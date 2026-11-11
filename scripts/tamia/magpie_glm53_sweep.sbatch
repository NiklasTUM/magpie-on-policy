#!/bin/bash
#SBATCH --job-name=mp-glm53-sweep
#SBATCH --account=aip-gidelgau
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=160G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Instruction-only sampling sweep for GLM-5.3-Flash. Smoke job 680693 ran Qwen's settings
# (temp 1.0, top_p 1.0) and Gemma rated ~94% of the instructions poor/very poor: GLM wrote
# assistant reasoning, <tool_call> traffic and code-switched gibberish instead of a user
# turn. Each variant (see SWEEP_VARIANTS) is 240 generations (20 x 12 templates), then has Gemma judge every usable instruction in one pass.
#
#   sbatch --exclude=tg10907 scripts/tamia/magpie_glm53_sweep.sbatch
#
# No --drop_unsanitizable: the sanitizer's drop rate is one of the things being compared.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Environment: identical to magpie_glm53.sbatch (see the comments there).
VENV="${VENV:-/project/aip-gidelgau/niklas/venvs/ddo-judge}"
module load gcc cuda
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
venv_cuda=$(ls -d "$VENV"/lib/python3*/site-packages/nvidia/cu*/ 2>/dev/null | head -1 || true)
[[ -n "$venv_cuda" ]] || { echo "no nvidia/cu* toolkit inside $VENV" >&2; exit 1; }
export CUDA_HOME="${venv_cuda%/}"
export PATH="$VENV/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VIRTUAL_ENV="$VENV"
export HOME="$SLURM_TMPDIR/home"; mkdir -p "$HOME"
export HF_HOME=/project/aip-gidelgau/niklas/huggingface
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1
unset HF_HUB_CACHE || true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# shellcheck disable=SC1091
source scripts/tamia/cache_env.sh

MODEL_PATH="${MODEL_PATH:-Intel/GLM-5.3-Flash-W4A16-AutoRound}"
TAG_MODEL_PATH="${TAG_MODEL_PATH:-google/gemma-4-31B-it}"
TS=$(date +%s)
SWEEP_DIR="${OUTPUT_FOLDER:-$SCRATCH/magpie-data}/glm53_sweep_${TS}"
mkdir -p "$SWEEP_DIR"
echo "host=$(hostname) job=${SLURM_JOB_ID:-none} sweep_dir=$SWEEP_DIR git=$(git rev-parse --short HEAD)"

ENGINE=(--tensor_parallel_size 4 --gpu_memory_utilization 0.90
        --language_model_only --enforce_eager --disable_custom_all_reduce)

# One variant per line: name|temperature|top_p|reasoning_effort|min_tokens|presence_penalty
# (min_tokens/presence 0 = off). Override with SWEEP_VARIANTS. The default is the second
# sweep: Qwen's two instruction fixes (min_tokens=96 against ending the turn immediately,
# presence_penalty=1.5 against repetition) with and without 'Reasoning Effort: Low'.
# First sweep (job 685195): Max/Low x temp 1.0/0.8/0.7, no fixes; best was 6.7% good+.
SWEEP_VARIANTS="${SWEEP_VARIANTS:-low_t0.8_p0.95_min96_pp1.5|0.8|0.95|Low|96|1.5
low_t0.8_p0.95_min96|0.8|0.95|Low|96|0
low_t1.0_p1.0_min96_pp1.5|1.0|1.0|Low|96|1.5
max_t1.0_p1.0_min96_pp1.5|1.0|1.0|Max|96|1.5}"

cd scripts
while IFS="|" read -r -u 3 name temp topp effort mint pp; do
    [[ -z "$name" ]] && continue
    extra=()
    [[ "$effort" != "Max" ]] && extra+=(--template_sub "Reasoning Effort: Max" "Reasoning Effort: $effort")
    [[ "$mint" != "0" ]] && extra+=(--min_tokens "$mint")
    [[ "$pp" != "0" ]] && extra+=(--presence_penalty "$pp")
    echo "=== variant $name: temp=$temp top_p=$topp effort=$effort min_tokens=$mint presence=$pp"
    CUDA_VISIBLE_DEVICES=0,1,2,3 python ../exp/gen_ins.py \
        --device 0,1,2,3 --model_path "$MODEL_PATH" \
        --total_prompts 240 --n 20 --mix_control_tasks all+free \
        --temperature "$temp" --top_p "$topp" \
        --max_tokens 1024 --max_model_len 4096 \
        --disable_early_stopping --sanitize \
        --checkpoint_every 1 --output_folder "$SWEEP_DIR" --job_name "$name" --timestamp "$TS" \
        ${extra[@]+"${extra[@]}"} "${ENGINE[@]}" \
        || echo "=== variant $name FAILED (continuing)"
done 3<<< "$SWEEP_VARIANTS"

# Merge the usable instructions of every variant and judge them in one Gemma pass.
python - "$SWEEP_DIR" <<'EOF'
import glob, json, os, sys
d = sys.argv[1]; out = []
for f in sorted(glob.glob(os.path.join(d, "*", "*_ins.json"))):
    v = os.path.basename(os.path.dirname(f))
    for r in json.load(open(f, encoding="utf-8")):
        if r["instruction"].strip():
            r["sweep_variant"] = v; out.append(r)
for i, r in enumerate(out):
    r["id"] = i
json.dump(out, open(os.path.join(d, "sweep_usable.json"), "w"), indent=2, ensure_ascii=False)
print(f"merged {len(out)} usable instructions")
EOF

CUDA_VISIBLE_DEVICES=0,1,2,3 python ../exp/unitag.py \
    --device 0,1,2,3 --tag_mission quality --model_path "$TAG_MODEL_PATH" \
    --input_file "$SWEEP_DIR/sweep_usable.json" \
    --batch_size 1000 --checkpoint_every 5 --max_tokens 512 --max_model_len 4096 \
    --quantization None --use_tokenizer_template --disable_prefix_caching \
    --save_as json --offline "${ENGINE[@]}"

python tamia/sweep_summary.py "$SWEEP_DIR"
echo "done"
