#!/bin/bash
#SBATCH --job-name=mp-quality
#SBATCH --account=aip-irina
# No --partition: the scheduler auto-routes by --time.
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=160G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# LLM-judge quality pass over a filtered instruction set, using Magpie's own
# unitag.py --tag_mission quality (rates very poor -> excellent with an explanation).
# Upstream's filter is literally `input_quality in ['good','excellent']`.
#
#   INPUT=/scratch/.../magpie_qwen36_pilot_clean.json sbatch scripts/tamia/quality_tag.sbatch
#
# WHY A JUDGE AND NOT MORE RULES. The deterministic filter already removed what rules do
# reliably: dangling context, phrase echoes recurring 3+ times, token leakage. What it
# deliberately left in is the echo==2 band (~25% of the corpus), where a random sample
# was roughly a third good / a third bad / a third borderline -- not separable by rule
# without discarding ~1000 good instructions. That ambiguous middle is what this scores.
#
# Same engine settings as the tagging stage: the tagger prefills '{' and stops at '}',
# which requires the CLOSED empty think block, hence --enable_thinking off (the default)
# and --use_tokenizer_template. --quantization None because unitag.py defaults to fp8.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

: "${INPUT:?set INPUT to a filtered *_clean.json}"
[[ -f "$INPUT" ]] || { echo "INPUT not found: $INPUT" >&2; exit 1; }

VENV="${VENV:-$SCRATCH/venvs/magpie}"
module load gcc python/3.12 arrow/24 opencv/4.13 cuda
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export CUDA_HOME=$EBROOTCUDA
export HF_HOME=/project/aip-irina/shared/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_CACHE || true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# shellcheck disable=SC1091
source scripts/tamia/cache_env.sh

MODEL="${MODEL:-Qwen/Qwen3.6-27B}"

# The GatedDeltaNet prefill kernel only exists on the Qwen3.5/3.6 hybrids; passing it for
# any other architecture is meaningless. Judging with a different model family (to avoid
# a model grading its own output) is the normal case, so gate it on the model name.
gdn_flags=()
case "${MODEL,,}" in
    *qwen3.5*|*qwen3.6*) gdn_flags+=(--gdn_prefill_backend "${GDN_PREFILL_BACKEND:-triton}") ;;
esac

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} input=$INPUT model=$MODEL"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# unitag.py derives its output name from the INPUT name ({stem}_quality.json), so two
# judges over the same file would silently overwrite each other -- and unitag opens the
# output with 'w', so a failed second run would destroy the first judge's grades before
# producing anything. Give each judge its own input copy, which keeps both verdicts and
# makes cross-judge comparison possible.
TAG=$(basename "$MODEL" | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9.-' '-' | sed 's/-*$//')
JUDGE_INPUT="${INPUT%.json}_judge-${TAG}.json"
cp -n "$INPUT" "$JUDGE_INPUT"
echo "[quality] judging via $JUDGE_INPUT"

cd scripts
CUDA_VISIBLE_DEVICES=0,1,2,3 python ../exp/unitag.py \
    --device 0,1,2,3 \
    --tag_mission quality \
    --model_path "$MODEL" \
    --input_file "$JUDGE_INPUT" \
    --batch_size "${BATCH_SIZE:-1000}" \
    --checkpoint_every 5 \
    --max_tokens 512 \
    --max_model_len 4096 \
    --tensor_parallel_size 4 \
    --gpu_memory_utilization 0.90 \
    --quantization None \
    --use_tokenizer_template \
    --disable_prefix_caching \
    --save_as json \
    --offline \
    --language_model_only \
    --enforce_eager \
    --disable_custom_all_reduce \
    ${gdn_flags[@]+"${gdn_flags[@]}"}

out="${JUDGE_INPUT%.json}_quality.json"
echo "[quality] wrote $out"

python - "$out" <<'PY'
import json, sys
from collections import Counter
d = json.load(open(sys.argv[1], encoding="utf-8"))
n = len(d)
q = Counter(r.get("input_quality") or "(unparsed)" for r in d)
order = ["excellent", "good", "average", "poor", "very poor", "(unparsed)"]
print(f"\n[quality] {n} instructions")
for k in order:
    if k in q:
        print(f"[quality]   {k:12s} {q[k]:6d} ({q[k]/n:5.1%})")
keep = sum(q.get(k, 0) for k in ("good", "excellent"))
print(f"[quality] upstream's cut (good|excellent): {keep}/{n} ({keep/n:.1%})")
PY

echo "done"
