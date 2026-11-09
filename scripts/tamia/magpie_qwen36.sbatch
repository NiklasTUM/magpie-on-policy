#!/bin/bash
#SBATCH --job-name=mp-qwen36
#SBATCH --account=aip-irina
# No --partition: the scheduler auto-routes by --time (<=3h -> b1, <=12h -> b2,
# <=24h -> b3, which is the hard cap). 8h lands in b2 with headroom.
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=160G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Magpie on-policy data generation from Qwen3.6-27B: instructions -> responses -> task
# category tags, in one job.
#
#   SMOKE=1 sbatch --time=00:30:00 scripts/tamia/magpie_qwen36.sbatch   # 240 pairs
#   sbatch scripts/tamia/magpie_qwen36.sbatch                          # 10k pairs
#   TOTAL_PROMPTS=50000 sbatch --time=24:00:00 scripts/tamia/magpie_qwen36.sbatch
#   STAGE=tag TIMESTAMP=1755440000 sbatch --time=02:00:00 scripts/tamia/magpie_qwen36.sbatch
#
# Smokes get a SHORT wall clock (30 min): they finish or fail in minutes, and a long
# --time only burns allocation and schedules slower.
#
# ALL THREE STAGES RUN IN ONE JOB, but not because of the JIT. Each stage builds its own
# vLLM engine, and on this architecture a FlashInfer GatedDeltaNet prefill engine build
# costs 301-772 s of JIT compile (measured across ~14 Qwen3.6 runs in the sibling repos)
# -- per ENGINE, not per job. The driver therefore defaults to the triton GDN kernel,
# which skips the JIT entirely; see the rationale in magpie-qwen3.6-27b.sh. One job is
# still right, because it means one queue wait and one set of output paths.
#
# House style: no --requeue / signal traps. Resilience is the per-stage checkpoint files
# plus STAGE=, so a job that dies partway is resubmitted for the stage that failed with
# the original TIMESTAMP.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

VENV="${VENV:-$SCRATCH/venvs/magpie}"

# arrow and opencv are runtime requirements, not build extras: the Alliance ships
# deliberately-failing dummy pyarrow/opencv wheels. cuda provides $EBROOTCUDA.
module load gcc python/3.12 arrow/24 opencv/4.13 cuda
# shellcheck disable=SC1091
source "$VENV/bin/activate"

export CUDA_HOME=$EBROOTCUDA
# Model weights come from the shared project cache; compute nodes are offline.
export HF_HOME=/project/aip-irina/shared/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_CACHE || true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Redirect every cache that would otherwise land in $HOME's 25 GiB quota. Not optional:
# exceeding it kills all concurrent jobs mid-run, hours in. See the file.
# shellcheck disable=SC1091
source scripts/tamia/cache_env.sh

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.6-27B}"
export OUTPUT_FOLDER="${OUTPUT_FOLDER:-$SCRATCH/magpie-data}"
mkdir -p "$OUTPUT_FOLDER"

if [[ "${SMOKE:-0}" == "1" ]]; then
    # Enough to prove the whole chain works and to see whether Qwen3.6 continues as the
    # USER rather than answering itself -- which is Magpie's core premise and is
    # untested for this model family. Not enough to judge diversity from.
    TOTAL_PROMPTS="${TOTAL_PROMPTS:-240}"
    export N="${N:-20}"                    # 20 per template x 12 templates = 240/round
    TAG=smoke
else
    TOTAL_PROMPTS="${TOTAL_PROMPTS:-10000}"
    TAG=full
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} mode=$TAG model=$MODEL_PATH prompts=$TOTAL_PROMPTS stage=${STAGE:-all}"
echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "vllm=$(python -c 'import vllm; print(vllm.__version__)')"
echo "transformers=$(python -c 'import transformers; print(transformers.__version__)')"
# No `| head`: with `set -o pipefail`, head closing early sends SIGPIPE to nvidia-smi
# and the pipeline returns 141, aborting the script. Invisible on 4-GPU nodes (output is
# exactly 4 lines), fatal on 8-GPU ones.
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd scripts
bash magpie-qwen3.6-27b.sh "$MODEL_PATH" "$TOTAL_PROMPTS"

echo "done"
