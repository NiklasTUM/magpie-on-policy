#!/bin/bash
#SBATCH --job-name=mp-glm53
#SBATCH --account=aip-gidelgau
# No --partition: the scheduler auto-routes by --time (<=3h -> b1, <=12h -> b2).
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=160G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Magpie on-policy data generation from GLM-5.3-Flash (W4A16 AutoRound): instructions ->
# responses -> category tags -> quality judge, in one job. Counterpart of
# magpie_qwen36.sbatch. Tagger and judge: Gemma-4-31B-it, from the same HF cache.
#
#   SMOKE=1 sbatch --time=01:30:00 scripts/tamia/magpie_glm53.sbatch   # 240 prompts
#   sbatch scripts/tamia/magpie_glm53.sbatch                          # 10k prompts
#   STAGE=res TIMESTAMP=<ts> sbatch scripts/tamia/magpie_glm53.sbatch  # resume one stage
#
# ENVIRONMENT. The magpie venv (vLLM 0.25) cannot load Glm5NextForConditionalGeneration,
# which needs vLLM >= 0.30. This job reuses the DDO project's judge venv and weights; every
# setting below is from ~/ddo/docs/glm53_judge_vllm.md, where each one is traced to a job
# that failed without it. That doc serves the model over HTTP; here the same environment
# runs vLLM in-process (gen_ins.py/gen_res.py/unitag.py build their own LLM engine).
#
# Smoke budget is 1 h, not Qwen's 30 min: the doc measures ~4 min startup plus first-use
# JIT, and each of the three stages builds its own engine.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

VENV="${VENV:-/project/aip-gidelgau/niklas/venvs/ddo-judge}"
[[ -x "$VENV/bin/python" ]] || { echo "venv not found: $VENV" >&2; exit 1; }

# FlashInfer, DeepGEMM and the AutoRound kernels JIT-compile and must COMPILE against the
# venv's own CUDA 13 toolkit, so CUDA_HOME/PATH/LD_LIBRARY_PATH point at the venv (without
# the venv's lib on LD_LIBRARY_PATH nvrtc cannot find libnvrtc-builtins.so.13.0).
# The cuda module (12.6) is still loaded, as in DDO's slurm/env.sh, for one reason: it puts
# cudacore/12.6.2/lib (libcudart.so) and cudacore/12.6.2/stubs/lib64 (libcuda.so) on
# LIBRARY_PATH, which FlashInfer's JIT LINK step needs -- without them the batch_mla
# attention kernel fails with 'ld: cannot find -lcudart / -lcuda' (job 680617). Its include
# paths must then be cleared, or its CCCL headers shadow CUDA 13's (DeepGEMM:
# 'cccl/.../monostate.h: error'). See ~/ddo/docs/glm53_judge_vllm.md.
module load gcc cuda
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
venv_cuda=$(ls -d "$VENV"/lib/python3*/site-packages/nvidia/cu*/ 2>/dev/null | head -1 || true)
[[ -n "$venv_cuda" ]] || { echo "no nvidia/cu* toolkit inside $VENV" >&2; exit 1; }
export CUDA_HOME="${venv_cuda%/}"
export PATH="$VENV/bin:$PATH"
export LD_LIBRARY_PATH="${CUDA_HOME}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export VIRTUAL_ENV="$VENV"
echo "LIBRARY_PATH=$LIBRARY_PATH"

# JIT kernel caches (e.g. ~/.humming) otherwise land in the shared 25 GiB $HOME quota.
export HOME="$SLURM_TMPDIR/home"
mkdir -p "$HOME"

# Weights live in the DDO project cache; compute nodes are offline.
export HF_HOME=/project/aip-gidelgau/niklas/huggingface
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
unset HF_HUB_CACHE || true
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Magpie samples at temperature 1.0, while the DDO judge only ever ran greedy, so this
# venv's sampling kernel path is untested on this cluster. FlashInfer's sampler failed to
# compile in another venv here (missing curand.h); the PyTorch sampler draws from the same
# distribution, so take it rather than risk a JIT failure hours in.
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
# shellcheck disable=SC1091
source scripts/tamia/cache_env.sh

MODEL_PATH="${MODEL_PATH:-Intel/GLM-5.3-Flash-W4A16-AutoRound}"
export OUTPUT_FOLDER="${OUTPUT_FOLDER:-$SCRATCH/magpie-data}"
mkdir -p "$OUTPUT_FOLDER"

if [[ "${SMOKE:-0}" == "1" ]]; then
    # 20 per template x 12 templates (11 categories + free) = 240, same as the Qwen smoke.
    # Enough to see whether GLM continues AS THE USER, stops on its own, and stays out of
    # repetition -- the three things that decided the Qwen fixes. Not enough for diversity.
    TOTAL_PROMPTS="${TOTAL_PROMPTS:-240}"
    export N="${N:-20}"
    TAG=smoke
else
    TOTAL_PROMPTS="${TOTAL_PROMPTS:-10000}"
    TAG=full
fi

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} mode=$TAG model=$MODEL_PATH prompts=$TOTAL_PROMPTS stage=${STAGE:-all}"
echo "git=$(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "python=$(command -v python) CUDA_HOME=$CUDA_HOME"
echo "vllm=$(python -c 'import vllm; print(vllm.__version__)')"
echo "transformers=$(python -c 'import transformers; print(transformers.__version__)')"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

cd scripts
bash magpie-glm5.3-flash.sh "$MODEL_PATH" "$TOTAL_PROMPTS"

echo "done"
