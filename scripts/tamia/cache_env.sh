# Redirect every cache that defaults to $HOME onto $SCRATCH.  source, don't run.
#
# $HOME on this cluster is a 25 GiB quota shared by every job. Exceeding it does
# NOT fail at startup -- it fails hours in, mid-forward-pass, with
#   OSError: [Errno 122] Disk quota exceeded
# and it kills every concurrent job at once because they share the quota. Six RL
# arms died that way ~14 h in (jobs 412543-412548, one of them at step 93/250),
# and the eval jobs launched alongside them failed the same way. vLLM's compile
# cache alone had reached 8.2 GB; Triton another 1.0 GB; offline W&B runs, which
# default to $PWD (the repo, in $HOME), another 3.3 GB.
#
# HF_HOME is set separately by each script (model weights live in the shared
# project cache). This file covers everything else.
#
# Sourced by every script that starts a torch/vLLM process:
#   magpie_qwen36.sbatch
#
# Each is overridable from the environment so a caller can point elsewhere.
#
# Copied from reward-hacking-misalignment/scripts/tamia/cache_env.sh; the job numbers
# in the comments below are from that repo's runs. Keep the two in sync.

: "${SCRATCH:?SCRATCH must be set}"

export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$SCRATCH/.cache}"

# COMPILE caches go PER-JOB on node-local $SLURM_TMPDIR, not on a shared path.
#
# Triton/inductor/vLLM write compiled kernels keyed by content hash, and their
# cache writes are not safe against concurrent first-time compilation: several
# jobs racing to produce the same key can read a torn or partial file. That does
# not surface as a cache error -- it surfaces as a corrupt kernel, i.e.
#   ncclUnhandledCudaError: Call to CUDA function failed
# Job 415867 died that way at step 0, ~20 min after the shared cache had been
# emptied, while the eval jobs launched alongside it were compiling the same
# kernels into the same directory. The next job on the SAME node, with a warm
# cache, ran fine -- so it was the cold-cache race, not the hardware.
#
# $SLURM_TMPDIR is node-local (~7 TB) and torn down with the job, so every job
# gets a private cache. The cost is recompiling per job (a few minutes); the
# benefit is that launching N jobs together can never corrupt a shared kernel.
_compile_cache="${SLURM_TMPDIR:-$SCRATCH/.cache/job-${SLURM_JOB_ID:-$$}}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$_compile_cache/vllm}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$_compile_cache/triton}"
export TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-$_compile_cache/torchinductor}"
export FLASHINFER_WORKSPACE_BASE="${FLASHINFER_WORKSPACE_BASE:-$_compile_cache/flashinfer}"
export TORCH_HOME="${TORCH_HOME:-$SCRATCH/.cache/torch}"
# wandb writes offline runs into $PWD unless told otherwise. Magpie does not use wandb,
# but a stray import must not land in $HOME either.
export WANDB_DIR="${WANDB_DIR:-$SCRATCH/magpie-data}"

mkdir -p "$XDG_CACHE_HOME" "$VLLM_CACHE_ROOT" "$TRITON_CACHE_DIR" \
         "$TORCHINDUCTOR_CACHE_DIR" "$FLASHINFER_WORKSPACE_BASE" \
         "$TORCH_HOME" "$WANDB_DIR"

# Fail loudly rather than hours later: if $HOME is already at its quota, no
# amount of redirection saves a job that still writes a dotfile somewhere.
if command -v diskusage_report >/dev/null 2>&1; then
    _home_line=$(diskusage_report 2>/dev/null | grep -E "^\s+/home" || true)
    if [[ -n "$_home_line" ]]; then
        echo "[cache_env] $_home_line"
    fi
    unset _home_line
fi
