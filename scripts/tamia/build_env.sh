#!/usr/bin/env bash
# Build the Magpie venv on Tamia (DRAC / Compute Canada).
#
# RUN THIS ON THE LOGIN NODE. Compute nodes have no internet.
#
#   bash scripts/tamia/build_env.sh          # build (safe to re-run)
#   FORCE=1 bash scripts/tamia/build_env.sh  # delete and rebuild
#   VENV=/path/to/venv bash scripts/tamia/build_env.sh
#
# Why the pins are what they are:
#   * Magpie's own requirements.txt pins vllm==0.6.5 (and a cu124/torch2.4 flashinfer
#     index). That is DISCARDED, not adapted: Qwen3.6-27B is
#     Qwen3_5ForConditionalGeneration -- a hybrid GatedDeltaNet/full-attention
#     multimodal architecture with a 248320-token vocab -- and it simply does not
#     exist in vLLM 0.6.5. The model card asks for vllm>=0.19.0.
#   * vllm 0.25.0 is the newest in the Alliance wheelhouse and hard-pins
#     torch==2.11.0 / torchvision==0.26.0 / torchaudio==2.11.0 and transformers>=5.5.3,
#     so those are not free choices. This is the same stack as $SCRATCH/venvs/rh-rl,
#     which has a track record running this model on these nodes.
#   * fschat and lingua-language-detector have NO wheelhouse copy. Both are optional
#     here (exp/utils.py and exp/unitag.py import them lazily), so they are attempted
#     from PyPI and skipped with a warning if unavailable.
set -euo pipefail

VENV="${VENV:-$SCRATCH/venvs/magpie}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# --- Pins (all present in the wheelhouse; verified before writing this) -------
TORCH=2.11.0
TORCHVISION=0.26.0
TORCHAUDIO=2.11.0
TRITON=3.6.0
VLLM=0.25.0
TRANSFORMERS=5.14.1
ACCELERATE=1.14.0
DATASETS=5.0.0
FLASHINFER=0.6.13
FLA=0.4.1

say() { printf '\n=== %s ===\n' "$*"; }

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "REFUSING: this must run on the login node (SLURM_JOB_ID is set)." >&2
    echo "Compute nodes have no internet and cannot fetch the optional PyPI packages." >&2
    exit 1
fi

say "Modules"
# arrow and opencv are NOT optional extras: the Alliance ships deliberately failing
# dummy pyarrow/opencv wheels whose error message tells you to load the module instead.
# opencv matters more than usual here because Qwen3.6 is a vision-language checkpoint.
# cuda provides $EBROOTCUDA for CUDA_HOME.
module load gcc python/3.12 arrow/24 opencv/4.13 cuda
python --version

# Login-node /tmp is small and RAM-backed: unpacking a ~300 MB vllm or ~900 MB torch
# wheel there gets cgroup-killed mid-install. $HOME is near quota, so keep pip's
# scratch space on Lustre.
export TMPDIR="$SCRATCH/.pip-tmp"
export PIP_CACHE_DIR="$SCRATCH/.pip-cache"
mkdir -p "$TMPDIR" "$PIP_CACHE_DIR"

if [[ -d "$VENV" ]]; then
    if [[ "${FORCE:-0}" == "1" ]]; then
        say "Removing existing venv $VENV"
        rm -rf "$VENV"
    else
        echo "venv already exists at $VENV -- set FORCE=1 to rebuild." >&2
        echo "Continuing to the install/verify steps (safe to re-run)." >&2
    fi
fi

if [[ ! -d "$VENV" ]]; then
    say "Creating venv at $VENV"
    virtualenv --no-download "$VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --no-index --upgrade pip setuptools wheel

# --- Everything below comes from the CVMFS wheelhouse -------------------------
# PIP_CONFIG_FILE (set by StdEnv) already points find-links at the wheelhouse, so
# --no-index resolves offline with no --find-links needed.

say "Torch stack (pinned by vllm $VLLM)"
pip install --no-index \
    "torch==$TORCH" "torchvision==$TORCHVISION" "torchaudio==$TORCHAUDIO" \
    "triton==$TRITON"

say "vLLM $VLLM"
pip install --no-index "vllm==$VLLM" "flashinfer_python==$FLASHINFER"

say "Transformers stack"
pip install --no-index \
    "transformers==$TRANSFORMERS" "accelerate==$ACCELERATE" \
    "flash_linear_attention==$FLA"

# datasets must be --no-deps: its pyarrow dependency resolves to the Alliance's
# deliberately-failing dummy wheel. pyarrow comes from `module load arrow`.
say "datasets $DATASETS (--no-deps; pyarrow from the arrow module)"
pip install --no-index --no-deps "datasets==$DATASETS"
pip install --no-index dill multiprocess xxhash pandas fsspec requests tqdm pyyaml

say "Magpie's own utility packages"
# One at a time: these are non-critical, and a single missing name in a combined pip
# call would take the whole group down with it.
for pkg in sentence_transformers tenacity ml_collections matplotlib ipykernel ipywidgets; do
    if pip install --no-index -q "$pkg"; then
        echo "  $pkg ok"
    else
        echo "  WARN: $pkg unavailable offline (skipped)"
    fi
done

# --- Optional PyPI-only packages ---------------------------------------------
# Both are lazily imported by the Magpie code (see the note at the top), so a failure
# here is a warning, not a build failure. Wheels only: building from source on the
# login node is what gets OOM-killed. PYTHONPATH= defeats the Alliance's _manylinux
# shim, which otherwise blocks manylinux wheels.
say "Optional: fschat, lingua-language-detector, faiss-cpu (PyPI; no wheelhouse copy)"
# faiss-cpu is needed by exp/gen_dis.py, the near-duplicate measurement (min neighbour
# distance over sentence embeddings). CPU build is sufficient: gen_dis.py only puts the
# SentenceTransformer on the GPU, the index itself is CPU (datasets.add_faiss_index).
for pkg in fschat lingua-language-detector faiss-cpu; do
    if PYTHONPATH= pip install --only-binary :all: -q "$pkg"; then
        echo "  $pkg ok"
    else
        echo "  WARN: $pkg not installed. Consequences:"
        echo "        fschat                   -> always pass --use_tokenizer_template"
        echo "        lingua-language-detector -> unitag.py --tag_mission language unavailable"
        echo "        faiss-cpu                -> exp/gen_dis.py (dedup) unavailable"
    fi
done

# A PyPI resolution may have pulled newer transformers / torch and broken vLLM's exact
# pins. Re-assert them; this is a no-op when nothing moved.
say "Re-asserting wheelhouse pins after the PyPI installs"
pip install --no-index \
    "torch==$TORCH" "torchvision==$TORCHVISION" "torchaudio==$TORCHAUDIO" \
    "transformers==$TRANSFORMERS" "vllm==$VLLM"

# torchcodec can make `import vllm` die outright: a CUDA-13 build links libnvrtc.so.13
# against a CUDA-12 torch (SLURM job 385274 in a sibling repo). But vLLM 0.25.0 pulls
# torchcodec in as a real dependency and the wheelhouse copy is a matching +torch211
# build, so removing it unconditionally risks breaking the very import the rule exists
# to protect. Decide empirically.
say "torchcodec / import vllm compatibility"
if python -c "import vllm" >/dev/null 2>&1; then
    echo "  import vllm ok -- leaving torchcodec as vLLM installed it"
elif pip show torchcodec >/dev/null 2>&1; then
    echo "  import vllm FAILED with torchcodec present -- removing it and retrying"
    pip uninstall -y torchcodec
    if python -c "import vllm" >/dev/null 2>&1; then
        echo "  import vllm ok after removing torchcodec"
    else
        echo "  import vllm STILL failing -- see the verification output below"
    fi
else
    echo "  import vllm FAILED and torchcodec is not installed -- something else is wrong"
fi

# --- Verify. Fail the BUILD, not the first job at 3am. ------------------------
say "Verification"
REPO="$REPO" python - <<'PY'
import json
import os
import sys

REQUIRED = {
    "torch": "2.11.0",
    "torchvision": "0.26.0",
    "vllm": "0.25.0",
    "transformers": "5.14.1",
}
MODEL = "Qwen/Qwen3.6-27B"

failed = []

from importlib.metadata import version, PackageNotFoundError
for pkg, want in REQUIRED.items():
    try:
        got = version(pkg)
    except PackageNotFoundError:
        failed.append(f"{pkg}: NOT INSTALLED (want {want})")
        continue
    # Alliance wheels carry a +computecanada local version suffix.
    base = got.split("+")[0]
    status = "ok" if base == want else "MISMATCH"
    print(f"  {pkg:16s} {got:30s} want {want:10s} {status}")
    if base != want:
        failed.append(f"{pkg}: got {got}, want {want}")

print()
for mod in ("torch", "vllm", "transformers", "pyarrow", "datasets"):
    try:
        __import__(mod)
        print(f"  import {mod:24s} ok")
    except Exception as exc:
        print(f"  import {mod:24s} FAILED: {type(exc).__name__}: {exc}")
        failed.append(f"import {mod}: {exc}")

# Optional, lazily imported by the Magpie code. Absence is reported, not fatal.
print()
for mod, why in (("fastchat", "FastChat conversation templates"),
                 ("lingua", "unitag.py --tag_mission language")):
    try:
        __import__(mod)
        print(f"  optional {mod:12s} present")
    except ImportError:
        print(f"  optional {mod:12s} ABSENT -- {why} unavailable")

# The whole point of this venv: vLLM must recognise Qwen3.6's architecture.
print()
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = set(ModelRegistry.get_supported_archs())
    for arch in ("Qwen3_5ForConditionalGeneration",):
        ok = arch in archs
        print(f"  vLLM registry: {arch} {'ok' if ok else 'MISSING'}")
        if not ok:
            failed.append(f"vLLM does not support {arch}")
except Exception as exc:
    print(f"  vLLM registry check FAILED: {type(exc).__name__}: {exc}")
    failed.append(f"vLLM registry check: {exc}")

# The model config entry and its category templates must be loadable, and the model
# itself resolvable from the shared offline HF cache.
repo = os.environ["REPO"]
try:
    cfg = json.load(open(f"{repo}/configs/model_configs.json", encoding="utf-8"))[MODEL]
    cats = sorted(k[len("pre_query_template_"):] for k in cfg
                  if k.startswith("pre_query_template_") and k != "pre_query_template_with_system_prompt")
    print(f"  model_configs.json: {MODEL} -> {len(cats)} category templates")
    if len(cats) < 2:
        failed.append(f"{MODEL} has too few category templates: {cats}")
except Exception as exc:
    print(f"  model_configs.json check FAILED: {type(exc).__name__}: {exc}")
    failed.append(f"model_configs.json: {exc}")

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    rendered = tok.apply_chat_template([{"role": "user", "content": "x"}], tokenize=False,
                                       add_generation_prompt=True, enable_thinking=False)
    assert rendered.startswith(cfg["pre_query_template"]), repr(rendered[:60])
    assert rendered.endswith("<think>\n\n</think>\n\n"), repr(rendered[-40:])
    print(f"  tokenizer: {MODEL} resolved offline; pre_query_template is a true prefix")
except Exception as exc:
    print(f"  tokenizer check FAILED: {type(exc).__name__}: {exc}")
    failed.append(f"tokenizer: {exc}")

print()
import torch
print(f"  torch.version.cuda        = {torch.version.cuda}")
print(f"  torch.cuda.is_available() = {torch.cuda.is_available()}  (False on login node is expected)")

if failed:
    print("\nBUILD FAILED:")
    for f in failed:
        print(f"  - {f}")
    sys.exit(1)
print("\nAll checks passed.")
PY

say "Disk usage"
# /scratch inode count is the one to watch: a vLLM venv is tens of thousands of files.
command -v diskusage_report >/dev/null 2>&1 && diskusage_report || true

say "Done"
cat <<EOF
venv: $VENV

Activate in job scripts with:
  module load gcc python/3.12 arrow/24 opencv/4.13 cuda
  source $VENV/bin/activate

Reminder: \`module load arrow\` is required at RUNTIME too -- pyarrow comes from the
module, not from pip, so \`import datasets\` fails without it. Same for opencv.
EOF
