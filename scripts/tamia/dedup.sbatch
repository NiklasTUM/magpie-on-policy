#!/bin/bash
#SBATCH --job-name=mp-dedup
#SBATCH --account=aip-irina
# No --partition: the scheduler auto-routes by --time (<=3h -> b1, the fast queue).
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=h100:4
#SBATCH --mem=160G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Near-duplicate measurement over a generated instruction set: exp/gen_dis.py embeds every
# instruction with all-mpnet-base-v2, builds a FAISS index, and records each one's minimum
# neighbour distance.
#
#   INPUT=/scratch/n/niklas/magpie-data/<job>/Magpie_..._ins_res_category.json \
#       sbatch scripts/tamia/dedup.sbatch
#
# WHY THIS EXISTS. Magpie draws every instruction in a shard from ONE fixed prompt -- the
# pilot takes ~1000 draws per template -- so within-shard near-duplication is the natural
# failure mode, and sampling entropy is the only thing preventing it. The smoke run showed
# 0 exact duplicates, but at n=198 that says nothing about n=12000. This is upstream's own
# tool for the problem (it is how the released 1M datasets were deduplicated), and its
# output is what should decide whether the category templates need sub-topic rotation --
# rather than adding that machinery on speculation.
#
# Only one GPU is used (the SentenceTransformer; the FAISS index is CPU), but tamia
# allocates whole nodes, so 4 is the floor regardless.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

: "${INPUT:?set INPUT to a *_ins_res_category.json (or any json with an 'instruction' field)}"
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
# shellcheck disable=SC1091
source scripts/tamia/cache_env.sh

# gen_dis.py writes to a hardcoded ../data/<name>_distance.jsonl relative to its CWD.
# The repo's data/ is a symlink onto $SCRATCH so this never touches the /home quota.
[[ -d data ]] || { echo "repo data/ missing; expected a symlink to \$SCRATCH" >&2; exit 1; }

echo "host=$(hostname) job=${SLURM_JOB_ID:-none} input=$INPUT"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# gen_dis.py is written for the ShareGPT-format file that data_concatenation.ipynb
# produces, so it assumes a `conversation_id` field -- and only touches it on the branch
# that fires when a near-duplicate IS found (gen_dis.py:127). A run with no duplicates
# therefore succeeds and a run with duplicates dies with KeyError: 'conversation_id'
# (job 418444). We feed it raw pipeline output, so add the field here rather than
# patching upstream.
PREPPED="${SLURM_TMPDIR:-/tmp}/$(basename "${INPUT%.json}")_withid.json"
python - "$INPUT" "$PREPPED" <<'PREP'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
data = json.load(open(src, encoding="utf-8"))
for i, rec in enumerate(data):
    rec.setdefault("conversation_id", rec.get("uuid") or rec.get("id") or i)
json.dump(data, open(dst, "w", encoding="utf-8"))
print(f"[dedup] prepared {len(data)} records with conversation_id -> {dst}")
PREP

cd exp
python gen_dis.py \
    --input_file "$PREPPED" \
    --device 0 \
    --encoding_batch_size "${ENCODING_BATCH_SIZE:-65536}" \
    --search_space_size "${SEARCH_SPACE_SIZE:-500}" \
    --search_batch_size "${SEARCH_BATCH_SIZE:-1024}"

name=$(basename "$PREPPED"); name=${name%.*}
out="../data/${name}_distance.jsonl"
echo "[dedup] wrote $out"

# Summarise, because the raw per-record distances are not the thing anyone wants to read.
python - "$out" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
d = sorted(r["min_neighbor_distance"] for r in rows if r.get("min_neighbor_distance") is not None)
n = len(d)
if not n:
    sys.exit("no min_neighbor_distance values found")
print(f"\n[dedup] {n} instructions")
print(f"[dedup] min neighbour distance: p1={d[n//100]:.4f} p5={d[n//20]:.4f} "
      f"median={d[n//2]:.4f} p95={d[19*n//20]:.4f}")
# 0.05 is the threshold gen_dis.py/data_filter.ipynb treat as "repetition".
for t in (0.01, 0.05, 0.10):
    k = sum(1 for x in d if x < t)
    print(f"[dedup] below {t:.2f}: {k} ({k/n:.1%})")
print("[dedup] Reading: a large mass below 0.05 means the shards are collapsing onto the "
      "same few instructions, which is the case for adding sub-topic rotation.")
PY

echo "done"
