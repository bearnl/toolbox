#!/bin/bash
# HI-RIDE paper-2 re-run: one-time BIWI preparation. CPU ONLY -- no GPU needed.
#
# Stages Training.rar + Testing.rar to node-local $SLURM_TMPDIR, builds the
# manifest, computes the 13 scalar cues + per-recording background plates, and
# writes ~12 large shard files back to $SCRATCH. Every later job reads those
# instead of ~196,000 loose frames, which is what keeps us inside the inode
# quota AND fixes the input-pipeline bottleneck that made the 2023 runs
# I/O-bound by roughly two orders of magnitude.
#
# Submit:
#   sbatch --account=def-czarnuch_cpu run_hiride_prep.slurm
# Cues only (fast, ~20 min, no big shards):
#   sbatch --account=def-czarnuch_cpu --export=ALL,NO_SHARDS=1 run_hiride_prep.slurm

#SBATCH --job-name=hiride-prep
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

SIZE="${SIZE:-256}"
STRIDE="${STRIDE:-1}"
NO_SHARDS="${NO_SHARDS:-0}"
RAR_DIR="${RAR_DIR:-$SCRATCH/datasets}"
OUT_DIR="${OUT_DIR:-$SCRATCH/hiride2/prep}"

module purge
module load StdEnv/2023 python/3.11
# NOTE: no opencv module. The venv cannot see its bindings anyway (cv2 imports
# as MISSING inside venv311), so hiride_pgm.py parses PGM with numpy only.
export PATH="$HOME/.local/rar:$HOME/bin/rar:$HOME/bin:$PATH"
source "$HOME/venvs/venv311/bin/activate"

DATA_DIR="$SLURM_TMPDIR/biwi"
mkdir -p "$DATA_DIR" "$OUT_DIR"
cd "$DATA_DIR"

echo "[$(date)] staging archives from $RAR_DIR"
for rar in "$RAR_DIR"/*.rar; do cp "$rar" .; done
for rar in *.rar; do echo "  extracting $rar"; unrar x -y "$rar" >/dev/null; done
rm -f ./*.rar

echo "[$(date)] extracted layout:"
ls "$DATA_DIR"
echo "  depth=$(find "$DATA_DIR" -name '*_depth.pgm' | wc -l)" \
     "userMap=$(find "$DATA_DIR" -name '*_userMap.pgm' | wc -l)" \
     "rgb=$(find "$DATA_DIR" -name '*_rgb.jpg' | wc -l)"

cd "$SLURM_SUBMIT_DIR"

# ---------------------------------------------------------------------------
# GATE: the split invariants must hold on the freshly staged tree before we
# spend anything on prep. This is the only place the extracted tree exists, so
# it is the only place the test can run -- $SLURM_TMPDIR does not exist on a
# login node. A non-zero exit here aborts the job (set -e), which is intended:
# a split that is not disjoint invalidates every number downstream.
# ---------------------------------------------------------------------------
echo "[$(date)] checking split invariants on the staged tree"
python -u test_hiride_splits.py --root "$DATA_DIR"

EXTRA=""
[ "$NO_SHARDS" = "1" ] && EXTRA="--no-shards"

python -u hiride_prep.py \
    --data "$DATA_DIR" \
    --out  "$OUT_DIR" \
    --size "$SIZE" \
    --stride "$STRIDE" \
    $EXTRA

echo "[$(date)] prep complete -> $OUT_DIR"
ls -lh "$OUT_DIR"
df -h "$SCRATCH" | tail -1
