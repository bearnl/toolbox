#!/bin/bash
# Rebuild the image shards with a MASK-AWARE resize (hiride_prep_edges.py).
# Stages BOTH archives -- unlike the FOV job, every sequence is needed.
#   sbatch --account=def-czarnuch_cpu run_hiride_edges.slurm

#SBATCH --job-name=hiride-edges
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs
RAR_DIR="${RAR_DIR:-$SCRATCH/datasets}"
SRC="${SRC:-$SCRATCH/hiride2/prep}"
OUT="${OUT:-$SCRATCH/hiride2/prep_edges}"

module purge
module load StdEnv/2023 python/3.11
export PATH="$HOME/.local/rar:$HOME/bin/rar:$HOME/bin:$PATH"
source "$HOME/venvs/venv311/bin/activate"

DATA_DIR="$SLURM_TMPDIR/biwi"
mkdir -p "$DATA_DIR"; cd "$DATA_DIR"
echo "[$(date)] staging archives"
for rar in "$RAR_DIR"/*.rar; do cp "$rar" .; done
for rar in *.rar; do echo "  extracting $rar"; unrar x -y "$rar" >/dev/null; done
rm -f ./*.rar
echo "  depth=$(find "$DATA_DIR" -name '*_depth.pgm' | wc -l)"

cd "$SLURM_SUBMIT_DIR"
python -u hiride_prep_edges.py --root "$DATA_DIR" --src "$SRC" --out "$OUT"
echo "[$(date)] done -> $OUT"
