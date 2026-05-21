#!/bin/bash
# Step-0 diagnostic: measure depth-derived metric anthropometry discriminative
# power on BIWI, with NO training. CPU-only, fast — clears the queue quickly.
#
# Usage:
#   sbatch --account=def-czarnuch anthro_probe.slurm
#   # override foreground slab for the measurement (default 1500mm):
#   sbatch --account=def-czarnuch --export=ALL,CLIP_MM=1200 anthro_probe.slurm

#SBATCH --account=def-czarnuch
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16000M
#SBATCH --job-name=anthro-probe
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p logs

CLIP_MM="${CLIP_MM:-1500}"
FULL_BODY="${FULL_BODY:-0}"          # 1 = pass --full-body-only (reject FOV-clipped frames)
RAR_DIR="${RAR_DIR:-$SCRATCH/datasets}"

module purge
module load StdEnv/2023 python/3.11 opencv
# numpy comes from the venv; cv2 from the opencv module. No TF/GPU needed.
source "$HOME/venvs/venv311/bin/activate"
export PATH="$HOME/.local/rar:$HOME/bin/rar:$HOME/bin:$PATH"

# Extract the dataset to node-local scratch (same pattern as run_biwi.slurm).
DATA_DIR="$SLURM_TMPDIR/biwi"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
echo "[$(date)] Copying + extracting .rar archives from $RAR_DIR ..."
for rar in "$RAR_DIR"/*.rar; do cp "$rar" .; done
for rar in *.rar; do echo "  $rar"; unrar x -y "$rar" >/dev/null; done
rm -f *.rar
echo "[$(date)] depth frame count: $(find "$DATA_DIR" -name '*_depth.pgm' | wc -l)"

cd "$SLURM_SUBMIT_DIR"
if [ "$FULL_BODY" = "1" ] || [ "$FULL_BODY" = "true" ]; then
    FBO_FLAG="--full-body-only"
else
    FBO_FLAG=""
fi
python -u anthro_probe.py --base-dir "$DATA_DIR" --clip-mm "$CLIP_MM" $FBO_FLAG
echo "[$(date)] anthro-probe complete."
