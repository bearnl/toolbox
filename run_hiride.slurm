#!/bin/bash
# HI-RIDE paper-2 re-run: one training cell.
#
# Wave 2 (the decisive ladder) as a job array, driven by a runs file:
#   python make_runs.py > runs.txt        # or write it by hand, one cell per line
#   sbatch --account=def-czarnuch_gpu --array=1-30%8 run_hiride.slurm
#
# A single cell, for a smoke test inside salloc:
#   POLICY=R1_block MODALITY=depth ARCH=alexnet SEED=0 bash run_hiride.slurm
#
# GPU accounting note: GPU jobs bill def-czarnuch_gpu, NOT def-czarnuch.
# AlexNet at 256^2 fits comfortably in the smallest MIG slice.

#SBATCH --job-name=hiride
#SBATCH --account=def-czarnuch_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64000M
#SBATCH --time=2:00:00
#SBATCH --output=logs/%x_%A_%a.out

set -euo pipefail
mkdir -p logs

PREP="${PREP:-$SCRATCH/hiride2/prep}"
OUT="${OUT:-$SCRATCH/hiride2/runs}"
RUNS_FILE="${RUNS_FILE:-}"

module purge
module load StdEnv/2023 python/3.11 cuda cudnn
source "$HOME/venvs/venv311/bin/activate"
nvidia-smi || true

# Array mode: take this task's arguments from line $SLURM_ARRAY_TASK_ID.
if [ -n "$RUNS_FILE" ] && [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
    LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$RUNS_FILE")
    [ -z "$LINE" ] && { echo "no line $SLURM_ARRAY_TASK_ID in $RUNS_FILE"; exit 1; }
    echo "[$(date)] cell: $LINE"
    # shellcheck disable=SC2086
    python -u hiride_train.py --prep "$PREP" --out "$OUT" $LINE
else
    python -u hiride_train.py --prep "$PREP" --out "$OUT" \
        --policy   "${POLICY:-R1_block}" \
        --modality "${MODALITY:-depth}" \
        --arch     "${ARCH:-alexnet}" \
        --condition "${CONDITION:-full}" \
        --guard    "${GUARD:-150}" \
        --seed     "${SEED:-0}"
fi

echo "[$(date)] cell complete -> $OUT"
