#!/bin/bash
# Adjudicate the paper-2 vs paper-3 FOV-clipping contradiction on RAW frames.
#
# The 256^2 shards are AREA-resized, and that box average mixes invalid (0)
# pixels into valid depth, manufacturing spuriously NEAR values -- precisely
# what a 1st-percentile slab anchor latches onto. So the shard-based answer is
# suggestive but confounded; this job stages Training.rar and reruns the same
# check on the original 480x640 PGM, where no such artefact exists.
#
#   sbatch --account=def-czarnuch_cpu run_hiride_fov.slurm

#SBATCH --job-name=hiride-fov-raw
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=32000M
#SBATCH --output=logs/%x_%j.out

set -euo pipefail
mkdir -p logs

RAR_DIR="${RAR_DIR:-$SCRATCH/datasets}"
PREP="${PREP:-$SCRATCH/hiride2/prep}"
OUT="${OUT:-$SCRATCH/hiride2/results}"
SAMPLE="${SAMPLE:-4000}"

module purge
module load StdEnv/2023 python/3.11
export PATH="$HOME/.local/rar:$HOME/bin/rar:$HOME/bin:$PATH"
source "$HOME/venvs/venv311/bin/activate"

DATA_DIR="$SLURM_TMPDIR/biwi"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"
echo "[$(date)] staging Training.rar from $RAR_DIR (Testing not needed -- the"
echo "          contradiction is about Training frames)"
cp "$RAR_DIR/Training.rar" .
unrar x -y Training.rar >/dev/null
rm -f ./Training.rar
echo "  depth=$(find "$DATA_DIR" -name '*_depth.pgm' | wc -l)" \
     "userMap=$(find "$DATA_DIR" -name '*_userMap.pgm' | wc -l)"

cd "$SLURM_SUBMIT_DIR"
python -u hiride_fov_check.py --prep "$PREP" --root "$DATA_DIR" \
       --sample "$SAMPLE" --out "$OUT"
echo "[$(date)] done -> $OUT/fov_check.json"
