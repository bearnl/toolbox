#!/bin/bash
# Overnight batch: the "use depth as GEOMETRY" arm, submitted as one command.
#
#   bash run_hiride_night.slurm
#
# Submits four jobs with dependencies so nothing waits on a human:
#   1 metric  (CPU, staged raw tree) extract metric 3D + skeleton features
#   2 floor   (CPU, after 1)         the ladder on those features
#   3 heads   (GPU, independent)     wave 10, the classifier-head ablation
#   4 signal  (CPU, independent)     probe on the exact CNN splits, unbalanced
#
# NOTHING EXISTING IS MODIFIED: every new switch defaults to current behaviour
# (--head gap, --frames 1, --depth-encoding raw), so runs already queued by
# another agent are unaffected, and waves 2-9 emit byte-identical runs files.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
PREP="${PREP:-$SCRATCH/hiride2/prep}"
RES="${RES:-$SCRATCH/hiride2/results}"
RUNS="${RUNS:-$SCRATCH/hiride2/runs}"

cat > .night_metric.sh <<'SH'
#!/bin/bash
#SBATCH --job-name=hiride-metric
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=48000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module purge; module load StdEnv/2023 python/3.11
export PATH="$HOME/.local/rar:$HOME/bin/rar:$HOME/bin:$PATH"
source "$HOME/venvs/venv311/bin/activate"
D="$SLURM_TMPDIR/biwi"; mkdir -p "$D"; cd "$D"
echo "[$(date)] staging archives"
cp "$SCRATCH/datasets/Training.rar" . && unrar x -y Training.rar >/dev/null && rm -f Training.rar
cp "$SCRATCH/datasets/Testing.rar"  . && unrar x -y Testing.rar  >/dev/null && rm -f Testing.rar
echo "  depth=$(find "$D" -name '*_depth.pgm' | wc -l) ground=$(find "$D" -name '*_groundCoeff.txt' | wc -l) skel=$(find "$D" -name '*_skel.txt' | wc -l)"
cd "$SLURM_SUBMIT_DIR"
python -u hiride_metric.py --prep "$PREP" --root "$D"
SH

cat > .night_floor.sh <<'SH'
#!/bin/bash
#SBATCH --job-name=hiride-metricfloor
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module purge; module load StdEnv/2023 python/3.11
source "$HOME/venvs/venv311/bin/activate"
python -u hiride_metric_floor.py --prep "$PREP" --out "$RES"
SH

cat > .night_signal.sh <<'SH'
#!/bin/bash
#SBATCH --job-name=hiride-signal-fair
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module purge; module load StdEnv/2023 python/3.11
source "$HOME/venvs/venv311/bin/activate"
# per-class 400 ~ "use everything": makes the probe comparable to the CNN, which
# trained on the full split. The earlier probe used 70/class, so the probe-beats-CNN
# gap (R1 interior_only 60.5 vs 40.9) still had a training-size confound.
python -u hiride_signal.py --prep "$PREP" --per-class 400 --out "$RES"
SH

export PREP RES RUNS
J1=$(sbatch --parsable --account=def-czarnuch_cpu --export=ALL,PREP="$PREP" .night_metric.sh)
J2=$(sbatch --parsable --dependency=afterok:$J1 --account=def-czarnuch_cpu \
     --export=ALL,PREP="$PREP",RES="$RES" .night_floor.sh)
J4=$(sbatch --parsable --account=def-czarnuch_cpu --export=ALL,PREP="$PREP",RES="$RES" .night_signal.sh)
python make_runs.py --wave 10 > runs10.txt
J3=$(sbatch --parsable --account=def-czarnuch_gpu --array=1-$(wc -l < runs10.txt)%8 \
     --export=ALL,RUNS_FILE=$PWD/runs10.txt,PREP="$PREP",OUT="$RUNS" run_hiride.slurm)
echo "submitted:"
echo "  $J1  hiride-metric      (CPU, stages both archives, ~2-4 h)"
echo "  $J2  hiride-metricfloor (CPU, runs after $J1)"
echo "  $J3  wave 10 head ablation (GPU, $(wc -l < runs10.txt) cells, ~5 h)"
echo "  $J4  hiride-signal-fair (CPU, unbalanced probe)"
echo
echo "collect with:  bash collect_night.sh"
cat > collect_night.sh <<'SH'
#!/bin/bash
cd "$(dirname "$0")"
clean() { grep -v "^2026-\|oneDNN\|cuInit\|TF-TRT\|To enable\|cpu_feature\|^  [0-9]*)" "$@"; }
for j in hiride-metric hiride-metricfloor hiride-signal-fair; do
  f=$(ls -t logs/${j}_*.out 2>/dev/null | head -1)
  echo "=================== $j  ($f)"; [ -n "$f" ] && clean "$f" | tail -60
done
echo "=================== wave 10 (head ablation)"
sacct -u "$USER" -S now-12hours -X -P --format=JobID,JobName,State -n | grep hiride | tail -5
source "$HOME/venvs/venv311/bin/activate" 2>/dev/null
python hiride_collate.py --runs "${RUNS:-$SCRATCH/hiride2/runs}" --floor "${RES:-$SCRATCH/hiride2/results}" \
  | grep -E "arch |flatten|stripe"
SH
chmod +x collect_night.sh
