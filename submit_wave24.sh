#!/bin/bash
# WAVE 24 -- does the pretrained network read the outline? (HIRIDE_HANDOFF 14.13).
#
#     cd ~/toolbox && git fetch && git reset --hard origin/master && bash submit_wave24.sh
#     ...next morning:  bash collect_wave24.sh
#
# Submits 20 GPU cells (make_runs.py --wave 24: ConvNeXt-Tiny/ImageNet, best recipe, R4, on
# the normalised silhouette, 2-bit depth, interior-only and the size-kept person, 5 seeds each,
# every line --skip-existing) at %6 on 1g.10gb slices -- ~20-30 min each, ~8-10 GPU-hours, 4 h wall --
# then one CPU analysis job (afterany) that regenerates: the 13-scalar floor and the
# metric floor under the new rungs, stats_final.json (all cells), and tables.tex /
# report.md through hiride_report.py, which now keys cells via hiride_keys (the previous
# mechanism and Z-precision tables averaged recipes into the gap-head rows).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PREP="${PREP:-$SCRATCH/hiride2/prep}"
RUNS="${RUNS:-$SCRATCH/hiride2/runs}"
RES="${RES:-$SCRATCH/hiride2/results}"
ACC_CPU=def-czarnuch_cpu
ACC_GPU=def-czarnuch_gpu

[ -f "$PREP/prep_meta.json" ] || { echo "no certified prep at $PREP"; exit 1; }
source ~/venvs/venv311/bin/activate
python make_runs.py --wave 24 > runs24.txt
echo "wave 24: $(wc -l < runs24.txt) cells"

cat > .wave24_analysis.sh <<'INNER'
#!/bin/bash
#SBATCH --job-name=hiride-w24-analysis
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
module purge; module load StdEnv/2023 python/3.11
source ~/venvs/venv311/bin/activate
R="$RES"; mkdir -p "$R"
step() { echo; echo "===================== $* ====================="; }

step "retag audit (must report NOTHING to rename)"
python hiride_retag.py --runs "$RUNS"

step "wave 24 cells as they landed"
python hiride_collate.py --runs "$RUNS" --floor "$R" | grep -E "cnxt|convnext|policy" || true

step "subject-cluster CIs over every cell (stats_final.json)"
python hiride_stats.py --runs "$RUNS" --boot 20000 --json "$R/stats_final.json"

step "tables and report, keyed through hiride_keys"
python hiride_report.py --runs "$RUNS" --floor "$R" --latex > "$R/tables.tex"
python hiride_report.py --runs "$RUNS" --floor "$R" > "$R/report.md"
grep -E "cnxt|convnext" "$R/report.md" || echo "(no convnext rows yet)"
echo; echo "analysis complete -> $R"
INNER

EXP="ALL,PREP=$PREP,RUNS=$RUNS,RES=$RES"
JW24=$(sbatch --parsable --account=$ACC_GPU --time=4:00:00 \
       --array=1-$(wc -l < runs24.txt)%6 \
       --export="$EXP,RUNS_FILE=$PWD/runs24.txt,OUT=$RUNS" run_hiride.slurm)
JANA=$(sbatch --parsable --account=$ACC_CPU --dependency=afterany:$JW24 \
       --export="$EXP" .wave24_analysis.sh)

cat <<EOT

submitted:
  $JW24  wave 24   (GPU array $(wc -l < runs24.txt) cells, %6)
  $JANA  ANALYSIS  (CPU, afterany wave 24)

tomorrow:  bash collect_wave24.sh
EOT

cat > collect_wave24.sh <<'INNER'
#!/bin/bash
cd "$(dirname "$0")"
R="${RES:-$SCRATCH/hiride2/results}"
echo "=================== job states (anything not COMPLETED is a problem) ==================="
sacct -u "$USER" -S $(date -d '-7 days' +%F) -X --format=JobID%16,JobName%18,State%12,Elapsed \
  | grep -E "hiride|JobID" | tail -60
echo
echo "--- failures, exact task logs ---"
BAD=$(sacct -u "$USER" -S $(date -d '-7 days' +%F) -X -n -P --format=JobID,State,JobName \
      | grep -Ev "COMPLETED|RUNNING|PENDING" | head -20)
if [ -z "$BAD" ]; then echo "  none"; else
  echo "$BAD"
  for jid in $(echo "$BAD" | cut -d'|' -f1 | head -8); do
    f=$(ls -t logs/*"$jid".out 2>/dev/null | head -1)
    [ -n "$f" ] && { echo "  --- $jid -> $f"; grep -E "Error|error:|Traceback|Killed|OOM" "$f" | tail -4; tail -2 "$f"; }
  done
fi
echo
echo "=================== the analysis job's full output ==================="
f=$(ls -t logs/hiride-w24-analysis_*.out 2>/dev/null | head -1)
[ -n "$f" ] && grep -v "oneDNN\|cuInit\|TF-TRT\|cpu_feature\|^2026-" "$f" || echo "(not finished yet)"
INNER
chmod +x collect_wave24.sh
