#!/bin/bash
# WAVE 22 -- the field's BIWI protocol as a commensurability rung (HIRIDE_HANDOFF 14.8).
#
#     cd ~/toolbox && git fetch && git reset --hard origin/master && bash submit_wave22.sh
#     ...next morning:  bash collect_wave22.sh
#
# Submits 50 GPU cells (make_runs.py --wave 22: R4_standard_walking / R4_standard_still x
# {full, person, scale_removed, scale_removed best recipe, rgb best recipe} x 5 seeds,
# every line --skip-existing) at %8 on 1g.10gb slices -- ~4-10 min each, ~4-8 GPU-hours --
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
python make_runs.py --wave 22 > runs22.txt
echo "wave 22: $(wc -l < runs22.txt) cells"

cat > .wave22_analysis.sh <<'INNER'
#!/bin/bash
#SBATCH --job-name=hiride-w22-analysis
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

step "wave 22 cells as they landed"
python hiride_collate.py --runs "$RUNS" --floor "$R" | grep -E "R4_standard|policy" || true

step "13-scalar trivial-cue floor, now including the standard rungs (floor_results.json)"
python hiride_floor.py --prep "$PREP" --out "$R"

step "metric floor with CIs -- full-body test gate"
python hiride_metric_floor.py --prep "$PREP" --test-eligibility full_body \
    --out "$R" && mv "$R/metric_floor.json" "$R/metric_floor_fbtest.json"
step "metric floor with CIs -- headline configuration (runs LAST so metric_floor.json is the headline file)"
python hiride_metric_floor.py --prep "$PREP" --out "$R"

step "subject-cluster CIs over every cell (stats_final.json)"
python hiride_stats.py --runs "$RUNS" --boot 20000 --json "$R/stats_final.json"

step "tables and report, keyed through hiride_keys"
python hiride_report.py --runs "$RUNS" --floor "$R" --latex > "$R/tables.tex"
python hiride_report.py --runs "$RUNS" --floor "$R" > "$R/report.md"
grep -A40 "field's BIWI protocol" "$R/report.md" || echo "(no standard-protocol rows yet)"
echo; echo "analysis complete -> $R"
INNER

EXP="ALL,PREP=$PREP,RUNS=$RUNS,RES=$RES"
JW22=$(sbatch --parsable --account=$ACC_GPU \
       --array=1-$(wc -l < runs22.txt)%8 \
       --export="$EXP,RUNS_FILE=$PWD/runs22.txt,OUT=$RUNS" run_hiride.slurm)
JANA=$(sbatch --parsable --account=$ACC_CPU --dependency=afterany:$JW22 \
       --export="$EXP" .wave22_analysis.sh)

cat <<EOT

submitted:
  $JW22  wave 22   (GPU array $(wc -l < runs22.txt) cells, %8)
  $JANA  ANALYSIS  (CPU, afterany wave 22)

tomorrow:  bash collect_wave22.sh
EOT

cat > collect_wave22.sh <<'INNER'
#!/bin/bash
cd "$(dirname "$0")"
R="${RES:-$SCRATCH/hiride2/results}"
echo "=================== job states (anything not COMPLETED is a problem) ==================="
sacct -u "$USER" -S now-24hours -X --format=JobID%16,JobName%18,State%12,Elapsed \
  | grep -E "hiride|JobID" | tail -60
echo
echo "--- failures, exact task logs ---"
BAD=$(sacct -u "$USER" -S now-24hours -X -n -P --format=JobID,State,JobName \
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
f=$(ls -t logs/hiride-w22-analysis_*.out 2>/dev/null | head -1)
[ -n "$f" ] && grep -v "oneDNN\|cuInit\|TF-TRT\|cpu_feature\|^2026-" "$f" || echo "(not finished yet)"
INNER
chmod +x collect_wave22.sh
