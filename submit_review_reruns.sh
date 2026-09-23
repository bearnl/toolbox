#!/bin/bash
# Reruns for the 2026-09-23 review of the manuscript (HIRIDE_HANDOFF 14.24). No training and
# no GPU. Four CPU jobs, independent of one another:
#
#     cd ~/toolbox && git fetch && git reset --hard origin/master && bash submit_review_reruns.sh
#     ...when all four have finished:  bash collect_review.sh
#
#   stats   hiride_stats.py on the BIWI and TVRID runs, and both metric-floor runs, with the
#           joint subject bootstrap (one resample of people shared by all seeds)
#   seq-cnxt, seq-alexnet
#           hiride_sequence.py, sum rule, three protocols x gated/ungated, joint intervals and
#           the measured duration of every decision window
#   errors  hiride_errors.py, sum rule, with the observed plurality vote next to the
#           independent-frames plurality ceiling
#
# Every output keeps its previous name in $RES, so the paper's figures and tables read the
# regenerated files without any change of path.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PREP="${PREP:-$SCRATCH/hiride2/prep}"
RUNS="${RUNS:-$SCRATCH/hiride2/runs}"
IRUNS="${IRUNS:-$SCRATCH/hiride2/runs_tvrid}"
RES="${RES:-$SCRATCH/hiride2/results}"
ACC_CPU=def-czarnuch_cpu

[ -f "$PREP/prep_meta.json" ] || { echo "no certified prep at $PREP"; exit 1; }
[ -d "$IRUNS" ] || { echo "no TVRID runs at $IRUNS"; exit 1; }

cat > .review_stats.sh <<'INNER'
#!/bin/bash
#SBATCH --job-name=hiride-review-stats
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module purge; module load StdEnv/2023 python/3.11
source ~/venvs/venv311/bin/activate
step() { echo; echo "===================== $* ====================="; }
step "BIWI cells, joint bootstrap (stats_final.json)"
python hiride_stats.py --runs "$RUNS" --boot 20000 --json "$RES/stats_final.json" > "$RES/stats_final.txt"
tail -5 "$RES/stats_final.txt"
step "TVRID cells, joint bootstrap (stats_tvrid.json)"
python hiride_stats.py --runs "$IRUNS" --boot 20000 --json "$RES/stats_tvrid.json" > "$RES/stats_tvrid.txt"
tail -5 "$RES/stats_tvrid.txt"
step "metric floor, full-body test frames (metric_floor_fbtest.json)"
python hiride_metric_floor.py --prep "$PREP" --test-eligibility full_body --out "$RES"
mv "$RES/metric_floor.json" "$RES/metric_floor_fbtest.json"
step "metric floor, headline configuration (metric_floor.json)"
python hiride_metric_floor.py --prep "$PREP" --out "$RES"
echo; echo "review stats complete"
INNER

for pair in "convnext_tiny/stripe/aug8/tf10 cnxt" "alexnet/stripe/aug8/tf10 alexnet"; do
  set -- $pair
  cat > ".review_seq_$2.sh" <<INNER
#!/bin/bash
#SBATCH --job-name=hiride-review-seq-$2
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module purge; module load StdEnv/2023 python/3.11
source ~/venvs/venv311/bin/activate
for pol in R4_cross_session R4_standard_walking R4_standard_still; do
  for gate in gated ungated; do
    flag=""; [ "\$gate" = gated ] && flag="--full-body"
    echo; echo "===================== \$pol \$gate  $1  sum rule ====================="
    python hiride_sequence.py --prep "\$PREP" --runs "\$RUNS" \\
      --policy \$pol --modality depth --arch $1 --condition scale_removed \\
      \$flag --agg mean --out "\$RES" --out-name sequence_$2-mean_\${pol}_\${gate}.json
  done
done
echo; echo "review sequence $2 complete"
INNER
done

cat > .review_errors.sh <<'INNER'
#!/bin/bash
#SBATCH --job-name=hiride-review-errors
#SBATCH --time=3:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module purge; module load StdEnv/2023 python/3.11
source ~/venvs/venv311/bin/activate
for arch in alexnet/stripe/aug8/tf10 convnext_tiny/stripe/aug8/tf10; do
  for gate in gated ungated; do
    flag=""; [ "$gate" = gated ] && flag="--full-body"
    echo; echo "===================== $arch  R4_cross_session  $gate  sum rule ====================="
    python hiride_errors.py --prep "$PREP" --runs "$RUNS" \
      --policy R4_cross_session --arch $arch --condition scale_removed $flag --agg mean --out "$RES"
  done
done
echo; echo "review errors complete"
INNER

EXP="ALL,PREP=$PREP,RUNS=$RUNS,IRUNS=$IRUNS,RES=$RES"
J1=$(sbatch --parsable --account=$ACC_CPU --export="$EXP" .review_stats.sh)
J2=$(sbatch --parsable --account=$ACC_CPU --export="$EXP" .review_seq_cnxt.sh)
J3=$(sbatch --parsable --account=$ACC_CPU --export="$EXP" .review_seq_alexnet.sh)
J4=$(sbatch --parsable --account=$ACC_CPU --export="$EXP" .review_errors.sh)

cat > collect_review.sh <<EOT
#!/bin/bash
cd "\$(dirname "\$0")"
echo "=================== job states (all four must be COMPLETED) ==================="
sacct -j $J1,$J2,$J3,$J4 -X --format=JobID%12,JobName%26,State%12,Elapsed
echo
for f in logs/hiride-review-stats_$J1.out logs/hiride-review-seq-cnxt_$J2.out \\
         logs/hiride-review-seq-alexnet_$J3.out logs/hiride-review-errors_$J4.out; do
  echo "=================== \$f ==================="
  grep -E "Traceback|Error|complete\$" "\$f" || echo "(no completion line yet)"
done
EOT
chmod +x collect_review.sh

cat <<EOT

submitted (CPU only, 0 GPU-hours):
  $J1  hiride-review-stats        BIWI + TVRID stats, two metric floors      (16 cores, <= 3 h)
  $J2  hiride-review-seq-cnxt     six sequence files, pretrained CNN         (8 cores, <= 3 h)
  $J3  hiride-review-seq-alexnet  six sequence files, CNN trained from scratch (8 cores, <= 3 h)
  $J4  hiride-review-errors       four error-structure files                 (8 cores, <= 3 h)

when finished:  bash collect_review.sh
EOT
