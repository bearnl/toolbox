#!/bin/bash
# Second part of the 2026-09-23 review reruns (HIRIDE_HANDOFF 14.25). One CPU job, no GPU:
# the two pre-registered hypotheses of the manuscript (Section VI-H) with the joint subject
# bootstrap, so that every interval in the paper uses the same construction.
#
#     cd ~/toolbox && git fetch && git reset --hard origin/master && bash submit_review_aux.sh
#
#   hiride_aux.py   distance as an extra CNN input, paired against the same CNN without it
#                   -> $RES/aux_dist.txt
#   hiride_fuse.py  combination rules and the per-frame oracle against the anthropometric
#                   classifier -> $RES/fusion.json (same name as before)
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs

PREP="${PREP:-$SCRATCH/hiride2/prep}"
RUNS="${RUNS:-$SCRATCH/hiride2/runs}"
RES="${RES:-$SCRATCH/hiride2/results}"
[ -f "$PREP/prep_meta.json" ] || { echo "no certified prep at $PREP"; exit 1; }

cat > .review_aux.sh <<'INNER'
#!/bin/bash
#SBATCH --job-name=hiride-review-aux
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
module purge; module load StdEnv/2023 python/3.11
source ~/venvs/venv311/bin/activate
echo "===================== distance as an extra input (aux_dist.txt) ====================="
python hiride_aux.py --runs "$RUNS" --prep "$PREP" > "$RES/aux_dist.txt"
cat "$RES/aux_dist.txt"
echo "===================== combination rules and oracle (fusion.json) ====================="
python hiride_fuse.py --prep "$PREP" --runs "$RUNS" --out "$RES"
echo; echo "review aux complete"
INNER

J=$(sbatch --parsable --account=def-czarnuch_cpu --export="ALL,PREP=$PREP,RUNS=$RUNS,RES=$RES" .review_aux.sh)
cat <<EOT

submitted (CPU only, 0 GPU-hours):
  $J  hiride-review-aux   distance hypothesis + combination oracle   (8 cores, <= 2 h)

when finished:  grep -E "Traceback|Error|complete\$" logs/hiride-review-aux_$J.out
EOT
