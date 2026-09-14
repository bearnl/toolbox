#!/bin/bash
# Error-structure analysis (hiride_errors.py) for the two best-recipe cells at R4, gated and
# ungated -- one CPU job, four JSONs, no training. Licenses (or forbids) the word "systematic"
# in Section VI: product rule vs arithmetic mean, float16 veto count, vote consistency, i.i.d.
# ceiling, confuser stability, both-wrong enrichment, early-stopping cost (HIRIDE_HANDOFF 14.9).
# The decision rule that defines wrong windows is the sum rule (--agg mean), the paper's rule
# since 2026-09-13 (HIRIDE_HANDOFF 14.15); pass geo as the first argument to re-run the product rule.
#
#     bash submit_errors.sh            # sum rule
#     bash submit_errors.sh geo        # product rule
set -euo pipefail
cd "$(dirname "$0")"
AGG=${1:-mean}
mkdir -p logs
cat > .errors.sh <<EOF
#!/bin/bash
#SBATCH --job-name=hiride-errors
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
module purge; module load StdEnv/2023 python/3.11
source ~/venvs/venv311/bin/activate
R=\$SCRATCH/hiride2/results
for arch in alexnet/stripe/aug8/tf10 convnext_tiny/stripe/aug8/tf10; do
  for gate in gated ungated; do
    flag=""; [ "\$gate" = gated ] && flag="--full-body"
    echo; echo "===================== \$arch  R4_cross_session  \$gate  ${AGG} rule ====================="
    python hiride_errors.py --prep \$SCRATCH/hiride2/prep --runs \$SCRATCH/hiride2/runs \\
      --policy R4_cross_session --arch \$arch --condition scale_removed \$flag --agg ${AGG} --out \$R
  done
done
EOF
sbatch --account=def-czarnuch_cpu .errors.sh
