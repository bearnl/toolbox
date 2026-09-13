#!/bin/bash
# Aggregate one CNN cell's stored posteriors with the 12 metric features over observation windows,
# gated and ungated, at the ladder's R4 and both standard rungs -- one CPU job, six JSONs.
#
#     bash submit_sequence.sh convnext_tiny/stripe/aug8/tf10 cnxt
#     bash submit_sequence.sh alexnet/stripe/aug8/tf10 alexnet
#
# Writes $RES/sequence_<TAG>_<policy>_<gated|ungated>.json (hiride_sequence.py; no training).
set -euo pipefail
cd "$(dirname "$0")"
ARCH="${1:?arch key, e.g. convnext_tiny/stripe/aug8/tf10}"
TAG="${2:?short tag for the output files}"
mkdir -p logs
cat > ".seq_${TAG}.sh" <<EOF
#!/bin/bash
#SBATCH --job-name=hiride-seq-${TAG}
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000M
#SBATCH --output=logs/%x_%j.out
set -uo pipefail
module purge; module load StdEnv/2023 python/3.11
source ~/venvs/venv311/bin/activate
R=\$SCRATCH/hiride2/results
for pol in R4_cross_session R4_standard_walking R4_standard_still; do
  for gate in gated ungated; do
    flag=""; [ "\$gate" = gated ] && flag="--full-body"
    echo; echo "===================== \$pol \$gate  ${ARCH} ====================="
    python hiride_sequence.py --prep \$SCRATCH/hiride2/prep --runs \$SCRATCH/hiride2/runs \\
      --policy \$pol --modality depth --arch ${ARCH} --condition scale_removed \\
      \$flag --out \$R --out-name sequence_${TAG}_\${pol}_\${gate}.json
  done
done
EOF
sbatch --account=def-czarnuch_cpu ".seq_${TAG}.sh"
