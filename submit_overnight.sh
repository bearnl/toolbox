#!/bin/bash
# Queue the whole overnight batch with dependencies, so nothing needs a human.
#   bash submit_overnight.sh
set -euo pipefail
cd ~/toolbox
source ~/venvs/venv311/bin/activate
GPU="--account=def-czarnuch_gpu"
CPU="--account=def-czarnuch_cpu"

python make_runs.py --wave 10 > runs10.txt
python make_runs.py --wave 11 > runs11.txt
python make_runs.py --wave 12 > runs12.txt
echo "wave10=$(wc -l < runs10.txt)  wave11=$(wc -l < runs11.txt)  wave12=$(wc -l < runs12.txt)"

# 1. the head test and the speculative wave run on the EXISTING prep, now.
W10=$(sbatch --parsable $GPU --array=1-$(wc -l < runs10.txt)%8 \
      --export=ALL,RUNS_FILE=$PWD/runs10.txt run_hiride.slurm)
W12=$(sbatch --parsable $GPU --array=1-$(wc -l < runs12.txt)%8 \
      --export=ALL,RUNS_FILE=$PWD/runs12.txt run_hiride.slurm)
echo "wave10=$W10  wave12=$W12"

# 2. the boundary-fix prep, then wave 11 ONLY IF it succeeded.
EDG=$(sbatch --parsable $CPU run_hiride_edges.slurm)
W11=$(sbatch --parsable $GPU --dependency=afterok:$EDG --kill-on-invalid-dep=yes \
      --array=1-$(wc -l < runs11.txt)%8 \
      --export=ALL,RUNS_FILE=$PWD/runs11.txt,PREP=$SCRATCH/hiride2/prep_edges,OUT=$SCRATCH/hiride2/runs_edges \
      run_hiride.slurm)
echo "edges=$EDG -> wave11=$W11 (afterok)"

# 3. the probe on the rebuilt shards, also gated on the prep.
SIG=$(sbatch --parsable $CPU --dependency=afterok:$EDG --kill-on-invalid-dep=yes \
      --time=1:00:00 --mem=48000M --cpus-per-task=4 -J hiride-signal-edges \
      -o logs/hiride-signal-edges_%j.out \
      --wrap "cd ~/toolbox && source ~/venvs/venv311/bin/activate && python hiride_signal.py --prep \$SCRATCH/hiride2/prep_edges --per-class 70 --out \$SCRATCH/hiride2/results_edges")
echo "signal-on-edges=$SIG (afterok:$EDG)"
echo; echo "submitted. collect with:  bash collect_overnight.sh"
