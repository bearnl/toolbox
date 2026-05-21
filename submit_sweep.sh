#!/bin/bash
# Multi-seed sweep launcher for siamese.py on ComputeCanada.
#
# Submits one SLURM job per seed. Each job runs both modalities (depth + RGB)
# under the BIWI full cross-clothing protocol with the same hyperparameters,
# differing only in random seed. Use the resulting results_*.json files to
# compute mean ± std for the paper.
#
# Usage:
#   bash submit_sweep.sh                       # 5 seeds, default config (BIWI full)
#   N_SEEDS=10 bash submit_sweep.sh            # 10 seeds
#   IMG_SIZE=384 WIDTH=32 bash submit_sweep.sh # override config
#   TEST_SUBDIR=TestingB bash submit_sweep.sh  # IAS-Lab cross-clothing
#
# After all jobs finish:
#   python collate_results.py $SCRATCH/biwi-results/biwi-reid_*/results_*.json

set -euo pipefail

N_SEEDS="${N_SEEDS:-5}"
START_SEED="${START_SEED:-0}"   # First seed in the range; useful for "extend the sweep"
                                # without re-running existing seeds. e.g. START_SEED=1 N_SEEDS=4
                                # fires seeds 1,2,3,4 (skipping the already-completed seed 0).
EPOCHS="${EPOCHS:-30}"
IMG_SIZE="${IMG_SIZE:-384}"
WIDTH="${WIDTH:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
PROTOCOL="${PROTOCOL:-open-set}"  # open-set is the paper protocol (train on Training\Testing IDs, eval on Testing IDs, disjoint). 'auto' uses full overlap and triggers cross-clothing GP eval which OOMs at 384² RGB.
CLIP_RANGE="${CLIP_RANGE:-600}"   # depth slab thickness (mm). Body ~250mm; tighter (300) gives more dynamic range, may clip extended limbs.
TEST_SUBDIR="${TEST_SUBDIR:-Testing}"
HARDNEG_FREQ="${HARDNEG_FREQ:-0}"
# Modern triplet + BNNeck defaults (Luo 2019 strong baseline). Override at submit
# time, e.g.:  TRAINING_MODE=pair bash submit_sweep.sh  (legacy ablation)
#              USE_BNNECK=0 bash submit_sweep.sh        (triplet-only ablation)
TRAINING_MODE="${TRAINING_MODE:-triplet}"
USE_BNNECK="${USE_BNNECK:-1}"
CE_WEIGHT="${CE_WEIGHT:-1.0}"
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"
PK_P="${PK_P:-8}"
PK_K="${PK_K:-4}"
TRIPLET_MARGIN="${TRIPLET_MARGIN:-0.3}"
EMB_DIM="${EMB_DIM:-256}"
PROBE_WINDOW="${PROBE_WINDOW:-10}"
# Backbone — smallresnet (v1, train-from-scratch) | convnext_tiny | efficientnet_b0 |
# efficientnet_b2 | resnet50v2. The pretrained options channel-average the first
# Conv2D so depth stays genuinely 1-channel.
BACKBONE="${BACKBONE:-smallresnet}"
# Pretrained init for applications backbones: 1=ImageNet transfer, 0=from scratch.
PRETRAINED="${PRETRAINED:-1}"
# Anthropometric aux loss weight (depth only). 0 disables; 0.1-0.5 recommended.
ANTHRO_WEIGHT="${ANTHRO_WEIGHT:-0.0}"
# Pose canonicalization (depth only): 1 enables PCA-based body-tilt removal at preprocess time.
CANONICALIZE_POSE="${CANONICALIZE_POSE:-0}"
# 3D variants (depth only).
ANTHRO_3D="${ANTHRO_3D:-0}"
CANONICALIZE_POSE_3D="${CANONICALIZE_POSE_3D:-0}"
# Temporal-stack depth input (depth only). FRAMES=1 disables; 3 = 3-frame stack.
TEMPORAL_STACK_FRAMES="${TEMPORAL_STACK_FRAMES:-1}"
TEMPORAL_STACK_STRIDE="${TEMPORAL_STACK_STRIDE:-5}"

mkdir -p logs

echo "Submitting $N_SEEDS jobs (seeds $START_SEED..$((START_SEED + N_SEEDS - 1))): img=$IMG_SIZE width=$WIDTH epochs=$EPOCHS protocol=$PROTOCOL test=$TEST_SUBDIR"
echo "  mode=$TRAINING_MODE  backbone=$BACKBONE  pretrained=$PRETRAINED  bnneck=$USE_BNNECK  ce_weight=$CE_WEIGHT  P=$PK_P K=$PK_K m=$TRIPLET_MARGIN"
echo "  anthro_w=$ANTHRO_WEIGHT  canonicalize_pose=$CANONICALIZE_POSE  anthro_3d=$ANTHRO_3D  canon_3d=$CANONICALIZE_POSE_3D  clip=$CLIP_RANGE  tstack=${TEMPORAL_STACK_FRAMES}x${TEMPORAL_STACK_STRIDE}"

for seed in $(seq $START_SEED $((START_SEED + N_SEEDS - 1))); do
  jobname="biwi_${BACKBONE}_pt${PRETRAINED}_size${IMG_SIZE}_w${WIDTH}_${TRAINING_MODE}_bn${USE_BNNECK}_aw${ANTHRO_WEIGHT}_a3d${ANTHRO_3D}_c3d${CANONICALIZE_POSE_3D}_clip${CLIP_RANGE}_tstack${TEMPORAL_STACK_FRAMES}_s${seed}"
  sbatch \
    --job-name="$jobname" \
    --export=ALL,SEED=$seed,EPOCHS=$EPOCHS,IMG_SIZE=$IMG_SIZE,WIDTH=$WIDTH,BATCH_SIZE=$BATCH_SIZE,PROTOCOL=$PROTOCOL,TEST_SUBDIR=$TEST_SUBDIR,HARDNEG_FREQ=$HARDNEG_FREQ,TRAINING_MODE=$TRAINING_MODE,USE_BNNECK=$USE_BNNECK,CE_WEIGHT=$CE_WEIGHT,LABEL_SMOOTHING=$LABEL_SMOOTHING,PK_P=$PK_P,PK_K=$PK_K,TRIPLET_MARGIN=$TRIPLET_MARGIN,EMB_DIM=$EMB_DIM,PROBE_WINDOW=$PROBE_WINDOW,BACKBONE=$BACKBONE,PRETRAINED=$PRETRAINED,ANTHRO_WEIGHT=$ANTHRO_WEIGHT,CANONICALIZE_POSE=$CANONICALIZE_POSE,ANTHRO_3D=$ANTHRO_3D,CANONICALIZE_POSE_3D=$CANONICALIZE_POSE_3D,CLIP_RANGE=$CLIP_RANGE,TEMPORAL_STACK_FRAMES=$TEMPORAL_STACK_FRAMES,TEMPORAL_STACK_STRIDE=$TEMPORAL_STACK_STRIDE \
    run_biwi.slurm
done

echo "Submitted. Monitor with: squeue -u \$USER"
