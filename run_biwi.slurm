#!/bin/bash
# ComputeCanada SLURM job template for siamese.py
#
# The dataset stays as .rar archives on nearline storage (one file per archive,
# trivial quota cost). At the start of each job we extract them to $SLURM_TMPDIR,
# which is node-local SSD scratch that doesn't count against any user quota and
# is auto-wiped when the job exits. So nothing on persistent storage ever has
# the >100k extracted files.
#
# Usage:
#   sbatch --account=def-czarnuch --export=ALL,SEED=0 run_biwi.slurm
#   sbatch --account=def-czarnuch --export=ALL,SEED=1,IMG_SIZE=384,WIDTH=32 run_biwi.slurm

#SBATCH --account=def-czarnuch
#SBATCH --time=16:00:00
# Was 8:00:00. ConvNeXt-Tiny both-modality at 384² takes ~6.5h total; with cache
# build + GP-eval + safety margin, 8h was tight enough to TIMEOUT under load.
# 16h covers ConvNeXt-Base too (heavier backbone). Override at submit:
#   sbatch --time=4:00:00 ... run_biwi.slurm   (for smallresnet which only needs ~1h)
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
# Nibi GPU lineup: full h100, a100, a5000, t4, plus H100 MIG slices.
# MIG slice 3g.40gb = 3/7 of an H100 (~40GB VRAM, ~21-30 TFLOPS FP32). Needed
# for ConvNeXt-Tiny RGB at 384²+batch=32: deepest-stage activation alone is
# 32×1536×24×24×4 = 113 MB and the backward pass needs many such tensors plus
# optimizer state — 2g.20gb (20GB) OOMs reliably. 3g.40gb fits ConvNeXt-Tiny/
# Small/Base RGB+depth at batch=32 with comfortable headroom.
# For smallresnet jobs you can drop back to 2g.20gb or 1g.10gb at submit time:
#   sbatch --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1 ... run_biwi.slurm
#SBATCH --mem=64000M
# Use M (megabytes) to avoid Slurm's G=1024M / 1000M ambiguity warning.
# Cache budget at 384²:
#   open-set (22 train IDs ≈ 10k images): depth ~1.5GB + RGB ~4.5GB + val ~7GB
#       + TF runtime ~8GB + rerank matrices ~3GB ≈ 24GB peak. 64GB ample.
#   auto (50 train IDs ≈ 24k images): depth ~3.5GB + RGB ~11GB + val ~7GB
#       + TF runtime ~8GB + GP-eval temporaries ~5GB + rerank ~3GB ≈ 38GB peak.
#       40GB was tight enough to OOM (Linux kills at ~85% RSS); 64GB safe.
# At 256², 32GB would suffice.
#SBATCH --cpus-per-task=4
#SBATCH --job-name=biwi-reid
#SBATCH --output=logs/%x_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail
mkdir -p logs

# -----------------------------------------------------------------------------
# Defaults — override via --export=ALL,VAR=value
# -----------------------------------------------------------------------------
SEED="${SEED:-0}"
EPOCHS="${EPOCHS:-30}"
IMG_SIZE="${IMG_SIZE:-384}"                  # was 256; 384 + BNNeck gives best paper numbers
WIDTH="${WIDTH:-32}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-1e-4}"
CLIP_RANGE="${CLIP_RANGE:-600}"
PROTOCOL="${PROTOCOL:-auto}"                # auto | disjoint | full | open-set
TEST_SUBDIR="${TEST_SUBDIR:-Testing}"       # for IAS-Lab: TestingA / TestingB / TestingC
MODALITY="${MODALITY:-both}"
HARDNEG_FREQ="${HARDNEG_FREQ:-0}"
# Modern siamese knobs (Hermans 2017 batch-hard triplet recipe)
TRAINING_MODE="${TRAINING_MODE:-triplet}"   # was pair; triplet + BNNeck is the strong baseline
PK_P="${PK_P:-8}"                            # identities per batch in triplet mode
PK_K="${PK_K:-4}"                            # images per identity per batch (batch_size = P*K)
TRIPLET_MARGIN="${TRIPLET_MARGIN:-0.3}"      # margin for batch-hard triplet
EMB_DIM="${EMB_DIM:-256}"                    # embedding dim (recommended 512 for triplet)
PROBE_WINDOW="${PROBE_WINDOW:-10}"           # N-frame multi-frame probe averaging at eval
# Hybrid-loss knobs (Luo 2019 BNNeck strong baseline)
USE_BNNECK="${USE_BNNECK:-1}"                # 1=enable BNNeck head, 0=disable (triplet-only)
CE_WEIGHT="${CE_WEIGHT:-1.0}"                # cross-entropy weight on aux classifier; 0 disables
LABEL_SMOOTHING="${LABEL_SMOOTHING:-0.1}"    # label-smoothing ε for CE loss
# Backbone selector. smallresnet (default, train-from-scratch v1) | convnext_tiny |
# convnext_small | convnext_base | efficientnet_b0 | efficientnet_b2 | resnet50v2
# (pretrained options use tf.keras.applications with channel-averaged first conv).
BACKBONE="${BACKBONE:-smallresnet}"
# Pretrained init for applications backbones: 1=ImageNet (transfer), 0=random (from scratch).
PRETRAINED="${PRETRAINED:-1}"
# Anthropometric aux-loss weight (depth only). 0 disables. Recommended 0.1-0.5.
ANTHRO_WEIGHT="${ANTHRO_WEIGHT:-0.0}"
# Pose canonicalization (depth only): 1 enables PCA-based body-tilt removal.
CANONICALIZE_POSE="${CANONICALIZE_POSE:-0}"
# 3D variants (depth only). Need Kinect intrinsics — defaults are standard Kinect v1.
ANTHRO_3D="${ANTHRO_3D:-0}"             # 1 = 3D-derived anthropometric features (9-dim, mm units)
CANONICALIZE_POSE_3D="${CANONICALIZE_POSE_3D:-0}"  # 1 = 3D PCA + re-project canonicalization
# Temporal-stack depth input (depth-only). FRAMES=1 disables; 3 = 3-frame stack as channels.
TEMPORAL_STACK_FRAMES="${TEMPORAL_STACK_FRAMES:-1}"
TEMPORAL_STACK_STRIDE="${TEMPORAL_STACK_STRIDE:-5}"

# Where the .rar archives live (must be compute-accessible — /nearline is NOT
# mounted on compute nodes, only login nodes, so .rar files must first be copied
# from /nearline to /home, /scratch, or /project once before any training job
# runs).  Expect <RAR_DIR>/Training.rar and (optionally) <RAR_DIR>/Testing.rar.
RAR_DIR="${RAR_DIR:-$SCRATCH/datasets}"

# Where to write checkpoints and results.json.
OUT_DIR="${OUT_DIR:-$SCRATCH/biwi-results/${SLURM_JOB_NAME}_${SLURM_JOB_ID}}"
mkdir -p "$OUT_DIR"

# -----------------------------------------------------------------------------
# Environment
# -----------------------------------------------------------------------------
module purge
module load StdEnv/2023 python/3.11 cuda cudnn opencv
# opencv module provides cv2; saves us from a 75MB pip install + ensures it's
# built against the same CUDA we load. TF and scikit-learn still come from the
# venv (pip-installed once into ~/venvs/venv311).
nvidia-smi || true
# Static unrar binary (handles rar5; system 7z on Nibi is too old).
export PATH="$HOME/.local/rar:$HOME/bin/rar:$HOME/bin:$PATH"
source "$HOME/venvs/venv311/bin/activate"

# Pick an extractor. Prefer unrar (correct for any rar version). Fall back to 7z.
# Removed silencing flags so we see actual errors in the log.
if command -v unrar >/dev/null 2>&1; then
    EXTRACT="unrar x -y"
    echo "[$(date)] Using unrar: $(unrar 2>&1 | head -1)"
elif command -v 7z >/dev/null 2>&1; then
    EXTRACT="7z x -y"
    echo "[$(date)] Using 7z: $(7z 2>&1 | head -3 | tail -1)"
elif command -v bsdtar >/dev/null 2>&1; then
    EXTRACT="bsdtar -xf"
    echo "[$(date)] Using bsdtar (libarchive): $(bsdtar --version)"
else
    echo "ERROR: no rar extractor available (tried: unrar, 7z, bsdtar)." >&2
    echo "       Available modules to try:" >&2
    module spider unrar 2>&1 | head >&2 || true
    module spider 7zip 2>&1 | head >&2 || true
    exit 1
fi

# -----------------------------------------------------------------------------
# Stage data: copy .rar to local SSD, extract there.
# $SLURM_TMPDIR is node-local, NOT counted against home/project quotas,
# auto-cleaned when this job exits.
# -----------------------------------------------------------------------------
DATA_DIR="$SLURM_TMPDIR/biwi"
mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

echo "[$(date)] Copying .rar archives from $RAR_DIR ..."
for rar in "$RAR_DIR"/*.rar; do
    cp "$rar" .
done

echo "[$(date)] Extracting (using: $EXTRACT) ..."
for rar in *.rar; do
    echo "  $rar"
    $EXTRACT "$rar"
done
rm -f *.rar    # free local SSD

echo "[$(date)] Extracted layout (top of $DATA_DIR):"
ls "$DATA_DIR"
echo "[$(date)] File counts: rgb=$(find "$DATA_DIR" -name '*_rgb.jpg' | wc -l)  depth=$(find "$DATA_DIR" -name '*_depth.pgm' | wc -l)"

# -----------------------------------------------------------------------------
# Train
# -----------------------------------------------------------------------------
RUN_TAG="${SLURM_JOB_NAME}_s${SEED}_size${IMG_SIZE}_w${WIDTH}_${TRAINING_MODE}_${BACKBONE}"
cd "$SLURM_SUBMIT_DIR"

# Build bnneck flag conditional on USE_BNNECK=0|1.
if [ "$USE_BNNECK" = "0" ] || [ "$USE_BNNECK" = "false" ] || [ "$USE_BNNECK" = "False" ]; then
    BNNECK_FLAG="--no-bnneck"
else
    BNNECK_FLAG="--use-bnneck"
fi

# Build canonicalize-pose flag conditional on CANONICALIZE_POSE=0|1.
if [ "$CANONICALIZE_POSE" = "1" ] || [ "$CANONICALIZE_POSE" = "true" ] || [ "$CANONICALIZE_POSE" = "True" ]; then
    CANON_FLAG="--canonicalize-pose"
else
    CANON_FLAG=""
fi

# 3D variants — same pattern.
if [ "$ANTHRO_3D" = "1" ] || [ "$ANTHRO_3D" = "true" ] || [ "$ANTHRO_3D" = "True" ]; then
    ANTHRO_3D_FLAG="--anthro-3d"
else
    ANTHRO_3D_FLAG=""
fi
if [ "$CANONICALIZE_POSE_3D" = "1" ] || [ "$CANONICALIZE_POSE_3D" = "true" ] || [ "$CANONICALIZE_POSE_3D" = "True" ]; then
    CANON3D_FLAG="--canonicalize-pose-3d"
else
    CANON3D_FLAG=""
fi
if [ "$PRETRAINED" = "0" ] || [ "$PRETRAINED" = "false" ] || [ "$PRETRAINED" = "False" ]; then
    PRETRAINED_FLAG="--no-pretrained"
else
    PRETRAINED_FLAG=""
fi

python -u siamese.py \
  --base-dir   "$DATA_DIR" \
  --protocol   "$PROTOCOL" \
  --test-subdir "$TEST_SUBDIR" \
  --modality   "$MODALITY" \
  --seed       "$SEED" \
  --epochs     "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --lr         "$LR" \
  --img-size   "$IMG_SIZE" \
  --width      "$WIDTH" \
  --emb-dim    "$EMB_DIM" \
  --clip-range "$CLIP_RANGE" \
  --hardneg-freq "$HARDNEG_FREQ" \
  --training-mode "$TRAINING_MODE" \
  --pk-p       "$PK_P" \
  --pk-k       "$PK_K" \
  --triplet-margin "$TRIPLET_MARGIN" \
  --probe-window "$PROBE_WINDOW" \
  $BNNECK_FLAG \
  --ce-weight  "$CE_WEIGHT" \
  --label-smoothing "$LABEL_SMOOTHING" \
  --backbone   "$BACKBONE" \
  $PRETRAINED_FLAG \
  --anthro-weight "$ANTHRO_WEIGHT" \
  $CANON_FLAG \
  $ANTHRO_3D_FLAG \
  $CANON3D_FLAG \
  --temporal-stack-frames "$TEMPORAL_STACK_FRAMES" \
  --temporal-stack-stride "$TEMPORAL_STACK_STRIDE" \
  --out-dir    "$OUT_DIR" \
  --run-tag    "$RUN_TAG"

echo "[$(date)] Job complete. Results at $OUT_DIR"
