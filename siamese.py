import os
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import layers, Model, applications
import random
from collections import defaultdict
from sklearn.metrics import pairwise_distances
import gc
import argparse
import json
import time
import zipfile
import tempfile

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

# Defaults — overridable via CLI in __main__.
# All values used by functions are read at call time, so the CLI block may mutate
# these globals after parsing and before training begins.
SEED = 2025
IMG_H, IMG_W = 480, 480
EMB_DIM = 256
DROPOUT_RATE = 0.2
MARGIN_DEPTH = 0.5
MARGIN_RGB = 0.5
BATCH_SIZE = 32
EPOCHS = 15
LR = 0.0001
LR_MIN = 5e-5
WEIGHT_DECAY_DEPTH = 1e-4
WEIGHT_DECAY_RGB = 1e-4
BASE_DIR = "/mnt/g/biwi"
USE_TRIPLET_LOSS = True
USE_REDUCE_LR_ON_PLATEAU = True
CLIP_RANGE = 600.0
FOREGROUND_CLOSEST_PERCENTILE = 1
MIN_FOREGROUND_PIXELS = 100
CROP_EXPAND = 1.8
CROP_STD_K = 2.5
EVAL_BATCH_SIZE = 16  # was 32 — reduced to halve peak predict-time VRAM on H100 MIG 1g.10gb (10 GB slice).

# Knobs only set via CLI (no module-default — see __main__).
RESNET_BASE_WIDTH = 16              # backbone base channels
AUGMENT = True                      # data augmentation toggle
HARDNEG_FREQ = 0                    # 0 disables; >0 = refresh every N epochs (pair mode only)
PATIENCE_ES = 8                     # EarlyStopping patience
PATIENCE_RLR = 3                    # ReduceLROnPlateau patience
OUT_DIR = "."                       # where to write weights + results.json
RUN_TAG = "default"                 # tag included in checkpoint filenames + JSON
# Triplet-mode (PK batch sampling + batch-hard triplet, Hermans 2017) knobs.
TRAINING_MODE = "pair"              # "pair" (legacy SiamesePairGenerator) or "triplet" (PK + batch-hard)
PK_P = 8                            # number of identities per batch
PK_K = 4                            # number of images per identity per batch (batch = P*K)
PK_STEPS_PER_EPOCH = None           # None = auto (~ len(unique_labels)//P * K)
TRIPLET_MARGIN = 0.3                # margin for batch-hard triplet (cosine distance space)
# Hybrid loss (BNNeck + auxiliary classification, Luo 2019). Only meaningful in triplet mode.
USE_BNNECK = True                   # Use BNNeck head (pre-BN ft for triplet, post-BN fi for CE+retrieval)
CE_WEIGHT = 1.0                     # Cross-entropy loss weight. 0 disables aux classifier.
LABEL_SMOOTHING = 0.1               # CE label smoothing ε (Luo 2019 recipe).
ANTHRO_WEIGHT = 0.0                 # Anthropometric MSE aux-loss weight. 0 disables the head.
                                    # Depth-only intervention — the head doesn't fire for RGB
                                    # since RGB doesn't have pose-invariant silhouette stats
                                    # that are meaningful from a 3-channel colour image.
CANONICALIZE_POSE = False           # Depth-only: PCA-align foreground principal axis to vertical
                                    # in the normalized [-1, 1] depth image, before any cached
                                    # rotation augmentation. Reduces body-tilt as a pose variable.
# 3D variants — exploit that depth is a 2.5D scalar field (per-pixel Z in mm),
# not just a silhouette. Both use Kinect intrinsics to unproject.
ANTHRO_3D = False                   # Depth-only: replace 2D-silhouette anthropometrics with
                                    # 3D-derived features (body height/width/depth in mm,
                                    # body volume, 3D aspect). Captures Z-thickness, which the
                                    # silhouette features ignored. ANTHRO_DIM becomes 9 when on.
CANONICALIZE_POSE_3D = False        # Depth-only: PCA-align in 3D space (using unprojected
                                    # point cloud), then re-project to a depth image. Captures
                                    # forward-lean / tilt-toward-camera that 2D PCA can't see.
# Temporal-stack depth input (depth-only). When >1, each input sample is a stack
# of N depth frames at temporal offsets [0, stride, 2*stride, ...] presented as
# channels. Captures gait/motion. Single-modality: every channel is depth from
# the same sensor, just at adjacent timestamps. With N=3 we use ConvNeXt's native
# 3-channel ImageNet weights directly (no 1ch adapter needed — cleaner transfer).
TEMPORAL_STACK_FRAMES = 1           # 1 = single-frame (current behavior); 3 = temporal stack
TEMPORAL_STACK_STRIDE = 5           # frames between stack elements (at 30 FPS, 5 ≈ 167 ms)
# Kinect v1 depth-camera intrinsics at 640×480 (Munaro 2014 used Kinect v1 for BIWI).
# Override at command line with --kinect-fx etc. if BIWI calibration specifies different values.
KINECT_FX = 575.816
KINECT_FY = 575.816
KINECT_CX = 320.0
KINECT_CY = 240.0
# Backbone selection. "smallresnet" = trained from scratch (v1). Other options use
# tf.keras.applications with ImageNet-pretrained weights; for 1-channel depth the
# first Conv2D's input-channel weights are averaged from 3 → 1 so the model is
# genuinely single-channel.
BACKBONE = "smallresnet"            # smallresnet | convnext_tiny | efficientnet_b0 | efficientnet_b2 | resnet50v2
USE_PRETRAINED = True               # for applications backbones: True = ImageNet weights (transfer),
                                    # False = random init (from scratch). Lets us isolate the
                                    # architecture from the pretraining in a clean 2×2 ablation.

def _set_seeds(seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

_set_seeds(SEED)

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"GPUs Found: {len(gpus)}")
    except RuntimeError as e:
        print(e)

def get_augmentation_params():
    # Random-erasing rectangle parameters (Zhong et al. 2017, ECCV). Standard
    # cross-clothing Re-ID augmentation: occludes a random rectangle of the image
    # so the model can't rely on any single body region's clothing for identity.
    erase_apply = np.random.rand() < 0.5
    target_area = np.random.uniform(0.02, 0.35) * IMG_H * IMG_W
    aspect = np.random.uniform(0.3, 3.3)
    erase_h = int(round(np.sqrt(target_area * aspect)))
    erase_w = int(round(np.sqrt(target_area / aspect)))
    erase_h = min(max(erase_h, 1), IMG_H - 1)
    erase_w = min(max(erase_w, 1), IMG_W - 1)
    erase_y = np.random.randint(0, IMG_H - erase_h + 1)
    erase_x = np.random.randint(0, IMG_W - erase_w + 1)
    return {
        # Stronger geometric aug: rotation ±15° (was ±5°) and scale 0.7-1.0 (was 0.9-1.0)
        # to simulate the body-pose variation we need for cross-pose Re-ID
        # (Still ↔ Walking). Also adds small random translation (±5% of frame size).
        # Helps depth particularly since silhouette is the only identity signal — the
        # model must learn invariance to whole-body rotation/scale/translation, not
        # just rely on the subject being centered at consistent pose.
        'angle': np.random.uniform(-15, 15),
        'scale': np.random.uniform(0.7, 1.0),
        'tx': float(np.random.uniform(-0.05, 0.05) * IMG_W),
        'ty': float(np.random.uniform(-0.05, 0.05) * IMG_H),
        'flip': np.random.rand() > 0.5,
        'noise': np.random.normal(0, 0.01, (IMG_H, IMG_W, 1)).astype(np.float32),
        'rgb_alpha': np.random.uniform(0.9, 1.1),
        'rgb_beta': np.random.uniform(-10.0, 10.0),
        # Aggressive colour jitter — forces RGB to learn clothing-COLOUR-invariant features.
        # Hue is the dominant clothing signal in BIWI; without large shifts here the model
        # just memorises "subject X = red shirt" and fails on cross-clothing eval.
        # OpenCV hue range is 0..179 (180 values). ±90 = half the colour wheel.
        'rgb_hue_shift': int(np.random.uniform(-90, 90)),
        'rgb_sat_scale': float(np.random.uniform(0.4, 1.6)),
        # Optional: random channel permutation (probability 0.5) — even more aggressive
        # colour-invariance forcing. Each call picks one of 6 permutations.
        'rgb_channel_perm': tuple(np.random.permutation(3)) if np.random.rand() < 0.5 else (0, 1, 2),
        # Random erasing — Cutout-style occlusion of a random rectangle, fills with
        # uniform-random pixel values. Standard for clothing-change Re-ID.
        'rgb_erase_apply': bool(erase_apply),
        'rgb_erase_y': int(erase_y),
        'rgb_erase_x': int(erase_x),
        'rgb_erase_h': int(erase_h),
        'rgb_erase_w': int(erase_w),
        'rgb_erase_fill': tuple(int(np.random.randint(0, 256)) for _ in range(3)),
    }

def apply_depth_augmentation(img, params):
    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D((w/2, h/2), params['angle'], 1.0)
    M[0, 2] += params.get('tx', 0.0)
    M[1, 2] += params.get('ty', 0.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT)

    scale = params['scale']
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
        img = cv2.copyMakeBorder(img, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)

    if params['flip']:
        img = cv2.flip(img, 1)

    return img

def apply_rgb_augmentation(img, params):
    h, w = img.shape[:2]

    M = cv2.getRotationMatrix2D((w/2, h/2), params['angle'], 1.0)
    M[0, 2] += params.get('tx', 0.0)
    M[1, 2] += params.get('ty', 0.0)
    img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

    scale = params['scale']
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
        img = cv2.copyMakeBorder(img, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w, cv2.BORDER_REFLECT)

    if params['flip']:
        img = cv2.flip(img, 1)

    # Brightness/contrast jitter.
    img = cv2.convertScaleAbs(img, alpha=params['rgb_alpha'], beta=params['rgb_beta'])

    # HSV: aggressive hue + saturation shift to destroy colour-as-identity cue.
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + params['rgb_hue_shift']) % 180
    hsv[..., 1] = np.clip(hsv[..., 1] * params['rgb_sat_scale'], 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Channel permutation (50% prob) — completely scrambles colour identity.
    perm = params['rgb_channel_perm']
    if perm != (0, 1, 2):
        img = img[..., list(perm)]

    # Random erasing — occlude a rectangle so the model can't rely on one body region.
    if params['rgb_erase_apply']:
        y, x = params['rgb_erase_y'], params['rgb_erase_x']
        h, w = params['rgb_erase_h'], params['rgb_erase_w']
        img = img.copy()  # don't mutate the cached uint8 image
        img[y:y + h, x:x + w] = np.array(params['rgb_erase_fill'], dtype=np.uint8)

    return img

def apply_cached_depth_augmentation(img_2d, params):
    h, w = img_2d.shape

    M = cv2.getRotationMatrix2D((w/2, h/2), params['angle'], 1.0)
    M[0, 2] += params.get('tx', 0.0)
    M[1, 2] += params.get('ty', 0.0)
    img = cv2.warpAffine(img_2d, M, (w, h), flags=cv2.INTER_NEAREST,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=-1.0)

    scale = params['scale']
    if scale < 1.0:
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        pad_h, pad_w = (h - new_h) // 2, (w - new_w) // 2
        img = cv2.copyMakeBorder(img, pad_h, h - new_h - pad_h, pad_w, w - new_w - pad_w,
                                 cv2.BORDER_CONSTANT, value=-1.0)

    if params['flip']:
        img = cv2.flip(img, 1)

    fg = img > -0.99
    img = img + params['noise'][:, :, 0] * fg
    img = np.clip(img, -1.0, 1.0)
    return img

def preprocess_depth_smart(img, augment=False, aug_params=None):
    img = img.astype(np.float32)

    if augment and aug_params is not None:
        img = apply_depth_augmentation(img, aug_params)

    H, W = img.shape
    valid_mask = img > 0
    if np.sum(valid_mask) < MIN_FOREGROUND_PIXELS:
        return np.full((IMG_H, IMG_W, 1), -1.0, dtype=np.float32)

    anchor = np.percentile(img[valid_mask], FOREGROUND_CLOSEST_PERCENTILE)
    fg_mask = valid_mask & (img >= anchor) & (img <= anchor + CLIP_RANGE)
    if np.sum(fg_mask) < MIN_FOREGROUND_PIXELS:
        return np.full((IMG_H, IMG_W, 1), -1.0, dtype=np.float32)

    # 3D pose canonicalization: rotate body in 3D space, re-project to 2D depth.
    # Applied on the raw mm depth using the foreground slab as the body mask.
    # After this, we re-extract the foreground (rotation may have shifted things)
    # and continue with the standard crop+resize+normalize pipeline.
    if CANONICALIZE_POSE_3D:
        img, fg_mask = _canonicalize_pose_3d(img, fg_mask)
        # Re-derive valid_mask + anchor in case re-projection shifted depth values
        valid_mask = img > 0
        if np.sum(valid_mask) < MIN_FOREGROUND_PIXELS or np.sum(fg_mask) < MIN_FOREGROUND_PIXELS:
            return np.full((IMG_H, IMG_W, 1), -1.0, dtype=np.float32)
        # Update anchor — rotation may have changed the closest-point depth slightly
        anchor = np.percentile(img[valid_mask], FOREGROUND_CLOSEST_PERCENTILE)

    ys, xs = np.where(fg_mask)
    cy, cx = float(ys.mean()), float(xs.mean())
    sy = max(float(ys.std()) * CROP_STD_K, H * 0.15)
    sx = max(float(xs.std()) * CROP_STD_K, W * 0.10)
    side = int(min(max(sy, sx) * CROP_EXPAND, min(H, W)))
    x0 = int(np.clip(cx - side / 2, 0, W - side))
    y0 = int(np.clip(cy - side / 2, 0, H - side))

    img_crop = img[y0:y0 + side, x0:x0 + side]
    fg_crop = fg_mask[y0:y0 + side, x0:x0 + side]

    img_resized = cv2.resize(img_crop, (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST)
    fg_resized = cv2.resize(fg_crop.astype(np.uint8), (IMG_W, IMG_H), interpolation=cv2.INTER_NEAREST).astype(bool)

    img_normalized = np.full_like(img_resized, -1.0)
    img_normalized[fg_resized] = ((img_resized[fg_resized] - anchor) / CLIP_RANGE) * 2.0 - 1.0

    # Pose canonicalization: rotate normalized depth so foreground principal axis
    # is vertical. Applied BEFORE noise so noise stays uncorrelated with rotation.
    # Augmentation rotation in apply_cached_depth_augmentation then acts on this
    # canonical baseline, giving the model rotation variance from a consistent
    # starting orientation (vs the natural-tilt variation otherwise).
    if CANONICALIZE_POSE:
        img_normalized = _canonicalize_pose_2d(img_normalized)

    if augment and aug_params is not None:
        img_normalized = img_normalized + aug_params['noise'][:, :, 0]
        img_normalized = np.clip(img_normalized, -1.0, 1.0)

    return np.expand_dims(img_normalized, -1)

ANTHRO_DIM = 6   # 2D silhouette anthros (when ANTHRO_3D=False). See _compute_anthropometrics.
ANTHRO_DIM_3D = 9  # 3D-derived anthros (when ANTHRO_3D=True). See _compute_anthropometrics_3d.


def _intrinsics_for_image_shape(H, W):
    """Return Kinect intrinsics scaled to a given image resolution. The cached
    KINECT_* globals are calibrated for the native 640×480 depth image; if we
    operate on a different resolution (e.g., a crop or resize), the intrinsics
    must scale proportionally."""
    scale_x = W / 640.0
    scale_y = H / 480.0
    return (KINECT_FX * scale_x, KINECT_FY * scale_y,
            KINECT_CX * scale_x, KINECT_CY * scale_y)


def _unproject_depth_to_3d(depth_2d_mm, fg_mask, fx, fy, cx, cy):
    """Convert a 2D depth image (raw mm values) + foreground mask + intrinsics
    into a 3D point cloud of the foreground pixels.

    Args:
        depth_2d_mm: (H, W) depth in mm. Background pixels may be 0.
        fg_mask: (H, W) bool foreground mask.
        fx, fy, cx, cy: pinhole intrinsics (in pixels), matched to image shape.

    Returns:
        (N, 3) array of (X, Y, Z) coordinates in mm. Y increases downward
        (image convention), X to the right, Z away from the camera.
    """
    ys, xs = np.where(fg_mask)
    if len(ys) == 0:
        return np.empty((0, 3), dtype=np.float32)
    zs = depth_2d_mm[ys, xs].astype(np.float32)
    Xs = zs * (xs.astype(np.float32) - cx) / fx
    Ys = zs * (ys.astype(np.float32) - cy) / fy
    return np.stack([Xs, Ys, zs], axis=1)


def _compute_anthropometrics(depth_img_2d, fg_threshold=-0.99):
    """Extract pose-mostly-invariant body-shape statistics from a preprocessed
    depth frame. Returns a 6-element vector in [0, 1] suitable as an auxiliary
    regression target.

    These features are computed from the foreground silhouette of a depth
    image that's already been through `preprocess_depth_smart` (i.e., cropped,
    resized to IMG_H × IMG_W, foreground=values > fg_threshold, background=−1).

    Features:
      0. body_height_norm    — silhouette y-extent / IMG_H. Captures stature.
      1. width_25_norm       — silhouette x-extent at y = ymin + 0.25*body_height,
                                divided by IMG_W. Captures lower-body width (legs).
      2. width_50_norm       — same at 50% (torso).
      3. width_75_norm       — same at 75% (shoulders / upper body).
      4. aspect_norm         — body_height / max(width). Long/tall vs short/wide.
                                Clipped to [0, 5] then divided by 5.
      5. compactness         — foreground area / bounding-box area. How "filled"
                                the silhouette is vs being lanky.

    These are anatomical statistics — they vary across people based on body
    proportions, much less based on the specific pose at this frame. (Compare
    to raw CNN features which encode "the projected silhouette at this pose"
    much more strongly than the underlying anatomy.) Used as an *auxiliary*
    regression target alongside triplet+CE, they force the depth embedding
    to retain pose-invariant body geometry.

    Returns zeros if the foreground is too small to extract meaningful features
    — caller can mask out the loss contribution from such frames.
    """
    if depth_img_2d.ndim == 3:
        depth_img_2d = depth_img_2d[..., 0]
    H, W = depth_img_2d.shape
    fg = depth_img_2d > fg_threshold
    n_fg = int(fg.sum())
    if n_fg < MIN_FOREGROUND_PIXELS:
        return np.zeros(ANTHRO_DIM, dtype=np.float32)

    ys, xs = np.where(fg)
    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    body_height_px = y_max - y_min + 1
    body_height_norm = body_height_px / float(H)

    # Widths at three vertical fractions of the body.
    widths_norm = []
    for frac in (0.25, 0.5, 0.75):
        y_target = int(np.clip(y_min + frac * body_height_px, y_min, y_max))
        row_fg = fg[y_target]
        if row_fg.any():
            row_xs = np.where(row_fg)[0]
            w_px = int(row_xs.max() - row_xs.min() + 1)
            widths_norm.append(w_px / float(W))
        else:
            widths_norm.append(0.0)

    # Aspect ratio (tall-thin vs short-wide). Clip to [0, 5] then scale to [0, 1].
    max_w_norm = max(widths_norm) if max(widths_norm) > 0 else 1e-6
    aspect = body_height_norm / max(max_w_norm, 1e-3)
    aspect_norm = float(np.clip(aspect / 5.0, 0.0, 1.0))

    # Compactness: how much of the bbox is foreground.
    bbox_area = body_height_px * (x_max - x_min + 1)
    compactness = float(n_fg) / max(float(bbox_area), 1.0)

    return np.array([body_height_norm, *widths_norm, aspect_norm, compactness],
                    dtype=np.float32)


def _compute_anthropometrics_3d(depth_2d_mm, fg_mask):
    """Compute 3D-derived anthropometric features from a raw mm depth image
    + foreground mask. Uses Kinect intrinsics to unproject pixels into 3D
    (X, Y, Z) coordinates in mm, then measures body geometry in real units.

    Key difference vs `_compute_anthropometrics`: that one operates on the
    silhouette (foreground mask) only — discards the actual depth values.
    This one uses the Z values, recovering body THICKNESS (front-to-back),
    real-world body height in mm (not pixels), and body volume. These are
    much more identity-discriminative AND much more pose-invariant than
    silhouette features (a person's chest depth doesn't change much when
    they walk; their silhouette outline changes a lot).

    Returns a length-ANTHRO_DIM_3D vector with all values in [0, 1] after
    normalization by plausible human bounds.

      0. body_height_norm     — y-extent in mm / 2000 (typical max ~2m)
      1. width_25_norm        — x-extent at 25% height / 600 (typical max ~60cm)
      2. width_50_norm        — same at 50% (torso)
      3. width_75_norm        — same at 75% (shoulders)
      4. depth_25_norm        — z-extent (body thickness) at 25% height / 400
      5. depth_50_norm        — body thickness at 50% (torso depth ~25-35cm)
      6. depth_75_norm        — body thickness at 75% (shoulder depth)
      7. volume_norm          — estimated body volume / 2e8 mm³
      8. aspect_3d_norm       — height / max(width, depth), clipped & scaled

    Features 4-7 (z-thickness + volume) are the new identity signals that
    2D silhouette anthropometrics can't see. Features 0-3 and 8 are now in
    real mm units instead of pixel ratios — comparable across recordings.

    Returns zeros if the foreground is too small to yield reliable measurements.
    """
    H, W = depth_2d_mm.shape
    fx, fy, cx, cy = _intrinsics_for_image_shape(H, W)
    pts3d = _unproject_depth_to_3d(depth_2d_mm, fg_mask, fx, fy, cx, cy)
    if len(pts3d) < MIN_FOREGROUND_PIXELS:
        return np.zeros(ANTHRO_DIM_3D, dtype=np.float32)

    Xs, Ys, Zs = pts3d[:, 0], pts3d[:, 1], pts3d[:, 2]
    y_min, y_max = float(Ys.min()), float(Ys.max())
    body_height_mm = max(y_max - y_min, 1.0)

    # Measure width (X-extent) and depth (Z-extent) at three vertical fractions,
    # taking the points within a thin horizontal slice (5% of body height wide).
    slice_half_thickness = 0.05 * body_height_mm
    widths_mm, depths_mm = [], []
    for frac in (0.25, 0.5, 0.75):
        y_target = y_min + frac * body_height_mm
        in_slice = np.abs(Ys - y_target) < slice_half_thickness
        if in_slice.sum() >= 5:
            slc = pts3d[in_slice]
            widths_mm.append(float(slc[:, 0].max() - slc[:, 0].min()))
            depths_mm.append(float(slc[:, 2].max() - slc[:, 2].min()))
        else:
            widths_mm.append(0.0)
            depths_mm.append(0.0)

    # Rough volume estimate: avg cross-sectional area × body height.
    avg_w = float(np.mean(widths_mm))
    avg_d = float(np.mean(depths_mm))
    volume_mm3 = avg_w * avg_d * body_height_mm  # ellipsoidal slice approx

    max_lateral = max(max(widths_mm) if widths_mm else 0.0,
                      max(depths_mm) if depths_mm else 0.0, 1e-3)
    aspect_3d = body_height_mm / max_lateral
    aspect_3d_norm = float(np.clip(aspect_3d / 5.0, 0.0, 1.0))

    features = np.array([
        body_height_mm / 2000.0,
        widths_mm[0] / 600.0,
        widths_mm[1] / 600.0,
        widths_mm[2] / 600.0,
        depths_mm[0] / 400.0,
        depths_mm[1] / 400.0,
        depths_mm[2] / 400.0,
        volume_mm3 / 2e8,
        aspect_3d_norm,
    ], dtype=np.float32)
    return np.clip(features, 0.0, 1.0)


def _canonicalize_pose_2d(img_normalized_2d, fg_threshold=-0.99):
    """Rotate a normalized depth image so the foreground silhouette's principal
    axis (largest-variance direction) is vertical.

    Args:
        img_normalized_2d: (H, W) float32 array in [-1, 1] with background=-1.
        fg_threshold: pixels > fg_threshold are foreground.

    Returns:
        rotated image of same shape & dtype. If foreground is too small or PCA
        fails, returns the input unchanged.

    Mechanism: extract foreground pixel coords, compute 2D PCA on them, find the
    rotation that aligns the principal eigenvector with the +y (downward) axis,
    apply the rotation via cv2.warpAffine. Rotation is normalized to [-90°, 90°]
    so we don't ever rotate by 180° (the principal axis is direction-ambiguous).

    Pose canonicalization is a depth-specific preprocessing step justified by the
    sensor characteristics — same rule as the foreground slab. It removes
    body-tilt as a pose variable, leaving the CNN to model only the harder
    pose variations (limb position, gait, articulation). For BIWI Still↔Walking,
    the residual tilt across a recording is small but non-zero; canonicalizing
    standardizes it so augmentation rotation acts on a consistent baseline.
    """
    if img_normalized_2d.ndim == 3:
        img_normalized_2d = img_normalized_2d[..., 0]
    fg = img_normalized_2d > fg_threshold
    n_fg = int(fg.sum())
    if n_fg < MIN_FOREGROUND_PIXELS:
        return img_normalized_2d

    ys, xs = np.where(fg)
    coords = np.stack(
        [xs.astype(np.float64) - xs.mean(), ys.astype(np.float64) - ys.mean()],
        axis=1,
    )
    try:
        cov = np.cov(coords.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
    except (np.linalg.LinAlgError, ValueError):
        return img_normalized_2d
    # Principal axis = eigenvector with largest eigenvalue.
    principal = eigvecs[:, int(np.argmax(eigvals))]
    # Eigenvectors are direction-ambiguous; flip so it points downward (+y in
    # image coords) so arctan2 gives a value in [0, π] consistently.
    if principal[1] < 0:
        principal = -principal
    angle_deg = float(np.rad2deg(np.arctan2(principal[1], principal[0])))
    # Target: principal axis at 90° from +x (along image-coord +y, "down").
    # cv2.getRotationMatrix2D treats positive angle as CCW in math convention,
    # which is *visual* CW in image coords (y-down). So if the bar is tilted to
    # angle θ, we need cv2 rotation of (θ - 90°) — positive when θ > 90° (bar
    # tilted left, rotate CW visually) and negative when θ < 90° (bar tilted
    # right, rotate CCW visually). Since we forced principal[1] ≥ 0, θ ∈ [0°, 180°]
    # and rotation_deg ∈ [-90°, 90°] — no 180° flip ambiguity to worry about.
    rotation_deg = angle_deg - 90.0

    H, W = img_normalized_2d.shape
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), rotation_deg, 1.0)
    rotated = cv2.warpAffine(
        img_normalized_2d, M, (W, H),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=-1.0,
    )
    return rotated


def _canonicalize_pose_3d(depth_2d_mm, fg_mask):
    """Rotate the foreground body in 3D space so its principal axis aligns
    with the vertical (image-coord +Y, "downward") direction, then re-project
    to a 2D depth image.

    Why 3D instead of 2D PCA (which we tried and which hurt by 6pp):
    - 2D PCA is computed on (x_pixel, y_pixel). It can only see in-image-plane
      tilt and is heavily polluted by limb pose (leg lift, arm swing).
    - 3D PCA is computed on the unprojected (X, Y, Z) point cloud. The principal
      axis is dominated by the torso (which has the most points). Forward-lean
      (a pose component invisible to 2D PCA) is captured. Leg/arm pose
      contributes much less because limbs are smaller proportion of total mass.

    Algorithm:
      1. Unproject foreground pixels to 3D (X, Y, Z) using Kinect intrinsics.
      2. Compute centroid; subtract to center the point cloud.
      3. PCA: covariance of centered points → principal eigenvector.
      4. Compute rotation matrix that maps principal eigenvector to +Y axis
         (Rodrigues' formula, no roll component).
      5. Apply rotation to centered points; add centroid back.
      6. Re-project rotated 3D points to a 2D depth image via the same intrinsics.
         Z-buffer (closer wins) for pixels receiving multiple points.

    Returns (rotated_depth_2d_mm, rotated_fg_mask). If too few foreground
    points or rotation is near-identity, returns the inputs unchanged.
    """
    H, W = depth_2d_mm.shape
    fx, fy, cx, cy = _intrinsics_for_image_shape(H, W)
    pts3d = _unproject_depth_to_3d(depth_2d_mm, fg_mask, fx, fy, cx, cy)
    if len(pts3d) < MIN_FOREGROUND_PIXELS:
        return depth_2d_mm, fg_mask

    centroid = pts3d.mean(axis=0)
    pts_centered = pts3d - centroid

    try:
        cov = np.cov(pts_centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
    except (np.linalg.LinAlgError, ValueError):
        return depth_2d_mm, fg_mask

    # Principal axis = eigenvector with largest eigenvalue.
    principal = eigvecs[:, int(np.argmax(eigvals))].astype(np.float32)
    # Direction-ambiguous; force +Y component so we rotate toward "down" not "up".
    if principal[1] < 0:
        principal = -principal

    target = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    # Rotation that maps `principal` onto `target` via Rodrigues' formula.
    cross = np.cross(principal, target)
    sin_a = float(np.linalg.norm(cross))
    cos_a = float(np.dot(principal, target))
    if sin_a < 1e-4:
        # Already aligned (cos_a ≈ +1) or anti-aligned (cos_a ≈ -1). In the
        # anti-aligned case a 180° rotation would mirror the body — which
        # is just the principal-axis direction ambiguity, not a real rotation.
        # Skip in both cases.
        return depth_2d_mm, fg_mask

    axis = cross / sin_a
    K = np.array([
        [    0.0, -axis[2],  axis[1]],
        [ axis[2],     0.0, -axis[0]],
        [-axis[1],  axis[0],     0.0],
    ], dtype=np.float32)
    R = np.eye(3, dtype=np.float32) + sin_a * K + (1.0 - cos_a) * (K @ K)

    pts_rotated = pts_centered @ R.T + centroid

    # Re-project to 2D depth via the same intrinsics. Z-buffer: closer point wins.
    new_depth = np.zeros_like(depth_2d_mm)
    new_fg = np.zeros((H, W), dtype=bool)

    Xs, Ys, Zs = pts_rotated[:, 0], pts_rotated[:, 1], pts_rotated[:, 2]
    valid = Zs > 1.0  # avoid division by ~0 and behind-camera points
    Xs, Ys, Zs = Xs[valid], Ys[valid], Zs[valid]
    us = np.round(fx * Xs / Zs + cx).astype(np.int32)
    vs = np.round(fy * Ys / Zs + cy).astype(np.int32)
    in_bounds = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
    us, vs, zs = us[in_bounds], vs[in_bounds], Zs[in_bounds]

    # Z-buffer: sort points by depth descending (far first), then write. The
    # final value at each pixel is the closest point (smallest z) since later
    # writes overwrite earlier ones.
    order = np.argsort(-zs)
    us, vs, zs = us[order], vs[order], zs[order]
    new_depth[vs, us] = zs
    new_fg[vs, us] = True

    # If the rotation produced very few points after re-projection (e.g.,
    # extreme rotation pushed body out of frame), fall back to the original.
    if int(new_fg.sum()) < MIN_FOREGROUND_PIXELS:
        return depth_2d_mm, fg_mask

    return new_depth, new_fg


def load_rgb_uint8(path):
    img = cv2.imread(path)
    if img is None:
        print(f"Warning: Failed to load RGB image: {path}")
        return np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_W, IMG_H))
    return img

def normalize_rgb(img_uint8):
    return (img_uint8.astype(np.float32) / 127.5) - 1.0

def quantize_depth_uint8(img_f32):
    return ((img_f32 + 1.0) * 127.5).clip(0, 255).astype(np.uint8)

def dequantize_depth_uint8(img_uint8):
    return (img_uint8.astype(np.float32) / 127.5) - 1.0

def load_and_preprocess_single(path, is_rgb, augment=False, aug_params=None):
    if is_rgb:
        img = load_rgb_uint8(path)
        if augment and aug_params is not None:
            img = apply_rgb_augmentation(img, aug_params)
        return normalize_rgb(img)
    else:
        img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
        if img is None:
            print(f"Warning: Failed to load depth image: {path}")
            return np.full((IMG_H, IMG_W, 1), -1.0, dtype=np.float32)
        img = preprocess_depth_smart(img, augment=augment, aug_params=aug_params)
        return img

def verify_data_quality(file_paths, labels):
    print(f"\nVerifying data quality...")
    issues = []
    
    if len(file_paths) != len(labels):
        issues.append(f"Mismatch: {len(file_paths)} files vs {len(labels)} labels")
    
    unique_labels = set(labels)
    label_counts = defaultdict(int)
    for label in labels:
        label_counts[label] += 1
    
    min_samples = min(label_counts.values())
    max_samples = max(label_counts.values())
    avg_samples = np.mean(list(label_counts.values()))
    
    print(f"  Identities: {len(unique_labels)}")
    print(f"  Samples per identity: min={min_samples}, max={max_samples}, avg={avg_samples:.1f}")
    
    if max_samples > 3 * min_samples:
        issues.append(f"High imbalance: max/min ratio = {max_samples/min_samples:.1f}")
    
    duplicates = len(file_paths) - len(set(file_paths))
    if duplicates > 0:
        issues.append(f"Found {duplicates} duplicate file paths")
    
    if issues:
        print(f"  ⚠ Issues found:")
        for issue in issues:
            print(f"    - {issue}")
    else:
        print(f"  ✓ No issues detected")
    
    return len(issues) == 0

def _scan_biwi_format(root):
    """Recursively walk root, group RGB/depth files by subject ID (NNN prefix in filename).

    Works for BIWI RGBD-ID and IAS-Lab RGBD-ID — both use the same Munaro et al. format.
    Training filenames: '022_<frame>-a_<ts>_rgb.jpg', etc.
    Testing  filenames: '022b_<frame>-a_<ts>_rgb.jpg' (trailing 'b' = second-session
                        recording with different clothing; same identity as Training '022').

    We extract leading digits only so '022_…' and '022b_…' both map to subject '022'.
    Handles flat (<root>/<subj>/<files>) or nested (<root>/<cond>/<subj>/<files>) layouts.
    """
    by_subj = defaultdict(lambda: {'rgb': [], 'depth': []})
    if not os.path.exists(root):
        return by_subj
    for dirpath, _dirnames, filenames in os.walk(root):
        for f in filenames:
            # Take leading digits as subject id; tolerate suffix letters like 'b'.
            head = f.split('_', 1)[0]
            digits = ''
            for c in head:
                if c.isdigit():
                    digits += c
                else:
                    break
            if not digits:
                continue
            subj_id = digits
            fpath = os.path.join(dirpath, f)
            fl = f.lower()
            if fl.endswith('_rgb.jpg') or fl.endswith('_rgb.png'):
                by_subj[subj_id]['rgb'].append(fpath)
            elif fl.endswith('_depth.pgm') or fl.endswith('_depth.png'):
                by_subj[subj_id]['depth'].append(fpath)
    return by_subj

def load_biwi_disjoint(base_dir, test_split=0.2):
    """Local diagnostic loader: identity-disjoint random split of Training/ only.
    Used when the official Testing/ portion isn't available locally."""
    print(f"Scanning {base_dir} (local-diagnostic disjoint split)...")
    by_subj = _scan_biwi_format(os.path.join(base_dir, 'Training'))

    all_ids = sorted(by_subj.keys())
    random.shuffle(all_ids)
    n_val = int(len(all_ids) * test_split)
    val_ids = set(all_ids[:n_val])
    train_ids = set(all_ids[n_val:])

    if len(train_ids) < 2 or len(val_ids) < 2:
        raise ValueError(f"Insufficient identities for train ({len(train_ids)}) or val ({len(val_ids)}). Need at least 2 each.")

    print(f"Identities -> Train: {len(train_ids)}, Val: {len(val_ids)}")

    def collect(id_set, modality):
        paths, labels = [], []
        for subj in id_set:
            ps = sorted(by_subj[subj][modality])
            paths.extend(ps)
            labels.extend([subj] * len(ps))
        return paths, labels

    rgb_train = collect(train_ids, 'rgb')
    rgb_val = collect(val_ids, 'rgb')
    depth_train = collect(train_ids, 'depth')
    depth_val = collect(val_ids, 'depth')

    print(f"\n--- RGB Data Quality ---")
    verify_data_quality(rgb_train[0], rgb_train[1])
    verify_data_quality(rgb_val[0], rgb_val[1])

    print(f"\n--- Depth Data Quality ---")
    verify_data_quality(depth_train[0], depth_train[1])
    verify_data_quality(depth_val[0], depth_val[1])

    return (rgb_train, rgb_val, depth_train, depth_val)

def load_biwi_full_protocol(base_dir, train_subdir='Training', test_subdir='Testing'):
    """Official BIWI/IAS-Lab cross-clothing protocol.

    Train on <base_dir>/<train_subdir>/ (one outfit per ID), evaluate on
    <base_dir>/<test_subdir>/ (different day, different clothing for most subjects).

    For BIWI:     train_subdir='Training', test_subdir='Testing'
    For IAS-Lab:  train_subdir='Training', test_subdir='TestingB' (different clothing)
                                       or  test_subdir='TestingA' (similar clothing, sanity check)
    """
    train_root = os.path.join(base_dir, train_subdir)
    test_root = os.path.join(base_dir, test_subdir)
    print(f"Scanning train: {train_root}")
    train_by_subj = _scan_biwi_format(train_root)
    print(f"Scanning test:  {test_root}")
    test_by_subj = _scan_biwi_format(test_root)

    train_ids = sorted(train_by_subj.keys())
    test_ids = sorted(test_by_subj.keys())
    overlap = sorted(set(train_ids) & set(test_ids))

    print(f"Identities -> Train: {len(train_ids)}, Test: {len(test_ids)}, Overlap (cross-clothing IDs): {len(overlap)}")
    if len(test_ids) < 2:
        raise ValueError(f"Test set must contain >=2 identities. Found {len(test_ids)} in {test_root}.")
    if len(overlap) == 0:
        print(f"  Note: zero overlap between train and test IDs — running open-set protocol.")

    def collect(ids, by_subj, modality):
        paths, labels = [], []
        for subj in ids:
            ps = sorted(by_subj.get(subj, {}).get(modality, []))
            paths.extend(ps)
            labels.extend([subj] * len(ps))
        return paths, labels

    rgb_train = collect(train_ids, train_by_subj, 'rgb')
    rgb_val = collect(test_ids, test_by_subj, 'rgb')
    depth_train = collect(train_ids, train_by_subj, 'depth')
    depth_val = collect(test_ids, test_by_subj, 'depth')

    print(f"\n--- RGB Data Quality ---")
    verify_data_quality(rgb_train[0], rgb_train[1])
    verify_data_quality(rgb_val[0], rgb_val[1])

    print(f"\n--- Depth Data Quality ---")
    verify_data_quality(depth_train[0], depth_train[1])
    verify_data_quality(depth_val[0], depth_val[1])

    return (rgb_train, rgb_val, depth_train, depth_val)

def load_biwi_open_set(base_dir, train_subdir='Training', test_subdir='Testing'):
    """Open-set siamese protocol — train and test identity sets are DISJOINT.

    For BIWI: train on subjects in <base_dir>/Training that are NOT in <base_dir>/Testing,
    test on the subjects in <base_dir>/Testing. The model has never seen the test identities
    in any clothing/recording, so the embedding must generalize to unseen people. This is
    the canonical setup for "siamese as a step forward from classification" — classification
    can't handle this protocol (no class label for unseen subjects at test time).
    """
    train_root = os.path.join(base_dir, train_subdir)
    test_root = os.path.join(base_dir, test_subdir)
    print(f"Open-set protocol")
    print(f"  Scanning train: {train_root}")
    train_by_subj = _scan_biwi_format(train_root)
    print(f"  Scanning test:  {test_root}")
    test_by_subj = _scan_biwi_format(test_root)

    all_train_ids = set(train_by_subj.keys())
    all_test_ids = set(test_by_subj.keys())
    # Disjoint: train IDs are those in Training/ but NOT in Testing/.
    open_train_ids = sorted(all_train_ids - all_test_ids)
    open_test_ids = sorted(all_test_ids)

    print(f"  Train identities (Training/ \\ Testing/): {len(open_train_ids)}")
    print(f"  Test identities (Testing/):              {len(open_test_ids)}")
    print(f"  Overlap (should be 0 for open-set):     {len(set(open_train_ids) & set(open_test_ids))}")
    if len(open_train_ids) < 2:
        raise ValueError(f"Open-set needs >=2 disjoint train identities. Found {len(open_train_ids)}.")
    if len(open_test_ids) < 2:
        raise ValueError(f"Open-set needs >=2 test identities. Found {len(open_test_ids)}.")

    def collect(ids, by_subj, modality):
        paths, labels = [], []
        for subj in ids:
            ps = sorted(by_subj.get(subj, {}).get(modality, []))
            paths.extend(ps)
            labels.extend([subj] * len(ps))
        return paths, labels

    rgb_train = collect(open_train_ids, train_by_subj, 'rgb')
    rgb_val = collect(open_test_ids, test_by_subj, 'rgb')
    depth_train = collect(open_train_ids, train_by_subj, 'depth')
    depth_val = collect(open_test_ids, test_by_subj, 'depth')

    # ---- Cross-clothing gallery (outfit A) for the held-out test subjects ----
    # The test subjects appear in BOTH Training/ (outfit A) and Testing/ (outfit B).
    # Our embedding is trained ONLY on the disjoint `open_train_ids` (22 subjects),
    # so the test subjects' Training-folder frames are unseen-identity data. They
    # form the gallery for an OPEN-SET CROSS-CLOTHING eval:
    #   gallery = test subjects in outfit A (Training/),
    #   probe   = same test subjects in outfit B (Testing/, == rgb_val/depth_val).
    # This is the regime where depth's clothing-invariant body-shape cue should
    # match or beat RGB (whose clothing cue flips to a different outfit). Only the
    # test subjects that actually appear in Training/ can contribute a gallery.
    xcloth_ids = sorted(set(open_test_ids) & all_train_ids)
    rgb_xgallery = collect(xcloth_ids, train_by_subj, 'rgb')
    depth_xgallery = collect(xcloth_ids, train_by_subj, 'depth')
    print(f"  Cross-clothing gallery (test subjects' Training/ frames, outfit A): "
          f"{len(xcloth_ids)} subjects, {len(rgb_xgallery[0])} rgb / {len(depth_xgallery[0])} depth frames")

    print(f"\n--- RGB Data Quality ---")
    verify_data_quality(rgb_train[0], rgb_train[1])
    verify_data_quality(rgb_val[0], rgb_val[1])
    print(f"\n--- Depth Data Quality ---")
    verify_data_quality(depth_train[0], depth_train[1])
    verify_data_quality(depth_val[0], depth_val[1])

    return (rgb_train, rgb_val, depth_train, depth_val,
            rgb_xgallery, depth_xgallery)

def load_dataset_auto(base_dir):
    """Auto-pick the right loader: if <base_dir>/Testing exists, use the full protocol
    (cross-clothing eval); otherwise fall back to identity-disjoint random split of Training/.
    """
    if os.path.exists(os.path.join(base_dir, 'Testing')):
        print("Detected Testing/ — using BIWI full protocol (cross-clothing eval)")
        return load_biwi_full_protocol(base_dir, train_subdir='Training', test_subdir='Testing')
    print("No Testing/ found — falling back to local-diagnostic disjoint split of Training/")
    return load_biwi_disjoint(base_dir)

def stage_zip_to_local(zip_path, dest_dir=None):
    """Extract a packed dataset ZIP to a local directory (typically $SLURM_TMPDIR).

    Cluster-friendly: $SLURM_TMPDIR is node-local SSD that doesn't count against
    file-count quotas. The extraction lasts only for the duration of the job and
    is auto-cleaned at exit.

    Returns the path to the extracted directory (suitable for --base-dir).
    """
    if dest_dir is None:
        dest_dir = os.environ.get("SLURM_TMPDIR") or tempfile.mkdtemp(prefix="biwi_")
    os.makedirs(dest_dir, exist_ok=True)
    print(f"Staging {zip_path} -> {dest_dir} ...")
    t0 = time.time()
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        zf.extractall(dest_dir)
    print(f"  Extracted {len(members)} entries in {time.time() - t0:.0f}s.")
    return dest_dir

class SiamesePairGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_paths, labels, batch_size, is_rgb, shuffle=True, cache_images=True, augment=False, hard_negatives=False, **kwargs):
        super().__init__(**kwargs)
        self.file_paths = np.array(file_paths)
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.is_rgb = is_rgb
        self.shuffle = shuffle
        self.cache_images = cache_images
        self.augment = augment
        self.hard_negatives = hard_negatives
        self.image_cache = {}
        self.embeddings = None
        self.backbone = None
        
        self.indices_by_label = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.indices_by_label[lbl].append(idx)
        self.unique_labels = [l for l in self.indices_by_label.keys() if len(self.indices_by_label[l]) > 1]
        
        if self.cache_images:
            print(f"Caching {len(self.file_paths)} images ({'RGB' if is_rgb else 'Depth'}, uint8)...")
            for i, path in enumerate(self.file_paths):
                if i % 500 == 0: print(f"  {i}/{len(self.file_paths)}...", end='\r')
                if self.is_rgb:
                    self.image_cache[path] = load_rgb_uint8(path)
                else:
                    self.image_cache[path] = quantize_depth_uint8(load_and_preprocess_single(path, self.is_rgb))
            print(f"  Done.")
        self.on_epoch_end()
    
    def embed_all(self, backbone, chunk=1024):
        """Embed every image in this generator's path list. Returns (N, emb_dim) array.

        Works whether or not the cache is populated. Used by hard-negative mining
        (sets self.embeddings) and by the gallery-probe evaluator.
        """
        N = len(self.file_paths)
        embs = []
        for start in range(0, N, chunk):
            chunk_paths = self.file_paths[start:start + chunk]
            if self.cache_images:
                if self.is_rgb:
                    imgs = np.array([normalize_rgb(self.image_cache[p]) for p in chunk_paths])
                else:
                    imgs = np.array([dequantize_depth_uint8(self.image_cache[p]) for p in chunk_paths])
            else:
                imgs = np.array([load_and_preprocess_single(p, self.is_rgb) for p in chunk_paths])
            embs.append(backbone.predict(imgs, batch_size=EVAL_BATCH_SIZE, verbose=0))
            del imgs
        return np.concatenate(embs, axis=0)

    def update_embeddings(self, backbone):
        if not self.hard_negatives:
            return
        self.backbone = backbone
        print(f"  Computing embeddings for hard negative mining...")
        self.embeddings = self.embed_all(backbone)
        print(f"  Done.")
        
    def __len__(self): return len(self.pairs) // self.batch_size

    def on_epoch_end(self):
        self.pairs = []
        self.pair_labels = []
        all_indices = np.arange(len(self.file_paths))
        if self.shuffle: np.random.shuffle(all_indices)
            
        for anchor_idx in all_indices:
            anchor_label = self.labels[anchor_idx]
            candidates = self.indices_by_label[anchor_label]
            if len(candidates) < 2: continue
            pos_idx = random.choice(candidates)
            while pos_idx == anchor_idx: pos_idx = random.choice(candidates)
            self.pairs.append([anchor_idx, pos_idx])
            self.pair_labels.append(1.0)
            
            if self.hard_negatives and self.embeddings is not None:
                anchor_emb = self.embeddings[anchor_idx]
                neg_candidates = []
                for neg_label in self.unique_labels:
                    if neg_label != anchor_label:
                        neg_candidates.extend(self.indices_by_label[neg_label])
                
                if len(neg_candidates) > 0:
                    neg_embs = self.embeddings[neg_candidates]
                    dists = np.sum((anchor_emb - neg_embs) ** 2, axis=1)
                    hardest_idx = neg_candidates[np.argmin(dists)]
                    neg_idx = hardest_idx
                else:
                    neg_label = random.choice(self.unique_labels)
                    while neg_label == anchor_label: neg_label = random.choice(self.unique_labels)
                    neg_idx = random.choice(self.indices_by_label[neg_label])
            else:
                neg_label = random.choice(self.unique_labels)
                while neg_label == anchor_label: neg_label = random.choice(self.unique_labels)
                neg_idx = random.choice(self.indices_by_label[neg_label])
            
            self.pairs.append([anchor_idx, neg_idx])
            self.pair_labels.append(0.0)
        
        zipped = list(zip(self.pairs, self.pair_labels))
        np.random.shuffle(zipped)
        self.pairs, self.pair_labels = zip(*zipped)
        self.pairs = np.array(self.pairs)
        self.pair_labels = np.array(self.pair_labels, dtype=np.float32)

    def __getitem__(self, index):
        indices = self.pairs[index * self.batch_size : (index + 1) * self.batch_size]
        labels = self.pair_labels[index * self.batch_size : (index + 1) * self.batch_size]
        A, B = [], []
        for (ia, ib) in indices:
            pa, pb = self.file_paths[ia], self.file_paths[ib]
            if self.cache_images:
                ca, cb = self.image_cache[pa], self.image_cache[pb]
                if self.augment:
                    ap_a = get_augmentation_params()
                    ap_b = get_augmentation_params()
                    if self.is_rgb:
                        A.append(normalize_rgb(apply_rgb_augmentation(ca, ap_a)))
                        B.append(normalize_rgb(apply_rgb_augmentation(cb, ap_b)))
                    else:
                        da = dequantize_depth_uint8(ca)
                        db = dequantize_depth_uint8(cb)
                        A.append(np.expand_dims(apply_cached_depth_augmentation(da[:, :, 0], ap_a), -1))
                        B.append(np.expand_dims(apply_cached_depth_augmentation(db[:, :, 0], ap_b), -1))
                else:
                    if self.is_rgb:
                        A.append(normalize_rgb(ca))
                        B.append(normalize_rgb(cb))
                    else:
                        A.append(dequantize_depth_uint8(ca))
                        B.append(dequantize_depth_uint8(cb))
            else:
                aug_params = get_augmentation_params() if self.augment else None
                A.append(load_and_preprocess_single(pa, self.is_rgb, augment=self.augment, aug_params=aug_params))
                B.append(load_and_preprocess_single(pb, self.is_rgb, augment=self.augment, aug_params=aug_params))
        return (np.array(A), np.array(B)), labels

def evaluate_gallery_probe(backbone, gallery_gen, probe_imgs_uint8, probe_labels, is_rgb,
                           chunk=1024, dist_chunk=2048):
    """Cross-set retrieval eval — the proper protocol for cross-clothing Re-ID.

    For each probe image, find its nearest neighbour among all gallery embeddings.
    R1 = predicted ID (= nearest gallery item's label) matches the probe's true ID.

    gallery_gen:        SiamesePairGenerator wrapping the training set (clothing A).
                        Its file_paths and labels become the gallery.
    probe_imgs_uint8:   pre-loaded uint8 tensor of the probe (test) images, shape
                        (N_probe, H, W, C). For RGB: 3-channel uint8 directly. For
                        depth: uint8-quantized via quantize_depth_uint8 (single channel).
    probe_labels:       array-like of length N_probe with each probe image's ID.

    Returns dict with rank1, rank5, mAP, sep, n_gallery, n_probe (all percentages).
    """
    print(f"  [gallery-probe] embedding {len(gallery_gen.file_paths)} gallery + "
          f"{len(probe_imgs_uint8)} probe images...")

    # Gallery (training set, clothing A).
    gallery_embs = gallery_gen.embed_all(backbone, chunk=chunk)
    gallery_labels = np.asarray(gallery_gen.labels)

    # Probe (test set, clothing B). Already pre-loaded as uint8.
    probe_embs = []
    deq = normalize_rgb if is_rgb else dequantize_depth_uint8
    for start in range(0, len(probe_imgs_uint8), EVAL_BATCH_SIZE):
        batch = deq(probe_imgs_uint8[start:start + EVAL_BATCH_SIZE])
        probe_embs.append(backbone.predict(batch, verbose=0))
    probe_embs = np.vstack(probe_embs)
    probe_labels = np.asarray(probe_labels)

    # Per-frame matching: each probe -> nearest training FRAME.
    # Chunked because the full (15k probe × 24k gallery) matrix is ~1.4GB.
    N_probe = probe_embs.shape[0]
    N_gallery = gallery_embs.shape[0]
    rank1 = 0
    rank5 = 0
    aps = []
    for start in range(0, N_probe, dist_chunk):
        end = min(start + dist_chunk, N_probe)
        D = pairwise_distances(probe_embs[start:end], gallery_embs, metric='cosine')
        sorted_idx = np.argsort(D, axis=1)
        for i_local in range(end - start):
            i_global = start + i_local
            matches = (gallery_labels[sorted_idx[i_local]] == probe_labels[i_global])
            if matches[0]:
                rank1 += 1
            if np.any(matches[:5]):
                rank5 += 1
            num_valid = int(np.sum(matches))
            if num_valid > 0:
                old_recall, old_precision = 0.0, 1.0
                ap = 0.0
                intersect = 0
                for j in range(N_gallery):
                    if matches[j]:
                        intersect += 1
                        recall = intersect / num_valid
                        precision = intersect / (j + 1)
                        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
                        old_recall, old_precision = recall, precision
                aps.append(ap)
        del D, sorted_idx

    r1 = rank1 / N_probe * 100.0
    r5 = rank5 / N_probe * 100.0
    mAP = float(np.mean(aps) * 100.0) if aps else 0.0

    # Per-subject-prototype matching: collapse gallery to one mean embedding per
    # training subject, then match each probe against those K=50 prototypes.
    proto_embs, proto_labels = _per_subject_prototypes(gallery_embs, gallery_labels)
    D_proto = pairwise_distances(probe_embs, proto_embs, metric='cosine')
    proto_metrics = _rank_metrics(D_proto, probe_labels, proto_labels)

    # K-reciprocal re-rank on the prototype distance matrix (small, fast).
    proto_rerank = None
    try:
        D_proto_rerank = _k_reciprocal_rerank(probe_embs, proto_embs)
        proto_rerank = _rank_metrics(D_proto_rerank, probe_labels, proto_labels)
    except (MemoryError, ValueError) as e:
        print(f"  [gallery-probe] skipped proto rerank: {type(e).__name__}: {e}")

    # Separation: same-ID vs different-ID cosine distance ratio over gallery.
    intra, inter = [], []
    for lbl in np.unique(gallery_labels):
        idx = np.where(gallery_labels == lbl)[0]
        if len(idx) > 1:
            d = pairwise_distances(gallery_embs[idx], metric='cosine')
            intra.extend(d[np.triu_indices_from(d, k=1)])
    # inter sampled (full pairwise inter is O(N^2) on 24k embeddings = 10 GB)
    sample = np.random.choice(N_gallery, size=min(5000, N_gallery), replace=False)
    d_sample = pairwise_distances(gallery_embs[sample], metric='cosine')
    labels_sample = gallery_labels[sample]
    for i in range(len(sample)):
        for j in range(i + 1, len(sample)):
            if labels_sample[i] != labels_sample[j]:
                inter.append(d_sample[i, j])
    intra_mean = float(np.mean(intra)) if intra else 0.0
    inter_mean = float(np.mean(inter)) if inter else 0.0
    sep = inter_mean / intra_mean if intra_mean > 0 else 0.0

    out = {
        # Per-frame matching (probe → nearest of all 24k gallery frames).
        "rank1": float(r1),
        "rank5": float(r5),
        "mAP": mAP,
        "sep": float(sep),
        # Per-subject-prototype matching (probe → nearest of 50 subject means).
        "proto_rank1": proto_metrics["rank1"],
        "proto_rank5": proto_metrics["rank5"],
        "proto_mAP":   proto_metrics["mAP"],
        "n_gallery":      int(N_gallery),
        "n_gallery_subj": int(len(proto_labels)),
        "n_probe":        int(N_probe),
    }
    if proto_rerank is not None:
        out.update({
            "rerank_proto_rank1": proto_rerank["rank1"],
            "rerank_proto_rank5": proto_rerank["rank5"],
            "rerank_proto_mAP":   proto_rerank["mAP"],
        })
    return out

def _per_subject_prototypes(embs, labels):
    """Collapse a (N, D) embedding bank into (K, D) prototypes — one mean embedding per
    unique label. Standard Re-ID trick: averaging multiple gallery frames per subject
    cancels per-frame noise (depth especially benefits). Re-normalises to unit length
    so cosine distance against original L2-normalised embeddings stays meaningful.
    """
    unique = sorted(set(labels.tolist())) if hasattr(labels, "tolist") else sorted(set(labels))
    proto = np.stack([embs[labels == lbl].mean(axis=0) for lbl in unique], axis=0)
    norms = np.linalg.norm(proto, axis=1, keepdims=True)
    proto = proto / np.maximum(norms, 1e-12)
    return proto, np.asarray(unique)

def _multi_frame_average(embs, labels, window):
    """Average consecutive same-label embeddings in non-overlapping windows of size N.

    Standard video-Re-ID trick (Hermans et al. 2017, "MARS" eval): instead of treating
    each probe frame independently, group N consecutive frames per subject into a
    single "video probe" by averaging their embeddings and re-normalising. This:
    - cancels per-frame motion-blur / silhouette noise (helps depth disproportionately)
    - represents the probe as a "sequence template" — closer to how Re-ID systems are
      actually deployed (a track of N frames is matched as a unit, not frame-by-frame)
    Assumes embs/labels are in temporal order WITHIN each label (which our
    _scan_biwi_format-based loaders guarantee).
    """
    if window is None or window <= 1:
        return embs, labels
    out_embs, out_labels = [], []
    labels = np.asarray(labels)
    # Process each subject's frames in temporal order.
    seen = set()
    label_order = []
    for l in labels:
        if l not in seen:
            seen.add(l); label_order.append(l)
    for lbl in label_order:
        idx = np.where(labels == lbl)[0]
        # idx will be a contiguous block since labels are grouped by subject; if not
        # contiguous, np.where still returns ascending indices, which is fine.
        n = len(idx)
        for start in range(0, n, window):
            end = min(start + window, n)
            avg = embs[idx[start:end]].mean(axis=0)
            avg = avg / max(float(np.linalg.norm(avg)), 1e-12)
            out_embs.append(avg)
            out_labels.append(lbl)
    return np.stack(out_embs), np.array(out_labels)

def _rank_metrics(D, probe_labels, gallery_labels):
    """Compute R1, R5, mAP given a (probe, gallery) distance matrix and label arrays."""
    sorted_idx = np.argsort(D, axis=1)
    rank1 = rank5 = 0
    aps = []
    N_probe = D.shape[0]
    for i in range(N_probe):
        matches = (gallery_labels[sorted_idx[i]] == probe_labels[i])
        if matches[0]: rank1 += 1
        if np.any(matches[:5]): rank5 += 1
        num_valid = int(np.sum(matches))
        if num_valid > 0:
            old_recall, old_precision = 0.0, 1.0
            ap, intersect = 0.0, 0
            for j, match in enumerate(matches):
                if match:
                    intersect += 1
                    recall = intersect / num_valid
                    precision = intersect / (j + 1)
                    ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
                    old_recall, old_precision = recall, precision
            aps.append(ap)
    return {
        "rank1": float(rank1 / N_probe * 100.0),
        "rank5": float(rank5 / N_probe * 100.0),
        "mAP": float(np.mean(aps) * 100.0) if aps else 0.0,
    }

def _k_reciprocal_rerank(probe_features, gallery_features, k1=20, k2=6, lambda_value=0.3):
    """K-reciprocal NN re-ranking (Zhong et al. 2017, CVPR — "Re-ranking Person
    Re-identification with k-reciprocal Encoding").

    Refines the (probe × gallery) cosine-distance matrix using the structure of
    mutual k-nearest-neighbors in the joint (probe + gallery) candidate pool.
    The final distance is a weighted combination of:
      - the original cosine distance (weight λ)
      - the Jaccard distance between k-reciprocal feature vectors (weight 1-λ)

    The intuition: if two embeddings are mutual k-NN of each other AND share
    many common k-reciprocal neighbors, they're more likely to be the same
    identity than a one-sided NN match. Particularly helpful for depth Re-ID
    where per-frame embeddings are noisy and the original ranking has many
    false positives in the top-k.

    Features are assumed L2-normalized. k1, k2 are auto-capped if (P+G) is small.

    Args:
        probe_features: (P, D) L2-normalized probe embeddings
        gallery_features: (G, D) L2-normalized gallery embeddings
        k1: number of k-reciprocal neighbors. Default 20 (Zhong's recommended).
            Auto-capped to (P+G)//2 - 1 for small candidate pools.
        k2: local-query-expansion neighborhood. Default 6.
        lambda_value: weight for original distance (vs Jaccard). Default 0.3.

    Returns: (P, G) reranked distance matrix.
    """
    P = probe_features.shape[0]
    G = gallery_features.shape[0]
    N = P + G

    # Cap k1, k2 to keep them sensible for small galleries (e.g., 28-prototype eval).
    k1 = max(1, min(k1, max(1, N // 2 - 1)))
    k2 = max(1, min(k2, k1))

    feat = np.vstack([probe_features, gallery_features]).astype(np.float32)
    original_dist = pairwise_distances(feat, metric='cosine').astype(np.float32)
    # Per-column min-max normalize (Zhong's recipe — keeps Jaccard combination in scale).
    col_max = original_dist.max(axis=0, keepdims=True)
    col_max = np.maximum(col_max, 1e-12)
    original_dist = original_dist / col_max

    initial_rank = np.argsort(original_dist, axis=1).astype(np.int32)
    V = np.zeros_like(original_dist, dtype=np.float32)

    half_k = max(1, int(round(k1 / 2)))
    for i in range(N):
        forward_neigh = initial_rank[i, :k1 + 1]
        backward_neigh = initial_rank[forward_neigh, :k1 + 1]
        fi = np.where(backward_neigh == i)[0]
        k_recip = forward_neigh[fi]
        k_recip_exp = k_recip.copy()
        # Local query expansion: include neighbors of neighbors if 2/3 overlap.
        for cand in k_recip:
            c_fwd = initial_rank[cand, :half_k + 1]
            c_bwd = initial_rank[c_fwd, :half_k + 1]
            c_fi = np.where(c_bwd == cand)[0]
            c_recip = c_fwd[c_fi]
            if len(c_recip) > 0 and len(np.intersect1d(c_recip, k_recip)) > 2.0 / 3 * len(c_recip):
                k_recip_exp = np.append(k_recip_exp, c_recip)
        k_recip_exp = np.unique(k_recip_exp)
        weight = np.exp(-original_dist[i, k_recip_exp])
        V[i, k_recip_exp] = weight / np.sum(weight)

    # k2-NN smoothing of V (further query expansion).
    if k2 > 1:
        V_qe = np.zeros_like(V, dtype=np.float32)
        for i in range(N):
            V_qe[i] = V[initial_rank[i, :k2]].mean(axis=0)
        V = V_qe

    # Sparse Jaccard distance via inverted index.
    inv_index = [np.where(V[:, j] != 0)[0] for j in range(N)]
    jaccard_probe_x_all = np.zeros((P, N), dtype=np.float32)
    for i in range(P):
        temp_min = np.zeros(N, dtype=np.float32)
        idx_nz = np.where(V[i, :] != 0)[0]
        for j in idx_nz:
            inds = inv_index[j]
            temp_min[inds] += np.minimum(V[i, j], V[inds, j])
        # V is row-normalized (sum = 1), so |x|=|y|=1 and Jaccard = 1 - min/(2-min).
        jaccard_probe_x_all[i] = 1.0 - temp_min / (2.0 - temp_min)

    # Combine original cosine + Jaccard, keep only probe × gallery sub-block.
    final_dist = (
        lambda_value * original_dist[:P, P:] +
        (1.0 - lambda_value) * jaccard_probe_x_all[:, P:]
    )
    return final_dist

# ---------------------------------------------------------------------------
# PK Batch Sampler + Batch-Hard Triplet Loss
# ---------------------------------------------------------------------------
# Modern siamese Re-ID recipe (Hermans et al. 2017, "In Defense of the Triplet
# Loss for Person Re-Identification"). Each batch contains P identities * K
# images per identity. The loss mines the hardest positive and hardest negative
# for every anchor WITHIN the batch. This is the standard replacement for the
# weaker random-pair contrastive loss we used in v1.
# ---------------------------------------------------------------------------

class PKBatchSampler(tf.keras.utils.Sequence):
    """Sample P identities x K images per batch for batch-hard triplet training.

    Yields (batch_imgs, batch_int_labels) where batch size = P * K.
    Labels are mapped from string subject IDs to integer indices internally.
    """
    def __init__(self, file_paths, labels, P, K, is_rgb,
                 cache_images=True, augment=True, steps_per_epoch=None,
                 compute_anthropometrics=False, **kwargs):
        super().__init__(**kwargs)
        self.file_paths = np.array(file_paths)
        self.labels = np.array(labels)
        self.P = int(P)
        self.K = int(K)
        self.batch_size = self.P * self.K
        self.is_rgb = is_rgb
        self.cache_images = cache_images
        self.augment = augment
        # When True (depth-only path with --anthro-weight > 0), compute the
        # per-frame anthropometric vector from the canonical (un-augmented)
        # preprocessed depth and serve it as an extra target in __getitem__.
        # The auxiliary head in HybridModel regresses against it.
        self.compute_anthropometrics = bool(compute_anthropometrics) and not is_rgb
        self.image_cache = {}
        self.anthro_cache = {}  # path → np.float32 ANTHRO_DIM-vector (depth only)

        # Index images by label, keep only labels with >=K images so we can sample K per ID.
        self.indices_by_label = defaultdict(list)
        for idx, lbl in enumerate(self.labels):
            self.indices_by_label[lbl].append(idx)
        self.sampleable_labels = sorted([l for l, idxs in self.indices_by_label.items() if len(idxs) >= self.K])
        if len(self.sampleable_labels) < self.P:
            raise ValueError(
                f"PK sampler needs at least P={self.P} identities with >=K={self.K} images each. "
                f"Found {len(self.sampleable_labels)} valid identities."
            )

        # String label → int label (for the loss).
        self.label_to_int = {lbl: i for i, lbl in enumerate(sorted(set(self.labels)))}

        # Default: enough batches per epoch to cover the full dataset once
        # (each frame seen ~once per epoch). With P=8, K=4 (batch=32) and 10k
        # training images, that's ~330 steps/epoch — comparable to pair mode.
        # The earlier default of max(1, n_ids//P)*K gave only ~8 steps/epoch,
        # which is 100x undertraining vs pair mode.
        if steps_per_epoch is None:
            steps_per_epoch = max(1, len(self.file_paths) // self.batch_size)
        self.steps_per_epoch = int(steps_per_epoch)

        if self.cache_images:
            anthro_tag = ""
            if self.compute_anthropometrics:
                anthro_tag = " + 3D anthros" if ANTHRO_3D else " + 2D anthros"
            print(f"Caching {len(self.file_paths)} images ({'RGB' if is_rgb else 'Depth'}, uint8) for PK sampler"
                  f"{anthro_tag}...")
            for i, path in enumerate(self.file_paths):
                if i % 500 == 0: print(f"  {i}/{len(self.file_paths)}...", end='\r')
                if self.is_rgb:
                    self.image_cache[path] = load_rgb_uint8(path)
                else:
                    if self.compute_anthropometrics and ANTHRO_3D:
                        # Need raw mm depth + foreground mask for 3D unprojection.
                        # Load the raw frame, derive the foreground slab the same way
                        # preprocess_depth_smart does, then compute 3D anthros.
                        raw = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
                        if raw is None:
                            self.anthro_cache[path] = np.zeros(ANTHRO_DIM_3D, dtype=np.float32)
                        else:
                            raw_f = raw.astype(np.float32)
                            valid = raw_f > 0
                            if int(valid.sum()) >= MIN_FOREGROUND_PIXELS:
                                anchor_mm = np.percentile(raw_f[valid], FOREGROUND_CLOSEST_PERCENTILE)
                                fg_raw = valid & (raw_f >= anchor_mm) & (raw_f <= anchor_mm + CLIP_RANGE)
                                if int(fg_raw.sum()) >= MIN_FOREGROUND_PIXELS:
                                    self.anthro_cache[path] = _compute_anthropometrics_3d(raw_f, fg_raw)
                                else:
                                    self.anthro_cache[path] = np.zeros(ANTHRO_DIM_3D, dtype=np.float32)
                            else:
                                self.anthro_cache[path] = np.zeros(ANTHRO_DIM_3D, dtype=np.float32)
                    canonical_depth = load_and_preprocess_single(path, self.is_rgb)  # float32, [-1, 1]
                    self.image_cache[path] = quantize_depth_uint8(canonical_depth)
                    if self.compute_anthropometrics and not ANTHRO_3D:
                        # 2D silhouette anthros from the canonical preprocessed depth.
                        self.anthro_cache[path] = _compute_anthropometrics(canonical_depth[..., 0])
            print("  Done.")

    def __len__(self):
        return self.steps_per_epoch

    def embed_all(self, backbone, chunk=1024):
        """Embed every image in this sampler's path list (in path order, not PK order).
        Returns (N, emb_dim). Needed by the cross-clothing gallery-probe evaluator.
        """
        N = len(self.file_paths)
        embs = []
        for start in range(0, N, chunk):
            chunk_paths = self.file_paths[start:start + chunk]
            if self.cache_images:
                if self.is_rgb:
                    imgs = np.array([normalize_rgb(self.image_cache[p]) for p in chunk_paths])
                else:
                    imgs = np.array([dequantize_depth_uint8(self.image_cache[p]) for p in chunk_paths])
            else:
                imgs = np.array([load_and_preprocess_single(p, self.is_rgb) for p in chunk_paths])
            embs.append(backbone.predict(imgs, batch_size=EVAL_BATCH_SIZE, verbose=0))
            del imgs
        return np.concatenate(embs, axis=0)

    def _read_one(self, path):
        ap = get_augmentation_params() if self.augment else None
        if self.cache_images:
            cached = self.image_cache[path]
            if self.is_rgb:
                if ap is not None:
                    return normalize_rgb(apply_rgb_augmentation(cached, ap))
                return normalize_rgb(cached)
            # depth
            f32 = dequantize_depth_uint8(cached)
            if ap is not None:
                return np.expand_dims(apply_cached_depth_augmentation(f32[:, :, 0], ap), -1)
            return f32
        # uncached fallback
        return load_and_preprocess_single(path, self.is_rgb, augment=self.augment, aug_params=ap)

    def _read_one_or_stack(self, paths):
        """Read either a single frame (len(paths)==1) or stack N depth frames as
        channels of one (H, W, N) input.

        For temporal stack mode, augmentation params are generated ONCE per stack
        and applied identically to all frames — preserving the temporal coherence
        of motion across the stack.
        """
        if len(paths) == 1:
            return self._read_one(paths[0])
        # Stack mode (depth only). Generate ONE set of aug params for all frames.
        ap = get_augmentation_params() if self.augment else None
        frames_2d = []
        for path in paths:
            if self.cache_images:
                cached = self.image_cache[path]
                f32 = dequantize_depth_uint8(cached)[..., 0]   # (H, W)
                if ap is not None:
                    f32 = apply_cached_depth_augmentation(f32, ap)
                frames_2d.append(f32)
            else:
                f32 = load_and_preprocess_single(path, is_rgb=False, augment=self.augment, aug_params=ap)[..., 0]
                frames_2d.append(f32)
        return np.stack(frames_2d, axis=-1)  # (H, W, N) — temporal stack as channels

    def __getitem__(self, index):
        # Sample P identities and K images each, with replacement only if too few IDs.
        if len(self.sampleable_labels) >= self.P:
            sampled = random.sample(self.sampleable_labels, self.P)
        else:
            sampled = random.choices(self.sampleable_labels, k=self.P)
        # Depth temporal stack: each "sample" is a stack of N consecutive (with stride)
        # frames from the same subject. RGB stays single-frame. Stack=1 is identical
        # to the original single-frame behavior.
        is_stack = (not self.is_rgb) and (TEMPORAL_STACK_FRAMES > 1)
        stack_frames = TEMPORAL_STACK_FRAMES if is_stack else 1
        stack_stride = TEMPORAL_STACK_STRIDE if is_stack else 1
        max_offset = (stack_frames - 1) * stack_stride
        imgs, ints, anthros = [], [], []
        for sid in sampled:
            pool = sorted(self.indices_by_label[sid])  # sorted = temporal order
            # Pick K anchor positions, leaving room for the stack at the end of the pool.
            anchors = self._stratified_sample(pool, self.K, max_offset=max_offset)
            for anchor_global_idx in anchors:
                # Find the anchor's position within the sorted pool, then build the
                # stack at positions [anchor_pos, anchor_pos + stride, anchor_pos + 2*stride, ...].
                anchor_pos = pool.index(anchor_global_idx)
                stack_path_indices = [
                    pool[min(anchor_pos + k * stack_stride, len(pool) - 1)]
                    for k in range(stack_frames)
                ]
                stack_paths = [self.file_paths[i] for i in stack_path_indices]
                imgs.append(self._read_one_or_stack(stack_paths))
                ints.append(self.label_to_int[sid])
                if self.compute_anthropometrics:
                    # Anthros come from the first (anchor) frame of the stack —
                    # they're identity features, roughly invariant within the
                    # 2*stride-frame temporal window of the stack.
                    anthros.append(self.anthro_cache[stack_paths[0]])
        if self.compute_anthropometrics:
            # Return labels as a (B, 1+dim) tensor: first column is the int
            # identity label, remaining columns are the anthropometric target.
            # HybridModel.train_step splits this back into int_labels + anthro_targets.
            dim = ANTHRO_DIM_3D if ANTHRO_3D else ANTHRO_DIM
            labels_arr = np.zeros((len(ints), 1 + dim), dtype=np.float32)
            labels_arr[:, 0] = np.array(ints, dtype=np.float32)
            labels_arr[:, 1:] = np.stack(anthros)
            return np.array(imgs), labels_arr
        return np.array(imgs), np.array(ints, dtype=np.int32)

    @staticmethod
    def _stratified_sample(pool, K, max_offset=0):
        """Temporal-stratified sampling: divide the (temporally-ordered) pool into K bins
        and pick one frame from each bin. Guarantees that the K same-ID samples in a PK
        batch span the subject's recording, exposing the triplet loss to within-subject
        POSE variation. Without this, random K-sampling tends to cluster temporally close
        frames together (similar poses), which leads to within-pose identity learning
        but cross-pose collapse at test time — the exact symptom we observed for depth.

        `max_offset` reserves the last `max_offset` frames of the pool from being
        chosen as anchor positions — used for temporal-stack input where each
        anchor position p must have p+max_offset still in pool for the stack to
        fit. For non-stack mode (max_offset=0), behaves as before.
        """
        n = len(pool)
        n_valid = max(1, n - max_offset)
        valid_pool = pool[:n_valid]
        if n_valid <= K:
            # Need replacement to reach K; replicate available valid prefix.
            return list(valid_pool) + random.choices(valid_pool, k=K - n_valid)
        return [random.choice(valid_pool[i * n_valid // K : (i + 1) * n_valid // K]) for i in range(K)]

    def on_epoch_end(self):
        pass  # purely random sampling per __getitem__; nothing to reshuffle

def batch_hard_triplet_loss(embeddings, labels, margin=0.3):
    """Hermans 2017 — In Defense of the Triplet Loss for Person Re-Identification.

    For each anchor in the batch:
        hardest positive = same-label sample with LARGEST distance
        hardest negative = different-label sample with SMALLEST distance
    Loss per anchor = relu(hardest_pos - hardest_neg + margin). Mean over batch.

    Embeddings are assumed L2-normalised, so 1 - cosine_sim is used as distance.
    """
    embeddings = tf.cast(embeddings, tf.float32)
    labels = tf.cast(labels, tf.int32)
    # cosine distance in [0, 2] for L2-normalised vectors
    pairwise_dist = 1.0 - tf.matmul(embeddings, embeddings, transpose_b=True)
    pairwise_dist = tf.maximum(pairwise_dist, 0.0)

    labels_eq = tf.equal(labels[:, None], labels[None, :])  # (B, B)
    eye = tf.eye(tf.shape(embeddings)[0], dtype=tf.bool)
    pos_mask = tf.logical_and(labels_eq, tf.logical_not(eye))
    neg_mask = tf.logical_not(labels_eq)

    # Hardest positive: max distance among positives. Set negatives to -inf so they're ignored by max.
    pos_dist = tf.where(pos_mask, pairwise_dist, tf.fill(tf.shape(pairwise_dist), -1e9))
    hardest_pos = tf.reduce_max(pos_dist, axis=1)

    # Hardest negative: min distance among negatives. Set positives to +inf so they're ignored by min.
    neg_dist = tf.where(neg_mask, pairwise_dist, tf.fill(tf.shape(pairwise_dist), 1e9))
    hardest_neg = tf.reduce_min(neg_dist, axis=1)

    # Drop anchors with no positives (e.g., if some PK row only sampled 1 image due to small K pool).
    has_positive = tf.reduce_any(pos_mask, axis=1)
    triplet = tf.nn.relu(hardest_pos - hardest_neg + margin)
    triplet = tf.where(has_positive, triplet, tf.zeros_like(triplet))
    n_valid = tf.reduce_sum(tf.cast(has_positive, tf.float32))
    return tf.cond(n_valid > 0,
                   lambda: tf.reduce_sum(triplet) / n_valid,
                   lambda: tf.constant(0.0, dtype=tf.float32))

class TripletModel(tf.keras.Model):
    """tf.keras.Model that wraps a backbone with a batch-hard triplet train_step.

    Input: (batch_imgs, batch_int_labels) from a PKBatchSampler.
    Output (during inference): the backbone's L2-normalised embeddings.
    """
    def __init__(self, backbone, margin=0.3, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.margin = float(margin)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")

    def call(self, inputs, training=False):
        return self.backbone(inputs, training=training)

    def train_step(self, data):
        imgs, labels = data
        with tf.GradientTape() as tape:
            embeddings = self.backbone(imgs, training=True)
            triplet_loss = batch_hard_triplet_loss(embeddings, labels, self.margin)
            reg_loss = tf.add_n(self.backbone.losses) if self.backbone.losses else tf.constant(0.0)
            loss = triplet_loss + reg_loss
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        return {"loss": self.loss_tracker.result()}

    @property
    def metrics(self):
        return [self.loss_tracker]


class HybridModel(tf.keras.Model):
    """BNNeck backbone + triplet (on pre-BN) + cross-entropy (on classifier logits).

    Implements the Luo 2019 strong baseline ("A Strong Baseline and Batch
    Normalization Neck for Deep Person Re-identification", CVPRW 2019). The
    auxiliary classification head forces the training-ID embeddings to be linearly
    separable, providing a much stronger supervision signal than triplet alone —
    which on small open-set datasets (BIWI: 22 train IDs) has trivial solutions
    where the model overfits within ~1 epoch.

    Loss = triplet(ft, labels) + ce_weight * CE(classifier(fi), labels) + reg

    Inference returns the same L2-normalised post-BN embedding as TripletModel
    (the classifier is discarded), so the eval code is unchanged.

    The backbone must have been built with `use_bnneck=True` so it exposes the
    named layers 'pre_bnneck' and 'bnneck'.
    """
    def __init__(self, backbone, n_classes, margin=0.3, ce_weight=1.0,
                 weight_decay=1e-4, label_smoothing=0.1,
                 anthro_weight=0.0, anthro_dim=0, **kwargs):
        super().__init__(**kwargs)
        self.backbone = backbone
        # Intermediate model exposing pre-BN ft and post-BN fi (not L2-normalized).
        ft_layer = backbone.get_layer('pre_bnneck')
        fi_layer = backbone.get_layer('bnneck')
        self.feature_model = Model(backbone.input, [ft_layer.output, fi_layer.output],
                                   name='feature_model')
        # Classifier head — no bias (standard BNNeck recipe).
        self.classifier = layers.Dense(
            n_classes, use_bias=False, name='classifier',
            kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
        )
        # Build classifier so its variables register as trainable.
        emb_dim = ft_layer.output.shape[-1]
        _ = self.classifier(tf.zeros((1, emb_dim)))

        # Anthropometric auxiliary regression head — depth-only, predicts the
        # pose-mostly-invariant body-shape statistics (height, width profile,
        # aspect, compactness; see _compute_anthropometrics). When weight=0
        # the head is not built and the loss term is zero.
        self.anthro_weight = float(anthro_weight)
        self.anthro_dim = int(anthro_dim)
        if self.anthro_weight > 0 and self.anthro_dim > 0:
            self.anthro_head = layers.Dense(
                self.anthro_dim, activation=None, name='anthro_head',
                kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
            )
            _ = self.anthro_head(tf.zeros((1, emb_dim)))
        else:
            self.anthro_head = None

        self.margin = float(margin)
        self.ce_weight = float(ce_weight)
        self.n_classes = int(n_classes)
        self.label_smoothing = float(label_smoothing)
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.triplet_tracker = tf.keras.metrics.Mean(name="triplet")
        self.ce_tracker = tf.keras.metrics.Mean(name="ce")
        self.acc_tracker = tf.keras.metrics.SparseCategoricalAccuracy(name="acc")
        if self.anthro_head is not None:
            self.anthro_tracker = tf.keras.metrics.Mean(name="anthro")
        else:
            self.anthro_tracker = None

    def call(self, inputs, training=False):
        # Inference output = backbone's L2-normalised post-BN embedding.
        return self.backbone(inputs, training=training)

    def train_step(self, data):
        imgs, raw_labels = data
        # If the sampler attached anthropometric targets, raw_labels has shape
        # (B, 1 + ANTHRO_DIM): column 0 is the int id-label, rest is the target.
        # Otherwise raw_labels is shape (B,) of int ids.
        if self.anthro_head is not None and len(raw_labels.shape) == 2:
            labels = tf.cast(raw_labels[:, 0], tf.int32)
            anthro_targets = tf.cast(raw_labels[:, 1:1 + self.anthro_dim], tf.float32)
        else:
            labels = tf.cast(raw_labels, tf.int32)
            anthro_targets = None

        with tf.GradientTape() as tape:
            ft, fi = self.feature_model(imgs, training=True)
            logits = self.classifier(fi)

            # Triplet loss on L2-normalised pre-BN features (cosine-distance space).
            ft_normed = tf.math.l2_normalize(ft, axis=1)
            triplet = batch_hard_triplet_loss(ft_normed, labels, self.margin)

            # Cross-entropy with label smoothing (Luo 2019 default ε=0.1).
            one_hot = tf.one_hot(labels, depth=self.n_classes)
            if self.label_smoothing > 0:
                one_hot = one_hot * (1.0 - self.label_smoothing) + \
                          self.label_smoothing / float(self.n_classes)
            ce = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=one_hot, logits=logits)
            )

            # Anthropometric auxiliary MSE loss (depth only; zero otherwise).
            if self.anthro_head is not None and anthro_targets is not None:
                anthro_pred = self.anthro_head(fi)
                anthro_loss = tf.reduce_mean(tf.square(anthro_pred - anthro_targets))
            else:
                anthro_loss = tf.constant(0.0, dtype=tf.float32)

            # Regularization (L2 on conv kernels + classifier + anthro head if present).
            reg_terms = list(self.feature_model.losses) + list(self.classifier.losses)
            if self.anthro_head is not None:
                reg_terms = reg_terms + list(self.anthro_head.losses)
            reg = tf.add_n(reg_terms) if reg_terms else tf.constant(0.0)

            loss = triplet + self.ce_weight * ce + self.anthro_weight * anthro_loss + reg

        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.triplet_tracker.update_state(triplet)
        self.ce_tracker.update_state(ce)
        self.acc_tracker.update_state(labels, logits)
        out = {
            "loss":    self.loss_tracker.result(),
            "triplet": self.triplet_tracker.result(),
            "ce":      self.ce_tracker.result(),
            "acc":     self.acc_tracker.result(),
        }
        if self.anthro_tracker is not None:
            self.anthro_tracker.update_state(anthro_loss)
            out["anthro"] = self.anthro_tracker.result()
        return out

    @property
    def metrics(self):
        m = [self.loss_tracker, self.triplet_tracker, self.ce_tracker, self.acc_tracker]
        if self.anthro_tracker is not None:
            m.append(self.anthro_tracker)
        return m

def evaluate_within_test_galprobe(backbone, val_imgs_uint8, val_paths, val_labels, is_rgb,
                                  gallery_token='still', probe_token='walking',
                                  probe_window=10):
    """Within-test gallery-probe eval — the standard BIWI/IAS-Lab Re-ID protocol.

    Splits the test set by directory token (default: 'Still' → gallery, 'Walking' → probe).
    Both gallery and probe come from the SAME test recording, so clothing is constant.
    The task is cross-pose matching: given a Walking frame of subject X, find the nearest
    Still frame and check whether it's also subject X.

    Reports BOTH per-frame matching (each probe → nearest gallery frame) AND
    per-subject-prototype matching (each probe → nearest of K mean embeddings, one per
    gallery subject). Prototype matching collapses per-frame noise and is the standard
    Re-ID "single-shot from gallery template" formulation.

    Returns None if the test set doesn't have both Still and Walking subdirs.
    """
    val_paths = [str(p).lower() for p in val_paths]
    gallery_idx = [i for i, p in enumerate(val_paths) if gallery_token in p]
    probe_idx = [i for i, p in enumerate(val_paths) if probe_token in p]
    if not gallery_idx or not probe_idx:
        print(f"  [within-test GP] cannot split val_paths on tokens "
              f"({gallery_token!r}/{probe_token!r}) — skipping (likely IAS-Lab format).")
        return None

    print(f"  [within-test GP] gallery={len(gallery_idx)} ({gallery_token}), "
          f"probe={len(probe_idx)} ({probe_token})")

    # Embed all val once, then index.
    deq = normalize_rgb if is_rgb else dequantize_depth_uint8
    all_embs = []
    for start in range(0, len(val_imgs_uint8), EVAL_BATCH_SIZE):
        batch = deq(val_imgs_uint8[start:start + EVAL_BATCH_SIZE])
        all_embs.append(backbone.predict(batch, verbose=0))
    all_embs = np.vstack(all_embs)
    val_labels = np.asarray(val_labels)

    gallery_embs = all_embs[gallery_idx]
    gallery_labels = val_labels[gallery_idx]
    probe_embs = all_embs[probe_idx]
    probe_labels = val_labels[probe_idx]

    # Per-frame matching: each probe -> nearest gallery FRAME.
    D = pairwise_distances(probe_embs, gallery_embs, metric='cosine')
    frame_metrics = _rank_metrics(D, probe_labels, gallery_labels)

    # Per-subject-prototype matching: each probe -> nearest gallery SUBJECT (mean embed).
    proto_embs, proto_labels = _per_subject_prototypes(gallery_embs, gallery_labels)
    D_proto = pairwise_distances(probe_embs, proto_embs, metric='cosine')
    proto_metrics = _rank_metrics(D_proto, probe_labels, proto_labels)

    # ---- K-reciprocal re-ranking (Zhong 2017) at frame and prototype levels.
    # Frame-level: thousands × thousands → biggest expected lift.
    # Prototype-level: thousands × 28 → still useful (probes cluster by identity).
    print(f"  [within-test GP] applying k-reciprocal re-ranking (k1=20, k2=6, λ=0.3)...")
    try:
        D_rerank = _k_reciprocal_rerank(probe_embs, gallery_embs)
        frame_rerank = _rank_metrics(D_rerank, probe_labels, gallery_labels)
    except (MemoryError, ValueError) as e:
        print(f"    skipped frame rerank: {type(e).__name__}: {e}")
        frame_rerank = None
    try:
        D_proto_rerank = _k_reciprocal_rerank(probe_embs, proto_embs)
        proto_rerank = _rank_metrics(D_proto_rerank, probe_labels, proto_labels)
    except (MemoryError, ValueError) as e:
        print(f"    skipped proto rerank: {type(e).__name__}: {e}")
        proto_rerank = None

    # Multi-frame probe averaging — group N consecutive Walking frames per subject
    # into a single "video probe", then re-match against gallery prototypes. Helps
    # depth especially (per-frame silhouette noise averages out across the window).
    seq_probe_embs, seq_probe_labels = _multi_frame_average(probe_embs, probe_labels,
                                                            window=probe_window)
    D_seq = pairwise_distances(seq_probe_embs, proto_embs, metric='cosine')
    seq_metrics = _rank_metrics(D_seq, seq_probe_labels, proto_labels)

    # Re-rank seq-probe distances too (small matrix — fast).
    try:
        D_seq_rerank = _k_reciprocal_rerank(seq_probe_embs, proto_embs)
        seq_rerank = _rank_metrics(D_seq_rerank, seq_probe_labels, proto_labels)
    except (MemoryError, ValueError) as e:
        print(f"    skipped seq rerank: {type(e).__name__}: {e}")
        seq_rerank = None

    out = {
        # Per-frame matching (probe → nearest of all gallery frames).
        "rank1": frame_metrics["rank1"],
        "rank5": frame_metrics["rank5"],
        "mAP":   frame_metrics["mAP"],
        # Per-subject-prototype matching (single probe frame → nearest subject mean).
        "proto_rank1": proto_metrics["rank1"],
        "proto_rank5": proto_metrics["rank5"],
        "proto_mAP":   proto_metrics["mAP"],
        # Multi-frame video probe (N-frame window) → nearest subject mean. RECOMMENDED.
        "seq_rank1": seq_metrics["rank1"],
        "seq_rank5": seq_metrics["rank5"],
        "seq_mAP":   seq_metrics["mAP"],
        "seq_probe_window": int(probe_window),
        "n_gallery":      int(len(gallery_idx)),
        "n_gallery_subj": int(len(proto_labels)),
        "n_probe":        int(len(probe_idx)),
        "n_seq_probe":    int(len(seq_probe_embs)),
    }
    # K-reciprocal rerank variants (Zhong 2017). Often +5-10pp R1 vs unranked.
    if frame_rerank is not None:
        out.update({
            "rerank_rank1": frame_rerank["rank1"],
            "rerank_rank5": frame_rerank["rank5"],
            "rerank_mAP":   frame_rerank["mAP"],
        })
    if proto_rerank is not None:
        out.update({
            "rerank_proto_rank1": proto_rerank["rank1"],
            "rerank_proto_rank5": proto_rerank["rank5"],
            "rerank_proto_mAP":   proto_rerank["mAP"],
        })
    if seq_rerank is not None:
        out.update({
            "rerank_seq_rank1": seq_rerank["rank1"],
            "rerank_seq_rank5": seq_rerank["rank5"],
            "rerank_seq_mAP":   seq_rerank["mAP"],
        })
    return out

def evaluate_cross_clothing_galprobe(backbone, gallery_imgs_uint8, gallery_labels,
                                     probe_imgs_uint8, probe_labels, is_rgb,
                                     probe_window=10):
    """Open-set CROSS-CLOTHING gallery-probe eval — the regime where depth's
    clothing-invariant body-shape cue should match or beat RGB.

    Unlike `evaluate_within_test_galprobe` (which splits ONE same-clothing
    recording into Still/Walking), this takes a SEPARATE gallery and probe:
      gallery = held-out test subjects in outfit A (their Training/ frames),
      probe   = same subjects in outfit B (their Testing/ frames).
    The subjects are unseen in training (open-set), and the clothing differs
    between gallery and probe. RGB's dominant clothing cue flips to a different
    outfit (becomes misleading); depth's body geometry is unchanged.

    Same metric suite as within-test: frame-match, prototype, seq-probe, plus
    k-reciprocal rerank variants. Returns a dict (keys identical to within-test,
    so the collate/results plumbing is reused with an 'xc_' prefix by the caller).
    """
    deq = normalize_rgb if is_rgb else dequantize_depth_uint8

    def _embed(imgs_uint8):
        embs = []
        for start in range(0, len(imgs_uint8), EVAL_BATCH_SIZE):
            batch = deq(imgs_uint8[start:start + EVAL_BATCH_SIZE])
            embs.append(backbone.predict(batch, verbose=0))
        return np.vstack(embs)

    print(f"  [cross-clothing GP] embedding gallery={len(gallery_imgs_uint8)} (outfit A) + "
          f"probe={len(probe_imgs_uint8)} (outfit B)...")
    gallery_embs = _embed(gallery_imgs_uint8)
    probe_embs = _embed(probe_imgs_uint8)
    gallery_labels = np.asarray(gallery_labels)
    probe_labels = np.asarray(probe_labels)

    # Per-frame matching: each probe -> nearest gallery FRAME (chunked for memory).
    frame_metrics = _rank_metrics(
        pairwise_distances(probe_embs, gallery_embs, metric='cosine'),
        probe_labels, gallery_labels)

    # Per-subject-prototype matching: collapse gallery (outfit A) to one mean per subject.
    proto_embs, proto_labels = _per_subject_prototypes(gallery_embs, gallery_labels)
    D_proto = pairwise_distances(probe_embs, proto_embs, metric='cosine')
    proto_metrics = _rank_metrics(D_proto, probe_labels, proto_labels)

    print(f"  [cross-clothing GP] applying k-reciprocal re-ranking (k1=20, k2=6, λ=0.3)...")
    try:
        frame_rerank = _rank_metrics(
            _k_reciprocal_rerank(probe_embs, gallery_embs), probe_labels, gallery_labels)
    except (MemoryError, ValueError) as e:
        print(f"    skipped frame rerank: {type(e).__name__}: {e}")
        frame_rerank = None
    try:
        proto_rerank = _rank_metrics(
            _k_reciprocal_rerank(probe_embs, proto_embs), probe_labels, proto_labels)
    except (MemoryError, ValueError) as e:
        print(f"    skipped proto rerank: {type(e).__name__}: {e}")
        proto_rerank = None

    # Multi-frame video probe (average N consecutive probe frames per subject).
    seq_probe_embs, seq_probe_labels = _multi_frame_average(probe_embs, probe_labels,
                                                            window=probe_window)
    D_seq = pairwise_distances(seq_probe_embs, proto_embs, metric='cosine')
    seq_metrics = _rank_metrics(D_seq, seq_probe_labels, proto_labels)
    try:
        seq_rerank = _rank_metrics(
            _k_reciprocal_rerank(seq_probe_embs, proto_embs), seq_probe_labels, proto_labels)
    except (MemoryError, ValueError) as e:
        print(f"    skipped seq rerank: {type(e).__name__}: {e}")
        seq_rerank = None

    out = {
        "rank1": frame_metrics["rank1"], "rank5": frame_metrics["rank5"], "mAP": frame_metrics["mAP"],
        "proto_rank1": proto_metrics["rank1"], "proto_rank5": proto_metrics["rank5"], "proto_mAP": proto_metrics["mAP"],
        "seq_rank1": seq_metrics["rank1"], "seq_rank5": seq_metrics["rank5"], "seq_mAP": seq_metrics["mAP"],
        "seq_probe_window": int(probe_window),
        "n_gallery": int(len(gallery_embs)),
        "n_gallery_subj": int(len(proto_labels)),
        "n_probe": int(len(probe_embs)),
        "n_seq_probe": int(len(seq_probe_embs)),
    }
    if frame_rerank is not None:
        out.update({"rerank_rank1": frame_rerank["rank1"], "rerank_rank5": frame_rerank["rank5"],
                    "rerank_mAP": frame_rerank["mAP"]})
    if proto_rerank is not None:
        out.update({"rerank_proto_rank1": proto_rerank["rank1"], "rerank_proto_rank5": proto_rerank["rank5"],
                    "rerank_proto_mAP": proto_rerank["mAP"]})
    if seq_rerank is not None:
        out.update({"rerank_seq_rank1": seq_rerank["rank1"], "rerank_seq_rank5": seq_rerank["rank5"],
                    "rerank_seq_mAP": seq_rerank["mAP"]})
    return out


def _build_eval_stacks(val_paths, val_labels, stack_frames, stack_stride):
    """Group val_paths by subject (preserving the provided temporal order), then
    build one temporal stack per valid anchor position per subject.

    Args:
        val_paths: list of file paths in temporal order PER SUBJECT (the loaders
            return paths sorted lexically, which matches temporal order for
            BIWI/IAS-Lab naming conventions).
        val_labels: parallel labels.
        stack_frames: number of frames per stack.
        stack_stride: temporal offset between consecutive frames in a stack.

    Returns:
        anchor_paths: list of length N_stacks; the anchor (first-frame) path
            of each stack. Used downstream to split Still vs Walking by path
            token, identifying the stack's "location" in the recording.
        anchor_labels: list of length N_stacks; the subject label per stack.
        stack_path_lists: list of length N_stacks; each entry is a list of
            stack_frames paths in temporal order.
    """
    max_offset = (stack_frames - 1) * stack_stride
    # Preserve input order so subjects appear in the same order as in val_paths.
    by_subj = defaultdict(list)
    for p, lbl in zip(val_paths, val_labels):
        by_subj[lbl].append(p)

    anchor_paths, anchor_labels, stack_path_lists = [], [], []
    for lbl, paths in by_subj.items():
        n = len(paths)
        n_valid = n - max_offset
        if n_valid <= 0:
            # Pool too short for a full stack; replicate the last frame to fill.
            paths = list(paths) + [paths[-1]] * (max_offset - n + 1)
            n_valid = 1
            n = len(paths)
        for anchor_pos in range(n_valid):
            stack = [paths[anchor_pos + k * stack_stride] for k in range(stack_frames)]
            anchor_paths.append(stack[0])
            anchor_labels.append(lbl)
            stack_path_lists.append(stack)
    return anchor_paths, anchor_labels, stack_path_lists


def _load_eval_images(paths, labels, is_rgb, tag="eval set"):
    """Load a set of frames as uint8 for evaluation, handling temporal-stack mode.

    Returns (imgs_uint8, out_paths, out_labels):
      - single-frame mode: imgs_uint8 has shape (N, H, W, C); out_paths/out_labels
        are the inputs unchanged.
      - temporal-stack mode (depth, TEMPORAL_STACK_FRAMES > 1): frames are grouped
        per subject in temporal order and assembled into (N_stacks, H, W, stack_frames);
        out_paths/out_labels are one-per-stack (anchor frame's path/label).

    Used for both the within-test val set and the cross-clothing gallery so they
    are encoded identically.
    """
    is_stack = (not is_rgb) and (TEMPORAL_STACK_FRAMES > 1)
    if is_stack:
        stack_frames = TEMPORAL_STACK_FRAMES
        stack_stride = TEMPORAL_STACK_STRIDE
        anchor_paths, anchor_labels, stack_path_lists = \
            _build_eval_stacks(paths, labels, stack_frames, stack_stride)
        print(f"Pre-loading {tag} (temporal stack: {stack_frames}-frame, stride {stack_stride}; "
              f"{len(anchor_paths)} stacks from {len(paths)} raw frames)...")
        unique_paths = sorted({p for plist in stack_path_lists for p in plist})
        frame_cache = {}
        for i, p in enumerate(unique_paths):
            if i % 500 == 0:
                print(f"  caching {i}/{len(unique_paths)}...", end='\r')
            frame_cache[p] = quantize_depth_uint8(load_and_preprocess_single(p, is_rgb=False))[..., 0]
        print(f"  done; assembling {len(stack_path_lists)} stacks.")
        imgs = np.stack(
            [np.stack([frame_cache[p] for p in plist], axis=-1) for plist in stack_path_lists],
            axis=0,
        )
        return imgs, anchor_paths, np.array(anchor_labels)
    # Single-frame mode.
    print(f"Pre-loading {tag}...")
    if is_rgb:
        imgs = np.array([load_rgb_uint8(p) for p in paths])
    else:
        imgs = np.array([quantize_depth_uint8(load_and_preprocess_single(p, is_rgb)) for p in paths])
    return imgs, list(paths), np.array(labels)


class EvaluationCallback(Callback):
    def __init__(self, backbone, val_paths, val_labels, is_rgb, results_dict):
        super().__init__()
        self.backbone = backbone
        self.results_dict = results_dict
        self.modality = "RGB" if is_rgb else "DEPTH"
        self.is_rgb = is_rgb
        self.val_imgs_uint8, self.val_paths, self.val_labels = _load_eval_images(
            val_paths, val_labels, is_rgb, tag=f"{self.modality} validation set")

    def on_epoch_end(self, epoch, logs=None):
        embs = []
        deq = normalize_rgb if self.is_rgb else dequantize_depth_uint8
        for i in range(0, len(self.val_imgs_uint8), EVAL_BATCH_SIZE):
            batch_imgs = deq(self.val_imgs_uint8[i:i + EVAL_BATCH_SIZE])
            embs.append(self.backbone.predict(batch_imgs, verbose=0))
        embs = np.vstack(embs)

        dist_matrix = pairwise_distances(embs, metric='cosine')
        np.fill_diagonal(dist_matrix, np.inf)
        
        rank1, rank5 = 0, 0
        aps = []
        total = len(self.val_labels)
        
        for i in range(total):
            dists = dist_matrix[i]
            sorted_idx = np.argsort(dists)
            matches = (self.val_labels[sorted_idx] == self.val_labels[i])
            
            if matches[0]: rank1 += 1
            if np.any(matches[:5]): rank5 += 1
            
            num_valid = np.sum(matches)
            if num_valid > 0:
                old_recall = 0.0
                old_precision = 1.0
                ap = 0.0
                intersect_size = 0
                for j, match in enumerate(matches):
                    if match:
                        intersect_size += 1
                        recall = intersect_size / num_valid
                        precision = intersect_size / (j + 1)
                        ap += (recall - old_recall) * ((old_precision + precision) / 2.0)
                        old_recall = recall
                        old_precision = precision
                aps.append(ap)
                
        r1 = rank1 / total * 100.0
        r5 = rank5 / total * 100.0
        mAP = np.mean(aps) * 100.0 if aps else 0.0

        intra_dists = []
        inter_dists = []
        unique_labels = np.unique(self.val_labels)
        
        for label in unique_labels:
            mask = self.val_labels == label
            if np.sum(mask) > 1:
                class_embs = embs[mask]
                class_dists = pairwise_distances(class_embs, metric='cosine')
                intra_dists.extend(class_dists[np.triu_indices_from(class_dists, k=1)])
        
        for i in range(len(unique_labels)):
            for j in range(i+1, len(unique_labels)):
                mask_i = self.val_labels == unique_labels[i]
                mask_j = self.val_labels == unique_labels[j]
                if np.sum(mask_i) > 0 and np.sum(mask_j) > 0:
                    inter_dist = pairwise_distances(embs[mask_i], embs[mask_j], metric='cosine')
                    inter_dists.extend(inter_dist.flatten())
        
        intra_mean = np.mean(intra_dists) if intra_dists else 0
        inter_mean = np.mean(inter_dists) if inter_dists else 0
        separation_ratio = inter_mean / intra_mean if intra_mean > 0 else 0
        
        if epoch % 10 == 0 or epoch == 0:
            top_k = min(5, len(unique_labels))
            confusion = np.zeros((top_k, top_k), dtype=int)
            top_labels = unique_labels[:top_k]
            
            for i in range(total):
                if self.val_labels[i] not in top_labels:
                    continue
                dists = dist_matrix[i]
                pred_idx = np.argmin(dists)
                pred_label = self.val_labels[pred_idx]
                
                true_idx = np.where(top_labels == self.val_labels[i])[0]
                pred_idx_in_top = np.where(top_labels == pred_label)[0]
                
                if len(true_idx) > 0 and len(pred_idx_in_top) > 0:
                    confusion[true_idx[0], pred_idx_in_top[0]] += 1
            
            if epoch % 10 == 0:
                print(f"\n   Confusion (top {top_k} IDs):")
                print(f"   {confusion}")
        
        print(f" - {self.modality} | R1: {r1:.2f}% | R5: {r5:.2f}% | mAP: {mAP:.2f}% | Sep: {separation_ratio:.2f}")
        
        if logs is not None:
            logs['val_rank1'] = r1
        
        if r1 > self.results_dict['best_rank1']:
            self.results_dict['best_rank1'] = r1
            self.results_dict['best_rank5'] = r5
            self.results_dict['best_mAP'] = mAP
            self.results_dict['best_separation'] = separation_ratio
            self.results_dict['best_epoch'] = epoch + 1

def _basic_block(x, filters, stride, weight_decay):
    shortcut = x
    x = layers.Conv2D(filters, 3, strides=stride, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, strides=1, padding='same',
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                      use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if stride != 1 or shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=stride, padding='same',
                                 kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                                 use_bias=False)(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

def _build_smallresnet_features(input_shape, weight_decay, base_width):
    """SmallResNet feature extractor (v1, trained from scratch).

    Returns (inputs, features_after_dense512) — features are post-Dense(512)+BN+Dropout
    so the BNNeck head can be attached uniformly to both this and pretrained backbones.
    """
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(base_width, 7, strides=2, padding='same', use_bias=False,
                      kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(3, strides=2, padding='same')(x)

    x = _basic_block(x, base_width,     stride=1, weight_decay=weight_decay)
    x = _basic_block(x, base_width,     stride=1, weight_decay=weight_decay)
    x = _basic_block(x, base_width * 2, stride=2, weight_decay=weight_decay)
    x = _basic_block(x, base_width * 2, stride=1, weight_decay=weight_decay)
    x = _basic_block(x, base_width * 4, stride=2, weight_decay=weight_decay)
    x = _basic_block(x, base_width * 4, stride=1, weight_decay=weight_decay)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    return inputs, x


_PRETRAINED_FACTORIES = {
    # name → (tf.keras.applications factory, preprocess_input function)
    # Each factory accepts (weights, include_top, input_shape, pooling).
    # ConvNeXt family (Liu et al. 2022): modern CNN matching ViT performance.
    # All four ConvNeXt variants share the same factory pattern — the 1-channel
    # adapter (channel-averaged stem conv) works identically for all of them.
    "convnext_tiny":   (lambda: tf.keras.applications.ConvNeXtTiny,
                        lambda: tf.keras.applications.convnext.preprocess_input),
    "convnext_small":  (lambda: tf.keras.applications.ConvNeXtSmall,
                        lambda: tf.keras.applications.convnext.preprocess_input),
    "convnext_base":   (lambda: tf.keras.applications.ConvNeXtBase,
                        lambda: tf.keras.applications.convnext.preprocess_input),
    # EfficientNet — note: built-in Normalization layer's mean/variance are
    # length-3, so the factory may refuse 1-channel input. The error path in
    # _build_pretrained_features catches it cleanly. RGB works fine.
    "efficientnet_b0": (lambda: tf.keras.applications.EfficientNetB0,
                        lambda: tf.keras.applications.efficientnet.preprocess_input),
    "efficientnet_b2": (lambda: tf.keras.applications.EfficientNetB2,
                        lambda: tf.keras.applications.efficientnet.preprocess_input),
    # ResNetV2 — no built-in preprocessing; expects [-1, 1] which matches our pipeline.
    "resnet50v2":      (lambda: tf.keras.applications.ResNet50V2,
                        lambda: tf.keras.applications.resnet_v2.preprocess_input),
}


def _iter_all_layers(model):
    """Yield every layer in `model`, recursing into nested Sequential/Functional
    sub-models. tf.keras.applications models often wrap layers (e.g., ConvNeXt's
    stem) inside Sequential containers, so a flat `model.layers` scan misses
    the inner Conv2D layers we want to surgically modify."""
    for layer in model.layers:
        if isinstance(layer, tf.keras.Model):  # Sequential & Functional both are Models
            yield from _iter_all_layers(layer)
        else:
            yield layer


def _copy_weights_3ch_to_1ch(dst_1ch_model, src_3ch_model):
    """Copy weights from `src` (3-channel pretrained) to `dst` (1-channel, freshly
    instantiated with the same architecture). Matches layers by name (recursing
    into nested Sequential/Model containers). For Conv2D kernels whose input-
    channel dimension shrinks from 3 → 1 (the stem conv that receives raw image
    input), the weights are *channel-averaged* over the input-channel axis —
    giving a genuine 1-channel first layer initialised from RGB pretraining.

    Layers present only in `src` (e.g., the PreStem Normalization that
    tf.keras.applications.ConvNeXt skips for non-3-channel inputs) are simply
    not looked up and their src-side weights are ignored. Layers present only
    in `dst` (rare) keep their random init.
    """
    # Index src layers by name (recursive walk into nested models).
    src_index = {}
    for layer in _iter_all_layers(src_3ch_model):
        src_index[layer.name] = layer

    n_copied = 0
    n_averaged = 0
    n_skipped = 0
    for dst_layer in _iter_all_layers(dst_1ch_model):
        src_layer = src_index.get(dst_layer.name)
        if src_layer is None:
            continue
        src_w = src_layer.get_weights()
        dst_w = dst_layer.get_weights()
        if len(src_w) != len(dst_w):
            n_skipped += 1
            continue
        new_w = []
        for sw, dw in zip(src_w, dst_w):
            if sw.shape == dw.shape:
                new_w.append(sw)
            elif (sw.ndim == 4 and dw.ndim == 4
                  and sw.shape[0] == dw.shape[0] and sw.shape[1] == dw.shape[1]
                  and sw.shape[3] == dw.shape[3]
                  and dw.shape[2] == 1 and sw.shape[2] == 3):
                # Stem Conv2D kernel: average input channels 3 → 1.
                new_w.append(sw.mean(axis=2, keepdims=True).astype(sw.dtype))
                n_averaged += 1
            else:
                # Shape mismatch we don't understand — keep dst's default init
                # rather than crash, and warn so we can investigate.
                print(f"  [1ch-adapter] WARNING: shape mismatch in {dst_layer.name}: "
                      f"src={sw.shape}, dst={dw.shape}; using default init for this weight.")
                new_w.append(dw)
                n_skipped += 1
        dst_layer.set_weights(new_w)
        n_copied += 1
    print(f"  [1ch-adapter] copied weights of {n_copied} layers "
          f"({n_averaged} kernel-averaged, {n_skipped} weight-mismatches skipped).")


def _build_pretrained_features(input_shape, weight_decay, backbone_name):
    """Build a tf.keras.applications backbone + post-GAP+Dense feature head.

    For 3-channel input: loads ImageNet weights directly.
    For 1-channel input: loads ImageNet weights then channel-averages the first
    Conv2D to make a genuine 1-channel model. No replication / no learnable
    projection — the architecture is honestly single-channel for depth.

    Input-range handling: our preprocessing pipeline outputs [-1, 1] for both
    modalities. ResNetV2 expects [-1, 1] natively (we're fine). ConvNeXt and
    EfficientNet have *built-in* Rescaling+Normalization layers that expect
    raw [0, 255] inputs — for these we insert a `(v + 1) * 127.5` rescaling
    Lambda at the entry so the model sees what it was trained on.

    Returns (inputs, features_after_dense512) matching _build_smallresnet_features.
    """
    H, W, C = input_shape
    if backbone_name not in _PRETRAINED_FACTORIES:
        raise ValueError(f"Unknown pretrained backbone: {backbone_name}")
    factory_thunk, _ = _PRETRAINED_FACTORIES[backbone_name]
    Factory = factory_thunk()

    if not USE_PRETRAINED:
        # FROM SCRATCH: random init, same architecture. Isolates architecture
        # from ImageNet transfer. No channel-averaging needed — there are no
        # pretrained weights to harvest; the factory builds a native C-channel net.
        print(f"  [{backbone_name}] FROM SCRATCH (weights=None, {C}-channel input)")
        ext = Factory(weights=None, include_top=False, input_shape=(H, W, C), pooling=None)
    elif C == 3:
        ext = Factory(weights="imagenet", include_top=False,
                      input_shape=(H, W, 3), pooling=None)
    elif C == 1:
        # Build the 3-channel pretrained model to harvest weights.
        src_3ch = Factory(weights="imagenet", include_top=False,
                          input_shape=(H, W, 3), pooling=None)
        # Build an untrained 1-channel version with the same architecture (the
        # factory natively supports arbitrary input_shape; for ConvNeXt it skips
        # the PreStem Normalization since that expects 3 channels).
        try:
            ext = Factory(weights=None, include_top=False,
                          input_shape=(H, W, 1), pooling=None)
        except Exception as e:
            raise RuntimeError(
                f"Backbone {backbone_name} does not support 1-channel input "
                f"natively (factory raised {type(e).__name__}: {e}). EfficientNet's "
                f"built-in Normalization layer for example expects 3 channels."
            )
        # Copy weights from the 3-channel pretrained model into the 1-channel
        # model. The stem Conv2D kernel gets channel-averaged from (k, k, 3, F)
        # to (k, k, 1, F); every other layer copies verbatim (same shapes).
        _copy_weights_3ch_to_1ch(ext, src_3ch)
        # Release the 3-channel source; we don't need it again.
        del src_3ch
    else:
        raise ValueError(f"Pretrained backbones support C ∈ {{1, 3}}, got C={C}")

    # Some applications models (ConvNeXt, EfficientNet) bake Rescaling+Normalization
    # into their graph expecting raw [0, 255] inputs; ResNetV2 expects [-1, 1] (matches
    # our preprocessing). The 1-channel adapter strips those pre-Conv2D layers, so the
    # Conv2D weights — trained on post-normalization inputs in roughly [-2, +2] — get
    # our [-1, 1] depth directly, which lands in a reasonable range that fine-tuning
    # absorbs over the first epoch or two.
    needs_rescale = (C == 3 and backbone_name in
                     {"convnext_tiny", "convnext_small", "convnext_base",
                      "efficientnet_b0", "efficientnet_b2"})

    inputs = layers.Input(shape=(H, W, C))
    if needs_rescale:
        x = layers.Lambda(lambda v: (v + 1.0) * 127.5,
                          name='rescale_to_0_255')(inputs)
    else:
        x = inputs
    feat = ext(x)

    # Pool + project to 512-d so the BNNeck head matches the SmallResNet path's
    # expected input dimension (the head's first layer projects 512 → emb_dim).
    x = layers.GlobalAveragePooling2D()(feat)
    x = layers.Dense(512, activation='relu',
                     kernel_regularizer=tf.keras.regularizers.l2(weight_decay),
                     name=f"{backbone_name}_proj_512")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(DROPOUT_RATE)(x)
    return inputs, x


def get_backbone(input_shape, emb_dim, weight_decay, base_width=None,
                 use_bnneck=False, backbone="smallresnet"):
    """Build a backbone whose output is the L2-normalised retrieval embedding.

    Dispatches on `backbone`:
      - "smallresnet" (default): trained-from-scratch SmallResNet (v1 path).
      - "convnext_tiny" / "efficientnet_b0" / "efficientnet_b2" / "resnet50v2":
            tf.keras.applications model with ImageNet-pretrained weights.
            For 1-channel depth, the first Conv2D is channel-averaged from the
            3-channel pretrained weights so the model is genuinely 1-channel.

    BNNeck head (`use_bnneck=True`, Luo 2019):
        feat → Dense(emb_dim)[name='pre_bnneck'] → BN[name='bnneck'] → L2norm
    where `pre_bnneck` feeds triplet loss, `bnneck` feeds the auxiliary
    classifier (in `HybridModel`) and L2(`bnneck`) is the retrieval embedding.
    """
    if base_width is None:
        base_width = RESNET_BASE_WIDTH

    if backbone == "smallresnet":
        inputs, x = _build_smallresnet_features(input_shape, weight_decay, base_width)
        name = "small_resnet_backbone"
    else:
        inputs, x = _build_pretrained_features(input_shape, weight_decay, backbone)
        name = f"{backbone}_backbone"

    if use_bnneck:
        ft = layers.Dense(emb_dim, name='pre_bnneck',
                          kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        fi = layers.BatchNormalization(name='bnneck', scale=False)(ft)
        outputs = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1),
                                name='retrieval_emb')(fi)
    else:
        x = layers.Dense(emb_dim,
                         kernel_regularizer=tf.keras.regularizers.l2(weight_decay))(x)
        outputs = layers.Lambda(lambda v: tf.math.l2_normalize(v, axis=1))(x)

    return Model(inputs, outputs, name=name)

def make_siamese_densenet(input_shape, weight_decay):
    backbone = get_backbone(input_shape, EMB_DIM, weight_decay, backbone=BACKBONE)
    
    a = tf.keras.Input(shape=input_shape)
    b = tf.keras.Input(shape=input_shape)
    
    feat_a = backbone(a)
    feat_b = backbone(b)
    
    def cosine_sim(feats):
        x, y = feats
        return tf.reduce_sum(x * y, axis=1, keepdims=True)
    
    distance = layers.Lambda(cosine_sim)([feat_a, feat_b])
    
    return Model(inputs=[a, b], outputs=distance), backbone

def loss_fn(margin):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        pos_loss = y_true * (1.0 - y_pred)
        neg_loss = (1.0 - y_true) * tf.nn.relu(y_pred - margin)
        
        return tf.reduce_mean(pos_loss + neg_loss)
    return loss

def triplet_loss_fn(margin):
    def loss(y_true, y_pred):
        y_true = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)
        
        pos_mask = y_true > 0.5
        neg_mask = y_true < 0.5
        
        pos_sim = tf.boolean_mask(y_pred, pos_mask)
        neg_sim = tf.boolean_mask(y_pred, neg_mask)
        
        pos_dist = 1.0 - pos_sim
        neg_dist = 1.0 - neg_sim
        
        pos_dist_expanded = tf.expand_dims(pos_dist, 1)
        neg_dist_expanded = tf.expand_dims(neg_dist, 0)
        
        triplet_loss = tf.nn.relu(pos_dist_expanded - neg_dist_expanded + margin)
        
        num_valid = tf.cast(tf.size(pos_sim) * tf.size(neg_sim), tf.float32)
        loss_sum = tf.reduce_sum(triplet_loss)
        
        return tf.where(num_valid > 0, loss_sum / num_valid, 0.0)
    return loss

def run_training_session(modality_name, train_p, train_l, val_p, val_l,
                         partial_writer=None, xgallery_p=None, xgallery_l=None):
    tf.keras.backend.clear_session()
    gc.collect()

    is_rgb = (modality_name == "RGB")
    # Channel count: RGB always 3 (single-frame colour). Depth: 1 single-frame,
    # or TEMPORAL_STACK_FRAMES if temporal stack input is enabled.
    depth_channels = TEMPORAL_STACK_FRAMES if (not is_rgb and TEMPORAL_STACK_FRAMES > 1) else 1
    input_shape = (IMG_H, IMG_W, 3) if is_rgb else (IMG_H, IMG_W, depth_channels)
    do_cache = True
    margin = MARGIN_RGB if is_rgb else MARGIN_DEPTH
    weight_decay = WEIGHT_DECAY_RGB if is_rgb else WEIGHT_DECAY_DEPTH

    augment_data = AUGMENT
    use_hard_negatives = HARDNEG_FREQ > 0  # only used in pair mode

    use_bnneck_here = (TRAINING_MODE == "triplet" and USE_BNNECK)
    use_hybrid_loss = (use_bnneck_here and CE_WEIGHT > 0)
    # Anthropometric aux head is only meaningful for depth (RGB doesn't have
    # pose-invariant silhouette stats that are derivable from 3-channel colour).
    # The PKBatchSampler conditional inside compute_anthropometrics enforces this too.
    use_anthro = (use_hybrid_loss and ANTHRO_WEIGHT > 0 and not is_rgb)

    print(f"\n{'='*60}")
    if BACKBONE == "smallresnet":
        arch_desc = f"SmallResNet w={RESNET_BASE_WIDTH}"
    else:
        arch_desc = f"{BACKBONE} (ImageNet-pretrained)"
    print(f"STARTING TRAINING: {modality_name} ({arch_desc}, {IMG_H}x{IMG_W})  mode={TRAINING_MODE}")
    if TRAINING_MODE == "triplet":
        print(f"  PK sampler: P={PK_P}, K={PK_K}, batch={PK_P*PK_K}, steps/epoch={PK_STEPS_PER_EPOCH or 'auto'}")
        print(f"  Batch-hard triplet margin={TRIPLET_MARGIN}")
        print(f"  BNNeck head: {use_bnneck_here}")
        if use_hybrid_loss:
            print(f"  Hybrid loss: triplet + {CE_WEIGHT} * CE (label_smoothing={LABEL_SMOOTHING})")
    else:
        print(f"  Pair generator: batch={BATCH_SIZE}, HardNeg={use_hard_negatives} (every {HARDNEG_FREQ})")
        print(f"  Pair loss margin={margin}")
    print(f"Cache: {do_cache} | Augment: {augment_data} | Weight Decay: {weight_decay} | Epochs: {EPOCHS} | LR: {LR}")
    print(f"{'='*60}")

    # Build data generator FIRST so we know n_classes (needed for BNNeck classifier head).
    if TRAINING_MODE == "triplet":
        # PK batch sampler + batch-hard triplet (Hermans 2017)
        train_gen = PKBatchSampler(train_p, train_l, P=PK_P, K=PK_K, is_rgb=is_rgb,
                                   cache_images=do_cache, augment=augment_data,
                                   steps_per_epoch=PK_STEPS_PER_EPOCH,
                                   compute_anthropometrics=use_anthro)
    else:
        # Original pair-based contrastive/triplet (pre-existing code path)
        train_gen = SiamesePairGenerator(train_p, train_l, BATCH_SIZE, is_rgb, shuffle=True,
                                          cache_images=do_cache, augment=augment_data,
                                          hard_negatives=use_hard_negatives)

    # Now build the backbone/model.
    if TRAINING_MODE == "triplet":
        back = get_backbone(input_shape, EMB_DIM, weight_decay,
                            use_bnneck=use_bnneck_here, backbone=BACKBONE)
        siam = None
    else:
        siam, back = make_siamese_densenet(input_shape, weight_decay)

    # Optimizer.
    if USE_REDUCE_LR_ON_PLATEAU:
        optimizer = tf.keras.optimizers.Adam(LR)
    else:
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(LR, EPOCHS * len(train_gen), LR_MIN)
        optimizer = tf.keras.optimizers.Adam(lr_schedule)

    # Compile + pick fit target.
    if TRAINING_MODE == "triplet":
        if use_hybrid_loss:
            n_classes = len(train_gen.label_to_int)
            print(f"  Hybrid model classifier: {n_classes} training identities")
            effective_anthro_dim = (ANTHRO_DIM_3D if ANTHRO_3D else ANTHRO_DIM)
            anthro_dim_arg = effective_anthro_dim if use_anthro else 0
            anthro_w_arg = ANTHRO_WEIGHT if use_anthro else 0.0
            if use_anthro:
                anthro_kind = "3D-derived" if ANTHRO_3D else "2D-silhouette"
                print(f"  Anthropometric aux head: {effective_anthro_dim} targets ({anthro_kind}), "
                      f"weight {ANTHRO_WEIGHT}")
            triplet_model = HybridModel(back, n_classes=n_classes, margin=TRIPLET_MARGIN,
                                        ce_weight=CE_WEIGHT, weight_decay=weight_decay,
                                        label_smoothing=LABEL_SMOOTHING,
                                        anthro_weight=anthro_w_arg,
                                        anthro_dim=anthro_dim_arg)
        else:
            triplet_model = TripletModel(back, margin=TRIPLET_MARGIN)
        triplet_model.compile(optimizer=optimizer)
        fit_target = triplet_model
        hard_neg_cb = None  # no across-dataset hard-neg in triplet mode; batch-hard is intrinsic
    else:
        loss_function = triplet_loss_fn(margin) if USE_TRIPLET_LOSS else loss_fn(margin)
        siam.compile(optimizer=optimizer, loss=loss_function)
        fit_target = siam

        class HardNegativeCallback(tf.keras.callbacks.Callback):
            def __init__(self, generator, backbone, update_freq=5):
                super().__init__()
                self.generator = generator
                self.backbone = backbone
                self.update_freq = update_freq

            def on_epoch_end(self, epoch, logs=None):
                if (epoch + 1) % self.update_freq == 0:
                    self.generator.update_embeddings(self.backbone)

        hard_neg_cb = HardNegativeCallback(train_gen, back, update_freq=HARDNEG_FREQ) if use_hard_negatives else None

    results = {'best_rank1': 0.0, 'best_rank5': 0.0, 'best_mAP': 0.0, 'best_separation': 0.0, 'best_epoch': 0}
    eval_cb = EvaluationCallback(back, val_p, val_l, is_rgb, results)

    os.makedirs(OUT_DIR, exist_ok=True)
    ckpt_path = os.path.join(OUT_DIR, f"best_{RUN_TAG}_{modality_name.lower()}.weights.h5")
    ckpt = tf.keras.callbacks.ModelCheckpoint(
        ckpt_path,
        save_best_only=True, save_weights_only=True,
        monitor='val_rank1', mode='max'
    )
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_rank1',
        mode='max',
        patience=PATIENCE_ES,
        restore_best_weights=True,
        verbose=1
    )

    callbacks = [eval_cb, ckpt, early_stop]
    if hard_neg_cb:
        callbacks.append(hard_neg_cb)

    if USE_REDUCE_LR_ON_PLATEAU:
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_rank1',
            mode='max',
            factor=0.5,
            patience=PATIENCE_RLR,
            min_lr=LR_MIN,
            verbose=1
        )
        callbacks.append(reduce_lr)

    fit_target.fit(train_gen, epochs=EPOCHS, callbacks=callbacks, verbose=1)

    # ----- End-of-training evaluation: three protocols of increasing difficulty -----
    # 1. within-val intra-clustering (per-epoch, already in results['best_rank1'] etc.):
    #    each test image's nearest other test image. Conflates pose+session+identity.
    # 2. within-test gallery-probe (Still vs Walking, BELOW):
    #    same recording, same clothing, cross-pose. STANDARD BIWI/IAS-Lab Re-ID protocol.
    #    Both modalities should produce usable (>50%) R1 here.
    # 3. cross-clothing gallery-probe (train vs test, BELOW):
    #    different recordings, different clothing. The hardest, honest cross-clothing
    #    benchmark — naive siamese typically collapses near random here.
    print(f"\n[{modality_name}] Running within-test Still↔Walking gallery-probe ...")
    wt = evaluate_within_test_galprobe(
        backbone=back,
        val_imgs_uint8=eval_cb.val_imgs_uint8,
        val_paths=eval_cb.val_paths,
        val_labels=eval_cb.val_labels,
        is_rgb=is_rgb,
        probe_window=PROBE_WINDOW,
    )
    if wt is not None:
        print(f"[{modality_name}] within-test FRAME-MATCH (gallery={wt['n_gallery']}, probe={wt['n_probe']})  "
              f"R1={wt['rank1']:.2f}%  R5={wt['rank5']:.2f}%  mAP={wt['mAP']:.2f}%")
        print(f"[{modality_name}] within-test PROTOTYPE   (gallery_subjects={wt['n_gallery_subj']})              "
              f"R1={wt['proto_rank1']:.2f}%  R5={wt['proto_rank5']:.2f}%  mAP={wt['proto_mAP']:.2f}%")
        print(f"[{modality_name}] within-test SEQ-PROBE   (window={wt['seq_probe_window']}, n_seq={wt['n_seq_probe']})  "
              f"R1={wt['seq_rank1']:.2f}%  R5={wt['seq_rank5']:.2f}%  mAP={wt['seq_mAP']:.2f}%")
        if 'rerank_seq_rank1' in wt:
            print(f"[{modality_name}] within-test SEQ-PROBE+RERANK (k-reciprocal)                   "
                  f"R1={wt['rerank_seq_rank1']:.2f}%  R5={wt['rerank_seq_rank5']:.2f}%  mAP={wt['rerank_seq_mAP']:.2f}%")
        if 'rerank_rank1' in wt:
            print(f"[{modality_name}] within-test FRAME+RERANK                                       "
                  f"R1={wt['rerank_rank1']:.2f}%  R5={wt['rerank_rank5']:.2f}%  mAP={wt['rerank_mAP']:.2f}%")
        if 'rerank_proto_rank1' in wt:
            print(f"[{modality_name}] within-test PROTO+RERANK                                       "
                  f"R1={wt['rerank_proto_rank1']:.2f}%  R5={wt['rerank_proto_rank5']:.2f}%  mAP={wt['rerank_proto_mAP']:.2f}%")
        results['wt_rank1'] = wt['rank1']
        results['wt_rank5'] = wt['rank5']
        results['wt_mAP'] = wt['mAP']
        results['wt_proto_rank1'] = wt['proto_rank1']
        results['wt_proto_rank5'] = wt['proto_rank5']
        results['wt_proto_mAP'] = wt['proto_mAP']
        results['wt_seq_rank1'] = wt['seq_rank1']
        results['wt_seq_rank5'] = wt['seq_rank5']
        results['wt_seq_mAP'] = wt['seq_mAP']
        results['wt_seq_window'] = wt['seq_probe_window']
        results['wt_n_gallery'] = wt['n_gallery']
        results['wt_n_gallery_subj'] = wt['n_gallery_subj']
        results['wt_n_probe'] = wt['n_probe']
        results['wt_n_seq_probe'] = wt['n_seq_probe']
        # K-reciprocal re-rank results (Zhong 2017).
        for k in ('rerank_rank1', 'rerank_rank5', 'rerank_mAP',
                  'rerank_proto_rank1', 'rerank_proto_rank5', 'rerank_proto_mAP',
                  'rerank_seq_rank1', 'rerank_seq_rank5', 'rerank_seq_mAP'):
            if k in wt:
                # Stored under 'wt_<key>' to match collate convention.
                results[f'wt_{k}'] = wt[k]

    # Flush partial results to disk BEFORE the heavy cross-clothing GP eval.
    # GP at 384² RGB has been OOM-killing jobs after within-test completes —
    # without this flush, the paper numbers (within-test SEQ-PROBE + rerank)
    # are lost. With it, even if GP gets SIGKILL'd by the OOM-killer, the
    # results JSON on disk already contains the within-test block.
    if partial_writer is not None:
        try:
            partial_writer(results)
        except Exception as e:
            print(f"  (partial JSON flush failed, continuing: {type(e).__name__}: {e})")

    # ---- OPEN-SET CROSS-CLOTHING eval (the decisive experiment) ----
    # gallery = held-out test subjects in outfit A (Training/ frames), probe =
    # same subjects in outfit B (Testing/ frames == the val set). Subjects unseen
    # in training; clothing differs between gallery and probe. This is the regime
    # where depth's clothing-invariant body-shape cue should match/beat RGB.
    if xgallery_p is not None and len(xgallery_p) > 0:
        print(f"\n[{modality_name}] Running OPEN-SET CROSS-CLOTHING gallery-probe "
              f"(gallery=outfit A / Training, probe=outfit B / Testing) ...")
        gc.collect()
        xg_imgs, _xg_paths, xg_labels = _load_eval_images(
            xgallery_p, xgallery_l, is_rgb, tag=f"{modality_name} cross-clothing gallery (outfit A)")
        # Restrict probe to the subjects that have a cross-clothing gallery entry
        # (should be all test subjects, but guard in case some lack Training frames).
        xg_label_set = set(np.asarray(xg_labels).tolist())
        probe_mask = np.array([lbl in xg_label_set for lbl in eval_cb.val_labels])
        xc_probe_imgs = eval_cb.val_imgs_uint8[probe_mask]
        xc_probe_labels = np.asarray(eval_cb.val_labels)[probe_mask]
        xc = evaluate_cross_clothing_galprobe(
            backbone=back,
            gallery_imgs_uint8=xg_imgs,
            gallery_labels=xg_labels,
            probe_imgs_uint8=xc_probe_imgs,
            probe_labels=xc_probe_labels,
            is_rgb=is_rgb,
            probe_window=PROBE_WINDOW,
        )
        print(f"[{modality_name}] X-CLOTH FRAME-MATCH (gallery={xc['n_gallery']}, probe={xc['n_probe']}, "
              f"subj={xc['n_gallery_subj']})  R1={xc['rank1']:.2f}%  R5={xc['rank5']:.2f}%  mAP={xc['mAP']:.2f}%")
        print(f"[{modality_name}] X-CLOTH PROTOTYPE                                  "
              f"R1={xc['proto_rank1']:.2f}%  R5={xc['proto_rank5']:.2f}%  mAP={xc['proto_mAP']:.2f}%")
        print(f"[{modality_name}] X-CLOTH SEQ-PROBE  (window={xc['seq_probe_window']}, n_seq={xc['n_seq_probe']})  "
              f"R1={xc['seq_rank1']:.2f}%  R5={xc['seq_rank5']:.2f}%  mAP={xc['seq_mAP']:.2f}%")
        if 'rerank_seq_rank1' in xc:
            print(f"[{modality_name}] X-CLOTH SEQ-PROBE+RERANK (PAPER CROSS-CLOTHING NUMBER)   "
                  f"R1={xc['rerank_seq_rank1']:.2f}%  R5={xc['rerank_seq_rank5']:.2f}%  mAP={xc['rerank_seq_mAP']:.2f}%")
        # Store all xc_* keys.
        for k, v in xc.items():
            results[f'xc_{k}'] = v
        if partial_writer is not None:
            try:
                partial_writer(results)
            except Exception as e:
                print(f"  (partial JSON flush failed, continuing: {type(e).__name__}: {e})")
        del xg_imgs, xc_probe_imgs
        gc.collect()

    # Skip cross-clothing GP if train/test identities are disjoint (open-set protocol).
    # The eval would report R1=0% by construction (no positives possible), so it's both
    # uninformative and the most memory-hungry step. Saves ~5 min wall time and reduces
    # peak VRAM, helping avoid OOM on small MIG slices.
    train_labels_set = set(train_gen.labels.tolist())
    val_labels_set = set(eval_cb.val_labels.tolist())
    overlap = train_labels_set & val_labels_set
    if not overlap:
        print(f"\n[{modality_name}] Skipping cross-clothing GP eval — train/test identities are disjoint "
              f"(open-set protocol). R1 would be 0% by construction.")
    else:
        print(f"\n[{modality_name}] Running cross-clothing gallery-probe (train↔test) ...")
        gc.collect()
        gp = evaluate_gallery_probe(
            backbone=back,
            gallery_gen=train_gen,
            probe_imgs_uint8=eval_cb.val_imgs_uint8,
            probe_labels=eval_cb.val_labels,
            is_rgb=is_rgb,
        )
        print(f"[{modality_name}] cross-clothing FRAME-MATCH (gallery={gp['n_gallery']}, probe={gp['n_probe']})  "
              f"R1={gp['rank1']:.2f}%  R5={gp['rank5']:.2f}%  mAP={gp['mAP']:.2f}%  Sep={gp['sep']:.2f}")
        print(f"[{modality_name}] cross-clothing PROTOTYPE   (gallery_subjects={gp['n_gallery_subj']})              "
              f"R1={gp['proto_rank1']:.2f}%  R5={gp['proto_rank5']:.2f}%  mAP={gp['proto_mAP']:.2f}%")
        if 'rerank_proto_rank1' in gp:
            print(f"[{modality_name}] cross-clothing PROTO+RERANK                                       "
                  f"R1={gp['rerank_proto_rank1']:.2f}%  R5={gp['rerank_proto_rank5']:.2f}%  mAP={gp['rerank_proto_mAP']:.2f}%")
        results['gp_rank1'] = gp['rank1']
        results['gp_rank5'] = gp['rank5']
        results['gp_mAP'] = gp['mAP']
        results['gp_separation'] = gp['sep']
        results['gp_proto_rank1'] = gp['proto_rank1']
        results['gp_proto_rank5'] = gp['proto_rank5']
        results['gp_proto_mAP'] = gp['proto_mAP']
        results['gp_n_gallery'] = gp['n_gallery']
        results['gp_n_gallery_subj'] = gp['n_gallery_subj']
        results['gp_n_probe'] = gp['n_probe']
        for k in ('rerank_proto_rank1', 'rerank_proto_rank5', 'rerank_proto_mAP'):
            if k in gp:
                results[f'gp_{k}'] = gp[k]
        # Flush again now that GP results have been merged in.
        if partial_writer is not None:
            try:
                partial_writer(results)
            except Exception as e:
                print(f"  (partial JSON flush failed, continuing: {type(e).__name__}: {e})")

    # Release the generator's cache + the model graph before the next modality starts.
    # Prevents OOM when running both modalities in one process with large caches.
    del train_gen, eval_cb, siam, back, fit_target
    if hard_neg_cb is not None:
        del hard_neg_cb
    gc.collect()
    tf.keras.backend.clear_session()
    return results

def _build_argparser():
    p = argparse.ArgumentParser(
        description="Siamese RGB vs Depth body Re-ID on BIWI / IAS-Lab RGBD-ID datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Data / dataset selection
    p.add_argument("--base-dir", default=BASE_DIR,
                   help="Dataset root. Expects <base-dir>/Training (and optionally <base-dir>/Testing).")
    p.add_argument("--train-subdir", default="Training",
                   help="Training subdirectory name under --base-dir.")
    p.add_argument("--test-subdir", default=None,
                   help="Testing subdirectory name. If omitted, auto-detects Testing/ (BIWI) "
                        "or falls back to identity-disjoint random split of Training/.")
    p.add_argument("--protocol", choices=["auto", "disjoint", "full", "open-set"], default="auto",
                   help="Dataset protocol. 'auto'=detect Testing/. 'disjoint'=local diagnostic. "
                        "'full'=closed-set cross-clothing (Training train IDs, Testing test IDs, "
                        "identities overlap). 'open-set'=siamese cross-subject: train on Training/ \\ "
                        "Testing/ identities, eval on Testing/ identities (disjoint, never seen).")
    # Modality selection
    p.add_argument("--modality", choices=["both", "depth", "rgb"], default="both",
                   help="Which modality to train. 'both' runs depth then RGB sequentially.")
    # Training schedule
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--lr-min", type=float, default=LR_MIN)
    p.add_argument("--patience-es", type=int, default=PATIENCE_ES, help="EarlyStopping patience (epochs).")
    p.add_argument("--patience-rlr", type=int, default=PATIENCE_RLR, help="ReduceLROnPlateau patience.")
    # Model
    p.add_argument("--img-size", type=int, default=IMG_H, help="Square input size HxW.")
    p.add_argument("--width", type=int, default=RESNET_BASE_WIDTH,
                   help="SmallResNet base channel count (stages double from here).")
    p.add_argument("--emb-dim", type=int, default=EMB_DIM)
    p.add_argument("--dropout", type=float, default=DROPOUT_RATE)
    p.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY_DEPTH,
                   help="L2 weight decay applied to both modalities.")
    p.add_argument("--margin", type=float, default=MARGIN_DEPTH,
                   help="Triplet/contrastive margin applied to both modalities.")
    # Augmentation / mining
    p.add_argument("--augment", dest="augment", action="store_true", default=True,
                   help="Enable per-batch data augmentation (default).")
    p.add_argument("--no-augment", dest="augment", action="store_false",
                   help="Disable data augmentation.")
    p.add_argument("--hardneg-freq", type=int, default=0,
                   help="Hard-negative mining refresh frequency (epochs). 0 disables. (pair mode only)")
    # Modern siamese recipe — PK batch sampling + batch-hard triplet (Hermans 2017).
    p.add_argument("--training-mode", choices=["pair", "triplet"], default="pair",
                   help="'pair' = legacy SiamesePairGenerator + pairwise loss; "
                        "'triplet' = PK batch sampler + batch-hard triplet (modern Re-ID standard).")
    p.add_argument("--pk-p", type=int, default=8,
                   help="Identities per PK batch (triplet mode).")
    p.add_argument("--pk-k", type=int, default=4,
                   help="Images per identity per PK batch (triplet mode). batch_size = P * K.")
    p.add_argument("--pk-steps-per-epoch", type=int, default=None,
                   help="Override PK sampler's batches-per-epoch. Default: auto.")
    p.add_argument("--triplet-margin", type=float, default=0.3,
                   help="Margin for batch-hard triplet loss (cosine-distance space, range 0..2).")
    p.add_argument("--probe-window", type=int, default=10,
                   help="Multi-frame probe averaging window at eval (sequence-Re-ID). "
                        "Group N consecutive probe frames per subject, average their embeddings. "
                        "0 or 1 disables. Default 10.")
    # Hybrid loss recipe (Luo 2019). Only meaningful in --training-mode=triplet.
    p.add_argument("--use-bnneck", dest="use_bnneck", action="store_true", default=True,
                   help="Use BNNeck head (Luo 2019). Pre-BN ft feeds triplet, post-BN fi feeds "
                        "auxiliary classifier and retrieval. Triplet mode only.")
    p.add_argument("--no-bnneck", dest="use_bnneck", action="store_false",
                   help="Disable BNNeck head; fall back to v1 simple Dense+L2norm.")
    p.add_argument("--ce-weight", type=float, default=1.0,
                   help="Weight for cross-entropy loss on the BNNeck classifier head. "
                        "0 disables the classifier (triplet-only). Default 1.0 (Luo 2019).")
    p.add_argument("--label-smoothing", type=float, default=0.1,
                   help="Label-smoothing ε for the auxiliary CE loss. Default 0.1 (Luo 2019).")
    p.add_argument("--anthro-weight", type=float, default=0.0,
                   help="Weight for the anthropometric MSE auxiliary loss (depth only). "
                        "0 disables the head. Recommended 0.1-0.5 for the first tries. "
                        "Targets are pose-mostly-invariant body-shape statistics computed "
                        "from the canonical (un-augmented) preprocessed depth frame; the "
                        "model is forced to predict them from the augmented view, baking "
                        "pose-invariant geometry into the embedding.")
    p.add_argument("--canonicalize-pose", dest="canonicalize_pose", action="store_true",
                   default=False,
                   help="Depth-only: PCA-align the foreground silhouette so its principal "
                        "axis is vertical. Removes body-tilt as a pose variable before the "
                        "CNN sees the image. Augmentation rotation still applies on top, "
                        "providing variance from the canonical baseline.")
    p.add_argument("--anthro-3d", dest="anthro_3d", action="store_true", default=False,
                   help="Depth-only: replace 2D-silhouette anthropometric features with "
                        "3D-derived features (body height/width/depth in mm, volume, 3D aspect) "
                        "using Kinect intrinsics. Captures Z-thickness — the body-shape signal "
                        "that silhouette features ignore. Aux head dim 6 → 9.")
    p.add_argument("--canonicalize-pose-3d", dest="canonicalize_pose_3d",
                   action="store_true", default=False,
                   help="Depth-only: 3D pose canonicalization. Unproject foreground to 3D "
                        "via Kinect intrinsics, compute 3D PCA, rotate body so principal axis "
                        "is vertical, re-project to 2D depth. Captures forward-lean / "
                        "tilt-toward-camera that 2D PCA cannot see.")
    p.add_argument("--temporal-stack-frames", type=int, default=1,
                   help="Depth-only: number of consecutive depth frames to stack as channels "
                        "of the network input. 1 = single-frame (current default); 3 = stack "
                        "of 3 frames at temporal offsets [0, stride, 2*stride]. Captures gait. "
                        "Stays single-modality (every channel from the same depth sensor, "
                        "just at adjacent timestamps). With 3 frames, ConvNeXt's native "
                        "3-channel ImageNet weights are used directly without the 1ch adapter.")
    p.add_argument("--temporal-stack-stride", type=int, default=5,
                   help="Number of frames between consecutive stack elements (at 30 FPS, "
                        "stride=5 ≈ 167 ms — about 1/6 of a typical walking gait cycle).")
    p.add_argument("--kinect-fx", type=float, default=575.816,
                   help="Kinect depth-camera focal length (x), in pixels at 640×480.")
    p.add_argument("--kinect-fy", type=float, default=575.816,
                   help="Kinect depth-camera focal length (y), in pixels at 640×480.")
    p.add_argument("--kinect-cx", type=float, default=320.0,
                   help="Kinect depth-camera principal-point x.")
    p.add_argument("--kinect-cy", type=float, default=240.0,
                   help="Kinect depth-camera principal-point y.")
    p.add_argument("--backbone", default="smallresnet",
                   choices=["smallresnet",
                            "convnext_tiny", "convnext_small", "convnext_base",
                            "efficientnet_b0", "efficientnet_b2", "resnet50v2"],
                   help="Backbone architecture. 'smallresnet' is the v1 trained-from-scratch "
                        "network. All other options use tf.keras.applications with ImageNet "
                        "pretraining; for 1-channel depth the first Conv2D's weights are "
                        "channel-averaged from 3 → 1 to keep the architecture genuinely "
                        "single-channel for depth. ConvNeXt-Tiny=28M params, Small=50M, "
                        "Base=88M — Small/Base may need batch_size≤16 on 2g.20gb MIG slice.")
    p.add_argument("--no-pretrained", dest="pretrained", action="store_false", default=True,
                   help="For applications backbones (convnext_*, efficientnet_*, resnet50v2): "
                        "initialize RANDOMLY instead of from ImageNet. Isolates the architecture "
                        "from the pretraining (clean from-scratch vs transfer ablation). "
                        "No effect on smallresnet (always from scratch).")
    # Depth-specific preprocessing
    p.add_argument("--clip-range", type=float, default=CLIP_RANGE,
                   help="Depth foreground slab width in mm.")
    # Output
    p.add_argument("--out-dir", default=".",
                   help="Where to write weights checkpoints and results.json.")
    p.add_argument("--run-tag", default=None,
                   help="Tag included in checkpoint filenames and results.json. "
                        "Default: smallresnet_w<width>_s<size>_seed<seed>.")
    return p

if __name__ == "__main__":
    args = _build_argparser().parse_args()

    # Override module globals from CLI.
    SEED = args.seed
    IMG_H, IMG_W = args.img_size, args.img_size
    EMB_DIM = args.emb_dim
    DROPOUT_RATE = args.dropout
    BATCH_SIZE = args.batch_size
    EPOCHS = args.epochs
    LR = args.lr
    LR_MIN = args.lr_min
    BASE_DIR = args.base_dir
    CLIP_RANGE = args.clip_range
    RESNET_BASE_WIDTH = args.width
    MARGIN_DEPTH = MARGIN_RGB = args.margin
    WEIGHT_DECAY_DEPTH = WEIGHT_DECAY_RGB = args.weight_decay
    AUGMENT = args.augment
    HARDNEG_FREQ = max(0, args.hardneg_freq)
    PATIENCE_ES = args.patience_es
    PATIENCE_RLR = args.patience_rlr
    OUT_DIR = args.out_dir
    # Triplet-mode globals.
    TRAINING_MODE = args.training_mode
    PK_P = args.pk_p
    PK_K = args.pk_k
    PK_STEPS_PER_EPOCH = args.pk_steps_per_epoch
    TRIPLET_MARGIN = args.triplet_margin
    PROBE_WINDOW = args.probe_window
    # Hybrid-loss globals.
    USE_BNNECK = args.use_bnneck
    CE_WEIGHT = args.ce_weight
    LABEL_SMOOTHING = args.label_smoothing
    ANTHRO_WEIGHT = args.anthro_weight
    CANONICALIZE_POSE = args.canonicalize_pose
    ANTHRO_3D = args.anthro_3d
    CANONICALIZE_POSE_3D = args.canonicalize_pose_3d
    TEMPORAL_STACK_FRAMES = max(1, args.temporal_stack_frames)
    TEMPORAL_STACK_STRIDE = max(1, args.temporal_stack_stride)
    KINECT_FX = args.kinect_fx
    KINECT_FY = args.kinect_fy
    KINECT_CX = args.kinect_cx
    KINECT_CY = args.kinect_cy
    # Backbone selector.
    BACKBONE = args.backbone
    USE_PRETRAINED = args.pretrained
    default_tag = f"{TRAINING_MODE}_smallresnet_w{args.width}_s{args.img_size}_seed{args.seed}"
    if args.protocol == "open-set":
        default_tag = "openset_" + default_tag
    RUN_TAG = args.run_tag or default_tag

    _set_seeds(SEED)

    # If --base-dir points at a packed .zip, extract it once to local node scratch.
    # This is the standard pattern for clusters with file-count quotas: the .zip
    # lives in persistent storage as a single file, the extracted form lives in
    # $SLURM_TMPDIR (node-local, no quota) only for the duration of the job.
    if BASE_DIR.endswith(".zip") and os.path.isfile(BASE_DIR):
        BASE_DIR = stage_zip_to_local(BASE_DIR)

    # Print effective config for log self-documentation.
    print("=" * 60)
    print(f"RUN CONFIG: {RUN_TAG}")
    print("=" * 60)
    for k in ["base_dir", "train_subdir", "test_subdir", "protocol", "modality",
              "seed", "epochs", "batch_size", "lr", "lr_min", "patience_es", "patience_rlr",
              "img_size", "width", "emb_dim", "dropout", "weight_decay", "margin",
              "augment", "hardneg_freq", "clip_range", "out_dir", "run_tag",
              "training_mode", "pk_p", "pk_k", "pk_steps_per_epoch", "triplet_margin",
              "probe_window", "use_bnneck", "ce_weight", "label_smoothing", "backbone",
              "pretrained", "anthro_weight", "canonicalize_pose",
              "anthro_3d", "canonicalize_pose_3d",
              "temporal_stack_frames", "temporal_stack_stride",
              "kinect_fx", "kinect_fy", "kinect_cx", "kinect_cy"]:
        v = getattr(args, k, None) if hasattr(args, k) else None
        if k == "run_tag": v = RUN_TAG
        print(f"  {k:>16}: {v}")
    print("=" * 60)

    # Dataset loading. xgallery (cross-clothing gallery, outfit A) is only
    # populated by the open-set loader; empty otherwise.
    rgb_xg, d_xg = ([], []), ([], [])
    if args.protocol == "disjoint":
        (rgb_tr_p, rgb_tr_l), (rgb_va_p, rgb_va_l), (d_tr_p, d_tr_l), (d_va_p, d_va_l) = \
            load_biwi_disjoint(BASE_DIR)
    elif args.protocol == "full":
        test_subdir = args.test_subdir or "Testing"
        (rgb_tr_p, rgb_tr_l), (rgb_va_p, rgb_va_l), (d_tr_p, d_tr_l), (d_va_p, d_va_l) = \
            load_biwi_full_protocol(BASE_DIR, train_subdir=args.train_subdir, test_subdir=test_subdir)
    elif args.protocol == "open-set":
        test_subdir = args.test_subdir or "Testing"
        (rgb_tr_p, rgb_tr_l), (rgb_va_p, rgb_va_l), (d_tr_p, d_tr_l), (d_va_p, d_va_l), \
            rgb_xg, d_xg = \
            load_biwi_open_set(BASE_DIR, train_subdir=args.train_subdir, test_subdir=test_subdir)
    else:  # auto
        if args.test_subdir is not None:
            (rgb_tr_p, rgb_tr_l), (rgb_va_p, rgb_va_l), (d_tr_p, d_tr_l), (d_va_p, d_va_l) = \
                load_biwi_full_protocol(BASE_DIR, train_subdir=args.train_subdir, test_subdir=args.test_subdir)
        else:
            (rgb_tr_p, rgb_tr_l), (rgb_va_p, rgb_va_l), (d_tr_p, d_tr_l), (d_va_p, d_va_l) = \
                load_dataset_auto(BASE_DIR)

    # Training.
    t0 = time.time()

    def _json_default(o):
        # numpy scalars (e.g. np.float32 from tf/sklearn) → Python native types.
        if hasattr(o, "item"):
            return o.item()
        if hasattr(o, "tolist"):
            return o.tolist()
        raise TypeError(f"Object of type {type(o).__name__} is not JSON serializable")

    json_path = os.path.join(OUT_DIR, f"results_{RUN_TAG}.json")
    os.makedirs(OUT_DIR, exist_ok=True)

    # Mutable container so nested partial-writer callbacks can update results
    # without needing 'global'/'nonlocal' (we're at module scope here, so
    # 'nonlocal' would be a SyntaxError).
    _state = {"depth": None, "rgb": None, "partial": True}

    def _flush_partial():
        """Write the running results JSON. Called incrementally so OOM kills
        after within-test eval (e.g., during cross-clothing GP at 384²) still
        leave the paper numbers on disk."""
        payload = {
            "run_tag": RUN_TAG,
            "config": vars(args),
            "depth": _state["depth"],
            "rgb":   _state["rgb"],
            "wall_minutes": (time.time() - t0) / 60.0,
            "partial": _state["partial"],
        }
        try:
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2, default=_json_default)
        except Exception as e:
            print(f"  (incremental JSON write failed: {type(e).__name__}: {e})")

    def _depth_partial(res):
        _state["depth"] = dict(res)
        _flush_partial()

    def _rgb_partial(res):
        _state["rgb"] = dict(res)
        _flush_partial()

    if args.modality in ("both", "depth"):
        _state["depth"] = run_training_session("DEPTH", d_tr_p, d_tr_l, d_va_p, d_va_l,
                                               partial_writer=_depth_partial,
                                               xgallery_p=d_xg[0], xgallery_l=d_xg[1])
        _flush_partial()
    if args.modality in ("both", "rgb"):
        _state["rgb"] = run_training_session("RGB", rgb_tr_p, rgb_tr_l, rgb_va_p, rgb_va_l,
                                             partial_writer=_rgb_partial,
                                             xgallery_p=rgb_xg[0], xgallery_l=rgb_xg[1])
        _flush_partial()
    depth_res = _state["depth"]
    rgb_res = _state["rgb"]
    wall_s = time.time() - t0

    # Summary tables.
    print(f"\n{'='*95}")
    print(f"FINAL RESULTS  run_tag={RUN_TAG}  wall={wall_s/60:.1f} min")
    print(f"{'='*95}")
    print("\n[within-val monitoring metric — not the paper number]")
    print(f"{'MODALITY':<10} | {'RANK-1':<10} | {'RANK-5':<10} | {'mAP':<10} | {'SEP':<8} | {'EPOCH':<6}")
    print("-" * 70)
    if depth_res is not None:
        print(f"{'DEPTH':<10} | {depth_res['best_rank1']:.2f}%     | {depth_res['best_rank5']:.2f}%     "
              f"| {depth_res['best_mAP']:.2f}%     | {depth_res['best_separation']:.2f}     | {depth_res['best_epoch']}")
    if rgb_res is not None:
        print(f"{'RGB':<10} | {rgb_res['best_rank1']:.2f}%     | {rgb_res['best_rank5']:.2f}%     "
              f"| {rgb_res['best_mAP']:.2f}%     | {rgb_res['best_separation']:.2f}     | {rgb_res['best_epoch']}")
    print("\n[within-test Still↔Walking gallery-probe — standard BIWI protocol]")
    print(f"{'MODALITY':<10} | {'WT-R1':<10} | {'WT-R5':<10} | {'WT-mAP':<10}")
    print("-" * 70)
    if depth_res is not None and 'wt_rank1' in depth_res:
        print(f"{'DEPTH':<10} | {depth_res['wt_rank1']:.2f}%     | {depth_res['wt_rank5']:.2f}%     "
              f"| {depth_res['wt_mAP']:.2f}%")
    if rgb_res is not None and 'wt_rank1' in rgb_res:
        print(f"{'RGB':<10} | {rgb_res['wt_rank1']:.2f}%     | {rgb_res['wt_rank5']:.2f}%     "
              f"| {rgb_res['wt_mAP']:.2f}%")

    print("\n[cross-clothing train↔test gallery-probe — hardest, honest]")
    print(f"{'MODALITY':<10} | {'GP-R1':<10} | {'GP-R5':<10} | {'GP-mAP':<10} | {'GP-SEP':<8}")
    print("-" * 70)
    if depth_res is not None and 'gp_rank1' in depth_res:
        print(f"{'DEPTH':<10} | {depth_res['gp_rank1']:.2f}%     | {depth_res['gp_rank5']:.2f}%     "
              f"| {depth_res['gp_mAP']:.2f}%     | {depth_res['gp_separation']:.2f}")
    if rgb_res is not None and 'gp_rank1' in rgb_res:
        print(f"{'RGB':<10} | {rgb_res['gp_rank1']:.2f}%     | {rgb_res['gp_rank5']:.2f}%     "
              f"| {rgb_res['gp_mAP']:.2f}%     | {rgb_res['gp_separation']:.2f}")
    print("-" * 95)

    # Mark the JSON as complete (incremental writes happened throughout training;
    # this is the final flush after both modalities + their evals all completed).
    _state["partial"] = False
    _flush_partial()
    print(f"\nResults written to {json_path}")
