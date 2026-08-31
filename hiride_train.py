"""HI-RIDE paper-2 re-run: the patched trainer.

Consumes the shards written by hiride_prep.py and trains one (policy, modality,
architecture, mask-condition, seed) cell. Every 2023 defect the audit found is
fixed here, and each fix is annotated so the paper can state it.

    python hiride_train.py --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/runs \
        --policy R4_cross_session --modality depth --arch alexnet --seed 0

Design decisions that are FIXED CONTROLS, never swept (paper 3 owns the
initialisation and dynamic-range questions):
  * one normalisation policy for both modalities, declared below;
  * identical optimiser settings across architectures;
  * no augmentation -- the 2023 runs had none, and adding it to one modality
    would break the "same pipeline" claim this paper defends.
"""
import os
import sys
import json
import time
import argparse

import numpy as np

from hiride_keys import AXES
import tensorflow as tf
from sklearn.metrics import f1_score, confusion_matrix

from hiride_data import (load_manifest, make_split, block_train_counts,
                         describe_split, eligible_mask)

# Global fixed depth scale, in millimetres. This is the honest counterpart to
# RGB's /255: a CONSTANT, so it preserves absolute body scale across frames.
# The 2023 "depth8" path used a PER-FRAME min-max stretch, which is
# room-relative and destroys exactly the cue depth is supposed to carry.
# Measured on BIWI: person median depth 1819-3975 mm, background median 3842 mm
# (max 4068); stray far readings reach 16524 mm and are clipped away here.
DEPTH_CLIP_MM = 6000.0
MASK_CONDITIONS = ("full", "person", "bg_hole", "silhouette", "scale_removed",
                   "sil_scaled", "person_centred", "bg_plate", "interior_only")

# Kinect v1 depth intrinsics at 640x480 (Munaro 2014); scaled to the shard size
# where needed. Used only by the surface-normal encoding.
FX_640 = FY_640 = 575.816
# Measured sensor floor, for reference when reading the normal-baseline sweep:
# depth quantisation is z^2/(f*b*8) ~ 25 mm at the person's 2955 mm median, and
# the frame-to-frame |delta depth| of 80.6 mm (adjacency_results.json) implies a
# per-pixel sigma of roughly 40-70 mm. A 1-px lateral step at 3 m spans only
# ~5 mm, so a finite-difference normal is pure noise unless the baseline is
# widened -- hence --normal-baseline, and hence --frames to denoise first.
NORMAL_BASELINE_PX = 4
PLATE_CONDITIONS = ("bg_plate",)          # need hiride_plates.py to have been run

# scale_removed: the person is cut out, rescaled so the bounding box is a fixed
# fraction of the frame height, pasted centred on a constant background, and
# (depth only) shifted so the person's median depth is a fixed distance. What
# is left is body SHAPE and internal depth structure -- apparent size, image
# position and standing distance, i.e. the cues that dominate the trivial-cue
# floor, are gone. sil_scaled is the same geometry on the binary silhouette.
SCALE_TARGET_H = 0.70            # bbox height as a fraction of the frame
SCALE_TARGET_DEPTH = 3000.0 / DEPTH_CLIP_MM   # person median -> 3.0 m, in [0,1] units


def spatial_head(x, n_classes, head):
    """Pool the final feature map in a way that respects ALIGNMENT.

    `scale_removed` / `interior_only` normalise the person to a fixed size and
    position, so pixel (i, j) denotes roughly the same body location in every
    frame. GlobalAveragePooling then averages that away -- "chest depth at the
    chest" becomes "mean depth over the body" -- and the preprocessing fights
    the architecture. The evidence this matters: on interior_only at R1 a linear
    nearest-class-mean probe over 96 PCA components of the ALIGNED pixels scores
    60.5 % while the GAP CNN scores 40.9 %, trained on MORE frames. A template
    matcher beats a 58M-parameter network, which only makes sense if the network
    is discarding the correspondence.

      gap      the existing head; translation-invariant, alignment-agnostic.
      stripe   average over width only, keeping H rows as separate features
               (PCB-style). Vertical body structure -- head, shoulders, torso,
               legs -- is preserved while horizontal jitter is still pooled out.
      flatten  keep the full spatial map. Most faithful to the template view,
               most parameters.
    """
    L = tf.keras.layers
    if head == "gap":
        return L.GlobalAveragePooling2D()(x)
    if head == "stripe":
        return L.Flatten()(L.Lambda(lambda t: tf.reduce_mean(t, axis=2),
                                    name="stripe_pool")(x))
    if head == "flatten":
        return L.Flatten()(L.AveragePooling2D(pool_size=2)(x))
    raise ValueError(head)


def attach_aux(L, inp, x, aux_dim):
    """Concatenate an auxiliary scalar vector to the pooled image features.

    WHY THIS EXISTS. What survives a session change is METRIC: millimetres of
    stature, width, thickness. Recovering millimetres from a depth image means
    multiplying pixel extent by the depth value and dividing by the focal
    length -- a multiplicative interaction between spatial extent and pixel
    intensity. Convolution and pooling are engineered to be INVARIANT to
    exactly that, and the network is never given the focal length or the
    subject's standing distance. Hand-computed metric features reach 28.86 % at
    R4 where this CNN reaches 18.40 % on identical frames, using arithmetic and
    no training data at all.

    So hand the network the term it cannot compute and see whether it uses it.
    The prediction that makes this a test rather than a tweak: distance should
    HELP the conditions that preserve apparent size (`person`,
    `person_centred`) and do almost NOTHING for `sil_scaled`, which normalises
    size away by construction and therefore has no metric content left for
    distance to unlock. If it lifts everything equally, the story is wrong and
    distance is acting as a recording-identity shortcut instead.
    """
    if not aux_dim:
        return inp, x
    aux = L.Input(shape=(aux_dim,), name="aux", dtype="float32")
    a = L.Dense(32, activation="relu")(aux)
    a = L.Dropout(0.2)(a)
    return [inp, aux], L.Concatenate()([x, a])


def build_alexnet(input_shape, n_classes, head="gap", aux_dim=0):
    """The 2023 architecture with its two structural defects repaired.

    2023 did Conv(activation='relu') -> BatchNorm -> Activation('relu'), i.e.
    conv, ReLU, BN, then a SECOND ReLU that clips everything BN shifted below
    zero. That is neither standard Conv-BN-ReLU nor original AlexNet. Fixed to
    Conv -> BN -> ReLU. Also: the classifier is forced to float32 so the softmax
    is not computed in float16 under a mixed-precision policy.
    """
    L = tf.keras.layers

    def conv_block(x, filters, kernel, strides=1, pool=False):
        x = L.Conv2D(filters, kernel, strides=strides, padding="same",
                     use_bias=False)(x)
        x = L.BatchNormalization()(x)
        x = L.Activation("relu")(x)
        if pool:
            x = L.MaxPooling2D(pool_size=3, strides=2)(x)
        return x

    inp = L.Input(shape=input_shape)
    x = conv_block(inp, 96, 11, strides=4, pool=True)
    x = conv_block(x, 256, 5, pool=True)
    x = conv_block(x, 384, 3)
    x = conv_block(x, 384, 3)
    x = conv_block(x, 256, 3, pool=True)
    # GAP, not Flatten->Dense(4096). The 2023 head held ~65% of all parameters
    # (37.7M in one layer), which made the "1-channel saves model size" claim
    # incoherent -- the channel change is 23,232 of 58,524,466 params (0.0397%).
    x = spatial_head(x, n_classes, head)
    x = L.Dropout(0.5)(x)
    inputs, x = attach_aux(L, inp, x, aux_dim)
    out = L.Dense(n_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, out, name=f"alexnet_{head}")


def build_convnext(input_shape, n_classes, pretrained=True, head="gap", aux_dim=0):
    """ConvNeXt-Tiny with an honest 1-channel stem when the input is depth.

    ImageNet weights are a FIXED CONTROL here, applied to both modalities.
    A 1-channel stem is obtained by channel-averaging the pretrained stem
    kernel, which preserves the response scale.
    """
    ch = input_shape[-1]
    # include_preprocessing=False everywhere. Keras' ConvNeXt prepends an
    # ImageNet mean/std `PreStem` to 3-channel inputs that expects 0-255 pixels;
    # this trainer feeds [-1, 1], which that layer would squash to a near-constant
    # (x - 124) / 58 ~ -2.1. The equivalent normalisation for our [-1, 1] input
    # (p in [0,1] -> x = 2p - 1, so (p - m)/s = (x - (2m - 1)) / (2s)) is applied
    # explicitly below for RGB; depth (1 channel) gets no ImageNet statistics.
    base3 = tf.keras.applications.ConvNeXtTiny(
        include_top=False, include_preprocessing=False,
        weights="imagenet" if pretrained else None,
        input_shape=input_shape[:2] + (3,))
    if ch == 3:
        base = base3
    else:
        base = tf.keras.applications.ConvNeXtTiny(
            include_top=False, include_preprocessing=False, weights=None,
            input_shape=input_shape)
        copied = averaged = 0
        if pretrained:
            # Pair layers POSITIONALLY, not by name. Keras 2.15's ConvNeXt names
            # every layer with the model_name prefix EXCEPT the include_top=False
            # tail `LayerNormalization(epsilon=1e-6)`, which is auto-named from a
            # process-global counter: `layer_normalization` in the first model
            # built, `layer_normalization_1` in the second. A by-name lookup
            # silently skipped it and left the depth model's final LN at default
            # init. With include_preprocessing=False on both models the layer
            # lists line up 1:1, which is asserted below.
            assert len(base3.layers) == len(base.layers), (len(base3.layers), len(base.layers))
            for src, dst in zip(base3.layers, base.layers):
                assert type(src) is type(dst), (src.name, dst.name)
                sw, dw = src.get_weights(), dst.get_weights()
                if not sw or not dw:
                    continue
                if sw[0].shape != dw[0].shape and sw[0].ndim == 4:
                    w = [sw[0].mean(axis=2, keepdims=True)] + sw[1:]
                    dst.set_weights(w)
                    averaged += 1
                else:
                    assert all(a.shape == b.shape for a, b in zip(sw, dw)), (src.name, dst.name)
                    dst.set_weights(sw)
                    copied += 1
            n_weighted = sum(1 for l in base.layers if l.get_weights())
            assert averaged == 1, f"expected exactly one stem kernel to average, got {averaged}"
            assert copied + averaged == n_weighted, (copied, averaged, n_weighted)
        print(f"[convnext] init={'imagenet' if pretrained else 'scratch'} "
              f"copied {copied} layers, channel-averaged {averaged} stem kernel(s)")
    inp = tf.keras.layers.Input(shape=input_shape)
    x = inp
    if ch == 3:
        m = np.array([0.485, 0.456, 0.406]); s = np.array([0.229, 0.224, 0.225])
        x = tf.keras.layers.Normalization(mean=(2 * m - 1), variance=(2 * s) ** 2,
                                          name="imagenet_norm_from_pm1")(x)
    x = spatial_head(base(x), n_classes, head)
    x = tf.keras.layers.Dropout(0.3)(x)
    inputs, x = attach_aux(tf.keras.layers, inp, x, aux_dim)
    out = tf.keras.layers.Dense(n_classes, activation="softmax", dtype="float32")(x)
    return tf.keras.Model(inputs, out, name="convnext_tiny")


# Every axis that makes two runs a DIFFERENT cell. `tag` must encode all of
# them or runs overwrite each other; the pre-write check below compares these
# fields against the file already at the target path so a missing axis fails
# loudly instead of silently costing seeds.
CELL_FIELDS = ("policy", "modality", "arch", "init", "condition", "seed", "guard",
               "permuted", "bits", "depth_slab_mm", "frames", "encoding", "erode",
               "head", "eligibility", "ref_eligibility", "augment", "test_fuse",
               "aux", "cohort", "cohort_seed", "mask_source")


def _axis(rec, field):
    """One cell-identity field, with an absent value resolved to its default."""
    v = rec.get(field)
    return AXES[field] if v is None and field in AXES else v


def run_tag(m):
    """Canonical filename stem for one training cell, keyed on the result dict.

    ONE definition, used both when writing a run and when auditing the runs
    directory, so the two can never disagree about what distinguishes a cell.
    Every field named in CELL_FIELDS must appear here; anything omitted means
    two different cells share a filename and the second silently destroys the
    first. That is not hypothetical -- `erode` was missing, so wave 12's e1 and
    e4 runs were overwritten by e6, taking seeds 0-2 of plain interior_only
    with them, and the array job reported COMPLETED for all of it.
    """
    g = m.get("guard")
    return (f"{m['policy']}_{m['modality']}_{m['arch']}_{m['condition']}"
            f"_s{m['seed']}" + ("_perm" if m.get("permuted") else "")
            + ("_scratch" if (m["arch"] == "convnext_tiny"
                              and m.get("init") == "scratch") else "")
            + (f"_b{m['bits']}" if m.get("bits", 16) < 16 else "")
            + (f"_slab{int(m['depth_slab_mm'])}"
               if m.get("depth_slab_mm", DEPTH_CLIP_MM) < DEPTH_CLIP_MM else "")
            + (f"_f{m['frames']}" if m.get("frames", 1) > 1 else "")
            + ("_nrm" if m.get("encoding") == "normals" else "")
            + (f"_{m['head']}" if m.get("head", "gap") != "gap" else "")
            + (f"_aug{m['augment']}" if m.get("augment") else "")
            + (f"_tf{m['test_fuse']}" if m.get("test_fuse", 1) > 1 else "")
            + (f"_{m['eligibility']}" if m.get("eligibility", "cues") != "cues" else "")
            + (f"_e{m['erode']}" if m.get("erode", 2) != 2 else "")
            + (f"_g{g}" if m["policy"].startswith("R1") and g not in (None, 150) else "")
            + (f"_ref{m['ref_eligibility']}"
               if m.get("ref_eligibility", "match") != "match" else "")
            + (f"_aux{m['aux']}" if m.get("aux", "none") != "none" else "")
            + (f"_k{m['cohort']}d{m['cohort_seed']}" if m.get("cohort") else "")
            + (f"_m{m['mask_source']}" if m.get("mask_source", "user") != "user" else ""))


def scale_remove(img, mask, fill, is_depth, target_h=SCALE_TARGET_H,
                 target_depth=SCALE_TARGET_DEPTH, slab_mm=DEPTH_CLIP_MM):
    """Person-only frame with apparent size, image position and (depth) standing
    distance normalised away. `img` is (S,S,C) in [0,1]; `mask` is the userMap.

    Steps: crop the mask's bounding box; zero everything outside the mask;
    for depth, shift valid person pixels so their median is `target_depth`;
    resize the crop so its height is `target_h` of the frame (aspect kept,
    width capped at the frame); paste centred on a `fill` canvas.
    """
    from scipy.ndimage import zoom              # scipy ships with sklearn on Nibi
    S = img.shape[0]
    m = mask > 0
    if not m.any():
        return np.full_like(img, fill)
    rows, cols = np.where(m)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    crop = np.where(m[r0:r1, c0:c1, None], img[r0:r1, c0:c1], fill).astype(np.float32)
    mc = m[r0:r1, c0:c1]
    if is_depth:
        valid = mc & (crop[..., 0] > 0)         # 0 = invalid depth, never shift it
        if valid.any():
            med = np.median(crop[..., 0][valid])
            v = crop[..., 0][valid] + (target_depth - med)
            # BODY-RELATIVE RANGE. The global 6000 mm scale is honest about
            # absolute distance but spends only ~7 % of the input range on the
            # body: the person's own depth extent (p99-p1) has a median of
            # 419 mm. Everything that encodes body SHAPE therefore arrives as a
            # near-flat plate. Rescaling a fixed slab_mm window, centred on the
            # person's median, onto the full range restores that contrast
            # without touching relative thickness (a fixed width keeps body
            # depth comparable between people -- normalising by each person's
            # OWN extent would delete a biometric). slab_mm = DEPTH_CLIP_MM is
            # the identity transform, so the default reproduces earlier runs.
            if slab_mm < DEPTH_CLIP_MM:
                v = target_depth + (v - target_depth) * (DEPTH_CLIP_MM / float(slab_mm))
            crop[..., 0][valid] = np.clip(v, 1e-3, 1.0)
    h, w = crop.shape[:2]
    scale = (target_h * S) / h
    if w * scale > S:                            # very wide (arms out / near camera)
        scale = S / w
    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
    zc = zoom(crop, (nh / h, nw / w, 1), order=1, mode="nearest")
    zm = zoom(mc.astype(np.float32), (nh / h, nw / w), order=0, mode="nearest") > 0.5
    zc = np.where(zm[..., None], zc, fill).astype(np.float32)   # keep the outline crisp
    out = np.full((S, S, img.shape[2]), fill, dtype=np.float32)
    top, left = (S - zc.shape[0]) // 2, (S - zc.shape[1]) // 2
    out[top:top + zc.shape[0], left:left + zc.shape[1]] = zc[:S, :S]
    return out.astype(img.dtype)


def centre_person(img, mask, fill):
    """Person-only, translated so the mask bbox centre sits at the frame centre.

    Pure integer translation: no interpolation, no rescale, so it separates the
    POSITION cue from the SIZE cue that `scale_removed` also normalises away.
    Anything shifted past the border is cropped, as in a real re-framing.
    """
    S = img.shape[0]
    m = mask > 0
    if not m.any():
        return np.full_like(img, fill)
    rows, cols = np.where(m)
    dy = S // 2 - (rows.min() + rows.max() + 1) // 2
    dx = S // 2 - (cols.min() + cols.max() + 1) // 2
    src = np.where(m[..., None], img, fill).astype(img.dtype)
    out = np.full_like(src, fill)
    ys, xs = slice(max(0, dy), min(S, S + dy)), slice(max(0, dx), min(S, S + dx))
    sy, sx = slice(max(0, -dy), min(S, S - dy)), slice(max(0, -dx), min(S, S - dx))
    out[ys, xs] = src[sy, sx]
    return out


def quantise_depth(img, bits):
    """Uniform quantisation to 2**bits levels at FIXED global range.

    `img` is already clipped to DEPTH_CLIP_MM and divided by it, so the levels
    are absolute distances, not per-frame. bits=1 gives a thresholded depth map
    (near/far) -- report it as that, NOT as the shipped-userMap silhouette,
    which this study measures separately as the `silhouette` condition.

    Values are BIN CENTRES in (0, 1), never exactly 0, because 0 is the
    invalid-depth sentinel everywhere else in this pipeline. A first version
    rounded to `floor(x*(L-1)+0.5)/(L-1)`, which sent every pixel nearer than
    half the range to exactly 0: at 1 bit that erased the whole person into the
    fill value instead of binarising them, and the axis read 4.1 % / 4.8 % --
    the null -- for reasons that had nothing to do with depth precision.
    """
    levels = float(2 ** bits)
    q = (np.minimum(np.floor(img * levels), levels - 1) + 0.5) / levels
    return np.where(img > 0, q, 0.0).astype(img.dtype)


def erode_mask(mask, k):
    """Shrink the person mask by k pixels (4-connected), no scipy dependency.

    Two jobs at once. (1) It creates the INTERIOR-ONLY condition, the cell the
    mechanism suite never had: every condition so far confounded outline with
    interior, so "the interior adds ~1 pp" was a MARGINAL measurement given the
    outline, not evidence the interior is empty. (2) It removes the rim, where
    hiride_prep's AREA resize averaged person depth (~2955 mm) against
    background (~3842 mm) before the NEAREST-resized mask was applied -- an
    error of up to ~440 mm on roughly 9 % of person pixels, concentrated exactly
    where shape information is richest.
    """
    m = mask > 0
    for _ in range(int(k)):
        e = m.copy()
        e[1:, :] &= m[:-1, :]; e[:-1, :] &= m[1:, :]
        e[:, 1:] &= m[:, :-1]; e[:, :-1] &= m[:, 1:]
        m = e
    return m


def depth_to_normals(depth01, mask, baseline_px=NORMAL_BASELINE_PX):
    """Unit surface normals from a normalised depth map, as 3 channels in [-1,1].

    Derivatives of depth are scale-free, so they sidestep the contrast problem
    entirely -- but they AMPLIFY noise, which is why the baseline is a parameter
    and why this is worth pairing with --frames. Points are unprojected with the
    Kinect v1 intrinsics (scaled to the shard size) so the normals are true
    surface orientations rather than image gradients.
    """
    S = depth01.shape[0]
    z = depth01[..., 0] * DEPTH_CLIP_MM                      # back to mm
    f = FX_640 * S / 640.0
    u = (np.arange(S) - S / 2.0)[None, :]
    v = (np.arange(S) - S / 2.0)[:, None]
    X, Y, Z = u * z / f, v * z / f, z
    d = int(max(1, baseline_px))
    def dif(A, axis):
        out = np.zeros_like(A)
        if axis == 1:
            out[:, d:-d] = A[:, 2 * d:] - A[:, :-2 * d]
        else:
            out[d:-d, :] = A[2 * d:, :] - A[:-2 * d, :]
        return out
    ax = np.stack([dif(X, 1), dif(Y, 1), dif(Z, 1)], -1)
    ay = np.stack([dif(X, 0), dif(Y, 0), dif(Z, 0)], -1)
    n = np.cross(ax, ay)
    ln = np.linalg.norm(n, axis=-1, keepdims=True)
    n = np.where(ln > 1e-6, n / np.maximum(ln, 1e-6), 0.0)
    if n[..., 2].sum() < 0:                                  # face the camera
        n = -n
    valid = (mask > 0) & (z > 0)
    return (n * valid[..., None]).astype(np.float32)          # already in [-1,1]


def apply_mask_condition(img, mask, condition, fill, slab_mm=DEPTH_CLIP_MM,
                         erode=2):
    """Single controlled edit to the input. `mask` is the shipped userMap."""
    if condition == "full":
        return img
    m = mask[..., None] > 0
    if condition == "person":
        return np.where(m, img, fill).astype(img.dtype)
    if condition == "bg_hole":
        return np.where(m, fill, img).astype(img.dtype)
    if condition == "silhouette":
        return np.where(m, np.asarray(1.0, img.dtype),
                        np.asarray(0.0, img.dtype)).astype(img.dtype)
    if condition == "scale_removed":
        return scale_remove(img, mask, fill, is_depth=(img.shape[-1] == 1),
                            slab_mm=slab_mm)
    if condition == "interior_only":
        # depth INSIDE an eroded mask, then the same size/position/distance
        # normalisation as scale_removed, so it is comparable to sil_scaled
        # (outline, no interior) and scale_removed (outline + interior).
        e = erode_mask(mask, erode)
        if not e.any():
            return np.full_like(img, fill)
        return scale_remove(np.where(e[..., None], img, fill).astype(img.dtype),
                            e.astype(np.uint8), fill,
                            is_depth=(img.shape[-1] == 1), slab_mm=slab_mm)
    if condition == "person_centred":
        return centre_person(img, mask, fill)
    if condition == "sil_scaled":
        sil = np.where(m, np.asarray(1.0, img.dtype),
                       np.asarray(0.0, img.dtype)).astype(img.dtype)
        return scale_remove(sil, mask, fill, is_depth=False)
    raise ValueError(f"unknown mask condition: {condition}")


SEQ_TAGS = ("training", "testing_still", "testing_walking")


def open_shards(prep, modality, mask_source="user"):
    """Map every manifest row to (image shard, mask shard, position).

    hiride_prep writes ONE SHARD PER SEQUENCE -- `training_depth.npy`,
    `testing_walking_depth.npy`, ... -- each with a `<tag>_index.npz` giving the
    manifest rows it holds, in shard order. A split like R4_cross_session spans
    two sequences, so the trainer has to resolve rows across shards.

    `mask_source` selects which mask shard the conditions cut with:
      user    <tag>_mask.npy     -- the shipped userMap (depth-derived; what a
                                    DEPTH system has at runtime)
      rgbseg  <tag>_maskrgb.npy  -- an RGB-derived person segmentation, written
                                    by hiride_rgbseg.py (what an RGB-ONLY system
                                    could actually obtain; closes the 13.15
                                    asymmetry as a measurement, not a caveat)
    """
    mask_file = {"user": "mask", "rgbseg": "maskrgb"}[mask_source]
    imgs, masks, where = {}, {}, {}
    for tag in SEQ_TAGS:
        ipath = os.path.join(prep, f"{tag}_index.npz")
        if not os.path.exists(ipath):
            continue
        rows = np.load(ipath)["manifest_row"]
        imgs[tag] = np.load(os.path.join(prep, f"{tag}_{modality}.npy"), mmap_mode="r")
        mp = os.path.join(prep, f"{tag}_{mask_file}.npy")
        # A prep with no segmentation (in-house) writes no mask shard. Carry None
        # rather than a zero array: a fake all-background mask would let every
        # mask condition run and return silent nonsense. Same rule for a missing
        # rgbseg shard -- refusing beats silently falling back to the userMap,
        # which would label an oracle-mask run as a deployable-mask one.
        if mask_source != "user" and not os.path.exists(mp):
            raise FileNotFoundError(
                f"{mp} missing. --mask-source {mask_source} needs the RGB-derived "
                f"mask shards:\n  python hiride_rgbseg.py --prep {prep}")
        masks[tag] = np.load(mp, mmap_mode="r") if os.path.exists(mp) else None
        for pos, row in enumerate(rows):
            where[int(row)] = (tag, pos)
    if not imgs:
        raise FileNotFoundError(
            f"no image shards in {prep}. Run the FULL prep (without NO_SHARDS=1):\n"
            f"  sbatch --account=def-czarnuch_cpu run_hiride_prep.slurm")

    # Refuse to train on shards prep did not certify. A killed prep leaves
    # sparse all-zero .npy files that load fine and train to chance in silence.
    meta_path = os.path.join(prep, "prep_meta.json")
    ok = None
    if os.path.exists(meta_path):
        with open(meta_path) as fh:
            ok = json.load(fh).get("shards_ok")
    if ok is not True:
        raise RuntimeError(
            f"prep_meta.json in {prep} does not certify the shards "
            f"(shards_ok={ok!r}). Re-run the full prep and check its exit status:\n"
            f"  sbatch --account=def-czarnuch_cpu run_hiride_prep.slurm")
    return imgs, masks, where


def open_plates(prep, modality, man):
    """row -> per-recording background plate, for the `bg_plate` condition."""
    kind = "rgb" if modality == "rgb" else "depth"
    p = os.path.join(prep, f"plates_{kind}.npy")
    gp = os.path.join(prep, "plate_groups.npy")
    if not (os.path.exists(p) and os.path.exists(gp)):
        raise FileNotFoundError(
            f"{p} missing. The bg_plate condition needs the plates:\n"
            f"  python hiride_plates.py --prep {prep}")
    plates = np.load(p, mmap_mode="r")
    sp = os.path.join(prep, f"plates_seen_{kind}.npy")
    if not os.path.exists(sp):
        raise FileNotFoundError(f"{sp} missing -- re-run hiride_plates.py "
                                f"(it also writes the observed-pixel masks).")
    seen = np.load(sp, mmap_mode="r")
    groups = [str(g) for g in np.load(gp, allow_pickle=False)]
    index = {g: i for i, g in enumerate(groups)}
    missing = sorted(set(man["group"].tolist()) - set(index))
    if missing:
        raise KeyError(f"{len(missing)} recordings have no plate (e.g. {missing[:3]}). "
                       f"Re-run hiride_plates.py against this prep directory.")
    return plates, index, seen


def temporal_windows(idx, man, n_frames):
    """row -> the n_frames rows nearest in time, RESTRICTED TO `idx` ITSELF.

    Multi-frame fusion is what the noise arithmetic points at: depth
    quantisation is ~25 mm at the person's 2955 mm median, and the measured
    frame-to-frame |delta depth| of 80.6 mm implies a per-pixel sigma of roughly
    40-70 mm, against identity-bearing curvature differences of 20-80 mm.
    Averaging N frames cuts that as sqrt(N).

    The window MUST stay inside the split. At R1 a test frame's temporal
    neighbours ARE training frames, so drawing a window from the whole recording
    would reintroduce precisely the adjacency leak this paper exists to measure,
    invisibly, on the rung used as the within-session reference.
    """
    if n_frames <= 1:
        return None
    idx = np.asarray(idx)
    out = {}
    for g in np.unique(man["group"][idx]):
        gi = idx[man["group"][idx] == g]
        gi = gi[np.argsort(man["frame"][gi])]
        for pos, row in enumerate(gi):
            lo = max(0, pos - n_frames // 2)
            hi = min(len(gi), lo + n_frames)
            lo = max(0, hi - n_frames)
            out[int(row)] = [int(r) for r in gi[lo:hi]]
    return out


def load_split_arrays(shards, idx, modality, condition, bits=16, plates=None,
                      man=None, slab_mm=DEPTH_CLIP_MM, n_frames=1,
                      encoding="raw", erode=2, normal_baseline=NORMAL_BASELINE_PX):
    """Load only the frames this split needs, normalised, into RAM."""
    imgs, masks, where = shards
    win = temporal_windows(idx, man, n_frames)
    missing = [int(i) for i in idx if int(i) not in where]
    if missing:
        raise KeyError(f"{len(missing)} manifest rows have no shard entry "
                       f"(e.g. {missing[:5]}). Prep and manifest are out of sync.")
    size = next(iter(imgs.values())).shape[1]
    ch = 3 if modality == "rgb" else 1
    if encoding == "normals":
        if modality != "depth":
            sys.exit("error: --depth-encoding normals applies to depth only.")
        ch = 3
    elif encoding == "depth_sil":
        # Hand the network BOTH cues explicitly: metric depth and the binary
        # outline. sil_scaled (outline only) beats scale_removed (outline +
        # interior) at R4, which may simply mean the net cannot separate the two
        # from one channel.
        if modality != "depth":
            sys.exit("error: --depth-encoding depth_sil applies to depth only.")
        ch = 2
    out = np.empty((len(idx), size, size, ch), dtype=np.float32)
    for j, i in enumerate(idx):
        tag, pos = where[int(i)]
        if win is not None:
            # MEDIAN, not mean: depth dropouts are stored as 0, and a mean would
            # drag every averaged pixel toward that invalid sentinel.
            stack = []
            for r in win[int(i)]:
                t2, p2 = where[int(r)]
                a2 = np.asarray(imgs[t2][p2]).astype(np.float32)
                stack.append(np.where(a2 > 0, a2, np.nan) if modality == "depth" else a2)
            with np.errstate(all="ignore"):
                raw = np.nan_to_num(np.nanmedian(np.stack(stack), axis=0), nan=0.0)
        else:
            raw = np.asarray(imgs[tag][pos])
        if modality == "depth":
            img = np.clip(raw.astype(np.float32), 0.0, DEPTH_CLIP_MM) / DEPTH_CLIP_MM
            if bits < 16:
                img = quantise_depth(img, bits)
            img = img[..., None]
        else:
            img = raw.astype(np.float32) / 255.0
        if masks[tag] is None:
            if condition != "full":
                raise SystemExit(
                    f"condition '{condition}' needs a person mask and this prep has no "
                    f"{tag}_mask.npy. The in-house corpus has no userMap and no usable "
                    f"slab foreground (see hiride_inhouse_probe.py), so it supports "
                    f"--condition full only.")
            out[j] = img
            continue
        m = np.asarray(masks[tag][pos])
        if condition == "bg_plate":
            # The exact complement of `person`: the person's pixels replaced by
            # what that camera sees when nobody stands there, so NO person-shaped
            # boundary survives. Anything above chance here is scene/recording
            # nuisance with the silhouette provably removed -- which is what
            # `bg_hole` cannot claim, since it leaves the hole behind.
            pl, index, seen_all = plates
            gidx = index[str(man["group"][int(i)])]
            raw_pl = np.asarray(pl[gidx])
            # Pixels never observed without the person in that whole recording.
            # Left as-is they would be a person-shaped hole in BOTH modalities
            # (0 = invalid depth; 0 = black in RGB), which is the very thing
            # this condition removes. Fill them with the plate's own median.
            seen_px = np.asarray(seen_all[gidx])
            if modality == "depth":
                plate = np.clip(raw_pl.astype(np.float32), 0.0, DEPTH_CLIP_MM) / DEPTH_CLIP_MM
                if bits < 16:
                    plate = quantise_depth(plate, bits)
                plate = plate[..., None]
                ok = seen_px[..., None] & (plate > 0)
            else:
                plate = raw_pl.astype(np.float32) / 255.0
                ok = np.repeat(seen_px[..., None], 3, axis=2)
            if not ok.all():
                med = np.median(plate[ok]) if ok.any() else 0.0
                plate = np.where(ok, plate, med)
            out[j] = np.where(m[..., None] > 0, plate, img).astype(np.float32)
        else:
            edited = apply_mask_condition(img, m, condition, 0.0, slab_mm=slab_mm,
                                          erode=erode)
            if encoding == "normals":
                keep_m = erode_mask(m, erode) if condition == "interior_only" else (m > 0)
                out[j] = depth_to_normals(edited, keep_m, normal_baseline)
            elif encoding == "depth_sil":
                sil = apply_mask_condition(img, m, "sil_scaled", 0.0, slab_mm=slab_mm,
                                           erode=erode)
                out[j] = np.concatenate([edited, sil], axis=-1)
            else:
                out[j] = edited
    # IN-PLACE. `out * 2.0 - 1.0` allocates a second full-size array, doubling
    # peak memory: an RGB R0 split is 13,335 x 256 x 256 x 3 x 4 B = 10.5 GB, so
    # the copy pushed the job past --mem and 10 array cells died OUT_OF_MEMORY.
    out *= 2.0
    out -= 1.0
    return out                                   # to [-1, 1], both modalities


class ArrayBatches(tf.keras.utils.Sequence):
    """Feed a host numpy array to the model ONE BATCH AT A TIME.

    Passing a whole numpy array to fit()/predict() makes Keras materialise it as
    a single GPU tensor (it builds a Dataset from a tf.constant of the array).
    On the 10 GB MIG slice that is fatal for RGB: an R0 RGB training split is
    10.5 GB, and R1/R4 RGB (5.4-5.8 GB) fitted but then died in predict() when
    the test set had to be resident beside it -- 15 array cells, all with
    `InternalError: ... Dst tensor is not initialized`. Depth is 1 channel and
    never reached the ceiling, which is why the depth arms completed.

    Shuffling is done here, at the SAMPLE level with a seeded numpy RNG, so
    fit() is called with shuffle=False. (Keras' own shuffle for a Sequence only
    permutes batch order.) With shuffle=False the order is the identity, which
    predict() relies on to line predictions up with the truth vector.
    """

    def __init__(self, X, y=None, batch_size=32, shuffle=False, seed=0,
                 augment=0, fill=-1.0, A=None):
        self.X, self.y, self.bs, self.shuffle = X, y, int(batch_size), shuffle
        # aux rows are indexed with the SAME permutation as X, so shuffling
        # cannot silently pair a frame with another frame's scalars
        self.A = A
        # augment = max translation in px. Random shift + horizontal flip, applied
        # ONLY to training batches. This targets the nuisance the mechanism suite
        # identified as decisive -- framing -- by making the network invariant to
        # it, rather than normalising it away in preprocessing. Deliberately no
        # scaling: apparent size is a real cue at R1 and rescaling would need
        # interpolation on depth, which the boundary analysis says is exactly
        # where artefacts come from.
        self.augment, self.fill = int(augment), float(fill)
        self.rng = np.random.default_rng(seed)
        self.order = np.arange(len(X))
        if shuffle:
            self.rng.shuffle(self.order)

    def __len__(self):
        return int(np.ceil(len(self.X) / self.bs))

    def _aug(self, xb):
        xb = xb.copy()
        k = self.augment
        for n in range(len(xb)):
            if self.rng.random() < 0.5:
                xb[n] = xb[n][:, ::-1]
            dy, dx = (int(self.rng.integers(-k, k + 1)) for _ in range(2))
            if dy or dx:
                out = np.full_like(xb[n], self.fill)
                H, W = xb[n].shape[:2]
                ys, xs = slice(max(0, dy), min(H, H + dy)), slice(max(0, dx), min(W, W + dx))
                sy, sx = slice(max(0, -dy), min(H, H - dy)), slice(max(0, -dx), min(W, W - dx))
                out[ys, xs] = xb[n][sy, sx]
                xb[n] = out
        return xb

    def __getitem__(self, i):
        idx = self.order[i * self.bs:(i + 1) * self.bs]
        xb = self.X[idx]
        if self.augment:
            xb = self._aug(xb)
        xb = xb if self.A is None else [xb, self.A[idx]]
        if self.y is None:
            return xb                     # plain x; see predict_seq() below
        return xb, self.y[idx]

    def on_epoch_end(self):
        if self.shuffle:
            self.rng.shuffle(self.order)


def predict_seq(model, seq):
    """Batch-wise predict that does NOT depend on how Keras unpacks a Sequence.

    `model.predict(sequence)` has to guess whether a returned container is
    `x`, `(x, y)`, or `(x, y, w)`. For a two-input model `__getitem__` must
    yield `[image, aux]`, which is indistinguishable from `(x, y)` -- Keras
    took the image as x and the aux vector as targets, so the model received
    one input and raised `expects 2 input(s), but it received 1`. Wave 17
    trained for 60 epochs before dying there, twice, and a 1-tuple did not fix
    it either.

    `predict_on_batch` takes x directly, with nothing to unpack. Order is the
    identity because the Sequence is built with shuffle=False, which is what
    lines predictions up against the truth vector.
    """
    return np.concatenate([np.asarray(model.predict_on_batch(seq[i]))
                           for i in range(len(seq))], axis=0)


class TestCurve(tf.keras.callbacks.Callback):
    """Record TEST frame accuracy after every epoch -- a diagnostic curve only.

    It is never used for stopping or model selection (EarlyStopping still
    watches val_accuracy on the training session). Its purpose is the
    estimator-sensitivity question a cross-session rung raises: when the
    within-session validation set saturates at 1.0 after one epoch (ImageNet
    ConvNeXt on RGB does exactly that), the "best" epoch is the first one and
    the reported R4 number is one epoch of fine-tuning. The curve shows what
    any other stopping rule would have reported, without re-running.
    """

    def __init__(self, seq, truth):
        super().__init__()
        self.seq, self.truth, self.accs = seq, truth, []

    def on_epoch_end(self, epoch, logs=None):
        prob = predict_seq(self.model, self.seq)
        acc = float((np.argmax(prob, axis=1) == self.truth).mean())
        self.accs.append(acc)
        if logs is not None:
            logs["test_acc"] = acc                 # also lands in hist.history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--policy", required=True)
    ap.add_argument("--modality", choices=("depth", "rgb"), required=True)
    ap.add_argument("--arch", choices=("alexnet", "convnext_tiny"), default="alexnet")
    ap.add_argument("--init", choices=("imagenet", "scratch"), default="imagenet",
                    help="convnext_tiny only: ImageNet init (fixed control) or scratch")
    ap.add_argument("--condition", choices=MASK_CONDITIONS, default="full")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--guard", type=int, default=150)
    ap.add_argument("--ref-guard", type=int, default=150)
    ap.add_argument("--cross-val-guard", type=int, default=50,
                    help="R3/R4 only: the guard between the training block and "
                         "the validation tail carved from each TRAINING "
                         "recording (hiride_data._policy_cross's `guard`; the "
                         "test set is untouched). Default 50 = the behaviour "
                         "of every BIWI run. TVRID tracklets are ~40 frames, "
                         "so its wave sets 5 -- like --ref-guard this is a "
                         "construction detail, constant within one runs dir, "
                         "and deliberately NOT a cell axis.")
    ap.add_argument("--ref-eligibility", choices=("match", "cues"), default="match",
                    help="which frame filter the R1 match_ntrain reference is computed "
                         "under. 'match' uses --eligibility (default); 'cues' uses the "
                         "standard filter even when --eligibility is full_body, which is "
                         "required at guard 150 because the full-body filter starves three "
                         "recordings of training frames outright.")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--head", choices=("gap", "stripe", "flatten"), default="gap",
                    help="alexnet only: how the final feature map is pooled. GAP is "
                         "translation-invariant and discards the correspondence that "
                         "scale_removed/interior_only establish; stripe and flatten keep it.")
    ap.add_argument("--frames", type=int, default=1,
                    help="multi-frame median over the N frames nearest in time, "
                         "restricted to rows inside the same split so no adjacency "
                         "leak is introduced. Attacks the sensor noise floor.")
    ap.add_argument("--augment", type=int, default=0,
                    help="train-time random translation up to N px + horizontal flip. "
                         "Makes the net invariant to framing instead of normalising it "
                         "away; 0 = the published behaviour.")
    ap.add_argument("--test-fuse", type=int, default=1,
                    help="TRACKLET evaluation: average the predicted probabilities over "
                         "N consecutive test frames of one recording and score the window "
                         "once. Unlike --frames this never blurs the input, which is why "
                         "input fusion failed; it is also how a deployed system sees a "
                         "person. Reported ALONGSIDE frame accuracy, never instead of it.")
    ap.add_argument("--depth-encoding", choices=("raw", "normals", "depth_sil"), default="raw",
                    help="'normals' replaces depth with 3-channel unit surface normals: "
                         "scale-free, so immune to the contrast problem, but noise "
                         "amplifying -- pair with --frames.")
    ap.add_argument("--cohort", type=int, default=0, metavar="K",
                    help="restrict the enrolled cohort to K subjects (0 = all). "
                         "Accuracy is bound to gallery size -- 49.90 %% at R4 is "
                         "on 28 people, where chance is 3.57 %% -- so a "
                         "deployment reader needs the CURVE, not one point. "
                         "Subjects are drawn from those present in BOTH pools "
                         "of the policy, so every drawn class has train and "
                         "test frames.")
    ap.add_argument("--cohort-seed", type=int, default=0, metavar="S",
                    help="which draw of K subjects. Vary it: a single subset is "
                         "one sample of an easy-or-hard cohort, not a "
                         "measurement of cohort size.")
    ap.add_argument("--aux", choices=("none", "dist", "metric"), default="none",
                    help="`dist` appends the subject's standing distance "
                         "(cues p_med, z-scored on TRAIN rows) to the pooled "
                         "image features. It is the one quantity needed to "
                         "turn pixel extent into millimetres, and the network "
                         "is otherwise never given it. Uses cues.npz, so the "
                         "frame set is UNCHANGED and these cells compare "
                         "directly against their --aux none partners. "
                         "`metric` appends the 12 published metric scalars "
                         "(hiride_metric.BASE_METRIC, z-scored on TRAIN rows) "
                         "-- FEATURE-level fusion. Score-level fusion proved "
                         "the two representations complementary (~8 pp oracle "
                         "headroom, 13.10) but no fixed posterior rule "
                         "captured it; a joint model can learn a "
                         "frame-dependent combination. Frames lacking metric "
                         "features get the train mean (= 0 after z-scoring), "
                         "so the frame set is again UNCHANGED.")
    ap.add_argument("--mask-source", choices=("user", "rgbseg"), default="user",
                    help="which person mask the mask conditions cut with. "
                         "`user` = the shipped userMap: depth-derived, so "
                         "masked-DEPTH rows describe a deployable depth "
                         "system while masked-RGB rows describe an RGB-D one "
                         "(13.15). `rgbseg` = an RGB-derived segmentation "
                         "(hiride_rgbseg.py), what an RGB-ONLY camera could "
                         "obtain -- converts that caveat into a measurement.")
    ap.add_argument("--erode", type=int, default=2,
                    help="pixels to erode the mask by for interior_only; also drops the "
                         "rim contaminated by prep's AREA resize.")
    ap.add_argument("--normal-baseline", type=int, default=NORMAL_BASELINE_PX,
                    help="lateral baseline in px for the surface-normal difference")
    ap.add_argument("--depth-slab-mm", type=float, default=DEPTH_CLIP_MM,
                    help="scale_removed only: rescale a slab_mm window centred on the "
                         "person's median depth onto the full input range. Default "
                         "6000 = the global scale (identity transform).")
    ap.add_argument("--eligibility", choices=("cues", "all", "full_body"), default="cues",
                    help="'all' skips the cue-based frame filter, for a prep that has "
                         "no masks and therefore no cues (the in-house corpus). "
                         "'full_body' additionally requires the person to touch neither "
                         "the top nor the bottom frame edge, which removes the 5x "
                         "train/test framing shift documented by hiride_range_profile.py. "
                         "It changes the evaluated population, so it is a separate "
                         "reported condition, never a swap-in for the headline number.")
    ap.add_argument("--bits", type=int, default=16, choices=(16, 8, 6, 4, 3, 2, 1),
                    help="depth only: quantise to 2**bits levels at fixed global "
                         "range (16 = untouched). The Z-precision axis.")
    ap.add_argument("--track-test", action="store_true",
                    help="diagnostic: record test accuracy after every epoch (never "
                         "used for selection); adds ~10 s/epoch")
    ap.add_argument("--permute-labels", action="store_true",
                    help="null control: permute labels within the rows this split "
                         "uses (train+val+test) before training")
    args = ap.parse_args()

    for f in (("manifest.npz",) if args.eligibility == "all" else ("manifest.npz", "cues.npz")):
        if not os.path.exists(os.path.join(args.prep, f)):
            sys.exit(f"error: {f} missing from {args.prep}. Run prep first:\n"
                     f"  sbatch --account=def-czarnuch_cpu run_hiride_prep.slurm")

    os.makedirs(args.out, exist_ok=True)
    tf.keras.utils.set_random_seed(args.seed)
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    ref_keep = None
    if args.eligibility == "all":
        keep = np.ones(len(man["subject"]), dtype=bool)
    else:
        cues = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
        feats = [str(f) for f in cues["feats"]]
        keep = eligible_mask(cues["cues"], feats,
                             full_body=(args.eligibility == "full_body"))
        if args.ref_eligibility == "cues":
            ref_keep = eligible_mask(cues["cues"], feats, full_body=False)

    kw = {}
    if args.policy.startswith(("R3", "R4")):
        kw = dict(guard=args.cross_val_guard)
    if args.policy.startswith("R1"):
        # The reference counts that match_ntrain subsamples toward. Under
        # --eligibility full_body, computing them from the SAME mask starves
        # three recordings of training frames entirely (guard 150 plus the
        # full-body filter leaves nothing), and block_train_counts refuses
        # rather than silently dropping those classes. --ref-eligibility cues
        # takes the reference from the standard mask instead: match_ntrain only
        # ever subsamples, so a recording with fewer full-body frames than the
        # target simply keeps all of them. The full-body arm therefore trains on
        # AT MOST what its cues partner does -- conservative, so a win cannot be
        # attributed to extra data.
        kw = dict(guard=args.guard,
                  match_ntrain=block_train_counts(
                      man, guard=args.ref_guard, seed=0,
                      keep=ref_keep if ref_keep is not None else keep))
    tr, va, te = make_split(man, args.policy, seed=args.seed, keep=keep, **kw)
    if args.cohort:
        # Draw from subjects present in BOTH pools, so a drawn class cannot land
        # in training with no test frames (or the reverse) and quietly change
        # what K means. Split once to find them, then re-split on the subset.
        subj = np.asarray(man["subject"], dtype=str)
        common = sorted(set(subj[tr].tolist()) & set(subj[te].tolist()))
        if args.cohort > len(common):
            raise SystemExit(f"error: --cohort {args.cohort} exceeds the "
                             f"{len(common)} subjects present in both pools of "
                             f"{args.policy}")
        pick = set(np.random.default_rng([args.cohort_seed, args.cohort]).choice(
            np.array(common), size=args.cohort, replace=False).tolist())
        keep = keep & np.isin(subj, list(pick))
        if args.policy.startswith("R1"):
            kw = dict(kw, match_ntrain=block_train_counts(
                man, guard=args.ref_guard, seed=0,
                keep=(ref_keep if ref_keep is not None else keep) & np.isin(subj, list(pick))))
        tr, va, te = make_split(man, args.policy, seed=args.seed, keep=keep, **kw)
        print(f"[cohort] K={args.cohort} draw {args.cohort_seed} of {len(common)} "
              f"eligible: {sorted(pick)}")
    info = describe_split(man, tr, va, te)

    classes = sorted(set(man["subject"][tr].tolist()))
    cls_index = {c: i for i, c in enumerate(classes)}
    unseen = sorted(set(man["subject"][te].tolist()) - set(classes))
    if unseen:
        sys.exit(f"error: {len(unseen)} test subjects absent from training "
                 f"({unseen[:5]}...). A closed-set classifier has no output unit "
                 f"for them; this policy is identity-disjoint and belongs to paper 3.")
    # Label vector over the WHOLE manifest, but the class map only covers the
    # TRAINING subjects. For R3/R4 the training set is the 28 subjects present
    # in both sessions, while the manifest holds all 50 -- so a direct
    # comprehension raises KeyError on the 22 Training-only subjects. Rows
    # outside the class map get -1 and must never be indexed; the splits only
    # ever touch subjects that are in the map, which is asserted below.
    y = np.full(len(man["subject"]), -1, dtype=np.int64)
    for cls, idx in cls_index.items():
        y[man["subject"] == cls] = idx
    for name, part in (("train", tr), ("val", va), ("test", te)):
        if (y[part] < 0).any():
            bad = sorted(set(man["subject"][part][y[part] < 0].tolist()))
            sys.exit(f"error: {name} split contains subjects absent from the "
                     f"class map ({bad[:5]}). This should be impossible -- the "
                     f"split library and the class map have diverged.")
    if args.permute_labels:
        # Permute WITHIN the rows this split touches, never over the whole
        # manifest. A global permutation dragged the -1 of the 22 Training-only
        # subjects into R4's train/test rows: sparse CE on label -1 gave
        # `loss: nan` from epoch 1 and np.bincount(truth) crashed on the
        # negative labels (all three R4 null cells, array 18539954). Permuting
        # within train+val+test keeps the rung's own class marginals -- the same
        # null hiride_floor.py measures -- and touches no out-of-map row.
        rows = np.concatenate([tr, va, te])
        assert len(np.unique(rows)) == len(rows), "train/val/test overlap"
        y[rows] = np.random.default_rng(5000 + args.seed).permutation(y[rows])
        assert (y[rows] >= 0).all(), "permutation leaked an out-of-map label"

    t0 = time.time()
    shards = open_shards(args.prep, args.modality, args.mask_source)
    plates = (open_plates(args.prep, args.modality, man)
              if args.condition in PLATE_CONDITIONS else None)
    Xtr = load_split_arrays(shards, tr, args.modality, args.condition, args.bits,
                             plates, man, args.depth_slab_mm, args.frames,
                             args.depth_encoding, args.erode, args.normal_baseline)
    Xva = load_split_arrays(shards, va, args.modality, args.condition, args.bits,
                             plates, man, args.depth_slab_mm, args.frames,
                             args.depth_encoding, args.erode, args.normal_baseline)
    Xte = load_split_arrays(shards, te, args.modality, args.condition, args.bits,
                             plates, man, args.depth_slab_mm, args.frames,
                             args.depth_encoding, args.erode, args.normal_baseline)
    print(f"[data] loaded in {time.time() - t0:.0f}s  train={Xtr.shape} "
          f"val={Xva.shape} test={Xte.shape}  classes={len(classes)}")

    Atr = Ava = Ate = None
    aux_dim = 0
    if args.aux == "dist":
        cz = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
        pm = cz["cues"][:, [str(f) for f in cz["feats"]].index("p_med")].astype(np.float32)
        # standardised on TRAIN rows only -- test statistics must not leak into
        # the input scaling
        mu, sd = float(pm[tr].mean()), float(pm[tr].std()) or 1.0
        Atr, Ava, Ate = ((pm[i] - mu).reshape(-1, 1) / sd for i in (tr, va, te))
        aux_dim = 1
        print(f"[aux] standing distance, train mean {mu:.0f} mm sd {sd:.0f} mm")
    elif args.aux == "metric":
        # The 12 published metric scalars (BASE_METRIC), aligned to manifest
        # rows exactly as hiride_fuse.load_metric does. Per-frame quantities
        # from that frame's own depth + mask -- nothing crosses frames, so no
        # split can leak through them. z-scored on TRAIN rows only; rows
        # without metric features (person < 200 px) get the train mean, i.e. 0,
        # so the frame set matches the --aux none partner exactly.
        from hiride_metric import BASE_METRIC
        mf = np.load(os.path.join(args.prep, "metric_features.npz"),
                     allow_pickle=False)
        mnames = [str(n) for n in mf["names"]]
        cols = [mnames.index(n) for n in BASE_METRIC if n in mnames]
        if len(cols) != len(BASE_METRIC):
            sys.exit(f"error: metric_features.npz holds {len(cols)} of the "
                     f"{len(BASE_METRIC)} published metric columns -- re-run "
                     f"hiride_metric.py before using --aux metric.")
        A = np.full((len(man["subject"]), len(cols)), np.nan, dtype=np.float32)
        A[mf["manifest_row"]] = mf["feats"][:, cols]
        mu = np.nanmean(A[tr], axis=0)
        sd = np.nanstd(A[tr], axis=0)
        sd[sd == 0] = 1.0
        A = (A - mu) / sd
        A = np.nan_to_num(A, nan=0.0)
        Atr, Ava, Ate = A[tr], A[va], A[te]
        aux_dim = len(cols)
        miss_tr = int((~np.isin(tr, mf["manifest_row"])).sum())
        miss_te = int((~np.isin(te, mf["manifest_row"])).sum())
        print(f"[aux] {aux_dim} metric scalars; imputed train {miss_tr}/{len(tr)}, "
              f"test {miss_te}/{len(te)} rows at the train mean")

    shape = Xtr.shape[1:]
    model = (build_alexnet(shape, len(classes), args.head, aux_dim=aux_dim)
             if args.arch == "alexnet"
             else build_convnext(shape, len(classes), pretrained=(args.init == "imagenet"),
                                 aux_dim=aux_dim,
                                 head=args.head))
    # Optimiser parity: identical settings for every architecture. 2023 gave
    # ViT clipvalue=0.1 and AlexNet none, confounding the architecture contrast.
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss="sparse_categorical_crossentropy", metrics=["accuracy"])

    # Batch-wise feeding (see ArrayBatches). Do NOT pass batch_size/shuffle to
    # fit()/predict() here: the Sequence owns both.
    train_seq = ArrayBatches(Xtr, y[tr], args.batch_size, shuffle=True, seed=args.seed,
                             A=Atr,
                             augment=args.augment)
    val_seq = ArrayBatches(Xva, y[va], args.batch_size, A=Ava)
    test_seq = ArrayBatches(Xte, None, args.batch_size, A=Ate)
    es = tf.keras.callbacks.EarlyStopping(
        # 2023 monitored val_loss with restore_best_weights=False, so the
        # reported number was whatever the model scored 10 epochs AFTER it
        # stopped improving. Both are fixed.
        monitor="val_accuracy", mode="max", patience=args.patience,
        restore_best_weights=True)
    # TestCurve goes BEFORE EarlyStopping so an epoch's test accuracy is
    # measured on that epoch's weights, not on the restored best.
    tracker = TestCurve(test_seq, y[te]) if args.track_test else None
    hist = model.fit(train_seq, validation_data=val_seq, shuffle=False,
                     epochs=args.epochs, verbose=2,
                     callbacks=([tracker] if tracker else []) + [es])
    if not np.isfinite(hist.history["loss"]).all():
        sys.exit("error: non-finite training loss -- refusing to write a result "
                 "for a diverged run.")
    # tf.keras 2.15 restores best weights ONLY when patience actually fires. A
    # run that hits --epochs with its best val_accuracy inside the last
    # `patience` epochs exits fit() on last-epoch weights while best_epoch
    # reports the argmax -- silently inconsistent. Restore unconditionally
    # (idempotent when the callback already did it).
    if es.best_weights is not None:
        model.set_weights(es.best_weights)
    hit_epoch_cap = es.stopped_epoch == 0 and len(hist.history["loss"]) >= args.epochs

    prob = predict_seq(model, test_seq)
    assert prob.shape[0] == len(te), (prob.shape, len(te))
    pred = np.argmax(prob, axis=1)
    truth = y[te]
    # VALIDATION posteriors, on the restored best weights. Until now only test
    # rows were stored, so no fusion weight could be chosen without touching
    # the test set -- which is what forced the fixed-weight rules in 13.10.
    # Val rows are within-session for R3/R4, so a weight tuned on them answers
    # "best within-session mixture", not the cross-session optimum; that caveat
    # travels with any use of these, but tuned-on-val is at least honest.
    vprob = predict_seq(model, ArrayBatches(Xva, None, args.batch_size, A=Ava))
    assert vprob.shape[0] == len(va), (vprob.shape, len(va))
    labels = list(range(len(classes)))          # 2023 omitted labels=, so the
                                                # macro denominator moved per epoch
    per_subject = [float((pred[truth == c] == c).mean())
                   for c in labels if (truth == c).any()]
    fused = {}
    if args.test_fuse > 1:
        # group consecutive TEST frames of one recording into windows and score
        # the mean probability once per window
        order = np.argsort(man["frame"][te])
        gp = man["group"][te]
        f_pred, f_true = [], []
        for g in np.unique(gp):
            gi = np.array([o for o in order if gp[o] == g])
            for lo in range(0, len(gi), args.test_fuse):
                w = gi[lo:lo + args.test_fuse]
                # require at least HALF a window (rounded up), so every scored
                # window rests on comparable evidence. max(2, N//2) let a 2-frame
                # stub through at N=5, i.e. 40 % of the evidence of a full window.
                if len(w) < max(2, (args.test_fuse + 1) // 2):
                    continue
                f_pred.append(int(np.argmax(prob[w].mean(0))))
                f_true.append(int(truth[w[0]]))
        if f_pred:
            f_pred, f_true = np.array(f_pred), np.array(f_true)
            fused = dict(fused_acc=float((f_pred == f_true).mean()),
                         fused_n=int(len(f_pred)), test_fuse=args.test_fuse,
                         fused_per_subject=float(np.mean(
                             [(f_pred[f_true == c] == c).mean()
                              for c in np.unique(f_true)])))
            print(f"[tracklet] window={args.test_fuse}  {fused['fused_n']} windows  "
                  f"acc={100 * fused['fused_acc']:.2f}%  "
                  f"per-subj={100 * fused['fused_per_subject']:.2f}%")

    res = dict(
        policy=args.policy, modality=args.modality, arch=args.arch,
        init=(args.init if args.arch == "convnext_tiny" else None),
        condition=args.condition, seed=args.seed,
        guard=args.guard if args.policy.startswith("R1") else None,
        permuted=bool(args.permute_labels),
        frame_acc=float((pred == truth).mean()),
        per_subject_acc=float(np.mean(per_subject)),
        macro_f1=float(f1_score(truth, pred, labels=labels, average="macro",
                                zero_division=0)),
        chance=1.0 / len(classes),
        majority_class_rate=float(np.bincount(truth).max() / len(truth)),
        best_epoch=int(np.argmax(hist.history["val_accuracy"])),
        epochs_run=len(hist.history["val_accuracy"]),
        hit_epoch_cap=bool(hit_epoch_cap),     # True => consider raising --epochs
        feeding="sequence_v2",                  # code-path provenance for collate
        depth_clip_mm=DEPTH_CLIP_MM, bits=args.bits, depth_slab_mm=args.depth_slab_mm,
        frames=args.frames, encoding=args.depth_encoding, erode=args.erode, head=args.head,
        eligibility=args.eligibility, ref_eligibility=args.ref_eligibility,
        aux=args.aux, cohort=args.cohort, cohort_seed=args.cohort_seed,
        mask_source=args.mask_source,
        cross_val_guard=(args.cross_val_guard
                         if args.policy.startswith(("R3", "R4")) else None),
        augment=args.augment, **fused,
        normal_baseline=args.normal_baseline,
        n_classes=len(classes),
        elapsed_s=round(time.time() - t0, 1),
        history={k: [float(v) for v in vals] for k, vals in hist.history.items()},
        test_curve=(tracker.accs if tracker else None),
        # describe_split also reports `chance`; drop it so the explicit value
        # above (computed from the training class count) wins.
        **{k: v for k, v in info.items() if k != "chance"})
    res["x_chance"] = res["frame_acc"] / res["chance"]

    tag = run_tag(res)
    path = os.path.join(args.out, f"results_{tag}.json")
    # A filename that does not encode every cell-defining axis loses runs
    # silently -- the array job still reports COMPLETED, and the loss only
    # shows up later as a cell with fewer seeds than were asked for. Compare
    # against whatever is already there and refuse rather than overwrite.
    # Re-running an IDENTICAL cell stays allowed: that is how a wave tops up
    # seeds without knowing which ones exist.
    if os.path.exists(path):
        try:
            prev = json.load(open(path))
        except (ValueError, OSError):
            prev = {}
        # A field ABSENT from the file on disk is not a difference -- it is an
        # older trainer that did not record that axis yet. Wave 17 lost 48
        # trained jobs to this: every run finished its 60 epochs and was then
        # refused at the write, because files from waves 2-9 have no `bits`,
        # `head` or `augment` key and None != 16 compares unequal. AXES already
        # states what each axis's default means, so resolve through it.
        clash = {k: (_axis(prev, k), _axis(res, k)) for k in CELL_FIELDS
                 if _axis(prev, k) != _axis(res, k)}
        if clash:
            raise SystemExit(
                f"error: {os.path.basename(path)} already holds a DIFFERENT cell -- "
                + "; ".join(f"{k}: on disk {o!r}, this run {n!r}"
                            for k, (o, n) in sorted(clash.items()))
                + ". The filename does not distinguish these, so writing would "
                  "destroy the run on disk. Add the differing axis to `tag`.")
    with open(path, "w") as fh:
        json.dump(res, fh, indent=1)
    # Per-frame predictions keyed on manifest row: RGB and depth cells of one
    # (policy, seed) score byte-identical frame sets, so this is what a McNemar
    # test or a subject-cluster bootstrap reads later. Cheap; do not drop it.
    np.savez_compressed(os.path.join(args.out, f"cm_{tag}.npz"),
                        cm=confusion_matrix(truth, pred, labels=labels),
                        classes=np.array(classes),
                        test_rows=np.asarray(te, dtype=np.int64),
                        test_subject=man["subject"][te],
                        truth=truth.astype(np.int64), pred=pred.astype(np.int64),
                        prob=prob.astype(np.float16),
                        val_rows=np.asarray(va, dtype=np.int64),
                        val_subject=man["subject"][va],
                        val_truth=y[va].astype(np.int64),
                        val_prob=vprob.astype(np.float16))
    print(f"[done] {tag}  acc={res['frame_acc'] * 100:.2f}% "
          f"({res['x_chance']:.1f}x chance {res['chance'] * 100:.2f}%)  "
          f"per-subj={res['per_subject_acc'] * 100:.2f}%  "
          f"macroF1={res['macro_f1']:.3f}  best_epoch={res['best_epoch']}")


if __name__ == "__main__":
    main()
