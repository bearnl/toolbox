"""Metric 3D features: use depth AS GEOMETRY, not as a grey image.

NO SKELETON. Paper 1 is published and skeleton-only, so _skel.txt is
off-limits to paper 2 entirely (territory decision, 2026-08-19).

    python hiride_metric.py --prep $SCRATCH/hiride2/prep --root $SLURM_TMPDIR/biwi

Everything this project has done to depth so far is a 2D IMAGE operation --
resize, mask, contrast-normalise, quantise, convolve, pool. Each one destroys
metric content, and the measured results follow: contrast changes bought
nothing, fusion hurt, normals hurt, and the CNN ends up using the silhouette.

The specific loss is `scale_removed`. It rescales every person to a fixed
bounding-box height because APPARENT size varies with standing distance. But
that conflates two different things: apparent size varying with distance (a
nuisance) and TRUE size varying between people (the single most stable biometric
a body has). The right operation is not "make everyone the same size" -- it is
"convert apparent size to true size", which depth supports and RGB cannot.

And it is exactly what R4 needs. The census showed the cross-session failure is
CAMERA POSE: background +620 mm for all 28 shared subjects, people ~30 cm closer
and 34 px lower in frame. In an image, "person 30 cm closer" and "person is
larger" are indistinguishable -- which is why scale_removed had to delete size.
Rectifying against the shipped ground plane cancels camera translation, camera
pitch and standing distance BY CONSTRUCTION while leaving true body size intact.

Features written per frame (all camera-invariant unless marked):
    stature_mm            p99.5 of height above the ground plane
    height_p50/p05        median / low percentile of body height
    w_XX                  body width in mm at 6 fixed fractions of stature
    depth_extent_mm       front-to-back extent along the viewing axis
    surface_area_m2       sum of per-pixel areas (z/f)^2 -- a build proxy
    volume_proxy_l        surface area x mean depth extent
    n_points, valid_frac
    top_clip/bot_clip     mask touches the frame edge -> stature unmeasurable
    stand_dist_mm         NUISANCE, kept only as a covariate

NOTHING EXISTING IS MODIFIED. This writes one new file, metric_features.npz,
and derives the skel/groundCoeff paths from the depth path via BIWI's fixed
slot letters (-a rgb, -b depth, -c userMap, -d skel, -e groundCoeff), so the
manifest schema is untouched.

The two .txt formats have never been opened by this project, so the parsers are
defensive: they log the raw first file, accept several plausible layouts, and
fall back (camera-Y as vertical; skeleton features omitted) rather than aborting
a multi-hour job. Read the log to learn the true format.
"""
import os
import json
import argparse

import numpy as np

from hiride_data import load_manifest, build_manifest
from hiride_pgm import read_pgm

FX = FY = 575.816
CX, CY = 320.0, 240.0
WIDTH_FRACS = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
_LOGGED = {"ground": False}


def sibling(depth_rel, slot, kind, ext):
    """`..._<frame>-b_<ts>_depth.pgm` -> the same frame's other file."""
    base = os.path.basename(depth_rel)
    parts = base.split("_")
    if len(parts) < 4:
        return None
    parts[1] = parts[1].rsplit("-", 1)[0] + "-" + slot
    parts[-1] = f"{kind}.{ext}"
    return os.path.join(os.path.dirname(depth_rel), "_".join(parts))


def read_floats(path, limit=None):
    try:
        with open(path) as fh:
            txt = fh.read()
    except OSError:
        return None, None
    vals = []
    for tok in txt.replace(",", " ").split():
        try:
            vals.append(float(tok))
        except ValueError:
            pass
    return np.array(vals, dtype=np.float64), txt


def read_ground(path):
    v, txt = read_floats(path)
    if not _LOGGED["ground"]:
        _LOGGED["ground"] = True
        print(f"[format] groundCoeff first file: {os.path.basename(path)}\n"
              f"         raw: {(txt or '')[:200]!r}\n"
              f"         parsed {0 if v is None else len(v)} floats: "
              f"{None if v is None else v[:8]}")
    if v is None or len(v) < 4:
        return None
    n = v[:3]
    if np.linalg.norm(n) < 1e-6:
        return None
    return v[:4]


# The 12 columns the published numbers were computed from (19.04 % at R4,
# 67.97 % at R1). PINNED: adding features to the npz must not silently change
# what "metric" means, or every earlier number becomes unreproducible. New
# features join `shape`, and `metric+shape` is where they are evaluated.
BASE_METRIC = ("depth_extent_mm", "height_p05", "height_p50", "stature_mm",
               "surface_area_m2", "volume_proxy_l",
               "w_15", "w_30", "w_45", "w_60", "w_75", "w_90")
SHAPE_PREFIXES = ("hw_", "ht_", "hc_", "ha_", "r_")
NUISANCE = ("stand_dist_mm", "ground", "n_points", "valid_frac",
            "top_clip", "bot_clip")

# Offsets below the crown, in millimetres, at which the body is measured.
# Absolute, not fractions of stature: a fraction of a clipped stature points at
# the wrong anatomy, which is the defect these replace.
HEAD_OFFSETS_MM = (100, 200, 300, 400, 500, 600, 700, 800)
BAND_HALF_MM = 40.0
# Scale-free width ratios: shoulder-ish over hip-ish, and two neighbours.
SHAPE_RATIOS = ((200, 700), (200, 400), (400, 700))


def frame_features(depth, user, ground):
    """One frame -> dict of metric features in a gravity-aligned frame."""
    m = (user > 0) & (depth > 0)
    out = {}
    H, W = depth.shape
    n_pts = int(m.sum())
    out["n_points"] = float(n_pts)
    out["valid_frac"] = float((depth > 0).mean())
    out["top_clip"] = float((user > 0)[0].any())
    out["bot_clip"] = float((user > 0)[-1].any())
    if n_pts < 200:
        return None
    ys, xs = np.nonzero(m)
    z = depth[ys, xs].astype(np.float64)
    X = (xs - CX) * z / FX
    Y = (ys - CY) * z / FY
    P = np.stack([X, Y, z], 1)
    out["stand_dist_mm"] = float(np.median(z))          # NUISANCE covariate

    if ground is not None:
        nvec = ground[:3] / np.linalg.norm(ground[:3])
        h = (P @ nvec + ground[3] / np.linalg.norm(ground[:3]))
        if np.median(h) < 0:                            # orient "up" positive
            nvec, h = -nvec, -h
        out["ground"] = 1.0
    else:
        nvec = np.array([0.0, -1.0, 0.0])               # fallback: camera -Y
        h = P @ nvec
        h = h - np.percentile(h, 0.5)
        out["ground"] = 0.0
    fwd = np.array([0.0, 0.0, 1.0]) - nvec * float(nvec[2])
    if np.linalg.norm(fwd) < 1e-6:
        fwd = np.array([1.0, 0.0, 0.0])
    fwd /= np.linalg.norm(fwd)
    right = np.cross(nvec, fwd)

    lat = P @ right
    dep = P @ fwd
    stature = float(np.percentile(h, 99.5) - np.percentile(h, 0.5))
    out["stature_mm"] = stature
    out["height_p50"] = float(np.percentile(h, 50) - np.percentile(h, 0.5))
    out["height_p05"] = float(np.percentile(h, 5) - np.percentile(h, 0.5))
    out["depth_extent_mm"] = float(np.percentile(dep, 99) - np.percentile(dep, 1))
    base = np.percentile(h, 0.5)
    for f in WIDTH_FRACS:
        lo, hi = base + stature * (f - 0.04), base + stature * (f + 0.04)
        sel = (h >= lo) & (h <= hi)
        out[f"w_{int(f * 100):02d}"] = (
            float(np.percentile(lat[sel], 97.5) - np.percentile(lat[sel], 2.5))
            if sel.sum() > 30 else 0.0)
    area = float(np.sum((z / FX) * (z / FY))) / 1e6                 # m^2
    out["surface_area_m2"] = area
    out["volume_proxy_l"] = area * out["depth_extent_mm"]           # m^2 * mm = litres

    # ---- HEAD-ANCHORED BAND FEATURES ---------------------------------------
    # The w_XX block above anchors at `base` -- the bottom of the VISIBLE body
    # -- and scales by `stature`. Both are corrupted the moment the feet leave
    # frame, which happens in 98 % of close-range frames: `base` becomes
    # mid-shin, `stature` shrinks, and w_45 lands on a different body part
    # depending only on how much got cut off. hiride_metric_bias.py measures the
    # consequence -- stature_mm drifting 89.5 mm per metre of standing distance
    # against a 105 mm between-subject SD, while w_15 and w_30, which sit near
    # the anchor and barely move with it, drift by ~0.
    #
    # These bands are measured DOWNWARD FROM THE CROWN in absolute millimetres.
    # The crown is visible far more often than the feet (top-touch 0.43 vs
    # bottom-touch 0.98 at close range), and an offset in mm is anatomically
    # stable however much of the legs is missing. `top_clip` already flags the
    # frames where even this anchor fails.
    crown = float(np.percentile(h, 99.5))
    for d in HEAD_OFFSETS_MM:
        sel = (h >= crown - d - BAND_HALF_MM) & (h <= crown - d + BAND_HALF_MM)
        if sel.sum() <= 30:
            out[f"hw_{d:03d}"] = out[f"ht_{d:03d}"] = 0.0
            out[f"hc_{d:03d}"] = out[f"ha_{d:03d}"] = 0.0
            continue
        w = float(np.percentile(lat[sel], 97.5) - np.percentile(lat[sel], 2.5))
        t = float(np.percentile(dep[sel], 97.5) - np.percentile(dep[sel], 2.5))
        out[f"hw_{d:03d}"] = w                       # width, mm
        out[f"ht_{d:03d}"] = t                       # thickness, mm -- NEW axis:
        # the old set had ONE global depth_extent_mm, so body thickness was a
        # single number for the whole person. Chest depth against waist depth is
        # a standard anthropometric discriminator and costs nothing here.
        a, b = w / 2.0, t / 2.0
        # Ramanujan's ellipse perimeter: the circumference an anthropometrist
        # would tape, from the two semi-axes the sensor can actually see.
        out[f"hc_{d:03d}"] = float(
            np.pi * (3 * (a + b) - np.sqrt(max((3 * a + b) * (a + 3 * b), 0.0))))
        # cross-section aspect. A RATIO, so it cancels any residual scale error
        # -- including whatever the crown anchor still gets wrong.
        out[f"ha_{d:03d}"] = float(t / w) if w > 1e-6 else 0.0
    for lo_d, hi_d in SHAPE_RATIOS:
        a_, b_ = out.get(f"hw_{lo_d:03d}", 0.0), out.get(f"hw_{hi_d:03d}", 0.0)
        out[f"r_{lo_d:03d}_{hi_d:03d}"] = float(a_ / b_) if b_ > 1e-6 else 0.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--root", required=True, help="staged BIWI tree ($SLURM_TMPDIR/biwi)")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=None, help="defaults to --prep")
    args = ap.parse_args()
    out_dir = args.out or args.prep

    ref = load_manifest(os.path.join(args.prep, "manifest.npz"))
    raw = build_manifest(args.root, verbose=True)
    key = lambda m, i: (str(m["seq"][i]), str(m["subject"][i]), int(m["frame"][i]))
    want = {key(ref, i): i for i in range(len(ref["frame"]))}

    names, rows, mrow = None, [], []
    n_ground = 0
    order = range(0, len(raw["frame"]), max(1, args.stride))
    for c, i in enumerate(order):
        k = key(raw, i)
        if k not in want:
            continue
        dpath = os.path.join(raw["root"], raw["depth"][i])
        try:
            depth, _ = read_pgm(dpath)
            user, _ = read_pgm(os.path.join(raw["root"], raw["user"][i]))
        except Exception:
            continue
        g = sibling(raw["depth"][i], "e", "groundCoeff", "txt")
        ground = read_ground(os.path.join(raw["root"], g)) if g else None
        f = frame_features(depth, user, ground)
        if f is None:
            continue
        n_ground += int(f.get("ground", 0) > 0)

        # _skel.txt IS NOT READ. Paper 1 is published, skeleton-only, and its
        # feature vector is 20-D bone-segment lengths; the territory decision of
        # 2026-08-19 puts _skel.txt off-limits to paper 2 ENTIRELY, including as
        # a pose covariate (Testing/Still provides pose control instead). This
        # is enforced here deliberately rather than left to the two parser bugs
        # that used to enforce it by accident -- see 13.16 for both, recorded
        # for whoever does own this material.

        if names is None:
            names = sorted(f)
        rows.append([f.get(n, 0.0) for n in names])
        mrow.append(want[k])
        if (c + 1) % 4000 == 0:
            print(f"[metric] {c + 1}/{len(raw['frame'])} scanned, {len(rows)} kept")

    F = np.array(rows, dtype=np.float32)
    np.savez_compressed(os.path.join(out_dir, "metric_features.npz"),
                        feats=F, names=np.array(names), manifest_row=np.array(mrow))
    print(f"\n[done] {F.shape[0]} frames x {F.shape[1]} features -> "
          f"{os.path.join(out_dir, 'metric_features.npz')}")
    print(f"  ground plane parsed on {n_ground}/{len(rows)} frames "
          f"({'USED' if n_ground else 'FELL BACK to camera-Y -- check the format log above'})")
    if F.shape[0]:
        for n in ("stature_mm", "w_45", "surface_area_m2", "volume_proxy_l", "stand_dist_mm"):
            if n in names:
                v = F[:, names.index(n)]
                v = v[v > 0]
                if v.size:
                    print(f"  {n:<18s} median {np.median(v):9.1f}  "
                          f"p5 {np.percentile(v, 5):9.1f}  p95 {np.percentile(v, 95):9.1f}")
        print("\n  Sanity: stature_mm should sit near 1500-1950 for standing adults. If it is")
        print("  wildly off, the ground plane sign or units are wrong -- see the format log.")


if __name__ == "__main__":
    main()
