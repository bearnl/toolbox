"""Metric 3D + skeleton features: use depth AS GEOMETRY, not as a grey image.

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
    bone_*                skeleton segment lengths, if _skel.txt parses

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
_LOGGED = {"ground": False, "skel": False}


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


def read_skel(path):
    v, txt = read_floats(path)
    if not _LOGGED["skel"]:
        _LOGGED["skel"] = True
        print(f"[format] skel first file: {os.path.basename(path)}\n"
              f"         raw head: {(txt or '')[:300]!r}\n"
              f"         parsed {0 if v is None else len(v)} floats")
    if v is None or len(v) < 9:
        return None
    for stride in (3, 4, 9, 12):                 # xyz / xyz+conf / +rotation
        if len(v) % stride == 0 and len(v) // stride >= 3:
            pts = v.reshape(-1, stride)[:, :3]
            if np.isfinite(pts).all() and np.abs(pts).max() < 1e5:
                return pts
    return None


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
    bone_names, n_ground, n_skel = None, 0, 0
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

        s = sibling(raw["depth"][i], "d", "skel", "txt")
        pts = read_skel(os.path.join(raw["root"], s)) if s else None
        if pts is not None and len(pts) >= 4:
            n_skel += 1
            k_j = min(len(pts), 15)
            d = []
            nm = []
            for a in range(k_j):
                for b in range(a + 1, k_j):
                    d.append(float(np.linalg.norm(pts[a] - pts[b])))
                    nm.append(f"bone_{a:02d}_{b:02d}")
            if bone_names is None:
                bone_names = nm
            if len(d) == len(bone_names):
                f.update(dict(zip(bone_names, d)))

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
    print(f"  skeleton parsed on {n_skel}/{len(rows)} frames"
          + (f", {len(bone_names)} segment lengths" if bone_names else " -- no skeleton features"))
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
