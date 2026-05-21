#!/usr/bin/env python3
"""Step-0 diagnostic: how much identity is carried by depth-derived METRIC
anthropometry on BIWI — with NO training at all.

Motivation
----------
Our CNN preprocessing crops each person and resizes-to-fill, which erases
absolute body scale (height/width). Human whole-body recognition at a distance
relies precisely on metric anthropometry: real-world height, segment widths,
body thickness, proportions. This script measures the *raw* discriminative
power of those measurements, extracted directly from the depth foreground via
camera intrinsics (NO skeleton, NO RGB, NO learning), so we know whether a
metric-preserving pipeline is worth building.

Protocols (28 BIWI test subjects)
---------------------------------
  within-Still   : gallery/probe both from Testing/Still (same session, same
                   clothing, ~same pose) — an easy sanity ceiling.
  cross-pose     : gallery = Testing/Still,  probe = Testing/Walking
                   (same clothing, different pose).
  cross-clothing : gallery = Training/ (outfit A),  probe = Testing/ (outfit B)
                   (different day + different clothing; the hard regime).

For each: prototype nearest-neighbour matching (one mean metric vector per
gallery subject), reported frame-level and sequence-averaged. Also reports
which features carry the signal (between/within-subject variance ratio), and
an ablation: all features vs height-only vs shape-only (height excluded).

Random R1 baseline = 100 / n_subjects.

Usage
-----
  python anthro_probe.py --base-dir /path/to/biwi   # expects Training/ + Testing/
"""
import os
import argparse
import numpy as np
import cv2

# Kinect v1 depth intrinsics at 640x480 (Munaro 2014). Override via CLI.
FX = FY = 575.816
CX, CY = 320.0, 240.0

# Foreground slab. Generous (vs the 300mm training clip) so we capture the FULL
# body for measurement — the diagnostic is about true body extent, not the
# high-resolution front surface the CNN trains on.
FG_CLIP_MM = 1500.0
FG_PERCENTILE = 1.0          # anchor = this percentile of valid depth (nearest)
MIN_FG_PIXELS = 200
FULL_BODY_ONLY = False       # if True, reject frames whose foreground touches the
                             # top/bottom image edge (head/feet clipped by the FOV)
EDGE_MARGIN = 2              # rows within this of the top/bottom edge count as "touching"
PLAUSIBLE_HEIGHT_MM = (900.0, 2400.0)  # discard frames whose measured height is implausible

# Relative heights at which we measure width (lateral) and thickness (front-back).
BAND_FRACS = [0.2, 0.35, 0.5, 0.65, 0.8]
FEATURE_NAMES = (
    ["height"]
    + [f"width_{int(f*100)}" for f in BAND_FRACS]
    + [f"thick_{int(f*100)}" for f in BAND_FRACS]
    + ["volume"]
)


def _scan_biwi(root):
    """Group depth files by subject id (leading digits of filename) under root.
    Returns {subj_id: [paths...]} and also keeps each path so callers can split
    on 'still'/'walking' tokens."""
    by_subj = {}
    if not os.path.isdir(root):
        return by_subj
    for dirpath, _dirs, files in os.walk(root):
        for f in files:
            fl = f.lower()
            if not (fl.endswith("_depth.pgm") or fl.endswith("_depth.png")):
                continue
            head = f.split("_", 1)[0]
            digits = ""
            for c in head:
                if c.isdigit():
                    digits += c
                else:
                    break
            if not digits:
                continue
            by_subj.setdefault(digits, []).append(os.path.join(dirpath, f))
    for s in by_subj:
        by_subj[s].sort()
    return by_subj


def _load_depth_mm(path):
    img = cv2.imread(path, cv2.IMREAD_ANYDEPTH)
    if img is None:
        return None
    return img.astype(np.float32)


def _foreground(depth_mm, reject_edge=False):
    """Foreground slab + largest connected component (removes detached
    background blobs like walls). Returns bool mask or None.

    If reject_edge, also returns None when the foreground touches the top or
    bottom image edge (head/feet clipped by the field of view) — used by the
    --full-body-only filter so anthropometry is only measured on un-clipped bodies."""
    valid = depth_mm > 0
    if int(valid.sum()) < MIN_FG_PIXELS:
        return None
    anchor = np.percentile(depth_mm[valid], FG_PERCENTILE)
    fg = valid & (depth_mm >= anchor) & (depth_mm <= anchor + FG_CLIP_MM)
    if int(fg.sum()) < MIN_FG_PIXELS:
        return None
    n_lab, labels = cv2.connectedComponents(fg.astype(np.uint8))
    if n_lab > 2:
        counts = np.bincount(labels.ravel())
        counts[0] = 0  # ignore background label
        fg = labels == int(np.argmax(counts))
    if int(fg.sum()) < MIN_FG_PIXELS:
        return None
    if reject_edge:
        rows = np.where(fg.any(axis=1))[0]
        Hh = fg.shape[0]
        if rows.min() <= EDGE_MARGIN or rows.max() >= Hh - 1 - EDGE_MARGIN:
            return None
    return fg


def _unproject(depth_mm, fg):
    """Foreground pixels → (N,3) point cloud in CAMERA coords (mm)."""
    ys, xs = np.where(fg)
    zs = depth_mm[ys, xs].astype(np.float64)
    Xs = zs * (xs.astype(np.float64) - CX) / FX
    Ys = zs * (ys.astype(np.float64) - CY) / FY
    return np.stack([Xs, Ys, zs], axis=1)


def _extents_along(axis0_vals, axis1_vals, axis2_vals):
    """Given per-point coordinates along (vertical, lateral, depth) axes, measure
    robust height + width/thickness profiles at BAND_FRACS along the vertical axis.
    Returns the 12-vector or None if degenerate."""
    v0, v98 = np.percentile(axis0_vals, [2, 98])
    height = float(v98 - v0)
    if not (PLAUSIBLE_HEIGHT_MM[0] <= height <= PLAUSIBLE_HEIGHT_MM[1]):
        return None
    widths, thicks = [], []
    band_half = 0.06 * height
    for fr in BAND_FRACS:
        c = v0 + fr * height
        m = np.abs(axis0_vals - c) < band_half
        if int(m.sum()) >= 20:
            widths.append(float(np.percentile(axis1_vals[m], 98) - np.percentile(axis1_vals[m], 2)))
            thicks.append(float(np.percentile(axis2_vals[m], 98) - np.percentile(axis2_vals[m], 2)))
        else:
            widths.append(0.0)
            thicks.append(0.0)
    volume = float(np.mean([widths[i] * thicks[i] for i in range(len(BAND_FRACS))]) * height)
    return np.array([height] + widths + thicks + [volume], dtype=np.float64)


def _metric_features_camera(depth_mm, fg):
    """CAMERA-frame anthropometry: vertical=camera-Y, lateral=camera-X,
    depth=camera-Z. Viewpoint-DEPENDENT (changes if camera tilts/rotates)."""
    pts = _unproject(depth_mm, fg)
    if len(pts) < MIN_FG_PIXELS:
        return None
    return _extents_along(pts[:, 1], pts[:, 0], pts[:, 2])  # Y, X, Z


def _canonical_extents(pts):
    """AZIMUTH-INVARIANT anthropometry. People stand upright, so the body's
    vertical ≈ gravity ≈ camera-Y; the dominant viewpoint difference between
    rooms/cameras is AZIMUTH (which way the person faces) — a rotation about the
    vertical. So we:
      - keep height = camera-Y extent (gravity-aligned, stable, sane frame gate),
      - canonicalize only the horizontal plane: 2D PCA on (X, Z) gives the
        shoulder-width axis (max horizontal variance) and the front-back
        thickness axis, recovered regardless of azimuth.
    This removes the main viewpoint variable without letting the vertical axis be
    hijacked by the depth-direction spread (the bug that free 3D PCA had).
    Residual: camera TILT and self-occlusion are not corrected — measured.
    Returns None only on the same height-plausibility gate as the camera frame,
    so it stays row-aligned with the camera features."""
    if len(pts) < MIN_FG_PIXELS:
        return None
    Y = pts[:, 1]
    XZ = pts[:, [0, 2]].astype(np.float64)
    XZ = XZ - XZ.mean(axis=0)
    try:
        evals, evecs = np.linalg.eigh(np.cov(XZ.T))
        order = np.argsort(evals)[::-1]      # [width axis, thickness axis]
        H = XZ @ evecs[:, order]
        w_axis, t_axis = H[:, 0], H[:, 1]
    except (np.linalg.LinAlgError, ValueError):
        w_axis, t_axis = XZ[:, 0], XZ[:, 1]  # fallback: raw camera X, Z
    return _extents_along(Y, w_axis, t_axis)


def _collect(paths, labels):
    """Compute BOTH camera-frame and canonical features for every path.
    Returns (cam[N,12], canon[N,12], labels[N], n_attempted, n_ok). A frame is
    'ok' only if both feature sets succeed (so the two are row-aligned)."""
    cam, canon, labs = [], [], []
    for p, lab in zip(paths, labels):
        d = _load_depth_mm(p)
        if d is None:
            continue
        fg = _foreground(d, reject_edge=FULL_BODY_ONLY)
        if fg is None:
            continue
        pts = _unproject(d, fg)
        vc = _extents_along(pts[:, 1], pts[:, 0], pts[:, 2])
        vk = _canonical_extents(pts)
        if vc is None or vk is None:
            continue
        cam.append(vc)
        canon.append(vk)
        labs.append(lab)
    if not cam:
        z = np.zeros((0, len(FEATURE_NAMES)))
        return z, z, np.array([]), len(paths), 0
    return np.array(cam), np.array(canon), np.array(labs), len(paths), len(cam)


def _rank_metrics(D, probe_labels, proto_labels):
    order = np.argsort(D, axis=1)
    r1 = r5 = 0
    aps = []
    for i in range(len(probe_labels)):
        matches = (proto_labels[order[i]] == probe_labels[i])
        if matches[0]:
            r1 += 1
        if matches[:5].any():
            r5 += 1
        ranks = np.where(matches)[0] + 1  # 1-indexed
        if len(ranks):
            aps.append(float(np.mean([(k + 1) / ranks[k] for k in range(len(ranks))])))
    n = len(probe_labels)
    return (100.0 * r1 / n, 100.0 * r5 / n, 100.0 * float(np.mean(aps)) if aps else 0.0)


def _match(gal_feats, gal_labels, prb_feats, prb_labels, feat_idx, seq=False):
    """z-score features (gallery stats), build per-subject prototypes, NN match."""
    G = gal_feats[:, feat_idx]
    P = prb_feats[:, feat_idx]
    mu = G.mean(axis=0)
    sd = G.std(axis=0) + 1e-6
    G = (G - mu) / sd
    P = (P - mu) / sd
    uniq = sorted(set(gal_labels.tolist()))
    proto = np.stack([G[gal_labels == s].mean(axis=0) for s in uniq])
    proto_labels = np.array(uniq)
    if seq:
        puniq = sorted(set(prb_labels.tolist()))
        P = np.stack([P[prb_labels == s].mean(axis=0) for s in puniq])
        prb_labels = np.array(puniq)
    # Euclidean distance matrix (P x proto) via (a-b)^2 expansion.
    D = (np.sum(P**2, axis=1, keepdims=True)
         - 2.0 * P @ proto.T
         + np.sum(proto**2, axis=1)[None, :])
    return _rank_metrics(D, prb_labels, proto_labels)


def _f_ratios(feats, labels):
    """Between-subject / within-subject variance ratio per feature (Fisher-like)."""
    uniq = sorted(set(labels.tolist()))
    overall = feats.mean(axis=0)
    n_feat = feats.shape[1]
    between = np.zeros(n_feat)
    within = np.zeros(n_feat)
    N = len(labels)
    for s in uniq:
        fs = feats[labels == s]
        ns = len(fs)
        ms = fs.mean(axis=0)
        between += ns * (ms - overall) ** 2
        within += ((fs - ms) ** 2).sum(axis=0)
    between /= max(len(uniq) - 1, 1)
    within /= max(N - len(uniq), 1)
    return between / (within + 1e-9)


def _report(name, gal, prb):
    """gal/prb are 5-tuples (cam[N,12], canon[N,12], labels[N], n_att, n_ok).
    Reports matching for BOTH camera-frame and canonical (viewpoint-invariant)
    features, side by side."""
    g_cam, g_canon, gl, ga, gok = gal
    p_cam, p_canon, pl, pa, pok = prb
    print(f"\n{'='*78}\n{name}\n{'='*78}")
    print(f"  gallery frames ok: {gok}/{ga}   probe frames ok: {pok}/{pa}")
    common = sorted(set(gl.tolist()) & set(pl.tolist()))
    print(f"  subjects (gallery∩probe): {len(common)}   random R1 = {100.0/max(len(common),1):.2f}%")
    if len(common) < 2 or gok == 0 or pok == 0:
        print("  (insufficient data — skipped)")
        return
    gm = np.isin(gl, common)
    pm = np.isin(pl, common)
    gl2, pl2 = gl[gm], pl[pm]

    idx_all = list(range(len(FEATURE_NAMES)))
    idx_height = [0]
    idx_shape = [i for i in idx_all if i != 0]

    for frame_tag, gf, pf in [("CAMERA frame (viewpoint-dependent)", g_cam[gm], p_cam[pm]),
                              ("CANONICAL frame (PCA, viewpoint-invariant)", g_canon[gm], p_canon[pm])]:
        print(f"\n  --- {frame_tag} ---")
        print(f"  {'feature set':<16}{'frame R1':>10}{'frame R5':>10}{'frame mAP':>11}"
              f"{'seq R1':>9}{'seq R5':>9}{'seq mAP':>9}")
        for tag, idx in [("all (12)", idx_all), ("height-only", idx_height), ("shape-only", idx_shape)]:
            fr = _match(gf, gl2, pf, pl2, idx, seq=False)
            sq = _match(gf, gl2, pf, pl2, idx, seq=True)
            print(f"  {tag:<16}{fr[0]:>9.2f}{fr[1]:>10.2f}{fr[2]:>11.2f}"
                  f"{sq[0]:>9.2f}{sq[1]:>9.2f}{sq[2]:>9.2f}")


def _consistency(name, gal, prb):
    """Per-feature cross-session consistency: Pearson r between each subject's
    gallery-mean and probe-mean, for camera vs canonical frames. High r ⇒ the
    measurement transfers across sessions; near 0 ⇒ it doesn't. This is the
    decisive test of whether canonicalization fixes the cross-session shift."""
    g_cam, g_canon, gl, _, gok = gal
    p_cam, p_canon, pl, _, pok = prb
    common = sorted(set(gl.tolist()) & set(pl.tolist()))
    print(f"\n{'-'*78}\n{name}: per-feature A-vs-B consistency (Pearson r over {len(common)} subjects)\n{'-'*78}")
    if len(common) < 3 or gok == 0 or pok == 0:
        print("  (insufficient data)")
        return
    print(f"  {'feature':<10}{'camera r':>12}{'canonical r':>14}")
    for fi, nm in enumerate(FEATURE_NAMES):
        def subj_means(feats, labs):
            return np.array([feats[labs == s, fi].mean() for s in common])
        rc = np.corrcoef(subj_means(g_cam, gl), subj_means(p_cam, pl))[0, 1]
        rk = np.corrcoef(subj_means(g_canon, gl), subj_means(p_canon, pl))[0, 1]
        print(f"  {nm:<10}{rc:>12.3f}{rk:>14.3f}")


def _audit_frame(depth_mm):
    """Lightweight per-frame audit: (height_mm, dist_mm, top_touch, bottom_touch).
    NO plausibility gate — we WANT to see truncated/implausible frames here, since
    partial-body capture (head/feet cut by the image edge) is the prime suspect for
    why height fails to transfer across sessions. Returns None only if no foreground."""
    fg = _foreground(depth_mm)
    if fg is None:
        return None
    pts = _unproject(depth_mm, fg)
    if len(pts) < MIN_FG_PIXELS:
        return None
    Y, Z = pts[:, 1], pts[:, 2]
    height = float(np.percentile(Y, 98) - np.percentile(Y, 2))
    dist = float(np.median(Z))
    rows = np.where(fg.any(axis=1))[0]
    H = fg.shape[0]
    top_touch = int(rows.min() <= 1)        # head at top image edge → likely cut
    bottom_touch = int(rows.max() >= H - 2)  # feet at bottom edge → likely cut
    return height, dist, top_touch, bottom_touch


def _audit_pool(by_subj, ids, stride=2):
    """Per-subject aggregates of the frame audit. Returns {subj: dict(...)}."""
    out = {}
    for s in ids:
        hs, ds, tt, bt = [], [], [], []
        for p in by_subj.get(s, [])[::stride]:
            d = _load_depth_mm(p)
            if d is None:
                continue
            a = _audit_frame(d)
            if a is None:
                continue
            hs.append(a[0]); ds.append(a[1]); tt.append(a[2]); bt.append(a[3])
        if hs:
            out[s] = dict(h=float(np.mean(hs)), h_sd=float(np.std(hs)),
                          dist=float(np.mean(ds)), top=100.0 * np.mean(tt),
                          bot=100.0 * np.mean(bt), n=len(hs))
    return out


def _spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def _height_audit(train_by, test_by, xcloth_ids):
    print(f"\n{'='*78}\nHEIGHT AUDIT — same person, Training (outfit A) vs Testing (outfit B)\n{'='*78}")
    A = _audit_pool(train_by, xcloth_ids)
    B = _audit_pool(test_by, xcloth_ids)
    common = sorted(set(A) & set(B))
    if len(common) < 3:
        print("  (insufficient data)")
        return
    print(f"  {'subj':<6}{'A_height':>9}{'B_height':>9}{'Δmm':>7}"
          f"{'A_dist':>8}{'B_dist':>8}{'A_topcut%':>10}{'A_botcut%':>10}{'B_topcut%':>10}{'B_botcut%':>10}")
    ha, hb = [], []
    for s in common:
        a, b = A[s], B[s]
        ha.append(a["h"]); hb.append(b["h"])
        print(f"  {s:<6}{a['h']:>9.0f}{b['h']:>9.0f}{b['h']-a['h']:>7.0f}"
              f"{a['dist']:>8.0f}{b['dist']:>8.0f}{a['top']:>10.0f}{a['bot']:>10.0f}{b['top']:>10.0f}{b['bot']:>10.0f}")
    ha, hb = np.array(ha), np.array(hb)
    pear = float(np.corrcoef(ha, hb)[0, 1])
    spear = _spearman(ha, hb)
    print(f"\n  per-subject mean height:  Pearson r = {pear:+.3f}   Spearman(rank) r = {spear:+.3f}")
    print(f"  A height: mean {ha.mean():.0f} ± {ha.std():.0f} mm   B height: mean {hb.mean():.0f} ± {hb.std():.0f} mm")
    print(f"  mean |Δheight| same person across sessions: {np.mean(np.abs(hb-ha)):.0f} mm")
    a_top = np.mean([A[s]['top'] for s in common]); a_bot = np.mean([A[s]['bot'] for s in common])
    b_top = np.mean([B[s]['top'] for s in common]); b_bot = np.mean([B[s]['bot'] for s in common])
    print(f"  edge-touch rates — A: head {a_top:.0f}% / feet {a_bot:.0f}%   "
          f"B: head {b_top:.0f}% / feet {b_bot:.0f}%   (high ⇒ body cut off ⇒ height truncated)")
    print("\n  READ: Spearman≈0 ⇒ identity ordering genuinely lost (not a scaling issue).")
    print("        High edge-touch and/or large |Δheight| with closer B_dist ⇒ partial-body")
    print("        capture artifact (potentially fixable). Offset+high-Spearman ⇒ calibration.")


def main():
    global FX, FY, CX, CY, FG_CLIP_MM, FULL_BODY_ONLY, EDGE_MARGIN
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-dir", required=True,
                    help="BIWI root containing Training/ and Testing/.")
    ap.add_argument("--train-subdir", default="Training")
    ap.add_argument("--test-subdir", default="Testing")
    ap.add_argument("--fx", type=float, default=FX)
    ap.add_argument("--fy", type=float, default=FY)
    ap.add_argument("--cx", type=float, default=CX)
    ap.add_argument("--cy", type=float, default=CY)
    ap.add_argument("--clip-mm", type=float, default=FG_CLIP_MM)
    ap.add_argument("--full-body-only", action="store_true",
                    help="Reject frames whose foreground touches the top/bottom image "
                         "edge (head/feet clipped by FOV). Anthropometry is then measured "
                         "only on un-clipped bodies. NOTE: BIWI Training may have ~0 such "
                         "frames — the survival counts will show this.")
    ap.add_argument("--edge-margin", type=int, default=EDGE_MARGIN,
                    help="Rows within this distance of the top/bottom edge count as touching.")
    args = ap.parse_args()

    FX, FY, CX, CY, FG_CLIP_MM = args.fx, args.fy, args.cx, args.cy, args.clip_mm
    FULL_BODY_ONLY = args.full_body_only
    EDGE_MARGIN = args.edge_margin

    train_root = os.path.join(args.base_dir, args.train_subdir)
    test_root = os.path.join(args.base_dir, args.test_subdir)
    print(f"Scanning train: {train_root}")
    train_by = _scan_biwi(train_root)
    print(f"Scanning test:  {test_root}")
    test_by = _scan_biwi(test_root)
    print(f"  train subjects: {len(train_by)}   test subjects: {len(test_by)}")
    print(f"  intrinsics fx={FX} fy={FY} cx={CX} cy={CY}   fg_clip={FG_CLIP_MM}mm")
    print(f"  full_body_only={FULL_BODY_ONLY} (edge_margin={EDGE_MARGIN}px)"
          + ("   ← measuring only un-clipped bodies; watch the 'frames ok' survival counts"
             if FULL_BODY_ONLY else ""))

    # Build path/label lists for each pool.
    def pool(by_subj, ids, token=None):
        paths, labels = [], []
        for s in ids:
            for p in by_subj.get(s, []):
                if token is None or token in p.lower():
                    paths.append(p)
                    labels.append(s)
        return paths, labels

    test_ids = sorted(test_by.keys())

    # Testing split into Still / Walking by path token.
    still_paths, still_labels = pool(test_by, test_ids, token="still")
    walk_paths, walk_labels = pool(test_by, test_ids, token="walking")
    have_split = len(still_paths) > 0 and len(walk_paths) > 0
    if not have_split:
        print("\n  NOTE: could not split Testing on 'still'/'walking' tokens; "
              "cross-pose uses an even/odd frame split of all Testing frames instead.")
        all_test_paths, all_test_labels = pool(test_by, test_ids)

    def slice_pool(p, idx):
        """Slice a 5-tuple pool (cam, canon, labels, n_att, n_ok) by row indices."""
        return (p[0][idx], p[1][idx], p[2][idx], len(idx), len(idx))

    print("\nComputing metric features (this reads every depth frame once)...")
    if have_split:
        still = _collect(still_paths, still_labels)
        walk = _collect(walk_paths, walk_labels)
        # within-Still sanity: split each subject's COLLECTED Still frames in half.
        slab = still[2]
        idx_by = {}
        for i, l in enumerate(slab.tolist()):
            idx_by.setdefault(l, []).append(i)
        g_idx, p_idx = [], []
        for l, ids in idx_by.items():
            half = max(1, len(ids) // 2)
            g_idx += ids[:half]
            p_idx += ids[half:] if len(ids) > 1 else ids[:half]
        _report("SANITY  within-Still (gallery=Still[:half], probe=Still[half:])",
                slice_pool(still, g_idx), slice_pool(still, p_idx))
        _report("CROSS-POSE  gallery=Still, probe=Walking (same clothing)", still, walk)
    else:
        allt = _collect(all_test_paths, all_test_labels)
        n = allt[4]
        ev_idx = list(range(0, n, 2))
        od_idx = list(range(1, n, 2))
        _report("CROSS-POSE (even/odd split of Testing)",
                slice_pool(allt, ev_idx), slice_pool(allt, od_idx))

    # Cross-clothing: gallery = test subjects' Training frames (outfit A),
    #                 probe   = their Testing frames (outfit B).
    xcloth_ids = sorted(set(test_ids) & set(train_by.keys()))
    galA_paths, galA_labels = pool(train_by, xcloth_ids)
    probeB_paths, probeB_labels = pool(test_by, xcloth_ids)
    galA = _collect(galA_paths, galA_labels)
    probeB = _collect(probeB_paths, probeB_labels)
    _report("CROSS-CLOTHING  gallery=Training (outfit A), probe=Testing (outfit B)", galA, probeB)
    _consistency("CROSS-CLOTHING", galA, probeB)
    _height_audit(train_by, test_by, xcloth_ids)

    print("\nDone.")


if __name__ == "__main__":
    main()
