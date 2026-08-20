#!/usr/bin/env python3
"""Does each corpus actually contain LOCOMOTION? A feasibility gate for gait.

Motion is measured as GROUND-PLANE displacement in millimetres -- the person's
median depth for the radial component, and their mask centroid column
unprojected at that depth for the lateral one. Radial distance alone is blind to
someone walking across the field of view at constant range, which would read as
a perfectly stationary subject.

    python hiride_motion.py --prep $SCRATCH/hiride2/prep

Gait is the one untested lever with a real chance of raising cross-session depth
identification (handoff 13.6 / the R3-R4 decomposition). But a gait feature is
undefined on a subject who is not walking, and it has to be defined on the
TRAINING pool, not just the test pool:

    R3  trains on Testing/Still   -> named "Still"
    R4  trains on Training        -> BIWI's head-pose corpus, subjects may be seated

If neither training pool contains walking, gait cannot be learned at either
recording-disjoint rung and the experiment is dead before it is written. Testing
that costs one pass over `cues.npz`, which already holds the person's median
depth per frame -- so radial motion is measurable without touching a shard.

Reports, per sequence and per recording: how far the person's standing distance
travels, and how fast. A static subject shows a few tens of mm of jitter; a
walker crosses metres.
"""
import os
import argparse
import numpy as np

from hiride_data import load_manifest

# A frame pair advancing more than this counts as motion. Kinect v1 depth
# quantisation is ~10-40 mm over 2-4 m, so anything below that is sensor
# noise on a stationary subject, not a step.
MOVE_MM = 20.0
# Frames a contiguous moving run must hold to contain a stride cycle.
MIN_RUN = 13
# Kinect v1 colour/depth intrinsics, as used by hiride_metric.py.
FX, CX = 575.816, 320.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--min-frames", type=int, default=30,
                    help="ignore recordings shorter than this")
    args = ap.parse_args()

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    z = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    cues, feats = z["cues"], [str(f) for f in z["feats"]]
    print(f"cue features available: {feats}\n")
    col = {f: i for i, f in enumerate(feats)}
    if "p_med" not in col:
        raise SystemExit(f"no p_med in cues.npz; have {feats}")

    p_med = cues[:, col["p_med"]].astype(np.float64)
    if "cent_x" not in col:
        raise SystemExit(f"no cent_x in cues.npz; have {feats}")
    cent_x = cues[:, col["cent_x"]].astype(np.float64)
    seq = np.asarray(man["seq"], dtype=str)
    subj = np.asarray(man["subject"], dtype=str)
    sess = np.asarray(man["session"], dtype=str)
    frame = np.asarray(man["frame"]).astype(np.int64)

    ts = np.asarray(man["ts"]).astype(np.float64)
    dts = np.diff(np.sort(ts[(seq == "Testing/Walking")]))
    dt = float(np.median(dts[dts > 0])) if (dts > 0).any() else 0.0
    if dt > 0:
        print(f"median inter-frame ts delta = {dt:g}; if ms, that is "
              f"{1000 / dt:.1f} fps, so a ~1.1 s stride cycle is ~{1.1 * 1000 / dt:.0f} frames")
    print(f"moving threshold = {MOVE_MM} mm/frame; a run must hold {MIN_RUN} frames\n")

    valid = np.isfinite(p_med) & (p_med > 0)
    rows = []
    for key in sorted(set(zip(seq, subj, sess))):
        m = (seq == key[0]) & (subj == key[1]) & (sess == key[2]) & valid
        if m.sum() < args.min_frames:
            continue
        order = np.argsort(frame[m])
        z_mm = p_med[m][order]
        # PLANAR displacement, not radial. p_med alone is blind to a subject
        # walking ACROSS the field of view at constant distance -- that reads as
        # a 0.0 mm step while the person is plainly walking. cent_x is a pixel
        # column, so unproject it at the person's own depth before differencing:
        # x_mm = (cent_x - cx) * z / fx. Ground-plane motion is then hypot(dx, dz).
        x_mm = (cent_x[m][order] - CX) * z_mm / FX
        d = np.hypot(np.diff(x_mm), np.diff(z_mm))
        # SPAN ALONE DOES NOT MEAN WALKING. A subject who stands at 2 m, is
        # repositioned, and stands at 3.3 m accumulates 1300 mm of span without
        # ever taking a stride -- which is exactly what Training's 1347 mm span
        # next to a 0.0 mm median step describes. What a gait feature needs is a
        # CONTIGUOUS RUN of frames that are all moving, long enough to hold a
        # stride cycle. Measure that directly.
        moving = d > MOVE_MM
        best = cur = 0
        for mv in moving:
            cur = cur + 1 if mv else 0
            best = max(best, cur)
        rows.append(dict(seq=key[0], subject=key[1], session=key[2], n=int(m.sum()),
                         span=float(np.percentile(z_mm, 95) - np.percentile(z_mm, 5)),
                         step=float(np.median(d)), p90=float(np.percentile(d, 90)),
                         moving=float(moving.mean()), run=int(best)))

    print(f"{len(rows)} recordings with >= {args.min_frames} usable frames\n")
    hdr = (f"{'sequence':<18s}{'recs':>5s}{'frames':>8s}{'span p50':>10s}"
           f"{'step p50':>10s}{'step p90':>10s}{'moving':>8s}{'run p50':>9s}{'usable':>8s}")
    print(hdr); print("-" * len(hdr))
    for s in sorted(set(r["seq"] for r in rows)):
        g = [r for r in rows if r["seq"] == s]
        arr = lambda k: np.array([r[k] for r in g])
        # "usable for gait" = holds a contiguous moving run of at least one
        # stride cycle. Span is deliberately NOT part of this test.
        usable = int((arr("run") >= MIN_RUN).sum())
        print(f"{s:<18s}{len(g):>5d}{sum(r['n'] for r in g):>8d}"
              f"{np.median(arr('span')):>9.0f}mm"
              f"{np.median(arr('step')):>9.1f}mm{np.median(arr('p90')):>9.1f}mm"
              f"{100 * np.median(arr('moving')):>7.1f}%{np.median(arr('run')):>9.0f}"
              f"{usable:>5d}/{len(g):<3d}")

    print("\nper-recording detail (largest travel first, top 12):")
    for r in sorted(rows, key=lambda r: -r["span"])[:12]:
        print(f"  {r['seq']:<18s} subj {r['subject']} sess {r['session']:<10s} "
              f"n={r['n']:<5d} span={r['span']:7.0f} mm  step p50={r['step']:5.1f} mm")
    print("\nsmallest travel (bottom 6):")
    for r in sorted(rows, key=lambda r: r["span"])[:6]:
        print(f"  {r['seq']:<18s} subj {r['subject']} sess {r['session']:<10s} "
              f"n={r['n']:<5d} span={r['span']:7.0f} mm  step p50={r['step']:5.1f} mm")

    print(f"\nREAD: `moving` is the share of consecutive frame pairs that advance more "
          f"than {MOVE_MM} mm; `run p50` is the longest UNBROKEN moving stretch, in frames. "
          f"\nGait needs a stride cycle inside one unbroken run -- {MIN_RUN} frames at the "
          f"rate above. A pool with large span but a near-zero moving share is a subject "
          f"\nbeing REPOSITIONED between static poses, not walking, and no gait feature "
          f"can be learned from it.")


if __name__ == "__main__":
    main()
