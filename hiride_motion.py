#!/usr/bin/env python3
"""Does each corpus actually contain LOCOMOTION? A feasibility gate for gait.

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
    seq = np.asarray(man["seq"], dtype=str)
    subj = np.asarray(man["subject"], dtype=str)
    sess = np.asarray(man["session"], dtype=str)
    frame = np.asarray(man["frame"]).astype(np.int64)

    valid = np.isfinite(p_med) & (p_med > 0)
    rows = []
    for key in sorted(set(zip(seq, subj, sess))):
        m = (seq == key[0]) & (subj == key[1]) & (sess == key[2]) & valid
        if m.sum() < args.min_frames:
            continue
        order = np.argsort(frame[m])
        z_mm = p_med[m][order]
        d = np.abs(np.diff(z_mm))
        rows.append(dict(seq=key[0], subject=key[1], session=key[2], n=int(m.sum()),
                         span=float(np.percentile(z_mm, 95) - np.percentile(z_mm, 5)),
                         step=float(np.median(d)), p90=float(np.percentile(d, 90))))

    print(f"{len(rows)} recordings with >= {args.min_frames} usable frames\n")
    hdr = (f"{'sequence':<18s}{'recs':>5s}{'frames':>8s}"
           f"{'span p50':>10s}{'span p90':>10s}{'step p50':>10s}{'walkers':>9s}")
    print(hdr); print("-" * len(hdr))
    for s in sorted(set(r["seq"] for r in rows)):
        g = [r for r in rows if r["seq"] == s]
        span = np.array([r["span"] for r in g])
        step = np.array([r["step"] for r in g])
        # a metre of travel is locomotion; tens of mm is a standing subject swaying
        walk = int((span > 1000).sum())
        print(f"{s:<18s}{len(g):>5d}{sum(r['n'] for r in g):>8d}"
              f"{np.median(span):>9.0f}mm{np.percentile(span, 90):>9.0f}mm"
              f"{np.median(step):>9.1f}mm{walk:>6d}/{len(g):<3d}")

    print("\nper-recording detail (largest travel first, top 12):")
    for r in sorted(rows, key=lambda r: -r["span"])[:12]:
        print(f"  {r['seq']:<18s} subj {r['subject']} sess {r['session']:<10s} "
              f"n={r['n']:<5d} span={r['span']:7.0f} mm  step p50={r['step']:5.1f} mm")
    print("\nsmallest travel (bottom 6):")
    for r in sorted(rows, key=lambda r: r["span"])[:6]:
        print(f"  {r['seq']:<18s} subj {r['subject']} sess {r['session']:<10s} "
              f"n={r['n']:<5d} span={r['span']:7.0f} mm  step p50={r['step']:5.1f} mm")

    print("\nREAD: a training pool whose recordings sit near a few tens of mm of span "
          "contains no locomotion, so no gait feature can be LEARNED there -- "
          "whatever the test pool does.")


if __name__ == "__main__":
    main()
