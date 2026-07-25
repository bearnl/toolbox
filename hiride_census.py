"""Per-sequence census of the 13 cues: what changed between BIWI sessions?

    python hiride_census.py --prep $SCRATCH/hiride2/prep

hiride_adjacency.py found that a Training frame and a Testing/Walking frame of
the SAME subject differ by a median 1.3 m of depth over the whole frame -- as
much as two different subjects across those sequences -- while two frames of
different subjects inside Training differ by ~0.2 m. So R4_cross_session is not
only "different day + different clothes": the scene (camera placement / room)
moved too. This prints the per-sequence medians of the cues so that shift can
be stated in numbers (background depth, person depth, apparent size, image
position, valid fraction), for eligible frames and for all frames.

Login node, seconds; reads cues.npz + manifest.npz only.
"""
import os
import argparse

import numpy as np

from hiride_data import load_manifest, eligible_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    args = ap.parse_args()

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    z = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    cues, feats = z["cues"], [str(f) for f in z["feats"]]
    keep = eligible_mask(cues, feats)
    seqs = sorted(set(man["seq"].tolist()))

    show = ["bg_med", "p_med", "p_p1", "p_range", "n_person_px", "bbox_h", "bbox_w",
            "cent_y", "cent_x", "valid_frac", "top_touch", "bot_touch"]
    for label, sel_fn in (("ELIGIBLE frames", lambda m: m & keep), ("ALL frames", lambda m: m)):
        print(f"\n== per-sequence median of each cue, {label} ==")
        hdr = f"{'sequence':<17s}{'frames':>7s}{'subj':>5s}{'elig%':>7s} " + "".join(f"{f:>12s}" for f in show)
        print(hdr)
        print("-" * len(hdr))
        for s in seqs:
            m = man["seq"] == s
            sel = sel_fn(m)
            row = f"{s:<17s}{int(m.sum()):>7d}{len(set(man['subject'][m].tolist())):>5d}" \
                  f"{100 * keep[m].mean():>6.1f}% "
            for f in show:
                v = cues[sel, feats.index(f)]
                row += f"{np.median(v):>12.1f}" if v.size else f"{'-':>12s}"
            print(row)

    # Same-subject shift, Training -> Testing/Walking, on eligible frames only:
    # per subject, median cue in each sequence, then the paired difference.
    shared = sorted(set(man["subject"][man["seq"] == "Training"].tolist())
                    & set(man["subject"][man["seq"] == "Testing/Walking"].tolist()))
    if shared:
        print(f"\n== same-subject shift Training -> Testing/Walking, {len(shared)} shared subjects, "
              f"eligible frames: median over subjects of (median_B - median_A) ==")
        for f in ("bg_med", "p_med", "p_range", "n_person_px", "bbox_h", "cent_y", "cent_x", "valid_frac"):
            col = feats.index(f)
            d = []
            for sub in shared:
                a = cues[(man["seq"] == "Training") & (man["subject"] == sub) & keep, col]
                b = cues[(man["seq"] == "Testing/Walking") & (man["subject"] == sub) & keep, col]
                if a.size and b.size:
                    d.append(np.median(b) - np.median(a))
            d = np.array(d)
            print(f"  {f:<12s} Δ median={np.median(d):+9.1f}   IQR=[{np.percentile(d, 25):+.1f}, "
                  f"{np.percentile(d, 75):+.1f}]   n={d.size}")

    # Depth range sanity per sequence -- did any sequence use a different scale?
    print("\n== person-depth quantiles per sequence (p_med over eligible frames) ==")
    col = feats.index("p_med")
    for s in seqs:
        v = cues[(man["seq"] == s) & keep, col]
        if v.size:
            q = np.percentile(v, [1, 10, 50, 90, 99])
            print(f"  {s:<17s} p1={q[0]:6.0f}  p10={q[1]:6.0f}  p50={q[2]:6.0f}  p90={q[3]:6.0f}  p99={q[4]:6.0f} mm")


if __name__ == "__main__":
    main()
