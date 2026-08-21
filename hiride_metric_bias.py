#!/usr/bin/env python3
"""Do the "metric" features actually measure the body, or partly the range?

    python hiride_metric_bias.py --prep $SCRATCH/hiride2/prep

hiride_metric.py's whole claim is UNITS: millimetres in a gravity-aligned frame
cancel camera pose by construction, where pixel counts do not. `stature_mm`
should therefore read the same whether a person stands at 2 m or 4 m.

Two results say it may not. Averaging features over a window DESTROYS accuracy
(18.83 % -> 5.37 % at 100 frames) while averaging posteriors over the same
frames IMPROVES it -- and the test subjects walk across 2 m while the training
subjects barely move, so a test window mixes ranges a training window never
does. And the range profile already shows probe accuracy varying 5.97-23.13 %
across depth bins.

If a feature drifts with standing distance, then part of what the classifier
calls "person 023" is "person 023 at 2.6 m", which is a nuisance of exactly the
kind this campaign exists to remove -- and one that a correction could remove
cheaply, improving every number that rests on these features.

Reports per feature:
  between-SD   spread of per-subject means. THE IDENTITY SIGNAL.
  within-SD    spread around a subject's own mean. The noise it competes with.
  dist-R2      fraction of that within-subject variance explained by standing
               distance alone -- i.e. how much of the "noise" is really a
               correctable bias.
  slope        within-subject drift, in feature units per metre.
  recoverable  how far between/within would improve if the drift were removed.
"""
import os
import argparse
import numpy as np

from hiride_data import load_manifest
from hiride_fuse import load_metric
from hiride_metric import NUISANCE, SHAPE_PREFIXES


# Testing/Walking spans 1585-3800 mm, so a feature's drift across the walk is
# |slope| * this. Divided by between-subject SD it gives one number for "does
# this measure the person or the range" -- below ~0.5 is usable, above 1 means
# the range moves it further than people differ.
TEST_RANGE_M = 2.2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--min-frames", type=int, default=40)
    args = ap.parse_args()

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    full, have, cols, keep = load_metric(args.prep, man)
    z = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    feats = [str(f) for f in z["feats"]]
    p_med = z["cues"][:, feats.index("p_med")].astype(np.float64)
    mf = np.load(os.path.join(args.prep, "metric_features.npz"), allow_pickle=False)
    names = [str(n) for n in mf["names"]]
    # Analyse EVERY measurable column, not the pinned publication set.
    # load_metric returns BASE_METRIC so the published numbers keep reproducing;
    # a diagnostic asking "which features are range-invariant" must see the new
    # candidates too, or it can never rank them. Nuisance covariates are
    # excluded because their range dependence is the point of them.
    cols = [i for i, n in enumerate(names)
            if n not in NUISANCE and not n.startswith("bone_")]
    print(f"[bias] analysing {len(cols)} of {len(names)} columns "
          f"({sum(n.startswith(SHAPE_PREFIXES) for n in names)} crown-anchored)")
    seq = np.asarray(man["seq"], dtype=str)
    subj = np.asarray(man["subject"], dtype=str)

    print("\nstanding-distance distribution per corpus (mm):")
    print(f"{'sequence':<20s}{'p05':>8s}{'p50':>8s}{'p95':>8s}{'IQR':>8s}")
    for s in sorted(set(seq)):
        m = keep & (seq == s) & np.isfinite(p_med) & (p_med > 0)
        if m.sum() < 100:
            continue
        q = np.percentile(p_med[m], [5, 50, 95, 25, 75])
        print(f"{s:<20s}{q[0]:>8.0f}{q[1]:>8.0f}{q[2]:>8.0f}{q[4]-q[3]:>8.0f}")

    ok = keep & np.isfinite(p_med) & (p_med > 0)

    # How much of the R4 TEST set sits at ranges the R4 TRAINING set barely
    # covers? A model cannot be blamed for frames drawn from a distance it never
    # saw, and fig 5's worst probe bin (0-2000 mm, 5.97 %) is exactly there.
    trn = ok & (seq == "Training")
    tst = ok & (seq == "Testing/Walking")
    lo, hi = np.percentile(p_med[trn], [5, 95])
    below, above = float((p_med[tst] < lo).mean()), float((p_med[tst] > hi).mean())
    print(f"\nR4 range overlap: training p05-p95 = {lo:.0f}-{hi:.0f} mm")
    print(f"  test frames BELOW the training p05: {100 * below:5.1f} %")
    print(f"  test frames ABOVE the training p95: {100 * above:5.1f} %")
    print(f"  test frames inside the training range: "
          f"{100 * (1 - below - above):5.1f} %")

    print(f"\nper-feature decomposition over {len(cols)} reported columns "
          f"({int(ok.sum())} frames):\n")
    print("A GLOBAL slope, not a per-subject one. Fitting each subject its own line "
          "\nuses that subject's identity, so it is not a correction anyone could apply "
          "\nat test time -- and Training/Still subjects barely move, leaving those fits "
          "\nill-conditioned (the RankWarnings an earlier version emitted). This is a "
          "\nfixed-effects estimate: subtract each subject's own mean, then regress the "
          "\ncentred feature on centred distance pooled over everyone. One slope, "
          "\napplicable to a stranger.\n")
    hdr = (f"{'feature':<18s}{'between-SD':>12s}{'within-SD':>11s}{'slope/m':>10s}"
           f"{'var expl':>10s}{'SNR gain':>10s}{'drift/sig':>10s}")
    print(hdr); print("-" * len(hdr))
    rows_out = []
    for c in cols:
        yc, xc, betw = [], [], []
        for s_ in sorted(set(seq)):
            for u in sorted(set(subj[ok & (seq == s_)])):
                m = ok & (seq == s_) & (subj == u)
                if m.sum() < args.min_frames:
                    continue
                y, x = full[m, c].astype(np.float64), p_med[m] / 1000.0
                if not np.isfinite(y).all():
                    continue
                betw.append(y.mean())
                yc.append(y - y.mean()); xc.append(x - x.mean())
        if len(betw) < 5:
            continue
        Y, X = np.concatenate(yc), np.concatenate(xc)
        vx = float((X * X).sum())
        slope = float((X * Y).sum() / vx) if vx > 0 else 0.0
        w_sd = float(Y.std())
        w_sd_c = float((Y - slope * X).std())
        b_sd = float(np.std(betw))
        expl = 1.0 - (w_sd_c ** 2) / max(w_sd ** 2, 1e-12)
        gain = (w_sd / w_sd_c - 1.0) * 100 if w_sd_c > 0 else 0.0
        # drift across the test corpus's own range, against the identity signal:
        # the ratio that says whether a feature measures the person or the range
        drift = abs(slope) * TEST_RANGE_M
        rows_out.append((drift / max(b_sd, 1e-9), names[c], b_sd, w_sd, slope,
                         expl, gain, drift))
    for ratio, nm, b_sd, w_sd, slope, expl, gain, drift in sorted(rows_out):
        print(f"{nm:<18s}{b_sd:>12.2f}{w_sd:>11.2f}{slope:>10.2f}"
              f"{100 * expl:>9.1f}%{gain:>9.1f}%{ratio:>10.2f}")

    print("\nREAD: `between-SD` is the identity signal; `within-SD` is what it must beat."
          "\n`slope/m` is how far the feature drifts per metre of standing distance -- for a"
          "\nquantity claimed to be in millimetres in a gravity-aligned frame, it should be"
          "\nnear zero. `SNR gain` is what removing that one global slope buys. Compare the"
          "\ndrift over the test range against `between-SD`: a feature drifting further"
          "\nacross the walk than it varies between people is measuring the range, not the"
          "\nperson.")


if __name__ == "__main__":
    main()
