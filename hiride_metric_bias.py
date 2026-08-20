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
    print(f"\nper-feature decomposition over {len(cols)} reported columns "
          f"({int(ok.sum())} frames, within-subject and within-corpus):\n")
    hdr = (f"{'feature':<18s}{'between-SD':>12s}{'within-SD':>11s}"
           f"{'dist-R2':>9s}{'slope/m':>10s}{'recoverable':>12s}")
    print(hdr); print("-" * len(hdr))
    for c in cols:
        vals, slopes, r2s, betw = [], [], [], []
        resid_all, raw_all = [], []
        for s in sorted(set(seq)):
            for u in sorted(set(subj[ok & (seq == s)])):
                m = ok & (seq == s) & (subj == u)
                if m.sum() < args.min_frames:
                    continue
                y, x = full[m, c].astype(np.float64), p_med[m] / 1000.0
                if not np.isfinite(y).all() or y.std() == 0 or x.std() == 0:
                    continue
                b = np.polyfit(x, y, 1)
                r = y - np.polyval(b, x)
                slopes.append(b[0])
                r2s.append(1.0 - r.var() / y.var())
                raw_all.append(y - y.mean()); resid_all.append(r - r.mean())
                betw.append(y.mean())
        if len(betw) < 5:
            continue
        b_sd = float(np.std(betw))
        w_sd = float(np.std(np.concatenate(raw_all)))
        w_sd_corr = float(np.std(np.concatenate(resid_all)))
        gain = (w_sd / w_sd_corr - 1.0) * 100 if w_sd_corr > 0 else 0.0
        print(f"{names[c]:<18s}{b_sd:>12.2f}{w_sd:>11.2f}"
              f"{np.median(r2s):>9.3f}{np.median(slopes):>10.2f}{gain:>11.1f}%")

    print("\nREAD: `between-SD` is the identity signal; `within-SD` is what it must beat. "
          "\n`dist-R2` is the share of within-subject variance that standing distance alone "
          "\nexplains -- variance that is not noise but a correctable bias. `recoverable` is "
          "\nthe percentage improvement in signal-to-noise from removing the linear drift. "
          "\nAnything above a few percent means these features are not as range-invariant as "
          "\ntheir units imply, and a per-frame correction would lift every metric result.")


if __name__ == "__main__":
    main()
