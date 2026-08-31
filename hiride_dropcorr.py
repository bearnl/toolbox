"""Does the R0->R1 drop track near-duplication, per identity?

    python hiride_dropcorr.py --runs $SCRATCH/hiride2/runs_inhouse \
        --probe $SCRATCH/hiride2/results/inhouse_probe.json \
        --out $SCRATCH/hiride2/results

Wave 8 replicated the frame-random -> contiguous-block collapse on the Azure
Kinect corpus and HIRIDE_HANDOFF 11.1 left this note: the drop was PREDICTED to
track the near-duplicate ratio (adjacent |ddepth| over far-apart |ddepth|; at
~0.9 adjacent frames are nearly as different as random ones, so there is little
frame-random leak left to demonstrate), the direction came out right across two
datasets, but "stating it as a per-recording correlation would need the drop
measured per identity ... and has not been done". This does it.

Per in-house identity (n = 10): the per-subject accuracy drop R0 - R1 (mean
over seeds, from the stored per-frame predictions) against that identity's
near-duplicate ratio from inhouse_probe.json. Spearman rank correlation, both
modalities. BIWI enters as one aggregate reference point (ratio ~0.38 from
adjacency_results.json; drop from the main runs dir if --biwi-runs is given).

READ THE RESULT FOR WHAT IT IS: ten points from one corpus. A clean monotone
relation supports the leak mechanism; a flat one says the drop has other
drivers too (the room-identity confound differs by recording directory here).
Either way it belongs in the paper as a measurement, not a hunch.
"""
import os
import json
import glob
import argparse

import numpy as np

from hiride_keys import cond_key, arch_key


def per_subject_acc(runs, policy, modality, condition_key, arch="alexnet"):
    """{subject: mean per-subject accuracy over seeds} for one cell."""
    accs = {}
    for f in sorted(glob.glob(os.path.join(runs, "results_*.json"))):
        try:
            m = json.load(open(f))
        except (ValueError, OSError):
            continue
        if (m.get("policy") != policy or m.get("modality") != modality
                or m.get("permuted") or arch_key(m) != arch
                or cond_key(m) != condition_key):
            continue
        cm = os.path.join(runs, "cm_" + os.path.basename(f)[len("results_"):-len(".json")] + ".npz")
        if not os.path.exists(cm):
            continue
        z = np.load(cm, allow_pickle=False)
        subj = z["test_subject"].astype(str)
        ok = (z["pred"] == z["truth"]).astype(float)
        for s in np.unique(subj):
            accs.setdefault(str(s), []).append(float(ok[subj == s].mean()))
    return {s: float(np.mean(v)) for s, v in accs.items()}, \
           {s: len(v) for s, v in accs.items()}


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="the in-house runs dir")
    ap.add_argument("--probe", required=True, help="inhouse_probe.json")
    ap.add_argument("--biwi-runs", default=None,
                    help="main BIWI runs dir, for the aggregate reference point")
    ap.add_argument("--biwi-ratio", type=float, default=0.38,
                    help="BIWI Training near-duplicate ratio: lag-1 |ddepth| "
                         "80.6 mm over between-subject 213.5 mm "
                         "(adjacency_results.json)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    probe = json.load(open(args.probe))
    ratios = {}
    for label, rec in probe.items():
        if not isinstance(rec, dict):
            continue
        adj, far = rec.get("adjacent_ddepth_median"), rec.get("far_ddepth_median")
        if adj and far:
            ratios[str(label)] = float(adj) / float(far)
    if not ratios:
        raise SystemExit(f"no per-label ratios found in {args.probe} -- keys: "
                         f"{list(probe)[:8]}")

    report = {}
    for modality in ("depth", "rgb"):
        # wave 8 ran --eligibility all, so the condition key is "full/all"
        r0, n0 = per_subject_acc(args.runs, "R0_frame_random", modality, "full/all")
        r1, n1 = per_subject_acc(args.runs, "R1_block", modality, "full/all")
        common = sorted(set(r0) & set(r1) & set(ratios))
        if len(common) < 5:
            print(f"[{modality}] only {len(common)} identities with both rungs "
                  f"and a probe ratio -- skipped (have runs for "
                  f"{sorted(set(r0) & set(r1))[:5]}..., ratios for "
                  f"{sorted(ratios)[:5]}...)")
            continue
        drops = np.array([r0[s] - r1[s] for s in common])
        rats = np.array([ratios[s] for s in common])
        rho = spearman(rats, drops)
        print(f"\n[{modality}] per-identity R0->R1 drop vs near-duplicate ratio "
              f"(n={len(common)}, seeds/cell ~{int(np.median(list(n0.values())))})")
        print(f"{'identity':<12s}{'ratio':>7s}{'R0':>8s}{'R1':>8s}{'drop':>8s}")
        for s in sorted(common, key=lambda k: ratios[k]):
            print(f"{s:<12s}{ratios[s]:7.2f}{100 * r0[s]:7.1f}%{100 * r1[s]:7.1f}%"
                  f"{100 * (r0[s] - r1[s]):+7.1f}")
        print(f"  Spearman rho = {rho:+.2f}  "
              f"(prediction: NEGATIVE -- higher ratio, less duplication, smaller drop)")
        report[modality] = dict(
            identities={s: dict(ratio=ratios[s], r0=r0[s], r1=r1[s],
                                drop=r0[s] - r1[s]) for s in common},
            spearman=rho, n=len(common))

    if args.biwi_runs:
        b0, _ = per_subject_acc(args.biwi_runs, "R0_frame_random", "depth", "full")
        b1, _ = per_subject_acc(args.biwi_runs, "R1_block", "depth", "full")
        common = sorted(set(b0) & set(b1))
        if common:
            drop = float(np.mean([b0[s] - b1[s] for s in common]))
            print(f"\n[biwi] aggregate reference: ratio {args.biwi_ratio:.2f}, "
                  f"mean per-subject R0->R1 drop {100 * drop:+.1f} pp "
                  f"({len(common)} subjects)")
            report["biwi_aggregate"] = dict(ratio=args.biwi_ratio, drop=drop,
                                            n=len(common))

    if args.out:
        path = os.path.join(args.out, "dropcorr.json")
        json.dump(report, open(path, "w"), indent=1)
        print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
