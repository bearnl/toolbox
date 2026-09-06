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


def shard_ratios(prep, far_pairs=300, seed=0):
    """{label: adjacent |ddepth| median / random-pair |ddepth| median}."""
    man = np.load(os.path.join(prep, "manifest.npz"), allow_pickle=False)
    dep = np.load(os.path.join(prep, "training_depth.npy"), mmap_mode="r")
    idx = np.load(os.path.join(prep, "training_index.npz"))["manifest_row"]
    pos = {int(r): i for i, r in enumerate(idx)}
    subj, frame = man["subject"].astype(str), man["frame"].astype(np.int64)
    rng = np.random.default_rng(seed)
    out = {}
    for s in sorted(set(subj)):
        rows = np.flatnonzero(subj == s)
        rows = rows[np.argsort(frame[rows])]
        p = np.array([pos[int(r)] for r in rows if int(r) in pos])
        if len(p) < 20:
            continue
        def d(i, j):
            a, b = np.asarray(dep[i], np.float32), np.asarray(dep[j], np.float32)
            v = (a > 0) & (b > 0)
            return float(np.abs(a[v] - b[v]).mean()) if v.any() else np.nan
        adj = np.array([d(p[k], p[k + 1]) for k in range(len(p) - 1)])
        far = np.array([d(*rng.choice(p, 2, replace=False)) for _ in range(far_pairs)])
        a, f = np.nanmedian(adj), np.nanmedian(far)
        if np.isfinite(a) and np.isfinite(f) and f > 0:
            out[s] = float(a / f)
            print(f"[shard] {s:<10s} adjacent {a:7.1f} mm  far {f:7.1f} mm  ratio {a / f:.3f}")
    return out


def spearman(x, y):
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    d = np.sqrt((rx * rx).sum() * (ry * ry).sum())
    return float((rx * ry).sum() / d) if d > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="the in-house runs dir")
    ap.add_argument("--probe", default=None, help="inhouse_probe.json (needs far medians)")
    ap.add_argument("--prep-inhouse", default=None,
                    help="in-house prep dir; measures the ratio from "
                         "training_depth.npy when the probe JSON lacks it")
    ap.add_argument("--far-pairs", type=int, default=300)
    ap.add_argument("--biwi-runs", default=None,
                    help="main BIWI runs dir, for the aggregate reference point")
    ap.add_argument("--biwi-ratio", type=float, default=0.38,
                    help="BIWI Training near-duplicate ratio: lag-1 |ddepth| "
                         "80.6 mm over between-subject 213.5 mm "
                         "(adjacency_results.json)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    ratios = {}
    if args.probe and os.path.exists(args.probe):
        probe = json.load(open(args.probe))
        for key, rec in probe.items():
            if not isinstance(rec, dict):
                continue
            label = str(key).split("/")[-1]           # keys are "<dir>/<label>"
            adj, far = rec.get("adjacent_ddepth_median"), rec.get("far_ddepth_median")
            if adj and far:
                ratios[label] = float(adj) / float(far)
    if not ratios and args.prep_inhouse:
        # The shipped inhouse_probe.json stores far_ddepth_median = None for
        # every label (its far-pair sampling did not run in the archived
        # call), so measure the ratio directly from the prep shard: median
        # adjacent |ddepth| over median random-pair |ddepth| within each
        # identity, valid pixels only. Same definition the probe prints.
        ratios = shard_ratios(args.prep_inhouse, args.far_pairs)
    if not ratios:
        raise SystemExit("no per-label near-duplicate ratios: pass --probe with "
                         "far medians, or --prep-inhouse to measure them from "
                         "the shard")

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
