#!/usr/bin/env python3
"""Does the CNN fail for want of METRIC SCALE? A pre-registered test.

    python hiride_aux.py --runs $SCRATCH/hiride2/runs --prep $SCRATCH/hiride2/prep

What survives a session change is millimetres of body size. Turning pixel
extent into millimetres requires the subject's standing distance, and a
convnet is never given it -- while convolution and pooling are engineered to be
invariant to precisely the quantity that would carry it. Hand-computed metric
features reach 28.86 % at R4 where this CNN reaches 18.40 % on identical
frames, from arithmetic and no training data at all. `--aux dist` hands the
network that one scalar.

THE PREDICTION, FIXED BEFORE THE RUN (make_runs.py WAVE17):

    conditions that PRESERVE apparent size   (person, person_centred)
        -> distance SHOULD help: it is the missing calibration term
    conditions that NORMALISE size away      (scale_removed, sil_scaled)
        -> distance should do almost NOTHING: there is no metric content
           left for it to unlock

    A UNIFORM LIFT ACROSS ALL FOUR REFUTES THE STORY. That would mean distance
    is serving as a recording-identity shortcut rather than a calibration term
    -- which is exactly what adding stand_dist_mm to the RandomForest did at
    R1, costing 10 pp (67.97 -> 57.59).

Pairing is exact: the split does not depend on --aux, so the two arms of each
seed score identical test frames, and the difference is taken per frame before
any averaging. Intervals are subject-cluster bootstrap, because seeds are not
subjects.
"""
import os
import argparse
import numpy as np

from hiride_data import load_manifest
from hiride_stats import cluster_boot, boot_rng
from hiride_fuse import cnn_cells

SIZE_KEPT = ("person", "person_centred")
SIZE_GONE = ("scale_removed", "sil_scaled")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--prep", required=True)
    ap.add_argument("--modality", default="depth")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    subj_all = np.asarray(man["subject"], dtype=str)
    hdr = (f"{'policy':<20s}{'condition':<16s}{'size':>8s}{'none':>9s}"
           f"{'+dist':>9s}{'delta':>9s}{'subject-cluster CI':>22s}{'n':>5s}")
    print(hdr); print("-" * len(hdr))
    verdict = {}
    for policy in ("R3_cross_recording", "R4_cross_session"):
        for cond in SIZE_KEPT + SIZE_GONE:
            base = {int(m["seed"]): p for m, p in
                    cnn_cells(args.runs, policy, args.modality, "alexnet", cond)}
            aux = {int(m["seed"]): p for m, p in
                   cnn_cells(args.runs, policy, args.modality, "alexnet/auxdist", cond)}
            seeds = sorted(set(base) & set(aux))
            if not seeds:
                continue
            diffs, a0, a1 = [], [], []
            for sd in seeds:
                B, A = np.load(base[sd]), np.load(aux[sd])
                if not np.array_equal(B["test_rows"], A["test_rows"]):
                    print(f"  !! {policy} {cond} seed {sd}: test rows differ, skipped")
                    continue
                bo = (B["pred"] == B["truth"]).astype(float)
                ao = (A["pred"] == A["truth"]).astype(float)
                a0.append(bo.mean()); a1.append(ao.mean())
                diffs.append((ao - bo, subj_all[B["test_rows"]]))
            if not diffs:
                continue
            cis = [cluster_boot(d, s, boot_rng(args.seed, ("aux", policy, cond, i)),
                                args.boot) for i, (d, s) in enumerate(diffs)]
            lo = float(np.mean([c[0] for c in cis])) * 100
            hi = float(np.mean([c[1] for c in cis])) * 100
            d = (np.mean(a1) - np.mean(a0)) * 100
            tag = "kept" if cond in SIZE_KEPT else "removed"
            star = "" if lo * hi > 0 else "  (straddles 0)"
            print(f"{policy:<20s}{cond:<16s}{tag:>8s}{100*np.mean(a0):>8.2f}%"
                  f"{100*np.mean(a1):>8.2f}%{d:>+9.2f}"
                  f"{f'[{lo:+.2f}, {hi:+.2f}]':>22s}{len(diffs):>4d}{star}")
            verdict.setdefault(tag, []).append(d)

    if verdict.get("kept") and verdict.get("removed"):
        k, r = float(np.mean(verdict["kept"])), float(np.mean(verdict["removed"]))
        print(f"\nmean delta  size-kept {k:+.2f} pp   size-removed {r:+.2f} pp")
        print("\nVERDICT")
        if k > 1.0 and abs(r) < 1.0:
            print("  SUPPORTED. Distance helps exactly where apparent size survives and")
            print("  not where it has been normalised away, which is what a missing")
            print("  CALIBRATION term looks like and not what a shortcut looks like.")
        elif k > 1.0 and r > 1.0:
            print("  REFUTED. Distance lifts conditions that retain no metric content,")
            print("  so it is acting as a recording-identity shortcut -- the same")
            print("  failure stand_dist_mm caused in the RandomForest at R1.")
        else:
            print("  NO EFFECT. The bottleneck is not the missing calibration term.")
            print("  With ~tens of independent samples per class against ~23M")
            print("  parameters, sample size is the remaining explanation, and no")
            print("  architecture change addresses that at n = 28 subjects.")


if __name__ == "__main__":
    main()
