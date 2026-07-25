"""Invariant tests for the HI-RIDE split library.

These assert the properties the paper will *claim* about its protocol. Run them
on any machine that has a BIWI tree (Training/ alone is enough for R0-R2):

    python test_hiride_splits.py --root /path/to/biwi
    python test_hiride_splits.py --root $SLURM_TMPDIR/biwi     # inside a job

Exit code 0 means every invariant held. Run this before submitting any wave;
a split that quietly stops being disjoint invalidates every number downstream.
"""
import sys
import argparse
import numpy as np

from hiride_data import (build_manifest, make_split, block_train_counts,
                         describe_split, interframe_stats)

FAILURES = []


def check(label, ok, detail=""):
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {label}" + (f"  -- {detail}" if detail else ""))
    if not ok:
        FAILURES.append(label)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="dir containing Training/ and/or Testing/")
    ap.add_argument("--ref-guard", type=int, default=150)
    ap.add_argument("--guards", default="0,25,50,100,150")
    args = ap.parse_args()

    man = build_manifest(args.root)
    guards = [int(g) for g in args.guards.split(",")]
    has_testing = bool((man["session"] == "B").any())
    print(f"\ninter-frame timestamp delta: {interframe_stats(man)}")

    print("\n1. reference guard must not starve any recording")
    try:
        targets = block_train_counts(man, guard=args.ref_guard, seed=0)
        check(f"guard={args.ref_guard} viable as matching reference", True,
              f"{len(targets)} recordings, {sum(targets.values())} train frames")
    except ValueError as exc:
        check(f"guard={args.ref_guard} viable", False, str(exc)[:120])
        return 1
    try:
        block_train_counts(man, guard=10 ** 6, seed=0)
        check("an absurd guard is rejected", False, "it was accepted")
    except ValueError:
        check("an absurd guard is rejected", True)

    print("\n2. test and val are byte-identical across the guard sweep")
    ref_te = ref_va = ref_n = None
    for g in guards:
        tr, va, te = make_split(man, "R1_block", seed=0, guard=g, match_ntrain=targets)
        if ref_te is None:
            ref_te, ref_va, ref_n = te, va, len(tr)
        check(f"guard={g:<4d} test identical", np.array_equal(te, ref_te))
        check(f"guard={g:<4d} val  identical", np.array_equal(va, ref_va))
        check(f"guard={g:<4d} N_train matched", len(tr) == ref_n,
              f"n_train={len(tr)} vs {ref_n}")

    print("\n3. train/val/test are pairwise disjoint")
    tr, va, te = make_split(man, "R1_block", seed=0, guard=50, match_ntrain=targets)
    check("train n val", len(np.intersect1d(tr, va)) == 0)
    check("train n test", len(np.intersect1d(tr, te)) == 0)
    check("val n test", len(np.intersect1d(va, te)) == 0)

    print("\n4. the guard actually separates train from the held-out block")
    worst = min(
        int(man["frame"][te[man["group"][te] == g]].min()
            - man["frame"][tr[man["group"][tr] == g]].max())
        for g in sorted(set(man["group"][te].tolist()))
        if (man["group"][tr] == g).any() and (man["group"][te] == g).any())
    check("min train->test frame gap > 1", worst > 1, f"gap={worst} frames")

    print("\n5. the 2023 policy leaves near-duplicate neighbours (this SHOULD be tiny)")
    r0tr, _, r0te = make_split(man, "R0_frame_random", seed=0)
    gaps = []
    for g in sorted(set(man["group"][r0te].tolist())):
        tf = np.sort(man["frame"][r0te[man["group"][r0te] == g]])
        trf = np.sort(man["frame"][r0tr[man["group"][r0tr] == g]])
        if len(tf) and len(trf):
            gaps.append(int(np.abs(tf[:, None] - trf[None, :]).min()))
    check("R0 min gap == 1 frame (documents the leak)", min(gaps) <= 1,
          f"min={min(gaps)} median={int(np.median(gaps))} over {len(gaps)} recordings")

    print("\n6. seed changes training subsample but never the test block")
    a = make_split(man, "R1_block", seed=0, guard=50, match_ntrain=targets)
    b = make_split(man, "R1_block", seed=7, guard=50, match_ntrain=targets)
    check("test stable across seeds", np.array_equal(a[2], b[2]))
    check("train varies across seeds", not np.array_equal(a[0], b[0]))

    print("\n7. cross-session policies")
    for name in ("R3_cross_recording", "R4_cross_session"):
        try:
            tr, va, te = make_split(man, name, seed=0)
            info = describe_split(man, tr, va, te)
            ok = (info["k_test"] > 0 and not info["unseen_test_subjects"]
                  and len(np.intersect1d(tr, te)) == 0)
            check(f"{name} well-formed", ok, str(info))
        except RuntimeError as exc:
            check(f"{name} unavailable -> raises clearly", not has_testing, str(exc)[:90])

    print("\n" + ("ALL INVARIANTS HELD" if not FAILURES
                  else f"{len(FAILURES)} FAILURE(S): {FAILURES}"))
    return 1 if FAILURES else 0


if __name__ == "__main__":
    sys.exit(main())
