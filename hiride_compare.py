"""Paired comparison along ONE axis, holding every other axis fixed.

    python hiride_compare.py --runs $SCRATCH/hiride2/runs $SCRATCH/hiride2/runs_edges \
        --axis prep
    python hiride_compare.py --runs $SCRATCH/hiride2/runs --axis eligibility

hiride_collate.py prints every cell as its own row, which is the right table for
reporting but the wrong one for the questions waves 13-15 were built to answer:
"does the boundary fix change anything?" and "does the framing shift explain
depth's R4 deficit?" Both are DIFFERENCES between two cells that are identical
except on one axis, and reading them off a 60-row table by eye invites pairing
the wrong rows.

This groups results by cell key with the chosen axis neutralised, then pairs the
levels of that axis SEED BY SEED inside each group. Seed pairing matters: seed
variance at R4 is ~2 pp, the same size as the effects being looked for, so an
unpaired difference of means can easily invert the sign of a real effect.

The reported interval is the sd over seed-level differences, NOT a confidence
interval on the population. n = 28 subjects is the binding constraint and only
the subject-cluster bootstrap in hiride_stats.py speaks to that.
"""
import os
import glob
import json
import argparse
import numpy as np

from hiride_keys import cell_key, AXES


def reduced_key(r, axis):
    """Cell key with `axis` forced to its default, so levels of it collapse."""
    if axis == "prep":
        return cell_key(r)
    if axis not in AXES:
        raise SystemExit(f"--axis {axis} is not a cell axis. Known: "
                         + ", ".join(sorted(AXES)) + ", prep")
    return cell_key(dict(r, **{axis: AXES[axis]}))


def level_of(r, axis):
    return r["_src"] if axis == "prep" else r.get(axis, AXES[axis])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--axis", required=True,
                    help="the axis to compare across; 'prep' compares --runs dirs")
    ap.add_argument("--metric", default="frame_acc",
                    choices=("frame_acc", "per_subject_acc", "macro_f1", "fused_acc"))
    ap.add_argument("--min-seeds", type=int, default=2,
                    help="skip pairs with fewer than this many shared seeds")
    args = ap.parse_args()

    rows = []
    for d in args.runs:
        src = os.path.basename(os.path.normpath(d))
        for f in sorted(glob.glob(os.path.join(d, "results_*.json"))):
            try:
                r = json.load(open(f))
            except Exception as exc:
                print(f"  (skipping {os.path.basename(f)}: {exc})")
                continue
            r["_src"] = src
            rows.append(r)
    if not rows:
        raise SystemExit("no results found in " + ", ".join(args.runs))

    groups = {}
    for r in rows:
        if r.get(args.metric) is None:
            continue                          # e.g. fused_acc on a non-tracklet cell
        groups.setdefault(reduced_key(r, args.axis), {}).setdefault(
            level_of(r, args.axis), {})[r["seed"]] = float(r[args.metric])

    print(f"\nPaired on seed, axis = {args.axis}, metric = {args.metric}   "
          f"({len(rows)} results)\n")
    hdr = (f"{'policy':<20s}{'mod':<6s}{'cell':<44s}{'A -> B':<26s}"
           f"{'n':>3s}{'A':>8s}{'B':>8s}{'B-A':>9s}{'sd':>7s}")
    print(hdr); print("-" * len(hdr))
    printed = 0
    for key in sorted(groups, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        levels = groups[key]
        if len(levels) < 2:
            continue
        names = sorted(levels, key=str)
        base = names[0]
        for other in names[1:]:
            shared = sorted(set(levels[base]) & set(levels[other]))
            if len(shared) < args.min_seeds:
                continue
            a = np.array([levels[base][s] for s in shared]) * 100
            b = np.array([levels[other][s] for s in shared]) * 100
            d = b - a
            policy, guard, mod, arch, cond, perm = key
            name = policy + (f"g{guard}" if guard is not None else "") + ("[perm]" if perm else "")
            print(f"{name:<20s}{mod:<6s}{(arch + '/' + cond):<44s}"
                  f"{f'{base} -> {other}':<26s}{len(shared):>3d}"
                  f"{a.mean():>7.2f}%{b.mean():>7.2f}%{d.mean():>+8.2f}"
                  f"{d.std(ddof=1) if len(d) > 1 else float('nan'):>7.2f}")
            printed += 1
    if not printed:
        print("  (no cell had two levels of this axis with shared seeds -- either the "
              "second arm has not finished, or the axis is not varied in these runs)")
    print(f"\nsd is over the {args.metric} DIFFERENCE across shared seeds, not a CI on the")
    print("population. Seeds are not subjects; n = 28 subjects binds, so a difference here")
    print("is suggestive until hiride_stats.py puts a subject-cluster interval on it.")


if __name__ == "__main__":
    main()
