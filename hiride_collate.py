"""Aggregate hiride_train.py results into the ladder table.

    python hiride_collate.py --runs $SCRATCH/hiride2/runs --floor $SCRATCH/hiride2/results

Groups results_*.json by (policy, modality, arch, condition, permuted), reports
mean +/- sd across seeds, and prints the trivial-cue floor alongside so the CNN
is read against what 13 hand-computed scalars already achieve rather than
against 1/K.
"""
import os
import json
import glob
import argparse
import numpy as np

from hiride_keys import cell_key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--floor", default=None, help="dir holding floor_results.json")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.runs, "results_*.json")))
    if not files:
        print(f"no results_*.json in {args.runs} yet")
        return
    rows = []
    for f in files:
        try:
            with open(f) as fh:
                rows.append(json.load(fh))
        except Exception as exc:                     # partial write while a job runs
            print(f"  (skipping unreadable {os.path.basename(f)}: {exc})")

    floor = {}
    if args.floor:
        fp = os.path.join(args.floor, "floor_results.json")
        if os.path.exists(fp):
            with open(fp) as fh:
                blob = json.load(fh)
            # hiride_floor writes {"results": [...], "importance": ...}, not a bare list.
            for r in (blob.get("results", []) if isinstance(blob, dict) else blob):
                if isinstance(r, dict) and r.get("model") == "rf":
                    floor[r["policy"]] = r

    groups = {}
    for r in rows:
        # The key lives in hiride_keys so collate, stats and figures cannot drift
        # apart -- they did, four times, and each drift silently averaged two
        # different experiments together.
        groups.setdefault(cell_key(r), []).append(r)

    print(f"\n{len(rows)} runs in {len(groups)} cells\n")
    hdr = (f"{'policy':<22s} {'mod':<6s} {'arch':<9s} {'cond':<11s} {'n':>2s} "
           f"{'frame acc':>14s} {'per-subj':>9s} {'macroF1':>8s} "
           f"{'1/K':>6s} {'floor':>7s} {'epochs':>10s}")
    print(hdr)
    print("-" * len(hdr))
    for key in sorted(groups, key=lambda k: tuple("" if v is None else str(v) for v in k)):
        policy, guard_k, mod, arch, cond, perm = key
        g = groups[key]
        acc = np.array([x["frame_acc"] for x in g]) * 100
        ps = np.array([x["per_subject_acc"] for x in g]) * 100
        f1 = np.array([x["macro_f1"] for x in g])
        be = np.array([x["best_epoch"] for x in g])
        er = np.array([x["epochs_run"] for x in g])
        chance = g[0]["chance"] * 100
        # The floor names block rungs by guard ("R1_block_guard150") while a run
        # records policy + guard separately, so a bare .get(policy) misses.
        # guard_k comes from the cell key, so it cannot disagree with the group.
        fl = floor.get(f"{policy}_guard{guard_k}" if guard_k is not None else policy)
        fl_s = f"{fl['frame_acc'] * 100:6.2f}" if fl else "     -"
        name = policy + ("[perm]" if perm else "")
        capped = sum(bool(x.get("hit_epoch_cap")) for x in g)
        cap_s = f"  cap:{capped}" if capped else ""
        arch_s = {"alexnet": "alexnet", "convnext_tiny": "cnxt-in", "convnext_tiny/scratch": "cnxt-scr"}.get(arch, arch)
        print(f"{name:<22s} {mod:<6s} {arch_s:<9s} {cond:<11s} {len(g):>2d} "
              f"{acc.mean():8.2f} ±{acc.std():4.2f} {ps.mean():8.2f}  {f1.mean():7.3f} "
              f"{chance:5.2f}% {fl_s} {be.mean():4.1f}/{er.mean():4.1f}{cap_s}")

    # Short cells are the visible symptom of a run that failed, was cancelled, or
    # never got submitted -- and reading a 3-seed mean as if it were 5 is exactly
    # the kind of thing that quietly ends up in a table. Name them.
    counts = [len(g) for g in groups.values()]
    modal = max(set(counts), key=counts.count)
    short = [(k, len(g)) for k, g in groups.items() if len(g) < modal]
    if short:
        print(f"\n{len(short)} cell(s) with fewer than the usual {modal} seeds "
              f"-- check these finished:")
        for k, n in sorted(short, key=lambda kv: kv[1]):
            policy, guard_k, mod, arch, cond, perm = k
            print(f"  n={n}  {policy}{'' if guard_k is None else f'g{guard_k}'} "
                  f"{mod} {arch} {cond}{' [perm]' if perm else ''}")

    print("\nfloor = trivial-cue RF on the same split (13 scalars, shipped masks).")
    print("epochs = mean best_epoch / mean epochs_run. best_epoch at 0-1 with an early")
    print("stop means the model is overfitting, not converging. cap:N = N seeds hit")
    print("--epochs without early stopping (best weights are still restored).")


if __name__ == "__main__":
    main()
