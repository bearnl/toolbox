"""Subject-level uncertainty and paired modality tests for the Wave-2 cells.

    python hiride_stats.py --runs $SCRATCH/hiride2/runs [--boot 2000] [--json out.json]

hiride_collate.py reports mean +/- sd over seeds. This script adds what the
design (HIRIDE_HANDOFF.md section 4.4) asks for and the trainer now saves per
cell in cm_*.npz (test_rows, test_subject, truth, pred):

  1. Subject-cluster bootstrap CI on frame accuracy for every cell -- frames
     within a recording are not independent, so the unit of resampling is the
     SUBJECT (n = 50 or 28), not the frame.
  2. Paired RGB-vs-depth contrast per (policy, seed): the two cells score the
     SAME test rows (asserted), so the difference is tested with an exact
     McNemar on the discordant frames and given a subject-cluster bootstrap CI.
     Seeds are then pooled by averaging the per-seed difference.
  3. Chance and majority-class rate next to every number.

Nothing here reads images; it runs on a login node in seconds.
"""
import os
import glob
import json
import argparse
from math import lgamma

import hashlib
import numpy as np

from hiride_keys import cell_key, cond_key, arch_key


def load_cells(runs):
    cells = {}
    for f in sorted(glob.glob(os.path.join(runs, "results_*.json"))):
        with open(f) as fh:
            r = json.load(fh)
        cm = f.replace("results_", "cm_").replace(".json", ".npz")
        if not os.path.exists(cm):
            continue
        z = np.load(cm, allow_pickle=False)
        if "test_rows" not in z.files:                 # old-path cell, no per-frame data
            continue
        # The key comes from hiride_keys so this file cannot drift from
        # hiride_collate.py -- it did, repeatedly, and each drift silently
        # collided cells (only the last file read survived) or averaged two
        # different experiments. Adding an axis there covers both consumers.
        key = cell_key(r) + (r["seed"],)
        cells[key] = dict(meta=r, rows=z["test_rows"], subj=z["test_subject"].astype(str),
                          truth=z["truth"], pred=z["pred"])
    return cells


def boot_rng(base_seed, key):
    """An independent generator per bootstrapped quantity, keyed on its identity.

    A single shared generator consumed in loop order makes every interval depend
    on how many OTHER cells happened to be processed first. Landing a wave that
    touches unrelated cells then shifts intervals that no new evidence bears on:
    adding three cells on 2026-08-20 moved the headline `sil_scaled` bound from
    +0.19 to +0.10 pp with its own data unchanged. Deriving the stream from the
    cell's own key makes an interval a function of that cell's data and the
    global --seed, and nothing else.
    """
    h = hashlib.blake2b(repr(key).encode(), digest_size=8).digest()
    return np.random.default_rng([base_seed, int.from_bytes(h, "big")])


def cluster_boot(correct, subj, rng, n_boot):
    """Bootstrap frame accuracy by resampling SUBJECTS with replacement.

    Vectorised: the mean over a resampled set of subjects is
    sum(their correct counts) / sum(their frame counts), so the whole bootstrap
    is two gathers and two row-sums -- no per-draw concatenation of frame
    arrays. Exactly equal to the loop it replaces, and fast enough that --boot
    is no longer a reason to keep the resample count low.
    """
    subjects, inv = np.unique(subj, return_inverse=True)
    k = len(subjects)
    sums = np.bincount(inv, weights=np.asarray(correct, dtype=np.float64), minlength=k)
    cnts = np.bincount(inv, minlength=k).astype(np.float64)
    pick = rng.integers(0, k, size=(n_boot, k))
    stats = sums[pick].sum(axis=1) / cnts[pick].sum(axis=1)
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def mcnemar_exact(b, c):
    """Two-sided exact McNemar p on discordant counts b (A right, B wrong), c.

    Computed in log space: `2.0 ** n` overflows a float once the discordant
    count passes 1,023, which R1's rgb-vs-depth contrast does (~1,100 frames).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    logs = np.array([lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1) for i in range(k + 1)])
    logs -= n * np.log(2.0)
    m = logs.max()
    log_p = m + np.log(np.exp(logs - m).sum())
    return float(min(1.0, 2.0 * np.exp(log_p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    cells = load_cells(args.runs)
    if not cells:
        print(f"no cm_*.npz with per-frame predictions in {args.runs} "
              f"(cells written before the sequence_v2 trainer have none)")
        return
    out = {"cells": [], "paired": []}

    # ---- 1. per-cell subject-cluster CIs, grouped over seeds -------------------
    groups = {}
    for key, c in cells.items():
        groups.setdefault(key[:-1], []).append(c)
    print(f"\n{len(cells)} cells with per-frame predictions\n")
    hdr = (f"{'policy':<20s}{'mod':<6s}{'arch':<26s}{'condition':<28s}{'perm':<5s}{'n':>2s} "
           f"{'frame acc':>12s} {'subj-boot 95% CI':>20s} {'per-subj':>9s} {'1/K':>6s} {'maj':>6s}")
    print(hdr)
    print("-" * len(hdr))
    for gkey in sorted(groups, key=lambda k: tuple(str(v) for v in k)):
        g = groups[gkey]
        policy, guard, mod, arch, cond, perm = gkey
        accs, los, his, per_subj = [], [], [], []
        for c in g:
            correct = (c["pred"] == c["truth"]).astype(float)
            lo, hi = cluster_boot(correct, c["subj"],
                                  boot_rng(args.seed, gkey + (c["meta"]["seed"],)),
                                  args.boot)
            accs.append(correct.mean()); los.append(lo); his.append(hi)
            per_subj.append(np.mean([correct[c["subj"] == s].mean() for s in np.unique(c["subj"])]))
        m = g[0]["meta"]
        rec = dict(policy=policy, guard=guard, modality=mod, arch=arch, condition=cond,
                   bits=m.get("bits", 16), frames=m.get("frames", 1),
                   encoding=m.get("encoding", "raw"), head=m.get("head", "gap"),
                   augment=m.get("augment", 0), test_fuse=m.get("test_fuse", 1),
                   eligibility=m.get("eligibility", "cues"),
                   base_condition=m["condition"], permuted=perm, n_seeds=len(g),
                   frame_acc_mean=float(np.mean(accs)), frame_acc_sd=float(np.std(accs)),
                   subj_ci_lo_mean=float(np.mean(los)), subj_ci_hi_mean=float(np.mean(his)),
                   per_subject_acc_mean=float(np.mean(per_subj)),
                   chance=m["chance"], majority=m.get("majority_class_rate"))
        out["cells"].append(rec)
        name = policy + (f"g{guard}" if guard is not None else "")
        arch_s = arch.replace("convnext_tiny", "cnxt-in")
        cond_s = cond                       # already composite, straight from cell_key
        # A CI whose lower bound clears the majority-class rate is the only kind
        # this study calls a positive result: at R4 there are 28 subjects, so a
        # frame-level number several points above 1/K can still be one lucky
        # subject.
        flag = " *" if rec["subj_ci_lo_mean"] > (m.get("majority_class_rate") or 0) else ""
        print(f"{name:<20s}{mod:<6s}{arch_s:<26s}{cond_s:<28s}{'perm' if perm else '':<5s}{len(g):>2d} "
              f"{100 * rec['frame_acc_mean']:6.2f} ±{100 * rec['frame_acc_sd']:4.2f} "
              f"[{100 * rec['subj_ci_lo_mean']:6.2f}, {100 * rec['subj_ci_hi_mean']:6.2f}]"
              f"{100 * rec['per_subject_acc_mean']:9.2f} "
              f"{100 * m['chance']:5.2f}% {100 * (m.get('majority_class_rate') or 0):5.2f}%{flag}")

    # ---- 2. paired RGB vs depth on identical test rows -----------------------
    print("\nPaired modality contrast (same policy, seed, condition; identical test frames):\n")
    hdr = (f"{'policy':<20s}{'arch/condition':<44s}{'seed':>4s} {'rgb':>7s} {'depth':>7s} "
           f"{'rgb-depth':>10s} {'subj-boot 95% CI':>18s} {'discordant b/c':>15s} {'McNemar p':>10s}")
    print(hdr)
    print("-" * len(hdr))
    pol_diffs = {}
    for key in sorted(cells):
        policy, guard, mod, arch, cond, perm, seed = key
        if mod != "rgb" or perm:
            continue
        # the depth twin of this cell: same everything, modality swapped. Built
        # through cell_key so it stays correct as axes are added.
        dkey = cell_key(dict(cells[key]["meta"], modality="depth")) + (seed,)
        if dkey not in cells:
            continue
        R, D = cells[key], cells[dkey]
        if not np.array_equal(R["rows"], D["rows"]):
            print(f"  !! {policy} seed {seed}: rgb/depth test rows differ -- not paired, skipped")
            continue
        r_ok = R["pred"] == R["truth"]; d_ok = D["pred"] == D["truth"]
        b = int((r_ok & ~d_ok).sum()); c = int((~r_ok & d_ok).sum())
        p = mcnemar_exact(b, c)
        diff = r_ok.astype(float) - d_ok.astype(float)
        lo, hi = cluster_boot(diff, R["subj"],
                              boot_rng(args.seed, ("paired",) + key), args.boot)
        rec = dict(policy=policy, guard=guard, arch=arch, condition=cond, seed=seed,
                   rgb=float(r_ok.mean()),
                   depth=float(d_ok.mean()), diff=float(diff.mean()), ci=[lo, hi],
                   discordant_rgb_only=b, discordant_depth_only=c, mcnemar_p=p,
                   n_test=int(len(diff)))
        out["paired"].append(rec)
        # arch MUST be in this key. Without it the summary pools gap, stripe and
        # stripe/aug8/tf10 into one "scale_removed" row -- three different models,
        # one printed mean, and an odd seed count (18) as the only hint.
        pol_diffs.setdefault((policy, guard, arch, cond), []).append(rec["diff"])
        name = policy + (f"g{guard}" if guard is not None else "")
        print(f"{name:<20s}{(arch + '/' + cond):<44s}{seed:>4d} {100 * rec['rgb']:6.2f}% {100 * rec['depth']:6.2f}% "
              f"{100 * rec['diff']:+9.2f}  [{100 * lo:+6.2f}, {100 * hi:+6.2f}] "
              f"{b:>7d}/{c:<7d} {p:10.2e}")
    if pol_diffs:
        print()
        for (policy, guard, arch, cond), ds in sorted(pol_diffs.items()):
            ds = np.array(ds) * 100
            name = (policy + (f"g{guard}" if guard is not None else "")) + f" [{arch}/{cond}]"
            print(f"  {name:<62s} rgb-depth over {len(ds)} seeds: {ds.mean():+6.2f} pp "
                  f"(sd {ds.std():4.2f}, se {ds.std(ddof=1) / np.sqrt(len(ds)) if len(ds) > 1 else float('nan'):4.2f})")

    # ---- 3. paired mask-condition vs full, same modality/seed, identical rows --
    print("\nPaired condition contrast vs `full` (same policy, modality, arch, seed; identical test frames):\n")
    hdr = (f"{'policy':<22s}{'mod':<6s}{'condition':<11s}{'n':>2s} {'cond':>7s} {'full':>7s} "
           f"{'cond-full':>10s} {'subj-boot 95% CI (mean over seeds)':>36s} {'McNemar p (median)':>19s}")
    print(hdr)
    print("-" * len(hdr))
    cond_rows = {}
    for key in sorted(cells):
        policy, guard, mod, arch, cond, perm, seed = key
        meta = cells[key]["meta"]
        if cond == "full" or perm:
            continue
        # the unedited twin: same model and training, `full` input at 16 bits.
        fkey = cell_key(dict(meta, condition="full", bits=16, frames=1,
                             encoding="raw")) + (seed,)
        if fkey not in cells:
            continue
        C, F = cells[key], cells[fkey]
        if not np.array_equal(C["rows"], F["rows"]):
            continue
        c_ok = C["pred"] == C["truth"]; f_ok = F["pred"] == F["truth"]
        b = int((c_ok & ~f_ok).sum()); c = int((~c_ok & f_ok).sum())
        diff = c_ok.astype(float) - f_ok.astype(float)
        lo, hi = cluster_boot(diff, C["subj"],
                              boot_rng(args.seed, ("cond",) + key), args.boot)
        rec = dict(policy=policy, guard=guard, modality=mod, arch=arch, condition=cond, seed=seed,
                   cond_acc=float(c_ok.mean()), full_acc=float(f_ok.mean()), diff=float(diff.mean()),
                   bits=meta.get("bits", 16), ci=[lo, hi], mcnemar_p=mcnemar_exact(b, c))
        out.setdefault("conditions", []).append(rec)
        cond_rows.setdefault((policy, guard, mod, arch, cond), []).append(rec)
    for (policy, guard, mod, arch, cond), rs in sorted(cond_rows.items()):
        name = policy + (f"g{guard}" if guard is not None else "")
        d = np.array([r["diff"] for r in rs]) * 100
        lo = np.mean([r["ci"][0] for r in rs]) * 100; hi = np.mean([r["ci"][1] for r in rs]) * 100
        print(f"{name:<22s}{mod:<6s}{cond:<28s}{len(rs):>2d} "
              f"{100 * np.mean([r['cond_acc'] for r in rs]):6.2f}% {100 * np.mean([r['full_acc'] for r in rs]):6.2f}% "
              f"{d.mean():+9.2f}  {'':>8s}[{lo:+6.2f}, {hi:+6.2f}] {'':>8s}"
              f"{np.median([r['mcnemar_p'] for r in rs]):10.2e}")

    print("\nCI = 95 % percentile bootstrap resampling SUBJECTS (n = k_test), mean over seeds.")
    print("* marks cells whose CI lower bound clears the majority-class rate -- the only")
    print("  cells this study reports as a positive identification result.")
    print("McNemar = exact two-sided binomial on the discordant frames of one seed; frames")
    print("are not independent, so read the subject-bootstrap CI as the honest interval and")
    print("McNemar as a lower bound on the p-value's order of magnitude.")

    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print(f"[written] {args.json}")


if __name__ == "__main__":
    main()
