"""Validate the metric-geometry result before it becomes the paper's headline.

    python hiride_metric_check.py --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/results

hiride_metric_floor.py reports 19.04 % at R4 from twelve millimetre-scale body
measurements -- above every CNN we have trained (best depth 18.01, best RGB
18.53) and far above the pixel-space floor's 5.35 %. That is the strongest
number in the study, so it needs the two checks it does not yet have.

1. DOES THE MEASUREMENT ITSELF TRANSFER?  Accuracy can rise for reasons that are
   not biometry. The direct test is whether a person's STATURE, measured in one
   session, predicts the same person's stature in the other. Paper 3 ran exactly
   this and got Pearson -0.04 / Spearman -0.15, and concluded BIWI's framing made
   metric measurement impossible. But hiride_fov_check.py showed paper 3's
   foreground merges the body with the floor and the wall (precision 0.204,
   covering 39 % of the frame), so that correlation was measured on the room.
   With the shipped userMap as the foreground the correlation should be strongly
   POSITIVE -- and if it is, it simultaneously validates this result and explains
   paper 3's failure. If it is near zero, 19.04 % is coming from something other
   than body size and must not be described as anthropometry.

   Reported alongside: within-subject vs between-subject spread. A biometric
   needs between >> within; the ratio is the discriminability that 19 % rests on.

2. WHAT IS THE UNCERTAINTY?  n = 28 subjects, so a frame-level 19.04 % can still
   be a handful of lucky subjects. Same subject-cluster bootstrap the CNN cells
   get, plus a within-split label permutation for the null.

CPU, ~1 min. Reads metric_features.npz; trains nothing new beyond one RF per
bootstrap-free fit.
"""
import os
import json
import argparse

import numpy as np

from hiride_data import load_manifest, make_split, block_train_counts, eligible_mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    z = np.load(os.path.join(args.prep, "metric_features.npz"), allow_pickle=False)
    F, names, mrow = z["feats"], [str(n) for n in z["names"]], z["manifest_row"]
    print(f"[load] {F.shape[0]} frames x {F.shape[1]} metric features")
    col = {n: i for i, n in enumerate(names)}
    subj, seq = man["subject"][mrow], man["seq"][mrow]

    # ---- 1. does stature transfer between sessions? ------------------------
    report = {}
    if "stature_mm" in col:
        s = F[:, col["stature_mm"]]
        ok = np.isfinite(s) & (s > 500) & (s < 2500)
        A, B = (seq == "Training") & ok, (seq == "Testing/Walking") & ok
        shared = sorted(set(subj[A]) & set(subj[B]))
        a = np.array([np.median(s[A & (subj == c)]) for c in shared])
        b = np.array([np.median(s[B & (subj == c)]) for c in shared])
        pr = float(np.corrcoef(a, b)[0, 1])
        ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
        sp = float(np.corrcoef(ra, rb)[0, 1])
        within = float(np.median([np.std(s[ok & (subj == c)]) for c in shared]))
        between = float(np.std(np.concatenate([a, b])))
        print(f"\n=== stature transfer, {len(shared)} subjects in both sessions ===")
        print(f"  Training median {np.median(a):7.1f} mm   Walking median {np.median(b):7.1f} mm"
              f"   bias {np.median(b - a):+6.1f} mm")
        print(f"  Pearson r = {pr:+.3f}   Spearman = {sp:+.3f}"
              f"   (paper 3, on its slab foreground: -0.04 / -0.15)")
        print(f"  within-subject sd {within:6.1f} mm   between-subject sd {between:6.1f} mm"
              f"   ratio {between / max(within, 1e-9):.2f}")
        print("  -> a positive r means the measurement IS the body and transfers; it also")
        print("     explains paper 3's null as a property of its foreground, not of BIWI.")
        report["stature"] = dict(n_subjects=len(shared), pearson=pr, spearman=sp,
                                 within_sd_mm=within, between_sd_mm=between,
                                 bias_mm=float(np.median(b - a)))

    # ---- 2. R4 with a subject-cluster CI and a permutation null -------------
    from sklearn.ensemble import RandomForestClassifier
    cues = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    keep = eligible_mask(cues["cues"], [str(f) for f in cues["feats"]])
    tr, va, te = make_split(man, "R4_cross_session", seed=0, keep=keep)
    pos = {int(r): i for i, r in enumerate(mrow)}
    tr = np.array([r for r in tr if int(r) in pos])
    te = np.array([r for r in te if int(r) in pos])
    drop = {"stand_dist_mm", "top_clip", "bot_clip"}
    use = [i for n, i in col.items() if n not in drop]
    Xtr = F[[pos[int(r)] for r in tr]][:, use]
    Xte = F[[pos[int(r)] for r in te]][:, use]
    good = np.isfinite(Xtr).all(1); Xtr, tr = Xtr[good], tr[good]
    good = np.isfinite(Xte).all(1); Xte, te = Xte[good], te[good]
    classes = sorted(set(man["subject"][tr]))
    cmap = {c: i for i, c in enumerate(classes)}
    ytr = np.array([cmap[c] for c in man["subject"][tr]])
    m_te = np.array([c in cmap for c in man["subject"][te]])
    Xte, te, yte = Xte[m_te], te[m_te], np.array(
        [cmap[c] for c in man["subject"][te][m_te]])
    ste = man["subject"][te]
    print(f"\n=== R4 metric floor: {len(tr)} train / {len(te)} test, "
          f"{len(classes)} subjects, {len(use)} features ===")

    accs, correct_runs = [], []
    for s in range(args.seeds):
        rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=s)
        rf.fit(Xtr, ytr)
        pred = rf.predict(Xte)
        correct_runs.append((pred == yte).astype(float))
        accs.append(float((pred == yte).mean()))
    corr = np.mean(correct_runs, axis=0)
    subs = np.unique(ste)
    per = {c: corr[ste == c] for c in subs}
    boot = np.empty(args.boot)
    for b in range(args.boot):
        pick = rng.choice(subs, size=len(subs), replace=True)
        boot[b] = np.concatenate([per[c] for c in pick]).mean()
    lo, hi = float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
    per_subj = float(np.mean([per[c].mean() for c in subs]))
    maj = float(np.bincount(yte).max() / len(yte))

    nulls = []
    for s in range(args.seeds):
        yp = np.random.default_rng(5000 + s).permutation(ytr)
        rf = RandomForestClassifier(n_estimators=400, n_jobs=-1, random_state=s)
        rf.fit(Xtr, yp)
        nulls.append(float((rf.predict(Xte) == yte).mean()))

    print(f"  frame accuracy {100 * np.mean(accs):6.2f} % ± {100 * np.std(accs):.2f}"
          f"   subject-cluster 95 % CI [{100 * lo:.2f}, {100 * hi:.2f}]")
    print(f"  per-subject    {100 * per_subj:6.2f} %   permutation null "
          f"{100 * np.mean(nulls):.2f} %   1/K {100 / len(classes):.2f} %   majority {100 * maj:.2f} %")
    verdict = ("CLEARS the majority rate" if lo > maj else
               "does NOT clear the majority rate -- report the interval, not the point")
    print(f"  -> CI lower bound {verdict}")
    report["r4"] = dict(acc=float(np.mean(accs)), sd=float(np.std(accs)), ci=[lo, hi],
                        per_subject=per_subj, null=float(np.mean(nulls)),
                        majority=maj, n_classes=len(classes), n_test=int(len(te)),
                        features=[names[i] for i in use])
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "metric_check.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\n[written] {os.path.join(args.out, 'metric_check.json')}")


if __name__ == "__main__":
    main()
