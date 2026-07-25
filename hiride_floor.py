"""The trivial-cue floor: how far do 13 hand-computed scalars get, per split policy?

This is the cheapest and most load-bearing experiment in the re-run. It needs
no GPU and finishes in seconds, and it decides how the CNN numbers must be read:

  * If simple geometry already saturates the task under the 2023 frame-random
    policy, then the published 0.99 is not evidence about body shape -- it is
    evidence that the split leaked.
  * The same features under progressively stricter policies give the leakage
    gradient, which is the paper's spine.
  * Feature attribution says WHICH cue carries it. On BIWI Training (measured
    locally, 50 subjects, chance 2%): frame-random 68.5% +/- 0.8 versus 34.3%
    +/- 0.3 for a contiguous block -- and the top cues are bbox height and
    vertical centroid, i.e. apparent size and where the person stood, not body
    geometry.

IMPORTANT SCOPE NOTE for the paper: these features are computed from the SHIPPED
userMap, i.e. a ground-truth segmentation the CNN never had. This is therefore a
floor on "geometry given perfect segmentation", not a like-for-like CNN
baseline. Say so in the caption.

Usage:
    python hiride_floor.py --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/results
"""
import os
import sys
import json
import time
import argparse
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from hiride_data import (load_manifest, make_split, block_train_counts,
                         describe_split, eligible_mask, POLICIES)


def fit_eval(X, y, tr, te, model, seed, logreg_max_iter=1000):
    sc = StandardScaler().fit(X[tr])
    if model == "logreg":
        clf = LogisticRegression(max_iter=logreg_max_iter, C=10.0, n_jobs=-1)
    else:
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    clf.fit(sc.transform(X[tr]), y[tr])
    pred = clf.predict(sc.transform(X[te]))
    frame_acc = float((pred == y[te]).mean())
    per_subj = [float((pred[y[te] == c] == c).mean()) for c in np.unique(y[te])]
    return frame_acc, float(np.mean(per_subj)), clf, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=5,
                    help="RF seeds. Ignored for logreg, which is deterministic.")
    # rf only by default: it is the stronger trivial baseline and it is fast.
    # Multinomial LBFGS over 50 classes on the full eligible pool takes minutes
    # per fit, which is too slow for a login node -- pass --models rf,logreg
    # inside a CPU job if you want it.
    ap.add_argument("--models", default="rf")
    ap.add_argument("--logreg-max-iter", type=int, default=1000)
    ap.add_argument("--perm-draws", type=int, default=10,
                    help="draws used to build the empirical null band")
    ap.add_argument("--guards", default="0,25,50,100,150")
    ap.add_argument("--ref-guard", type=int, default=150,
                    help="N_train of every block rung is matched to this guard")
    args = ap.parse_args()

    missing = [f for f in ("manifest.npz", "cues.npz")
               if not os.path.exists(os.path.join(args.prep, f))]
    if missing:
        sys.exit(
            f"error: {', '.join(missing)} not found in {args.prep}\n"
            "  This script consumes the output of the prep job. Either it has not run yet,\n"
            "  or it is still running. Check with:\n"
            "      squeue -u $USER\n"
            "      tail -20 logs/hiride-prep_*.out\n"
            "  and submit it if needed:\n"
            "      sbatch --account=def-czarnuch_cpu --export=ALL,NO_SHARDS=1 run_hiride_prep.slurm")

    os.makedirs(args.out, exist_ok=True)
    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    d = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    X, feats = d["cues"], [str(f) for f in d["feats"]]
    keep = eligible_mask(X, feats)
    subj = sorted(set(man["subject"].tolist()))
    y = np.array([subj.index(s) for s in man["subject"]])

    print(f"[floor] frames={len(y)}  eligible={int(keep.sum())} "
          f"({keep.mean() * 100:.1f}%)  subjects={len(subj)}")

    results = []
    guards = [int(g) for g in args.guards.split(",")]
    # Reference counts MUST be computed under the same eligibility filter the
    # rungs use, otherwise the "matched N_train" claim is false for any rung
    # whose eligible pool is smaller than the raw one.
    targets = block_train_counts(man, guard=args.ref_guard, seed=0, keep=keep)

    def record(name, tr, va, te, extra=None):
        info = describe_split(man, tr, va, te)
        for model in args.models.split(","):
            # LogisticRegression is deterministic: the seed only reseeds the RF
            # bootstrap. Running it `--seeds` times costs N-fold and yields
            # sd == 0, which is what made the full-data run appear to hang.
            n_seeds = args.seeds if model == "rf" else 1
            accs, psubj = [], []
            for s in range(n_seeds):
                print(f"    .. {name} {model} seed {s}", flush=True)
                fa, pa, clf, _ = fit_eval(X, y, tr, te, model, s,
                                          logreg_max_iter=args.logreg_max_iter)
                accs.append(fa); psubj.append(pa)
            # Majority-class rate, not 1/K, is the baseline a trivial predictor
            # actually achieves when the classes are imbalanced. Report both.
            maj = float(np.bincount(y[te]).max() / len(te))
            row = dict(policy=name, model=model,
                       frame_acc=float(np.mean(accs)), frame_sd=float(np.std(accs)),
                       per_subject_acc=float(np.mean(psubj)),
                       chance=info["chance"], majority_class_rate=maj,
                       x_chance=float(np.mean(accs)) / info["chance"],
                       **{k: v for k, v in info.items() if k != "chance"})
            if extra:
                row.update(extra)
            results.append(row)
            print(f"  {name:<26s} {model:<7s} acc={row['frame_acc'] * 100:6.2f}"
                  f" ±{row['frame_sd'] * 100:.2f}  = {row['x_chance']:5.1f}x chance"
                  f"  (1/K {info['chance'] * 100:.2f}%, maj {maj * 100:.2f}%,"
                  f" per-subj {row['per_subject_acc'] * 100:5.2f}%,"
                  f" n_train={info['n_train']}, n_test={info['n_test']})")

    # --- the ladder -------------------------------------------------------
    def subsample_train(tr, seed):
        """Cut R0's training set down to the block rungs' per-recording counts.

        R0 holds out only 20% per recording, so it trains on ~1.8x what the
        block rungs get. Without this, R0-vs-R1 confounds split policy with
        training-set size and overstates the leakage gap.
        """
        rng = np.random.default_rng(2000 + seed)
        out, short = [], []
        for grp, want in targets.items():
            idx = tr[man["group"][tr] == grp]
            if len(idx) > want:
                idx = rng.choice(idx, size=want, replace=False)
            elif len(idx) < want:
                short.append(grp)
            out.append(idx)
        if short:
            print(f"    (note: {len(short)} recordings had fewer than the matched "
                  f"target and were kept whole)")
        return np.sort(np.concatenate(out))

    tr, va, te = make_split(man, "R0_frame_random", seed=0, keep=keep)
    record("R0_frame_random", tr, va, te)
    record("R0_frame_random_matched", subsample_train(tr, 0), va, te)

    for g in guards:
        tr, va, te = make_split(man, "R1_block", seed=0, guard=g,
                                match_ntrain=targets, keep=keep)
        record(f"R1_block_guard{g}", tr, va, te, {"guard": g})

    for name in ("R3_cross_recording", "R4_cross_session"):
        try:
            tr, va, te = make_split(man, name, seed=0, keep=keep)
            record(name, tr, va, te)
        except Exception as exc:
            print(f"  {name:<26s} UNAVAILABLE: {exc}")

    # --- null calibration (NOT a leakage probe) ---------------------------
    # A GLOBAL per-frame label permutation destroys the subject<->feature
    # relationship, so a test frame's near-duplicate twin in train now carries
    # an unrelated label. Matching the twin therefore yields chance, and this
    # control CANNOT detect adjacency leakage -- an earlier comment here claimed
    # it could, which was wrong.
    #
    # What it does measure is whether the harness is sound (no test labels
    # reaching fit, no indexing/scaler error) and where the null actually sits.
    # It lands slightly ABOVE 1/K by construction: a permutation preserves the
    # class marginals, so a model fitted to noise drifts toward frequent
    # classes. We therefore report an empirical null over several draws and
    # compare the observed value against that band, not against 1/K.
    #
    # The real evidence for adjacency leakage is elsewhere and already
    # measured: the 1-frame nearest-neighbour gap under R0
    # (test_hiride_splits.py check 5) and the monotone guard sweep above, which
    # is N_train-matched and therefore attributable.
    def null_band(label, tr, te):
        """Empirical null for THIS rung, so `x chance` can be read honestly.

        The null must be computed per rung: it depends on the class count and
        the test-set marginals, and for the 28-class cross-session rungs it sits
        well above 1/K = 3.57%.
        """
        accs = []
        idx = np.concatenate([tr, te])
        for d in range(args.perm_draws):
            print(f"    .. null {label} draw {d}", flush=True)
            # Permute WITHIN this rung's own label space. Permuting the global
            # 50-class vector would pit a 50-way null against a 28-way task and
            # badly understate the null: the first version of this did exactly
            # that and reported 2.66% for R4, BELOW its own 1/K of 3.57%, which
            # is impossible for a correctly specified null.
            yp = y.copy()
            yp[idx] = np.random.default_rng(5000 + d).permutation(y[idx])
            sc = StandardScaler().fit(X[tr])
            clf = RandomForestClassifier(n_estimators=300, random_state=0, n_jobs=-1)
            clf.fit(sc.transform(X[tr]), yp[tr])
            accs.append(float((clf.predict(sc.transform(X[te])) == yp[te]).mean()))
        k = len(set(y[tr].tolist()))
        maj = float(np.bincount(y[te]).max() / len(te))
        mean, sd, hi = float(np.mean(accs)), float(np.std(accs)), float(np.max(accs))
        print(f"  {'null: ' + label:<26s} {'rf':<7s} "
              f"mean={mean * 100:6.2f} ±{sd * 100:.2f}  max={hi * 100:.2f}"
              f"  (1/K={100.0 / k:.2f}%, majority-class={maj * 100:.2f}%, "
              f"n={args.perm_draws} draws)")
        results.append(dict(policy=f"null_{label}", model="rf", frame_acc=mean,
                            frame_sd=sd, null_max=hi, draws=accs,
                            chance=1.0 / k, majority_class_rate=maj))

    tr, va, te = make_split(man, "R1_block", seed=0, guard=max(guards),
                            match_ntrain=targets, keep=keep)
    null_band(f"R1_block_guard{max(guards)}", tr, te)
    # The cross-session rung needs its own null: with K=28 and an imbalanced
    # test set, 1/K badly understates what a label-free model already achieves,
    # so "1.5x chance" is not the right way to read R4.
    for pol in ("R3_cross_recording", "R4_cross_session"):
        try:
            tr, va, te = make_split(man, pol, seed=0, keep=keep)
            null_band(pol, tr, te)
        except Exception as exc:
            print(f"  {'null: ' + pol:<26s} UNAVAILABLE: {str(exc)[:60]}")

    # --- feature attribution at the strictest available rung --------------
    tr, va, te = make_split(man, "R1_block", seed=0, guard=max(guards),
                            match_ntrain=targets, keep=keep)
    _, _, clf, _ = fit_eval(X, y, tr, te, "rf", 0)
    imp = sorted(zip(feats, clf.feature_importances_.tolist()), key=lambda t: -t[1])
    print("\n[floor] feature importance at guard=%d:" % max(guards))
    for f, v in imp:
        print(f"   {f:<14s} {v:.3f}")

    base, _, _, _ = fit_eval(X, y, tr, te, "rf", 0)
    loo = {}
    for i, f in enumerate(feats):
        cols = [j for j in range(len(feats)) if j != i]
        a, _, _, _ = fit_eval(X[:, cols], y, tr, te, "rf", 0)
        loo[f] = round(100 * (base - a), 2)
    print("[floor] leave-one-out drop (pp):", json.dumps(loo))

    out = os.path.join(args.out, "floor_results.json")
    with open(out, "w") as fh:
        json.dump({"results": results, "importance": imp, "leave_one_out": loo,
                   "feats": feats, "n_subjects": len(subj),
                   "eligible_frac": float(keep.mean())}, fh, indent=1)
    print("\n[floor] wrote", out)


if __name__ == "__main__":
    main()
