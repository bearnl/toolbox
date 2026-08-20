"""The ladder on METRIC features -- the test of "use depth as geometry".

    python hiride_metric_floor.py --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/results

The existing trivial-cue floor (hiride_floor.py) uses IMAGE-space cues: pixel
counts, bounding boxes in pixels, centroids in pixels. That is why it scored
89.4 % under the 2023 frame-random policy and collapsed to 5.35 % across a
session -- pixel measurements are not camera-invariant, and the camera moved
(+620 mm of background depth for all 28 shared subjects, people ~30 cm closer
and 34 px lower).

These features are millimetres in a gravity-aligned frame, so camera
translation, camera pitch and standing distance cancel by construction. The
sharp prediction: the metric floor should transfer across the session gap far
better than the image floor, even if its within-session number is lower.

Feature sets are reported separately so the contributions are attributable:
    metric      depth geometry only (stature, width profile, volume, extent)
    skeleton    segment lengths from the shipped _skel.txt, if it parsed
    both        concatenated
    +nuisance   metric plus stand_dist_mm, to show how much a camera-dependent
                covariate flatters the within-session rungs and then fails
Chance, majority-class rate and a label-permutation null accompany every rung,
same as everywhere else in this project.
"""
import os
import json
import argparse

import numpy as np

from hiride_data import load_manifest, make_split, block_train_counts, eligible_mask

LADDER = [("R0_frame_random", {}), ("R1_block", dict(guard=150)),
          ("R3_cross_recording", {}), ("R4_cross_session", {})]


def fit_eval(Xtr, ytr, Xte, yte, seed, model="rf"):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if model == "rf":
        clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        clf.fit(Xtr, ytr)
        pred = clf.predict(Xte)
        imp = clf.feature_importances_
    else:
        sc = StandardScaler().fit(Xtr)
        clf = LogisticRegression(max_iter=2000, random_state=seed)
        clf.fit(sc.transform(Xtr), ytr)
        pred = clf.predict(sc.transform(Xte))
        imp = None
    acc = float((pred == yte).mean())
    per = float(np.mean([(pred[yte == c] == c).mean() for c in np.unique(yte)]))
    return acc, per, imp, pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--perm-draws", type=int, default=5)
    # Both default to prior behaviour, so an un-flagged run reproduces the
    # published 19.04 % exactly.
    ap.add_argument("--eligibility", choices=("cues", "full_body"), default="cues",
                    help="full_body scores only frames whose whole body is in "
                         "shot. hiride_metric_bias.py measures stature_mm "
                         "drifting 89.5 mm per metre of standing distance "
                         "against a between-subject SD of 105 mm -- because a "
                         "clipped body is a SHORTER body, so the height "
                         "features partly encode range. w_15 and w_30, which "
                         "need no vertical extent, do not drift at all.")
    ap.add_argument("--detrend", action="store_true",
                    help="remove one global fixed-effects slope per feature "
                         "against standing distance, fitted on TRAIN rows only "
                         "and applied to both sides. Not the same as feeding "
                         "stand_dist_mm as a feature, which lets the model use "
                         "range as a shortcut and cost 10 pp at R1 (12.1); "
                         "this removes its influence instead.")
    ap.add_argument("--range-match", action="store_true",
                    help="score only test frames inside the training p05-p95 "
                         "standing-distance band. 36.3 %% of R4 test frames "
                         "fall outside it, so the headline number is partly a "
                         "model being asked about distances it never saw. "
                         "Reports the retained fraction; the resulting accuracy "
                         "answers a DIFFERENT question and is not comparable "
                         "to the unrestricted number.")
    args = ap.parse_args()

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    z = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    cue_feats = [str(f) for f in z["feats"]]
    keep_cue = eligible_mask(z["cues"], cue_feats,
                             full_body=(args.eligibility == "full_body"))
    p_med = z["cues"][:, cue_feats.index("p_med")].astype(np.float64)

    mf = np.load(os.path.join(args.prep, "metric_features.npz"), allow_pickle=False)
    F, names, rows = mf["feats"], [str(n) for n in mf["names"]], mf["manifest_row"]
    print(f"[load] {F.shape[0]} frames x {F.shape[1]} features")

    full = np.full((len(man["frame"]), F.shape[1]), np.nan, dtype=np.float32)
    full[rows] = F
    have = ~np.isnan(full[:, 0])
    keep = keep_cue & have
    print(f"[load] {int(have.sum())} frames have metric features; "
          f"{int(keep.sum())} also pass the cue eligibility filter")

    bone = [i for i, n in enumerate(names) if n.startswith("bone_")]
    nuis = [i for i, n in enumerate(names) if n in ("stand_dist_mm", "ground",
                                                    "n_points", "valid_frac",
                                                    "top_clip", "bot_clip")]
    metric = [i for i in range(len(names)) if i not in bone and i not in nuis]
    SETS = {"metric": metric}
    if bone:
        SETS["skeleton"] = bone
        SETS["both"] = metric + bone
    SETS["metric+nuisance"] = metric + [i for i, n in enumerate(names)
                                        if n == "stand_dist_mm"]
    print("[sets] " + "  ".join(f"{k}={len(v)}" for k, v in SETS.items()))

    report = {}
    hdr = (f"{'split':<20s}{'feature set':<17s}{'n':>2s}{'acc':>9s}{'per-subj':>10s}"
           f"{'null':>8s}{'1/K':>7s}{'majority':>10s}{'n_train':>9s}")
    print("\n" + hdr); print("-" * len(hdr))
    for pol, kw in LADDER:
        if pol.startswith("R1"):
            kw = dict(kw, match_ntrain=block_train_counts(man, guard=150, seed=0, keep=keep))
        try:
            tr, va, te = make_split(man, pol, seed=0, keep=keep, **kw)
        except Exception as e:
            print(f"{pol:<20s} unavailable: {type(e).__name__}")
            continue
        classes = sorted(set(man["subject"][tr].tolist()))
        cmap = {c: i for i, c in enumerate(classes)}
        ytr = np.array([cmap[s] for s in man["subject"][tr]])
        m_te = np.array([s in cmap for s in man["subject"][te]])
        te = te[m_te]
        if args.range_match:
            lo, hi = np.percentile(p_med[tr], [5, 95])
            inband = (p_med[te] >= lo) & (p_med[te] <= hi)
            print(f"{pol:<20s} range-match {lo:.0f}-{hi:.0f} mm keeps "
                  f"{100 * inband.mean():.1f} % of test frames")
            te = te[inband]
            if len(te) < 50:
                continue
        yte = np.array([cmap[s] for s in man["subject"][te]])
        if len(te) < 50:
            continue
        maj = float(np.bincount(yte).max() / len(yte))
        Xall = full
        if args.detrend:
            # One slope per feature, fitted on TRAIN rows with each subject
            # centred on its own mean, so the fit cannot borrow identity and is
            # well-conditioned even though many subjects barely move. Applied to
            # train and test alike; test rows never inform the slope.
            Xall = full.copy()
            zt = p_med[tr] / 1000.0
            for j in range(full.shape[1]):
                yc, xc = [], []
                for u in sorted(set(man["subject"][tr].tolist())):
                    sm = man["subject"][tr] == u
                    if sm.sum() < 20:
                        continue
                    yy = full[tr][sm, j].astype(np.float64)
                    if not np.isfinite(yy).all():
                        continue
                    yc.append(yy - yy.mean()); xc.append(zt[sm] - zt[sm].mean())
                if not yc:
                    continue
                Y, X = np.concatenate(yc), np.concatenate(xc)
                vx = float((X * X).sum())
                if vx <= 0:
                    continue
                Xall[:, j] = full[:, j] - float((X * Y).sum() / vx) * (p_med / 1000.0)

        for sname, cols in SETS.items():
            accs, pers = [], []
            for s in range(args.seeds):
                a, p, imp, _ = fit_eval(Xall[tr][:, cols], ytr, Xall[te][:, cols], yte, s)
                accs.append(a); pers.append(p)
            nulls = []
            for d in range(args.perm_draws):
                rng = np.random.default_rng(5000 + d)
                a, _, _, _ = fit_eval(Xall[tr][:, cols], rng.permutation(ytr),
                                      Xall[te][:, cols], yte, d)
                nulls.append(a)
            print(f"{pol:<20s}{sname:<17s}{args.seeds:>2d}"
                  f"{100 * np.mean(accs):8.2f}%{100 * np.mean(pers):9.2f}%"
                  f"{100 * np.mean(nulls):7.2f}%{100 / len(classes):6.2f}%"
                  f"{100 * maj:9.2f}%{len(tr):9d}")
            report[f"{pol}|{sname}"] = dict(
                policy=pol, feature_set=sname, n_features=len(cols),
                acc=float(np.mean(accs)), acc_sd=float(np.std(accs)),
                per_subject=float(np.mean(pers)), null=float(np.mean(nulls)),
                chance=1.0 / len(classes), majority=maj,
                n_train=int(len(tr)), n_test=int(len(te)))
        # which metric features carry it
        _, _, imp, _ = fit_eval(Xall[tr][:, metric], ytr, Xall[te][:, metric], yte, 0)
        if imp is not None:
            top = np.argsort(imp)[::-1][:6]
            print(f"{'':<20s}{'top metric cues':<17s}"
                  + "  ".join(f"{names[metric[t]]}={imp[t]:.3f}" for t in top))

    print("\nHOW TO READ THIS")
    print("  Compare the R4 row against the IMAGE floor's 5.35 % and against the CNN's best")
    print("  depth number (sil_scaled 14.59 %). Metric features are camera-invariant by")
    print("  construction, so if the geometry framing is right this is where they show it --")
    print("  a lower R0/R1 than the image floor is FINE and expected, since the image floor's")
    print("  89.4 % came from exactly the pixel-space nuisance that does not transfer.")
    print("  'metric+nuisance' exists to quantify that: adding stand_dist_mm should lift the")
    print("  within-session rungs and then fail at R4.")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "metric_floor.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\n[written] {os.path.join(args.out, 'metric_floor.json')}")


if __name__ == "__main__":
    main()
