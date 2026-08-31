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
from hiride_metric import BASE_METRIC, SHAPE_PREFIXES
from hiride_stats import cluster_boot, boot_rng

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
    ap.add_argument("--boot", type=int, default=20000,
                    help="subject-cluster bootstrap draws per (rung, set, seed). "
                         "Every CNN number in the paper carries a subject-level "
                         "interval; until now the metric floor's 19.04 %% at R4 "
                         "had none, with n = 28 subjects the binding constraint "
                         "as everywhere else.")
    ap.add_argument("--boot-seed", type=int, default=0)
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
    ap.add_argument("--test-eligibility", choices=("same", "full_body"),
                    default="same",
                    help="restrict only the TEST side. Full-body frames are "
                         "easier frames as well as better-measured ones, so "
                         "comparing a full_body run against the headline "
                         "conflates 'the features improved' with 'the test set "
                         "got easier'. Training on cues and testing on "
                         "full_body isolates the second; --eligibility "
                         "full_body then adds the first on top of it.")
    ap.add_argument("--invariance-max", type=float, default=None,
                    metavar="R",
                    help="add an `invariant` feature set: keep only columns "
                         "whose drift across the TRAINING range, divided by "
                         "their between-subject SD, is below R. Selection uses "
                         "training rows only -- no test frame, and no test "
                         "label, informs which columns survive. This is the "
                         "honest form of automatic feature selection here: "
                         "picking features by cross-session accuracy on 28 "
                         "subjects would be fitting the test set, whereas "
                         "range-invariance is a physics argument that can be "
                         "made before anyone looks at who is who. Try 0.5.")
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
    # The R1 reference counts must come from the STANDARD mask. Under
    # --eligibility full_body, guard 150 leaves subjects 036/037/045 with no
    # training frames at all and block_train_counts refuses rather than
    # silently dropping those classes -- the same infeasibility that killed
    # wave 15's R1 arm (13.3). match_ntrain only ever subsamples, so taking the
    # reference from the cues mask is conservative: the full-body arm trains on
    # at most what its partner does.
    keep_ref = eligible_mask(z["cues"], cue_feats, full_body=False)
    keep_test_fb = eligible_mask(z["cues"], cue_feats, full_body=True)
    p_med = z["cues"][:, cue_feats.index("p_med")].astype(np.float64)

    mf = np.load(os.path.join(args.prep, "metric_features.npz"), allow_pickle=False)
    F, names, rows = mf["feats"], [str(n) for n in mf["names"]], mf["manifest_row"]
    print(f"[load] {F.shape[0]} frames x {F.shape[1]} features")

    full = np.full((len(man["frame"]), F.shape[1]), np.nan, dtype=np.float32)
    full[rows] = F
    have = ~np.isnan(full[:, 0])
    keep = keep_cue & have
    print(f"[load] {int(have.sum())} frames have metric features; "
          f"{int(keep.sum())} also pass eligibility={args.eligibility}")
    seqs = np.asarray(man["seq"], dtype=str)
    for sname in sorted(set(seqs)):
        m = seqs == sname
        print(f"  {sname:<18s} {int((keep & m).sum()):>6d} / {int(m.sum()):>6d} kept "
              f"({100 * (keep & m).sum() / max(m.sum(), 1):5.1f} %)")

    bone = [i for i, n in enumerate(names) if n.startswith("bone_")]
    # `metric` is PINNED to the 12 published columns. Deriving it as
    # "everything that is not excluded" would have quietly redefined it the
    # moment head-anchored features were added to the npz, and every number in
    # section 8 would have stopped reproducing without anything looking wrong.
    metric = [i for i, n in enumerate(names) if n in BASE_METRIC]
    shape = [i for i, n in enumerate(names) if n.startswith(SHAPE_PREFIXES)]
    missing = set(BASE_METRIC) - {names[i] for i in metric}
    if missing:
        print(f"[warn] published metric columns absent from the npz: "
              f"{sorted(missing)} -- section 8 numbers will NOT reproduce")
    SETS = {"metric": metric}
    if shape:
        SETS["shape"] = shape
        SETS["metric+shape"] = metric + shape
    if bone:
        SETS["skeleton"] = bone
        SETS["both"] = metric + bone
    SETS["metric+nuisance"] = metric + [i for i, n in enumerate(names)
                                        if n == "stand_dist_mm"]
    if shape:
        SETS["shape+nuisance"] = shape + [i for i, n in enumerate(names)
                                          if n == "stand_dist_mm"]
    print("[sets] " + "  ".join(f"{k}={len(v)}" for k, v in SETS.items()))

    report = {}
    hdr = (f"{'split':<20s}{'feature set':<17s}{'n':>2s}{'acc':>9s}{'per-subj':>10s}"
           f"{'null':>8s}{'1/K':>7s}{'majority':>10s}{'n_train':>9s}"
           f"{'subj-boot 95% CI':>21s}")
    print("\n" + hdr); print("-" * len(hdr))
    for pol, kw in LADDER:
        try:
            if pol.startswith("R1"):
                kw = dict(kw, match_ntrain=block_train_counts(
                    man, guard=150, seed=0, keep=keep_ref & have))
            tr, va, te = make_split(man, pol, seed=0, keep=keep, **kw)
        except Exception as e:
            # skip the rung, never the ladder -- this runs unattended in the
            # overnight analysis chain
            print(f"{pol:<20s} unavailable: {type(e).__name__}: {e}")
            continue
        classes = sorted(set(man["subject"][tr].tolist()))
        cmap = {c: i for i, c in enumerate(classes)}
        ytr = np.array([cmap[s] for s in man["subject"][tr]])
        m_te = np.array([s in cmap for s in man["subject"][te]])
        te = te[m_te]
        if args.test_eligibility == "full_body":
            before = len(te)
            te = te[keep_test_fb[te]]
            print(f"{pol:<20s} test-side full_body keeps {len(te)}/{before} "
                  f"({100 * len(te) / max(before, 1):.1f} %) test frames")
        if args.range_match:
            lo, hi = np.percentile(p_med[tr], [5, 95])
            inband = (p_med[te] >= lo) & (p_med[te] <= hi)
            print(f"{pol:<20s} range-match {lo:.0f}-{hi:.0f} mm keeps "
                  f"{100 * inband.mean():.1f} % of test frames")
            te = te[inband]
            if len(te) < 50:
                print(f"{pol:<20s} SKIPPED: only {len(te)} test frames after "
                      f"range-match")
                continue
        yte = np.array([cmap[s] for s in man["subject"][te]])
        if len(te) < 50 or len(set(ytr.tolist())) < 2:
            # A silent `continue` here dropped every rung under
            # --eligibility full_body and printed an empty table, which reads
            # like "no effect" rather than "not runnable". Say which it is.
            print(f"{pol:<20s} SKIPPED: {len(tr)} train / {len(te)} test frames "
                  f"survive eligibility={args.eligibility} AND having metric "
                  f"features ({len(set(ytr.tolist()))} classes) -- need >=50 test")
            continue
        maj = float(np.bincount(yte).max() / len(yte))
        if args.invariance_max is not None:
            # Fixed-effects drift per candidate column, fitted on TRAIN rows.
            cand = sorted(set(metric) | set(shape))
            zt = p_med[tr] / 1000.0
            span = float(np.diff(np.percentile(p_med[tr], [5, 95]))[0]) / 1000.0
            subs = man["subject"][tr]
            kept, rej = [], []
            for j in cand:
                yc, xc, mu = [], [], []
                for u in sorted(set(subs.tolist())):
                    sm = subs == u
                    if sm.sum() < 20:
                        continue
                    yy = full[tr][sm, j].astype(np.float64)
                    if not np.isfinite(yy).all():
                        continue
                    mu.append(yy.mean())
                    yc.append(yy - yy.mean()); xc.append(zt[sm] - zt[sm].mean())
                if len(mu) < 5:
                    continue
                Y, X = np.concatenate(yc), np.concatenate(xc)
                vx = float((X * X).sum())
                slope = float((X * Y).sum() / vx) if vx > 0 else 0.0
                bsd = float(np.std(mu))
                ratio = abs(slope) * span / bsd if bsd > 1e-9 else np.inf
                (kept if ratio < args.invariance_max else rej).append((names[j], ratio, j))
            if kept:
                SETS["invariant"] = [j for _, _, j in kept]
                print(f"{pol:<20s} invariant<{args.invariance_max}: kept "
                      f"{len(kept)}/{len(cand)} -- "
                      + ", ".join(f"{n}({r:.2f})" for n, r, _ in sorted(kept,
                                                                       key=lambda t: t[1])[:8])
                      + (" ..." if len(kept) > 8 else ""))
            else:
                print(f"{pol:<20s} invariant<{args.invariance_max}: NOTHING kept "
                      f"of {len(cand)} candidates")
                SETS.pop("invariant", None)

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

        te_subj = np.asarray(man["subject"][te], dtype=str)
        for sname, cols in SETS.items():
            accs, pers, los, his = [], [], [], []
            for s in range(args.seeds):
                a, p, imp, pred = fit_eval(Xall[tr][:, cols], ytr, Xall[te][:, cols], yte, s)
                accs.append(a); pers.append(p)
                # Subject-cluster CI per seed, mean of bounds over seeds --
                # exactly hiride_stats.py's convention for the CNN cells, with
                # the interval's RNG keyed on the quantity's own identity so it
                # cannot depend on processing order (13.7).
                lo, hi = cluster_boot((pred == yte).astype(float), te_subj,
                                      boot_rng(args.boot_seed,
                                               ("mfloor", pol, sname, s,
                                                args.eligibility,
                                                args.test_eligibility)),
                                      args.boot)
                los.append(lo); his.append(hi)
            nulls = []
            for d in range(args.perm_draws):
                rng = np.random.default_rng(5000 + d)
                a, _, _, _ = fit_eval(Xall[tr][:, cols], rng.permutation(ytr),
                                      Xall[te][:, cols], yte, d)
                nulls.append(a)
            flag = " *" if np.mean(los) > maj else ""
            print(f"{pol:<20s}{sname:<17s}{args.seeds:>2d}"
                  f"{100 * np.mean(accs):8.2f}%{100 * np.mean(pers):9.2f}%"
                  f"{100 * np.mean(nulls):7.2f}%{100 / len(classes):6.2f}%"
                  f"{100 * maj:9.2f}%{len(tr):9d}"
                  f"   [{100 * np.mean(los):5.2f}, {100 * np.mean(his):5.2f}]{flag}")
            report[f"{pol}|{sname}"] = dict(
                policy=pol, feature_set=sname, n_features=len(cols),
                acc=float(np.mean(accs)), acc_sd=float(np.std(accs)),
                per_subject=float(np.mean(pers)), null=float(np.mean(nulls)),
                subj_ci_lo=float(np.mean(los)), subj_ci_hi=float(np.mean(his)),
                boot=args.boot, chance=1.0 / len(classes), majority=maj,
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
