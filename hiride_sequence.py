#!/usr/bin/env python3
"""How much identity does depth recover from a SEQUENCE rather than one frame?

    python hiride_sequence.py --prep $SCRATCH/hiride2/prep --runs $SCRATCH/hiride2/runs \
        --condition sil_scaled --out $SCRATCH/hiride2/results

Author's goal, 2026-08-21: bring depth at R4 up to what colour reaches at R3
(79.5 %). Per FRAME the best measured single-frame result is 19-22 % (13.10),
a long way short. Note that the `either-correct` rate of 27.6 % reported there
is NOT a ceiling -- it bounds rules that SELECT between two models, not rules
that combine their scores, and it says nothing at all about a retrained model
or about the aggregation done here.

But one frame is not the budget a deployment has. RGB's 79.5 % at R3 is a
single-frame number only because clothing is a high-entropy cue that needs no
integration; a depth system watching someone walk gets ~200 frames, and this
campaign has been discarding all but one of them at a time. Sensor noise on a
body measurement is close to independent across frames, so it averages down;
the person does not change.

This script spends nothing to find out. The posteriors are already stored per
frame (`cm_*.npz`), so it aggregates them over consecutive windows WITHIN one
recording and reports accuracy against window length, for the CNN, the metric
RandomForest, and their log-space fusion.

WHAT TO WATCH. Accuracy must rise with window length, but it cannot rise
without limit: frames inside a recording are strongly correlated, so the
effective sample size is far below the window length, and any SYSTEMATIC
confusion -- subject A's body genuinely resembling subject B's -- never
averages out no matter how long you watch. Where the curve flattens is the
honest answer to "how much identity is in a depth observation of this person",
and the last row (whole tracklet) is what a deployment would actually achieve.

The decision count falls as the window grows: at whole-tracklet there is ONE
decision per subject per recording, so 28 decisions and a wide interval. That
is reported alongside, because an accuracy over 28 decisions is not the same
kind of number as one over 5,642 and must not be read as though it were.
"""
import os
import json
import argparse
import numpy as np

from hiride_data import load_manifest, make_split, eligible_mask
from hiride_stats import cluster_boot, boot_rng
from hiride_fuse import cnn_cells, load_metric, select_columns


def agg_windows(F, rec, frame, w, stride):
    """Window-mean feature vectors and the row index each window starts at.

    Averaging the MEASUREMENT beats averaging the posterior when the error is
    sensor noise: 25 frames cut a stature error by five, and the classifier then
    sees a sharper feature vector instead of a blurrier vote. Posterior
    averaging can only reweight decisions already made on noisy inputs.

    Training uses stride 1 (sliding windows) so shortening the window does not
    also shrink the training set by a factor of w -- at w=25 non-overlapping
    windows would leave ~14 examples per subject. Test uses stride w, so no test
    frame is counted twice and the decision count is honest.
    """
    blocks = []
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        m = m[np.argsort(frame[m])]
        if w <= 0 or w >= len(m):
            blocks.append(m)
            continue
        blocks.extend(m[i:i + w] for i in range(0, len(m) - w + 1, stride))
    X = np.stack([F[b].mean(0) for b in blocks]).astype(np.float32)
    return X, blocks


def windows(order, w):
    """Consecutive non-overlapping index blocks of length w within one recording."""
    if w <= 0 or w >= len(order):
        return [order]
    return [order[i:i + w] for i in range(0, len(order) - w + 1, w)]


def sequence_acc(prob, truth, rec, frame, w, agg="geo"):
    """Accuracy of one decision per window, windows taken within a recording."""
    ok, n = 0, 0
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        m = m[np.argsort(frame[m])]
        for blk in windows(m, w):
            p = (np.log(prob[blk] + 1e-12).mean(0) if agg == "geo"
                 else prob[blk].mean(0))
            ok += int(p.argmax() == truth[blk[0]])
            n += 1
    return ok / max(n, 1), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--policy", default="R4_cross_session")
    ap.add_argument("--modality", default="depth")
    ap.add_argument("--arch", default="alexnet")
    ap.add_argument("--condition", action="append", default=None)
    ap.add_argument("--windows", default="1,2,5,10,25,50,100,0",
                    help="frames per decision; 0 = the whole tracklet")
    ap.add_argument("--features", default="metric",
                    choices=("metric", "shape", "metric+shape"),
                    help="metric+shape reaches 32.06 %% at R4 frame-level "
                         "against 28.06 %% for the pinned 12")
    ap.add_argument("--invariance-max", type=float, default=None, metavar="R")
    ap.add_argument("--min-snr", type=float, default=0.0, metavar="S",
                    help="with --invariance-max, also require between-subject "
                         "SD over within-subject SD above S. Drift alone keeps "
                         "range-stable but uninformative columns.")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full-body", action="store_true",
                    help="score only frames whose whole body is in shot. "
                         "hiride_metric_floor.py puts unclipped frames at "
                         "28.06 %% against 9.3 %% for clipped ones at R4, "
                         "because a clipped body yields a wrong stature -- so "
                         "this is a validity gate, not a difficulty filter. "
                         "Rejecting frames a sensor cannot measure is a real "
                         "deployment choice; the retained fraction is printed "
                         "and the resulting accuracy is NOT comparable to the "
                         "unrestricted headline.")
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-name", default="sequence.json",
                    help="filename within --out. Gated and ungated runs must "
                         "not overwrite each other; figure 7 reads several.")
    args = ap.parse_args()
    conds = args.condition or ["sil_scaled", "scale_removed"]
    W = [int(x) for x in args.windows.split(",")]

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    full, have, cols, keep = load_metric(args.prep, man, args.features)
    gate = None
    if args.full_body:
        zc = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
        gate = eligible_mask(zc["cues"], [str(f) for f in zc["feats"]],
                             full_body=True)
        print(f"[gate] full-body frames: {int(gate.sum())} of {len(gate)}")
    zc2 = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    p_med_all = zc2["cues"][:, [str(f) for f in zc2["feats"]].index("p_med")].astype(float)
    names_all = [str(n) for n in np.load(
        os.path.join(args.prep, "metric_features.npz"), allow_pickle=False)["names"]]
    from sklearn.ensemble import RandomForestClassifier

    report = {}
    for cond in conds:
        cells = cnn_cells(args.runs, args.policy, args.modality, args.arch, cond)
        if not cells:
            print(f"\n{cond}: no runs with per-frame posteriors -- skipped")
            continue
        print(f"\n=== {args.policy}  {args.modality}  {args.arch}  {cond} ===")
        KEYS = ("cnn", "metric", "geo", "met_agg", "geo_agg")
        acc = {k: {w: [] for w in W} for k in KEYS}
        ndec = {w: [] for w in W}
        tail = {k: [] for k in ("cnn", "metric", "geo")}
        for meta, cm_path in cells:
            seed = int(meta["seed"])
            d = np.load(cm_path, allow_pickle=False)
            te_rows, classes = d["test_rows"], [str(c) for c in d["classes"]]
            sel = have[te_rows]
            if gate is not None:
                sel = sel & gate[te_rows]
            rows_s = te_rows[sel]
            if len(rows_s) < 50:
                print(f"  seed {seed}: only {len(rows_s)} test frames survive "
                      f"the gate -- skipped")
                continue
            p_cnn = d["prob"].astype(np.float64)[sel]
            p_cnn /= np.clip(p_cnn.sum(1, keepdims=True), 1e-12, None)
            truth = d["truth"].astype(int)[sel]

            cmap = {c: i for i, c in enumerate(classes)}
            tr, _, _ = make_split(man, args.policy, seed=seed,
                                  keep=(keep & gate) if gate is not None else keep)
            tr = tr[np.array([str(s) in cmap for s in man["subject"][tr]], bool)]
            ytr = np.array([cmap[str(s)] for s in man["subject"][tr]])
            use = cols
            if args.invariance_max is not None:
                use, table = select_columns(
                    full, cols, tr, np.asarray(man["subject"], str)[tr],
                    p_med_all, args.invariance_max, args.min_snr, names_all)
                if not use:
                    print(f"  seed {seed}: selection kept 0 of {len(cols)} "
                          f"columns -- loosen the thresholds"); continue
                if seed == int(cells[0][0]["seed"]):
                    top = sorted(table, key=lambda t: -t[2])[:8]
                    print(f"  [select] kept {len(use)}/{len(cols)} "
                          f"(drift<{args.invariance_max}, snr>{args.min_snr}); "
                          f"best by snr: "
                          + ", ".join(f"{n}({d:.2f}/{sn:.2f})" for n, d, sn in top))
            rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
            rf.fit(full[tr][:, use], ytr)
            p_rf = np.zeros_like(p_cnn)
            p_rf[:, rf.classes_.astype(int)] = rf.predict_proba(full[rows_s][:, use])
            p_geo = np.exp(np.log(p_cnn + 1e-12) + np.log(p_rf + 1e-12))
            p_geo /= np.clip(p_geo.sum(1, keepdims=True), 1e-12, None)

            # a "recording" is one subject in one sequence/session
            rec = np.array([f"{a}|{b}|{c}" for a, b, c in
                            zip(np.asarray(man["seq"], str)[rows_s],
                                np.asarray(man["subject"], str)[rows_s],
                                np.asarray(man["session"], str)[rows_s])])
            frame = np.asarray(man["frame"])[rows_s].astype(np.int64)
            # training-side recordings, for the window-averaged feature model
            tr_rec = np.array([f"{a}|{b}|{c}" for a, b, c in
                               zip(np.asarray(man["seq"], str)[tr],
                                   np.asarray(man["subject"], str)[tr],
                                   np.asarray(man["session"], str)[tr])])
            tr_frame = np.asarray(man["frame"])[tr].astype(np.int64)
            for w in W:
                for k, P in (("cnn", p_cnn), ("metric", p_rf), ("geo", p_geo)):
                    a, n = sequence_acc(P, truth, rec, frame, w)
                    acc[k][w].append(a)
                    if k == "cnn":
                        ndec[w].append(n)
                # average the MEASUREMENT over the window, then classify
                Xw, btr = agg_windows(full[tr][:, use], tr_rec, tr_frame, w, 1)
                yw = np.array([ytr[b[0]] for b in btr])
                rfw = RandomForestClassifier(n_estimators=300, random_state=seed,
                                             n_jobs=-1)
                rfw.fit(Xw, yw)
                # ONE set of test blocks, shared by both aggregations, so the
                # feature-averaged and posterior-averaged columns are decided on
                # exactly the same frames and are directly comparable
                Xt, bte = agg_windows(full[rows_s][:, use], rec, frame, w, max(w, 1))
                yte = np.array([truth[b[0]] for b in bte])
                pw = np.zeros((len(bte), p_cnn.shape[1]))
                pw[:, rfw.classes_.astype(int)] = rfw.predict_proba(Xt)
                acc["met_agg"][w].append(float((pw.argmax(1) == yte).mean()))
                pc = np.stack([np.log(p_cnn[b] + 1e-12).mean(0) for b in bte])
                acc["geo_agg"][w].append(
                    float(((pc + np.log(pw + 1e-12)).argmax(1) == yte).mean()))
            # per-tracklet correctness, for a subject-clustered interval
            for k, P in (("cnn", p_cnn), ("metric", p_rf), ("geo", p_geo)):
                cor, sub = [], []
                for r in np.unique(rec):
                    m = np.flatnonzero(rec == r)
                    p = np.log(P[m] + 1e-12).mean(0)
                    cor.append(float(p.argmax() == truth[m[0]]))
                    sub.append(r.split("|")[1])
                tail[k].append((np.array(cor), np.array(sub)))

        hdr = (f"{'frames/decision':>16s}{'decisions':>11s}{'cnn':>9s}"
               f"{'metric':>9s}{'geo':>9s}{'met_agg':>10s}{'geo_agg':>10s}")
        print(hdr); print("-" * len(hdr))
        for w in W:
            lab = "whole tracklet" if w == 0 else str(w)
            print(f"{lab:>16s}{int(np.mean(ndec[w])):>11d}"
                  + "".join(f"{100 * np.mean(acc[k][w]):>8.2f}%"
                            for k in ("cnn", "metric", "geo"))
                  + "".join(f"{100 * np.mean(acc[k][w]):>9.2f}%"
                            for k in ("met_agg", "geo_agg")))
        print("  whole-tracklet subject-cluster CI (mean over seeds):")
        cis = {}
        for k in ("cnn", "metric", "geo"):
            per = [cluster_boot(c, s, boot_rng(args.seed, ("seq", cond, k, i)), args.boot)
                   for i, (c, s) in enumerate(tail[k])]
            cis[k] = [float(np.mean([p[0] for p in per])),
                      float(np.mean([p[1] for p in per]))]
            print(f"    {k:<8s}{100 * np.mean(acc[k][0]):>7.2f}%  "
                  f"[{100 * cis[k][0]:+.1f}, {100 * cis[k][1]:+.1f}]")
        report[cond] = dict(windows=W,
                            acc={k: {str(w): float(np.mean(v)) for w, v in d.items()}
                                 for k, d in acc.items()},
                            n_decisions={str(w): float(np.mean(v)) for w, v in ndec.items()},
                            tracklet_ci=cis)

    if args.out:
        # settings travel WITH the numbers: a curve read months later must say
        # whether it was gated and which features it used, or it is unreadable
        report["_meta"] = dict(policy=args.policy, modality=args.modality,
                               arch=args.arch, features=args.features,
                               full_body=bool(args.full_body),
                               invariance_max=args.invariance_max,
                               min_snr=args.min_snr, windows=W)
        path = os.path.join(args.out, args.out_name)
        json.dump(report, open(path, "w"), indent=1)
        print(f"\n[written] {path}")
    print("\nREAD: frames inside a recording are strongly correlated, so the effective "
          "sample size\nis far below the window length, and systematic confusions never "
          "average out. Where the\ncurve FLATTENS is the honest answer, not where it "
          "ends. The whole-tracklet row is one\ndecision per subject -- 28 of them -- "
          "so read its interval, not its point estimate.")


if __name__ == "__main__":
    main()
