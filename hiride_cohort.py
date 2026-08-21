#!/usr/bin/env python3
"""How does the operating point scale with COHORT SIZE?

    python hiride_cohort.py --prep $SCRATCH/hiride2/prep --runs $SCRATCH/hiride2/runs \
        --arch alexnet/stripe/aug8/tf10 --condition scale_removed --full-body \
        --out $SCRATCH/hiride2/results

49.90 % at R4 is on 28 enrolled people, where chance is 3.57 %. That figure is
bound to the gallery it was measured on, and identification gets harder as the
gallery grows. A deployment reader -- a care bedroom enrolling six residents, a
ward enrolling sixty -- needs the CURVE so they can find their own cohort on
it. This is one of the two open external-validity axes (handoff section 3); the
other is a second corpus, which no analysis of this one can supply.

TWO ARMS, WITH DIFFERENT STANDING:

  metric  RETRAINED per cohort. The RandomForest is refitted on exactly the K
          enrolled subjects, so this is a real measurement.
  cnn     APPROXIMATED. The network was trained once on all 28 classes; only
          the decision is restricted to the K enrolled columns. That is not the
          same as training on K, and it is stated rather than hidden -- wave 18
          retrains the CNN per cohort at K = 7/14/21 precisely so the gap
          between the two can be quantified instead of assumed.

THREE DRAWS PER K by default. One subset of seven people is a sample of an
easy-or-hard cohort, not a measurement of cohort size; the spread across draws
is reported next to the mean because at small K it is large.
"""
import os
import json
import argparse
import numpy as np

from hiride_data import load_manifest, make_split, eligible_mask
from hiride_fuse import cnn_cells, load_metric
from hiride_sequence import windows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--policy", default="R4_cross_session")
    ap.add_argument("--modality", default="depth")
    ap.add_argument("--arch", default="alexnet/stripe/aug8/tf10")
    ap.add_argument("--condition", default="scale_removed")
    ap.add_argument("--features", default="metric",
                    choices=("metric", "shape", "metric+shape"))
    ap.add_argument("--cohorts", default="4,7,10,14,21,28")
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--full-body", action="store_true")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    Ks = [int(x) for x in args.cohorts.split(",")]

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    full, have, cols, keep = load_metric(args.prep, man, args.features)
    subj_all = np.asarray(man["subject"], dtype=str)
    if args.full_body:
        zc = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
        gate = eligible_mask(zc["cues"], [str(f) for f in zc["feats"]], full_body=True)
        keep = keep & gate
    else:
        gate = None

    cells = cnn_cells(args.runs, args.policy, args.modality, args.arch, args.condition)
    if not cells:
        raise SystemExit(f"no runs for {args.arch} {args.condition} at {args.policy}")
    from sklearn.ensemble import RandomForestClassifier

    hdr = (f"{'K':>4s}{'chance':>9s}{'draws':>7s}{'cnn frame':>11s}{'metric frame':>14s}"
           f"{'cnn @W':>9s}{'metric @W':>12s}{'geo @W':>9s}{'spread @W':>11s}")
    print(f"\n{args.policy}  {args.modality}  {args.arch}  {args.condition}"
          f"{'  [full-body gated]' if args.full_body else ''}"
          f"   W={args.window} frames/decision\n")
    print(hdr); print("-" * len(hdr))
    report = {"_meta": dict(policy=args.policy, arch=args.arch,
                            condition=args.condition, features=args.features,
                            full_body=bool(args.full_body), window=args.window,
                            cohorts=Ks, draws=args.draws)}
    for K in Ks:
        per = {k: [] for k in ("cnn_f", "met_f", "cnn_w", "met_w", "geo_w")}
        for d in range(args.draws):
            accs = {k: [] for k in per}
            for meta, cm_path in cells:
                seed = int(meta["seed"])
                z = np.load(cm_path, allow_pickle=False)
                classes = [str(c) for c in z["classes"]]
                te_rows = z["test_rows"]
                sel = have[te_rows] & (gate[te_rows] if gate is not None else True)
                rows_s = te_rows[sel]
                truth = z["truth"].astype(int)[sel]
                prob = z["prob"].astype(np.float64)[sel]

                tr, _, _ = make_split(man, args.policy, seed=seed, keep=keep)
                common = sorted(set(subj_all[tr].tolist())
                                & set(subj_all[rows_s].tolist()))
                if K > len(common):
                    continue
                # SAME draw rule as hiride_train.py --cohort-seed, so the CPU
                # curve and wave 18's retrained points use identical cohorts
                pick = np.random.default_rng([d, K]).choice(
                    np.array(common), size=K, replace=False)
                cidx = np.array([classes.index(p) for p in pick if p in classes])
                if len(cidx) < K:
                    continue
                keepK = keep & np.isin(subj_all, pick)
                trK, _, _ = make_split(man, args.policy, seed=seed, keep=keepK)
                cmapK = {c: i for i, c in enumerate(sorted(pick.tolist()))}
                trK = trK[np.array([s in cmapK for s in subj_all[trK]], bool)]
                if len(trK) < 40:
                    continue
                mrow = np.isin(subj_all[rows_s], pick)
                if mrow.sum() < 30:
                    continue
                rs, tt = rows_s[mrow], truth[mrow]
                # CNN: restrict the decision to the enrolled columns only
                sub = prob[mrow][:, cidx]
                gold = np.array([list(cidx).index(classes.index(s))
                                 for s in subj_all[rs]])
                accs["cnn_f"].append(float((sub.argmax(1) == gold).mean()))
                # metric: genuinely retrained on the K enrolled subjects
                rf = RandomForestClassifier(n_estimators=300, random_state=seed,
                                            n_jobs=-1)
                rf.fit(full[trK][:, cols],
                       np.array([cmapK[s] for s in subj_all[trK]]))
                pm = np.zeros((len(rs), K))
                pm[:, rf.classes_.astype(int)] = rf.predict_proba(full[rs][:, cols])
                goldK = np.array([cmapK[s] for s in subj_all[rs]])
                accs["met_f"].append(float((pm.argmax(1) == goldK).mean()))
                # and the same, one decision per W frames within a recording
                sn = sub / np.clip(sub.sum(1, keepdims=True), 1e-12, None)
                pg = np.exp(np.log(sn + 1e-12) + np.log(pm + 1e-12))
                rec = np.array([f"{a}|{b}" for a, b in
                                zip(np.asarray(man["seq"], str)[rs], subj_all[rs])])
                frame = np.asarray(man["frame"])[rs].astype(np.int64)
                for name, P, g in (("cnn_w", sn, gold), ("met_w", pm, goldK),
                                   ("geo_w", pg, goldK)):
                    ok = n = 0
                    for r in np.unique(rec):
                        m = np.flatnonzero(rec == r)
                        m = m[np.argsort(frame[m])]
                        for blk in windows(m, args.window):
                            ok += int(np.log(P[blk] + 1e-12).mean(0).argmax() == g[blk[0]])
                            n += 1
                    accs[name].append(ok / max(n, 1))
            for k in per:
                if accs[k]:
                    per[k].append(float(np.mean(accs[k])))
        if not per["met_w"]:
            print(f"{K:>4d}   (no usable draws)")
            continue
        m = {k: 100 * float(np.mean(v)) for k, v in per.items() if v}
        spread = 100 * (max(per["geo_w"]) - min(per["geo_w"])) if per["geo_w"] else 0
        print(f"{K:>4d}{100/K:>8.2f}%{len(per['met_w']):>7d}{m['cnn_f']:>10.2f}%"
              f"{m['met_f']:>13.2f}%{m['cnn_w']:>8.2f}%{m['met_w']:>11.2f}%"
              f"{m['geo_w']:>8.2f}%{spread:>10.1f}pp")
        report[str(K)] = dict(mean=m, spread_geo_w=spread,
                              draws={k: v for k, v in per.items()})

    print("\nREAD: `metric` is RETRAINED per cohort and is a real measurement. `cnn` is")
    print("APPROXIMATED -- trained once on all 28 classes, with only the decision")
    print("restricted to the K enrolled columns; wave 18 retrains it per cohort so the")
    print("size of that approximation can be quantified. `spread` is the range across")
    print("draws: at small K a cohort can be easy or hard, and the mean hides that.")
    if args.out:
        path = os.path.join(args.out, "cohort.json")
        json.dump(report, open(path, "w"), indent=1)
        print(f"\n[written] {path}")


if __name__ == "__main__":
    main()
