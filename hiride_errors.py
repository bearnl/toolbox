#!/usr/bin/env python3
"""Error structure of a CNN cell and the metric random forest on identical decisions.

    python hiride_errors.py --prep $SCRATCH/hiride2/prep --runs $SCRATCH/hiride2/runs \
        --arch alexnet/stripe/aug8/tf10 --full-body --out $SCRATCH/hiride2/results

Why this exists (HIRIDE_HANDOFF 14.9; 09_cnn_failure_refute_ERRORS). The manuscript wants
to write that the network's cross-session errors are SYSTEMATIC -- the same wrong identity
on every frame of a clip, which integration cannot average away -- while the measurements'
errors are partly independent and do average. Two measurements must exist before that
sentence is licensed, and this script makes them from stored arrays without training:

  1. The veto. hiride_train.py stores posteriors as float16 and hiride_sequence.py
     aggregates with the product rule (mean log posterior, floor 1e-12). One frame whose
     true-class probability rounded to zero contributes log(1e-12) ~ -27.6 against typical
     per-frame log posteriors of -1 to -5 and can veto the whole window. Every window is
     therefore decided twice, by the product rule and by the arithmetic mean of posteriors,
     and the number of vetoing frames is counted. If the arithmetic mean recovers accuracy,
     the flat integration curve was partly an artefact of the rule, not of the network.
  2. Vote consistency. Within a window the model gets wrong, do the per-frame decisions
     concentrate on one wrong identity (modal share high, vote entropy low, the window's
     decision persisting frame after frame) or spread (noise-like)? Reported for the CNN,
     the forest and their fusion, split by whether the window decision was right.

Also reported, because each bounds a sentence in Section VI: the lag-k agreement between
wrong per-frame decisions within a recording; the i.i.d. ceiling -- the accuracy W frames
drawn independently from each class's frame-level confusion row would give, whose gap to
the measured curve is the effective correlation of the errors; how stable each recording's
modal confuser is across seeds (a confuser that recurs across seeds is a property of the
data, one that does not is a property of the fit); whether both models fail on the same
clipped or out-of-range frames; and, for cells trained with --track-test, what stopping on
within-session validation cost against the best test epoch.

Cells are located through hiride_fuse.cnn_cells, so the cell key is never re-derived here.
"""
import os
import json
import argparse
from math import ceil

import numpy as np

from hiride_data import load_manifest, make_split, eligible_mask
from hiride_fuse import cnn_cells, load_metric


def windows(order, w):
    """Consecutive non-overlapping index blocks of length w within one recording; w<=0 = whole."""
    if w <= 0 or w >= len(order):
        return [order]
    return [order[i:i + w] for i in range(0, len(order) - w + 1, w)]


def log_aggregate(P, blk, rule):
    """Window log-score under the product rule ('geo') or the arithmetic mean of posteriors."""
    if rule == "geo":
        return np.log(P[blk] + 1e-12).mean(0)
    return np.log(P[blk].mean(0) + 1e-12)


def window_rows(P, truth, rec, frame, w):
    """One record per window: decisions under both rules and the vote-consistency statistics."""
    out = []
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        m = m[np.argsort(frame[m])]
        for blk in windows(m, w):
            t = int(truth[blk[0]])
            votes = P[blk].argmax(1)
            counts = np.bincount(votes, minlength=P.shape[1])
            f = counts[counts > 0] / float(len(blk))
            entropy = float(-(f * np.log(f)).sum() / np.log(len(blk))) if len(blk) > 1 else 0.0
            lp_geo = log_aggregate(P, blk, "geo")
            d_geo = int(lp_geo.argmax())
            d_mean = int(log_aggregate(P, blk, "mean").argmax())
            out.append(dict(rec=str(r), subject=str(r).split("|")[1], truth=t, n=int(len(blk)),
                            d_geo=d_geo, d_mean=d_mean, modal=int(counts.argmax()),
                            share=float(counts.max() / len(blk)), entropy=entropy,
                            persist=float((votes == d_geo).mean()),
                            rank_geo=int((lp_geo > lp_geo[t]).sum()) + 1,
                            veto=int((P[blk, t] <= 0).sum())))
    return out


def lag_agreement(pred, truth, rec, frame, lags):
    """P(a_{t+k} == a_t | a_t wrong), pooled over recordings, for each lag k."""
    agree = {k: [0, 0] for k in lags}
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        m = m[np.argsort(frame[m])]
        a, t = pred[m], truth[m]
        for k in lags:
            if len(m) <= k:
                continue
            wrong = a[:-k] != t[:-k]
            agree[k][0] += int((a[:-k][wrong] == a[k:][wrong]).sum())
            agree[k][1] += int(wrong.sum())
    return {str(k): (v[0] / v[1] if v[1] else float("nan")) for k, v in agree.items()}


def iid_ceiling(pred, truth, n_classes, W, sims, rng):
    """Accuracy of a plurality vote over W frames drawn i.i.d. from each class's confusion row."""
    out = {}
    classes = np.unique(truth)
    weights = np.array([(truth == c).sum() for c in classes], dtype=float)
    weights /= weights.sum()
    rows = {}
    for c in classes:
        q = np.bincount(pred[truth == c], minlength=n_classes).astype(float)
        rows[c] = q / q.sum()
    for w in W:
        ww = w if w > 0 else 0
        if ww <= 0:
            continue
        acc = 0.0
        for c, wt in zip(classes, weights):
            draws = rng.choice(n_classes, size=(sims, ww), p=rows[c])
            plural = np.array([np.bincount(d, minlength=n_classes).argmax() for d in draws])
            acc += wt * float((plural == c).mean())
        out[str(w)] = acc
    return out


def modal_confuser(pred, truth, rec):
    """Per recording, the most frequent wrong identity (or None if never wrong)."""
    out = {}
    for r in np.unique(rec):
        m = np.flatnonzero(rec == r)
        wrong = pred[m][pred[m] != truth[m]]
        out[str(r)] = int(np.bincount(wrong).argmax()) if len(wrong) else None
    return out


def confuser_stability(per_seed):
    """Fraction of recordings whose modal confuser agrees in at least 80 % of seeds."""
    recs = set().union(*[set(d) for d in per_seed])
    need = ceil(0.8 * len(per_seed))
    stable = 0
    for r in recs:
        vals = [d.get(r) for d in per_seed if d.get(r) is not None]
        if len(vals) >= need and np.bincount(vals).max() >= need:
            stable += 1
    return stable / max(len(recs), 1)


def stopping_cost(meta):
    """Best test epoch minus the selected epoch, from a --track-test cell's stored curve."""
    curve, best = meta.get("test_curve"), meta.get("best_epoch")
    if not curve or best is None or best >= len(curve):
        return None
    return dict(best_epoch=int(best), epochs_run=int(meta.get("epochs_run", len(curve))),
                hit_epoch_cap=bool(meta.get("hit_epoch_cap", False)),
                test_at_selected=float(curve[best]), test_max=float(max(curve)),
                argmax_epoch=int(np.argmax(curve)), cost_pp=100 * (float(max(curve)) - float(curve[best])))


def summarise(rows, key_decision):
    """Mean vote statistics split by whether the window decision under `key_decision` was right."""
    right = [r for r in rows if r[key_decision] == r["truth"]]
    wrong = [r for r in rows if r[key_decision] != r["truth"]]

    def stats(rs):
        if not rs:
            return None
        return dict(n=len(rs), share=float(np.mean([r["share"] for r in rs])),
                    entropy=float(np.mean([r["entropy"] for r in rs])),
                    persist=float(np.mean([r["persist"] for r in rs])),
                    veto_frames=float(np.mean([r["veto"] for r in rs])),
                    runner_up_2_3=float(np.mean([2 <= r["rank_geo"] <= 3 for r in rs])))
    return dict(right=stats(right), wrong=stats(wrong))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--policy", default="R4_cross_session")
    ap.add_argument("--modality", default="depth")
    ap.add_argument("--arch", required=True, help="hiride_keys arch key, e.g. alexnet/stripe/aug8/tf10")
    ap.add_argument("--condition", default="scale_removed")
    ap.add_argument("--full-body", action="store_true")
    ap.add_argument("--window", type=int, default=25)
    ap.add_argument("--windows", default="1,2,5,10,25,50,100,0")
    ap.add_argument("--sims", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-name", default=None)
    args = ap.parse_args()
    W = [int(x) for x in args.windows.split(",")]

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    full, have, cols, keep = load_metric(args.prep, man, "metric")
    zc = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    feats = [str(f) for f in zc["feats"]]
    cues = zc["cues"]
    gate = eligible_mask(cues, feats, full_body=True) if args.full_body else None
    bot = cues[:, feats.index("bot_touch")] > 0
    p_med = cues[:, feats.index("p_med")].astype(float)
    from sklearn.ensemble import RandomForestClassifier

    cells = cnn_cells(args.runs, args.policy, args.modality, args.arch, args.condition)
    if not cells:
        raise SystemExit(f"no cells with posteriors for {args.policy} {args.modality} {args.arch} {args.condition}")
    print(f"=== {args.policy}  {args.modality}  {args.arch}  {args.condition}  "
          f"{'gated' if args.full_body else 'ungated'}  {len(cells)} seeds ===")

    rng = np.random.default_rng(args.seed)
    acc = {m: {rule: {str(w): [] for w in W} for rule in ("geo", "mean")} for m in ("cnn", "metric", "geo")}
    ndec = {str(w): [] for w in W}
    consistency = {m: [] for m in ("cnn", "metric", "geo")}
    lags = {m: [] for m in ("cnn", "metric")}
    ceiling = {m: [] for m in ("cnn", "metric")}
    confusers = {m: [] for m in ("cnn", "metric")}
    both_wrong = []
    stopping = []
    for meta, cm_path in cells:
        seed = int(meta["seed"])
        d = np.load(cm_path, allow_pickle=False)
        te_rows, classes = d["test_rows"], [str(c) for c in d["classes"]]
        sel = have[te_rows]
        if gate is not None:
            sel = sel & gate[te_rows]
        rows_s = te_rows[sel]
        if len(rows_s) < 50:
            print(f"  seed {seed}: only {len(rows_s)} frames survive the gate -- skipped")
            continue
        p_cnn = d["prob"].astype(np.float64)[sel]
        p_cnn /= np.clip(p_cnn.sum(1, keepdims=True), 1e-12, None)
        truth = d["truth"].astype(int)[sel]
        cmap = {c: i for i, c in enumerate(classes)}
        tr, _, _ = make_split(man, args.policy, seed=seed,
                              keep=(keep & gate) if gate is not None else keep)
        tr = tr[np.array([str(s) in cmap for s in man["subject"][tr]], bool)]
        ytr = np.array([cmap[str(s)] for s in man["subject"][tr]])
        rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
        rf.fit(full[tr][:, cols], ytr)
        p_rf = np.zeros_like(p_cnn)
        p_rf[:, rf.classes_.astype(int)] = rf.predict_proba(full[rows_s][:, cols])
        p_geo = np.exp(np.log(p_cnn + 1e-12) + np.log(p_rf + 1e-12))
        p_geo /= np.clip(p_geo.sum(1, keepdims=True), 1e-12, None)
        rec = np.array([f"{a}|{b}|{c}" for a, b, c in zip(np.asarray(man["seq"], str)[rows_s],
                                                           np.asarray(man["subject"], str)[rows_s],
                                                           np.asarray(man["session"], str)[rows_s])])
        frame = np.asarray(man["frame"])[rows_s].astype(np.int64)

        for m, P in (("cnn", p_cnn), ("metric", p_rf), ("geo", p_geo)):
            for w in W:
                rows = window_rows(P, truth, rec, frame, w)
                acc[m]["geo"][str(w)].append(float(np.mean([r["d_geo"] == r["truth"] for r in rows])))
                acc[m]["mean"][str(w)].append(float(np.mean([r["d_mean"] == r["truth"] for r in rows])))
                if m == "cnn":
                    ndec[str(w)].append(len(rows))
                if w == args.window:
                    consistency[m].append(summarise(rows, "d_geo"))
        for m, P in (("cnn", p_cnn), ("metric", p_rf)):
            pred = P.argmax(1)
            lags[m].append(lag_agreement(pred, truth, rec, frame, (1, 5, 10)))
            ceiling[m].append(iid_ceiling(pred, truth, P.shape[1], W, args.sims, rng))
            confusers[m].append(modal_confuser(pred, truth, rec))

        c_wrong = p_cnn.argmax(1) != truth
        r_wrong = p_rf.argmax(1) != truth
        lo, hi = np.percentile(p_med[tr], [5, 95])
        oob = (p_med[rows_s] < lo) | (p_med[rows_s] > hi)
        bw = c_wrong & r_wrong
        er = ~bw
        both_wrong.append(dict(frac_both_wrong=float(bw.mean()),
                               bot_touch_both_wrong=float(bot[rows_s][bw].mean()) if bw.any() else None,
                               bot_touch_either_right=float(bot[rows_s][er].mean()) if er.any() else None,
                               oob_both_wrong=float(oob[bw].mean()) if bw.any() else None,
                               oob_either_right=float(oob[er].mean()) if er.any() else None,
                               independence_either=float(1 - (c_wrong.mean() * r_wrong.mean())),
                               measured_either=float(1 - bw.mean())))
        sc = stopping_cost(meta)
        if sc:
            stopping.append(sc)

    def mean_of(lst, key):
        vals = [x[key] for x in lst if x.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    print(f"\n{'frames/decision':>16s}{'decisions':>10s}" + "".join(f"{m + '/' + r:>13s}" for m in ("cnn", "metric", "geo") for r in ("geo", "mean")))
    for w in W:
        lab = "whole tracklet" if w == 0 else str(w)
        print(f"{lab:>16s}{int(np.mean(ndec[str(w)])):>10d}"
              + "".join(f"{100 * np.mean(acc[m][r][str(w)]):>12.2f}%" for m in ("cnn", "metric", "geo") for r in ("geo", "mean")))
    print("  product rule (geo) vs arithmetic mean (mean), same decisions; a gap is the float16 veto")

    print(f"\nvote consistency at W={args.window} (mean over seeds; share = modal vote fraction, "
          f"entropy normalised, persist = frames voting the window's own decision):")
    for m in ("cnn", "metric", "geo"):
        for side in ("right", "wrong"):
            parts = [c[side] for c in consistency[m] if c and c[side]]
            if not parts:
                continue
            print(f"  {m:<7s}{side:<6s} n={np.mean([p['n'] for p in parts]):6.1f}  share={np.mean([p['share'] for p in parts]):.2f}  "
                  f"entropy={np.mean([p['entropy'] for p in parts]):.2f}  persist={np.mean([p['persist'] for p in parts]):.2f}  "
                  f"veto frames/window={np.mean([p['veto_frames'] for p in parts]):.2f}  "
                  f"true class ranked 2-3={100 * np.mean([p['runner_up_2_3'] for p in parts]):.0f}%")
    print("\nlag-k agreement of WRONG per-frame decisions within a recording, P(a_t+k = a_t | a_t wrong):")
    for m in ("cnn", "metric"):
        print(f"  {m:<7s}" + "  ".join(f"k={k}: {np.nanmean([l[k] for l in lags[m]]):.2f}" for k in ("1", "5", "10")))
    print("\ni.i.d. ceiling (plurality of W frames drawn independently from the frame-level confusion rows):")
    for m in ("cnn", "metric"):
        print(f"  {m:<7s}" + "  ".join(f"W={w}: {100 * np.mean([c[str(w)] for c in ceiling[m]]):.1f}% (measured {100 * np.mean(acc[m]['geo'][str(w)]):.1f}%)"
                                        for w in W if w > 0))
    print("\nmodal confuser stable across >=80% of seeds:")
    for m in ("cnn", "metric"):
        print(f"  {m:<7s}{100 * confuser_stability(confusers[m]):.0f}% of recordings")
    print("\nboth-wrong frames (frame level, this frame set):")
    print(f"  both wrong {100 * mean_of(both_wrong, 'frac_both_wrong'):.1f}%  |  either right measured "
          f"{100 * mean_of(both_wrong, 'measured_either'):.1f}% vs independence {100 * mean_of(both_wrong, 'independence_either'):.1f}%")
    bt_bw, bt_er = mean_of(both_wrong, "bot_touch_both_wrong"), mean_of(both_wrong, "bot_touch_either_right")
    ob_bw, ob_er = mean_of(both_wrong, "oob_both_wrong"), mean_of(both_wrong, "oob_either_right")
    if bt_bw is not None and bt_er is not None:
        print(f"  body clipped: {100 * bt_bw:.1f}% of both-wrong frames vs {100 * bt_er:.1f}% of either-right")
    if ob_bw is not None and ob_er is not None:
        print(f"  outside training distance band: {100 * ob_bw:.1f}% of both-wrong vs {100 * ob_er:.1f}% of either-right")
    if stopping:
        print("\nearly stopping vs best test epoch (--track-test cells):")
        for s in stopping:
            print(f"  selected epoch {s['best_epoch']:3d} (test {100 * s['test_at_selected']:.2f}%)  best epoch "
                  f"{s['argmax_epoch']:3d} (test {100 * s['test_max']:.2f}%)  cost {s['cost_pp']:.2f} pp  "
                  f"ran {s['epochs_run']}{' cap' if s['hit_epoch_cap'] else ''}")
        print(f"  mean cost {np.mean([s['cost_pp'] for s in stopping]):.2f} pp over {len(stopping)} seeds")
    else:
        print("\nno --track-test curves in these cells; early-stopping cost not measurable here")

    if args.out:
        report = dict(_meta=dict(policy=args.policy, modality=args.modality, arch=args.arch,
                                 condition=args.condition, full_body=bool(args.full_body),
                                 window=args.window, seeds=len(cells)),
                      acc={m: {r: {w: float(np.mean(v)) for w, v in d.items()} for r, d in dd.items()} for m, dd in acc.items()},
                      n_decisions={w: float(np.mean(v)) for w, v in ndec.items()},
                      consistency=consistency, lags=lags, ceiling=ceiling,
                      confuser_stability={m: confuser_stability(confusers[m]) for m in confusers},
                      both_wrong=both_wrong, stopping=stopping)
        name = args.out_name or (f"errors_{args.arch.replace('/', '-')}_{args.policy}_"
                                 f"{'gated' if args.full_body else 'ungated'}.json")
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, name), "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\n[written] {os.path.join(args.out, name)}")


if __name__ == "__main__":
    main()
