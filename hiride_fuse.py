#!/usr/bin/env python3
"""Are the metric 3D features and the CNN reading the SAME identity signal?

    python hiride_fuse.py --prep $SCRATCH/hiride2/prep --runs $SCRATCH/hiride2/runs \
        --condition sil_scaled --condition scale_removed --out $SCRATCH/hiride2/results

At R4 the hand-built metric scalars reach 19.04 % and the best CNN 18.01 %, from
completely disjoint representations of identical frames: millimetres of stature
and width profile on one side, learned pixels on the other. Two numbers that
close together can mean either of two things, and they have opposite
consequences for the paper:

  REDUNDANT  -- both are reading the same small pool of body-shape information,
                and ~19 % is the information content of a single Kinect v1 frame
                of a person. Fusion buys nothing and the ceiling claim is firm.
  COMPLEMENTARY -- they fail on different frames, the pool is larger than either
                measures alone, and the honest cross-session number is higher
                than anything reported so far.

This script decides it without training anything new. The CNN already stored a
per-frame posterior over classes for its exact test rows (`cm_*.npz`), and the
metric features are keyed on manifest row, so the two can be joined directly.

WHAT IT REPORTS, and why in this order:

  cnn / metric  -- each alone, on the SHARED rows (frames lacking metric features
                   are dropped from both, so every number in a row is computed on
                   one identical frame set; the CNN figure will therefore differ
                   slightly from its headline).
  fused         -- equal-weight average of the two posteriors, and the geometric
                   (log-space) mean. EQUAL WEIGHT IS FIXED, NOT TUNED. The CNN
                   saved posteriors for test rows only, so there is no held-out
                   split on which a mixing weight could honestly be chosen; the
                   sweep printed underneath is a diagnostic, and picking its
                   argmax would be fitting the test set.
  oracle        -- the frame is counted correct if EITHER model is correct. This
                   is the ceiling any score-level fusion could reach, and it is
                   the number that actually answers the question: an oracle close
                   to max(cnn, metric) means redundant, full stop.
  rescued       -- P(metric correct | CNN wrong). Complementarity, stated as the
                   quantity a reader will want.
"""
import os
import json
import glob
import argparse
import numpy as np

from hiride_data import load_manifest, make_split, eligible_mask
from hiride_keys import cond_key, arch_key
from hiride_stats import cluster_boot, boot_rng


def cnn_cells(runs, policy, modality, arch, condition):
    """Every run belonging to ONE cell, as (meta, cm_path).

    Keyed through hiride_keys, not through a hand-listed set of fields. The
    first version of this compared `m["condition"]` directly, which is the RAW
    condition string -- so scale_removed@1b, scale_removed/f10 and
    scale_removed/dsil all matched "scale_removed" and were averaged together
    as if they were seeds of one cell. That produced 15 "seeds" for a 5-seed
    condition, a row with 2,933 test frames next to rows with 5,642, and CNN
    accuracies from 3.7 % to 23.6 % inside a single mean.

    hiride_keys exists precisely to stop this; its docstring records the same
    bug shipping four separate times. Use it rather than re-deriving identity.
    """
    out, seen = [], {}
    for f in sorted(glob.glob(os.path.join(runs, "results_*.json"))):
        try:
            m = json.load(open(f))
        except (ValueError, OSError):
            continue
        if (m.get("policy") != policy or m.get("modality") != modality
                or m.get("permuted") or arch_key(m) != arch
                or cond_key(m) != condition):
            continue
        cm = os.path.join(runs, "cm_" + os.path.basename(f)[len("results_"):-len(".json")] + ".npz")
        if not os.path.exists(cm):
            continue
        # two runs of one cell sharing a seed means the key is still too loose
        seed = int(m["seed"])
        if seed in seen:
            raise SystemExit(
                f"error: {condition} seed {seed} matched twice "
                f"({os.path.basename(seen[seed])} and {os.path.basename(f)}). "
                f"The cell key is not distinguishing them -- add the differing "
                f"axis to hiride_keys.AXES.")
        seen[seed] = f
        out.append((m, cm))
    return sorted(out, key=lambda t: int(t[0]["seed"]))


def load_metric(prep, man):
    """Metric features aligned to manifest rows, plus the eligibility mask.

    Returns (full, have, cols, keep): `full` is (n_manifest_rows, n_features)
    with NaN where a frame has none, `have` marks the rows that do, `cols`
    selects the reported `metric` feature set -- skeleton and nuisance columns
    dropped, exactly as hiride_metric_floor.py does, so this is the same model
    that scores 19.04 % at R4 -- and `keep` is `have` intersected with cue
    eligibility.
    """
    z = np.load(os.path.join(prep, "cues.npz"), allow_pickle=False)
    keep_cue = eligible_mask(z["cues"], [str(f) for f in z["feats"]])
    mf = np.load(os.path.join(prep, "metric_features.npz"), allow_pickle=False)
    F, names, rows = mf["feats"], [str(n) for n in mf["names"]], mf["manifest_row"]
    full = np.full((len(man["frame"]), F.shape[1]), np.nan, dtype=np.float32)
    full[rows] = F
    have = ~np.isnan(full[:, 0])
    nuis = {"stand_dist_mm", "ground", "n_points", "valid_frac", "top_clip", "bot_clip"}
    cols = [i for i, n in enumerate(names) if n not in nuis and not n.startswith("bone_")]
    print(f"[load] {int(have.sum())} frames with metric features, "
          f"{int((keep_cue & have).sum())} also cue-eligible; "
          f"using {len(cols)} metric columns")
    return full, have, cols, keep_cue & have


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--policy", default="R4_cross_session")
    ap.add_argument("--modality", default="depth")
    ap.add_argument("--arch", default="alexnet")
    ap.add_argument("--condition", action="append", default=None,
                    help="repeatable; defaults to the three best R4 depth conditions")
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    conditions = args.condition or ["sil_scaled", "scale_removed", "person_centred"]

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    full, have, cols, keep = load_metric(args.prep, man)

    from sklearn.ensemble import RandomForestClassifier
    report = {}
    hdr = (f"{'condition':<16s}{'seed':>4s}{'rows':>7s}{'cnn':>8s}{'metric':>8s}"
           f"{'fused':>8s}{'geo':>8s}{'oracle':>8s}{'rescued':>9s}")
    for cond in conditions:
        cells = cnn_cells(args.runs, args.policy, args.modality, args.arch, cond)
        if not cells:
            print(f"\n{cond}: no CNN runs with per-frame posteriors -- skipped")
            continue
        print(f"\n{hdr}"); print("-" * len(hdr))
        per_seed, paired = [], []
        for meta, cm_path in cells:
            seed = int(meta["seed"])
            d = np.load(cm_path, allow_pickle=False)
            te_rows, classes = d["test_rows"], [str(c) for c in d["classes"]]
            prob = d["prob"].astype(np.float64)
            truth = d["truth"].astype(int)

            tr, va, _ = make_split(man, args.policy, seed=seed, keep=keep)
            cmap = {c: i for i, c in enumerate(classes)}
            tr = tr[np.array([str(s) in cmap for s in man["subject"][tr]], bool)]
            if len(tr) < 50:
                print(f"{cond:<16s}{seed:>4d}   too few metric-feature training rows")
                continue
            ytr = np.array([cmap[str(s)] for s in man["subject"][tr]])

            # the shared frame set: CNN test rows that also have metric features
            sel = have[te_rows]
            if sel.sum() < 50:
                print(f"{cond:<16s}{seed:>4d}   too few shared test rows")
                continue
            rows_s, p_cnn, y = te_rows[sel], prob[sel], truth[sel]
            p_cnn = p_cnn / np.clip(p_cnn.sum(1, keepdims=True), 1e-12, None)

            rf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
            rf.fit(full[tr][:, cols], ytr)
            p_rf = np.zeros_like(p_cnn)
            p_rf[:, rf.classes_.astype(int)] = rf.predict_proba(full[rows_s][:, cols])

            c_ok = (p_cnn.argmax(1) == y)
            m_ok = (p_rf.argmax(1) == y)
            f_ok = ((0.5 * p_cnn + 0.5 * p_rf).argmax(1) == y)
            g_ok = ((np.log(p_cnn + 1e-12) + np.log(p_rf + 1e-12)).argmax(1) == y)
            resc = float(m_ok[~c_ok].mean()) if (~c_ok).any() else float("nan")
            print(f"{cond:<16s}{seed:>4d}{len(y):>7d}{100*c_ok.mean():>7.2f}%"
                  f"{100*m_ok.mean():>7.2f}%{100*f_ok.mean():>7.2f}%{100*g_ok.mean():>7.2f}%"
                  f"{100*(c_ok | m_ok).mean():>7.2f}%{100*resc:>8.2f}%")
            subj = np.asarray(man["subject"][rows_s], dtype=str)
            per_seed.append(dict(seed=seed, n=int(len(y)), cnn=float(c_ok.mean()),
                                 metric=float(m_ok.mean()), fused=float(f_ok.mean()),
                                 geo=float(g_ok.mean()),
                                 oracle=float((c_ok | m_ok).mean()), rescued=resc))
            paired.append(dict(seed=seed, subj=subj, cnn=c_ok, metric=m_ok,
                               fused=f_ok, geo=g_ok, oracle=(c_ok | m_ok)))

        if not per_seed:
            continue
        mean = {k: float(np.mean([r[k] for r in per_seed]))
                for k in ("cnn", "metric", "fused", "geo", "oracle", "rescued")}
        print(f"{'mean':<16s}{len(per_seed):>4d}{'':>7s}"
              + "".join(f"{100*mean[k]:>7.2f}%" for k in ("cnn", "metric", "fused",
                                                          "geo", "oracle"))
              + f"{100*mean['rescued']:>8.2f}%")
        # Paired contrasts, every one on identical frames. The comparison that
        # decides whether fusion is worth reporting is against `metric` -- the
        # best SINGLE representation -- not against the CNN, which both fusions
        # beat trivially because metric alone already does.
        cis = {}
        for a, b in (("fused", "cnn"), ("fused", "metric"),
                     ("geo", "cnn"), ("geo", "metric"), ("oracle", "metric")):
            per = [cluster_boot(P[a].astype(float) - P[b].astype(float), P["subj"],
                                boot_rng(args.seed, ("fuse", cond, a, b, P["seed"])),
                                args.boot) for P in paired]
            cis[f"{a}-{b}"] = [float(np.mean([c[0] for c in per])),
                               float(np.mean([c[1] for c in per]))]
        print(f"  paired contrasts (subject-cluster CI, mean over {len(paired)} seeds):")
        for k, (lo, hi) in cis.items():
            a, b = k.split("-")
            d = 100 * (mean[a] - mean[b])
            flag = "" if lo * hi > 0 else "   (interval straddles zero)"
            print(f"    {k:<16s}{d:>+7.2f} pp   [{100*lo:+.2f}, {100*hi:+.2f}]{flag}")
        print(f"  headroom: oracle - max(cnn, metric) = "
              f"{100*(mean['oracle'] - max(mean['cnn'], mean['metric'])):+.2f} pp")
        report[cond] = dict(seeds=per_seed, mean=mean, ci=cis)

    if args.out:
        path = os.path.join(args.out, "fusion.json")
        with open(path, "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\n[written] {path}")
    print("\nREAD: `oracle` is the ceiling of any score-level fusion. If it sits near "
          "max(cnn, metric),\nthe two representations are reading the same information "
          "and ~19 % is the frame's content.\nThe equal-weight `fused` column is fixed, "
          "not tuned -- there is no held-out split to tune it on.")


if __name__ == "__main__":
    main()
