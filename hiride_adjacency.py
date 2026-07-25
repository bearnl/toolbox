"""Pixel-level near-duplicate measurement for the guard sweep.

    python hiride_adjacency.py --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/results

The guard sweep shows that temporal proximity between a test frame and its
nearest training frame is worth 14 points (31.8 -> 17.7 %, guard 0 -> 150) at
matched N_train on byte-identical test frames. That MEASURES the leak; this
script EXPLAINS it: how different, in pixels, are two frames of the same
recording k frames apart, against the distances that matter for the ladder?

For lag k in LAGS it samples pairs (f, f+k) inside one recording (both frames
eligible under eligible_mask, so the pose regime is the same one every other
condition sees) and reports, per pair,
  * mean |delta depth| in millimetres over pixels valid in BOTH frames
    (depth clipped at DEPTH_CLIP_MM like the trainer);
  * silhouette IoU from the shipped userMap;
  * mean |delta RGB| in 8-bit units (RGB shard is mmapped; fewer pairs).
and the same three numbers for the reference distances:
  * between-subject, same sequence          -- what "different person" looks like
  * same subject, Testing/Still  vs Walking  -- R3's train->test distance
  * same subject, Training       vs Walking  -- R4's train->test distance
Everything is reported as median and IQR over pairs; means are dominated by
walk-away frames.

CPU only, reads the prep shards, a few minutes:
    sbatch --account=def-czarnuch_cpu --time=0:30:00 --mem=24000M --cpus-per-task=2 \
        --wrap 'cd ~/toolbox && source ~/venvs/venv311/bin/activate && \
                python hiride_adjacency.py --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/results'
"""
import os
import json
import argparse

import numpy as np

from hiride_data import load_manifest, eligible_mask

DEPTH_CLIP_MM = 6000.0
LAGS = (1, 2, 5, 10, 25, 50, 150, 500)
SEQ_TAG = {"Training": "training", "Testing/Still": "testing_still",
           "Testing/Walking": "testing_walking"}


def _load(prep, tag, kind, mmap):
    p = os.path.join(prep, f"{tag}_{kind}.npy")
    if not os.path.exists(p):
        return None
    return np.load(p, mmap_mode="r" if mmap else None)


def pair_metrics(dA, dB, mA, mB, cA=None, cB=None):
    a = np.minimum(dA.astype(np.float32), DEPTH_CLIP_MM)
    b = np.minimum(dB.astype(np.float32), DEPTH_CLIP_MM)
    valid = (dA > 0) & (dB > 0)
    dd = float(np.abs(a[valid] - b[valid]).mean()) if valid.any() else float("nan")
    ma, mb = mA > 0, mB > 0
    union = (ma | mb).sum()
    iou = float((ma & mb).sum() / union) if union else float("nan")
    out = {"d_depth_mm": dd, "sil_iou": iou}
    if cA is not None:
        out["d_rgb"] = float(np.abs(cA.astype(np.int16) - cB.astype(np.int16)).mean())
    return out


def summarise(rows):
    keys = sorted({k for r in rows for k in r})
    out = {"n_pairs": len(rows)}
    for k in keys:
        v = np.array([r[k] for r in rows if k in r and np.isfinite(r[k])])
        if v.size:
            out[k] = {"median": float(np.median(v)), "q1": float(np.percentile(v, 25)),
                      "q3": float(np.percentile(v, 75)), "mean": float(v.mean()), "n": int(v.size)}
    return out


def fmt(s):
    """One line per summary; tolerant of empty lag buckets (short recordings)."""
    if not s.get("n_pairs") or "d_depth_mm" not in s:
        return f"pairs={s.get('n_pairs', 0):5d}  (no eligible pairs)"
    line = (f"pairs={s['n_pairs']:5d}  |dDepth| med={s['d_depth_mm']['median']:7.1f} mm "
            f"IQR=[{s['d_depth_mm']['q1']:.1f},{s['d_depth_mm']['q3']:.1f}]  "
            f"silIoU med={s['sil_iou']['median']:.3f}")
    if "d_rgb" in s:
        line += f"  |dRGB| med={s['d_rgb']['median']:.1f}"
    return line


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--pairs-per-group", type=int, default=40,
                    help="pairs sampled per recording per lag (depth+mask)")
    ap.add_argument("--rgb-pairs-per-group", type=int, default=8,
                    help="of those, how many also read the (mmapped) RGB shard")
    ap.add_argument("--baseline-pairs", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    cues = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    keep = eligible_mask(cues["cues"], [str(f) for f in cues["feats"]])
    n = len(man["frame"])

    # row -> (tag, pos); depth+mask fully in RAM (~8 GB total), RGB mmapped.
    depth, mask, rgb, where = {}, {}, {}, {}
    for seq, tag in SEQ_TAG.items():
        ipath = os.path.join(args.prep, f"{tag}_index.npz")
        if not os.path.exists(ipath):
            continue
        rows = np.load(ipath)["manifest_row"]
        depth[tag] = _load(args.prep, tag, "depth", mmap=False)
        mask[tag] = _load(args.prep, tag, "mask", mmap=False)
        rgb[tag] = _load(args.prep, tag, "rgb", mmap=True)
        for pos, row in enumerate(rows):
            where[int(row)] = (tag, pos)
        print(f"[load] {tag}: {len(rows)} frames")

    def get(row, want_rgb):
        tag, pos = where[int(row)]
        return (depth[tag][pos], mask[tag][pos],
                rgb[tag][pos] if (want_rgb and rgb[tag] is not None) else None)

    def metrics(ra, rb, want_rgb):
        dA, mA, cA = get(ra, want_rgb)
        dB, mB, cB = get(rb, want_rgb)
        return pair_metrics(dA, dB, mA, mB, cA, cB)

    results = {"lags": {}, "baselines": {}, "config": vars(args)}

    # ---- within-recording lags (Training only: that is where R0/R1 live) ----
    tr_rows = np.where((man["seq"] == "Training") & keep)[0]
    groups = sorted(set(man["group"][tr_rows].tolist()))
    for lag in LAGS:
        rows_out = []
        for g in groups:
            gi = tr_rows[man["group"][tr_rows] == g]
            fr = man["frame"][gi]
            lookup = {int(f): int(r) for f, r in zip(fr, gi)}
            cands = [(r, lookup[int(f) + lag]) for f, r in zip(fr, gi) if int(f) + lag in lookup]
            if not cands:
                continue
            pick = rng.choice(len(cands), size=min(args.pairs_per_group, len(cands)), replace=False)
            for j, c in enumerate(pick):
                ra, rb = cands[c]
                rows_out.append(metrics(ra, rb, want_rgb=j < args.rgb_pairs_per_group))
        results["lags"][str(lag)] = summarise(rows_out)
        print(f"[lag {lag:>3d}] " + fmt(results['lags'][str(lag)]))

    # ---- baselines ----
    def sample_pairs(rows_a, rows_b, same_subject, k):
        out = []
        subj = man["subject"]
        if same_subject:
            by_b = {}
            for r in rows_b:
                by_b.setdefault(subj[r], []).append(int(r))
            cand_a = [int(r) for r in rows_a if subj[r] in by_b]
            for _ in range(k):
                ra = int(rng.choice(cand_a))
                rb = int(rng.choice(by_b[subj[ra]]))
                out.append((ra, rb))
        else:
            while len(out) < k:
                ra, rb = int(rng.choice(rows_a)), int(rng.choice(rows_b))
                if subj[ra] != subj[rb]:
                    out.append((ra, rb))
        return out

    walk = np.where((man["seq"] == "Testing/Walking") & keep)[0]
    still = np.where((man["seq"] == "Testing/Still") & keep)[0]
    specs = {
        "between_subject_Training": (tr_rows, tr_rows, False),
        "same_subject_Still_vs_Walking (R3)": (still, walk, True),
        "same_subject_Training_vs_Walking (R4)": (tr_rows, walk, True),
        "between_subject_Training_vs_Walking": (tr_rows, walk, False),
    }
    for name, (a, b, same) in specs.items():
        if len(a) == 0 or len(b) == 0:
            print(f"[baseline {name}] skipped (missing sequence)")
            continue
        pairs = sample_pairs(a, b, same, args.baseline_pairs)
        rows_out = [metrics(ra, rb, want_rgb=(j % 5 == 0)) for j, (ra, rb) in enumerate(pairs)]
        results["baselines"][name] = summarise(rows_out)
        print(f"[baseline {name}] " + fmt(results['baselines'][name]))

    with open(os.path.join(args.out, "adjacency_results.json"), "w") as fh:
        json.dump(results, fh, indent=1)
    print(f"[done] -> {os.path.join(args.out, 'adjacency_results.json')}")


if __name__ == "__main__":
    main()
