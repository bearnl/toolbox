"""Measure the sensor floor: does depth quantisation explain the missing interior?

    python hiride_noise.py --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/results

The claim under test, arithmetic first. Kinect v1 derives depth from disparity
quantised to 1/8 px with baseline b ~ 75 mm and f ~ 575.8 px, so the depth step
grows as z^2/(f*b*8): ~25 mm at the person's 2955 mm median. Identity-bearing
body curvature differs between adults by ~20-80 mm. If that arithmetic is right
the in-body signal sits AT the sensor floor at 3 m, which would explain -- with
no appeal to our pipeline -- why every interior condition matches the silhouette
and why 15x more input contrast bought nothing.

Two things are measured directly rather than assumed:

  1. QUANTISATION STEP vs range. Raw Kinect depth takes discrete values, so the
     spacing between neighbouring distinct values inside a range bin IS the step.
     Measured from the shipped 16-bit PGM values, per 250 mm bin.
  2. TEMPORAL NOISE vs range. For consecutive frames of one recording, pixels
     that are BACKGROUND in both (per the shipped userMap) view a static scene,
     so their frame-to-frame difference is pure sensor noise. sigma is estimated
     robustly as 1.4826 * MAD(delta) / sqrt(2).

Read against the reference the paper already has: person depth extent (p99-p1)
median 419 mm, and between-person curvature differences of tens of mm. If sigma
at 3 m is comparable to those differences, the interior is unrecoverable per
frame at this range -- which makes multi-frame fusion (sigma/sqrt(N)) and closer
standoff the only levers, and predicts the interior becomes usable at shorter
range. That prediction is testable within BIWI, whose person depth spans
1240-3885 mm.

CPU, minutes. Needs the staged raw tree for the quantisation part (--root),
which only exists inside a job; the noise part works from the shards alone.
"""
import os
import json
import argparse
from collections import defaultdict

import numpy as np

from hiride_data import load_manifest, eligible_mask, build_manifest
from hiride_pgm import read_pgm

FX = 575.816
BASELINE_MM = 75.0
SUBPIX = 8.0
SEQ_TAG = {"Training": "training", "Testing/Still": "testing_still",
           "Testing/Walking": "testing_walking"}


def predicted_step(z):
    return z * z / (FX * BASELINE_MM * SUBPIX)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--root", default=None, help="staged BIWI tree, for quantisation")
    ap.add_argument("--pairs", type=int, default=400, help="consecutive frame pairs")
    ap.add_argument("--quant-frames", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    z = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    cues, feats = z["cues"], [str(f) for f in z["feats"]]
    keep = eligible_mask(cues, feats)
    report = {"predicted_step_mm": {}, "quantisation_mm": {}, "noise_mm": {}}

    print("Predicted depth step from the disparity model, z^2/(f*b*8):")
    for zz in (1250, 1750, 2250, 2750, 3250, 3750):
        report["predicted_step_mm"][str(zz)] = round(float(predicted_step(zz)), 2)
        print(f"   {zz:5d} mm -> {predicted_step(zz):6.1f} mm")

    # ---- 1. quantisation, from RAW pgm values ------------------------------
    if args.root:
        raw = build_manifest(args.root, verbose=False)
        n = len(raw["frame"])
        pick = rng.choice(n, size=min(args.quant_frames, n), replace=False)
        by_bin = defaultdict(list)
        for i in pick:
            d, _ = read_pgm(os.path.join(raw["root"], raw["depth"][int(i)]))
            v = d[d > 0].astype(np.int64)
            if v.size < 1000:
                continue
            for lo in range(1000, 4000, 250):
                sel = v[(v >= lo) & (v < lo + 250)]
                if sel.size > 200:
                    u = np.unique(sel)
                    if u.size > 3:
                        by_bin[lo].append(float(np.median(np.diff(u))))
        print("\nMEASURED quantisation step (median spacing of distinct raw values):")
        print(f"{'range bin':>14s}{'measured':>11s}{'predicted':>11s}{'n frames':>10s}")
        for lo in sorted(by_bin):
            m = float(np.median(by_bin[lo]))
            report["quantisation_mm"][str(lo)] = m
            print(f"{lo:6d}-{lo+250:4d} mm{m:11.1f}{predicted_step(lo + 125):11.1f}"
                  f"{len(by_bin[lo]):10d}")
    else:
        print("\n[quantisation] skipped -- pass --root <staged tree> to measure it")

    # ---- 2. temporal noise on static background, from the shards ----------
    print("\nMEASURED temporal noise on STATIC BACKGROUND (sigma from MAD):")
    print(f"{'range bin':>14s}{'sigma':>9s}{'predicted step':>16s}{'n pairs':>9s}")
    for seq, tag in SEQ_TAG.items():
        ip = os.path.join(args.prep, f"{tag}_index.npz")
        if not os.path.exists(ip):
            continue
        rows = np.load(ip)["manifest_row"]
        dep = np.load(os.path.join(args.prep, f"{tag}_depth.npy"), mmap_mode="r")
        msk = np.load(os.path.join(args.prep, f"{tag}_mask.npy"), mmap_mode="r")
        pos_of = {int(r): p for p, r in enumerate(rows)}
        got = defaultdict(list)
        groups = np.unique(man["group"][rows])
        for _ in range(args.pairs):
            g = rng.choice(groups)
            gi = rows[man["group"][rows] == g]
            gi = gi[np.argsort(man["frame"][gi])]
            if len(gi) < 3:
                continue
            k = int(rng.integers(0, len(gi) - 1))
            if man["frame"][gi[k + 1]] - man["frame"][gi[k]] != 1:
                continue
            a = np.asarray(dep[pos_of[int(gi[k])]]).astype(np.float32)
            b = np.asarray(dep[pos_of[int(gi[k + 1])]]).astype(np.float32)
            ma = np.asarray(msk[pos_of[int(gi[k])]]) > 0
            mb = np.asarray(msk[pos_of[int(gi[k + 1])]]) > 0
            bg = (~ma) & (~mb) & (a > 0) & (b > 0)      # static in both frames
            if bg.sum() < 500:
                continue
            zz, dd = a[bg], (b - a)[bg]
            for lo in range(1000, 4000, 250):
                sel = (zz >= lo) & (zz < lo + 250)
                if sel.sum() > 200:
                    # sigma of a single frame from the difference of two
                    got[lo].append(1.4826 * np.median(np.abs(dd[sel])) / np.sqrt(2.0))
        for lo in sorted(got):
            sig = float(np.median(got[lo]))
            report["noise_mm"][f"{seq}|{lo}"] = sig
            print(f"{seq[:8]:>8s}{lo:5d}+{sig:9.1f}{predicted_step(lo + 125):16.1f}"
                  f"{len(got[lo]):9d}")

    print("\nHOW TO READ THIS")
    print("  If sigma at ~3000 mm is comparable to between-person curvature differences")
    print("  (tens of mm), the in-body signal is at the sensor floor PER FRAME, and the")
    print("  levers are multi-frame fusion (sigma/sqrt(N)) and shorter standoff -- not")
    print("  contrast, not architecture. Both are then testable: --frames N, and the")
    print("  range-binned probe in hiride_signal.py.")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "noise_floor.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\n[written] {os.path.join(args.out, 'noise_floor.json')}")


if __name__ == "__main__":
    main()
