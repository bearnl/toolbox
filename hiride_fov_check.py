"""Adjudicate the FOV-clipping contradiction between paper 2 and paper 3.

    python hiride_fov_check.py --prep $SCRATCH/hiride2/prep

The two handoffs disagree about the SAME files:

  paper 2 (HIRIDE_HANDOFF section 5.3, from the shipped userMap, Training,
      eligible frames): top edge 3.6 %, bottom 13.1 %, both 2.2 %, FULL-BODY 85.4 %.
  paper 3 (PAPER_HANDOFF section 4.6, height audit, 28 subjects):
      "100 % of Training frames clip the body at top AND bottom edge; 0 of 13,426
      are full-body", height std across 28 people 16 mm ("FOV-saturated").

A reviewer who reads both papers damages both, so one of them has to be wrong.
The two use DIFFERENT foreground definitions, which is the obvious suspect:

  paper 2: the dataset's shipped `_userMap.pgm` (OpenNI user index > 0).
  paper 3: a DEPTH SLAB -- anchor = 1st percentile of valid depth (the nearest
      body surface), foreground = valid & depth in [anchor, anchor + CLIP_RANGE],
      CLIP_RANGE 300 or 600 mm -- followed by a square crop around the
      foreground centroid.

Hypothesis under test: the slab anchors on whatever is nearest, and if that is
the floor, a near wall, or sensor noise rather than the body, the "foreground"
becomes a depth band through the room that reaches the frame edges in every
frame. The crop would do it too: a box sized from the foreground's own spread
makes the foreground touch its edges by construction.

This script measures, on the same frames, the edge-touch rate of BOTH
definitions plus how much of the slab is actually the person, so the
disagreement is settled with numbers rather than argument.

Reports:
  * userMap edge-touch at FULL resolution, straight from cues.npz -- the
    authoritative restatement of paper 2's number;
  * userMap edge-touch recomputed from the 256^2 shards -- validates that the
    shard-based slab numbers below are comparable;
  * slab edge-touch at CLIP_RANGE 300 and 600, and whether the anchor pixel
    lands on the person at all;
  * slab-vs-userMap IoU / precision / recall.

CPU only, a few minutes:
    sbatch --account=def-czarnuch_cpu --time=0:30:00 --mem=24000M --cpus-per-task=2 \
      -J hiride-fov -o logs/hiride-fov_%j.out --wrap 'cd ~/toolbox && \
      source ~/venvs/venv311/bin/activate && python hiride_fov_check.py --prep $SCRATCH/hiride2/prep'
"""
import os
import json
import argparse

import numpy as np

from hiride_data import load_manifest, eligible_mask, build_manifest
from hiride_pgm import read_pgm

CLIP_RANGES = (300.0, 600.0)
SEQ_TAG = {"Training": "training", "Testing/Still": "testing_still",
           "Testing/Walking": "testing_walking"}


def pct(x):
    return 100.0 * float(np.mean(x)) if len(x) else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--seq", default="Training")
    ap.add_argument("--sample", type=int, default=4000,
                    help="frames sampled for the slab computation (0 = all)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--root", default=None,
                    help="staged BIWI tree ($SLURM_TMPDIR/biwi). Use this for the "
                         "AUTHORITATIVE answer: the 256^2 shards are AREA-resized, and "
                         "that box average mixes invalid (0) pixels into valid depth, "
                         "manufacturing spuriously NEAR values -- exactly what a "
                         "1st-percentile anchor latches onto. Raw PGM has no such artefact.")
    ap.add_argument("--paper3-exact", action="store_true",
                    help="replicate anthro_probe.py's _foreground/_audit_frame EXACTLY: "
                         "slab at FG_CLIP_MM=1500 (not 300/600), FG_PERCENTILE=1, then "
                         "LARGEST CONNECTED COMPONENT (which is meant to drop detached "
                         "background blobs like walls), MIN_FG_PIXELS=200, and the audit's "
                         "own edge rule (row<=1 / row>=H-2). This is the fair test: the "
                         "prose description omits the LCC step, and LCC could recover the "
                         "person from a slab that also caught a wall.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    z = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    cues, feats = z["cues"], [str(f) for f in z["feats"]]
    keep = eligible_mask(cues, feats)
    seq_m = man["seq"] == args.seq

    # ---- 1. paper 2's number, restated from the full-resolution cues ---------
    ti, bi = feats.index("top_touch"), feats.index("bot_touch")
    for label, sel in (("ALL frames", seq_m), ("ELIGIBLE frames", seq_m & keep)):
        t, b = cues[sel, ti] > 0, cues[sel, bi] > 0
        print(f"[userMap @ full res, {args.seq}, {label}] n={int(sel.sum())}  "
              f"top {pct(t):5.2f} %  bottom {pct(b):5.2f} %  both {pct(t & b):5.2f} %  "
              f"full-body {pct(~t & ~b):5.2f} %")

    # ---- 2. the slab, on the same frames ------------------------------------
    if args.root:
        raw = build_manifest(args.root, verbose=False)
        rkeep = eligible_mask(cues, feats)
        # map raw manifest rows by (seq, subject, frame) so the SAME eligibility
        # filter applies to the raw tree as to the prep manifest
        key_of = lambda m, i: (str(m["seq"][i]), str(m["subject"][i]), int(m["frame"][i]))
        elig_keys = {key_of(man, i) for i in np.where(seq_m & rkeep)[0]}
        cand = [i for i in range(len(raw["frame"]))
                if str(raw["seq"][i]) == args.seq and key_of(raw, i) in elig_keys]
        if args.sample and len(cand) > args.sample:
            cand = sorted(rng.choice(cand, size=args.sample, replace=False).tolist())
        print(f"\n[slab] {len(cand)} eligible {args.seq} frames, RAW 480x640 PGM "
              f"(no resize artefact)\n")

        def frames():
            for i in cand:
                d, _ = read_pgm(os.path.join(raw["root"], raw["depth"][i]))
                u, _ = read_pgm(os.path.join(raw["root"], raw["user"][i]))
                yield d.astype(np.float32), (u > 0)
        source, n_frames, side = frames(), len(cand), "480x640 raw"
    else:
        tag = SEQ_TAG[args.seq]
        depth = np.load(os.path.join(args.prep, f"{tag}_depth.npy"), mmap_mode="r")
        mask = np.load(os.path.join(args.prep, f"{tag}_mask.npy"), mmap_mode="r")
        rows = np.load(os.path.join(args.prep, f"{tag}_index.npz"))["manifest_row"]
        elig_pos = np.where(keep[rows])[0]
        if args.sample and len(elig_pos) > args.sample:
            elig_pos = np.sort(rng.choice(elig_pos, size=args.sample, replace=False))
        print(f"\n[slab] {len(elig_pos)} eligible {args.seq} frames "
              f"(shards are {depth.shape[1]}^2, AREA-resized from 480x640 -- see --root)\n")

        def frames():
            for pos in elig_pos:
                yield np.asarray(depth[pos]).astype(np.float32), np.asarray(mask[pos]) > 0
        source, n_frames, side = frames(), len(elig_pos), f"{depth.shape[1]}^2 shard"

    summary = {}
    res = {c: dict(top=[], bot=[], iou=[], prec=[], rec=[], anchor_on_person=[],
                   frac_person=[]) for c in CLIP_RANGES}
    um_top, um_bot = [], []
    anchor_mm, slab_area, person_p1 = [], [], []
    for d, person in source:
        if not person.any():
            continue
        um_top.append(person[0].any()); um_bot.append(person[-1].any())
        valid = d > 0
        if not valid.any():
            continue
        anchor = np.percentile(d[valid], 1)
        anchor_mm.append(float(anchor))
        person_p1.append(float(np.percentile(d[person & valid], 1)) if (person & valid).any() else np.nan)
        # the pixel that defines the anchor: nearest valid depth
        ay, ax = np.unravel_index(np.argmin(np.where(valid, d, np.inf)), d.shape)
        for c in CLIP_RANGES:
            fg = valid & (d >= anchor) & (d <= anchor + c)
            r = res[c]
            r["top"].append(bool(fg[0].any())); r["bot"].append(bool(fg[-1].any()))
            inter = float((fg & person).sum())
            r["iou"].append(inter / max(1.0, float((fg | person).sum())))
            r["prec"].append(inter / max(1.0, float(fg.sum())))
            r["rec"].append(inter / max(1.0, float(person.sum())))
            r["frac_person"].append(float(person.sum()) / max(1.0, float(fg.sum())))
            if c == CLIP_RANGES[-1]:
                slab_area.append(float(fg.mean()))
        res[CLIP_RANGES[0]]["anchor_on_person"].append(bool(person[ay, ax]))

    print(f"[userMap @ {side}, same frames] "
          f"top {pct(um_top):5.2f} %  bottom {pct(um_bot):5.2f} %  "
          f"both {pct(np.array(um_top) & np.array(um_bot)):5.2f} %  "
          f"full-body {pct(~np.array(um_top) & ~np.array(um_bot)):5.2f} %")
    print("  (if this differs much from the full-res line above, the 256^2 resize "
          "is smearing the mask and the slab numbers need the raw frames instead)\n")

    if args.paper3_exact:
        from scipy import ndimage
        FG_CLIP_MM, FG_PCT, MIN_FG = 1500.0, 1.0, 200
        p3 = dict(top=[], bot=[], iou=[], prec=[], rec=[], empty=0, area=[])
        um_t2, um_b2 = [], []
        for d, person in frames():
            H = d.shape[0]
            valid = d > 0
            if int(valid.sum()) < MIN_FG:
                p3["empty"] += 1
                continue
            anchor = np.percentile(d[valid], FG_PCT)
            fg = valid & (d >= anchor) & (d <= anchor + FG_CLIP_MM)
            if int(fg.sum()) < MIN_FG:
                p3["empty"] += 1
                continue
            lb, n = ndimage.label(fg)
            if n > 1:
                sizes = ndimage.sum(fg, lb, range(1, n + 1))
                fg = lb == (int(np.argmax(sizes)) + 1)
            if int(fg.sum()) < MIN_FG:
                p3["empty"] += 1
                continue
            rows = np.where(fg.any(axis=1))[0]
            p3["top"].append(bool(rows.min() <= 1))
            p3["bot"].append(bool(rows.max() >= H - 2))
            inter = float((fg & person).sum())
            p3["iou"].append(inter / max(1.0, float((fg | person).sum())))
            p3["prec"].append(inter / max(1.0, float(fg.sum())))
            p3["rec"].append(inter / max(1.0, float(person.sum())))
            p3["area"].append(float(fg.mean()))
            prow = np.where(person.any(axis=1))[0]
            um_t2.append(bool(prow.min() <= 1)); um_b2.append(bool(prow.max() >= H - 2))
        t, b = np.array(p3["top"]), np.array(p3["bot"])
        print("\n=== EXACT replication of anthro_probe.py (_foreground + _audit_frame) ===")
        print(f"  slab 1500 mm + largest connected component, edge margin 2 rows, "
              f"{len(t)} frames scored ({p3['empty']} rejected as too small)")
        print(f"  paper-3 foreground : top {pct(t):6.2f} %  bottom {pct(b):6.2f} %  "
              f"both {pct(t & b):6.2f} %  full-body {pct(~t & ~b):6.2f} %")
        print(f"  shipped userMap    : top {pct(um_t2):6.2f} %  bottom {pct(um_b2):6.2f} %  "
              f"both {pct(np.array(um_t2) & np.array(um_b2)):6.2f} %  "
              f"full-body {pct(~np.array(um_t2) & ~np.array(um_b2)):6.2f} %   (same rule, same frames)")
        print(f"  overlap with the person: IoU {np.median(p3['iou']):.3f}  "
              f"precision {np.median(p3['prec']):.3f}  recall {np.median(p3['rec']):.3f}  "
              f"| foreground covers {100 * np.median(p3['area']):.1f} % of the frame")
        summary["paper3_exact"] = dict(
            n=int(len(t)), rejected=int(p3["empty"]), top=pct(t), bottom=pct(b),
            both=pct(t & b), full_body=pct(~t & ~b),
            iou_median=float(np.median(p3["iou"])), precision_median=float(np.median(p3["prec"])),
            recall_median=float(np.median(p3["rec"])),
            usermap_full_body_same_rule=pct(~np.array(um_t2) & ~np.array(um_b2)))

    for c in CLIP_RANGES:
        r = res[c]
        t, b = np.array(r["top"]), np.array(r["bot"])
        summary[f"slab_{int(c)}"] = dict(
            top=pct(t), bottom=pct(b), both=pct(t & b), full_body=pct(~t & ~b),
            iou_median=float(np.median(r["iou"])), precision_median=float(np.median(r["prec"])),
            recall_median=float(np.median(r["rec"])))
        print(f"[slab CLIP={int(c)} mm] top {pct(t):6.2f} %  bottom {pct(b):6.2f} %  "
              f"both {pct(t & b):6.2f} %  full-body {pct(~t & ~b):6.2f} %")
        print(f"                 vs userMap: IoU {np.median(r['iou']):.3f}  "
              f"precision {np.median(r['prec']):.3f} (how much of the slab IS the person)  "
              f"recall {np.median(r['rec']):.3f}")
    am, pp = np.array(anchor_mm), np.array(person_p1)
    print(f"\n[anchor depth] slab anchor (1st pct of valid) median {np.median(am):7.0f} mm  "
          f"p10 {np.percentile(am, 10):6.0f}  p90 {np.percentile(am, 90):6.0f}")
    print(f"[person depth] nearest person surface (1st pct) median {np.nanmedian(pp):7.0f} mm  "
          f"-> the anchor sits {np.nanmedian(pp) - np.median(am):5.0f} mm IN FRONT of the person")
    print(f"[slab area]    CLIP={int(CLIP_RANGES[-1])} covers {100 * np.median(slab_area):5.1f} % "
          f"of the frame (the person is ~8 %)")
    aop = res[CLIP_RANGES[0]]["anchor_on_person"]
    print(f"\n[anchor] the nearest valid pixel lands ON the person in {pct(aop):5.2f} % of frames")
    print("  (a low number means the slab is anchored on something else -- floor, near wall,")
    print("   or sensor noise -- and the 'foreground' is a depth band through the room)")

    summary["usermap_full_res"] = dict(
        top=pct(cues[seq_m & keep, ti] > 0), bottom=pct(cues[seq_m & keep, bi] > 0))
    summary["anchor_on_person_pct"] = pct(aop)
    summary["anchor_depth_median_mm"] = float(np.median(am))
    summary["person_p1_depth_median_mm"] = float(np.nanmedian(pp))
    summary["slab_area_median"] = float(np.median(slab_area))
    summary["source"] = side
    summary["n_frames"] = int(n_frames)
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "fov_check.json"), "w") as fh:
            json.dump(summary, fh, indent=1)
        print(f"\n[written] {os.path.join(args.out, 'fov_check.json')}")

    print("\nHOW TO READ THIS. If the slab touches both edges in ~100 % of frames while the")
    print("userMap touches them in a few per cent, the two handoffs are both 'right' about")
    print("their own foreground and paper 3's claim is a property of the slab, not of BIWI's")
    print("framing -- in which case paper 3's FOV-clipping DIAGNOSIS needs revisiting, since")
    print("it attributes a cross-session failure to the dataset. If instead both definitions")
    print("agree, paper 2's 85.4 % full-body figure is wrong and section 5.3 must be corrected.")


if __name__ == "__main__":
    main()
