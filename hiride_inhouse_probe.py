"""Two follow-up questions the in-house feasibility check raised.

    python hiride_inhouse_probe.py --root /project/6005175/chenzz/datasets/inhouse

`hiride_inhouse_check.py` established that 26 source .mkv recordings were
collapsed into 10 unbroken index runs by the overwrite in kinect2png.py, and
that slab+LCC finds a person in 100 % of frames for three identities and ~0 %
for seven. Two things follow that decide what in-house can actually do, and
neither can be settled by reading code:

1. IS EACH LABEL TEMPORALLY COHERENT? `idx` restarts per .mkv, so for a given
   index the surviving frame comes from whichever file was written last. If the
   last-written file is the longest, one recording overwrote all the others and
   the run is a single coherent sequence -- which is all R1 (contiguous block +
   guard) needs. If it is shorter, the tail survives from an earlier recording
   and there is a SPLICE at that index. A splice is visible without any
   provenance metadata: inside a recording, adjacent frames are near-duplicates
   (measured on BIWI: median |delta depth| 80.6 mm at lag 1), whereas across
   recordings they are as far apart as two different people (1313 mm). So a scan
   of |delta depth| against index localises every splice, or shows there is none.
   **R1 on in-house is only meaningful if the run is coherent (or its splices are
   known and can be treated as recording boundaries).**

2. WHY DOES THE SLAB FAIL FOR SEVEN IDENTITIES? Reported without the
   plausibility filter, the largest-connected-component area and the anchor
   depth say whether the foreground is empty, the whole frame, or a slab through
   a wall -- i.e. whether a better foreground is worth writing (different anchor,
   depth histogram mode, RGB-driven) or whether the mechanism suite simply stays
   BIWI-only and in-house replicates the ladder alone.

CPU, minutes. Prints per label; writes inhouse_probe.json with --out.
"""
import os
import re
import json
import argparse
from collections import defaultdict

import numpy as np

NAME = re.compile(r"^(?P<label>.+)_(?P<idx>\d+)_depth\.png$")


def read_depth(path):
    import tensorflow as tf
    return np.asarray(tf.io.decode_png(tf.io.read_file(path), dtype=tf.uint16))[..., 0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--stride", type=int, default=5,
                    help="scan every Nth adjacent pair (1 = every pair)")
    ap.add_argument("--far-pairs", type=int, default=200,
                    help="random far-apart pairs per label, for the reference distance")
    ap.add_argument("--seg-sample", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from scipy import ndimage

    subdirs = sorted((d for d in os.scandir(args.root) if d.is_dir()), key=lambda e: e.name)
    report = {}
    for sd in subdirs:
        byl = defaultdict(list)
        for f in os.listdir(sd.path):
            m = NAME.match(f)
            if m:
                byl[m.group("label")].append(int(m.group("idx")))
        for lab, idxs in sorted(byl.items()):
            idxs = np.array(sorted(idxs))
            key = f"{sd.name}/{lab}"
            print(f"\n== {key}  ({len(idxs)} frames) ==")

            def load(i):
                return read_depth(os.path.join(sd.path, f"{lab}_{i}_depth.png")).astype(np.float32)

            # ---- 1. splice scan -------------------------------------------
            step = max(1, args.stride)
            probe = idxs[:-1:step]
            deltas, prev_i, prev = [], None, None
            for i in probe:
                a = load(i) if prev_i != i else prev
                b = load(i + 1)
                v = (a > 0) & (b > 0)
                deltas.append(float(np.abs(a[v] - b[v]).mean()) if v.any() else np.nan)
                prev_i, prev = i + 1, b
            deltas = np.array(deltas, dtype=float)
            far = []
            for _ in range(args.far_pairs):
                i, j = rng.choice(idxs, 2, replace=False)
                a, b = load(int(i)), load(int(j))
                v = (a > 0) & (b > 0)
                if v.any():
                    far.append(float(np.abs(a[v] - b[v]).mean()))
            far = np.array(far, dtype=float)
            adj_med = np.nanmedian(deltas)
            far_med = np.nanmedian(far) if far.size else np.nan
            # A splice looks like a between-recording distance between two
            # supposedly adjacent frames.
            thr = far_med if np.isfinite(far_med) else np.inf
            splices = probe[np.where(deltas > thr)[0]] if np.isfinite(thr) else np.array([])
            print(f"  adjacent |dDepth| median {adj_med:8.1f} mm   "
                  f"far-apart median {far_med:8.1f} mm   ratio {adj_med / far_med:.3f}"
                  if np.isfinite(far_med) else f"  adjacent |dDepth| median {adj_med:8.1f} mm")
            print(f"  splice candidates (adjacent pair >= far-apart median): {len(splices)}"
                  + (f"  at indices {splices[:10].tolist()}{' ...' if len(splices) > 10 else ''}"
                     if len(splices) else "  -> single coherent recording"))

            # ---- 2. why the slab fails ------------------------------------
            pick = rng.choice(idxs, size=min(args.seg_sample, len(idxs)), replace=False)
            areas, anchors, meds, ncomp = [], [], [], []
            cents, masks_kept = [], []
            for i in sorted(pick):
                d = load(int(i))
                valid = d > 0
                if not valid.any():
                    continue
                anchor = float(np.percentile(d[valid], 1))
                anchors.append(anchor); meds.append(float(np.median(d[valid])))
                fg = valid & (d >= anchor) & (d <= anchor + 600.0)
                lb, n = ndimage.label(fg)
                ncomp.append(n)
                if n:
                    sizes = ndimage.sum(fg, lb, range(1, n + 1))
                    areas.append(float(sizes.max()) / d.size)
                    best = lb == (int(np.argmax(sizes)) + 1)
                    ys, xs = np.nonzero(best)
                    cents.append((ys.mean(), xs.mean()))
                    if len(masks_kept) < 12:
                        masks_kept.append(best)
            a = np.array(areas) if areas else np.array([np.nan])
            print(f"  slab anchor (1st pct) median {np.median(anchors):7.0f} mm   "
                  f"frame median depth {np.median(meds):7.0f} mm   "
                  f"gap {np.median(meds) - np.median(anchors):6.0f} mm")
            print(f"  largest component area: median {100 * np.nanmedian(a):5.2f} %  "
                  f"p10 {100 * np.nanpercentile(a, 10):5.2f} %  p90 {100 * np.nanpercentile(a, 90):5.2f} %  "
                  f"(plausible person = 1-35 %)   components/frame median {int(np.median(ncomp))}")
            # Does the blob MOVE? A person walking shifts its centroid and its
            # mask overlaps only partly between frames. A static near-field
            # artefact (Azure Kinect speckle, the invalid-region border) sits
            # still and overlaps almost perfectly -- which an area-only
            # plausibility filter cannot tell apart from a person.
            static = ""
            if len(cents) > 3:
                c = np.array(cents)
                ious = [float((masks_kept[i] & masks_kept[j]).sum())
                        / max(1.0, float((masks_kept[i] | masks_kept[j]).sum()))
                        for i in range(len(masks_kept)) for j in range(i + 1, len(masks_kept))]
                miou = float(np.median(ious)) if ious else float("nan")
                print(f"  blob centroid std: y {c[:, 0].std():5.1f} px  x {c[:, 1].std():5.1f} px   "
                      f"cross-frame mask IoU median {miou:.3f}")
                if miou > 0.8:
                    static = " -- but the blob is STATIC (IoU>0.8): an artefact, not a person"
            verdict = ("slab OK" + static if 0.01 <= np.nanmedian(a) <= 0.35 else
                       "slab EMPTY -- anchor likely on noise" if np.nanmedian(a) < 0.01 else
                       "slab FLOODS the frame -- anchor on a surface spanning the scene")
            print(f"  -> {verdict}")
            report[key] = dict(
                n_frames=int(len(idxs)), adjacent_ddepth_median=adj_med,
                far_ddepth_median=(far_med if np.isfinite(far_med) else None),
                n_splice_candidates=int(len(splices)),
                splice_indices=[int(x) for x in splices[:50]],
                anchor_median_mm=float(np.median(anchors)),
                frame_median_depth_mm=float(np.median(meds)),
                lcc_area_median=float(np.nanmedian(a)), verdict=verdict)

    print("\nWHAT THIS DECIDES")
    print("  * 0 splice candidates for a label => that index run is ONE recording, so R1")
    print("    (contiguous block + guard) is meaningful on it and the ladder replication")
    print("    can proceed; splices found => treat each segment as a recording boundary")
    print("    or restrict in-house to R0 vs R1-within-segment.")
    print("  * 'slab FLOODS'/'slab EMPTY' for most identities => no usable foreground")
    print("    without new segmentation work, so the mechanism suite stays BIWI-only and")
    print("    in-house contributes the ladder (R0 vs R1) on depth and RGB.")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "inhouse_probe.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\n[written] {os.path.join(args.out, 'inhouse_probe.json')}")


if __name__ == "__main__":
    main()
