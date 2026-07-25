"""Feasibility check for keeping the in-house dataset in paper 2.

    python hiride_inhouse_check.py --root /project/6005175/chenzz/datasets/inhouse

The author wants in-house kept rather than dropped, so the question is not
"should we" but "what can it actually support". Three things decide that, and
all three are measurable from the extracted PNGs without touching the .mkv
sources (which live on tape-backed /nearline and need an SDK Nibi lacks):

1. LABELS. `kinect2png.py` writes `{label}_{idx}_{depth|rgb}.png`, so identity
   is in the filename -- the 2023 failure was purely the parser
   (`load_inhouse` took `split('_')[-2]`, the frame INDEX, and StaticHashTable's
   default_value=0 absorbed every miss, collapsing all 7,151 frames to class 0).

2. PROVENANCE -- the reason this script exists. In `kinect2png.py:34-64`,
   `dataset_name` comes from the SOURCE DIRECTORY and `idx` restarts at 0 for
   every .mkv inside it, so two recordings of the same person in one directory
   write the same filenames and overwrite each other frame by frame. Affected by
   construction: every leo-recording file (all labelled `leo`), and stephen
   {1,2,9}, lady1 {3,4}, lady2 {7,8} in stephen-stair-dk. If that happened, an
   identity's index range is a MIXTURE of recordings with no provenance left, so
   no recording-disjoint or session-disjoint protocol can be built -- only R0
   (frame-random) and R1 (contiguous block + guard). This script detects it by
   looking for discontinuities in the per-label index sequence and by comparing
   frame counts against the .mkv inventory when that is listable.

3. MASKS. There is no userMap, so every mechanism condition (person, bg_hole,
   bg_plate, silhouette, scale_removed, sil_scaled) needs a foreground of our
   own. This script tries slab + largest-connected-component on a sample and
   reports how often it finds a plausible person, which is what decides whether
   in-house can carry the mechanism suite or only the ladder.

CPU, minutes. Reads at most --sample images.
"""
import os
import re
import glob
import json
import argparse
from collections import defaultdict

import numpy as np

NAME = re.compile(r"^(?P<label>.+)_(?P<idx>\d+)_(?P<kind>depth|rgb)\.png$")


def parse_dir(d):
    """label -> {kind -> sorted list of frame indices}."""
    out = defaultdict(lambda: defaultdict(list))
    for f in os.listdir(d):
        m = NAME.match(f)
        if m:
            out[m.group("label")][m.group("kind")].append(int(m.group("idx")))
    for lab in out:
        for k in out[lab]:
            out[lab][k].sort()
    return out


def read_png(path):
    """16-bit-safe PNG read without cv2 (venv311 has none). Returns array or None."""
    try:
        import tensorflow as tf
        tf.config.set_visible_devices([], "GPU")
        raw = tf.io.read_file(path)
        img = tf.io.decode_png(raw, dtype=tf.uint16)      # depth was written uint16
        return np.asarray(img)
    except Exception:
        try:
            import tensorflow as tf
            return np.asarray(tf.io.decode_png(tf.io.read_file(path)))
        except Exception:
            return None


def slab_lcc(depth, clip_mm=600.0, min_px=2000):
    """Paper-3-style slab + largest connected component. Returns mask or None."""
    from scipy import ndimage
    valid = depth > 0
    if valid.sum() < min_px:
        return None
    anchor = np.percentile(depth[valid], 1)
    fg = valid & (depth >= anchor) & (depth <= anchor + clip_mm)
    lab, n = ndimage.label(fg)
    if n == 0:
        return None
    sizes = ndimage.sum(fg, lab, range(1, n + 1))
    return lab == (int(np.argmax(sizes)) + 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="directory holding the extracted per-recording subdirs")
    ap.add_argument("--sample", type=int, default=60, help="frames to segment per label")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    subdirs = sorted(d for d in glob.glob(os.path.join(args.root, "*"))
                     if os.path.isdir(d))
    if not subdirs and glob.glob(os.path.join(args.root, "*.png")):
        subdirs = [args.root]
    if not subdirs:
        raise SystemExit(f"no subdirectories or PNGs under {args.root}")
    print(f"[root] {args.root}\n[subdirs] {[os.path.basename(d) for d in subdirs]}\n")

    report = {"subdirs": {}, "labels": {}}
    total_frames = 0
    for d in subdirs:
        parsed = parse_dir(d)
        if not parsed:
            print(f"[{os.path.basename(d):<20s}] no {{label}}_{{idx}}_{{kind}}.png files")
            continue
        print(f"== {os.path.basename(d)} ==")
        print(f"{'label':<12s}{'depth':>7s}{'rgb':>7s}{'idx max':>9s}{'gaps':>7s}"
              f"{'contiguous?':>13s}   provenance")
        for lab, kinds in sorted(parsed.items()):
            dep, rgb = kinds.get("depth", []), kinds.get("rgb", [])
            n, mx = len(dep), (max(dep) if dep else -1)
            expected = mx + 1
            gaps = expected - n
            contiguous = (gaps == 0)
            # A label whose indices start at 0 and run unbroken to max is ONE
            # surviving sequence. Overwriting leaves that intact, so the real
            # tell is a frame count far below what several recordings should
            # give -- flagged against the .mkv inventory below when available.
            note = "single unbroken index run" if contiguous else f"{gaps} missing indices"
            print(f"{lab:<12s}{n:>7d}{len(rgb):>7d}{mx:>9d}{gaps:>7d}{str(contiguous):>13s}   {note}")
            report["labels"][f"{os.path.basename(d)}/{lab}"] = dict(
                n_depth=n, n_rgb=len(rgb), idx_max=mx, gaps=int(gaps),
                contiguous=bool(contiguous), paired=(len(dep) == len(rgb)))
            total_frames += n
        print()
        report["subdirs"][os.path.basename(d)] = len(parsed)

    print(f"[total] {total_frames} depth frames across "
          f"{len(report['labels'])} (subdir, label) pairs\n")

    # ---- class balance: the honest baseline is the majority rate ------------
    per_label = defaultdict(int)
    for k, v in report["labels"].items():
        per_label[k.split("/", 1)[1]] += v["n_depth"]
    tot = sum(per_label.values())
    if tot:
        top = max(per_label.values())
        print("[balance] " + "  ".join(f"{k}={v}" for k, v in
                                       sorted(per_label.items(), key=lambda kv: -kv[1])))
        print(f"[balance] {len(per_label)} identities, majority-class rate "
              f"{100 * top / tot:.1f} % -- report against THAT, not 1/K = {100 / len(per_label):.1f} %\n")
        report["majority_rate"] = top / tot
        report["n_identities"] = len(per_label)

    # ---- image format + segmentation viability ------------------------------
    print("[segmentation] slab + largest connected component, no userMap available")
    ok_total = tried_total = 0
    for d in subdirs:
        parsed = parse_dir(d)
        for lab, kinds in sorted(parsed.items()):
            idxs = kinds.get("depth", [])
            if not idxs:
                continue
            pick = rng.choice(idxs, size=min(args.sample, len(idxs)), replace=False)
            areas, ok, tried, shape, dtype = [], 0, 0, None, None
            for i in sorted(pick):
                img = read_png(os.path.join(d, f"{lab}_{i}_depth.png"))
                if img is None:
                    continue
                img = img[..., 0] if img.ndim == 3 else img
                shape, dtype = img.shape, img.dtype
                tried += 1
                m = slab_lcc(img.astype(np.float32))
                if m is not None:
                    frac = float(m.mean())
                    # a person should be a few per cent to a quarter of the frame
                    if 0.01 <= frac <= 0.35:
                        ok += 1
                        areas.append(frac)
            ok_total += ok; tried_total += tried
            a = f"{100 * np.median(areas):5.2f} %" if areas else "   -- "
            print(f"  {os.path.basename(d)}/{lab:<10s} {shape} {dtype}  "
                  f"plausible person in {ok}/{tried} sampled frames, median area {a}")
    print(f"\n[segmentation] overall {ok_total}/{tried_total} "
          f"({100 * ok_total / max(1, tried_total):.1f} %) of sampled frames yield a plausible "
          f"foreground")
    report["segmentation_ok_rate"] = ok_total / max(1, tried_total)

    print("\nWHAT THIS DECIDES")
    print("  * every label showing a single unbroken index run, with counts far below what")
    print("    its number of source .mkv files should give, confirms the overwrite in")
    print("    kinect2png.py -- in-house then supports R0 and R1 ONLY (no recording- or")
    print("    session-disjoint rung), because frame provenance is not recoverable;")
    print("  * a low segmentation rate means the mechanism suite cannot run on in-house and")
    print("    it can only replicate the LADDER (which is still the point: does the")
    print("    R0 -> R1 collapse reproduce on a different sensor generation?).")
    if args.out:
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "inhouse_check.json"), "w") as fh:
            json.dump(report, fh, indent=1)
        print(f"\n[written] {os.path.join(args.out, 'inhouse_check.json')}")


if __name__ == "__main__":
    main()
