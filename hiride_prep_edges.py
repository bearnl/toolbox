"""Rebuild the image shards with a MASK-AWARE resize. Nothing else changes.

    (inside a job that has staged the archives)
    python hiride_prep_edges.py --root $SLURM_TMPDIR/biwi --src $SCRATCH/hiride2/prep \
        --out $SCRATCH/hiride2/prep_edges

THE DEFECT. hiride_prep.py AREA-resizes depth and RGB over the WHOLE 640x480
frame and only then applies a NEAREST-resized mask. The box average therefore
mixes person pixels (~2955 mm) with background (~3842 mm) wherever the two meet:
a rim pixel that is half person comes out ~440 mm behind the body, on roughly
9 % of person pixels, concentrated exactly where shape information is richest.
RGB has the same problem with colour.

EVIDENCE IT MATTERS. Eroding the mask by 2 px -- i.e. throwing the rim away --
RAISES the linear probe from 57.57 to 60.52 % at R1 and 13.72 to 14.54 % at R4
(`interior_only` vs `scale_removed`, signal_diagnostic.json). Erosion is a
workaround that also discards real outline; this fixes the cause instead.

THE FIX. Resize the two populations separately and recombine:
    person_out     = AREA(depth * person) / AREA(person)
    background_out = AREA(depth * bg)     / AREA(bg)
    mask_out       = AREA(person) >= 0.5
    depth_out      = where(mask_out, person_out, background_out)
so no output pixel ever averages across the silhouette. Depth additionally
excludes invalid (0) readings from its numerator and denominator, which the
original code also failed to do -- invalid pixels were averaged in as if they
were surfaces at 0 mm.

Only the shards are rewritten. manifest.npz and cues.npz are copied from --src
unchanged, so splits, the eligibility filter and every published number stay
comparable; the ONLY difference between prep and prep_edges is the resize.
"""
import os
import json
import shutil
import argparse

import numpy as np

from hiride_data import build_manifest, load_manifest
from hiride_pgm import read_pgm
from hiride_prep import _StreamWriter, SEQ_TAG


def _edges(src, dst):
    return np.linspace(0, src, dst + 1).astype(np.int64)


def area_sum(a, size):
    """Box SUM over an arbitrary rescale (float in, float out)."""
    if a.ndim == 2:
        a = a[..., None]
    h, w = a.shape[:2]
    ry, rx = _edges(h, size), _edges(w, size)
    rows = np.add.reduceat(a.astype(np.float64), ry[:-1], axis=0)
    return np.add.reduceat(rows, rx[:-1], axis=1)


def masked_resize(img, keep, size):
    """AREA-resize `img` using ONLY pixels where `keep`, plus the kept fraction."""
    num = area_sum(img * keep[..., None] if img.ndim == 3 else img * keep, size)
    den = area_sum(keep.astype(np.float64), size)
    out = np.divide(num, np.maximum(den, 1e-9))
    return out, den


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="staged BIWI tree")
    ap.add_argument("--src", required=True, help="existing prep, for manifest + cues")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    S = args.size

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    # Reuse the ORIGINAL manifest so row order, splits and cues stay identical.
    man = load_manifest(os.path.join(args.src, "manifest.npz"))
    for f in ("manifest.npz", "manifest.csv", "cues.npz"):
        p = os.path.join(args.src, f)
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(args.out, f))
    raw = build_manifest(args.root, verbose=False)
    key = lambda m, i: (str(m["seq"][i]), str(m["subject"][i]), int(m["frame"][i]))
    raw_of = {key(raw, i): i for i in range(len(raw["frame"]))}

    stats = {"n": 0, "rim_frac": []}
    for seq, tag in SEQ_TAG.items():
        rows = np.where(man["seq"] == seq)[0]
        if not len(rows):
            continue
        w_d = _StreamWriter(os.path.join(args.out, f"{tag}_depth.npy"), np.uint16, (len(rows), S, S))
        w_c = _StreamWriter(os.path.join(args.out, f"{tag}_rgb.npy"), np.uint8, (len(rows), S, S, 3))
        w_m = _StreamWriter(os.path.join(args.out, f"{tag}_mask.npy"), np.uint8, (len(rows), S, S))
        np.savez(os.path.join(args.out, f"{tag}_index.npz"), manifest_row=rows)
        print(f"[{tag}] {len(rows)} frames")
        for j, r in enumerate(rows):
            k = key(man, int(r))
            i = raw_of.get(k)
            if i is None:
                raise SystemExit(f"frame {k} present in prep but not in {args.root}")
            d, _ = read_pgm(os.path.join(raw["root"], raw["depth"][i]))
            u, _ = read_pgm(os.path.join(raw["root"], raw["user"][i]))
            c = np.asarray(tf.io.decode_jpeg(tf.io.read_file(
                os.path.join(raw["root"], raw["rgb"][i])), channels=3))
            person = u > 0
            valid = d > 0
            # BIWI ships RGB at 960x1280 and depth/userMap at 480x640 -- exactly
            # 2x. The mask has to be upsampled to the colour grid before it can
            # select colour pixels. The original prep never hit this because it
            # resized each stream independently and only applied the mask at the
            # shared 256^2 output, which also means its RGB rim was masked with a
            # mask from a DIFFERENT resolution.
            if c.shape[:2] != person.shape:
                fy = c.shape[0] // person.shape[0]
                fx = c.shape[1] // person.shape[1]
                if (fy, fx) != (2, 2) or c.shape[:2] != (person.shape[0] * 2,
                                                         person.shape[1] * 2):
                    raise SystemExit(f"unexpected rgb {c.shape} vs depth {d.shape}")
                person_c = np.repeat(np.repeat(person, fy, axis=0), fx, axis=1)
            else:
                person_c = person
            # depth: person and background resized from DISJOINT, VALID pixels
            p_ok, p_den = masked_resize(d.astype(np.float64), person & valid, S)
            b_ok, b_den = masked_resize(d.astype(np.float64), (~person) & valid, S)
            frac, _ = masked_resize(np.ones_like(d, np.float64), person, S)
            cover = area_sum(person.astype(np.float64), S) / area_sum(
                np.ones_like(d, np.float64), S)
            m_out = (cover[..., 0] >= 0.5)
            dep = np.where(m_out & (p_den[..., 0] > 0), p_ok[..., 0],
                           np.where(b_den[..., 0] > 0, b_ok[..., 0], 0.0))
            # rgb: same recombination, no validity notion
            cp, cpd = masked_resize(c.astype(np.float64), person_c, S)
            cb, cbd = masked_resize(c.astype(np.float64), ~person_c, S)
            rgb = np.where((m_out & (cpd[..., 0] > 0))[..., None], cp,
                           np.where((cbd[..., 0] > 0)[..., None], cb, 0.0))
            w_d.append(np.clip(dep, 0, 65535).astype(np.uint16), expect=j)
            w_c.append(np.clip(rgb, 0, 255).astype(np.uint8), expect=j)
            w_m.append(m_out.astype(np.uint8), expect=j)
            if j % 2000 == 0:
                mixed = float(((cover[..., 0] > 0) & (cover[..., 0] < 1)).mean())
                stats["rim_frac"].append(mixed)
                print(f"   {j}/{len(rows)}  mixed-boundary output pixels {100 * mixed:.2f} %")
            stats["n"] += 1
        for w in (w_d, w_c, w_m):
            w.close()

    meta = dict(shards_ok=True, size=S, source_prep=args.src,
                resize="mask-aware: person and background resized separately, "
                       "invalid depth excluded from both",
                n_frames=stats["n"],
                mixed_boundary_pixel_frac=float(np.mean(stats["rim_frac"])) if stats["rim_frac"] else None)
    with open(os.path.join(args.out, "prep_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\n[done] {stats['n']} frames -> {args.out}")
    print(f"  output pixels that straddled the silhouette (and were previously "
          f"averaged across it): {100 * meta['mixed_boundary_pixel_frac']:.2f} %")


if __name__ == "__main__":
    main()
