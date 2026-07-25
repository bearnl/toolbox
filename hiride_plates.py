"""Per-recording background plates, for the exact-complement scene control.

    python hiride_plates.py --prep $SCRATCH/hiride2/prep

`bg_hole` (person -> constant) leaves a silhouette-shaped hole, so a result
above chance there is ambiguous: the model may be reading the SCENE, or it may
be reading the HOLE. The control that separates them is a plate: the person's
pixels replaced by what the camera sees when nobody is standing there, so no
person-shaped boundary survives anywhere in the frame.

Method, per recording (seq|subject): take EVERY frame of that recording, keep
the pixels the shipped userMap says are not the person (and, for depth, are
valid), and take the per-pixel median over frames. This is why the empty-mask
frames matter -- 22.3 % of Training frames have no person tracked at all and
show the bare room, which is exactly the observation a plate wants. Pixels
never seen empty in the whole recording are then filled from a GLOBAL plate
(the per-pixel median across all recordings), so no person-shaped hole and no
recording-specific fill pattern survives.

Writes plates_{depth,rgb}.npy, plates_global_{depth,rgb}.npy, plate_groups.npy
and plates_seen_{depth,rgb}.npy to --prep. CPU only, a few minutes.

READ THIS BEFORE INTERPRETING bg_plate: the condition is only meaningful at
RECORDING-DISJOINT rungs (R3, R4). Removing the only moving object from a
fixed-camera recording makes every frame of that recording the same image, so
under a within-recording split (R0, R1) train and test are near-duplicates
whatever the temporal guard, and accuracy goes to ~100 % by construction --
measured: depth 98.06 %, rgb 99.96 % at R1 guard 150. That is a demonstration
of the leak mechanism, not a measurement of the scene.
"""
import os
import json
import argparse

import numpy as np

from hiride_data import load_manifest

SEQ_TAG = {"Training": "training", "Testing/Still": "testing_still",
           "Testing/Walking": "testing_walking"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--chunk", type=int, default=64,
                    help="rows of the frame processed at once (memory knob)")
    args = ap.parse_args()

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    imgs, masks, where = {}, {}, {}
    for seq, tag in SEQ_TAG.items():
        ipath = os.path.join(args.prep, f"{tag}_index.npz")
        if not os.path.exists(ipath):
            continue
        rows = np.load(ipath)["manifest_row"]
        imgs[tag] = (np.load(os.path.join(args.prep, f"{tag}_depth.npy"), mmap_mode="r"),
                     np.load(os.path.join(args.prep, f"{tag}_rgb.npy"), mmap_mode="r"))
        masks[tag] = np.load(os.path.join(args.prep, f"{tag}_mask.npy"), mmap_mode="r")
        for pos, row in enumerate(rows):
            where[int(row)] = (tag, pos)
        print(f"[load] {tag}: {len(rows)} frames")

    groups = sorted(set(man["group"].tolist()))
    S = masks[next(iter(masks))].shape[1]
    pd_all = np.zeros((len(groups), S, S), np.uint16)
    pr_all = np.zeros((len(groups), S, S, 3), np.uint8)
    # Explicit "was this pixel ever observed without the person?" masks, ONE PER
    # MODALITY. Needed because 0 is a legitimate value in both (invalid depth,
    # black pixel), and an unfilled hole is precisely the person-shaped boundary
    # the bg_plate condition exists to eliminate. They differ: depth also
    # requires a valid reading, and ~25 % of pixels have none (census
    # valid_frac 0.7-0.8), while their RGB is perfectly good -- so using the
    # depth mask for RGB would median-fill a quarter of the RGB plate.
    seen_d_all = np.zeros((len(groups), S, S), bool)
    seen_c_all = np.zeros((len(groups), S, S), bool)
    stats = {}
    for gi, g in enumerate(groups):
        rows = np.where(man["group"] == g)[0]
        pos_by_tag = {}
        for r in rows:
            tag, pos = where[int(r)]
            pos_by_tag.setdefault(tag, []).append(pos)
        # One recording lives in one sequence, but resolve generally anyway.
        depth_med = np.zeros((S, S), np.float32)
        rgb_med = np.zeros((S, S, 3), np.float32)
        seen_d = np.zeros((S, S), bool)
        seen_c = np.zeros((S, S), bool)
        for tag, ps in pos_by_tag.items():
            ps = np.array(sorted(ps))
            dsh, rsh = imgs[tag]
            msh = masks[tag]
            for y0 in range(0, S, args.chunk):
                y1 = min(S, y0 + args.chunk)
                d = np.asarray(dsh[ps, y0:y1]).astype(np.float32)
                m = np.asarray(msh[ps, y0:y1]) > 0
                bg = (~m) & (d > 0)                     # background AND valid depth
                dm = np.where(bg, d, np.nan)
                with np.errstate(all="ignore"):
                    md = np.nanmedian(dm, axis=0)
                ok = np.isfinite(md)
                depth_med[y0:y1][ok] = md[ok]
                seen_d[y0:y1] |= ok
                c = np.asarray(rsh[ps, y0:y1]).astype(np.float32)
                cm = np.where((~m)[..., None], c, np.nan)
                with np.errstate(all="ignore"):
                    mc = np.nanmedian(cm, axis=0)
                okc = np.isfinite(mc).all(-1)
                rgb_med[y0:y1][okc] = mc[okc]
                seen_c[y0:y1] |= okc
        seen_d_all[gi] = seen_d
        seen_c_all[gi] = seen_c
        hole = float(1.0 - seen_d.mean())
        hole_c = float(1.0 - seen_c.mean())
        pd_all[gi] = np.clip(depth_med, 0, 65535).astype(np.uint16)
        pr_all[gi] = np.clip(rgb_med, 0, 255).astype(np.uint8)
        stats[g] = dict(n_frames=int(len(rows)), hole_frac=round(hole, 5),
                        hole_frac_rgb=round(hole_c, 5),
                        bg_median_mm=float(np.median(depth_med[seen_d])) if seen_d.any() else 0.0)
        if gi % 20 == 0 or gi == len(groups) - 1:
            print(f"[plate] {gi + 1}/{len(groups)} {g:<24s} frames={len(rows):5d} "
                  f"hole d={100 * hole:5.2f}% c={100 * hole_c:5.2f}%  "
                  f"bg_med={stats[g]['bg_median_mm']:.0f} mm")

    # Fill each recording's unseen pixels from a GLOBAL plate (the per-pixel
    # median across every recording that did see them) rather than from a flat
    # scalar. A flat fill leaves a recording-specific blob wherever that subject
    # stood longest, and at R3 a subject's two recordings share such blobs -- so
    # the "is it the scene?" control would be reading a fingerprint of the
    # person's own movement. The global plate carries no recording-specific
    # content: BIWI is one room with a fixed camera (plates differ by a median
    # of 55 mm), which is exactly what makes this substitution defensible here.
    for arr, seen_all, label in ((pd_all, seen_d_all, "depth"), (pr_all, seen_c_all, "rgb")):
        m = seen_all if arr.ndim == 3 else seen_all[..., None]
        stack = np.where(m, arr, np.nan).astype(np.float32)
        with np.errstate(all="ignore"):
            glob = np.nanmedian(stack, axis=0)
        gseen = np.isfinite(glob) if glob.ndim == 2 else np.isfinite(glob).all(-1)
        glob = np.nan_to_num(glob, nan=0.0)
        before = 1.0 - seen_all.mean()
        for gi in range(len(groups)):
            hole = ~seen_all[gi] & gseen
            if hole.any():
                arr[gi][hole] = glob[hole].astype(arr.dtype)
        seen_all |= gseen[None]
        print(f"[global-fill] {label}: hole fraction {100 * before:.3f} % -> "
              f"{100 * (1.0 - seen_all.mean()):.3f} % "
              f"(global plate undefined on {100 * (1 - gseen.mean()):.3f} % of pixels)")
        np.save(os.path.join(args.prep, f"plates_global_{label}.npy"), glob.astype(arr.dtype))

    np.save(os.path.join(args.prep, "plates_seen_depth.npy"), seen_d_all)
    np.save(os.path.join(args.prep, "plates_seen_rgb.npy"), seen_c_all)
    np.save(os.path.join(args.prep, "plates_depth.npy"), pd_all)
    np.save(os.path.join(args.prep, "plates_rgb.npy"), pr_all)
    np.save(os.path.join(args.prep, "plate_groups.npy"), np.array(groups))
    for label, key in (("depth", "hole_frac"), ("rgb", "hole_frac_rgb")):
        h = np.array([s[key] for s in stats.values()])
        print(f"\n[done] {len(groups)} {label} plates: hole fraction BEFORE the "
              f"global fill: median {100 * np.median(h):.3f} %  max {100 * h.max():.3f} %")
    print("  (holes are pixels never seen without the person in that whole recording;")
    print("   they are filled above from the global plate, so neither a person-shaped")
    print("   boundary nor a recording-specific fill pattern survives. -> " + args.prep + ")")
    with open(os.path.join(args.prep, "plates_meta.json"), "w") as fh:
        json.dump(stats, fh, indent=1)


if __name__ == "__main__":
    main()
