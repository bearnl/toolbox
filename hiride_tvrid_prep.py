"""Build a BIWI-format prep directory for TVRID (external validity, axis 2).

    python hiride_tvrid_prep.py --root $SLURM_TMPDIR/tvrid --out $SCRATCH/hiride2/prep_tvrid

TVRID (ICPR 2026 top-view RGB-D re-ID corpus, Zenodo 10.5281/zenodo.20070280,
CC-BY-4.0) is the second corpus for the ladder replication. VERIFIED BY
INSPECTION of the zip central directory + labels (2026-08-31):

  original.zip:  train/<person>/<cam>/<passage>/<timestamp>_{depth,RGB}.png
                 test_public/<hash>/<timestamp>_{depth,RGB}.png
                 depth = 640x480 16-bit grayscale PNG; RGB = 640x480 PNG
  labels:        train_labels.csv + test_secret_map.csv (hash -> person/cam/
                 passage), 62 + 26 = 88 persons, 4 cameras (flat/upward/
                 downward/upsideDown), passage1 + passage2 per (person, cam),
                 ~25-65 frames per tracklet at 15 fps (2-4 s each)

MAPPING INTO THE BIWI CONVENTIONS (deliberate, so hiride_data's policies apply
unchanged -- same trick as hiride_inhouse_prep.py):

  passage1  -> seq "Training",        session "A"
  passage2  -> seq "Testing/Walking", session "B"
  group     -> "<seq>|<person>|<cam>"   (one tracklet = one recording)

WHAT THE RUNGS THEN MEAN ON THIS CORPUS -- state this in the paper, the names
are BIWI's, the semantics here are NOT:
  R0  frame-random over passage1        (the leaky policy, as everywhere)
  R1  contiguous block within a tracklet (+ small guard; tracklets are only
      ~40 frames, so the guard is 5, not BIWI's 150)
  R4_cross_session  -> **CROSS-PASSAGE**: same day, same clothes, same
      cameras, minutes apart. It is an R3-analogue (recording-disjoint), NOT a
      clothing/session change. TVRID has no multi-day session -- that axis
      remains BIWI-only (and IAS-Lab RGBD-ID, which would supply it, is no
      longer publicly downloadable; see HIRIDE_HANDOFF 14.3).

No person masks exist -> no mask shard, no cues; run the trainer with
`--eligibility all --condition full` (wave 21 does). Ladder only, by design.
"""
import os
import csv
import json
import argparse

import numpy as np

from hiride_data import save_manifest

CAMS = ("flat", "upward", "downward", "upsideDown")


def scan_tracklets(root, labels_dir):
    """-> list of (person, cam, passage, dirpath) for every tracklet on disk."""
    out = []
    train_root = os.path.join(root, "train")
    if os.path.isdir(train_root):
        for person in sorted(os.listdir(train_root)):
            pdir = os.path.join(train_root, person)
            if not os.path.isdir(pdir):
                continue
            for cam in sorted(os.listdir(pdir)):
                for passage in sorted(os.listdir(os.path.join(pdir, cam))):
                    out.append((f"t{person}", cam, passage,
                                os.path.join(pdir, cam, passage)))
    smap = os.path.join(labels_dir, "test_secret_map.csv")
    test_root = os.path.join(root, "test_public")
    if os.path.isdir(test_root) and os.path.exists(smap):
        with open(smap, newline="") as fh:
            for row in csv.DictReader(fh):
                d = os.path.join(test_root, row["public_gallery_id"])
                if os.path.isdir(d):
                    out.append((f"s{row['person_id']}", row["cam_name"],
                                row["passage_name"], d))
    elif os.path.isdir(test_root):
        print("[warn] test_public/ present but no test_secret_map.csv -- "
              "test identities cannot be labelled and are SKIPPED")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True,
                    help="extracted original.zip (holds train/ and test_public/)")
    ap.add_argument("--labels", default=None,
                    help="dir holding test_secret_map.csv (default: --root, then "
                         "--root/TVRID_labels)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--depth-scale", type=float, default=1.0,
                    help="divide raw 16-bit depth by this to get millimetres. "
                         "RealSense D455 PNG exports are normally already mm "
                         "(scale 1); the sanity print below is the check.")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")
    from hiride_prep import _resizer
    rs_depth, rs_rgb, _ = _resizer(args.size)

    labels_dir = args.labels or args.root
    if not os.path.exists(os.path.join(labels_dir, "test_secret_map.csv")):
        alt = os.path.join(args.root, "TVRID_labels")
        if os.path.exists(os.path.join(alt, "test_secret_map.csv")):
            labels_dir = alt
    tracklets = scan_tracklets(args.root, labels_dir)
    if not tracklets:
        raise SystemExit(f"no tracklets under {args.root} -- expected train/ "
                         f"and/or test_public/ from original.zip")
    n_missing_cam = sum(1 for c in CAMS if c not in {t[1] for t in tracklets})
    print(f"[scan] {len(tracklets)} tracklets, "
          f"{len({t[0] for t in tracklets})} persons"
          + (f" ({n_missing_cam} camera names unseen)" if n_missing_cam else ""))

    # ---- per-frame inventory ----------------------------------------------
    rows = []          # (seq, person, cam, frame_idx, ts_us, depth_rel, rgb_rel)
    for person, cam, passage, tdir in tracklets:
        seq = "Training" if passage == "passage1" else "Testing/Walking"
        stems = sorted({f[:-len("_depth.png")] for f in os.listdir(tdir)
                        if f.endswith("_depth.png")})
        kept = 0
        for i, stem in enumerate(stems):
            dp, cp = f"{stem}_depth.png", f"{stem}_RGB.png"
            if not os.path.exists(os.path.join(tdir, cp)):
                continue
            # timestamp: ..._HH_MM_SS_micro -> microseconds within the day
            parts = stem.split("_")
            try:
                hh, mm, ss, us = (int(p) for p in parts[-4:])
                ts = ((hh * 3600 + mm * 60 + ss) * 1_000_000 + us)
            except ValueError:
                ts = i
            rel = os.path.relpath(tdir, args.root).replace("\\", "/")
            rows.append((seq, person, cam, kept, ts, f"{rel}/{dp}", f"{rel}/{cp}"))
            kept += 1
    rows.sort(key=lambda r: (r[0], r[1], r[2], r[3]))
    n = len(rows)
    print(f"[inventory] {n} paired frames")

    # frame numbers must be unique WITHIN a group and ordered in time; index
    # within the tracklet does both (ts kept for the record)
    man = {
        "root": args.root,
        "seq": np.array([r[0] for r in rows]),
        "subject": np.array([r[1] for r in rows]),
        "sess_letter": np.array(["a" if r[0] == "Training" else "b" for r in rows]),
        "session": np.array(["A" if r[0] == "Training" else "B" for r in rows]),
        "frame": np.array([r[3] for r in rows], dtype=np.int64),
        "ts": np.array([r[4] for r in rows], dtype=np.int64),
        "depth": np.array([r[5] for r in rows]),
        "user": np.array([""] * n),
        "rgb": np.array([r[6] for r in rows]),
    }
    man["group"] = np.array([f"{r[0]}|{r[1]}|{r[2]}" for r in rows])
    save_manifest(man, os.path.join(args.out, "manifest.npz"))
    for seq in ("Training", "Testing/Walking"):
        m = man["seq"] == seq
        gl = [int((man["group"] == g).sum()) for g in np.unique(man["group"][m])]
        print(f"  {seq:<18s} frames={int(m.sum()):6d} "
              f"subjects={len(set(man['subject'][m]))} "
              f"tracklets={len(gl)} (frames/tracklet p10 {np.percentile(gl, 10):.0f} "
              f"median {np.median(gl):.0f} p90 {np.percentile(gl, 90):.0f})")

    # ---- shards, one per mapped sequence ----------------------------------
    S = args.size
    tags = {"Training": "training", "Testing/Walking": "testing_walking"}
    for seq, tag in tags.items():
        idx = np.where(man["seq"] == seq)[0]
        dep = np.zeros((len(idx), S, S), np.uint16)
        rgb = np.zeros((len(idx), S, S, 3), np.uint8)
        for j, i in enumerate(idx):
            a = np.asarray(tf.io.decode_png(
                tf.io.read_file(os.path.join(args.root, str(man["depth"][i]))),
                dtype=tf.uint16))[..., 0]
            b = np.asarray(tf.io.decode_png(
                tf.io.read_file(os.path.join(args.root, str(man["rgb"][i]))),
                channels=3))
            if args.depth_scale != 1.0:
                a = (a.astype(np.float64) / args.depth_scale).astype(np.uint16)
            dep[j] = rs_depth(a)
            rgb[j] = rs_rgb(b)
            if (j + 1) % 2000 == 0 or j == len(idx) - 1:
                print(f"[shard {tag}] {j + 1}/{len(idx)}")
        np.save(os.path.join(args.out, f"{tag}_depth.npy"), dep)
        np.save(os.path.join(args.out, f"{tag}_rgb.npy"), rgb)
        np.savez(os.path.join(args.out, f"{tag}_index.npz"),
                 manifest_row=idx.astype(np.int64))

    dep_tr = np.load(os.path.join(args.out, "training_depth.npy"), mmap_mode="r")
    valid = np.asarray(dep_tr[::17]) > 0
    med = float(np.median(np.asarray(dep_tr[::17])[valid])) if valid.any() else 0.0
    if not (500 <= med <= 8000):
        print(f"[WARN] training depth median {med:.0f} raw units does not look "
              f"like millimetres of an indoor scene -- check --depth-scale "
              f"before trusting any depth run.")
    meta = dict(
        shards_ok=True, dataset="tvrid", size=S, n_frames=n,
        n_identities=len(set(man["subject"].tolist())),
        depth_scale=args.depth_scale, depth_median_raw=med,
        has_mask=False, has_cues=False,
        sequence_semantics={
            "Training": "passage1 (all cameras)",
            "Testing/Walking": "passage2 -- SAME day, SAME clothes, minutes "
                               "apart. The R4-named rung is CROSS-PASSAGE "
                               "(an R3-analogue), NOT a session/clothing "
                               "change. State this wherever a number from "
                               "this prep is quoted."},
        note="Run with --eligibility all --condition full (no masks, no cues). "
             "Tracklets are ~25-65 frames, so R1 uses guard 5 (ref-guard 5) "
             "and the cross rung needs --cross-val-guard 5.")
    with open(os.path.join(args.out, "prep_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\n[done] {args.out}  training depth median {med:.0f} "
          f"(expect ~1500-4500 mm for a top-view indoor corpus)")


if __name__ == "__main__":
    main()
