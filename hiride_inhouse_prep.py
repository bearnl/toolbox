"""Build a prep directory for the in-house dataset, in the BIWI prep's format.

    python hiride_inhouse_prep.py --root /project/6005175/chenzz/datasets/inhouse \
        --out $SCRATCH/hiride2/prep_inhouse

The in-house corpus is kept in paper 2 (author's call) as a SENSOR-GENERATION
replication: does the frame-random -> contiguous-block collapse reproduce on an
Azure Kinect, or is it a property of BIWI's Kinect capture? What it can and
cannot support was measured by hiride_inhouse_check.py / hiride_inhouse_probe.py:

  CAN   R0_frame_random and R1_block (the ladder), depth and RGB.
  CANNOT any recording- or session-disjoint rung: kinect2png.py restarted `idx`
        at 0 for every .mkv while writing them all into one directory, so 26
        source recordings were overwritten into 10 index runs and the frame
        provenance is gone. The probe found ~no splices, so each surviving run
        is one coherent recording (the longest file overwrote the rest), which
        is what R1 needs -- but the recordings that were overwritten are not
        recoverable, including `bear`'s aug-11 vs mar-22 pair, a real
        cross-session contrast destroyed at extraction.
  CANNOT the mechanism suite: there is no userMap, and slab+LCC finds no person
        for ANY identity -- the anchor lands 1.4-2.7 m in front of the scene
        (Azure Kinect near-field speckle, hundreds of components per frame), and
        where a blob does survive the area filter it is STATIC across frames
        (cross-frame mask IoU 0.96-0.98), i.e. an artefact, not a body.

So this writes depth and RGB shards and NO mask shard, and the trainer must be
run with `--eligibility all` (there are no cues to filter on) and
`--condition full`.

Layout written, deliberately identical to hiride_prep.py's so hiride_train.py,
hiride_collate.py and hiride_stats.py need no in-house special case:
    manifest.npz              seq="Training" for every row, subject=<label>,
                              frame=<idx>, group="Training|<label>"
    training_depth.npy        (N,256,256) uint16 mm
    training_rgb.npy          (N,256,256,3) uint8
    training_index.npz        manifest_row
    prep_meta.json            shards_ok + provenance notes

`seq="Training"` is a deliberate convention, not a claim about the data: it is
what makes hiride_data's existing R0/R1 policies apply unchanged, and it makes
R3/R4 raise, which is the correct behaviour here.

CPU, a few minutes.
"""
import os
import re
import json
import argparse

import numpy as np

from hiride_data import save_manifest

NAME = re.compile(r"^(?P<label>.+)_(?P<idx>\d+)_(?P<kind>depth|rgb)\.png$")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import tensorflow as tf
    tf.config.set_visible_devices([], "GPU")

    # ---- inventory ---------------------------------------------------------
    rows = []
    for sd in sorted((d for d in os.scandir(args.root) if d.is_dir()), key=lambda e: e.name):
        found = {}
        for f in os.listdir(sd.path):
            m = NAME.match(f)
            if m:
                found.setdefault((m.group("label"), int(m.group("idx"))), set()).add(m.group("kind"))
        for (lab, idx), kinds in sorted(found.items()):
            if {"depth", "rgb"} <= kinds:          # both modalities or drop the frame
                rows.append((sd.name, lab, idx))
        print(f"[scan] {sd.name}: {sum(1 for r in rows if r[0] == sd.name)} paired frames")
    if not rows:
        raise SystemExit(f"no {{label}}_{{idx}}_{{depth|rgb}}.png pairs under {args.root}")
    rows.sort(key=lambda r: (r[1], r[2]))          # by subject, then frame

    n = len(rows)
    man = {
        "root": args.root,
        # every in-house frame is "Training": one session, no cross-session rung.
        "seq": np.array(["Training"] * n),
        "subject": np.array([r[1] for r in rows]),
        "sess_letter": np.array(["a"] * n),
        "session": np.array(["A"] * n),
        "frame": np.array([r[2] for r in rows], dtype=np.int64),
        "ts": np.array([r[2] for r in rows], dtype=np.int64),
        "depth": np.array([f"{r[0]}/{r[1]}_{r[2]}_depth.png" for r in rows]),
        "user": np.array([""] * n),                # no userMap exists
        "rgb": np.array([f"{r[0]}/{r[1]}_{r[2]}_rgb.png" for r in rows]),
    }
    man["group"] = np.array([f"Training|{r[1]}" for r in rows])
    # `dir` is not part of the BIWI manifest; keep it so the room/identity
    # confound (each recording directory holds a disjoint set of people) stays
    # visible to anything that reads this manifest later.
    man["dir"] = np.array([r[0] for r in rows])
    save_manifest(man, os.path.join(args.out, "manifest.npz"))
    subs = sorted(set(man["subject"].tolist()))
    print(f"\n[manifest] {n} frames, {len(subs)} identities")
    counts = {s: int((man["subject"] == s).sum()) for s in subs}
    maj = max(counts.values()) / n
    for s in sorted(counts, key=lambda k: -counts[k]):
        print(f"   {s:<10s} {counts[s]:5d}  ({100 * counts[s] / n:5.2f} %)")
    print(f"[balance] majority-class rate {100 * maj:.1f} % -- report against THAT, "
          f"not 1/K = {100 / len(subs):.1f} %")

    # ---- shards ------------------------------------------------------------
    S = args.size
    dep = np.zeros((n, S, S), np.uint16)
    rgb = np.zeros((n, S, S, 3), np.uint8)
    for i, (d, lab, idx) in enumerate(rows):
        dp = os.path.join(args.root, d, f"{lab}_{idx}_depth.png")
        cp = os.path.join(args.root, d, f"{lab}_{idx}_rgb.png")
        # 16-bit depth needs dtype=tf.uint16 -- omitting it is the 2023 bug that
        # silently made every depth image 8-bit.
        a = np.asarray(tf.io.decode_png(tf.io.read_file(dp), dtype=tf.uint16))
        b = np.asarray(tf.io.decode_png(tf.io.read_file(cp), channels=3))
        a = a[..., 0] if a.ndim == 3 else a
        if a.shape[:2] != (S, S) or b.shape[:2] != (S, S):
            raise SystemExit(f"{dp}: expected {S}x{S}, got depth {a.shape} rgb {b.shape}")
        dep[i], rgb[i] = a, b
        if (i + 1) % 1000 == 0 or i == n - 1:
            print(f"[shard] {i + 1}/{n}")
    np.save(os.path.join(args.out, "training_depth.npy"), dep)
    np.save(os.path.join(args.out, "training_rgb.npy"), rgb)
    np.savez(os.path.join(args.out, "training_index.npz"), manifest_row=np.arange(n))

    valid = dep > 0
    meta = dict(
        shards_ok=True, dataset="inhouse", n_frames=n, n_identities=len(subs),
        majority_class_rate=maj, size=S, has_mask=False, has_cues=False,
        per_identity=counts,
        depth_median_mm=float(np.median(dep[valid])) if valid.any() else 0.0,
        valid_frac_median=float(np.median(valid.mean(axis=(1, 2)))),
        note=("No userMap and no usable slab foreground -- run with "
              "--eligibility all --condition full. R0/R1 only: frame provenance "
              "was destroyed by kinect2png.py's per-.mkv idx reset."))
    with open(os.path.join(args.out, "prep_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"\n[done] {args.out}  depth median {meta['depth_median_mm']:.0f} mm  "
          f"valid fraction {100 * meta['valid_frac_median']:.1f} %")


if __name__ == "__main__":
    main()
