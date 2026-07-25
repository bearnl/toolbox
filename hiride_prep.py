"""One-pass BIWI preparation for the HI-RIDE re-run.

Run this ONCE inside a job that has extracted the archives to $SLURM_TMPDIR.
It writes a handful of large files back to $SCRATCH, which is what makes every
later job cheap:

  * inode cost: ~12 files instead of ~196,000 loose frames
  * I/O cost:   the 2023 runs were input-pipeline-bound by ~2 orders of
                magnitude (0.67 s/step for a 1.35 GMAC model on 2xP100) because
                every frame went through tf.py_function + a per-file read.
                Shards are read as memmapped arrays instead.

Outputs (in --out):
    manifest.npz / manifest.csv     one row per frame: subject, session, seq, frame, ts
    cues.npz                        13 scalar cues per frame + background plates
    <seq>_depth.npy  (N,S,S) uint16 millimetres, area-resized
    <seq>_rgb.npy    (N,S,S,3) uint8, area-resized (proper box filter -- the 2023
                                code gave depth INTER_AREA but RGB an aliased
                                bilinear, which quietly favoured depth)
    <seq>_mask.npy   (N,S,S) uint8 binary person mask from the shipped userMap
                                (userMap > 0 -- values are OpenNI USER INDICES and
                                run 0..6, so '== 1' silently deletes part of a
                                recording)
    <seq>_index.npz  row -> manifest index

Usage inside a job:
    python hiride_prep.py --data $SLURM_TMPDIR/biwi --out $SCRATCH/hiride2/prep --size 256
"""
import os
import sys
import json
import time
import argparse
import numpy as np

from hiride_pgm import read_pgm
from hiride_data import build_manifest, save_manifest, interframe_stats

CUES = [
    "n_person_px", "bbox_h", "bbox_w", "cent_y", "cent_x",
    "p_med", "p_p1", "p_std", "p_range", "bg_med", "valid_frac",
    "top_touch", "bot_touch",
]
SEQ_TAG = {"Training": "training", "Testing/Still": "testing_still",
           "Testing/Walking": "testing_walking"}


def frame_cues(depth, user):
    """13 scalars per frame, computed from depth + the shipped person mask.

    These are the 'trivial-cue floor': whatever a CNN reports has to be read
    against what this gets, because if simple geometry already saturates the
    task then the CNN result is not evidence about body shape.
    """
    depth = depth.astype(np.int32)
    person = user > 0
    valid = depth > 0
    bg = valid & ~person
    n = int(person.sum())
    bg_med = float(np.median(depth[bg])) if bg.any() else 0.0
    if n < 200:
        return [float(n), 0., 0., 0., 0., 0., 0., 0., 0., bg_med,
                float(valid.mean()), 0., 0.]
    ys, xs = np.nonzero(person)
    pd = depth[person & valid]
    if pd.size == 0:
        pd = np.array([0])
    return [
        float(n),
        float(ys.max() - ys.min() + 1), float(xs.max() - xs.min() + 1),
        float(ys.mean()), float(xs.mean()),
        float(np.median(pd)), float(np.percentile(pd, 1)), float(pd.std()),
        float(np.percentile(pd, 99) - np.percentile(pd, 1)),
        bg_med, float(valid.mean()),
        float(person[0, :].any()), float(person[-1, :].any()),
    ]


class _StreamWriter:
    """Append-only .npy writer with bounded memory.

    Replaces np.lib.format.open_memmap. A 15 GB memmap on Lustre keeps its
    pages resident and counts against the cgroup limit, which is how prep
    reached MaxRSS 56.5 GB and was killed. Here frames are buffered and written
    sequentially, so peak memory is one buffer regardless of corpus size, and
    the I/O pattern is sequential -- which is what Lustre wants anyway.

    Frames MUST be appended in shard order; `expect` asserts that.
    """

    def __init__(self, path, dtype, shape, buffer_frames=256):
        self.path, self.dtype, self.shape = path, np.dtype(dtype), shape
        self.frame_shape = shape[1:]
        self.n_written = 0
        self.buf, self.buffer_frames = [], buffer_frames
        self.fh = open(path, "wb")
        np.lib.format.write_array_header_2_0(
            self.fh, {"descr": np.lib.format.dtype_to_descr(self.dtype),
                      "fortran_order": False, "shape": tuple(shape)})

    def append(self, frame, expect):
        if expect != self.n_written + len(self.buf):
            raise RuntimeError(
                f"{os.path.basename(self.path)}: out-of-order write "
                f"(got position {expect}, expected "
                f"{self.n_written + len(self.buf)}). Frames must be visited in "
                f"shard order.")
        self.buf.append(np.ascontiguousarray(frame, dtype=self.dtype))
        if len(self.buf) >= self.buffer_frames:
            self.flush()

    def flush(self):
        if self.buf:
            self.fh.write(np.stack(self.buf).tobytes())
            self.n_written += len(self.buf)
            self.buf = []
        self.fh.flush()

    def close(self):
        self.flush()
        if self.n_written != self.shape[0]:
            raise RuntimeError(
                f"{os.path.basename(self.path)}: wrote {self.n_written} frames, "
                f"header declares {self.shape[0]}. The file is truncated.")
        self.fh.close()


def _rss_gb():
    try:
        with open("/proc/self/statm") as fh:
            return int(fh.read().split()[1]) * os.sysconf("SC_PAGE_SIZE") / 1e9
    except Exception:
        return float("nan")


def _resizer(size):
    """Return resize fns.

    depth -> AREA  (box filter; averaging is defensible on a range map at this
                    scale and matches what the 2023 code did via INTER_AREA)
    rgb   -> AREA  (the fix: 2023 used bilinear with antialias=False, which
                    aliases badly at 640x480 -> 256x256)
    mask  -> NEAREST (never interpolate a label map)

    Implemented in pure numpy. The previous version called three eager
    tf.image.resize ops per frame (~120k calls over the corpus); together with
    resident mmap pages that drove MaxRSS to 56.5 GB and got prep OOM-killed.
    TensorFlow is now used only to decode JPEG, which numpy cannot do.
    """
    def _edges(src, dst):
        return np.linspace(0, src, dst + 1).astype(np.int64)

    def area(arr):
        """True box/area average for arbitrary (non-integer) scale factors."""
        a = arr.astype(np.float64)
        if a.ndim == 2:
            a = a[..., None]
        h, w = a.shape[:2]
        ry, rx = _edges(h, size), _edges(w, size)
        # rows first, then columns; reduceat gives sums, divide by bin widths
        rows = np.add.reduceat(a, ry[:-1], axis=0) / np.diff(ry)[:, None, None]
        out = np.add.reduceat(rows, rx[:-1], axis=1) / np.diff(rx)[None, :, None]
        return out

    def nearest(arr):
        h, w = arr.shape[:2]
        iy = np.minimum((np.arange(size) * h) // size, h - 1)
        ix = np.minimum((np.arange(size) * w) // size, w - 1)
        return arr[iy][:, ix]

    def _cast(out, dtype):
        if out.shape[-1] == 1:
            out = out[..., 0]
        return np.clip(np.rint(out), 0, np.iinfo(dtype).max).astype(dtype)

    return (lambda a: _cast(area(a), np.uint16),
            lambda a: _cast(area(a), np.uint8),
            lambda a: (nearest(a) > 0).astype(np.uint8))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="dir containing Training/ and/or Testing/")
    ap.add_argument("--out", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--stride", type=int, default=1, help="frame subsampling (1 = all)")
    ap.add_argument("--no-shards", action="store_true", help="cues only, skip image shards")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()

    man = build_manifest(args.data)
    if args.stride > 1:
        sel = np.arange(0, len(man["frame"]), args.stride)
        for k in man:
            if k != "root":
                man[k] = man[k][sel]
        print(f"[prep] stride={args.stride} -> {len(man['frame'])} frames")
    save_manifest(man, os.path.join(args.out, "manifest.npz"))

    n = len(man["frame"])
    cues = np.zeros((n, len(CUES)), dtype=np.float64)
    plates, plate_acc = {}, {}
    shards = {}
    rs_depth, rs_rgb, rs_mask = _resizer(args.size)   # pure numpy; cheap even for cues-only
    if not args.no_shards:
        import tensorflow as tf
        for seq, tag in SEQ_TAG.items():
            m = man["seq"] == seq
            if not m.any():
                continue
            cnt = int(m.sum())
            shards[seq] = {
                "rows": np.where(m)[0],
                "depth": _StreamWriter(os.path.join(args.out, f"{tag}_depth.npy"),
                                       np.uint16, (cnt, args.size, args.size)),
                "rgb": _StreamWriter(os.path.join(args.out, f"{tag}_rgb.npy"),
                                     np.uint8, (cnt, args.size, args.size, 3)),
                "mask": _StreamWriter(os.path.join(args.out, f"{tag}_mask.npy"),
                                      np.uint8, (cnt, args.size, args.size)),
                "pos": {int(r): i for i, r in enumerate(np.where(m)[0])},
            }
            np.savez(os.path.join(args.out, f"{tag}_index.npz"),
                     manifest_row=np.where(m)[0])
            print(f"[prep] shard {tag}: {cnt} frames -> "
                  f"{cnt * args.size * args.size * (2 + 3 + 1) / 1e9:.1f} GB")

    _tf = None
    if shards:
        import tensorflow as _tf                    # JPEG decode only
        _tf.config.set_visible_devices([], "GPU")

    # _StreamWriter requires ordered appends, so visit one sequence at a time in
    # shard order rather than walking the manifest front to back.
    if shards:
        order = np.concatenate([shards[s]["rows"] for s in SEQ_TAG if s in shards])
    else:
        order = np.arange(n)

    root = man["root"]
    for c, row in enumerate(order):
        i = int(row)
        depth, _ = read_pgm(os.path.join(root, man["depth"][i]))
        user, _ = read_pgm(os.path.join(root, man["user"][i]))
        cues[i] = frame_cues(depth, user)

        g = man["group"][i]
        if plate_acc.get(g, 0) < 15:                      # background plate sample
            bg = np.where((depth > 0) & (user == 0), depth, 0).astype(np.uint16)
            # Store the plate at SHARD resolution, not 480x640: at full res this
            # dict alone held ~1 GB (106 recordings x 15 frames x 614 kB).
            plates.setdefault(g, []).append(rs_depth(bg))
            plate_acc[g] = plate_acc.get(g, 0) + 1

        if shards:
            sh = shards[man["seq"][i]]
            j = sh["pos"][i]
            sh["depth"].append(rs_depth(depth), expect=j)
            sh["mask"].append(rs_mask(user), expect=j)
            rgb_path = os.path.join(root, man["rgb"][i])
            try:
                rgb = _tf.io.decode_jpeg(_tf.io.read_file(rgb_path), channels=3).numpy()
            except Exception as exc:
                # build_manifest already drops zero-byte files; this catches a
                # file that is non-empty but truncated/corrupt, and names it
                # instead of dying with an anonymous DecodeJpeg error 20k
                # frames into a 30-minute job.
                raise RuntimeError(
                    f"failed to decode {rgb_path} (manifest row {i}, "
                    f"{os.path.getsize(rgb_path)} bytes): {exc}") from None
            sh["rgb"].append(rs_rgb(rgb), expect=j)

        if c % 2000 == 0:
            el = time.time() - t0
            print(f"[prep] {c}/{n}  {el:.0f}s  eta {el / max(c, 1) * (n - c):.0f}s"
                  f"  rss={_rss_gb():.1f}GB", flush=True)

    plate_keys = sorted(plates)
    plate_arr = []
    for g in plate_keys:
        st = np.stack(plates[g]).astype(np.float32)
        st[st == 0] = np.nan
        with np.errstate(all="ignore"):
            plate_arr.append(np.nanmedian(st, axis=0))
    np.savez_compressed(os.path.join(args.out, "cues.npz"),
                        cues=cues, feats=np.array(CUES),
                        plates=np.nan_to_num(np.stack(plate_arr), nan=0.0),
                        plate_groups=np.array(plate_keys))

    shards_ok = None
    if shards:
        for sh in shards.values():
            for k in ("depth", "rgb", "mask"):
                sh[k].close()                 # raises if the file is truncated

        # VERIFY, then declare. Prep previously wrote prep_meta.json even when
        # the frame loop had died partway, so downstream jobs consumed sparse
        # all-zero shards and only failed much later (or worse, trained on
        # them). Re-open each file from disk and sample it.
        bad = []
        for seq, sh in shards.items():
            tag = SEQ_TAG[seq]
            d = np.load(os.path.join(args.out, f"{tag}_depth.npy"), mmap_mode="r")
            cnt = d.shape[0]
            probe = sorted({0, cnt // 4, cnt // 2, (3 * cnt) // 4, cnt - 1})
            empty = [p for p in probe if not (np.asarray(d[p]) > 0).any()]
            if empty:
                bad.append(f"{seq}: {len(empty)}/{len(probe)} sampled frames empty "
                           f"(rows {empty})")
            del d
        shards_ok = not bad
        if bad:
            for b in bad:
                print(f"[prep] SHARD VERIFICATION FAILED -- {b}", file=sys.stderr)
            print("[prep] shards are incomplete; prep_meta.json will record "
                  "shards_ok=false and downstream jobs will refuse to run.",
                  file=sys.stderr)

    meta = {
        "shards_ok": shards_ok,
        "n_frames": int(n), "size": args.size, "stride": args.stride,
        "interframe": interframe_stats(man),
        "sequences": {s: int((man["seq"] == s).sum()) for s in sorted(set(man["seq"].tolist()))},
        "subjects": {s: len(set(man["subject"][man["seq"] == s].tolist()))
                     for s in sorted(set(man["seq"].tolist()))},
        "elapsed_s": round(time.time() - t0, 1),
    }
    with open(os.path.join(args.out, "prep_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    print("[prep] done", json.dumps(meta, indent=1))
    if shards_ok is False:
        return 1                                   # non-zero exit: the job FAILS


if __name__ == "__main__":
    sys.exit(main())
