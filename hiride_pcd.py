"""numpy-only PCD (Point Cloud Data) reader -- venv311 has no PCL and no cv2.

Reads PCD v0.7 files in all three DATA modes (ascii, binary,
binary_compressed -- the last via a pure-python LZF decompressor, ~30 lines,
same algorithm as PCL's lzf.c). Written while scouting second corpora for the
external-validity replication; the corpora actually adopted (TVRID) or
requested (IAS-Lab RGBD-ID) both turned out to ship per-frame IMAGES, so
nothing depends on this yet -- it is kept, tested (round-trip, all three
modes), for the next candidate corpus that ships ORGANIZED clouds
(WIDTH x HEIGHT = the sensor grid), from which per-frame images follow:

    depth_mm[v, u] = z[v, u] * 1000        (invalid points are NaN -> 0)
    rgb[v, u]      = unpack(rgb_float)     (PCL packs r,g,b into one float)

Nothing here assumes which fields exist beyond x/y/z; `rgb` is optional and
some IAS-Lab clouds are XYZRGBA. Everything is returned in file order --
row-major over the organized grid.
"""
import numpy as np

_DTYPES = {("F", 4): "<f4", ("F", 8): "<f8",
           ("I", 1): "<i1", ("I", 2): "<i2", ("I", 4): "<i4",
           ("U", 1): "<u1", ("U", 2): "<u2", ("U", 4): "<u4"}


def _lzf_decompress(data, expected):
    """PCL/liblzf decompression (the format binary_compressed uses)."""
    out = bytearray()
    i, n = 0, len(data)
    while i < n:
        ctrl = data[i]; i += 1
        if ctrl < 32:                       # literal run of ctrl+1 bytes
            run = ctrl + 1
            out += data[i:i + run]
            i += run
        else:                               # back-reference
            length = ctrl >> 5
            if length == 7:
                length += data[i]; i += 1
            ref = len(out) - ((ctrl & 0x1f) << 8) - data[i] - 1
            i += 1
            for _ in range(length + 2):
                out.append(out[ref]); ref += 1
    if len(out) != expected:
        raise ValueError(f"lzf: expected {expected} bytes, got {len(out)}")
    return bytes(out)


def read_pcd(path):
    """-> (points structured array, header dict). Organized if height > 1."""
    with open(path, "rb") as fh:
        header = {}
        while True:
            line = fh.readline().decode("ascii", "replace").strip()
            if not line or line.startswith("#"):
                continue
            k, _, v = line.partition(" ")
            header[k] = v
            if k == "DATA":
                break
        fields = header["FIELDS"].split()
        sizes = [int(s) for s in header["SIZE"].split()]
        types = header["TYPE"].split()
        counts = [int(c) for c in header.get("COUNT", " ".join("1" * len(fields))).split()]
        n_pts = int(header["POINTS"])
        dtype = np.dtype([(f if c == 1 else f, _DTYPES[(t, s)], (c,) if c > 1 else ())
                          for f, s, t, c in zip(fields, sizes, types, counts)])
        mode = header["DATA"]
        if mode == "ascii":
            raw = np.loadtxt(fh, dtype=np.float64, max_rows=n_pts)
            pts = np.zeros(n_pts, dtype=dtype)
            col = 0
            for f, c in zip(fields, counts):
                v = raw[:, col:col + c]
                pts[f] = v[:, 0] if c == 1 else v
                col += c
        elif mode == "binary":
            pts = np.frombuffer(fh.read(dtype.itemsize * n_pts), dtype=dtype,
                                count=n_pts).copy()
        elif mode == "binary_compressed":
            comp_len, full_len = np.frombuffer(fh.read(8), dtype="<u4")
            blob = _lzf_decompress(fh.read(int(comp_len)), int(full_len))
            # compressed PCD stores each FIELD contiguously, not interleaved
            pts = np.zeros(n_pts, dtype=dtype)
            off = 0
            for f, s, t, c in zip(fields, sizes, types, counts):
                nbytes = s * c * n_pts
                arr = np.frombuffer(blob[off:off + nbytes],
                                    dtype=_DTYPES[(t, s)])
                pts[f] = arr if c == 1 else arr.reshape(n_pts, c)
                off += nbytes
        else:
            raise ValueError(f"unsupported DATA mode {mode!r} in {path}")
    header["_fields"] = fields
    return pts, header


def unpack_rgb(rgb_field):
    """PCL packs r,g,b (and sometimes a) into one little-endian float32."""
    as_u32 = rgb_field.astype(np.float32).view(np.uint32)
    r = (as_u32 >> 16) & 0xFF
    g = (as_u32 >> 8) & 0xFF
    b = as_u32 & 0xFF
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def pcd_to_frames(path):
    """Organized cloud -> (depth_mm uint16 HxW, rgb uint8 HxWx3 or None)."""
    pts, hdr = read_pcd(path)
    w, h = int(hdr["WIDTH"]), int(hdr["HEIGHT"])
    if h <= 1:
        raise ValueError(f"{path}: unorganized cloud ({w}x{h}); cannot form an image")
    z = np.asarray(pts["z"], dtype=np.float64).reshape(h, w)
    depth = np.where(np.isfinite(z) & (z > 0), z * 1000.0, 0.0)
    depth = np.clip(depth, 0, 65535).astype(np.uint16)
    rgb = None
    for f in ("rgb", "rgba"):
        if f in hdr["_fields"]:
            rgb = unpack_rgb(np.asarray(pts[f])).reshape(h, w, 3)
            break
    return depth, rgb


if __name__ == "__main__":
    import sys
    d, c = pcd_to_frames(sys.argv[1])
    v = d[d > 0]
    print(f"{sys.argv[1]}: depth {d.shape}, valid {100 * (d > 0).mean():.1f} %, "
          f"median {np.median(v) if v.size else 0:.0f} mm, rgb "
          f"{'yes ' + str(c.shape) if c is not None else 'no'}")
