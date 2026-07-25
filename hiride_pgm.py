"""Dependency-free PGM (P5) reader.

The Nibi venv311 has no cv2 (the `opencv` module's bindings are not visible from
inside the venv), so every BIWI file type we need is parsed here with numpy only.

BIWI RGBD-ID file types, per frame:
    <id>[<sess>]_<frame>-a_<ts>_rgb.jpg           8-bit colour
    <id>[<sess>]_<frame>-b_<ts>_depth.pgm         16-bit big-endian, millimetres
    <id>[<sess>]_<frame>-c_<ts>_userMap.pgm       8-bit OpenNI user index (0 = no user)
    <id>[<sess>]_<frame>-d_<ts>_skel.txt
    <id>[<sess>]_<frame>-e_<ts>_groundCoeff.txt
"""
import numpy as np

__all__ = ["read_pgm"]


def read_pgm(path):
    """Read a binary PGM (P5). Returns (ndarray, maxval).

    dtype is uint8 when maxval <= 255, otherwise big-endian uint16 ('>u2'),
    which is what the BIWI depth maps use (values in millimetres).
    """
    with open(path, "rb") as fh:
        data = fh.read()
    if data[:2] != b"P5":
        raise ValueError(f"not a binary PGM (P5): {path}")

    pos = 2
    fields = []
    while len(fields) < 3:
        while pos < len(data) and data[pos:pos + 1].isspace():
            pos += 1
        if data[pos:pos + 1] == b"#":                 # comment runs to end of line
            while pos < len(data) and data[pos:pos + 1] not in (b"\n", b"\r"):
                pos += 1
            continue
        start = pos
        while pos < len(data) and not data[pos:pos + 1].isspace():
            pos += 1
        fields.append(int(data[start:pos]))
    pos += 1                                          # exactly one whitespace byte follows

    width, height, maxval = fields
    dtype = np.dtype(">u2") if maxval > 255 else np.dtype("u1")
    count = width * height
    if len(data) - pos < count * dtype.itemsize:
        raise ValueError(f"truncated PGM: {path}")
    return np.frombuffer(data, dtype=dtype, count=count, offset=pos).reshape(height, width), maxval
