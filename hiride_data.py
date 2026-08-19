"""BIWI manifest + split-policy library for the HI-RIDE re-run (paper 2).

WHY THIS FILE EXISTS
--------------------
The 2023 experiments derived the class label from the *directory* name and split
the data by shuffling individual frames.  Both are wrong, and together they are
the whole reason the reported accuracy was ~0.99:

  * BIWI names Testing subject folders exactly like Training ones ('000', '001',
    ...), so a folder-derived label unions to 50 classes and silently merges
    outfit A (Training) with outfit B (Testing) into ONE class.  The session
    marker lives in the *filename* ('001a_000153-d_...'), never in the path.
  * Frames inside one recording are a continuous video, so a frame-random split
    puts near-duplicate neighbours on both sides of the boundary.

This module fixes both by (a) parsing identity + session from the filename and
(b) making the split an explicit, named policy.  Nothing else about the input
pipeline changes, so the ladder below is a like-for-like comparison against the
published numbers: only the split varies.

USAGE
-----
    man = build_manifest("/path/to/extracted/biwi")
    tr, va, te = make_split(man, "R4_cross_session", seed=0)
"""
import os
import re
import json
import numpy as np

__all__ = [
    "FRAME_RE", "build_manifest", "save_manifest", "load_manifest",
    "make_split", "POLICIES", "eligible_mask", "interframe_stats", "describe_split",
]

# <id><optional session letter>_<frame>-<type letter>_<timestamp>_<kind>.<ext>
FRAME_RE = re.compile(
    r"^(?P<subj>\d+)(?P<sess>[a-zA-Z]?)_(?P<frame>\d+)-(?P<slot>[a-z])_(?P<ts>\d+)_"
    r"(?P<kind>rgb|depth|userMap|skel|groundCoeff)\.(?P<ext>jpg|pgm|txt)$"
)

# Sequence directories we understand, and the capture session each belongs to.
# 'A' = Training (outfit A, day 1).  'B' = Testing (outfit B, different day).
SEQUENCES = {
    "Training": "A",
    "Testing/Still": "B",
    "Testing/Walking": "B",
}


def _rel(root, path):
    return os.path.relpath(path, root).replace("\\", "/")


def build_manifest(root, verbose=True):
    """Walk an extracted BIWI tree and return a manifest dict of parallel arrays.

    `root` is the directory that directly contains 'Training/' and/or 'Testing/'.
    One record per FRAME (not per file); a frame is kept only if depth, userMap
    and rgb are all present.
    """
    frames = {}
    for dirpath, _dirnames, filenames in os.walk(root):
        rel_dir = _rel(root, dirpath)
        seq = None
        for cand, _sess in SEQUENCES.items():
            if rel_dir == cand or rel_dir.startswith(cand + "/"):
                seq = cand
        if seq is None:
            continue
        for fn in filenames:
            m = FRAME_RE.match(fn)
            if not m:
                continue
            key = (seq, m.group("subj"), m.group("sess").lower(), int(m.group("frame")))
            rec = frames.setdefault(key, {"ts": int(m.group("ts"))})
            rec[m.group("kind")] = os.path.join(dirpath, fn)

    present = sorted(k for k, v in frames.items()
                     if "depth" in v and "userMap" in v and "rgb" in v)
    dropped = len(frames) - len(present)

    # "Present" is not the same as "readable". BIWI ships at least one zero-byte
    # _rgb.jpg, which sails through the completeness check above and then kills
    # the prep job with `InvalidArgumentError: Input is empty. [Op:DecodeJpeg]`
    # thousands of frames later. Drop such frames HERE so every consumer -- the
    # cue floor, the split library and the trainer -- sees the same corpus. A
    # frame with no colour data cannot serve the RGB arm, and substituting a
    # black image would quietly corrupt the modality comparison.
    keys, empty = [], []
    for k in present:
        bad = [kind for kind in ("depth", "userMap", "rgb")
               if os.path.getsize(frames[k][kind]) == 0]
        if bad:
            empty.append((k, bad))
        else:
            keys.append(k)

    man = {
        "root": root,
        "seq": np.array([k[0] for k in keys]),
        "subject": np.array([k[1] for k in keys]),
        "sess_letter": np.array([k[2] for k in keys]),
        "session": np.array([SEQUENCES[k[0]] for k in keys]),
        "frame": np.array([k[3] for k in keys], dtype=np.int64),
        "ts": np.array([frames[k]["ts"] for k in keys], dtype=np.int64),
        "depth": np.array([_rel(root, frames[k]["depth"]) for k in keys]),
        "user": np.array([_rel(root, frames[k]["userMap"]) for k in keys]),
        "rgb": np.array([_rel(root, frames[k]["rgb"]) for k in keys]),
    }
    # Stable ordering: sequence, subject, frame.
    order = np.lexsort((man["frame"], man["subject"], man["seq"]))
    for k in ("seq", "subject", "sess_letter", "session", "frame", "ts", "depth", "user", "rgb"):
        man[k] = man[k][order]
    man["group"] = np.array([f"{s}|{sub}" for s, sub in zip(man["seq"], man["subject"])])

    if verbose:
        print(f"[manifest] {len(keys)} complete frames "
              f"({dropped} incomplete frame groups dropped, "
              f"{len(empty)} dropped for zero-byte files)")
        for k, bad in empty[:10]:
            print(f"  zero-byte {'+'.join(bad):<8s} -> {k[0]}/{k[1]}{k[2]} frame {k[3]}")
        for seq in sorted(set(man["seq"].tolist())):
            sel = man["seq"] == seq
            print(f"  {seq:<18s} frames={sel.sum():6d}  subjects={len(set(man['subject'][sel]))}")
    return man


def save_manifest(man, path):
    np.savez_compressed(path, **{k: v for k, v in man.items() if k != "root"},
                        root=np.array(man["root"]))
    with open(os.path.splitext(path)[0] + ".csv", "w") as fh:
        fh.write("seq,subject,session,frame,ts,depth,user,rgb\n")
        for i in range(len(man["frame"])):
            fh.write(f"{man['seq'][i]},{man['subject'][i]},{man['session'][i]},"
                     f"{man['frame'][i]},{man['ts'][i]},{man['depth'][i]},"
                     f"{man['user'][i]},{man['rgb'][i]}\n")


def load_manifest(path):
    d = np.load(path, allow_pickle=False)
    man = {k: d[k] for k in d.files}
    man["root"] = str(man["root"])
    return man


def interframe_stats(man):
    """Median inter-frame timestamp delta per sequence.

    Lets a temporal guard be quoted in seconds instead of assumed. The BIWI
    timestamp field's unit is not documented, so we report the raw delta and
    let the caller decide what to claim.
    """
    out = {}
    for g in sorted(set(man["group"].tolist())):
        sel = np.where(man["group"] == g)[0]
        sel = sel[np.argsort(man["frame"][sel])]
        if len(sel) > 2:
            out[g] = float(np.median(np.diff(man["ts"][sel])))
    vals = np.array(list(out.values()))
    return {"median_delta": float(np.median(vals)),
            "p10": float(np.percentile(vals, 10)),
            "p90": float(np.percentile(vals, 90))}


def eligible_mask(cues, feats, min_person_px=2000, max_person_frac=0.25,
                  full_body=False):
    """The frame-eligibility rule, applied identically to EVERY condition.

    22.3% of BIWI Training frames carry a completely empty userMap, and those
    empty frames form exactly ONE contiguous mid-recording run per subject in
    all 50 subjects -- a walk-away / tracker-loss pose regime, not random
    dropout.  If masked conditions silently drop them and full-frame baselines
    do not, the ablation compares different data AND a different pose
    distribution.  So the filter is applied once, up front, to everything.
    """
    n = cues[:, feats.index("n_person_px")]
    frac = n / float(640 * 480)
    ok = (n >= min_person_px) & (frac <= max_person_frac)
    if full_body:
        # BIWI clips the body at the frame edge in 47.8 % of R4 TEST frames but
        # only 9.2 % of TRAIN frames (hiride_range_profile.py) -- a 5x covariate
        # shift in framing between the two sessions. That matters because
        # scale_remove normalises by the person's bounding box: when the feet
        # are outside the frame the visible box is not the body, so the same
        # image row means a different body part in train and test, and every
        # alignment-preserving head (stripe, flatten) is reading a corrupted
        # correspondence. Requiring a complete body removes the shift.
        ok &= (cues[:, feats.index("top_touch")] <= 0)
        ok &= (cues[:, feats.index("bot_touch")] <= 0)
    return ok


# ---------------------------------------------------------------------------
# Split policies
# ---------------------------------------------------------------------------

def _shared_subjects(man):
    """Subjects present in BOTH Training and Testing (the 28 with two sessions)."""
    tr = set(man["subject"][man["seq"] == "Training"].tolist())
    te = set(man["subject"][man["session"] == "B"].tolist())
    return sorted(tr & te)


def _tail_val(man, idx, val_frac, guard, rng):
    """Carve a validation set out of `idx`: the contiguous tail of each group,
    with a guard band so val is not adjacent to what remains in train."""
    tr, va = [], []
    for g in sorted(set(man["group"][idx].tolist())):
        gi = idx[man["group"][idx] == g]
        gi = gi[np.argsort(man["frame"][gi])]
        cut = int(len(gi) * (1 - val_frac))
        va.extend(gi[cut:])
        if cut:
            boundary = man["frame"][gi[cut]]
            tr.extend([j for j in gi[:cut] if boundary - man["frame"][j] > guard])
    return (np.array(sorted(tr), dtype=np.int64),
            np.array(sorted(va), dtype=np.int64))


def _block_test(man, pool, test_frac):
    """Contiguous last-`test_frac` block of each group -- the held-out test set."""
    te, rest = [], []
    for g in sorted(set(man["group"][pool].tolist())):
        gi = pool[man["group"][pool] == g]
        gi = gi[np.argsort(man["frame"][gi])]
        cut = int(len(gi) * (1 - test_frac))
        te.extend(gi[cut:])
        rest.append((g, gi[:cut], man["frame"][gi[cut]] if cut < len(gi) else None))
    return np.array(sorted(te), dtype=np.int64), rest


def block_train_counts(man, policy="R1_block", guard=0, seed=0, test_frac=0.2, keep=None):
    """Per-group training-set sizes a block policy would produce at `guard`.

    Feed the result back as `match_ntrain` to EVERY rung of a guard sweep so
    that all rungs train on the same number of frames per recording.  Without
    this, a larger guard removes adjacency AND training data at the same time,
    and the sweep cannot attribute the drop to either.  Matching only ever
    subsamples, so pass the counts from the LARGEST guard in the sweep.
    """
    spec = POLICIES[policy]
    valid = np.ones(len(man["frame"]), dtype=bool) if keep is None else np.asarray(keep)
    pool = np.where((man["seq"] == spec["seq"]) & valid)[0]
    tr, _, _ = _policy_block(man, pool, np.random.default_rng(seed),
                             guard=guard, test_frac=test_frac)
    groups = sorted(set(man["group"][pool].tolist()))
    counts = {g: int((man["group"][tr] == g).sum()) for g in groups}
    starved = [g for g, c in counts.items() if c == 0]
    if starved:
        raise ValueError(
            f"guard={guard} leaves {len(starved)}/{len(groups)} recordings with NO training "
            f"frames (e.g. {starved[:3]}). Matching against this reference would silently "
            f"drop whole classes. Pick a smaller reference guard: recordings hold "
            f"{int(np.median([ (man['group']==g).sum() for g in groups ]))} frames on median.")
    return counts


def _policy_R0(man, pool, rng, test_frac=0.2, val_frac=0.1, **kw):
    """R0 -- frame-random over continuous video. The 2023 policy. Leaky by construction."""
    idx = rng.permutation(pool)
    n_te = int(len(idx) * test_frac)
    te, restv = idx[:n_te], idx[n_te:]
    n_va = int(len(restv) * val_frac)
    return np.sort(restv[n_va:]), np.sort(restv[:n_va]), np.sort(te)


def _policy_block(man, pool, rng, guard=0, test_frac=0.2, val_frac=0.1,
                  match_ntrain=None, **kw):
    """R1/R2 -- contiguous held-out block per recording, with a temporal guard.

    Layout per recording, in frame order:

        [ ........ train ........ ][ guard ][ val ][ test ]

    TEST and VAL are byte-identical for every guard value -- the guard only
    ever removes frames from the tail of the TRAIN block.  So a drop across the
    guard sweep is attributable to temporal adjacency and nothing else.

    `match_ntrain` then subsamples train to a per-recording target so that a
    larger guard does not also mean less training data.  Matching can only
    shrink, so pass counts from the LARGEST guard in the sweep (see
    block_train_counts).
    """
    tr_all, va_all, te_all = [], [], []
    for g in sorted(set(man["group"][pool].tolist())):
        gi = pool[man["group"][pool] == g]
        gi = gi[np.argsort(man["frame"][gi])]
        n = len(gi)
        te_cut = int(n * (1 - test_frac))
        te_all.extend(gi[te_cut:])
        rest = gi[:te_cut]
        if len(rest) == 0:
            continue
        va_cut = int(len(rest) * (1 - val_frac))
        va_all.extend(rest[va_cut:])
        tr_pool = rest[:va_cut]
        if len(tr_pool) == 0:
            continue
        boundary = man["frame"][rest[va_cut]] if va_cut < len(rest) else man["frame"][gi[te_cut]]
        keep = np.array([j for j in tr_pool if boundary - man["frame"][j] > guard],
                        dtype=np.int64)
        if match_ntrain is not None:
            target = match_ntrain.get(g, len(keep))
            if len(keep) > target:
                keep = np.sort(rng.choice(keep, size=target, replace=False))
        tr_all.extend(keep.tolist())
    return (np.array(sorted(tr_all), dtype=np.int64),
            np.array(sorted(va_all), dtype=np.int64),
            np.array(sorted(te_all), dtype=np.int64))


def _policy_cross(man, rng, train_seq, test_seq, val_frac=0.15, guard=50, **kw):
    """R3/R4 -- train on one sequence, test on another, restricted to subjects
    that appear in both.  R3 changes the recording; R4 changes the day AND the
    clothing.  Test set is byte-identical between them, so the R3->R4 delta
    isolates session+clothing."""
    shared = set(_shared_subjects(man))
    if not shared:
        raise RuntimeError(
            f"no subject appears in both '{train_seq}' and '{test_seq}' -- this policy "
            "needs BOTH Training/ and Testing/ extracted (Testing.rar not staged?)")
    in_subj = np.array([s in shared for s in man["subject"]])
    tr_pool = np.where((man["seq"] == train_seq) & in_subj)[0]
    te = np.where((man["seq"] == test_seq) & in_subj)[0]
    if len(tr_pool) == 0 or len(te) == 0:
        raise RuntimeError(f"empty pool: train '{train_seq}'={len(tr_pool)} "
                           f"test '{test_seq}'={len(te)} frames")
    tr, va = _tail_val(man, tr_pool, val_frac, guard, rng)
    return tr, va, np.sort(te).astype(np.int64)


POLICIES = {
    # name                          pool                     builder
    "R0_frame_random":  dict(seq="Training", fn=_policy_R0,
                             doc="frame-random within Training (the 2023 policy)"),
    "R1_block":         dict(seq="Training", fn=_policy_block, kw=dict(guard=0),
                             doc="contiguous held-out block, no guard"),
    "R2_block_guard":   dict(seq="Training", fn=_policy_block, kw=dict(guard=50),
                             doc="contiguous block + 50-frame guard, N_train matched to R1"),
    "R3_cross_recording": dict(fn=_policy_cross,
                               kw=dict(train_seq="Testing/Still", test_seq="Testing/Walking"),
                               doc="same day, same outfit B, different recording"),
    "R4_cross_session": dict(fn=_policy_cross,
                             kw=dict(train_seq="Training", test_seq="Testing/Walking"),
                             doc="different day, different clothes -- PRIMARY"),
}


def make_split(man, policy, seed=0, keep=None, **overrides):
    """Return (train_idx, val_idx, test_idx) for a named policy.

    `keep` is an optional boolean mask (the eligibility filter) applied to the
    whole manifest before splitting, identically for every policy.
    """
    if policy not in POLICIES:
        raise KeyError(f"unknown policy {policy!r}; have {sorted(POLICIES)}")
    spec = POLICIES[policy]
    rng = np.random.default_rng(seed)
    kw = dict(spec.get("kw", {}))
    kw.update(overrides)

    valid = np.ones(len(man["frame"]), dtype=bool) if keep is None else np.asarray(keep)

    if spec["fn"] is _policy_cross:
        tr, va, te = _policy_cross(man, rng, **kw)
        keep_only = lambda a: a[valid[a]]
        return keep_only(tr), keep_only(va), keep_only(te)

    pool = np.where((man["seq"] == spec["seq"]) & valid)[0]
    return spec["fn"](man, pool, rng, **kw)


def describe_split(man, tr, va, te):
    def k(a):
        return len(set(man["subject"][a].tolist()))
    ktest = k(te)
    return {
        "n_train": int(len(tr)), "n_val": int(len(va)), "n_test": int(len(te)),
        "k_train": k(tr), "k_test": ktest,
        "chance": (1.0 / ktest) if ktest else float("nan"),
        "train_seq": sorted(set(man["seq"][tr].tolist())),
        "test_seq": sorted(set(man["seq"][te].tolist())),
        "unseen_test_subjects": sorted(set(man["subject"][te].tolist())
                                       - set(man["subject"][tr].tolist())),
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Build/inspect the BIWI manifest.")
    ap.add_argument("root", help="directory containing Training/ and/or Testing/")
    ap.add_argument("--save", default=None, help="write manifest npz+csv here")
    args = ap.parse_args()
    man = build_manifest(args.root)
    print("[interframe]", json.dumps(interframe_stats(man), indent=1))
    print("[shared subjects Training&Testing]", len(_shared_subjects(man)))
    targets = block_train_counts(man, guard=50, seed=0)   # match every rung to the widest guard
    for name in POLICIES:
        try:
            kw = {"match_ntrain": targets} if name in ("R1_block", "R2_block_guard") else {}
            tr, va, te = make_split(man, name, seed=0, **kw)
            print(f"  {name:<22s} {json.dumps(describe_split(man, tr, va, te))}")
        except Exception as exc:                       # a policy may need data we lack
            print(f"  {name:<22s} UNAVAILABLE: {exc}")
    if args.save:
        save_manifest(man, args.save)
        print("saved", args.save)
