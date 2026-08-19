"""Why depth and RGB have different accuracy-vs-distance profiles.

    python hiride_range_profile.py --prep $SCRATCH/hiride2/prep \
        --signal $SCRATCH/hiride2/results/signal_diagnostic_pc0_prep.json \
        --out $SCRATCH/hiride2/results

The fair probe (all frames, R4, whitened NCM) reports accuracy per person-median
depth bin, and depth and RGB disagree in a specific way:

    bin (mm)          0-2000  2000-2500  2500-3000  3000-3500  3500+
    depth interior     5.97 %    10.04 %    20.18 %    23.13 %  16.04 %
    rgb  scale_rm     10.10 %    15.18 %    18.73 %    13.48 %  10.24 %

Depth RISES with distance to a peak at 3-3.5 m; RGB peaks at 2.5-3 m and then
falls off twice as fast. The z^2 quantisation argument predicts the OPPOSITE for
depth -- the sensor's depth step grows from ~4.5 mm at 1.25 m to ~40.7 mm at
3.75 m, so a sensor-limited interior must get MORE discriminable as the subject
approaches, not less. It does not. So either the interior is not what carries
depth's identity signal, or something else costs more at close range than
quantisation buys.

This script tests the competing explanation -- FRAME TRUNCATION -- from the cues
already on disk. At 2 m the Kinect's 43 deg vertical FOV spans ~1.6 m, less than
a standing adult, so near frames must clip the body; and depth's usable cue
(body extent and shape) is destroyed by clipping in a way that RGB's (face and
clothing texture) is not, because RGB gets MORE pixels on the face as the person
approaches. If that is the mechanism, edge-touch rate and bbox height must both
degrade sharply in exactly the bins where depth loses.

Reports, per range bin and per split: edge-touch rate, bbox height, person pixel
count, and the accuracy the probe measured, so the association is read off one
table rather than eyeballed across two.
"""
import os
import sys
import json
import argparse
import numpy as np

from hiride_data import load_manifest, make_split, eligible_mask

# Same edges the probe uses, so bins line up row for row.
DEFAULT_BINS = "0,2000,2500,3000,3500,9999"


def bin_index(z, edges):
    """Bin id per frame; -1 for anything outside the outermost edges."""
    b = np.digitize(z, edges[1:-1], right=False)
    b[(z < edges[0]) | (z >= edges[-1])] = -1
    return b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--signal", default=None,
                    help="signal_diagnostic_*.json, to print probe accuracy alongside")
    ap.add_argument("--policy", default="R4_cross_session")
    ap.add_argument("--guard", type=int, default=150)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bins", default=DEFAULT_BINS)
    ap.add_argument("--eligibility", default="default", choices=("default", "all"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    man = load_manifest(os.path.join(args.prep, "manifest.npz"))
    z = np.load(os.path.join(args.prep, "cues.npz"), allow_pickle=False)
    cues, feats = z["cues"], [str(f) for f in z["feats"]]
    if len(cues) != len(man["subject"]):
        sys.exit(f"cues ({len(cues)}) and manifest ({len(man['subject'])}) disagree; "
                 f"re-run hiride_prep.py against this directory.")
    keep = (np.ones(len(cues), bool) if args.eligibility == "all"
            else eligible_mask(cues, feats))

    col = {f: i for i, f in enumerate(feats)}
    p_med = cues[:, col["p_med"]]
    bbox_h = cues[:, col["bbox_h"]]
    bbox_w = cues[:, col["bbox_w"]]
    npx = cues[:, col["n_person_px"]]
    top = cues[:, col["top_touch"]] > 0
    bot = cues[:, col["bot_touch"]] > 0

    kw = dict(guard=args.guard) if args.policy.startswith("R1") else {}
    tr, va, te = make_split(man, args.policy, seed=args.seed, keep=keep, **kw)

    edges = np.array([float(x) for x in args.bins.split(",")])
    probe = {}
    if args.signal and os.path.exists(args.signal):
        with open(args.signal) as fh:
            blob = json.load(fh)
        probe = read_probe_bins(blob, args.policy, edges)

    print(f"\nRange profile -- {args.policy}, {'all' if args.eligibility == 'all' else 'eligible'} "
          f"frames, bins {args.bins}\n")
    for name, idx in (("TEST", te), ("TRAIN", tr)):
        b = bin_index(p_med[idx], edges)
        print(f"[{name}] n={len(idx)}")
        hdr = (f"  {'bin (mm)':<14s}{'n':>6s} {'top-edge':>9s} {'bot-edge':>9s} "
               f"{'either':>8s} {'bbox_h px':>10s} {'bbox_w px':>10s} {'person px':>10s} "
               f"{'subjects':>9s}")
        print(hdr)
        print("  " + "-" * (len(hdr) - 2))
        for k in range(len(edges) - 1):
            sel = idx[b == k]
            if not len(sel):
                print(f"  {int(edges[k])}-{int(edges[k+1]):<9d}{0:>6d}"); continue
            print(f"  {int(edges[k])}-{int(edges[k+1]):<9d}{len(sel):>6d} "
                  f"{100*top[sel].mean():8.1f}% {100*bot[sel].mean():8.1f}% "
                  f"{100*(top|bot)[sel].mean():7.1f}% {bbox_h[sel].mean():10.1f} "
                  f"{bbox_w[sel].mean():10.1f} {npx[sel].mean():10.0f} "
                  f"{len(set(man['subject'][sel].tolist())):>9d}")
        print()

    if probe:
        print("Probe accuracy in the same bins (whitened NCM, from --signal):\n")
        w = max(len(r) for r in probe)
        print(f"  {'representation':<{w}s}" + "".join(
            f"{int(edges[k])}-{int(edges[k+1]):>9d}" for k in range(len(edges) - 1)))
        for rep, accs in probe.items():
            print(f"  {rep:<{w}s}" + "".join(f"{100*a:>13.2f}%" if a is not None else f"{'-':>14s}"
                                             for a in accs))
        print()

    print("HOW TO READ THIS. The truncation account predicts that the near bins where depth")
    print("loses are the bins where the body is clipped: high edge-touch and small bbox_h at")
    print("0-2000 mm, both recovering by 3000-3500 mm. If instead edge-touch is flat across")
    print("bins, truncation is NOT the mechanism and depth's near-range deficit needs another")
    print("explanation -- check bbox_w (the person may be wider than the frame) and the")
    print("subject count per bin, since a bin dominated by few subjects inflates or deflates")
    print("accuracy for reasons that have nothing to do with range.")

    if args.out:
        os.makedirs(args.out, exist_ok=True)
        rec = {"policy": args.policy, "bins": edges.tolist(), "splits": {}}
        for name, idx in (("test", te), ("train", tr)):
            b = bin_index(p_med[idx], edges)
            rec["splits"][name] = [
                dict(bin=[float(edges[k]), float(edges[k + 1])], n=int((b == k).sum()),
                     top_touch=float(top[idx[b == k]].mean()) if (b == k).any() else None,
                     bot_touch=float(bot[idx[b == k]].mean()) if (b == k).any() else None,
                     bbox_h=float(bbox_h[idx[b == k]].mean()) if (b == k).any() else None,
                     bbox_w=float(bbox_w[idx[b == k]].mean()) if (b == k).any() else None,
                     person_px=float(npx[idx[b == k]].mean()) if (b == k).any() else None,
                     n_subjects=int(len(set(man["subject"][idx[b == k]].tolist())))
                     if (b == k).any() else 0)
                for k in range(len(edges) - 1)]
        p = os.path.join(args.out, "range_profile.json")
        with open(p, "w") as fh:
            json.dump(rec, fh, indent=1)
        print(f"\n[written] {p}")


def read_probe_bins(blob, policy, edges):
    """Pull per-range-bin accuracy out of a signal_diagnostic JSON.

    hiride_signal.py keys each representation as "<policy>|<label>" and stores
    {"range_bins": {"0-2000": {"n": .., "acc": ..}, ...}}. Bins with fewer than
    30 test frames are omitted there, so a missing key means "not enough data",
    not zero -- it is reported as "-" rather than silently plotted as 0.
    """
    want = [f"{int(edges[k])}-{int(edges[k + 1])}" for k in range(len(edges) - 1)]
    out = {}
    for key, entry in blob.items():
        if not (isinstance(entry, dict) and "range_bins" in entry):
            continue
        if not key.startswith(f"{policy}|"):
            continue
        rb = entry["range_bins"]
        out[entry.get("representation", key.split("|", 1)[1])] = [
            rb[b]["acc"] if b in rb else None for b in want]
    if not out:
        reps = sorted({k.split("|")[0] for k in blob if "|" in k})
        print(f"  (no range bins for policy {policy} in the signal file; "
              f"it holds: {reps or 'no representation entries'})")
    return out


if __name__ == "__main__":
    main()
