"""Consolidate every results_*.json into the paper's tables.

    python hiride_report.py --runs $SCRATCH/hiride2/runs --floor $SCRATCH/hiride2/results
    python hiride_report.py --runs ... --floor ... --latex > tables.tex

hiride_collate.py prints one flat row per cell, which is the right shape for
watching a wave land and the wrong shape for writing a paper. This groups the
same data the way the argument runs:

  A. the protocol ladder      -- one model, five splits, both modalities: the
                                 89.4 -> 5.4 spine, with the trivial-cue floor
                                 and the permuted-label null beside every rung
  B. the mechanism suite      -- what each single controlled edit does, per rung
  C. the Z-precision axis     -- accuracy vs depth quantisation
  D. the standard protocol    -- the field's BIWI rung (every Training recording,
                                 50 classes -> Walking / Still of the 28), single-
                                 and multi-shot, beside the ladder's own R4

Cells are identified through hiride_keys, never by the raw condition string.
The first version of this file keyed on (policy, modality, arch, condition,
bits) alone, so every recipe layered on a condition -- the stripe head,
augmentation, tracklet fusion, the flatten head -- was averaged into the row of
the plain gap-head cell: the 2026-08-31 tables.tex reported R4 depth
scale_removed as 18.12 +- 7.45 where the gap-head cell is 13.62 over 5 seeds
(stats_final.json; HIRIDE_HANDOFF 14.6 quotes the latter). Tables A-C now take
the default recipe only -- arch_key equal to the bare architecture, cond_key
equal to the bare condition -- and table D names the recipe in every row.

Every cell carries chance, the majority-class rate and n_seeds, because "usable"
here is reported as a multiple of chance, never as a benchmark number.
"""
import os
import re
import glob
import json
import argparse
from collections import defaultdict

import numpy as np

from hiride_keys import arch_key, cond_key

LADDER = ["R0_frame_random", "R1_block", "R3_cross_recording", "R4_cross_session"]
STANDARD = ["R4_standard_walking", "R4_standard_still"]
SHORT = {"R0_frame_random": "R0 frame-random", "R1_block": "R1 block g150",
         "R3_cross_recording": "R3 cross-recording", "R4_cross_session": "R4 cross-session",
         "R4_standard_walking": "standard: Training(50) -> Walking",
         "R4_standard_still": "standard: Training(50) -> Still"}
ARCH_LABEL = {"alexnet": "alexnet", "convnext_tiny": "cnxt-imagenet",
              "convnext_tiny/scratch": "cnxt-scratch"}
COND_ORDER = ["full", "person", "person_centred", "scale_removed", "bg_hole",
              "bg_plate", "silhouette", "sil_scaled", "interior_only"]
COND_LABEL = {"full": "full frame", "person": "person only",
              "person_centred": "person, re-centred", "scale_removed": "person, size+position removed",
              "bg_hole": "person removed (hole kept)", "bg_plate": "person removed (plate, no hole)",
              "silhouette": "silhouette", "sil_scaled": "silhouette, size+position removed",
              "interior_only": "interior only"}


def load(runs):
    """Group results into cells keyed (policy, guard, modality, arch_key, cond_key, permuted).

    arch_key and cond_key come from hiride_keys, so a stripe-head or augmented
    run can never be averaged with the gap-head run of the same condition."""
    cells = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(runs, "results_*.json"))):
        try:
            with open(f) as fh:
                r = json.load(fh)
        except Exception:
            continue
        key = (r["policy"], r.get("guard"), r["modality"], arch_key(r), cond_key(r),
               bool(r.get("permuted", False)))
        cells[key].append(r)
    return cells


def cell(cells, policy, modality, arch, cond, permuted=False, guard=None):
    """The seeds of one cell. R1 rows are the guard-150 reference unless a guard is given."""
    if guard is None and policy.startswith("R1"):
        guard = 150
    return cells.get((policy, guard, modality, arch, cond, permuted))


def agg(rs):
    a = np.array([x["frame_acc"] for x in rs]) * 100
    s = np.array([x["per_subject_acc"] for x in rs]) * 100
    tl = [x["tracklet_acc"] for x in rs if x.get("tracklet_acc") is not None]
    tn = [x["tracklet_n"] for x in rs if x.get("tracklet_n") is not None]
    return dict(n=len(rs), mean=a.mean(), sd=a.std(ddof=1) if len(a) > 1 else 0.0,
                se=(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else float("nan"),
                per_subj=s.mean(), chance=rs[0]["chance"] * 100,
                majority=(rs[0].get("majority_class_rate") or 0) * 100,
                macro_f1=float(np.mean([x["macro_f1"] for x in rs])),
                tracklet=(100 * float(np.mean(tl))) if tl else None,
                tracklet_n=int(round(float(np.median(tn)))) if tn else None)


def load_floor(d):
    out = {}
    if not d:
        return out
    p = os.path.join(d, "floor_results.json")
    if not os.path.exists(p):
        return out
    with open(p) as fh:
        blob = json.load(fh)
    for r in (blob.get("results", []) if isinstance(blob, dict) else blob):
        if isinstance(r, dict) and r.get("model") == "rf":
            out[re.sub(r"_guard\d+$", "_guard150", r["policy"])] = r["frame_acc"] * 100
            out[r["policy"]] = r["frame_acc"] * 100
    return out


def floor_for(floor, policy, guard=150):
    return floor.get(f"{policy}_guard{guard}", floor.get(policy))


def fmt(v, nd=2):
    return "--" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}f}"


def emit(rows, headers, title, note, latex):
    if not latex:
        w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) if rows else len(str(h))
             for i, h in enumerate(headers)]
        print(f"\n### {title}\n")
        print("| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)) + " |")
        print("|" + "|".join("-" * (w[i] + 2) for i in range(len(headers))) + "|")
        for r in rows:
            print("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
        if note:
            print(f"\n{note}")
        return
    print("\n\\begin{table}[t]\n\\centering\n\\caption{" + title + "}")
    print("\\begin{tabular}{l" + "r" * (len(headers) - 1) + "}\n\\toprule")
    print(" & ".join(str(h) for h in headers) + " \\\\\n\\midrule")
    for r in rows:
        print(" & ".join(str(c) for c in r) + " \\\\")
    print("\\bottomrule\n\\end{tabular}")
    if note:
        print("\\\\[2pt]\\footnotesize " + note.replace("%", "\\%"))
    print("\\end{table}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--floor", default=None)
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--arch", default="alexnet",
                    help="arch_key (hiride_keys) for tables B and C; the default is the "
                         "ladder's gap head, so a recipe such as alexnet/stripe/aug8/tf10 "
                         "must be asked for explicitly")
    args = ap.parse_args()
    cells, floor = load(args.runs), load_floor(args.floor)
    if not cells:
        print(f"no results in {args.runs}")
        return
    pct = "\\%" if args.latex else "%"

    rows = []
    for pol in LADDER:
        for mod in ("depth", "rgb"):
            for arch in ("alexnet", "convnext_tiny"):
                rs = cell(cells, pol, mod, arch, "full")
                if not rs:
                    continue
                a = agg(rs)
                null = cell(cells, pol, "depth", "alexnet", "full", permuted=True)
                fl = floor_for(floor, pol)
                rows.append([SHORT[pol], mod, ARCH_LABEL[arch],
                             a["n"], f"{fmt(a['mean'])} ± {fmt(a['sd'])}", fmt(a["per_subj"]),
                             fmt(a["macro_f1"], 3), fmt(a["chance"]), fmt(a["majority"]),
                             fmt(fl) if fl else "--",
                             fmt(agg(null)["mean"]) if null else "--",
                             fmt(a["mean"] / a["chance"], 1) + "x"])
    emit(rows, ["split", "mod", "arch", "n", f"accuracy ({pct})", "per-subj", "macro-F1",
                "chance", "majority", "floor", "null", "x chance"],
         "The protocol ladder: one model, one dataset, only the split changes",
         "floor = 13 hand-computed scalars + random forest on the SAME split, using the "
         "dataset's shipped segmentation (a floor on geometry given perfect masks, not a "
         "like-for-like CNN baseline). null = the same CNN with labels permuted within the "
         "split. Read every rung against the floor and the null, never against 1/K.",
         args.latex)

    for mod in ("depth", "rgb"):
        rows, seeds = [], set()
        for cond in COND_ORDER:
            row, any_cell = [COND_LABEL[cond]], False
            for pol in LADDER:
                rs = cell(cells, pol, mod, args.arch, cond)
                if rs:
                    a = agg(rs)
                    row.append(f"{fmt(a['mean'])} ± {fmt(a['sd'])}")
                    seeds.add(a["n"])
                    any_cell = True
                else:
                    row.append("--")
            if any_cell:
                rows.append(row)
        if not rows:
            continue
        base = cell(cells, "R4_cross_session", mod, args.arch, "full")
        n_label = "/".join(str(n) for n in sorted(seeds))
        emit(rows, ["condition"] + [SHORT[p] for p in LADDER],
             f"Mechanism suite, {mod} ({args.arch}, {n_label} seeds): a single controlled edit per row",
             f"chance is 2.00{pct} at R0/R1 (50 subjects) and 3.57{pct} at R3/R4 (28). "
             + (f"The R4 full-frame reference is {fmt(agg(base)['mean'])}{pct}. " if base else "")
             + "'plate, no hole' replaces the person with that recording's own background, so "
               "no person-shaped boundary survives -- it is the exact complement of 'person only'. "
               "Every cell is the default recipe of the named architecture (hiride_keys.arch_key); "
               "recipes such as the stripe head are separate cells and never averaged in.",
             args.latex)

    bits_pols = [p for p in LADDER
                 if any(k[0] == p and k[4].startswith("scale_removed@") for k in cells)]
    if bits_pols:
        rows = []
        for b in (16, 8, 4, 3, 2, 1):
            row = [f"{b} bit" + ("s" if b > 1 else "") + (" (as captured)" if b == 16 else ""),
                   str(2 ** b) if b < 16 else "65536"]
            hit = False
            for pol in bits_pols:
                cond = "scale_removed" if b == 16 else f"scale_removed@{b}b"
                rs = cell(cells, pol, "depth", args.arch, cond)
                if rs:
                    a = agg(rs)
                    row.append(f"{fmt(a['mean'])} ± {fmt(a['sd'])}")
                    hit = True
                else:
                    row.append("--")
            if hit:
                rows.append(row)
        extra = []
        for pol in bits_pols:
            rs = cell(cells, pol, "depth", args.arch, "sil_scaled")
            extra.append(f"{fmt(agg(rs)['mean'])}" if rs else "--")
        if any(e != "--" for e in extra):
            rows.append(["binary silhouette (shipped mask)", "2"] + extra)
        emit(rows, ["depth precision", "levels"] + [SHORT[p] for p in bits_pols],
             "Z-precision axis: accuracy vs depth quantisation at fixed global range",
             "Quantisation is uniform over a FIXED 0--6000 mm range, so a level is an absolute "
             "distance, not a per-frame stretch. All rows use the size+position-removed person, "
             "the condition that carries the cross-session signal, under the default recipe. The "
             "last row is the shipped binary mask under the same geometry -- an outline with no "
             "depth at all.",
             args.latex)

    std_cells = sorted({(k[2], k[3], k[4]) for k in cells if k[0] in STANDARD and not k[5]})
    if std_cells:
        rows = []
        for mod, arch, cond in std_cells:
            for pol in STANDARD + ["R4_cross_session"]:
                rs = cell(cells, pol, mod, arch, cond)
                if not rs:
                    continue
                a = agg(rs)
                fl = floor_for(floor, pol)
                rows.append([SHORT[pol], mod, arch, cond, a["n"],
                             f"{fmt(a['mean'])} ± {fmt(a['sd'])}",
                             fmt(a["tracklet"]) if a["tracklet"] is not None else "--",
                             a["tracklet_n"] if a["tracklet_n"] is not None else "--",
                             fmt(a["chance"]), fmt(fl) if fl else "--",
                             fmt(a["mean"] / a["chance"], 1) + "x"])
        emit(rows, ["split", "mod", "recipe", "input", "n", f"single-shot ({pct})",
                    f"multi-shot ({pct})", "decisions", "chance", "floor", "x chance"],
             "The field's BIWI protocol beside the ladder's R4: same probe frames, "
             "different training pool",
             "single-shot = one decision per test frame; multi-shot = one decision per test "
             "recording (mean posterior over all its frames; decisions = recordings, 28 on "
             "BIWI). The standard rows train on EVERY Training recording -- 50 classes, chance "
             "2.00% -- as Haque et al. (2016) and Karianakis et al. (2018) did; the R4 row trains "
             "on the 28 re-recorded subjects only (chance 3.57%). Walking test frames are "
             "byte-identical between the standard and R4 rows. recipe is hiride_keys.arch_key: "
             "'alexnet' is the ladder's gap head, '/stripe/aug8/tf10' the best cross-session "
             "recipe. floor = 13 hand-computed scalars + random forest on the same split.",
             args.latex)


if __name__ == "__main__":
    main()
