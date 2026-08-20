#!/usr/bin/env python3
"""Audit the runs directory against the canonical filename for each run.

WHY THIS EXISTS. `results_<tag>.json` is the only thing that keeps two training
cells apart on disk, and for most of this campaign `tag` did not encode
`--erode`. Every `--erode N` run therefore landed on the filename of the plain
run with the same (policy, modality, arch, condition, seed) and overwrote it.
Slurm reported COMPLETED for all of them; the damage surfaced only as cells
holding fewer seeds than the wave had asked for.

The overwritten files are not recoverable -- but the SURVIVING files carry
correct metadata inside, so a file's true identity is whatever `run_tag` says
about its own contents. This tool renames each file to that name, which both
puts the erode runs under honest filenames and frees the plain names so the
lost runs can be regenerated without tripping the new pre-write guard.

    python hiride_retag.py --runs $SCRATCH/hiride2/runs           # report only
    python hiride_retag.py --runs $SCRATCH/hiride2/runs --apply   # rename

Read the report before applying: every rename names a cell whose own run was
destroyed, so the report doubles as the list of what needs re-running.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # never grab a GPU to read JSON
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hiride_train import run_tag, CELL_FIELDS       # noqa: E402  single definition


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--apply", action="store_true",
                    help="perform the renames (default: report only)")
    args = ap.parse_args()

    files = sorted(f for f in os.listdir(args.runs)
                   if f.startswith("results_") and f.endswith(".json"))
    moves, unreadable = [], []
    for f in files:
        try:
            m = json.load(open(os.path.join(args.runs, f)))
        except (ValueError, OSError) as e:
            unreadable.append((f, str(e)))
            continue
        have = f[len("results_"):-len(".json")]
        try:
            want = run_tag(m)
        except KeyError as e:
            unreadable.append((f, f"missing field {e}"))
            continue
        if want != have:
            moves.append((have, want, m))

    print(f"{len(files)} run files in {args.runs}")
    if unreadable:
        print(f"\n{len(unreadable)} unreadable:")
        for f, why in unreadable:
            print(f"  {f}: {why}")
    if not moves:
        print("\nEvery file already sits under the name its own metadata implies. "
              "Nothing to do.")
        return

    # A file living under the wrong name means the cell that OWNS that name was
    # overwritten -- that run no longer exists anywhere.
    print(f"\n{len(moves)} file(s) are misnamed. Each one is occupying the "
          f"filename of a different cell, whose run was destroyed:\n")
    lost = []
    for have, want, m in moves:
        axes = ", ".join(f"{k}={m.get(k)!r}" for k in CELL_FIELDS
                         if k in ("erode", "guard", "ref_eligibility"))
        print(f"  {have}")
        print(f"    is really -> {want}   ({axes})")
        lost.append(have)
    collide = [w for _, w, _ in moves if os.path.exists(
        os.path.join(args.runs, f"results_{w}.json"))]
    if collide:
        print(f"\nrefusing to rename: {len(collide)} target name(s) already exist "
              f"-- {collide[:4]}")
        return
    print(f"\nDESTROYED (re-run these): {len(lost)} cell(s) whose names were taken:")
    for name in lost:
        print(f"  {name}")

    if not args.apply:
        print("\n(report only -- pass --apply to perform the renames)")
        return
    n = 0
    for have, want, _ in moves:
        for pre, ext in (("results_", ".json"), ("cm_", ".npz")):
            src = os.path.join(args.runs, f"{pre}{have}{ext}")
            if os.path.exists(src):
                os.rename(src, os.path.join(args.runs, f"{pre}{want}{ext}"))
                n += 1
    print(f"\nrenamed {n} file(s). Re-run the destroyed cells listed above; the "
          f"pre-write guard in hiride_train.py now makes this failure loud.")


if __name__ == "__main__":
    main()
