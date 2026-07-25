# HI-RIDE (paper 2) re-run — operating instructions

Everything here is new code for the paper-2 re-run. It does not touch
`siamese.py`, `anthro_probe.py` or anything else belonging to paper 3.

## Why the re-run exists

An audit of the 2023 experiments found the reported numbers do not measure what
the paper says they measure:

| finding | evidence |
|---|---|
| No held-out set. Frames were split at random inside continuous video. | BIWI `Training/022` is 528 consecutively numbered frames of one person, one outfit, one room. Measured: under that policy the **minimum gap from a test frame to the nearest training frame of the same recording is 1 frame** (median 1 across all 50 recordings). |
| Classes came from directory names, which merged the two sessions. | BIWI names Testing folders `000`,`001`,… exactly like Training. The session marker is in the *filename* (`001a_000153-…`). Union = 50 classes, so outfit A and outfit B of one person became ONE class. |
| `mensa` / in-house validation sets were re-drawn every epoch. | `load_mensa` (`depth_alexnet.py:184`) and `load_inhouse` (`:245`) omit `reshuffle_each_iteration=False`, which `load_biwi` (`:136-137`) passes. |
| The modality comparison was unpaired and asymmetrically preprocessed. | Two independent unseeded shuffles (`:136`,`:137`); RGB got aliased bilinear, depth got `INTER_AREA`. |
| Quoted numbers are TensorBoard-smoothed endpoints. | EMA(0.6, debiased) on the final point reproduces 7 of 9 published numbers exactly. |
| A trivial baseline already explains most of the result. | 13 hand-computed scalars + a random forest reach **68.5 % ± 0.8** on 50-way ID under the 2023 policy (chance 2 %), and **34.3 % ± 0.3** under a contiguous block. |

## Files

| file | role |
|---|---|
| `hiride_pgm.py` | numpy-only PGM reader. venv311 has **no cv2**, so nothing may depend on it. |
| `hiride_data.py` | manifest (identity parsed from the **filename**) + named split policies. Run it directly to inspect a tree. |
| `hiride_prep.py` | one pass over the extracted archives → manifest, 13 cues, background plates, and image shards. |
| `hiride_floor.py` | the trivial-cue floor across the ladder + label-permutation control + feature attribution. CPU, seconds. |
| `run_hiride_prep.slurm` | CPU job: stage `.rar` → `$SLURM_TMPDIR`, extract, run prep. |

## The split ladder

Reporting one model across progressively stricter splits *is* the answer to
"why". Identity always comes from the filename prefix, so `022` and `022b` are
one person with two sessions.

| policy | train → test | K | chance | isolates |
|---|---|---|---|---|
| `R0_frame_random` | random 80/20 inside Training | 50 | 2.00 % | nothing — the 2023 policy |
| `R1_block` (guard 0) | contiguous block per recording | 50 | 2.00 % | block baseline |
| `R1_block` (guard *g*) | same, guard band before val | 50 | 2.00 % | **temporal adjacency, cleanly** |
| `R3_cross_recording` | Testing/Still → Testing/Walking | 28 | 3.57 % | recording + motion regime |
| `R4_cross_session` | Training(A) → Testing/Walking(B) | 28 | 3.57 % | **different day + clothes — PRIMARY** |

Guaranteed by assertion, not by assumption:

* test **and** validation sets are byte-identical across every guard value —
  only the training block moves;
* `block_train_counts(..., keep=…)` matches `N_train` per recording across the
  whole sweep, so a wider guard never also means less data (verified: 9,670
  training frames at every guard on the full Training set);
* `block_train_counts` **raises** if the reference guard would starve any
  recording of training frames, rather than silently leaving it unmatched;
* `R3`/`R4` **raise** a clear error if `Testing.rar` was not staged.

## Running it

The repo checkout on Nibi is **`~/toolbox`** — `cd` there first, or every command fails
with "No such file".

```bash
cd ~/toolbox && git fetch origin && git reset --hard origin/master

# 1. Prep — CPU only. Cues-only first for a fast answer (~20-40 min).
#    This also runs the split-invariant gate on the staged tree and aborts on failure.
sbatch --account=def-czarnuch_cpu --export=ALL,NO_SHARDS=1 run_hiride_prep.slurm

# 2. Floor — seconds of CPU, fine on a login node.
python hiride_floor.py --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/results

# 3. Full prep incl. image shards (~13 GB, ~12 files, a few hours) once the floor looks right.
sbatch --account=def-czarnuch_cpu run_hiride_prep.slurm
```

**`test_hiride_splits.py` cannot be run from a login node.** It needs the extracted BIWI
tree, which only exists at `$SLURM_TMPDIR` inside a job — on a login node that variable is
unset, so `--root $SLURM_TMPDIR/biwi` resolves to `/biwi` and fails. The prep job runs it for
you. To run it by hand, do it inside the job or from an `salloc` that has staged the archives.

`hiride_floor.py` prints the ladder with chance multipliers and writes
`floor_results.json`.

## Gates — read before spending GPU time

1. **Label permutation must come back at chance.** `hiride_floor.py` runs it on
   the strictest block rung. Measured locally: 1.45 % against 2.00 % chance.
   Anything materially above chance means the split still leaks; fix it before
   submitting anything.
2. **`R4_cross_session` must be available.** It needs `Testing.rar` staged. The
   archive is complete — 15,377 frames × {rgb, depth, userMap, skel,
   groundCoeff} — so masks and RGB exist for the cross-session rung too.

## Environment notes (Nibi, measured)

* `module load StdEnv/2023 python/3.11`, venv `~/venvs/venv311`
* **TF 2.15.1**, numpy 1.26.4, sklearn 1.4.2, scipy 1.14.1 — **cv2 and pandas absent**
* static `unrar` on `$HOME/.local/rar:$HOME/bin/rar:$HOME/bin`
* GPU jobs bill `def-czarnuch_gpu`; CPU jobs `def-czarnuch_cpu`
* archives: `$SCRATCH/datasets/{Training,Testing}.rar` (5.3 G / 2.9 G)
* `/scratch` quota 1024 GiB, 1000 K inodes — prep writes ~12 files, not 196 K

## Measured facts worth keeping

* Training: 23,904 frames, 50 subjects, 0 incomplete frame groups.
* Testing: 15,377 frames, `Still` + `Walking`, folders named like Training's.
* Inter-frame timestamp delta is a constant **84 units**, so guards can be
  quoted from data rather than assumed.
* **22.3 %** of Training frames have an empty `userMap`, forming exactly **one
  contiguous mid-recording run per subject in all 50 subjects** — a pose
  regime, not random dropout. `eligible_mask()` drops them from *every*
  condition so ablations never compare different pose distributions.
* `userMap` values are OpenNI **user indices running 0–6**; binarise `> 0`,
  never `== 1`.
* Depth reaches **16,524 mm**, not the ~4,500 assumed in earlier planning.
* Background plates differ between recordings by a median of only **55 mm**
  (one room, fixed camera), so "the model read the room" is a weak hypothesis
  here — the leak is temporal adjacency.
* From the shipped masks: **85.4 %** of Training frames are full-body (top edge
  3.6 %, bottom 13.1 %, both 2.2 %). This **contradicts** the paper-3 handoff's
  "100 % clipped at both edges, 0 of 13,426 full-body". Adjudicate before either
  paper is submitted — the two numbers describe the same files.
