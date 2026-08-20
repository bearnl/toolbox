# HI-RIDE paper-2 re-run — session handoff

> **Purpose.** Self-contained record so a new conversation on a different machine can
> resume with no prior context and **no local dataset**. Every number below was measured,
> not assumed; each is tagged with where it came from and whether it can be re-derived.
>
> **Read this first, then `HIRIDE_RERUN.md` for how to run the code.**

**Provenance tags used throughout**
- `[CLUSTER]` — measured on Nibi; re-derivable there.
- `[LOCAL]` — measured on the previous laptop from a copy of BIWI `Training/` at `G:\biwi`.
  **That machine is gone. These cannot be re-derived on the new laptop.** They can be
  re-derived on Nibi, and the code to do it is committed.
- `[AUDIT]` — derived from the committed 2023 artifacts (`paper/paper-depth/data/*.csv`,
  `toolbox/depth_alexnet.py`). Re-derivable anywhere both repos are checked out.
- `[UNVERIFIED]` — believed but **not** confirmed. Do not put in the paper as-is.

---

## 0. Where things stand

The 2023 paper (`paper/paper-depth/main.tex`, "HI-RIDE") reports ~0.99 accuracy for depth
person identification. An audit established that this measures the evaluation split, not the
modality (§2). Every experiment has now been re-run on Nibi under leak-free protocols.

### THE EXPERIMENTAL CAMPAIGN IS COMPLETE (2026-08-18; top-ups through 08-20)

**If you are picking this up after 2026-08-19, read §13 before quoting any number involving `interior_only` or the erode sweep** — a filename bug destroyed six runs and scrambled that sweep, and the repair is described there.

459 training cells across seven waves, all `COMPLETED`, 5 seeds each, plus the trivial-cue
floor, per-rung permutation nulls, subject-cluster bootstrap CIs and two pixel-level
diagnostics. Every element of the §4 design is measured or explicitly ruled out with a
stated reason. **Nothing computational is outstanding.** The next stage is the manuscript
and its figures — see §11.

**READ §12 FIRST if you are picking this up after 2026-08-18** — metric 3D features
and a changed CNN head beat every number in §8, and six other interventions were
refuted by measurement.

**Where the numbers live** (regenerate any table from these, do not retype numbers):
```bash
cat $SCRATCH/hiride2/results/report.md        # the three paper tables, markdown
cat $SCRATCH/hiride2/results/tables.tex       # the same, LaTeX (hiride_report.py --latex)
cat $SCRATCH/hiride2/results/stats_final.txt  # subject-cluster CIs + paired tests
```
Archived off scratch (which Alliance purges) at
`~/projects/def-czarnuch/chenzz/hiride2/` — `results/` plus a 90 MB tarball of all ~460
runs with per-frame predictions. Code: `git@github.com:bearnl/toolbox`, `master`.

### The result, in five lines

1. **The spine.** Thirteen hand-computed scalars reach **89.4 %** on 50-way ID under the
   2023 frame-random policy and **5.4 %** across a session change. Same features, same
   model, same data; only the split changes.
2. **The CNN's within-session accuracy is mostly the recording, not the person.** At R3,
   the background with the person inpainted away (`bg_plate`, 38.6 %) matches the full
   frame (34.8 %), while the person alone scores 8.7 % — on the floor (8.45 %).
3. **Across a session change both modalities collapse** to the null: depth 6.7 %, RGB
   5.6 %, null 4.3 %, majority 4.5 %. An ImageNet ConvNeXt does not rescue them (8.5 /
   10.7), so it is not a from-scratch-AlexNet artefact.
4. **What survives is the framing-normalised outline.** Removing apparent size and image
   position recovers depth to 12.5 % and RGB to 17.7 %; a normalised binary silhouette
   reaches **14.6 % [8.8, 21.0]**, beating the full metric-depth frame (+7.9 pp, CI
   [+0.16, +15.5] — the only R4 contrast whose subject-level interval excludes zero).
5. **Depth precision barely matters.** Four quantisation levels (2 bits) identify as well
   as 65,536 at R4, with the CI still clear of the majority rate.

**The deployment claim this supports.** Lowering Z-precision is not a privacy control, and
neither is masking the person out (`bg_hole` leaves a silhouette-shaped hole and scores
*above* the full frame at R3). What leaks is the outline. The honest operating point is
~15 % on a 28-person enrolled cohort — several times chance, far from usable
identification, and enough to refute "depth is anonymous because humans cannot read it".

**What must NOT be misread** (both documented in the code, restate both in the paper):
- `bg_plate` at R0/R1 scores 97–100 % and that is **structural, not a scene measurement**.
  Removing the only moving object from a fixed-camera recording makes every frame of that
  recording the same image, so a within-recording split trains and tests on near-duplicates
  whatever the guard. It is interpretable **only** at the recording-disjoint rungs R3/R4.
- The trivial-cue floor uses the dataset's **shipped ground-truth masks**, so it bounds
  "geometry given perfect segmentation" — it is not a like-for-like CNN baseline. Say so in
  the caption.

### Waves run

| wave | what | cells | array |
|---|---|---|---|
| 2 | the ladder × 2 modalities × 5 seeds + permutation nulls | 49 | `19904770` |
| 3 | mechanism part 1: `person`, `bg_hole`, `silhouette` | 100 | `19907128` |
| 4 | ConvNeXt-Tiny/ImageNet at R1/R3/R4, with per-epoch test curves | 30 | `19939466` |
| 5 | mechanism part 2: `scale_removed`, `sil_scaled` | 60 | `19939787` + array |
| 6 | `person_centred` + the Z-precision axis (8/4/3/2/1 bits) | 90 | `19939...` |
| 7 | `bg_plate`, the exact-complement scene control | 40 | `20003651` |

Plus: prep `18533754`; the trivial-cue floor; `hiride_adjacency.py`; `hiride_plates.py`;
`hiride_census.py`. **Do not re-run prep** — it is certified (`shards_ok: true`, 39,280
frames).

**Deliberately not run:** SWAP (person A on background B) — BIWI Training is one room with
a fixed camera and plates differ by a median of 55 mm (§5.3), so it is null by construction
here; say that in the limitations rather than reporting an uninformative null. The
human-rater study stays deferred: it needs ethics, not compute.

## 1. Context you cannot infer from the repo

- **Three papers.** `paper/paper1-new` (published, de-identified video), `paper/paper-depth`
  (**this one**, paper 2, closed-set classification), and an open-set siamese re-ID study
  (paper 3) whose material is `toolbox/PAPER_HANDOFF.md` + `siamese.py` + `anthro_probe.py`.
- **Hard boundary.** Paper 2 must NOT be re-cast as open-set re-ID; that cannibalises
  paper 3. Paper 2 stays closed-set classification. Paper 3 owns: identity-disjoint
  protocols, metric learning (triplet/BNNeck/re-ranking), CMC/mAP, ConvNeXt backbones,
  the slab-width dynamic-range finding, the training-free anthropometric probe, and the
  FOV-clipping diagnosis. Paper 2 may cite these; it may not claim them.
- **The author does not pursue SOTA.** Stated explicitly: the research focus is
  applications and using depth/alternative image formats for *comparable/usable* results.
  Never frame anything as beating a benchmark. The deliverable is an operating point.
- **The author wants a consistent story arc with defensible novelty**, on the grounds that
  the area is researched but the specific question is not.
- Repo: `git@github.com:bearnl/toolbox`, branch `master`. Six untracked
  `best_*.weights.h5` files (~230 MB) belong to paper 3 — **do not commit them**.

---

## 2. What the audit established `[AUDIT]`

| # | Finding | Evidence |
|---|---|---|
| 1 | **No held-out set.** Frames split at random inside continuous single-session video. | `depth_alexnet.py:129-146`; `VALIDATION_SIZE=400` of ~23,900 frames = 1.7 %. |
| 2 | **mensa and in-house validation sets were re-drawn every epoch and overlapped training.** | `load_mensa:184` and `load_inhouse:245` omit `reshuffle_each_iteration=False`, which `load_biwi:136-137` passes. The mensa 0.99 is a training-set accuracy. |
| 3 | **Classes came from directory names, merging the two sessions.** | BIWI names Testing folders `000`,`001`,… like Training's. Union = **50 classes**, so outfit A and outfit B of one person became ONE class. Confirmed independently two ways: metric quantisation (1/(K·(1−recall)) is integral only at K=50) and the archive listing `[CLUSTER]`. |
| 4 | **Quoted numbers are TensorBoard-smoothed endpoints.** | EMA(0.6, debiased) applied to each series' final point reproduces **7 of 9** published numbers exactly (e.g. biwi-rgb-alexnet raw final 0.8350 → smoothed 0.9166 → paper 0.91). |
| 5 | **The two that don't reproduce are both ViT-RGB, and each equals that dataset's AlexNet-RGB number.** The ViT row's RGB entries were copied from the AlexNet row. | `biwi-rgb-vit` is exactly **1.0000** on accuracy, precision, recall AND f1 for every step 31–41; the paper reports 0.91. |
| 6 | **Under a like-for-like estimator the headline reverses or ties.** | Best-epoch: BIWI AlexNet **RGB 1.0000 vs depth 0.9950**; mensa AlexNet **tie 1.0000/1.0000**; BIWI ViT **RGB 1.0000 vs depth 0.9150** (a 0.118 gap, not the reported 0.02). |
| 7 | **All in-house results are a degenerate one-class task.** | `load_inhouse:211` parses the label as `split(path,'_')[-2]` → the frame index; `StaticHashTable(...,0)` sends every sample to class 0. Proven algebraically: with one true class and *k* predicted, macro-P = 1/k and macro-R = acc/k. Observed ratios are exactly 10.00, 9.00, 2.00, 1.00 and macro-P exactly 1/k, to 4 dp, across all 12 epoch-0 numbers. |
| 8 | **Efficiency claims are arithmetically false.** | 1-vs-3 input channels = 23,232 of 58,524,466 AlexNet params = **0.0397 %**; MACs −6.62 %; the `Flatten→Dense(4096)` head is ~65 % of the model. End-to-end, depth is **1.7–2.7× slower** to any fixed accuracy. No latency benchmark exists anywhere. |
| 9 | **The modality comparison was unpaired and asymmetric.** | Two independent unseeded shuffles (`:136`,`:137`) → RGB and depth scored on different frames. RGB got aliased bilinear (`antialias=False`); depth got `INTER_AREA`. |
| 10 | **`depth8` is not a bit-depth ablation.** | It is a per-frame min–max stretch over the whole frame (`:64-68`), i.e. room-relative, destroying absolute scale. Worse, `decode_png(channels=1)` at `:162`/`:218` omits `dtype=tf.uint16`, so mensa/in-house depth was silently 8-bit anyway. |
| 11 | **mensa classes are annotation *tracks*, not verified people.** | `mensa_people_extractor.py` derives the class from the track filename. Also `:52-53` prints "skipping" for non-visible frames but has **no `continue`**, so occluded crops were written with identity labels. |
| 12 | **In-house room and identity are confounded.** | `kinect2png.py` `label_map`: `leo-recording`→{leo}, `mycapture`→{bear,ranyi,xiaoyu,lirunze}, `stephen-stair-dk`→{lady1,lady2,man1,man2,stephen}. Each room holds a disjoint set of people, so cross-room generalisation is untestable. True count is **10 identities**, not the 80/90 claimed. |
| 13 | Statistical resolution. | n_val=400 → one image = 0.0025 accuracy; difference SE ≈ 0.021; the old 0.99-vs-0.91 is p ≈ 0.32. One run per cell, no seed anywhere in `depth_alexnet.py`. |

---

## 3. The story arc and the novelty claim

**Thesis.** Illegible to humans ≠ anonymous to machines. Under a leak-free protocol a
standard classifier identifies members of an enrolled cohort from single-channel depth at
many times chance; and the near-ceiling numbers in the closed-set depth literature — ours
included — are a property of the evaluation split.

**Why this frame.** Depth cameras are deployed in bathrooms, care bedrooms and clinical
spaces on a stated premise that depth doesn't identify people — a claim about *human*
perception, while deployed risk is a property of *machines*. The paper asks: how much
identity is in a depth frame, which part of the frame carries it, and what does removing it
cost. It needs no SOTA (the deliverable is an operating point), it needs no new data, and
it sits in the group's own applied domain. It also converts the 0.99 retraction from an
embarrassment into Table 1.

### How to open the paper — framing decision (author's call, 2026-07-26)

**The opening step must NOT be "we had a flaw in our script."** The author asked for this
explicitly, and it is also the more accurate framing:

- **There is no bug in the BIWI split.** `depth_alexnet.py:136-145` pools all frames,
  shuffles once, and holds out a random subset — the standard image-classification
  protocol, the default in every toolkit. The implementation is *correct*, and it even
  passes `reshuffle_each_iteration=False` so the split stays fixed, which is the careful
  choice.
- **The error is an assumption, not an implementation.** Random hold-out is valid when
  samples are independent; frames from continuous video are not. That is a domain-assumption
  mistake — easy to make, invisible in the accuracy number, and it generalises beyond this
  project, which is what turns it into a contribution rather than an erratum.
- Opening line of the argument: *we adopted the standard protocol for image classification;
  frames from continuous video violate its independence assumption, and we quantify how
  badly.*

**Which 2026-audit items may NOT ride this framing** (they are genuine defects, not
assumptions): the mensa/in-house validation set re-drawn every epoch (a missing argument);
the in-house label parser collapsing every sample to class 0; the ViT row's RGB entries
duplicated from the AlexNet row; numbers read off TensorBoard-smoothed curves. **None are
load-bearing for the new paper** — mensa is dropped as a training set, in-house is
re-derived from corrected labels, and the reporting issues vanish once correct numbers are
reported. A methods sentence noting the protocol change is sufficient disclosure. Disclose;
do not narrate.

**The trivial-cue floor is the narrative device, not the retraction.** It lets the paper open
with no self-criticism at all: *both modalities saturated under the standard protocol; to
test whether that reflected identity we asked what a model with no learned features could do,
and thirteen hand-computed scalars reached 89.4 %. A number a bounding box can reproduce is
not a statement about body shape.* That reads as someone checking their own result, not
confessing.

**Three-beat opening:** (1) premise — depth should carry identity through body geometry, and
is deployed on that assumption; (2) observation — the standard protocol saturates and the
trivial-cue floor reaches 89.4 %, so it is not resolving the question; (3) question — what
protocol *does* measure identity, and where in the frame does the signal live?

**Keep R0 inside the ladder table**, labelled "random frame hold-out (standard image
protocol)". Showing it as a rung is what converts it from an embarrassment into a
measurement: the reader sees 89.4 → 5.4 and understands why protocol choice matters.

**Novelty, stated so a reviewer can check it**
1. **The exact-complement scene control.** No published depth person-ID work reports
   accuracy with the *person removed* (pixel-exact, using the dataset's own shipped masks,
   inpainted so no silhouette-shaped hole survives) alongside the person-only complement,
   at fixed protocol and model. Prior work crops or segments first — i.e. assumes the answer.
2. **A continuous Z-precision axis terminating at the 1-bit silhouette.** Accuracy vs depth
   quantisation, 16→1 bits at *fixed global range*, with the binary silhouette recovered as
   the exact 1-bit limit of the same axis. This isolates what metric depth adds over an outline.

**Claims that must NOT be made:** "depth is under-explored" (it is a mature subfield with a
benchmark, public code, a survey and a competition track); "we tweak AlexNet/ViT for depth"
as a contribution; anything about model size or inference speed (see §2.8); "depth is
privacy-preserving because humans can't read it" as a *supporting premise* — it is the
hypothesis under test, and this paper's own result is evidence against it.

**Citations — VERIFIED 2026-08-16** (each by two independent web agents reading the full
text; the earlier `[UNVERIFIED]` tag is lifted). What each actually says, and how to use it:

- **Liu, Bouazizi, Xing, Ohtsuki, "A Comparison Study of Person Identification Using IR
  Array Sensors and LiDAR", *Sensors* 25(1):271, Jan 2025, DOI 10.3390/s25010271
  (PMC11723478).** Confirmed: RGB 100.0 % / depth 99.54 % / thermal 97.93 % — but those are
  **ResNet34 at 640×480 only**; ViT depth peaks at 94.21 %. Depth and RGB from one RealSense
  L515; the "IR array" is a FLIR C5 thermal camera. **Six subjects** (2 F / 4 M), four
  scripted walking paths × 10 repetitions each. Split protocol, the paper's only sentence:
  *"split based on predefined walking scenarios to ensure consistent representation in both
  training and validation sets"* — no ratio, no held-out test set, no session/repetition
  disjointness; the reported numbers are validation accuracies. Sweep is **spatial
  resolution** (16×12 → 640×480), not bit depth. YOLO cropping succeeded on ~0 % of depth
  frames (their Table 4), so depth was effectively full-frame. **It does not close our claim
  — it is an instance of it**: near-ceiling closed-set depth ID on a handful of subjects
  under a protocol that cannot separate identity from recording. Cite it as the most recent
  example of the pattern the ladder quantifies, and note our BITS axis is orthogonal to
  their resolution axis.
- **Mucha & Kampel, "Addressing Privacy Concerns in Depth Sensors", ICCHP-AAATE 2022,
  LNCS, Part II pp. 526–533, DOI 10.1007/978-3-031-08645-8_62** and **Mucha & Kampel,
  "Beyond Privacy of Depth Sensors in Active and Assisted Living Devices", PETRA '22 pp.
  425–429, DOI 10.1145/3529190.3534764.** Same authors, same experiments (depth *face*
  recognition: 97.95 % cross-dataset on HRRFaceD; RGB vs depth 98.79 vs 80.50 on BIWI
  faces, 85.15 vs 53.23 on Pandora). Both explicitly reject the "depth is private, full
  stop" assumption in AAL and warn of identity disclosure with < 100 individuals and
  high-precision sensors — **but both also conclude depth is *more* private than RGB.**
  Cite them for the premise (depth is deployed on an anonymity assumption the literature
  itself disputes) and be precise: they are face-scale, near-range, and they do not test
  session-disjoint protocols; our contribution is whole-frame body-scale ID and the
  protocol decomposition.
- **Delécluse, Wannous, Guimas, "ICPR 2026 Competition on Privacy-Preserving Person
  Re-Identification from Top-View RGB-Depth Camera (TVRID)", arXiv:2605.04977 (author
  preprint; footnote says the authenticated version appears in the ICPR 2026 Springer
  proceedings).** 86 identities (Zenodo says 88), four overhead RealSense D455, RGB / depth /
  cross-modal *re-ID* tracks, mAP+CMC-1; average depth mAP 57.2 % vs RGB 89.7 %, top depth
  entry 99.4 % mAP. **No thermal modality** (an earlier note said "depth/thermal/video" —
  wrong). It is open-set re-ID, i.e. paper 3's territory: cite it in paper 2 only as
  evidence that depth person-ID is a mature subfield with a benchmark and a competition
  track (§3 "claims that must NOT be made").

---

## 4. Experiment design

### 4.1 The protocol ladder — the decision that drives everything

A closed-set softmax classifier **cannot** be evaluated identity-disjoint (an unseen
identity has no output unit). So disjointness must live in session/recording/time. State
this in the paper as a design fact, not an apology; a reviewer asking for subject-disjoint
evaluation is asking for paper 3.

| policy | train → test | K | chance | isolates |
|---|---|---|---|---|
| `R0_frame_random` | random 80/20 inside Training | 50 | 2.00 % | nothing — the 2023 policy |
| `R1_block` guard 0 | contiguous block per recording | 50 | 2.00 % | block baseline |
| `R1_block` guard *g* | same, guard band before val | 50 | 2.00 % | **temporal adjacency, cleanly** |
| `R3_cross_recording` | Testing/Still → Testing/Walking | 28 | 3.57 % | recording + motion regime |
| `R4_cross_session` | Training(A) → Testing/Walking(B) | 28 | 3.57 % | **different day + clothes — PRIMARY** |

`Testing/Still` is a *training resource*; `Testing/Walking` is the single universal test
set. That makes R3→R4 a one-variable contrast on byte-identical test frames.

### 4.2 Controlled modality comparison
One manifest, identical frame IDs for both modalities (enables McNemar / paired bootstrap);
one decode+resize+normalise policy (RGB gets AREA, fixing the 2023 aliasing asymmetry);
ImageNet init for both architectures as a **fixed control** (paper 3 owns init attribution);
identical optimiser settings (2023 gave `clipvalue=0.1` to ViT only).

### 4.3 Mechanism suite — this is what answers "why"
All conditions are single controlled edits at fixed protocol, run at R0 and the strictest
rung. BIWI ships per-frame `_userMap.pgm`, so person/background separation is exact and free.

| condition | edit | if geometry | if scene |
|---|---|---|---|
| FULL | unmodified | — | — |
| PERSON | background → constant | ≈ FULL | ≪ FULL |
| BG-PLATE | person removed, **inpainted** | ≈ chance | ≫ chance |
| BG-HOLE | person removed, hole kept | controls silhouette leakage via the hole | — |
| **SWAP** | person A composited into background B | prediction follows **A** | prediction follows **B** |
| SIL | binarised mask, depth discarded | < PERSON | — |
| SCALE-REMOVED | fixed centroid distance / constant apparent size | drop ⇒ it was a size detector | — |
| BITS | 16→1 bits at fixed global range | flat then knee | — |

**SWAP is the centrepiece** — it is causal, not an ablation, so it can't be dismissed with
"you damaged the input". **Prioritise SCALE-REMOVED**: §5.4 shows apparent size and image
position dominate the trivial-cue floor.

### 4.4 Measurement
Unit of analysis is the **subject**, not the frame (frames within a recording are not
independent). Cluster-bootstrap over subjects. Pre-register an MDE (~8 pp) and refuse finer
comparisons. 5 seeds standard, 10 on headline cells. Every table carries a chance column and
a chance-multiplier column — that is how "usable" is reported without a SOTA claim.

### 4.5 Phases
- **Phase 0** — forensics + gates (mostly done; §5, §7).
- **Phase 1** — harness + legacy reproduction + determinism check (~5 GPU-h).
- **Phase 2** — the ladder, ×2 modalities ×5 seeds + label permutation (~40 GPU-h) → **gate**.
- **Phase 3** — mechanism suite (~45 GPU-h) → **gate**.
- **Phase 4** — parity/scaling: INIT×MODALITY, ConvNeXt, bits axis, accuracy-vs-K (~60 GPU-h).
- **Phase 5** — confirmatory: seed top-ups, efficiency benchmark as a measured negative
  result, estimator-sensitivity appendix (~30 GPU-h).

Total ≈ 180 GPU-h, ~8 days wall at `%16` array concurrency. Defer the human-rater study —
it needs ethics, not compute, and the thesis survives without it.

---

## 5. Measured facts `[LOCAL]` — cannot be re-derived on the new laptop

Source: BIWI `Training/` only, at `G:\biwi`, via the committed `hiride_*.py`.
Re-derive on Nibi with the prep + floor jobs.

### 5.1 Census
23,904 frames, 50 subjects (`000`–`049`), 5 files per frame, **0 incomplete frame groups**.
Per-subject 305–761 frames (median 475). Inter-frame timestamp delta is a **constant 84
units** (p10 = p90 = 84), so guards can be quoted from data rather than assumed.

**Confirmed on the full staged tree `[CLUSTER]`** (prep job 18530985, both archives):

| corpus | frames | subjects |
|---|---|---|
| `Training` (outfit A) | 23,904 | 50 |
| `Testing/Still` (outfit B) | 3,931 | **28** |
| `Testing/Walking` (outfit B) | 11,446 | **28** |
| **total** | **39,281** | — |

`depth = userMap = rgb = 39,281` and **0 incomplete frame groups**. Extraction of both
archives to `$SLURM_TMPDIR` takes ~4 min; the cue pass runs at ~320 frames/s (~2 min total).
Both Testing sequences carry the **same 28 subjects**, so `R3_cross_recording` and
`R4_cross_session` are both constructible — they could only *raise* during local development,
where only `Training/` existed.

### 5.2 userMap forensics
- Values run **0–6** — OpenNI *user indices*. Binarise `> 0`, **never `== 1`** (subject 022
  changes index mid-recording; `== 1` silently deletes part of a recording).
- **22.3 %** of frames have a completely empty userMap, and in **all 50 subjects** these form
  **exactly one contiguous mid-recording run** (000: 376–468 of 596; 001: 228–320 of 421;
  002: 284–380 of 511; 003: 260–344 of 412; 004: 176–272 of 344; 005: 236–340 of 419).
  A pose regime (walk-away / tracker loss), not random dropout. Hence `eligible_mask()` is
  applied to **every** condition so ablations never compare different pose distributions.

### 5.3 Geometry
- FOV clipping from the shipped masks, eligible frames: top edge 3.6 %, bottom 13.1 %,
  **both 2.2 %, full-body 85.4 %**.
- **This contradicts paper 3's handoff** ("100 % of Training frames clip at top AND bottom;
  0 of 13,426 are full-body"). Paper 3's figure is most likely an artifact of its slab+LCC
  foreground picking up edge-touching background blobs. **Adjudicate internally before
  either paper is submitted** — both describe the same files, and a reviewer who reads both
  damages both.
- Person median depth 1819–3975 mm (median 2955); person depth extent (p99−p1) median
  419 mm; background median depth 3842 mm.
- Depth reaches **16,524 mm**, not the ~4,500 assumed in earlier planning. Redo any
  dynamic-range arithmetic accordingly.
- Background plates differ between recordings by a median of only **55 mm** (min 23, max
  408; nearest-other median 27). One room, fixed camera — so "the model read the room" is a
  *weak* hypothesis on BIWI Training. **The leak is temporal adjacency, not scene
  memorisation.** Keep BG-PLATE (any above-chance result is then pure recording nuisance),
  but do not build the story on it.

### 5.4 The trivial-cue floor — the cheapest and most load-bearing result
13 hand-computed scalars (person px count, bbox h/w, centroid x/y, person depth
median/p1/std/range, background median, valid fraction, edge-touch flags) + random forest.
Chance = 2 %. **Features use the shipped userMap, i.e. ground-truth segmentation the CNN
never had — this is a floor on "geometry given perfect segmentation", not a like-for-like
CNN baseline. Say so in the caption.**

*(a) Exploratory, stride 4, N_train **unmatched** (5 model seeds):*

| split | acc | × chance | n_train |
|---|---|---|---|
| R0 frame-random | **68.48 % ± 0.79** | 34.2× | 3,706 |
| block guard 0 | 34.26 % ± 0.29 | 17.1× | 3,706 |
| block guard 50 | 30.05 % ± 0.53 | 15.0× | 3,367 |
| block guard 150 | 25.38 % ± 0.16 | 12.7× | 2,842 |

n_train shrinks with the guard, so this sweep conflates adjacency with data starvation.
That confound is why matching was built into the library.

*(b) Committed pipeline, stride 8, N_train **matched at 867** (3 seeds) — the reference:*

| split | acc | × chance |
|---|---|---|
| R0 frame-random | 54.98 % ± 0.77 | 27.5× |
| block guard 0 | 20.70 % ± 0.29 | 10.4× |
| block guard 25 | 18.15 % ± 0.52 | 9.1× |
| block guard 50 | 16.56 % ± 0.34 | 8.3× |
| block guard 100 | 16.43 % ± 0.35 | 8.2× |
| block guard 150 | 14.98 % ± 0.26 | 7.5× |
| **label permutation** | **1.45 %** | 0.7× (chance 2.00 %) ✓ |

Absolute values are lower than (a) because stride 8 halves the data; the *pattern* is the
same. Re-run at stride 1 on Nibi for publishable numbers.

*(c) **Full data, stride 1, both archives `[CLUSTER]`** — the publishable configuration:*
eligible **27,983 of 39,281 frames (71.2 %)** — note this is lower than Training-only's
77.7 %, so `Testing/` carries proportionally more empty/degenerate masks.

| rung | acc | per-subj | n_train | n_test | 1/K | majority |
|---|---|---|---|---|---|---|
| **`R0_frame_random` (2023 policy)** | **89.38 % ± 0.13** | 89.12 % | 13,335 | 3,703 | 2.00 % | 3.70 % |
| `R0_frame_random_matched` | 79.97 % ± 0.12 | 77.91 % | 7,324 | 3,703 | 2.00 % | 3.70 % |
| `R1_block` guard 0 | 31.81 % ± 0.23 | 31.22 % | 7,324 | 3,724 | 2.00 % | 3.33 % |
| `R1_block` guard 150 | 17.73 % ± 0.09 | 16.71 % | 7,324 | 3,724 | 2.00 % | 3.33 % |
| `R3_cross_recording` | 8.55 % ± 0.19 | 9.10 % | **1,821** | 5,642 | 3.57 % | 4.47 % |
| **`R4_cross_session`** | **5.35 % ± 0.22** | 5.73 % | 7,894 | 5,642 | 3.57 % | **4.47 %** |
| null (50-class, guard 150) | 2.58 % ± 0.26 (max 3.01) | — | — | — | 2.00 % | 3.33 % |

**The fully attributed decomposition of the 2023 number.** Each step changes exactly one
thing, and the R0-vs-R1 steps now share a matched `n_train` and near-identical test sizes:

| step | Δ | what it isolates |
|---|---|---|
| 89.38 → 79.97 | −9.4 pp | training-set size (R0 holds out only 20 % per recording) |
| 79.97 → 31.81 | **−48.2 pp** | **interleaved vs contiguous split — the core leak** |
| 31.81 → 17.73 | −14.1 pp | temporal adjacency inside the block boundary (guard sweep) |
| 17.73 → 5.35 | −12.4 pp | session + clothing change |

**Read R4 against the majority-class rate, not 1/K.** Its 5.35 % sits barely above the
4.47 % obtained by always predicting the most frequent test subject. `hiride_floor.py`
computes an empirical null band per rung so this is measured rather than argued.

**Measured null bands `[CLUSTER]`** (10 permutation draws each, permuted within the rung's
own label space):

| rung | accuracy | null | Δ over null | 1/K | majority |
|---|---|---|---|---|---|
| `R1_block` guard 150 | 17.73 % ± 0.09 | 2.98 % ± 0.29 | **+14.75 pp** | 2.00 % | 3.33 % |
| `R3_cross_recording` | 8.55 % ± 0.19 | 3.67 % ± 0.31 | **+4.88 pp** | 3.57 % | 4.47 % |
| `R4_cross_session` | 5.35 % ± 0.22 | 3.92 % ± 0.16 | **+1.43 pp** | 3.57 % | 4.47 % |

**R4 is detectably above its null but the effect is tiny.** 5.35 % against a null of
3.92 ± 0.16 is ~9 null-SD and clears the null's max (4.17 %), so it is not zero — but it is
only 1.43 pp above the permutation null and 0.88 pp above simply predicting the most frequent
test subject. On 28 subjects, that is not a usable identification signal.

**The R3-vs-R4 contrast is the sharp result.** Both are scored on byte-identical test frames;
R3 trains on 4.3× *less* data. Above their own nulls: R3 **+4.88 pp**, R4 **+1.43 pp**. The
session change destroys roughly 70 % of the recoverable signal, and it does so despite R4
having far more training data. Session shift, not data quantity, is what breaks it.

⚠️ **Do not quote the first per-rung null run** (R1 2.58, R3 2.34, R4 2.66). It permuted the
**global 50-class** label vector, pitting a 50-way null against a 28-way task — which is why
R4's null came out *below* its own 1/K of 3.57 %, an impossible value that is what exposed the
bug. Fixed to permute within `tr ∪ te`. An earlier claim in this document that the 28-class
null "lands near 5 %" was also unfounded extrapolation; the measured value is 3.92 %.

**R3 earns its place after all, and is back in Wave 2.** It trains on 1,821 frames (~65 per
subject, **4.3× fewer** than R4's 7,894) yet scores 8.55 % against R4's 5.35 % **on a
byte-identical test set**. Less data, better accuracy — the only difference being whether
training came from the same session. That is a clean demonstration that the session gap
dominates the data-quantity effect. Its starvation must still be stated whenever the
absolute number is quoted; the caveat is about interpreting 8.55 %, not about the contrast.

**Full guard sweep** (all at matched `n_train = 7,324`, byte-identical test frames, so the
fall is caused by temporal adjacency alone): guard 0 → 31.81 ± 0.23, 25 → 26.18 ± 0.09,
50 → 19.44 ± 0.05, 100 → 18.70 ± 0.13, 150 → 17.73 ± 0.09.

**This is the paper's spine: 89.4 % → 5.4 %, same features, same model, same data, only the
split changes.** Thirteen hand-computed scalars reach 89.4 % on 50-way ID under the 2023
policy; the published CNN number was 0.99, so a 58.5 M-parameter network buys roughly ten
points over person-pixel-count plus bounding-box geometry.

Four things to read carefully:

1. **The guard sweep is attributable** — `N_train` is matched at 7,324 across all five rungs
   and the test frames are byte-identical, so the 31.81 → 17.73 fall (**14.1 points**) is
   caused by temporal adjacency alone.
2. **R0-vs-R1 was NOT attributable** as first run: R0 held out only 20 % per recording and so
   trained on 13,335 vs the block rungs' 7,324. `R0_frame_random_matched` now subsamples R0
   to the same per-recording counts. Locally, matching costs ~7 points (54.98 → 48.05), so
   expect the true R0→R1 gap to be ~50 points, not ~58.
3. **The null calibration passes, and it is not a leakage probe.** A *global* per-frame label
   permutation destroys the subject↔feature relationship, so a test frame's near-duplicate
   twin carries an unrelated label and matching it yields chance — it **cannot** detect
   adjacency leakage. It measures harness soundness and locates the null, which sits above
   1/K because a permutation preserves class marginals and a noise-fitted RF drifts toward
   frequent classes. Empirically (5 draws, local): mean 2.24 % ± 0.40, max 2.90 %, against a
   majority-class rate of 3.31 %. The cluster's 2.39 % is inside that band. **The real
   leakage evidence is the 1-frame nearest-neighbour gap and the guard sweep.**
4. **R3 is data-starved, not merely strict.** `Testing/Still` is 3,931 frames over 28
   subjects, ~71 % eligible → `n_train = 1,821`, about **65 training frames per subject**.
   Its 8.55 % conflates protocol strictness with starvation. Per the original design
   criterion (drop R3 if < 100 frames/subject) it is **not viable for CNN training** — report
   it as a limitation and keep the CNN sweep on R0/R1/R4.

**The headline finding at R4: hand-crafted geometry does not survive a session change.**
5.35 % against 3.57 % chance is ~8 frame-level SE above chance but the correct unit of
analysis is the **subject** (n = 28), where it is not defensibly above chance. Consistent
with attribution: at guard 150 the dominant cues are `cent_y` (importance 0.169, leave-one-out
−3.76 pp) and `bg_med` (−2.04 pp) — **vertical image position and background depth**, i.e.
framing, which is exactly what fails to transfer to a different day's capture.

**This puts us in the pre-registered fallback branch _for the floor_ — but not yet for the
paper.** The open question is whether a CNN beats 5.35 % at R4. If it does, that is the
positive contribution: it finds transferable structure that 13 scalars do not. If it does
not, the finding is that closed-set depth ID on BIWI does not transfer across sessions and
the literature's numbers are protocol artifacts. `hiride_train.py` must therefore prioritise
**R4, with R1 guard 150 as the within-session reference.**

**Performance note.** `LogisticRegression` is deterministic — the seed only reseeds the RF
bootstrap — so running it `--seeds` times cost N-fold for zero variance information, and
multinomial LBFGS over 50 classes on the full pool takes minutes per fit. That is what made
the first full-data run appear to hang. `--models` now defaults to `rf` alone; logreg is
opt-in, single-seed, and `--logreg-max-iter` defaults to 1000. Per-fit progress is printed.

**Feature attribution** (strictest guard): `cent_y` 0.154, `bbox_h` 0.140, `cent_x` 0.109,
`n_person_px` 0.105, `p_med` 0.087. Leave-one-out: removing `cent_y` costs the most
(−4.4 pp; −5.6 pp in config (a)). **The dominant cue is where the person stood and how big
they appeared — framing/position nuisance, not body geometry.** This is why SCALE-REMOVED
is the priority mechanism condition.

### 5.5 Split-library validation (full Training, stride 1)
`test_hiride_splits.py --root G:\biwi` → **ALL INVARIANTS HELD**. n_train = **9,670 at every
guard**, val 1,931, test 4,803, all byte-identical across guards; min train→test gap 77
frames; test stable across seeds while train varies.

**And the single most quotable measurement in the whole audit:** under the 2023
frame-random policy the **minimum gap from a test frame to the nearest training frame of the
same recording is 1 frame** — median 1 across all 50 recordings. That one number is the
mechanical explanation of the 0.99.

---

## 6. Code — one amended commit on `master` (`git@github.com:bearnl/toolbox`)

| file | role |
|---|---|
| `hiride_pgm.py` | numpy-only PGM reader — venv311 has **no cv2**, nothing may depend on it |
| `hiride_data.py` | manifest keyed on the **filename** + named split policies R0–R4 |
| `hiride_prep.py` | one pass → manifest, 13 cues, background plates, ~12 large shards |
| `hiride_floor.py` | trivial-cue floor across the ladder + label permutation + attribution |
| `test_hiride_splits.py` | invariant tests; run before submitting any wave |
| `hiride_train.py` | the patched trainer: one (policy, modality, arch, condition, seed) cell |
| `hiride_collate.py` | ladder table over `results_*.json` with the floor alongside |
| `hiride_stats.py` | subject-cluster bootstrap CIs + paired RGB-vs-depth McNemar from `cm_*.npz` (needs `sequence_v2` cells) |
| `hiride_adjacency.py` | pixel-level near-duplicate measurement: \|Δdepth\|, silhouette IoU, \|ΔRGB\| vs lag, against between-subject / R3 / R4 distances (CPU, minutes) |
| `make_runs.py` | emits every wave's runs file: `--wave {2,3,4,5,6,7}` → 49/100/30/60/90/40 lines |
| `hiride_census.py` | per-sequence medians of the 13 cues + the same-subject Training→Walking shift |
| `hiride_signal.py` | CPU diagnostic: why the depth-RGB margin is what it is; gates the slab wave |
| `hiride_fov_check.py` | adjudicates the paper-2 vs paper-3 FOV-clipping contradiction (§11.1) |
| `hiride_inhouse_check.py` | what the in-house dataset can support: labels, provenance, masks (§11.1) |
| `hiride_inhouse_probe.py` | follow-up: splice scan (is each index run one recording?) + why the slab fails |
| `hiride_inhouse_prep.py` | in-house → a BIWI-format prep dir (no cues, no mask); ladder only |
| `hiride_plates.py` | per-recording background plates + global hole fill, for `bg_plate` |
| `hiride_report.py` | the three paper tables from all `results_*.json`; `--latex` for `tables.tex` |
| `run_hiride_prep.slurm` | CPU-only staging job (`.rar` → `$SLURM_TMPDIR`) |
| `run_hiride.slurm` | GPU job, single cell or array mode via `RUNS_FILE` |
| `HIRIDE_RERUN.md` | operating instructions |
| `.gitattributes` | forces LF so the `.slurm` doesn't reach Linux with CRLF |

Invariants are asserted, not assumed: test+val byte-identical across guards; N_train matched
per recording; a reference guard that would starve a recording **raises**; R3/R4 **raise** if
`Testing.rar` was not staged.

Three real bugs were caught by local testing before commit: matching ran the wrong direction
(11,156 vs 9,326); guard=300 silently un-matched 15 of 50 recordings; and the matching
reference ignored the eligibility filter.

---

## 7. Cluster facts `[CLUSTER]` and what to run next

**Access.** `ssh nibi.alliancecan.ca`, user `chenzz`, auth via the 1Password SSH agent.
`$SCRATCH` = `/scratch/chenzz`.

**Data.** `$SCRATCH/datasets/Training.rar` (5.3 G, 119,571 entries = 23,904 × 5 + dirs) and
`Testing.rar` (2.9 G, 76,944 entries = 15,377 × 5 + dirs).
**Testing is complete — it ships `_rgb.jpg` AND `_userMap.pgm`**, so the cross-session
modality comparison and the mask-based mechanism conditions are all available.
Layout: `Testing/Still/<subj>` and `Testing/Walking/<subj>`, folders named `000`,`001`,… —
*identical to Training's*, which is exactly why folder-based labelling merged the sessions.
Filenames carry the session letter: `001a_000153-d_106271072_skel.txt`.

**Repo location on Nibi: `~/toolbox`** (a checkout containing `depth_alexnet.py`,
`siamese.py`). `cd ~/toolbox` before running anything — `python test_hiride_splits.py` from
`~` fails with "No such file".

**Quota.** `/home` 8,410 MiB / 50 GiB, 100 K / 500 K files. `/scratch` 48 GiB / 1024 GiB,
**1,024 / 1,000 K files**. `/project` (def-czarnuch) **457 GiB / 931 GiB, 264 K / 500 K
files**. `/nearline` (group) 20 GiB / 9537 GiB, 117 / 1025 files. Prep writes ~12 files, not
~196 K — never extract to persistent storage. Default `OUT_DIR` is `$SCRATCH/hiride2/prep`;
consider `/project` instead if Alliance's scratch purge becomes a concern (re-running prep is
only ~20–40 min, so this is a convenience, not a risk).

**`$SLURM_TMPDIR` does not exist on login nodes**, so the staged BIWI tree — and therefore
`test_hiride_splits.py` — can only run inside a job. `run_hiride_prep.slurm` now runs the
invariant test itself, right after extraction and before prep, and aborts the job if any
invariant fails.

**`find ~/projects` and `find ~/nearline` do not work.** `~/projects/def-czarnuch` and
`~/nearline/def-czarnuch` are **symlinks** (to `/project/6005175` and `/nearline/6005175`),
and `find` does not follow symlinks without `-L`. A recon sweep that missed mensa/in-house
this way proves nothing — use `find -L` or the resolved paths. `id` gives the group as
`6005175(def-czarnuch)`.

**2023 job configuration, for reference** (`~/paper-depth.sh`): `gpu:t4:2`, `mem=187G`,
`StdEnv/2020 gcc/9.3.0 cuda/11.1 opencv/4.4.0 cudnn`, `venvs/venv38`, array `18-26`,
`datadirs=('biwi' 'mensa' 'inhouse')`, `models=('alexnet' 'old' 'vit')`,
`base_data_dir=/home/chenzz/scratch/datasets/`. That directory now holds only the two
`.rar` files — the extracted datasets are gone from scratch, consistent with the 1,024-file
count. `~/depth.py` + `~/paper2.sh` belong to the *pre-segment* draft, not paper 2.

**Environment.** `module load StdEnv/2023 python/3.11 cuda cudnn opencv`; venv
`~/venvs/venv311` = Python 3.11.5, **TF 2.15.1**, numpy 1.26.4, sklearn 1.4.2, scipy 1.14.1,
**cv2 MISSING, pandas MISSING** (the opencv module's bindings are not visible inside the
venv). Static `unrar` 7.00 at `$HOME/.local/rar/unrar`; prepend
`$HOME/.local/rar:$HOME/bin/rar:$HOME/bin` to `PATH`.

**Scheduler.** Accounts `def-czarnuch_cpu` (FairShare 0.385) and **`def-czarnuch_gpu`**
(0.262) — GPU jobs must use the `_gpu` account. `MaxArraySize` 10000. GPU GRES per paper 3's
`run_biwi.slurm`: `nvidia_h100_80gb_hbm3_{1g.10gb,2g.20gb,3g.40gb}`; `gpubackfill` also
exposes `gpu:a100:8`. AlexNet at 256² needs only `1g.10gb`.

**Everything below has been run.** Kept as the recipe if prep or the floor ever needs
re-deriving (e.g. on a different cluster). Prep is certified — do NOT re-run it to "check".
```bash
sbatch --account=def-czarnuch_cpu --export=ALL,NO_SHARDS=1 run_hiride_prep.slurm  # cues only, ~20-40 min
python hiride_floor.py --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/results # seconds, CPU
sbatch --account=def-czarnuch_cpu run_hiride_prep.slurm                            # full shards, ~13 GB
sbatch --account=def-czarnuch_cpu --time=0:40:00 --mem=32000M --cpus-per-task=2 \
  --wrap 'cd ~/toolbox && source ~/venvs/venv311/bin/activate && python hiride_plates.py --prep $SCRATCH/hiride2/prep'
python make_runs.py --wave N > runsN.txt                                           # then the GPU array
sbatch --account=def-czarnuch_gpu --array=1-$(wc -l < runsN.txt)%8 \
       --export=ALL,RUNS_FILE=$PWD/runsN.txt run_hiride.slurm
# wave 4 (ConvNeXt) additionally needs --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1 and the
# ImageNet weights pre-cached on a LOGIN node (compute nodes have no internet):
#   python -c "import tensorflow as tf; tf.keras.applications.ConvNeXtTiny(include_top=False, weights='imagenet', input_shape=(256,256,3))"
```

**Three gates before any GPU time**
1. **The invariant test must pass** on the staged tree. Now automatic inside the prep job.
2. **Label permutation must return chance.** Locally 1.45 % vs 2.00 %. Anything materially
   above chance means the split still leaks — fix before submitting anything.
3. **`R4_cross_session` must not raise.** It needs `Testing.rar` staged; the archive listing
   confirms the frames exist, so a failure here is a staging bug, not a data gap.

**All datasets located `[CLUSTER]`.** The earlier sweep that appeared to show "only logs
remain" had searched `~/projects` and `~/nearline`, which are **symlinks `find` refuses to
enter without `-L`**. With `-L`:

| path | contents |
|---|---|
| `/project/6005175/chenzz/datasets/inhouse` | in-house **extracted PNGs** |
| `/project/6005175/chenzz/datasets/mensa`, `mensa_extracted`, `pre-seg/mensa` | mensa, extracted |
| `/nearline/6005175/chenzz/rawdata/biwi/{Training,Testing}.rar` | BIWI archive backup |
| `/nearline/6005175/chenzz/rawdata/{leo-recording,mycapture,stephen-stair-dk,stephen-record}` | in-house `.mkv` sources |
| `/nearline/6005175/chenzz/rawdata/myrecording` | Ottawa / Nanjing Rd clips — not a person dataset |

**The in-house dataset is recovered `[CLUSTER]`, and the `.mkv` files are not needed.**
`kinect2png.py` writes `{label}_{idx}_{depth|rgb}.png`, so the person's name is already *in
the filename* of the extracted PNGs on `/project`. The 2023 failure was purely the **parser**
— `load_inhouse` took `split('_')[-2]`, the frame index, and `StaticHashTable`'s
`default_value=0` absorbed every miss. Reconstructing labels from filenames gives:

| leo | stephen | ranyi | lirunze | man1 | bear | lady1 | lady2 | man2 | xiaoyu | total |
|---|---|---|---|---|---|---|---|---|---|---|
| 2454 | 855 | 665 | 545 | 527 | 527 | 523 | 373 | 370 | 312 | **7151** |

14,302 files = 7,151 depth + 7,151 RGB, so pairing is complete. **10 identities, 7,151
frames** — note the paper's *frame* count (7,151) was right and only its participant count
(80 / 90) was wrong.

Two properties that must be handled before in-house is used for anything:
1. **Severe class imbalance.** `leo` is 2,454 / 7,151 = **34.3 %**, so the honest baseline is
   the *majority-class rate of 34.3 %*, not uniform chance of 10 %. Report against 34.3 %.
2. **There are no `userMap` masks** — `kinect2png.py` writes only `_depth.png` and `_rgb.png`.
   All 13 cues and every mask-based mechanism condition (PERSON / BG-PLATE / SIL / SWAP)
   require a person mask, so in-house needs its own slab + largest-connected-component
   foreground first. Keep it as a **Phase 4/5 sensor-generation replication**; do not let it
   delay the BIWI spine. Also: reading its 16-bit PNGs needs `tf.io.decode_png(...,
   dtype=tf.uint16)` since the venv has no cv2 and `hiride_pgm.py` only parses PGM.

`mensa_extracted` holds 8,996 files (`mensa` itself only 62). Together with in-house that is
23,298 of `/project`'s 264 K used inodes — **leave both where they are, do not copy to
scratch**.

**Do not go down the `.mkv` route.** `/nearline` is tape-backed and, per `run_biwi.slurm:88-91`,
**not mounted on compute nodes**; and `kinect2png.py` imports `segment_k4a`, which needs the
Azure Kinect SDK that Nibi almost certainly lacks. The extracted PNGs make it unnecessary.

Two traps in the in-house naming, for whoever fixes the parser:
- `mycapture` labels come from `filename.split('-')[2]`: `mar-22-ranyi-dk-1.mkv` → `ranyi` ✓.
- `stephen-stair-dk` files are all named `record-bear-dk-<N>.mkv`, but their labels come from
  `label_map[<N>]` → {1,2,9→stephen; 3,4→lady1; 7,8→lady2; 5→man1; 6→man2}. **The "bear" in
  those filenames is misleading** — the identity is the trailing index, not the name.
- `stephen-record/` also exists, with *overlapping* basenames (`record-bear-dk-5.mkv`,
  `-11.mkv`) and was deliberately **not** in `kinect2png.py`'s directory list. Index `11` is
  absent from `label_map`, so adding that directory naively raises `KeyError`. Leave it out.

What the audit settled regardless: **mensa classes are annotation tracks, not verified
identities**, so no leak-free split exists for it — it belongs in the paper as a documented
exclusion. **In-house room is confounded with identity** (each recording directory holds a
disjoint set of people), so cross-room generalisation is untestable at 10 identities; its role
is a sensor-generation replication of the *shape* of the BIWI result, chance 10 %. BIWI carries
the spine, and its Testing archive is complete.

---

## 8. Results — the complete set `[CLUSTER]`

**Regenerate, never retype:** `python hiride_report.py --runs $SCRATCH/hiride2/runs --floor
$SCRATCH/hiride2/results` (add `--latex` for `tables.tex`), and `python hiride_stats.py
--runs $SCRATCH/hiride2/runs --boot 2000` for the subject-cluster CIs. The tables below are
that output as of 2026-08-18, kept here so a reader with no cluster access can follow the
argument. `±` is the SD over 5 seeds (`hiride_report.py` uses ddof=1; `hiride_collate.py`
uses ddof=0, which is why its ± is slightly smaller — quote the report).

### 8.1 The protocol ladder

| split | mod | arch | acc (%) | per-subj | macro-F1 | chance | maj | floor | null |
|---|---|---|---|---|---|---|---|---|---|
| R0 frame-random | depth | alexnet | 98.36 ± 2.47 | 98.42 | 0.984 | 2.00 | 3.70 | 89.38 | — |
| R0 frame-random | rgb | alexnet | 99.99 ± 0.02 | 99.99 | 1.000 | 2.00 | 3.70 | 89.38 | — |
| R1 block g150 | depth | alexnet | 62.25 ± 6.05 | 60.26 | 0.587 | 2.00 | 3.33 | 17.73 | 4.40 |
| R1 block g150 | depth | cnxt-in | 78.02 ± 3.38 | 76.43 | 0.761 | 2.00 | 3.33 | 17.73 | 4.40 |
| R1 block g150 | rgb | alexnet | 91.94 ± 1.18 | 90.44 | 0.900 | 2.00 | 3.33 | 17.73 | 4.40 |
| R1 block g150 | rgb | cnxt-in | 99.22 ± 0.40 | 98.96 | 0.990 | 2.00 | 3.33 | 17.73 | 4.40 |
| R3 cross-recording | depth | alexnet | 34.82 ± 1.68 | 35.60 | 0.384 | 3.57 | 4.47 | 8.45 | 3.79 |
| R3 cross-recording | depth | cnxt-in | 36.74 ± 3.24 | 37.52 | 0.376 | 3.57 | 4.47 | 8.45 | 3.79 |
| R3 cross-recording | rgb | alexnet | 79.51 ± 2.26 | 79.58 | 0.785 | 3.57 | 4.47 | 8.45 | 3.79 |
| R3 cross-recording | rgb | cnxt-in | 73.23 ± 1.42 | 74.10 | 0.735 | 3.57 | 4.47 | 8.45 | 3.79 |
| **R4 cross-session** | depth | alexnet | **6.70 ± 0.97** | 6.81 | 0.024 | 3.57 | 4.47 | 5.35 | 4.30 |
| R4 cross-session | depth | cnxt-in | 8.45 ± 1.65 | 8.60 | 0.050 | 3.57 | 4.47 | 5.35 | 4.30 |
| **R4 cross-session** | rgb | alexnet | **5.59 ± 2.15** | 5.96 | 0.018 | 3.57 | 4.47 | 5.35 | 4.30 |
| R4 cross-session | rgb | cnxt-in | 10.72 ± 3.46 | 10.48 | 0.058 | 3.57 | 4.47 | 5.35 | 4.30 |

**Paired modality contrast** (identical test frames, subject-cluster CI over 5 seeds):
R1 rgb−depth **+29.7 pp** (se 2.8), R3 **+44.7** (se 1.5), R4 **−1.1** (se 0.8), every R4
seed's CI straddling zero. Within a session RGB dominates (clothing colour); across a
session neither modality survives. R4 seed 2 puts RGB at 1.90 %, *below* chance — the
clothing cue actively misleads once the clothes change.

### 8.2 Mechanism suite — depth (alexnet, 5 seeds)

| condition | R0 | R1 g150 | R3 | R4 |
|---|---|---|---|---|
| full frame | 98.36 | 62.25 | 34.82 | 6.70 |
| person only | 78.72 | 23.31 | 8.65 | 7.87 |
| person, re-centred | 69.77 | 23.81 | 9.08 | **11.92** |
| person, size+position removed | 76.59 | 47.62 | 13.76 | **12.45** |
| person removed (hole kept) | 92.58 | 55.41 | **35.80** | 5.69 |
| person removed (plate, no hole) | 99.91 | 97.16 † | **38.57** | 4.92 |
| silhouette | 93.63 | 27.93 | 7.48 | 10.06 |
| silhouette, size+position removed | 88.75 | 52.73 | 13.71 | **14.59** |

† structural, not a scene measurement — see §0.

### 8.3 Mechanism suite — rgb (alexnet, 5 seeds)

| condition | R0 | R1 g150 | R3 | R4 |
|---|---|---|---|---|
| full frame | 99.99 | 91.94 | 79.51 | 5.59 |
| person only | 99.86 | 84.48 | 47.81 | **14.46** |
| person, re-centred | 99.49 | 85.01 | 48.88 | 14.45 |
| person, size+position removed | 99.34 | 96.33 | 66.77 | **17.72** |
| person removed (hole kept) | 97.09 | 58.27 | 55.31 | 5.60 |
| person removed (plate, no hole) | 99.97 | 99.95 † | 41.11 | 3.00 |

**How to read these two tables.**
- *Within a session the depth CNN reads the room.* `bg_hole` ≈ `full` at R1 (55.4 vs 62.3)
  and *exceeds* it at R3 (35.8 vs 34.8); `bg_plate` at R3 is 38.6 while the person alone is
  8.7, i.e. on the floor. R3's accuracy is a recording signature, not a person.
- *Across a session the background is actively harmful.* Deleting it nearly triples RGB
  (5.6 → 14.5) and lifts depth (6.7 → 7.9→11.9 once re-centred). The model latches onto a
  room that has moved: the census shows background depth +620 mm for **every one** of the
  28 shared subjects, people ~30 cm closer and 34 px lower in frame.
- *Position and size do different jobs at different rungs.* Re-centring is worth nothing
  within a session (R1 23.3 → 23.8) and everything across one (R4 7.9 → 11.9), because
  within a session the person stands in the same place. Rescaling is the opposite.
- *The outline beats metric depth across sessions.* `sil_scaled` 14.59 vs `full` 6.70,
  **+7.89 pp, subject CI [+0.16, +15.51]** — the only R4 condition-vs-full contrast whose
  interval excludes zero.

### 8.4 Z-precision axis (depth, `scale_removed`, fixed 0–6000 mm range)

| precision | levels | R1 g150 | R4 | R4 subject CI | clears majority |
|---|---|---|---|---|---|
| 16 bits | 65536 | 47.62 | 12.45 | [6.03, 19.85] | yes |
| 8 bits | 256 | 46.40 | 12.45 | [6.26, 19.93] | yes |
| 4 bits | 16 | 43.94 | 13.02 | [6.79, 20.15] | yes |
| 3 bits | 8 | 42.53 | 11.22 | [4.92, 18.80] | yes |
| 2 bits | 4 | 33.61 | 12.17 | [6.78, 18.38] | yes |
| 1 bit | 2 | 27.91 | 9.51 | [3.93, 16.28] | **no** |
| binary silhouette (shipped mask) | 2 | 52.73 | 14.59 | [8.82, 20.95] | yes |

R1 is flat to 3 bits then knees; R4 is flat within noise to **2 bits**. The 1-bit row is a
*thresholded depth map*, not a silhouette — a global 3 m threshold cuts through the body,
which is why it under-performs the shipped mask. Do not call it "the silhouette".

### 8.5 Subject-level uncertainty — which claims survive n = 28

`hiride_stats.py` resamples SUBJECTS, not frames. At R4 the cells whose 95 % CI lower bound
clears the majority-class rate (4.47 %) are: depth `sil_scaled` 14.59 [8.82, 20.95],
`scale_removed` 12.45 [6.03, 19.85], `person_centred` 11.92 [5.21, 20.00], `silhouette`
10.06 [5.10, 16.49], the 2–8 bit rows above, and rgb `scale_removed` 17.72 [5.99, 30.86].
**Everything unnormalised fails**: depth `full` [1.40, 13.66], rgb `full` [0.21, 13.31],
rgb `person` [3.86, 27.29], depth cnxt `full` [3.07, 15.32]. Report the interval, not the
point estimate, and state that n = 28 is the binding constraint.

### 8.6 Pixel-level diagnostics

`hiride_adjacency.py` (`adjacency_results.json`) — why the guard sweep flattens, and what
R4 actually changes. Median over pairs:

| pair | \|Δdepth\| | silhouette IoU | \|ΔRGB\| |
|---|---|---|---|
| lag 1 (adjacent frames) | 80.6 mm | 0.954 | 2.4 |
| lag 25 | 139.3 mm | 0.687 | 5.4 |
| lag 150 | 179.4 mm | 0.566 | 7.4 |
| different subject, Training | 213.5 mm | 0.544 | 10.0 |
| same subject, Still→Walking (R3) | 322.7 mm | 0.323 | 6.7 |
| **same subject, Training→Walking (R4)** | **1313.6 mm** | 0.309 | **65.4** |
| different subject, Training→Walking | 1319.5 mm | 0.314 | 65.9 |

Two things follow. Adjacent frames are near-duplicates (IoU 0.95, |ΔRGB| 2.4/255), and by
lag ~150 a frame is as far from its own recording as from a *different subject* — which is
exactly where the guard sweep flattens. And **a same-subject R4 pair is as far apart as a
between-subject pair** (1314 vs 1320 mm; 65 vs 66): the scene moved between sessions, so R4
is day + clothes + camera/room. State that as a property of BIWI, not as a design choice.

`hiride_census.py` quantifies the same shift per cue: `bg_med` +620 mm (IQR 549–709, all 28
subjects), `p_med` −302 mm, `n_person_px` +3009, `cent_y` +34 px. Also worth a methods
sentence: only **49.3 %** of `Testing/Walking` frames are eligible (the tracker loses the
person in half of them) against 97.2 % of `Testing/Still` and 77.5 % of `Training`.

## 9. Cluster gotchas already paid for `[CLUSTER]`

Each of these cost a failed job. Do not rediscover them.

- **BIWI ships at least one zero-byte `_rgb.jpg`, in `Testing/`.** It passes any
  "does the file exist" check and then kills prep with
  `InvalidArgumentError: Input is empty. [Op:DecodeJpeg]` thousands of frames later.
  `build_manifest` now stats all three files per frame and drops zero-byte members at
  manifest level, so cues, splits and trainer all see one corpus. Local `Training/` has
  none (23,904 frames unchanged), so the loss is confined to `Testing/`.
- **Changing iteration order relocates that bug rather than fixing it.** The original loop
  walked the manifest front-to-back and manifest order sorts `Testing/Still` first, so it
  died at frame ~2–4 k. The streaming rewrite walks `Training` first, so the same file
  surfaced at ~25.9 k and the job merely *looked* healthy for longer.
- **`MaxRSS` from `sacct`/`sstat` is not the process's memory.** It is the cgroup figure and
  includes page cache — `unrar` writing ~39 GB into `$SLURM_TMPDIR` plus every read and
  write since. Prep reports its own RSS (`/proc/self/statm`) in the progress line: that
  number sat **flat at 0.9 GB** while `sstat` showed 57.5 GB. Page cache is reclaimable;
  do not diagnose an OOM from it.
- **`NonZeroExitCode` in `squeue`'s reason column means the process raised**, not that it
  was killed. An OOM kill reports `OUT_OF_MEMORY` or a signal. Always read the log tail
  before theorising — inferring OOM from a memory figure cost a whole debugging cycle here.
- **Do not use `np.lib.format.open_memmap` for multi-GB shards on Lustre.** Pages stay
  resident, `flush()` msyncs without evicting, and a mid-loop death leaves silently partial
  files that load fine and train to chance. `_StreamWriter` writes an `.npy` header then
  appends buffered chunks sequentially, raises on truncation, and requires ordered appends.
- **Resize in numpy, not TensorFlow.** Three eager `tf.image.resize` calls per frame is
  ~120 k calls over the corpus and TF's CPU allocator does not return memory. TF is now used
  only for JPEG decode, which numpy cannot do.
- **`$SLURM_TMPDIR` does not exist on login nodes**, so anything needing the extracted tree
  must run inside a job. `run_hiride_prep.slurm` runs the invariant test itself.
- **`find ~/projects` and `find ~/nearline` silently return nothing** — both are symlinks and
  `find` will not traverse them without `-L`.
- The `cuInit: CUDA_ERROR_NO_DEVICE` line at the top of every prep log is harmless: TF probing
  for a GPU on a CPU node, and prep disables GPU visibility anyway.

**Two questions the author has already raised — answer them from here, don't re-derive:**

- *"Wasn't the original methodology k-fold?"* No, not for this paper. `depth_alexnet.py:10`
  imports `StratifiedKFold` **and never calls it**; the only other `fold` matches in the file
  are the words "folder name" in comments. The split is a single `shuffle → skip(400) /
  take(400)`. The k-fold recollection is `~/depth.py`, which does `from
  sklearn.model_selection import KFold` — that is the **pre-segment** paper's script, not
  paper 2's. Easy to conflate; both are the author's and both concern depth.
  **And k-fold would not have fixed it**: folding pooled frames from continuous video still
  puts each held-out frame's temporal neighbours in the training folds by construction. It
  averages over K leaky estimates instead of reporting one. The leak is a property of which
  frames may be adjacent across the boundary, not of how many times you redraw it.
- *"Why call adjacent frames near-duplicates?"* Separate what is measured from what is
  inferred. **Measured:** the minimum gap from an R0 test frame to the nearest training frame
  of the same recording is **1 frame** (median 1, all 50 recordings); and the guard sweep, at
  matched `n_train` with byte-identical test frames, falls 31.81 → 17.73 as the guard widens
  0 → 150, i.e. **14.1 points caused by temporal proximity with everything else fixed**.
  **Inferred, not yet measured:** that the frames are *visually* near-duplicates. The
  inference is well supported (if adjacent frames were dissimilar the guard could not cost 14
  points) but no pixel-level number exists yet. **Measurement scripted, not yet run:**
  `hiride_adjacency.py` gives median |Δdepth| (mm, valid pixels, clipped at 6000), silhouette
  IoU and |ΔRGB| between frames k ∈ {1,2,5,10,25,50,150,500} apart within a recording, against
  four reference distances: between-subject (Training), same-subject Still→Walking (R3's
  train→test gap), same-subject Training→Walking (R4's gap), between-subject Training→Walking.
  Cheap CPU job (§8 recipe); it belongs beside the guard sweep because it *explains* it
  rather than restating it. Expected shape: lag-1 ≪ R3 gap < R4 gap ≈ between-subject.

- **Two distinct Slurm failure codes, two distinct causes.** `0:125` = cgroup OOM kill.
  `1:0` = a Python exception (including numpy's `MemoryError`, which is *not* an OOM kill).
  Do not treat them as the same thing; read the log either way.
- **The class map must be built from the training subjects but applied only to split rows.**
  `cls_index` covers the 28 subjects R3/R4 train on while the manifest holds all 50, so
  mapping the whole manifest raised `KeyError: '004'` and killed every R3 and R4 cell.
  Rows outside the map now get `-1` and an assertion checks no split touches them.
- **Never return a scaled copy of a loaded split.** `out * 2.0 - 1.0` allocates a second
  full-size array; an RGB R0 split is 10.5 GB, so the copy doubled peak and killed 10 cells.
  Scale in place.
- **Slurm parses `#SBATCH` at submit time but reads the Python fresh at run time.** A pulled
  code fix reaches queued array tasks; a changed `--mem` does not.
- **Cells run 2–6 minutes**, so the full 46 is ≈ 2.7 GPU-h. The `sm_90` PTX JIT warning is
  harmless — CUDA caches to `~/.nv/ComputeCache` on shared `$HOME`.

**Prep job history:** `18530985` cues-only, OK. `18532680` killed mid-loop, partial shards,
undetected. `18533099` failed on the zero-byte JPEG (misdiagnosed as OOM at the time).
`18533484` cancelled — same JPEG, relocated by the new iteration order. `18533754` is the
first clean run, `shards_ok: true`.

**Wave 2 job history:** `18535526` — 46 cells, 13 completed (R0/R1 depth + R1 perm), 10
OUT_OF_MEMORY (R0 RGB, the copy bug), 20 FAILED (`KeyError` on R3/R4, plus R1 RGB memory).
`18539849` cancelled — submitted before the `KeyError` fix. `18539954` — 33 cells resubmitted:
15 COMPLETED (R3 depth+rgb, R4 depth), 18 FAILED — the GPU-OOM and permuted-label bugs
listed below. Waves 3–7 (`19907128`, `19939466`, `19939787`, wave 6, `20003651`) all
finished 100 % `COMPLETED` once those were fixed; §0 has the table.

---

**Learned during waves 2–7 — every one of these would have corrupted a reported number:**

- **`model.fit(numpy_array)` puts the WHOLE array on the GPU.** Keras 2.15 wraps a
  tensor-like input in a `tf.constant`, which lands on the device; the 10 GB MIG slice has
  an 8.47 GB pool. R0-RGB (10.5 GB) died in `fit`, R1/R4-RGB died later in `predict` when
  the test set had to sit beside the train set. Symptom: `InternalError: Failed copying
  input tensor … Dst tensor is not initialized`, exit `1:0` — a Python exception, NOT a
  Slurm OOM, and `MaxRSS` looks healthy. Depth (1 channel) never hit the ceiling, which is
  why every depth arm passed and hid it. Fixed with a `keras.utils.Sequence` (`ArrayBatches`).
- **A permuted-label null must permute INSIDE the split.** Permuting the whole manifest
  dragged the `-1` of the 22 Training-only subjects into R4's rows: sparse CE on label −1
  gives `loss: nan` from epoch 1 and `np.bincount` raises at the end. Remember the symptom:
  a permuted run whose *training* loss is `nan` is a labelling bug, not a null result.
- **tf.keras 2.15 `EarlyStopping(restore_best_weights=True)` restores ONLY if patience
  fires.** A cell that reaches `--epochs` is scored on last-epoch weights while
  `best_epoch` reports the argmax. Restore unconditionally after `fit()`; results now carry
  `hit_epoch_cap`. (Keras 3 moved this into `on_train_end`; 2.15 did not.)
- **Keras 2.15's ConvNeXt leaves its `include_top=False` tail LayerNorm unnamed**, so it is
  auto-named from a process-global counter — `layer_normalization` in the first model built,
  `layer_normalization_1` in the second. A by-name weight copy silently skips it. Pair the
  two models' layer lists POSITIONALLY and assert `copied + averaged == n_weighted`.
- **`ConvNeXtTiny` applies ImageNet mean/std to 3-channel input expecting 0–255 pixels.**
  This trainer feeds `[-1, 1]`, which that `PreStem` would squash to a near-constant −2.1.
  Pass `include_preprocessing=False` and normalise explicitly.
- **Quantisation must not collide with the invalid-depth sentinel.** A first `--bits`
  implementation rounded to `floor(x·(L−1)+0.5)/(L−1)`, sending everything nearer than half
  the range to exactly 0 = "invalid": at 1 bit the person was erased into the fill and the
  axis read the null for reasons unrelated to depth precision. Use bin centres in (0, 1).
- **A background plate needs an explicit "was this pixel ever seen?" mask, per modality.**
  0 is legitimate in both (invalid depth, black pixel), and an unfilled hole is exactly the
  person-shaped boundary `bg_plate` exists to remove. Depth and RGB masks differ: ~25 % of
  pixels have no valid depth while their RGB is fine. Fill holes from a GLOBAL plate, not a
  flat scalar — a flat fill leaves a blob where that subject stood longest, and at R3 a
  subject's two recordings share it.
- **Any key that groups result cells must include `bits`.** Omitting it made the six
  quantisation variants of one condition collide in `hiride_stats.py`, silently keeping only
  the last file read instead of erroring.
- **`sacct -j <id> -X` on a still-queued array** prints one row for the pending elements, so
  a "1 PENDING 1 RUNNING" summary does not mean the array is two cells.

## 10. Working agreements

- Single commit, **amended** as work proceeds (the user asked for this). Once it has been
  pulled on a second machine, an amend needs a force-push and `git fetch && git reset --hard
  origin/master` on the consumer side. Offer stacked commits instead if that becomes awkward.
- The agent's `git push` is blocked by a permission classifier — the user pushes.
- No compute on login nodes beyond `find`/`ls`/small text ops.
- Everything new lives under `$SCRATCH/hiride2/`; existing data, results and paper-3
  material are never modified or deleted.
- No `sbatch` of a wave without first stating run count and GPU-hours.

---

## 11. Next stage — the manuscript

The experiments are done (§0, §8). What follows is writing, and two things that must be
settled before either paper is submitted.

### 11.1 Blockers — settle these first

1. **RESOLVED 2026-08-18 — the FOV-clipping contradiction. Paper 2 is right; paper 3's
   §4.6 measures the room, not the body.** `hiride_fov_check.py --root <staged tree>
   --paper3-exact` replicates `anthro_probe.py`'s `_foreground` + `_audit_frame` exactly
   (slab `FG_CLIP_MM=1500`, `FG_PERCENTILE=1`, largest connected component,
   `MIN_FG_PIXELS=200`, edge margin 2 rows), on 4,000 eligible raw 480×640 Training frames:

   | foreground, same frames, same edge rule | top | bottom | both | full-body |
   |---|---|---|---|---|
   | paper 3 (`anthro_probe._foreground`) | 99.95 % | 99.98 % | 99.92 % | **0.00 %** |
   | shipped userMap | 3.25 % | 13.23 % | 1.98 % | **85.50 %** |

   Overlap of paper 3's foreground with the person: **recall 0.976, precision 0.204**, and
   it covers **39 % of the frame** (the person is ~8 %). So the LCC step does recover the
   body — and then merges it with the floor and the wall behind, because a standing person
   is depth-continuous with the floor at a 1500 mm slab. The merged blob necessarily runs
   to the bottom edge and up to the top, which is the entire "100 % clipped" observation.

   **This also explains the rest of §4.6 without any FOV saturation.** A "height" taken as
   the p2–p98 Y extent of that blob is mostly the ROOM's vertical extent, identical for
   everyone — hence "height std across 28 distinct people = 16 mm" and "same-person height
   correlation Training↔Testing: Pearson −0.04, Spearman −0.15". Those are what measuring
   the room looks like. **Paper 3's conclusion "cross-session failure = BIWI's tight
   Training framing (FOV clipping), not depth" is therefore unsupported by the evidence
   offered for it**, and needs either a corrected foreground (the shipped userMap is free
   and exact) or a different explanation.

   **Scope of this finding, stated precisely.** It refutes the *audit* and the diagnosis
   built on it. It does NOT touch paper 3's re-ID accuracy numbers, which come from a
   different code path (`siamese.preprocess_depth_smart`, `CLIP_RANGE=600`, **no LCC**,
   crop around the foreground centroid). But that path deserves its own check by paper 3's
   author: measured here on the same frames, the bare 600 mm slab has **IoU 0.000 and
   precision 0.000** with the person — the anchor sits at 1975 mm (±20 mm across 4,000
   frames from 50 recordings, i.e. a static scene element) while the nearest person surface
   is at 2810 mm, 835 mm behind it. If that holds inside paper 3's own loader, its models
   were trained on a background band, and its within-session numbers may be the same
   recording-fingerprint effect this paper documents as `bg_plate` (97 % at R1). **Verify
   with paper 3's own data loader before acting on it** — the trainer may crop or reload
   differently than the audit does, and the two foregrounds in that repo already disagree
   with each other (1500 mm + LCC vs 600 mm without).

   *(original blocker, kept for context)* ~~Adjudicate the FOV-clipping contradiction with paper 3.~~ §5.3 measures, from the
   shipped masks on eligible frames, top-edge clipping 3.6 %, bottom 13.1 %, both 2.2 %,
   **full-body 85.4 %**. `toolbox/PAPER_HANDOFF.md` (paper 3) states "100 % of Training
   frames clip at top AND bottom; 0 of 13,426 are full-body". Both describe the same files.
   The likely cause is paper 3's slab+LCC foreground picking up edge-touching background
   blobs, but that is a hypothesis, not a finding — check it and fix whichever document is
   wrong. **A reviewer who reads both papers damages both.**
2. **In-house STAYS — the author's call, 2026-08-18.** The question is therefore what it
   can support, not whether to keep it. `hiride_inhouse_check.py` answers that; one answer
   is already visible from the code. In `kinect2png.py:34-64` the output directory comes
   from the SOURCE directory while `idx` restarts at 0 for every `.mkv` inside it, so two
   recordings of one person overwrite each other frame by frame — every `leo-recording`
   file (all labelled `leo`), and `stephen` {1,2,9}, `lady1` {3,4}, `lady2` {7,8} in
   `stephen-stair-dk`. Where that happened an identity's index range is a MIXTURE of
   recordings with no provenance left in the filename, so **in-house can carry R0 and R1
   but no recording- or session-disjoint rung**. That is still worth having: it asks
   whether the R0→R1 collapse reproduces on a different sensor generation (Azure Kinect
   vs BIWI's Kinect). It also has no userMap, so the mechanism suite needs a slab+LCC
   foreground of our own — viability measured by the same script. Report against the
   **majority-class rate (leo = 34.3 %)**, never 1/K = 10 %.

   **MEASURED 2026-08-18** (`inhouse_check.json`). The overwrite is confirmed by
   arithmetic: 26 source `.mkv` files → 10 unbroken index runs, 7,151 frames.
   `leo-recording` 3 files all labelled `leo` (2,454 frames); `mycapture` 14 files → 4
   labels (ranyi 5 files/665 frames, lirunze 4/545, xiaoyu 3/312, bear 2/527);
   `stephen-stair-dk` 9 files → 5 labels (stephen 3/855, lady1 2/523, lady2 2/373, man1
   1/527, man2 1/370). **Only `man1` and `man2` come from a single recording.** `bear`'s
   two files are `aug-11` and `mar-22` — different days, i.e. a real cross-session pair,
   overwritten together. `stephen-record/` (13 further files, indices 1–13) was never in
   `kinect2png.py`'s directory list and remains unextracted. Images are 256×256 uint16,
   depth and RGB fully paired.

   Slab+LCC foreground is **bimodal, not merely weak**: 60/60 frames for `lirunze`,
   `ranyi`, `xiaoyu`; ~0/60 for `bear`, `lady1`, `lady2`, `man1`, `man2`, and 4–7/60 for
   `leo`, `stephen` — 31.8 % overall. Consistent with capture geometry (a stairwell puts a
   step or railing nearer than the person, so the 1st-percentile anchor lands off-body —
   the same mechanism `hiride_fov_check.py` tests on BIWI).

   **Plan.** In-house replicates the **ladder (R0 vs R1) on depth and RGB** — no masks
   needed — as a sensor-generation replication: does the frame-random → contiguous-block
   collapse reproduce on an Azure Kinect? The **mechanism suite stays BIWI-only**. Run
   `hiride_inhouse_probe.py` first: R1 is only meaningful if each index run is one
   coherent recording, which the splice scan settles.

   **PROBE RESULT 2026-08-18** (`inhouse_probe.json`). Runs ARE coherent — 0 splice
   candidates for every label, so the longest `.mkv` overwrote the rest and each index
   run is one recording. **R1 is therefore sound.** But the near-duplicate ratio
   (adjacent |Δdepth| ÷ far-apart |Δdepth|) varies enormously by capture: `leo` 0.26,
   `mycapture` 0.34–0.61, `stephen-stair-dk` 0.74–0.91, against BIWI Training's 0.38. At
   0.9 adjacent frames are nearly as different as random ones, so **little frame-random
   leak is left to demonstrate on the stairwell recordings** — which turns into a
   quantitative prediction worth testing: the R0→R1 gap should track the ratio.

   Masks are settled and the answer is no. The slab anchor sits 1.4–2.7 m in front of the
   scene median for EVERY identity (Azure Kinect near-field speckle; 590–985 connected
   components per frame), and the three identities that passed an area filter have
   cross-frame mask IoU **0.96–0.98** with centroid std < 13 px — a STATIC artefact, not a
   body. So `hiride_inhouse_prep.py` writes no mask shard, and the trainer refuses every
   mask condition on that prep rather than fabricating an all-background mask.

   **Wave 8 (`make_runs.py --wave 8`, 26 lines = 20 cells + 6 nulls, ~1–2 GPU-h):**
   ```bash
   python hiride_inhouse_prep.py --root /project/6005175/chenzz/datasets/inhouse \
       --out $SCRATCH/hiride2/prep_inhouse
   python make_runs.py --wave 8 > runs8.txt
   sbatch --account=def-czarnuch_gpu --array=1-26%8 \
     --export=ALL,RUNS_FILE=$PWD/runs8.txt,PREP=$SCRATCH/hiride2/prep_inhouse,OUT=$SCRATCH/hiride2/runs_inhouse \
     run_hiride.slurm
   ```
   **Quote the R0→R1 DROP, not the level.** Each recording directory holds a disjoint set
   of people, so identity and room are confounded and the absolute numbers are inflated —
   and unlike BIWI we cannot run `bg_plate` to decompose it. The confound is identical on
   both rungs, so the drop is still attributable.

   **WAVE 8 RESULT 2026-08-18** — array `20024361`, 26/26 `COMPLETED`,
   `$SCRATCH/hiride2/runs_inhouse`. **The collapse replicates on a second sensor
   generation.**

   | | R0 frame-random | R1 block g150 | drop | per-subj R0 → R1 | drop |
   |---|---|---|---|---|---|
   | depth | 93.69 ± 1.31 | 80.81 ± 0.25 | −12.9 pp | 89.74 → 67.44 | **−22.3 pp** |
   | rgb | 99.72 ± 0.17 | 81.77 ± 0.84 | −18.0 pp | 99.60 → 68.80 | **−30.8 pp** |
   | depth [permuted] | 35.41 ± 1.13 | 41.29 ± 1.30 | — | 10.00 / 10.00 | — |

   **Three baselines, and the third is the one to quote.** 1/K = 10 % is meaningless here;
   the majority class (`leo`) is 34.3 %; and because each of the three recording
   directories holds a disjoint set of people, an oracle that identifies only the ROOM and
   then names that room's most frequent person scores **55.6 % frame / 30.0 % per-subject**.
   R1 clears even that by +25 pp (frame) and +37 pp (per-subject), so there is real
   within-room identity signal — but the room-only oracle, not chance, is the honest
   reference and must appear in the table.

   **The permutation nulls behave exactly as they should** and are worth a sentence in the
   paper: 35.4 % / 41.3 % frame accuracy with per-subject accuracy of exactly 10.00 %, i.e.
   the model predicts the majority class for every frame. That is what a null looks like
   under severe imbalance, and it is why frame accuracy alone cannot be the headline metric
   on in-house.

   **Against the prediction.** The R0→R1 drop was predicted to track the near-duplicate
   ratio. BIWI Training (ratio 0.38) drops 36.1 pp on depth; in-house (0.26 for `leo`, but
   0.34–0.61 and 0.74–0.91 for the other two rooms, so a higher mixture) drops 12.9 pp on
   frames and 22.3 pp per-subject. Direction is right — less near-duplication, smaller leak
   — but this is two datasets and a mixture of ratios, so it is **suggestive, not
   established**. Stating it as a per-recording correlation would need the drop measured
   per identity, which is possible from `cm_*.npz` and has not been done.

   *(superseded)* ~~Decide the in-house dataset's role, or drop it explicitly.~~ It is recovered and
   usable (§7): 10 identities, 7,151 frames, labels reconstructible from the PNG filenames
   on `/project`. But it has **no userMap**, so every mask-based condition — which is now
   the entire mechanism story — needs a slab + largest-connected-component foreground
   first, and `leo` is 34.3 % of frames so the honest baseline is the majority rate, not
   10 %. Cost: a segmentation pass plus ~40 GPU-cells. Value: a sensor-generation
   replication of the *shape* of the BIWI result. **Recommendation: state it as a
   documented exclusion** unless a reviewer asks — the BIWI spine stands on its own and the
   in-house room/identity confound (§2.12) limits what it could show.

### 11.1b Why is the depth-RGB margin so wide? (author's question, 2026-08-18)

RGB beats depth by ~30 pp at R1 and ~45 pp at R3. The author's position — correct — is that
theory does not predict a gap that wide, so the pipeline is probably handicapping depth. The
suspect is our own normalisation, and it is quantified: the global 0-6000 mm scale gives the
body, whose depth extent (p99-p1) has a median of **419 mm**, about **7 % of the input
range**. Shape arrives as a near-flat plate while RGB's clothing texture uses the full range.
Paper 3 independently swept this and found it dominant (`clip 600 -> 32.8 %`,
`clip 300 -> 47.6 %`, "**No RGB analog**").

`--depth-slab-mm` now rescales a fixed window centred on the person's median depth onto the
full range (default 6000 = exact identity transform, so every completed run is unaffected).
A fixed width is deliberate: normalising by each person's OWN extent would delete body
thickness, which is itself a biometric. Measured on a synthetic body: **~15x more contrast**
at slab 400.

**This is also the paper's honest "benefit" claim.** Depth admits cheap, physically grounded
preprocessing that RGB cannot: background removal by distance threshold (no segmenter),
body-relative range normalisation (no colour analogue), illumination invariance. The claim
is not that depth beats RGB — it is that both are *usable*, and that depth gets there with
preprocessing RGB cannot perform.

**`hiride_signal.py` gates the GPU spend** by testing the chain on CPU: contrast statistics,
separability across the split gap, and a linear probe (PCA + nearest-class-mean). Two design
errors were caught by adversarial review before it ran, both worth remembering:
- **A separability ratio must have BOTH pair types crossing the gap.** The first version took
  the numerator across train->test and the denominator inside train, so a common-mode shift
  (the camera moved between BIWI sessions) inflated it alone. §8.6's own numbers show the
  damage: 1313.6 / 213.5 = 6.2 ("impossible") versus the matched 1313.6 / 1319.5 = **1.00** —
  and depth `sil_scaled` still reaches 14.59 % there. The rule would have declared the
  paper's central result impossible.
- **Un-whitened nearest-class-mean is blind to low-variance directions**, i.e. to exactly the
  compressed interior the hypothesis is about, so a flat result could not distinguish "no
  signal" from "probe cannot see it". The probe now reports raw, whitened, and
  test-centred variants; **trust the whitened one for a null**, and read
  `te_centred >> whitened` as "the cross-session failure is a fixable offset, not lost
  information".

### 11.1c Overnight batch, submitted 2026-08-19 — collect these first

Four independent jobs. **`hiride_signal.py`'s whitened probe gates the GPU arm**, so read
the CPU results before acting on wave 9's numbers.

| job | id | question | ~ |
|---|---|---|---|
| `hiride-signal2` | `20036772` | interior-only vs outline-only; `--frames` 1/5/10/25; normals; accuracy per range bin | 40 min CPU |
| `hiride-noise` | `20036773` | temporal sigma on static background, per range bin | 15 min CPU |
| **wave 9** | `20036778` | `interior_only`, `--frames 10` (depth/rgb/sil), normals+fusion, at R1 and R4 | 60 cells, ~4 GPU-h |
| `hiride-quant` | `20036786` | measured quantisation step vs range, from raw PGM | 20 min CPU |

**Collect with:**
```bash
cd ~/toolbox && for j in 20036772 20036773 20036786; do echo "=== $j"; grep -v "^2026-\|oneDNN\|cuInit\|TF-TRT\|To enable\|cpu_feature\|module" logs/*_$j.out; done
sacct -j 20036778 -X -P --format=State -n | sort | uniq -c
python hiride_collate.py --runs $SCRATCH/hiride2/runs --floor $SCRATCH/hiride2/results | grep -E "arch |interior|_f10|nrm"
```

**What each result decides.** The chain being tested is that the in-body depth signal sits at
the SENSOR floor at BIWI's 3 m standoff, not that our pipeline lost it — the dynamic-range
sweep already refuted the pipeline hypothesis (R4 flat at 13.9 → 14.3 → 11.4 % from slab
6000 → 300 despite 15x more input contrast, because BatchNorm absorbs a linear rescale, and
paper 3's clip effect was foreground SELECTION rather than contrast).

- *interior_only above chance* ⇒ the interior has signal and the earlier "~1 pp" was a
  MARGINAL result given the outline (redundancy), not absence. If it is at chance, depth ID
  here really is silhouette ID and the paper says so.
- *accuracy rising with `--frames`* ⇒ the bottleneck is noise, and sigma/sqrt(N) is the fix.
  Predicted from: quantisation ~25 mm at the person's 2955 mm median, measured sigma 40-70 mm
  (implied by the 80.6 mm frame-to-frame delta), against 20-80 mm of between-person curvature.
- *accuracy rising as people get CLOSER* (range bins) ⇒ the strongest version of the same
  claim, since quantisation grows as z^2: ~4.5 mm at 1.25 m to ~40.7 mm at 3.75 m across
  BIWI's own 1240-3885 mm span. This would give a deployment statement nobody else has:
  in-body depth shape becomes usable below ~X metres.
- *normals only working ON TOP of fusion* ⇒ expected, and the `depth normals` (no fusion) row
  exists to show it: a 1-px lateral step at 3 m spans ~5 mm against 40-70 mm of sigma.

**Two safeguards built into wave 9, do not remove them.** `temporal_windows` restricts every
fusion window to rows inside its OWN split — at R1 a test frame's temporal neighbours are
training frames, so a recording-wide window would silently restore the adjacency leak this
paper exists to measure, on the rung used as the within-session reference. And the fusion is
a MEDIAN, not a mean, because depth dropouts are stored as 0 and a mean would drag every
fused pixel toward that invalid sentinel.

**Territory, settled 2026-08-19.** Paper 1 is PUBLISHED and skeleton-only (a 20-D
bone-segment-length feature vector, 14 participants), so `_skel.txt` is off-limits to paper 2
entirely — including as a pose covariate; `Testing/Still` provides pose control instead.
Paper 3 is still a DRAFT and a natural extension of paper 2, so its material (multi-frame
fusion, the dynamic-range finding, anthropometry) is available to use here, with citation.
`_groundCoeff.txt` is unclaimed by either and still unused — paper 3 approximates gravity as
camera-Y (`anthro_probe.py:171`); height-above-ground from the shipped ground plane remains
an open lever.

### 11.2 Figures — DONE, see §13.1

`hiride_figures.py` exists and emits six figures. The list below is the original plan, kept because the reasoning about what each figure must carry still holds; §13.1 records what was actually built and the five defects that only became visible once they were rendered.

~~No plotting exists.~~ Four figures carry the argument, and all four are computable from
artifacts already on disk (`report.md`, `stats_final.json`, the prep shards):

- **Fig. 1 — the ladder.** Accuracy vs rung, depth and RGB, with the trivial-cue floor and
  the permutation null as shaded references. This is the 89.4 → 5.4 spine.
- **Fig. 2 — what each condition looks like.** One example frame rendered under `full`,
  `person`, `bg_hole`, `bg_plate`, `silhouette`, `scale_removed`, `sil_scaled`. Without
  this a reader cannot judge the mechanism suite at all. Read straight from the shards with
  `hiride_train.apply_mask_condition`; pick a subject whose mask is clean.
- **Fig. 3 — mechanism heatmap.** Conditions × rungs, one panel per modality, cells shaded
  by accuracy with the R4 column annotated by subject-cluster CI.
- **Fig. 4 — the Z-precision axis.** Accuracy vs bits at R1 and R4 with CI bands, the
  shipped-mask silhouette as a horizontal reference line, and the majority rate marked.

Suggested: `hiride_figures.py`, matplotlib only (venv311 has no seaborn), writing PDF to
`$SCRATCH/hiride2/results/figs/`. **Check matplotlib is present in venv311 first** — the
venv was built for TF and may not have it; if absent, generate figures on the laptop from
the copied JSON rather than installing on the cluster.

### 11.3 Writing — what changes in `paper/paper-depth/main.tex`

The 2023 manuscript is built around the retracted 0.99. The framing decision in §3 stands
and should be re-read before drafting: **open with the trivial-cue floor, not with a
confession.** Three-beat opening, R0 kept inside the ladder table labelled "random frame
hold-out (standard image protocol)", and a methods sentence disclosing the protocol change
— disclose, do not narrate.

Sections needing new text: Methods (the ladder, the eligibility filter, the one
normalisation policy, the fixed controls), Results (§8's four tables), Discussion (the
deployment claim in §0, bounded by n = 28), Limitations (SWAP null by construction; R3
data-starved at ~65 frames/subject; mensa excluded as annotation tracks not verified
identities; in-house room/identity confound; no human-rater study).

**Claims that must not be made** are listed in §3 and have not changed. Two additions from
this campaign: do not call the 1-bit quantisation row "the silhouette" (§8.4), and do not
quote `bg_plate` at R0/R1 as a scene result (§0).

### 11.4 Optional, only if a reviewer demands it

- Efficiency benchmark as a *measured negative result* — §2.8 shows the 2023 efficiency
  claims are arithmetically false; a latency table would replace them with something true.
- Seed top-ups on the headline R4 cells. n = 28 subjects, not seeds, is the binding
  constraint, so this buys little.
- The human-rater study (needs ethics, not compute; the thesis survives without it).

---

## 12. Session record 2026-08-18/19 — the geometry turn

Read this with §0 and §8. It supersedes the reading of §8 that treated depth as
weak: **two interventions found here beat every earlier number**, and six others
were refuted by measurement. The whole section is written so the next agent can
act without re-deriving anything.

### 12.1 The two results that changed the conclusion

**(a) METRIC 3D FEATURES BEAT EVERY CNN WE HAVE TRAINED.** `hiride_metric.py`
unprojects each frame with the Kinect intrinsics (fx=fy=575.816, cx=320, cy=240)
and measures the body in MILLIMETRES: stature, a width profile at six fixed
fractions of stature, depth extent, surface area, volume proxy.
`hiride_metric_floor.py` runs the ladder on them (`metric_floor.json`):

| rung | metric (12 scalars, RF) | image floor | best CNN depth | best CNN rgb |
|---|---|---|---|---|
| R0 frame-random | 92.11 % | 89.38 % | 98.36 | 99.99 |
| R1 block g150 | **67.97 %** | 17.73 % | 69.67 (flatten/stripe) | 96.15 |
| R3 cross-recording | **15.56 %** | 8.45 % | 36.74 | 79.51 |
| **R4 cross-session** | **19.04 %** (per-subj 20.06) | **5.35 %** | 18.01 | 18.53 |

Null 3.58 %, chance 3.57 %, majority 4.47 %. Top cues: `stature_mm`,
`height_p50`, `w_75`, `w_45`, `surface_area_m2`.

The point is not the model, it is the UNITS. The image floor uses pixel counts
and pixel bounding boxes and collapses across a session because pixels are not
camera-invariant; the census measured that collapse (background +620 mm for all
28 shared subjects, people ~30 cm closer, 34 px lower). Millimetres in a
gravity-aligned frame cancel camera pose by construction. `metric+nuisance`
confirms the mechanism from the other side: adding `stand_dist_mm` back DROPS R1
67.97 -> 57.59 and R3 15.56 -> 10.76.

**(b) THE CNN HEAD WAS THE BOTTLENECK FOR DEPTH, AND ONLY FOR DEPTH.**
Wave 10 (`make_runs.py --wave 10`, `--head {gap,flatten,stripe}`, default `gap`
unchanged):

| R1 depth | gap | flatten/stripe | gain |
|---|---|---|---|
| interior_only | 40.92 | **69.59** | +28.7 |
| scale_removed | 47.62 | 66.90 | +19.3 |
| sil_scaled | 52.73 | **69.67** | +16.9 |
| **rgb** scale_removed | 96.33 | 96.15 | **-0.2** |

At R4 the same change takes depth from 12.5-14.6 to 16.5-18.0. The asymmetry is
the evidence: after `scale_removed`, pixel (i,j) is the same body location for
everyone, so spatial ARRANGEMENT is the identity signal -- and global average
pooling answers "is this feature present?" but never "where?". Fine for clothing
texture, fatal for body geometry. The 2023 `Flatten->Dense` head we removed for
parameter-count reasons may have been accidentally right for depth.

**Consequence for the paper's thesis.** At R4, depth 18.01 vs rgb 18.53 --
indistinguishable -- and metric features 19.04 beat both. "RGB is usable, depth
is similarly usable" now holds AT THE CROSS-SESSION RUNG with measurements
behind it. Within a session RGB still wins by ~26 pp, from clothing colour, the
cue that then fails across sessions.

### 12.2 Refuted by measurement — do not re-try these without new reasoning

| lever | result | why it failed |
|---|---|---|
| body-relative depth range (`--depth-slab-mm`, 15x more contrast) | flat: R4 13.9 -> 14.3 -> 11.4 across slab 6000->300 | BatchNorm absorbs a linear input rescale. Paper 3's clip effect was foreground SELECTION, a different operation |
| multi-frame fusion (`--frames 5/10/25`) | monotonically harmful, both modalities | measured sensor sigma is only 6-12 mm (`noise_floor.json`), so there is little noise to remove and the median blurs pose |
| surface normals (`--depth-encoding normals`) | neutral at R1, harmful at R4 (6.4 vs 13.7) | derivatives amplify noise; a 1-px lateral step at 3 m spans ~5 mm |
| closer standing range | R4 interior_only 4.2 % at 0-2000 mm vs 20.9 % at 2500-3000 | FOV clips partial bodies below ~2 m; geometry, not quantisation, dominates |
| rim erosion alone (`interior_only`, gap head) | probe up, CNN down | the CNN was using the boundary GAP left it |
| ImageNet ConvNeXt | +2-5 pp | real but inside the ~6 pp MDE |

**My disparity-quantisation prediction was wrong and was corrected by direct
measurement**: predicted sigma 40-70 mm at 3 m, measured 6-12 mm. BIWI depth is
evidently post-processed, not raw disparity. Measure before building on physics.

### 12.3 Dataset facts learned (cost real time — do not rediscover)

- **`_groundCoeff.txt` and `_skel.txt` are effectively EMPTY** in this
  distribution: parsed on 130/28,037 and 168/28,037 frames respectively, raw
  content `''`, despite 39,281 of each file existing. The metric result above
  therefore used the CAMERA-Y FALLBACK, not the shipped ground plane, and still
  produced a plausible stature (median 1744 mm, p5 1520, p95 1897). Skeleton
  features contributed nothing. Anyone planning to use either file should verify
  emptiness first.
- Paper 1 is PUBLISHED and skeleton-only (20-D bone-segment lengths, 14
  participants). Paper 3 is a DRAFT and a natural extension of paper 2, so its
  material is available to paper 2 with citation (author's call, 2026-08-19).

### 12.4 Bugs fixed this session (each would have corrupted a number)

- `hiride_collate.py` / `hiride_stats.py` grouped cells WITHOUT `frames`,
  `encoding` or `head`, silently merging different experiments into one mean
  (wave 9's `interior_only` first read as n=10 mixing f1 and f10).
- The separability ratio in `hiride_signal.py` crossed the session gap in the
  numerator only, so a common-mode shift inflated it; §8.6's own numbers show
  the rule would have declared the paper's central result impossible. Both pair
  types now cross the gap, and d-prime is reported.
- Its probe was un-whitened nearest-class-mean, blind to low-variance
  directions -- exactly the signal being hunted. Now reports raw / whitened /
  test-centred; **trust `whitened` for a null**.
- `contrast()` selected the whole frame for rgb (background is (-1,-1,-1), so
  `abs().sum() > 0` is always true).
- `np.linalg.svd` on (n x 65,536) made `--pool 1` look hung; replaced with
  Gram-side PCA, verified identical to the dense path.

### 12.5 The paper-2 vs paper-3 FOV contradiction is RESOLVED

Paper 2 is right. Replicating `anthro_probe.py`'s `_foreground` + `_audit_frame`
EXACTLY (slab 1500 mm, LCC, 2-row margin) on 4,000 raw Training frames gives top
99.95 % / bottom 99.98 % / full-body 0.00 %, against the shipped userMap's
3.25 / 13.23 / **85.50 %** on the same frames with the same rule. Overlap with
the person: recall 0.976 but **precision 0.204**, covering 39 % of the frame --
the LCC merges the body with the floor and the wall behind, so the blob
necessarily touches both edges. That also explains paper 3's "height std 16 mm
across 28 people" and "Training<->Testing height correlation -0.04": those are
what measuring the ROOM looks like. **Paper 3's conclusion "cross-session
failure = FOV clipping, not depth" is unsupported by the evidence offered for
it.** Separately, its TRAINING path (600 mm slab, no LCC) has IoU 0.000 with the
person on these frames -- worth checking inside paper 3's own loader.

### 12.6 In-house: scoped and delivered (wave 8, 26/26)

Kept, per the author. `kinect2png.py` restarted `idx` per `.mkv` inside one
output directory, so 26 recordings were overwritten into 10 unbroken index runs
(7,151 frames); only `man1` and `man2` come from a single recording, and `bear`'s
aug-11/mar-22 pair -- a real cross-session contrast -- was destroyed at
extraction. Splice scan found runs coherent, so R0/R1 are sound; no
recording-disjoint rung is possible. No userMap, and slab+LCC finds no person
for ANY identity (the three that passed an area filter have cross-frame mask IoU
0.96-0.98, i.e. a static artefact), so the mechanism suite stays BIWI-only.
Result: depth 93.69 -> 80.81 (per-subj 89.74 -> 67.44), rgb 99.72 -> 81.77
(99.60 -> 68.80). **Quote against the ROOM-ONLY oracle, 55.6 % frame / 30.0 %
per-subject**, not chance -- each directory holds a disjoint set of people.

### 12.7 New code (all additive; every new switch defaults to prior behaviour)

`hiride_metric.py`, `hiride_metric_floor.py`, `hiride_signal.py`,
`hiride_noise.py`, `hiride_fov_check.py`, `hiride_inhouse_{check,probe,prep}.py`,
`hiride_census.py`, `hiride_plates.py`, `run_hiride_night.slurm` (+
`collect_night.sh`). Trainer gained `--head`, `--frames`, `--depth-encoding`,
`--erode`, `--depth-slab-mm`, `--bits`, `--eligibility`, `--track-test`,
`--init`, and the `interior_only` condition. **Waves 2-9 emit byte-identical
runs files**, verified, so concurrent work is unaffected.

### 12.8 Open items, in priority order

1. **Re-run `hiride_metric_floor.py` with subject-cluster CIs** — 19.04 % at R4
   has no interval yet, and n = 28 subjects is the binding constraint.
2. **`n=8` cells in the wave-10 collate** — some keys collected more runs than
   the 5 seeds submitted. Resolve before quoting wave-10 numbers.
3. **`hiride-signal-fair` hit its 2 h wall** partway through R4; R1 completed.
   Re-run with a longer wall for the unbalanced probe-vs-CNN comparison.
4. **Combine the two wins**: metric features are not yet fed to a network, and
   the `flatten`/`stripe` head has not been tried on top of them.
5. ~~**Figures** (§11.2)~~ — done, §13.1.

---

## 13. Session record 2026-08-19/20 — figures, and a filename bug that ate runs

Two things happened: the figure pipeline was built, and building it surfaced a
data-loss bug that had been running since the `--erode` flag was added. Read
13.2 before 13.1 if you are short of time — it is the one that changes numbers.

### 13.1 `hiride_figures.py` — six figures, and what rendering them found

Read-only over `stats_final.json` and `range_profile.json`. Nothing is
recomputed, so a number in a figure and the same number in a table cannot
disagree; that is the whole design constraint. matplotlib **is** present in
venv311 (3.11.1, from the Compute Canada wheelhouse), so §11.2's fallback plan
is moot.

```bash
python hiride_figures.py --stats $SCRATCH/hiride2/results/stats_final.json \
    --range $SCRATCH/hiride2/results/range_profile.json \
    --prep $SCRATCH/hiride2/prep --out $SCRATCH/hiride2/results/figs
```

| fig | what |
|---|---|
| 1 | `rgb − depth` per (arch, condition) cell at each rung — the collapse, as spread |
| 2 | the ladder, both modalities, subject-cluster CI bands |
| 3 | mechanism heatmap, conditions × rungs, one panel per modality |
| 4 | Z-precision axis at R1 and R4 |
| 5 | range profile: linear-probe accuracy and frame-edge clipping on shared axes |
| 6 | one frame under every condition, via the trainer's own `apply_mask_condition` |

**Figure 1 is the centrepiece.** One point per (architecture, condition) cell,
paired by seed on identical test frames, so the visible spread within a rung IS
the robustness argument: **+19.5 / +39.9 / +32.4 pp, then +1.2 at R4**, across
both architectures, every head, augmented and not. (Numbers as of the 08-19
collate; wave 16b will move them slightly.)

**Five defects were found by LOOKING at the rendered output, not by reading the
code.** All five had passed a clean compile and a synthetic smoke test:

1. **Fig 6 was inverted.** It passed `fill=1.0`; `hiride_train.py` passes `0.0`.
   The `sil_scaled` panel came out completely blank — silhouette interior and
   background both land on 1.0 — and every other panel had an inverted
   background. Calling the trainer's real function was supposed to stop the
   panel drifting from what the network sees; the wrong argument gave that up.
   The constant is now `TRAINER_FILL`, with the call site named in a comment.
2. **Fig 2 substituted architectures.** A `fallback_arch` meant R0 and R3 came
   from the gap-head baseline while R1 and R4 came from `stripe/aug8/tf10`,
   under a title naming only the latter — inflating the R1→R3 drop by the head
   effect (~13 pp at R1). Substitution removed; a rung an arm was not run at is
   left empty and reported on stdout.
3. **Fig 2 then interpolated across the gap it had just admitted to**, drawing
   one straight segment R1→R4 through R3's x-position at a value nothing
   measured. Only adjacent rungs are connected now.
4. **Fig 4 stacked two points at 16 bits.** The filter listed `head`,
   `augment`, `frames`, `encoding` — but not `test_fuse`, so
   `alexnet/tf10 scale_removed` (48.20 %) was drawn on the plain gap cell
   (55.40 %). It now matches on the **composed `arch` string**, which cannot
   omit an axis. The duplicate-detection that found this stays in.
5. **Fig 1 weighted saturated cells equally.** `bg_plate` at R0 is 99.97 % vs
   99.91 % — 0.06 pp between two ceilinged cells, carrying the same weight as a
   61 pp one. It pulled the R0 mean to 13.3 and made the collapse look like it
   began a rung early. Cells where both modalities clear 95 % are now drawn open
   and excluded from the mean (R0 mean 13.3 → 19.5).

The lesson worth carrying: **render the figure and look at it.** Every one of
these survived the checks that do not involve eyes.

### 13.2 `--erode` was missing from the run filename, and it destroyed runs

`results_<tag>.json` is the only thing keeping two training cells apart on disk.
`tag` did not encode `--erode`. Every `--erode N` run therefore landed on the
filename of the plain run with the same (policy, modality, arch, condition,
seed) and **overwrote it**. Slurm reported `COMPLETED` for all of it.

The only visible symptom was the total cell count *falling* from 130 to 127
after a wave that added 38 runs. Six cells were destroyed:
`{R1_block,R4_cross_session}_depth_alexnet_interior_only_s{0,1,2}`.

Worse than the loss: array tasks run concurrently, so **the survivor for each
seed was whichever erode value happened to finish last**. At R1 that left `e4`
seed 0 next to `e6` seeds 1–2. The erode sweep as collated before 2026-08-20 was
a random mixture wearing correct-looking labels — which is why `e1` appeared to
be "never run". **Any statement about erosion made before this date rests on
`e6` alone, and not even cleanly.**

`--guard` and `--ref-eligibility` were missing from the tag for the same reason.
Nothing has varied them within one policy yet, so no data was lost there.

Three fixes, all in place:

- **`run_tag(m)`** is one function keyed on the result dict, used both when
  writing a run and when auditing the runs directory, so writer and auditor
  cannot disagree about what makes a cell distinct. Default-valued axes stay out
  of the tag, so every already-correct filename is unchanged.
- **A pre-write check** compares `CELL_FIELDS` against whatever occupies the
  target path and raises rather than overwrites. Re-running an *identical* cell
  stays allowed — that is how a wave tops up seeds without knowing which exist.
  A future missing axis now fails loudly at the first colliding run.
- **`hiride_retag.py`** renames each file to the name its own metadata implies.
  The overwritten runs are gone, but survivors carry correct metadata, so this
  recovered the erode runs under honest names and freed the plain names for
  regeneration. Its report doubles as the re-run list. Run it read-only first.

```bash
python hiride_retag.py --runs $SCRATCH/hiride2/runs           # report
python hiride_retag.py --runs $SCRATCH/hiride2/runs --apply   # rename
```

This is the **seventh** instance in this campaign of an axis not reaching a key
— see §12.8 item 2 for the `n=8` version — and the first that destroyed data
rather than mislabelling it. The pattern is always the same: a hand-maintained
list of fields somewhere that has to be updated when a flag is added, and is
not. Both fixes here replace a hand-maintained list with something derived.

### 13.3 The R1 full-body arm is infeasible, not broken

All 15 tasks of the wave-15 R1 arm (`20144984`) failed with:

> `error: 3 test subjects absent from training (['036', '037', '045']...). A
> closed-set classifier has no output unit for them; this policy is
> identity-disjoint and belongs to paper 3.`

At guard 150 the `--eligibility full_body` filter leaves those three subjects
with **no training frames at all**. The `--ref-eligibility cues` switch added
for this wave fixed the *reference* construction but not the training set.

This is a dataset fact, not a bug — the guard is doing exactly its job.
**Report the full-body arm at R4 only** and state the R1 infeasibility in
Methods. The control exists where it matters: `interior_only stripe full_body`
at R4 has 5 seeds. Lowering the guard would make the arm non-comparable to
every other R1 cell; do not do it to fill the table.

### 13.4 Wave 16 — top-ups

38 cells, both policies, all `alexnet` gap-head depth: `scale_removed` and
`interior_only` at seeds 0–4 (they were sitting at n=2 while printing like any
other row, and they are the baselines the paired condition contrast subtracts
against), plus the erode sweep at `e1`/`e4`/`e6` × 3 seeds. Idempotent by
design — re-running an existing seed rewrites its own file — so it does not need
to know which seeds are missing. First submission `20145586` predates the
filename fix and its erode arm was lost to it; `20165081` is the good one.

### 13.5 A packaging near-miss worth knowing about

`pip install matplot` (an unrelated pyloco wrapper, not `matplotlib`) upgraded
`packaging` from `24.1+computecanada` to PyPI's `26.3` inside venv311 **while 15
training jobs were live**. TensorFlow 2.15.1 was verified to import and fit
after the fact, so nothing broke. The venv rule to follow: **`pip install
--no-deps`** for anything added after the TF install, and if a CC-patched wheel
does get replaced, `pip install --force-reinstall "packaging==24.1+computecanada"`.

### 13.7 Bootstrap intervals depended on processing order

`hiride_stats.py` seeded one generator (`default_rng(--seed)`) and consumed it
sequentially across every cell, so an interval depended on how many OTHER cells
happened to be bootstrapped first. Landing wave 16 — which touched
`scale_removed` and `interior_only`, not `sil_scaled` — moved the `sil_scaled`
bound from +0.19 to +0.10 pp with its own data untouched. Every other row
shifted too.

Fixed: `boot_rng(base_seed, key)` derives an independent stream from the
quantity's own key (blake2b of the cell key), so an interval is a function of
that cell's data and `--seed` and nothing else. Verified: identical interval for
the same key after 40 unrelated cells are bootstrapped in between.

`cluster_boot` is also vectorised — the mean over a resampled subject set is
`sum(correct counts) / sum(frame counts)`, two gathers and two row-sums instead
of concatenating frame arrays per draw. Byte-identical to the loop it replaces
(verified draw-for-draw), and fast enough that `--boot` default went 2000 →
20000, halving the Monte-Carlo spread of the 2.5 % bound from 0.064 to 0.031 pp.

**Every interval in the paper must be regenerated with this version.** Earlier
`stats_final.json` files carry order-dependent intervals.

### 13.6 Open items

1. **Wave 16b (`20165081`) must land before any `interior_only` or erode number
   is quoted.** Then `bash collect_all.sh` and re-render figures.
2. **`sil_scaled` vs `full` at R4: reproducibly positive, but quote the
   interval.** `+7.89 pp`, and the 95 % subject-cluster lower bound across five
   independent bootstrap seeds (0–4, `--boot 20000`) is
   **+0.21, +0.21, +0.15, +0.12, +0.18** — positive every time, spread 0.09 pp.
   It is the only R4 contrast whose interval clears zero, and that statement is
   now stable rather than an artefact of processing order (§13.7 — the earlier
   `+0.10` that suggested otherwise came from the buggy shared RNG).

   So it may be reported as significant. What it must not be is *oversold*: the
   lower bound is ~0.2 pp on a 7.89 pp effect, which is "barely clears zero at
   n = 28 subjects", not a robust margin. Always print the interval next to the
   point estimate, state `--boot` and `--seed` in Methods, and do not let the
   paper's central argument rest on this one row. It does not need to: the
   collapse (+39.9 → +1.2 pp, Fig. 1) and the finding that what survives is
   framing-normalised outline rather than metric depth both follow from the
   ORDERING of conditions across many cells, which no single interval controls.

3. Figures 1–5 are final in form; only their inputs change.
4. §12.8 items 1, 3 and 4 are untouched by this session and still stand.
