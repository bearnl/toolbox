# Depth as a Drop-in Image Modality for Open-Set Person Re-Identification — Paper Handoff

> **Purpose.** Self-contained record of the full research effort (motivation, method,
> every experiment with verified numbers, failures, diagnostics, related work, honest
> novelty assessment, and open decisions) so a paper-writing agent can draft the paper
> from this file alone. All numbers are from our own ComputeCanada runs unless cited.
> `n=` is the seed count; treat `n=1` as **preliminary** and multi-seed as **robust**.

---

## 0. One-paragraph summary

We study whether **depth images can serve as a drop-in alternative to RGB in a single,
modern, off-the-shelf open-set person Re-ID pipeline** (ImageNet-class backbone + BNNeck
+ batch-hard triplet + cross-entropy + k-reciprocal re-ranking + multi-frame probe),
applied identically to both modalities with only modality-necessary preprocessing
differing — **no hand-crafted depth descriptors, no skeleton, no RGB-D fusion**. On BIWI
RGBD-ID (open-set: 22 train identities disjoint from 28 test identities) the same-session
within-test result is **depth Rank-1 ≈ 48% (mAP ≈ 81%) vs RGB ≈ 91% (mAP ≈ 97%)** — depth
usable and RGB-comparable on ranking metrics (mAP 0.83×, R5 0.82×) though not Rank-1
parity under constant clothing. The single largest lever is a **depth-specific
dynamic-range preprocessing** (foreground-slab width 600→300 mm, +13 pp), which has no RGB
analog. We show depth Re-ID is **viable from scratch** (40% R1, ~11× random) and that a
training-free **metric-anthropometry analysis** reaches up to **79% R1** within session,
confirming the identity signal is genuine body geometry. We **diagnose** the cross-session
(cross-clothing) collapse as a **BIWI capture artifact** (100% of Training frames clip the
body at the field-of-view edge), not a limitation of the depth modality.

---

## 1. Motivation & thesis

- **Predecessor (the user's prior work, "HI-RIDE")**: a *classification* approach to RGBD
  person ID, reporting near-ceiling closed-set accuracy (depth ~0.99, RGB ~0.91). Limitation:
  closed-set classification cannot handle identities unseen at training time.
- **This work's step forward**: open-set **metric learning** (siamese/triplet) so the
  embedding generalizes to unseen identities — the deployable Re-ID formulation.
- **Core thesis**: RGB and depth are *both usable* for open-set siamese Re-ID. They encode
  identity through different physics (RGB = appearance: colour/texture/clothing/face;
  depth = 3-D body geometry), so they legitimately require **drastically different
  preprocessing** — but plugged into the *same* pipeline, depth is a viable modality.
- **Framing claim to defend in the paper**: "depth is a drop-in image modality" — we change
  the input format and the necessary preprocessing, *not* the architecture, losses, or
  training recipe. We do not engineer depth-specific descriptors (contrast: Wu 2017's
  Eigen-depth) and do not use skeleton or RGB-D fusion (contrast: Liu 2017, Patruno 2019).

---

## 2. Dataset & evaluation protocol

**BIWI RGBD-ID** (Munaro et al. 2014), Kinect v1, 640×480 depth (mm, uint16).
- `Training/`: 50 subjects, one outfit ("outfit A"), Still + Walking sequences.
- `Testing/`: 28 of those subjects re-recorded on a **different day in different clothes**
  ("outfit B"), with Still + Walking sub-sequences.
- File naming: `022_..._depth.pgm` (Training), `022b_..._depth.pgm` (Testing). Leading
  digits = subject ID; `022` and `022b` are the **same person**, different session/outfit.

**Protocols we use:**
- **Open-set (primary)**: train on the 22 subjects in `Training/` that are **not** in
  `Testing/` (disjoint IDs); evaluate on the 28 `Testing/` subjects (never seen in training).
- **Within-test (same-session, same-clothing)**: gallery = Testing Still, probe = Testing
  Walking — cross-*pose* matching, clothing constant. **This is the primary evaluation.**
- **Cross-clothing / cross-session (hard)**: gallery = the 28 test subjects' `Training/`
  frames (outfit A), probe = their `Testing/` frames (outfit B); identities held out of
  training. This is where everything collapses (Section 4.8).

**Metrics**: Rank-1 (R1), Rank-5 (R5), mAP. **R1 is the meaningful metric** (28 subjects →
R5 is near-saturated, random R5 = 17.9%). **Random R1 = 1/28 = 3.6%.**

**Matching variants reported** (increasing strength):
- FRAME-MATCH: each probe frame → nearest gallery frame.
- PROTOTYPE: each probe frame → nearest of K per-subject mean embeddings.
- SEQ-PROBE: average N=10 consecutive probe frames per subject → nearest prototype.
- `+ k-reciprocal RERANK`: Zhong 2017 re-ranking applied to the above.
- **Headline number = within-test SEQ-PROBE + k-reciprocal RERANK R1.**

**Dataset issues (and our responses):**
1. Only 28 test IDs → R5 near-meaningless; we report R1 + mAP, use multi-frame seq-probe.
2. Within-test is same-clothing → favours RGB's constant clothing cue; we built the
   cross-clothing eval to test the regime that should favour depth.
3. **Training frames clip 100% of bodies at the FOV edge** → cross-session metric transfer
   impossible (diagnosed in 4.8); this is the dataset's fundamental limitation for us.

---

## 3. Method (full pipeline; identical across modalities except preprocessing)

### 3.1 Depth preprocessing (`preprocess_depth_smart`)
1. Load raw depth (mm). `valid = depth > 0`.
2. **Foreground slab (background removal by distance)**: `anchor = 1st-percentile of valid
   depth` (nearest body surface); `fg = valid & (depth ∈ [anchor, anchor + CLIP_RANGE])`.
   **CLIP_RANGE (slab width) is the key hyperparameter — see 4.3.**
3. Crop a square box around the foreground centroid (size from foreground pixel std × expand
   factor); resize to 384×384 (INTER_NEAREST).
4. Normalize foreground to [−1, 1]: `(depth − anchor)/CLIP_RANGE × 2 − 1`; background = −1.
5. Single channel. (Optional augmentation: rotation ±15°, scale 0.7–1.0, translation ±5%,
   flip, additive noise — applied to cached frames at train time.)
- **This background-by-depth-threshold step is impossible for RGB** (no depth to threshold).

### 3.2 RGB preprocessing
- Resize to 384×384, normalize to [−1, 1]. Aggressive augmentation to destroy
  "person = shirt colour" memorization: HSV hue shift ±90, saturation ×0.4–1.6, brightness/
  contrast jitter, **random channel permutation (p=0.5)**, random erasing (Zhong 2017).
- **This colour augmentation is meaningless for depth.** (Justifies "different nature →
  different preprocessing".)

### 3.3 Backbone & 1-channel adaptation
- **ConvNeXt-Tiny** (28 M params), `tf.keras.applications`, ImageNet-pretrained (default) or
  random init (`--no-pretrained`).
- **1-channel depth adaptation** (genuinely single-channel, not replication): build the
  3-channel ImageNet model, build a fresh 1-channel model of the same architecture, copy all
  weights by layer name, and **channel-average the stem conv** kernel `(4,4,3,96)→(4,4,1,96)`.
  Verified: stem kernel is `(4,4,1,96)`; 134 layers copied, 1 kernel-averaged, 0 mismatches.
  → The "1 vs 3 channel" contribution is preserved (architecture is honestly 1-channel).
- Also implemented: ConvNeXt-Small/Base, EfficientNet-B0/B2, ResNet50V2, and a from-scratch
  **SmallResNet** (0.34 M params) baseline.
- Head: GAP → Dense(512) → BN → Dropout → **BNNeck** (Luo 2019): `Dense(emb=256)[pre_bnneck]`
  feeds the triplet loss; `BatchNorm(scale=False)[bnneck]` feeds the classifier and is
  L2-normalized for retrieval.

### 3.4 Losses (`HybridModel`)
- **Batch-hard triplet** (Hermans 2017) on L2-normalized pre-BN features, margin 0.3, cosine
  distance. **PK sampling**: P=8 identities × K=4 frames = batch 32; temporal-stratified
  sampling (K frames span the recording, not adjacent).
- **+ Cross-entropy** on the BNNeck classifier over the 22 training IDs, label smoothing 0.1
  (Luo 2019 strong baseline). Weight 1.0.
- **+ Optional anthropometric MSE aux head** (depth only): predicts pose-mostly-invariant
  body statistics; weight `ANTHRO_WEIGHT` (best 0.3, but effect is marginal — see 4.5).
- Regularization: L2 weight decay 1e-4.

### 3.5 Inference-time enhancements
- **Multi-frame seq-probe averaging** (window 10): group consecutive same-subject probe
  frames, average + renormalize embeddings → "video probe".
- **k-reciprocal re-ranking** (Zhong 2017, k1=20, k2=6, λ=0.3): applied to the probe×gallery
  (and probe×prototype) distance matrices at eval. Big lift for RGB; helps depth once the
  embedding has top-K structure.
- **Temporal stack (depth only)**: stack 3 consecutive frames at stride s as the 3 input
  channels (single sensor, adjacent timestamps — stays "depth alone"); uses ConvNeXt's
  native 3-channel ImageNet weights directly (no channel-averaging). Small gain (4.2).

### 3.6 Training config
- Image 384², emb 256, Adam LR 1e-4 (ReduceLROnPlateau), 30 epochs, EarlyStopping on
  within-val R1 (patience 8). ComputeCanada Nibi, H100 MIG slices (2g.20gb / 3g.40gb),
  ~64 GB host RAM, dataset extracted from `.rar` to node-local `$SLURM_TMPDIR` per job.

---

## 4. Results (all within-test SEQ-PROBE+RERANK R1 unless stated; random R1 = 3.6%)

### 4.1 Headline — depth vs RGB (same pipeline, ConvNeXt-Tiny + ImageNet, clip=300)

| Modality | R1 | R5 | mAP | n |
|---|---|---|---|---|
| RGB   | 90.7 ± 1.9 | 96.5 ± 0.6 | 96.8 ± 0.6 | 5 |
| Depth | 47.6 ± 4.6 | 79.1 ± 2.5 | 80.7 ± 1.9 | 15 |
| **D/RGB ratio** | **0.52** | 0.82 | **0.83** | |

Single-config peaks (n=1): depth **52.6** (no aux), **54.9** (+temporal stack 3×2).

### 4.2 The climb (progression, all open-set 384², depth)

| Configuration | Depth R1 | n |
|---|---|---|
| Triplet+BNNeck, SmallResNet (scratch), clip 600 | 25.0 ± 1.3 | 5 |
| + ConvNeXt-Tiny, ImageNet, clip 600 | 32.8 ± 2.2 | 11 |
| + anthropometric aux (w=0.3), clip 600 | 34.7 ± 1.7 | 5 |
| **+ tighter slab (clip 300)** | **47.6 ± 4.6** | 15 |
| + temporal stack 3×2 (single config) | 54.9 | 1 |
| RGB (best, same pipeline) | 90.7 ± 1.9 | 5 |

Pre-modern pair-contrastive baseline ≈ 16% (closed-set 256² "auto" protocol — *different
protocol; mention in prose only, not in the comparable table*).

### 4.3 Foreground-slab (dynamic-range) ablation — the dominant lever (ConvNeXt+ImageNet, aux 0.3)

| Slab width (mm) | Depth R1 | n |
|---|---|---|
| 600 | 34.7 ± 1.7 | 5 |
| 400 | 46.8 | 1 |
| **300** | **47.6 ± 4.6** | 15 |
| 200 | 41.2 | 1 |

Mechanism: body ≈250 mm deep; 600 mm slab → body occupies ~42% of input range; 300 mm →
~83%, ~2× the Z-resolution on the body. **+12.9 pp** from a single preprocessing constant.
Unimodal optimum at 300; 200 clips limbs, 400 wastes range. **No RGB analog.**

### 4.4 Architecture × initialization 2×2 (depth R1, clip=300, aux 0.3)

| Backbone | From scratch | ImageNet transfer |
|---|---|---|
| SmallResNet (0.34 M) | 36.0 ± 2.9 (n=5) | — (no pretrained weights) |
| ConvNeXt-Tiny (28 M) | 40.2 ± 2.3 (n=5) | 47.6 ± 4.6 (n=15) |

- **From scratch is viable**: 40.2% R1, ~11× random.
- Transfer adds **+7.4 pp** (not essential; scratch reaches 85% of transfer).
- Architecture adds **+4.2 pp** from scratch (ConvNeXt vs SmallResNet).
- The clip-300 effect is **initialization-independent**: SmallResNet scratch clip 600→300
  = 25.9 → 36.0 (+10.1 pp).
- Caveat: from-scratch ConvNeXt peaks at epoch 1 then overfits (28 M params on ~10 k frames);
  result needs early stopping (5-seed mean still stable, ±2.3). Transfer acts as a regularizer.

### 4.5 Anthropometric-aux weight sweep (ConvNeXt+ImageNet, clip 600)

| weight | Depth R1 | n |
|---|---|---|
| 0.0 | 32.8 ± 2.2 | 11 |
| 0.1 | 32.8 | 1 |
| 0.3 | 34.7 ± 1.7 | 5 |
| 0.5 | 32.7 | 1 |
| 1.0 | 32.5 | 1 |

Only 0.3 helps (+1.9 pp, marginal, p≈0.13); delays overfit (best_epoch 7.7→12.8). **At
clip=300 the aux effect is within noise / possibly negative** (no-aux single seed = 52.6 vs
aux-0.3 n=15 = 47.6). Recommendation: report as neutral or drop from the headline config.

### 4.6 Failed / neutral interventions (depth)

| Intervention | Depth R1 | vs baseline | Why it failed |
|---|---|---|---|
| Capacity ↑ (SmallResNet w32→w64) | 25.0→25.9 | ~0 | depth capacity-saturated on 10 k frames |
| Capacity ↑ (ConvNeXt Tiny→Small) | 32.8→31.8 (n=1) | ~0 | same |
| 2D pose canonicalization (clip 600) | 26.8 (n=1) | −6 | limb pose pollutes 2D PCA + interp noise; erases postural cues |
| 3D pose canonicalization (clip 600, aux) | 23.7 (n=1) | −9 | re-projection holes; overfit (best_ep=2) |
| 3D anthropometric features (clip 600, aux) | 33.2 (n=1) | ~0 | redundant once clip=300 exposes Z to the CNN |
| RGBD fusion | — | — | excluded by design (depth-alone goal) |

### 4.7 Training-free metric-anthropometry diagnostic (no learning; `anthro_probe.py`)

Hand-crafted body measurements (height, segment widths, **front-back thickness**, volume in
mm via Kinect intrinsics — **no skeleton, no CNN**), prototype NN matching.

**Within-session, full-body frames (12 subjects, random 8.3%):**

| Feature set | frame R1 | seq R1 |
|---|---|---|
| All 12 (camera frame) | 66.9 | 91.7 |
| All 12 (canonical / azimuth-invariant) | 79.0 | 83.3 |
| Height only (camera) | 74.2 | 61.5 |
| Shape only, no height (canonical) | 79.0 | 83.3 |

→ **Body geometry alone reaches up to 79% frame / 92% seq R1**, confirming depth carries
strong clothing-independent identity; **thickness + height** (the cues a 2-D silhouette
cannot capture) are the carriers (highest between/within-subject variance ratios).
Including FOV-clipped frames drops this to 33% frame R1 — clipping halves the signal even
within session.

### 4.8 Cross-session collapse — a dataset artifact, not a modality limit

**Cross-clothing SEQ-PROBE+RERANK (gallery outfit A, probe outfit B, unseen IDs):**

| Method | R1 | mAP |
|---|---|---|
| CNN depth (clip 300) | 5.3 ± 2.7 | 57.6 ± 1.0 |
| CNN RGB (clip 300) | 10.2 ± 4.1 | 61.0 ± 1.5 |
| Hand-crafted anthropometry | ~1.4 | ~12 |

All collapse toward random (3.6%). **Height audit** (28 subjects, `--full-body-only`):
- **100% of Training frames clip the body at top AND bottom edge**; **0 of 13,426 are
  full-body.** Measured "height" std across 28 distinct people = **16 mm** (FOV-saturated).
- Same-person height correlation Training↔Testing: **Pearson −0.04, Spearman −0.15** —
  measurements do not transfer across sessions.
- Azimuth (viewpoint) canonicalization does **not** recover it (camera ≈ canonical, both
  ~random), ruling out viewpoint angle as the cause.
- **Conclusion**: cross-session failure = BIWI's tight Training framing (FOV clipping), not
  depth. Prior depth-only work reaches 30–60% cross-clothing on BIWI, so the modality can do
  it; the standard BIWI Training capture precludes consistent metric measurement *for us*.

---

## 5. Story arcs / contributions (for intro + section ordering)

1. **Depth as a drop-in modality** — unified pipeline, swap the input, only preprocessing
   differs (Sec 3, 4.1).
2. **Dynamic-range preprocessing is the dominant lever** (clip 600→300, +13 pp, no RGB
   analog) — the strongest genuinely-new empirical nugget (4.3).
3. **From-scratch viability + clean architecture×init attribution** (4.4) — directly engages
   Karianakis's "depth needs RGB transfer" premise; we show preprocessing, not transfer,
   carries most of the gain.
4. **Training-free proof that body geometry is the signal** (up to 79% R1) (4.7).
5. **Diagnosis that the cross-session ceiling is a dataset capture artifact** (FOV clipping),
   not a modality limit (4.8) — an honest, useful negative result.

---

## 6. Related work & positioning (verified via web search; cite primary papers)

| Work | Cue | BIWI Rank-1 (per survey) | Relation to us |
|---|---|---|---|
| Munaro 2014 (skeleton) | skeleton | 26.6 Still / 21.1 Walk | dataset creators; we exclude skeleton |
| Munaro 2014 (point-cloud/3D) | depth geometry | 32.5 / 22.4 | hand-crafted geometry |
| **Wu 2017** (Eigen-depth, DVCov) | **depth only** | **30.5** | **closest depth-only; hand-crafted descriptors (we use a general CNN pipeline, no bespoke descriptors)** |
| **Karianakis 2018** (split-rate RGB→depth transfer + temporal attention) | depth (RGB-transferred) | — | **closest method; we show from-scratch viability and isolate preprocessing vs transfer** |
| Hafner 2018/2021 (cross-modal distillation) | RGB↔depth | ~59.8 (depth–depth) | uses RGB at train time (distillation); not depth-alone |
| Liu 2017 (feature funnel) | **RGB+skeleton** | 91.4 | fusion+skeleton upper bound we forgo |
| Patruno 2019 (skeleton std postures) | **skeleton** | 97.84 | skeleton upper bound we forgo |

**Positioning paragraph (honest):** depth-only Re-ID on BIWI is established (Wu 2017;
Karianakis 2018); 90%+ results use skeleton or RGB+skeleton fusion, which we deliberately
exclude. We differ by using a *unified, modern, off-the-shelf image pipeline with no bespoke
depth descriptors and no skeleton*, and by identifying **preprocessing (dynamic range)
rather than pretraining** as the dominant factor — not previously reported. **Protocols
across these works vary** (closed/open-set, same/cross-session); a number-to-number table
requires matching their protocol (Wu has public code: github.com/wuancong/depth_reid).

**References (verify/format):**
- Wu, Zheng, Lai. *Robust Depth-Based Person Re-Identification.* IEEE TIP 2017. arXiv:1703.09474. Code: github.com/wuancong/depth_reid
- Karianakis, Liu, Chen, Soatto. *Reinforced Temporal Attention and Split-Rate Transfer for Depth-Based Person Re-ID.* ECCV 2018. arXiv:1705.09882
- Hafner et al. *Cross-Modal Distillation for RGB-Depth Person Re-ID.* arXiv:1810.11641 (CVIU 2021)
- Munaro et al. *One-shot person re-identification with a consumer depth camera* / BIWI RGBD-ID, 2014.
- Liu et al. 2017 (feature funnel); Patruno et al. 2019 (skeleton standard postures).
- Survey: *Person Re-ID with RGB-D and RGB-IR Sensors: A Comprehensive Survey*, Sensors 23(3):1504, 2023 (PMC9919319).
- Methods cited for the pipeline: Hermans et al. 2017 (in-defense triplet); Luo et al. 2019 (BNNeck strong baseline); Zhong et al. 2017 (k-reciprocal re-ranking).

---

## 7. Honest novelty assessment (read before claiming SOTA)

- The **task** (depth-only Re-ID on BIWI), the **thesis** (depth body-shape is
  clothing/illumination-invariant), and **open-set metric learning** are all pre-existing.
- We **do not beat SOTA**: depth-only SOTA is 30–60% (Wu, Hafner); skeleton/fusion 91–98%.
  Our same-session ~48% is on an easier protocol; cross-clothing ~5% is below depth-only SOTA
  (blamed on the diagnosed FOV clipping).
- **Defensible contributions** (modest, honest): (a) the dynamic-range preprocessing finding;
  (b) clean architecture×init ablation showing from-scratch viability; (c) training-free
  anthropometry analysis attributing the signal to body geometry; (d) the FOV-clipping
  dataset diagnosis. **Best framed as a focused empirical study / short paper**, or a journal
  extension of HI-RIDE (classification vs open-set metric learning + preprocessing analysis),
  **not** a SOTA claim.

---

## 8. Limitations (for the limitations section)
- Single dataset (BIWI); 28 test IDs (R5 saturated; R1 the meaningful metric).
- Cross-session/cross-clothing not demonstrable here (BIWI Training FOV-clips 100% of bodies).
- From-scratch large-backbone overfits early on ~10 k frames; depends on early stopping.
- Same-clothing within-session protocol structurally favours RGB; depth's clothing-invariance
  advantage is untestable on this dataset.
- Anthropometric aux loss effect is marginal and not robust at the best clip setting.

---

## 9. Implementation / reproducibility

- **Repo files** (`F:\workspaces\toolbox\`): `siamese.py` (main: loaders, PK sampler,
  HybridModel, backbones + 1-ch adapter, eval incl. seq-probe/rerank/cross-clothing,
  anthropometric aux, temporal stack); `collate_results.py` (aggregates `results_*.json`);
  `run_biwi.slurm` (job template, .rar→$SLURM_TMPDIR staging); `submit_sweep.sh` (multi-seed
  launcher with all env-var knobs); `anthro_probe.py` (training-free diagnostic: camera +
  azimuth-canonical metric anthropometry, consistency report, height audit, `--full-body-only`).
- **Key flags / env vars**: `BACKBONE`, `PRETRAINED` (1/0), `CLIP_RANGE` (mm), `ANTHRO_WEIGHT`,
  `TEMPORAL_STACK_FRAMES`/`STRIDE`, `PROTOCOL=open-set`, `IMG_SIZE=384`, `PK_P=8 PK_K=4`,
  `TRIPLET_MARGIN=0.3`, `EMB_DIM=256`, `PROBE_WINDOW=10`.
- **Kinect v1 intrinsics used** (anthro_probe): fx=fy=575.816, cx=320, cy=240 (640×480).
- **Headline reproduce**: `MODALITY=both BACKBONE=convnext_tiny CLIP_RANGE=300 N_SEEDS=5 bash submit_sweep.sh`.

---

## 10. Open decisions for the paper writer

1. **Headline depth number**: use robust multi-seed **47.6 ± 4.6 (n=15)** as the main figure;
   cite 52.6 (no-aux) and 54.9 (temporal stack) as "up to ~55% in single-seed configs" unless
   multi-seeded. (Recommended: run a clean 5-seed **clip=300, no-aux, no-tstack** depth+RGB to
   anchor the headline on the simplest config with tight error bars.)
2. **Anthro aux**: present as neutral/marginal or omit from the main config (it's within noise
   at clip=300, possibly negative).
3. **SOTA comparison table**: only if protocol-matched (e.g., run Wu's public protocol);
   otherwise present prior work as context with explicit protocol caveats, not as beaten baselines.
4. **Venue framing**: empirical-study / short-paper, or HI-RIDE journal extension — *not* a
   SOTA paper.

---

## 11. Verified result appendix (raw multi-seed means for tables)

Within-test SEQ-PROBE+RERANK (depth):
- SmallResNet w32 scratch clip600: R1 25.04±1.30, mAP 70.51 (n=5)
- ConvNeXt-Tiny ImageNet clip600 (no aux): R1 32.79±2.22, mAP 73.82 (n=11)
- ConvNeXt-Tiny ImageNet clip600 aux0.3: R1 34.67±1.69, mAP 74.33 (n=5)
- ConvNeXt-Tiny ImageNet clip300 aux0.3: R1 47.56±4.60, mAP 80.72 (n=15)
- ConvNeXt-Tiny scratch clip300 aux0.3: R1 40.19±2.26 (n=5)
- SmallResNet w64 scratch clip300 aux0.3: R1 36.00±2.87 (n=5)

Within-test SEQ-PROBE+RERANK (RGB):
- SmallResNet w32 clip600: R1 68.33±6.70 (n=5)
- ConvNeXt-Tiny ImageNet clip600: R1 91.23±2.35, mAP 96.96 (n=5)
- ConvNeXt-Tiny ImageNet clip300: R1 90.73±1.94, mAP 96.80 (n=6)

Cross-clothing SEQ-PROBE+RERANK:
- depth ConvNeXt clip300 aux0.3: R1 5.34±2.73, mAP 57.57 (n=15)
- RGB ConvNeXt clip300: R1 10.15±4.09, mAP 61.04 (n=6)

Hand-crafted anthropometry (no training), within-Still full-body, 12 subj (random 8.3%):
- canonical all-12: frame R1 79.0, seq R1 83.3; camera all-12: frame 66.9, seq 91.7.

Height audit (cross-clothing, 28 subj): A_height 2012±16 mm, B 2160±117 mm, |Δ| 161 mm;
Pearson −0.045, Spearman −0.151; edge-touch A head/feet 100%/100%, B 84%/45%;
full-body Training frames 0/13426.
