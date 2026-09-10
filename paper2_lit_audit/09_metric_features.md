# 09 — The twelve metric features: definitions, selection, error model, lineage

Status legend: **[M]** measured in this campaign (handoff section or results JSON named); **[C]** read directly from code; **[V]** literature item confirmed on Crossref/DOI record or in a saved primary-text extract; **[V-sec]** confirmed only through a secondary source; **[UNVERIFIED]** not confirmed; **[inferred]** my inference from the record, stated as such.

Sources read: `HIRIDE_HANDOFF.md` §12.1–12.5, §13.10, §13.12–13.18, §14.2, §14.6, §14.8; `hiride_metric.py`, `hiride_metric_floor.py`, `hiride_metric_bias.py`, `hiride_metric_check.py`, `hiride_range_profile.py`, `hiride_fuse.py` (`select_columns`, `load_metric`), `hiride_sequence.py`, `hiride_data.py` (`eligible_mask`), `hiride_train.py` (`--aux`); results `metric_floor.json`, `metric_floor_fbtest.json`, `metric_check.json`, `range_profile.json`, `noise_floor.json`, `sequence_gated_best.json`, `sequence_ungated_best.json`, `tables.tex`; extracts `barbosa.txt`, `andersson.txt`; audit reports 03 and 08_refute_C3C4 for the Munaro/Paolanti rows.

---

## 1. From a depth frame to a feature vector — the common pipeline [C]

Everything below is `hiride_metric.frame_features(depth, user, ground)`, one call per frame, on the RAW BIWI tree.

1. **Person pixels.** `m = (user > 0) & (depth > 0)`. `user` is the shipped `_userMap.pgm` — the Microsoft Kinect SDK v1 player-index map produced online at acquisition (§14.2, verified against the dataset page and the Munaro 2014 chapter). Any non-zero player index counts; `depth == 0` (no return) is excluded. Frames with fewer than 200 person pixels return `None` and are dropped. Upstream, `eligible_mask` (`hiride_data.py`) has already removed frames with fewer than 2,000 person pixels or a person covering more than 25 % of the 640×480 frame; this is applied identically to every condition, CNN and metric alike.
2. **Unprojection.** Pinhole model with `FX = FY = 575.816`, `CX = 320`, `CY = 240`: `X = (x − CX)·z/FX`, `Y = (y − CY)·z/FY`, `z` = raw depth in millimetres. The 575.816 px focal length is the OpenNI/PCL default for the Kinect v1 depth camera at VGA [UNVERIFIED as to provenance; it is the constant the code uses and the handoff quotes throughout].
3. **Vertical axis.** If a ground plane `(a, b, c, d)` was parsed from the frame's `_groundCoeff.txt`, `h = P·n̂ + d/|n|`, sign-flipped so that the median body height is positive; `ground = 1`. Otherwise the fallback: `n̂ = (0, −1, 0)` (camera −Y), `h = P·n̂ − p0.5(h)`; `ground = 0`.
4. **Other axes.** `fwd` = camera Z with its vertical component removed, normalised; `right = n̂ × fwd`. `lat = P·right`, `dep = P·fwd`. In the fallback these reduce to `lat = −X`, `dep = Z`.
5. **Robust statistics throughout.** Extents are percentile differences (p99.5 − p0.5 for height, p99 − p1 for depth extent, p97.5 − p2.5 for widths), so a handful of stray mask pixels at the boundary do not set the value; a mask edge that is systematically wrong (erosion, a merged floor) does.
6. **Classifier.** `RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)` on the raw (unscaled) 12 columns; 3 seeds; per-(rung, set, seed) subject-cluster bootstrap with 20,000 draws (`hiride_metric_floor.py`). In `hiride_sequence.py` the same RF's `predict_proba` is averaged over W consecutive frames of one recording; the `geo` fusion is the renormalised product of CNN and RF posteriors.

### 1.1 The ground-plane caveat — must be settled before the paper says "gravity-aligned"

§13.14 describes the frame as "GRAVITY-ALIGNED … using the per-frame ground plane". §12.3, written when the features were first computed, records the opposite: `_groundCoeff.txt` parsed on **130 of 28,037 frames**, so the metric result "used the CAMERA-Y FALLBACK, not the shipped ground plane" and still gave a plausible stature (median 1744 mm, p5 1520, p95 1897) [M, §12.3]. §13.16 later found that `sibling()` derives the companion filename by re-using the DEPTH stream's timestamp, and each BIWI stream carries its own, so the derived name "usually does not exist" — that diagnosis was made for `_skel.txt` (168/28,037), but `hiride_metric.py` still builds the groundCoeff path with the same `sibling()` call [C]. Nothing in §13 or §14 records a fix or a re-run with the ground plane, and §13.13 names "probably the ground plane" as a still-open third defect.

**Inference [inferred, high confidence]:** the published 12 features were computed in the camera frame with `h = −Y`, i.e. camera translation and standing distance are cancelled (the unprojection does that) but camera **pitch is not**. A pitched camera mixes Y and Z into `h`, which is a scale error in the height axis that grows with the depth extent of the body — consistent with the §13.13 observation that crown-anchored drift rises with band offset. The features are still metric (millimetres) and still transfer (stature Pearson 0.848 across sessions, §5.4 below), but the paper should say "camera-frame millimetres, unprojected; the vertical axis is the camera's −Y" unless the check below shows otherwise.

**Check (one line on the cluster, seconds):**
`python -c "import numpy as np; z=np.load('$SCRATCH/hiride2/prep/metric_features.npz'); n=list(z['names']); print(z['feats'][:, n.index('ground')].mean())"` — the fraction of frames on which a real plane was used. If it is ≈0.005 the fallback stands. If the paper wants the true gravity frame, glob the frame prefix for `groundCoeff` (the fix §13.16 prescribes for `skel`) and recompute; every metric number would then need re-running.

---

## 2. The twelve `BASE_METRIC` columns

Pinned tuple in `hiride_metric.py` [C]: `depth_extent_mm, height_p05, height_p50, stature_mm, surface_area_m2, volume_proxy_l, w_15, w_30, w_45, w_60, w_75, w_90`. Headline numbers on exactly these 12: R4 19.04 % [14.00, 24.40] (5,642 test frames, 7,894 training frames, 28 classes, chance 3.57 %, majority 4.47 %, permutation null ≈3.0–3.6 %); full-body test frames 28.06 % [21.29, 34.66] (2,933 frames); gated W = 25: 43.30 % [30.3, 56.1] on 103 decisions [M, `metric_floor.json`, `metric_floor_fbtest.json`, `sequence_gated_best.json`].

`drift/sig` below is §13.12's figure: |global fixed-effects slope per metre of standing distance| × 2.2 m (the span of `Testing/Walking`) ÷ between-subject SD. Below ~0.5 is usable; above 1 the feature moves further with range than people differ. Between-subject SDs are recorded in the handoff for four columns only; the rest are marked "not recorded".

### Summary table

| # | column | definition (in the body frame of §1) | units | anthropometric target | drift/sig [M] | between-subject SD [M] |
|---|---|---|---|---|---|---|
| 1 | `stature_mm` | p99.5(h) − p0.5(h) | mm | standing height | 1.87 | 105.2 mm |
| 2 | `height_p50` | p50(h) − p0.5(h) | mm | height of the body's median pixel: vertical mass distribution / leg–torso proportion | 2.11 | 67.4 mm |
| 3 | `height_p05` | p5(h) − p0.5(h) | mm | height of the lowest 5 % of body pixels: foot/ankle mass | 1.92 | not recorded |
| 4 | `depth_extent_mm` | p99(dep) − p1(dep) | mm | whole-body front-to-back thickness | 1.54 | not recorded |
| 5 | `w_15` | p97.5(lat) − p2.5(lat) in the band 0.11–0.19 × stature above base | mm | lower-leg width (both shins, stance-separated) | **0.04** | 49.9 mm |
| 6 | `w_30` | same, band 0.26–0.34 | mm | knee / lower-thigh width | **0.25** | 38.5 mm |
| 7 | `w_45` | same, band 0.41–0.49 | mm | thigh / hip width, hands hanging at this level | 1.26 | not recorded |
| 8 | `w_60` | same, band 0.56–0.64 | mm | hip–waist width incl. forearms | 2.07 | not recorded |
| 9 | `w_75` | same, band 0.71–0.79 | mm | chest / upper-arm width ("shoulders" in the handoff) | **2.34** | not recorded |
| 10 | `w_90` | same, band 0.86–0.94 | mm | head / neck width | 0.93 | not recorded |
| 11 | `surface_area_m2` | Σ over person pixels of (z/FX)(z/FY) / 10⁶ | m² | frontal projected area of the visible body — build proxy | 1.55 | not recorded |
| 12 | `volume_proxy_l` | `surface_area_m2` × `depth_extent_mm` | litres (m²·mm) | bounding-slab volume — build proxy | 1.78 | not recorded |

Feature importances (RF, R4): the top cues are `stature_mm`, `height_p50`, `w_75`, `w_45`, `surface_area_m2` [M, §12.1]. Note that three of those five are among the four worst-drifting columns.

The band-to-anatomy labels for `w_XX` are my reading against standard stature-fraction landmarks (knee ≈0.29 H, hip ≈0.53 H, wrist with hanging arms ≈0.49 H, elbow ≈0.63 H, shoulder ≈0.82 H, chin ≈0.87 H — Drillis & Contini segment proportions [UNVERIFIED]); the handoff's own labels are "shoulders" for `w_75` and "neck" for `w_90`. State them as approximate; the bands are 8 % of stature tall (~140 mm on a 1750 mm adult) and are anchored at the visible base, so on a clipped frame none of these labels holds (see §5.3).

### 2.1 `stature_mm`
- **Definition/units.** p99.5 − p0.5 of `h`, millimetres. On a full-body frame this is sole-to-crown height as seen by the sensor, minus whatever the top percentiles trim (hair, a hat).
- **Computation.** Percentiles over all person pixels after unprojection; no fitting, no landmark.
- **Why included.** Stature is the single most stable soft biometric a body has and the one every predecessor found most informative (Barbosa 2012 `d1`/`d3`; Andersson 2015 height; Paolanti 2018 `d1`) — §6. It is clothing-invariant except for footwear and headwear, and day-invariant to within a few millimetres of diurnal change. Top RF importance.
- **Failure mode.** Frame clipping. When the feet leave the frame, p0.5 becomes mid-shin and the stature reads short; the measured global slope is **89.5 mm per metre** of standing distance against a between-subject SD of 105 mm, i.e. ~197 mm across the 2.2 m walk, 1.9× the identity signal [M, §13.12]. Camera pitch (unresolved ground plane, §1.1) adds a scale error. Hair/headwear sit inside the top 0.5 % and are mostly trimmed. Cross-session median bias −45 mm (Walking minus Training, per-subject medians) [M, `metric_check.json`], consistent with the Testing session being closer and more clipped.

### 2.2 `height_p50`, 2.3 `height_p05`
- **Definition/units.** p50 (resp. p5) of `h` minus p0.5 of `h`, millimetres.
- **Computation.** Same percentile pass as stature.
- **Why included.** They summarise the vertical distribution of body mass without a skeleton: a person with long legs relative to the torso has a higher median-pixel height at a given stature; `height_p05` is the height of the lowest 5 % of pixels (feet and ankles on a full body), which encodes foot size and stance. They were the cheapest skeleton-free proxies for the limb-proportion cues that paper 1 owns (§13.16) and that this paper may not compute from `_skel.txt`.
- **Failure mode.** Both are anchored at the visible base and scale with the visible height, so they inherit the clipping drift in full: drift/sig 2.11 and 1.92, the two worst after `w_75` [M]. `height_p50` also moves with arm pose (raised arms lift the median) and gait phase. On clipped frames `height_p05` is a slice of shin, not the feet.

### 2.4 `depth_extent_mm`
- **Definition/units.** p99 − p1 of `dep`, millimetres: the front-to-back extent of the visible surface along the camera's viewing axis.
- **Computation.** Percentiles of the projected `dep` coordinate; since the sensor only sees the near surface, this is not chest depth but the spread between the nearest and farthest visible body points (a leading foot or swinging hand to the trailing shoulder).
- **Why included.** One whole-body thickness number; thickness is the axis a silhouette cannot see and the one the sensor measures directly. In the shape block the per-band thicknesses `ht_ddd` turned out to be the most range-stable family of all 47 columns (0.10–0.91) [M, §13.13], which retrospectively justifies having a thickness axis at all.
- **Failure mode.** Pose-dominated: during walking the leading limbs extend `dep` by hundreds of millimetres, so the value swings with gait phase far more than between people. Range-dependent too (drift 1.54): at close range clipping removes the feet, at long range quantisation coarsens the far tail. Uses p1/p99 rather than p2.5/p97.5, so it is more sensitive to mask stragglers than the widths.

### 2.5–2.10 `w_15` … `w_90`
- **Definition/units.** For fraction f ∈ {0.15, 0.30, 0.45, 0.60, 0.75, 0.90}: select pixels with `base + stature·(f − 0.04) ≤ h ≤ base + stature·(f + 0.04)`; width = p97.5 − p2.5 of `lat` over that band; millimetres. If the band holds ≤ 30 pixels the value is written as **0.0** (a sentinel the RF reads as a measurement — a known code defect, not a design choice) [C].
- **Computation.** `base = p0.5(h)`; `stature` as in §2.1. So each band's position depends on two quantities that are themselves clipping-sensitive.
- **Why included.** A width profile at fixed fractions of stature is the skeleton-free version of "shoulder breadth, hip breadth, knee breadth": it captures frame size independent of colour and, at 6 heights, a coarse body-shape silhouette in metric units. Widths near the feet (`w_15`, `w_30`) barely need the vertical axis and turn out range-stable (0.04, 0.25) — the only two of the twelve below the 0.5 line [M].
- **Failure mode (the documented defect of this block).** The band position error is `stature_error × f` and the width error is that times the width gradient at that height. Hence `w_75`, on the steep shoulder/chest gradient, is worst (2.34); `w_60` 2.07; `w_45` 1.26; `w_90` sits on the slowly varying neck/head and drifts less (0.93) despite the largest position error; `w_15`/`w_30` sit beside the anchor and barely move [M, §13.12]. This is a positional artefact of anchoring at the visible base, not a sensor limit — which is what the crown-anchored shape block was written to fix. Pose adds to it: arm swing enters `w_45`–`w_75` because hanging hands sit near 0.49 H and elbows near 0.63 H; leg separation during stance enters `w_15`/`w_30`. Clothing: a coat widens `w_60`–`w_75` by tens of millimetres, so these are the least clothing-invariant of the set; Barbosa 2012 already noted heavy clothes bias the measurements [V, extract].

### 2.11 `surface_area_m2`
- **Definition/units.** Σ_pixels (z/FX)(z/FY), converted mm² → m². Each person pixel contributes the physical area of its footprint at its own depth; at 3 m a pixel covers ≈ 27 mm², so a 25,000-pixel person reads ≈ 0.68 m².
- **Computation.** Mask and depth only — independent of the ground plane and of any percentile.
- **Why included.** A metric build proxy (frame size × stature) that a pixel count cannot give, because pixel count falls as 1/z² and the census showed the test session ~30 cm closer. It converts the strongest pixel-space cue of the image floor (person pixel count) into a camera-invariant quantity.
- **Failure mode.** Anything that removes pixels removes area: clipping (drift 1.55), mask erosion, tracker loss on limbs. Arm pose changes the visible frontal area (arms akimbo). Loose clothing inflates it. It is the frontal projection, so a turned body reads smaller — BIWI Walking is roughly frontal, Training includes rotation.

### 2.12 `volume_proxy_l`
- **Definition/units.** `surface_area_m2 × depth_extent_mm`; m²·mm = litres.
- **Computation.** Product of §2.11 and §2.4.
- **Why included.** A bounding-slab volume as a second build proxy, combining the two axes the sensor sees. Named "proxy" in the code; it is not a body volume.
- **Failure mode.** Multiplies the failure modes of its factors (drift 1.78) and is dominated by the pose swing of `depth_extent_mm`. Redundant with its factors for a tree model; kept because the set was pinned before this was examined.

---

## 3. The six nuisance/validity columns (excluded from `metric`) [C, §13.14]

| column | meaning | role |
|---|---|---|
| `stand_dist_mm` | median z of person pixels | THE range covariate. As a feature it is a recording shortcut: `metric+nuisance` DROPS R1 67.97 → 57.59 and R3 15.56 → 10.76, and is neutral at R4 (19.27) [M, `metric_floor.json`]. The drift model regresses on the equivalent `p_med` from `cues.npz` [inferred from usage; cues prep not read]. |
| `ground` | 1 if a real plane was used, 0 if camera −Y | the §1.1 check |
| `n_points`, `valid_frac` | person pixel count; fraction of the frame with valid depth | quality covariates |
| `top_clip`, `bot_clip` | mask touches row 0 / row 479 | validity flags. `bot_clip` is the one that matters; the full-body gate is `top_touch ≤ 0 & bot_touch ≤ 0` on the cue equivalents (`eligible_mask(full_body=True)`) |

---

## 4. Selection rationale — why these 12 and not the 53

### 4.1 Pinning
`BASE_METRIC` is an explicit tuple, and `hiride_metric_floor.py` selects `metric` by name membership rather than "everything not excluded", with a warning if any pinned column is missing [C]. The comment gives the reason: deriving the set by exclusion would have silently redefined "metric" the moment the shape block was appended to the npz, and every §8/§12 number would have stopped reproducing "without anything looking wrong". The 12 are therefore the set that produced 19.04 % at R4 and 67.97 % at R1 when the geometry turn happened (§12.1), frozen there; all later feature work is evaluated as `shape` and `metric+shape` beside them, never inside them.

### 4.2 The 53 columns: 6 + 12 + 35, and no skeleton
- **35 crown-anchored `shape` columns** (`hw_`, `ht_`, `hc_`, `ha_` at 100…800 mm below p99.5(h), ±40 mm bands, plus three width ratios `r_200_700`, `r_200_400`, `r_400_700`) were written specifically to fix the base-anchoring defect of §2.5: the crown is visible far more often than the feet (top-touch 0.43 vs bottom-touch 0.98 at close range) and an absolute millimetre offset is anatomically stable however much leg is missing [C, §13.14]. What they bought [M, `metric_floor_fbtest.json`]: full-body R4 frame-level `metric` 28.06 → `metric+shape` 32.06 % [24.38, 39.58]; R3 21.91 → 33.62 %. `shape` alone is worse than the 12 at R4 (18.70 %). Under aggregation the gain **inverts**: W = 25 43.30 → 42.33, whole tracklet 50.71 → 48.57 [M, §13.13], because within a tracklet the range sweeps systematically, a range-driven error is correlated across the window, and integration cannot remove it. Thickness `ht_ddd` (0.10–0.91) is the one physically new and range-stable family; crown-anchored widths beyond 200 mm still drift ~2.0, which says the residual defect is a scale error in the height axis (the ground plane, §1.1), not the offset error crown-anchoring fixed.
- **Skeleton columns: none exist.** 53 = 6 + 12 + 35 exactly; the `bone_` path produced nothing, and `hiride_metric.py` now deliberately declines to read `_skel.txt` [C]. Territory, not capability (§13.16): paper 1 is published and skeleton-only (a 20-D bone-segment-length vector on 14 participants), so limb lengths and limb-proportion ratios are another paper's result and are excluded from the headline entirely, including as a pose covariate; `Testing/Still` provides pose control instead. What paper 2 may say: a CNN on raw depth does not recover geometry that tracked joints make explicit — citing paper 1 for the joints.

### 4.3 What `--invariance-max` / `--min-snr` found
`select_columns` (`hiride_fuse.py`, called by `hiride_sequence.py`) keeps a column only if (a) its fixed-effects drift across the TRAINING range, divided by its between-subject SD, is below R and (b) its between-subject SD over its within-subject SD exceeds S — both from training rows only, so no test frame or label informs the choice [C]. The `min_snr` term exists because drift alone kept `ht_300`, range-stable (0.10) but nearly uninformative (between-SD 26.7 vs within-SD 133.1) [C, docstring].

Two results, both recorded so they are not retried [M, §13.13]:
1. **Label-free drift selection cannot work on this dataset.** Estimated from training rows, `stature_mm` reads drift **0.02**; the pooled diagnostic over the test walk reads **1.87** for the same column. Not a bug: `Training` spans 2187–3565 mm and the drift is produced by frame clipping below ~2200 mm, which the training pool barely contains (36 training frames under 2 m vs 921 test frames, `range_profile.json`). The bias is out-of-distribution by construction, so no criterion fitted on the training pool can see it. The full-body gate works precisely because it is a per-frame validity check rather than anything learned.
2. **Do not quote the threshold sweep.** Four (drift, snr) pairs were tried; the best read 46.02 % at W = 25 and 57.86 % whole-tracklet — the maximum of four attempts scored on the test set, the exact failure the criterion was meant to prevent. And nothing in it is resolvable: at 103 decisions the binomial SE near 45 % is 4.9 pp, at 28 decisions 9.4 pp; every variant (43.11–46.02) sits within one SE of plain `metric` (43.30). The published operating point was run with `invariance_max: None, min_snr: 0.0, features: metric` [M, `sequence_gated_best.json` `_meta`].

### 4.4 Tried and rejected [M]
| lever | result | why |
|---|---|---|
| `stand_dist_mm` as a 13th feature | R1 67.97 → 57.59, R3 15.56 → 10.76, R4 19.27 | range is a recording shortcut within a session, useless across it (§12.1) |
| `--detrend` (one global slope per feature, fitted on train) | R1 67.97 → 44.30, R3 → 9.23, R4 → 19.28 | distance-correlated variance is a legitimate shared cue when train and test share a session; removing it helps only where distributions differ, and there only marginally (§13.12) |
| averaging FEATURES over the window (`met_agg`) | gated: 28.86 → 23.30 (W 25) → 20.0 (W 100) → 16.43 (tracklet); ungated 18.83 → 5.37 at 100 frames per §13.12 | a test window spans standing distances a training window never does; posterior averaging survives because each frame still votes separately |
| `--range-match` (test frames inside the training p05–p95 band) | 36.3 % of R4 test frames fall outside | answers a different question; not comparable to the headline |
| `shape` alone / `metric+shape` | §4.2 | frame-level gain, inverted after integration |
| whole-tracklet number (50.71 %) | not quoted | 28 decisions, interval 36 pp wide; quote W = 25 |

**Where 28 subjects stop resolving.** The gated operating point rests on 103 decisions (SE ≈ 4.9 pp); no feature-set variant separates from `metric` there. The only part of the feature work with enough decisions for a claim is the frame-level full-body gain (28.86 → 31.41 on 2,933 frames per §13.13; 28.06 → 32.06 in `metric_floor_fbtest.json`), and that needs a subject-clustered interval before it becomes a sentence. What survives the feature question is the gate and the observation budget — measurement discipline, not a feature. Settling it needs more subjects, not more columns.

### 4.5 Three ways of giving the CNN this information, none of which moved it [M]
Score-level fusion (§13.10: oracle − metric +7.4 to +8.8 pp with intervals clear of zero, but no fixed rule captures it); the standing distance at the head (`--aux dist`, §13.17, pre-registered and refuted: ≤ +2.4 pp where the test has power); the 12 scalars z-scored on train rows at the head (`--aux metric`, wave 19, §14.6: +0.18/+1.39/+2.65 pp at R4, inside seed variance). The complementarity is real, the headroom is real, no combination realises it at n = 28. This is why the 12 are reported as a model in their own right rather than as an input to the network.

---

## 5. Measurement-error model

### 5.1 Sensor: Kinect v1 structured light
Depth is triangulated from disparity, so the depth step grows as z². The campaign's predicted step (`noise_floor.json`, `predicted_step_mm`): 4.5 mm at 1.25 m, 8.9 at 1.75, 14.7 at 2.25, 21.9 at 2.75, 30.6 at 3.25, 40.7 at 3.75 m — ≈ 26 mm at 3 m, the figure §13.17 uses against ~15 mm of facial relief. The **measured** frame-to-frame depth sigma on BIWI is far smaller: 5–7 mm below 3 m rising to 9–13 mm at 3.25–3.75 m in all three sequences [M, `noise_floor.json` `noise_mm`]. §12.2 records the correction: the disparity-quantisation prediction of 40–70 mm sigma at 3 m "was wrong and was corrected by direct measurement"; BIWI depth is evidently post-processed. Lateral resolution: one pixel spans z/575.8 ≈ 3.5 mm at 2 m, 5.2 mm at 3 m, 6.1 mm at 3.5 m, so a width percentile difference carries ~±1 px ≈ ±5–10 mm of quantisation at working range. The independent reference for the z² law and the "few mm to ~4 cm at 5 m" random error is Khoshelham & Elberink 2012 [V].

Consequence: sensor noise is NOT the binding error for any of the 12. It is an order of magnitude below the between-subject SDs (38–105 mm) and averages away; the errors that matter are systematic.

### 5.2 Segmentation (the userMap)
The mask is SDK player-index output, not annotation (§14.2), so every metric number is "given segmentation at least this good": the naive alternative (1500 mm slab + largest component) has recall 0.976 but precision 0.204 and merges body with floor and wall (§12.5) — which is exactly what made paper 3's "stature" a room measurement (stature Training↔Testing correlation −0.04 there vs +0.848 here, §5.4). The percentile definitions tolerate stray pixels; a systematically eroded or dilated boundary shifts widths by the erosion width on each side and area proportionally. The 22.3 % of Training frames with an empty map (tracker loss, one contiguous run per subject) are removed for every condition by `eligible_mask`.

### 5.3 Clipping — the dominant error, and why it makes a frame invalid rather than hard
Geometry: the Kinect v1 vertical FOV is nominally 43° (the intrinsics used imply 2·atan(240/575.8) ≈ 45°); at 2 m the frame spans ≈ 1.67 m vertically, at 2.5 m ≈ 2.08 m, so a standing adult cannot fit below ~2.1–2.2 m even when centred. Measured on the cue-eligible R4 frames [M, `range_profile.json`]:

| person median depth | test frames | test bottom-touch | test top-touch | train frames | train bottom-touch |
|---|---|---|---|---|---|
| < 2000 mm | 921 | 98.3 % | 43.2 % | 36 | 100 % |
| 2000–2500 | 1285 | 87.4 % | 2.8 % | 963 | 71.0 % |
| 2500–3000 | 1303 | 48.0 % | 0 | 3244 | 0.18 % |
| 3000–3500 | 1254 | 3.3 % | 0 | 3431 | 0 |
| ≥ 3500 | 879 | 0 | 1.6 % | 220 | 0 |

Weighted: 47.8 % of the 5,642 R4 test frames touch the bottom row (48.0 % fail the full-body gate; it retains 2,933 = 52.0 %) against 9.2 % of the 7,894 training frames — a 5× covariate shift in framing between sessions (`eligible_mask` comment) [C, M]. The handoff's "74 % of walking frames clip the body / only 25.6 % contain a whole body" (§13.12(b), §14.8) uses all `Testing/Walking` frames as denominator, before cue eligibility removes the closest frames [inferred — the two denominators must be named when either figure is quoted].

Effect on the features: a clipped body is a shorter body, so `base` is mid-shin, `stature` shrinks, and every height-anchored quantity — the three heights and the six band positions — reads a different anatomy depending only on how much was cut. That is the entire content of the 89.5 mm/m slope and of the drift ordering in §2. Ten of the twelve drift more across the walk than they vary between people [M, §13.12].

Why "invalid", not "hard" [M, §13.12(b)]: on unclipped R4 test frames the 12 score 28.06 %, on clipped 9.3 % (back-solved from 19.04 % over 5,642); **training on unclipped frames as well adds only +0.83 pp** (28.06 → 28.89), so no training regime repairs a clipped measurement — the number is wrong, not merely noisy. The same signature appears under integration: systematic error does not average away, noise does. The CNN gains little from the observation budget (gated 21.6 → 28.2 → 31.4 % at W = 1/25/100), the metric model nearly doubles (28.9 → 43.3 → 54.3 %) [M, `sequence_gated_best.json`]; ungated, the metric curve peaks at W = 25 (30.14 %) and then FALLS (27.96 at 50, 25.85 at 100) because longer windows mix in more clipped, range-shifted frames [M, `sequence_ungated_best.json`]. Rejecting frames the sensor cannot measure is quality-gated frame selection — ordinary in biometrics (Barbosa 2012 chose the frame "not cropped by the sensors fields of view" [V, extract]; Karianakis 2018 removed frames occluded by the image boundary [V, audit 03]) and a real option for a system watching someone walk. The retained fraction (52.0 %) must always be quoted beside the gated number.

### 5.4 Does the measurement transfer between sessions? [M, `metric_check.json`, `hiride_metric_check.py`]
Per-subject median `stature_mm` in `Training` vs `Testing/Walking`, 28 shared subjects: Pearson **+0.848**, Spearman **+0.776**; median bias (Walking − Training) **−45.3 mm**; within-subject SD (median over subjects of per-subject frame SD) **106.2 mm** vs between-subject SD **97.2 mm**. Read: the quantity is the body and it transfers (paper 3's −0.04 was a property of its foreground), but at the FRAME level within ≥ between — which is exactly why a single frame gives ~19–28 % and why the gate (removes the clipped tail of the within-subject spread) and the observation budget (averages the rest) are where the accuracy comes from. The −45 mm bias is the session-level face of the clipping drift (Testing is closer and more clipped).

### 5.5 Camera pose
Translation and standing distance cancel by unprojection (the point of the exercise: the census measured +620 mm of background depth for all 28 shared subjects, people ~30 cm closer and 34 px lower in the test session — §12.1). Pitch does NOT cancel under the camera-Y fallback (§1.1); its size is unmeasured. Yaw of the body (not the camera) affects `dep`, `lat` and area because BIWI Training includes rotation while Walking is roughly frontal.

### 5.6 Pose and clothing
No pose normalisation is applied. Gait phase enters `depth_extent_mm` (leading limbs), `w_15`/`w_30` (stance separation), `w_45`–`w_75` (arm swing), `height_p50` (raised arms). Clothing enters the widths and the area (a coat) and footwear/headwear enter stature by a few centimetres; Barbosa 2012 flags both [V, extract]. `Testing/Still` exists as the pose control; §13.8 records that gait itself is not testable under this protocol (2.5-s probes, most frames clipped).

### 5.7 Code-level defects to disclose
Zero sentinels for empty bands (`w_XX = 0.0` when ≤ 30 pixels; likewise the shape block) are read as measurements by the RF. The `ground` column question (§1.1). `volume_proxy_l` is a product of two reported columns.

---

## 6. Lineage — anthropometric soft biometrics from consumer depth

| work | what it measured | protocol | status |
|---|---|---|---|
| Barbosa, Cristani, Del Bue, Bazzani, Murino, "Re-identification with RGB-D Sensors", ECCV 2012 Workshops, LNCS, pp. 433–442, DOI 10.1007/978-3-642-33863-2_43 | 10 cues pruned from 121: skeleton-based `d1` floor–head, `d2` torso/legs ratio, `d3` height estimate, `d4` floor–neck, `d5`–`d7` joint distances; surface geodesics `d8`–`d10` on a mesh. Floor by RANSAC plane fit; joints from OpenNI/NITE (15 joints); ONE frame per acquisition, best joint confidence, nearest, not cropped by the FOV (~2.5 m); height cues most informative | 79 people, four groups over different days, clothing changed | **[V]** Crossref + saved extract |
| Munaro, Fossati, Basso, Menegatti, Van Gool, "One-Shot Person Re-identification with a Consumer Depth Camera", in *Person Re-Identification*, Springer 2014, pp. 161–181, DOI 10.1007/978-1-4471-6296-4_8 | introduces BIWI RGBD-ID; skeleton descriptor (limb lengths and ratios) + NN/SVM; point-cloud matching after a standard-pose transform; samples selected by face detection | gallery = Training, probes = Still/Walking (28 people, new day, new clothes); skeleton NN Walking 21.1 %, PCM 22.4 %, PCM+skeleton 27.4 single / 42.9 multi | **[V]** DOI/Crossref; feature details and numbers **[V-sec]** via Haque 2016 / Karianakis 2018 / Uddin 2023 (audit 03: primary chapter not read) |
| Munaro, Basso, Fossati, Van Gool, Menegatti, "3D reconstruction of freely moving persons for re-identification with a depth sensor", ICRA 2014, pp. 4512–4519, DOI 10.1109/ICRA.2014.6907518 | companion; the IAS-Lab dataset used OpenNI+NITE (§14.2) | — | **[V]** Crossref |
| Andersson & Araujo, "Person Identification Using Anthropometric and Gait Data from Kinect Sensor", AAAI 2015, 29(1), DOI 10.1609/aaai.v29i1.9212 | 20 anthropometric attributes = per-walk means of Kinect SDK joint-to-joint segment lengths after a > 2 SD outlier discard; height = neck + upper/lower spine + mean hip + thigh + lower leg; plus 60 gait attributes (step/stride length, cycle time, velocity, joint angles); KNN | 140 people, single session, 10-fold CV; anthropometric alone ~85 %, gallery-size curve with random re-draws | **[V]** Crossref + saved extract |
| Paolanti et al., Sensors 18(10):3471, 2018, DOI 10.3390/s18103471; Liciotti et al., LNCS 2017, pp. 1–11, DOI 10.1007/978-3-319-56687-0_1 (TVPR) | skeleton-free, top-view depth: floor–head, floor–shoulder, head area/circumference, shoulder circumference/breadth, thoracic depth + colour | 100 people, within-session per audit 08_refute_C3C4 [inference there] | **[V]** Crossref (both); feature list per audit 08 (Europe PMC full text) |
| John, Englebienne, Kröse, "Person re-identification using height-based gait in colour depth camera", ICIP 2013, pp. 3345–3349, DOI 10.1109/ICIP.2013.6738689 | height-based gait from a colour-depth camera | — | **[V]** Crossref record; content abstract-level only (audit 08) |
| Khoshelham & Oude Elberink, "Accuracy and Resolution of Kinect Depth Data for Indoor Mapping Applications", Sensors 12(2):1437–1454, 2012, DOI 10.3390/s120201437 | Kinect v1 error model: random depth error grows with distance, a few mm to ~4 cm at maximum range; resolution ∝ z² | — | **[V]** Crossref |

**What is the same.** The target quantities: stature/floor–head distance (Barbosa `d1`/`d3`, Andersson height, Paolanti `d1`), a torso-versus-legs proportion (Barbosa `d2` ≈ our `height_p50`), breadths at named heights (Paolanti shoulder breadth ≈ our `w_75`), thickness (Paolanti thoracic depth ≈ our `depth_extent_mm`/`ht_`), a frontal-view walking corpus with a clothing change across days (Barbosa; BIWI), and the frame gate — Barbosa's "not cropped by the sensors fields of view" is the same rule as our `bot_clip`, twelve years earlier. Height being the most informative cue is common to Barbosa 2012, Andersson 2015 and our RF importances. "Metric anthropometry from consumer depth" is therefore 2012–2018 practice and is **refuted as a mechanism discovery** (§14.8).

**What is different.**
1. **No skeleton tracker.** Barbosa, Munaro and Andersson measure between SDK/NITE joints; we take percentiles of the unprojected point set through the sensor mask. That removes the joint-fitting failure modes and the need for a tracked pose, and it is what keeps paper 2 clear of paper 1's territory (§13.16). Paolanti 2018 is also skeleton-free but top-view, where floor–head distance is a direct range reading and clipping does not arise.
2. **Frontal, walking, across sessions, with the sensor's own mask.** TVPR is top-view and within-session; Andersson is single-session; Barbosa's evaluation is nAUC on single selected frames. Ours is the standard BIWI cross-session probe set, every eligible frame, with the segmentation that a depth system has at runtime (§13.15/§14.2).
3. **A measured drift model.** Nobody in the lineage quantifies clipping prevalence (74 % of walking frames / 48 % of eligible test frames), the per-metre drift of height features (89.5 mm/m vs 105 mm between-subject SD), the drift ranking of every column, the gate's cross-session effect (28.06 vs 9.3 %), or the session-level stature transfer (r = 0.85, bias −45 mm). Prior gates were applied, never ablated (audit 03). Claim first quantification, not invention.
4. **The comparison against learned models on identical frames.** The 12 are the reference against which every CNN in the suite is judged on the same frames, split and mask — and the statement is "beats the CNNs we train" (single-frame 19.04 vs 6.70 % full-frame), never "every CNN": published depth networks on whole sequences under Training(50) → Walking are higher (Haque 2016 30.1 % single-shot; Karianakis 2018 25.4 % single / 50.0 % multi), as are hand-crafted depth descriptors (Wu 2017 ED+SKL 24.47 % Walking; Munaro 2014 22.4 %) [V-sec, audit 03]. Our gated 2.5-s operating point (43–47 %) sits inside the prior multi-shot band; wave 22 puts the two side by side on one pipeline.
5. **Camera frame, not (yet) gravity frame** — §1.1. Barbosa fitted the floor by RANSAC; BIWI ships a plane per frame; ours appears not to have been read. This is a difference to disclose, not to claim.

**Safe wording** (consistent with 08_refute_C3C4, corrected for dimensionality): "Anthropometric soft biometrics from consumer depth sensors are established (Barbosa et al., 2012; Munaro et al., 2014; Andersson and Araujo, 2015), including skeleton-free floor-referenced distances from top-view depth (Paolanti et al., 2018). We recover twelve such quantities in millimetres from unprojected frontal depth through the sensor's runtime mask, without a skeleton tracker, measure their drift with standing distance, and use them as the reference against which learned models on the same frames are judged across sessions."

---

## 7. Numbers to quote, numbers not to quote, and open checks

Quote: 12 features; R4 19.04 % [14.00, 24.40] on 5,642 frames; full-body 28.06 % [21.29, 34.66] on 2,933 (52.0 % retained); gated W = 25 43.30 % [30.3, 56.1] on 103 decisions; drift 89.5 mm/m vs 105.2 mm; drift/sig for all 12 (§2 table); stature transfer r = 0.848, bias −45 mm; sensor sigma 5–13 mm measured vs 4.5–40.7 mm predicted step.

Do not quote: 50.71 % whole-tracklet as the headline; any (drift, snr) sweep value; `metric+shape` under aggregation as a gain; 22.13 % geo fusion as an improvement on 19.04 %; the "13-dimensional descriptor" phrase in 08_refute_C3C4's safe wording (BASE_METRIC is 12; 13 is `metric+nuisance`, which scores 19.27 %); any mechanism or bit-depth number from the 2026-08-31 `tables.tex` (its 16-bit anchor reads 18.12 where the gap cell is 13.62 — §14.8 keying bug; use `stats_final.json`).

Open checks before submission: (1) the `ground` column fraction (§1.1) and the wording "gravity-aligned"; (2) between-subject SDs for the eight columns not recorded in the handoff — re-run `hiride_metric_bias.py` and save its table (it prints and does not write); (3) a subject-clustered interval for the frame-level full-body gain 28.06 → 32.06 before it becomes a sentence; (4) the denominator behind "74 %" stated explicitly beside the 48 % figure; (5) replace the 0.0 band sentinel with NaN handling if the features are ever recomputed — the current numbers were produced with the sentinel and must be described as such.
