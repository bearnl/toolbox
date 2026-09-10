# Lens 3 — Error structure, aggregation and biometric information

Audit report for the HI-RIDE paper-2 manuscript (BIWI RGBD-ID, Kinect v1, 28 people re-recorded on another day). Written 2026-09-09 from the measured record in `/Volumes/Workspace/study/toolbox/HIRIDE_HANDOFF.md` (§12.1, §12.2, §13.10, §13.12–§13.18, §14.6, §14.8), the result JSONs in `/Volumes/Workspace/study/hiride2-results/` (`sequence_gated_best.json`, `sequence_ungated_best.json`, `noise_floor.json`), the code (`hiride_sequence.py`, `hiride_fuse.py`, `hiride_train.py`, `hiride_data.py`, `hiride_metric_floor.py`, `hiride_noise.py`), and the saved literature extracts in `lit-audit/extracts/`. Every literature item is tagged [V] (confirmed this session on Crossref / a DOI landing page / a saved primary-text extract) or [UNVERIFIED]. Where a number in the brief handed to this lens disagrees with the record, the record wins and the disagreement is flagged.

**Headline of this lens.** The fig-7 asymmetry is real and is the single most legible piece of evidence about *what kind* of error each representation makes — but the tidy story "sensor noise averages as 1/√N, CNN confusions do not" is only half right and the record itself says so. The measured temporal sensor noise is 6–12 mm (`noise_floor.json`), an order of magnitude below the 105 mm between-subject stature SD; what integration removes from the metric vote is frame-to-frame pose/segmentation jitter, and what it cannot remove — shown by the *ungated* metric curve flattening at ~30 % and by feature-averaging *falling* with window length — is the distance-coupled bias (89.5 mm/m). The CNN's errors are the systematic kind throughout. All of this is checkable from the stored per-frame posteriors, and the exact checks are specified below (§3).

---

## 1. The measured record this lens rests on

### 1.1 Fig-7 source numbers (`sequence_gated_best.json`, best recipe `alexnet/stripe/aug8/tf10`, `scale_removed`, R4, full-body gate, product-rule aggregation)

| frames/decision | decisions | CNN | metric (12 + RF) | fusion (geo) | **feature-averaged metric** (`met_agg`) |
|---|---|---|---|---|---|
| 1 | 2,933 | 21.60 | 28.86 | 32.51 | 28.86 |
| 2 | 1,461 | 22.26 | 31.75 | 34.88 | 29.69 |
| 5 | 576 | 23.47 | 34.69 | 38.92 | 28.06 |
| 10 | 281 | 24.91 | 39.72 | 42.92 | 26.19 |
| **25 (2.5 s)** | **103** | **28.16** [14.8, 43.0] | **43.30** [30.3, 56.1] | **46.99** [32.9, 61.4] | 23.30 |
| 50 | 45 | 30.22 | 49.33 | 49.33 | 17.78 |
| 100 | 28 | 31.43 | 54.29 | 55.71 | 20.00 |
| whole tracklet | 28 | 29.29 | 50.71 | 55.00 | 16.43 |

Paired, subject-clustered contrasts at W = 25 (§14.6): geo − metric +3.69 pp [−13.78, +20.39]; cnn − metric −15.15 [−31.48, +1.34]; geo − cnn +18.83 [+9.01, +29.39]. Chance 3.57 %.

### 1.2 The same sweep without the gate (`sequence_ungated_best.json`, 5,642 → 211 → 28 decisions)

| frames/decision | CNN | metric | fusion | `met_agg` |
|---|---|---|---|---|
| 1 | 18.36 | 18.83 | 25.31 | 18.83 |
| 25 | 24.08 | 30.14 | 37.25 | 15.64 |
| 100 | 22.44 | 25.85 | 39.02 | 5.37 |
| whole tracklet | 25.00 | 26.43 | 38.57 | 13.57 |

### 1.3 Other anchors used below

- Sensor floor (`noise_floor.json`, method in `hiride_noise.py`): predicted disparity-quantisation step z²/(f·b·8) = 21.9 mm at 2.75 m, 30.6 mm at 3.25 m (≈25 mm at the 2,955 mm median standing distance); **measured** frame-to-frame σ on static background pixels (robust MAD estimator) = 6.3 mm at 1.5–2.75 m rising to 9.4–12.6 mm at 3.0–3.75 m, in all three sequences. §12.2 records the prediction (40–70 mm) as *refuted by this measurement*; BIWI depth is evidently post-processed.
- Between-subject SD of `stature_mm` 105.2 mm; range slope 89.5 mm/m; drift over the 2.2 m walk ≈197 mm = 1.9× the identity signal; ten of twelve features drift more than they vary between people (§13.12 a).
- Only 25.6 % of `Testing/Walking` frames hold a whole body; the full-body gate (`hiride_data.eligible_mask(full_body=True)`: mask touches neither the top nor the bottom frame row) lifts single-frame metric from 19.04 to 28.06 % and retains 52.0 % of R4 test frames (§13.12 b, §14.6).
- Z-precision axis at R4 (§14.6, `stats_final.json`): 16 bit 13.62, 8/4/3/2 bit 12.45/13.02/11.22/12.17, 1 bit 9.51; the normalised binary silhouette (`sil_scaled`) is the CNN's best depth input. Quote nothing from the 08-31 `tables.tex` (§14.8 bug).
- `--aux dist` (§13.17) and `--aux metric` (wave 19, §14.6) change nothing within seed variance; ImageNet ConvNeXt-tiny full frame R4 8.45 %.
- Frame-level fusion (§13.10; regenerated contrasts §14.6): oracle-either − metric +7.7 / +9.0 / +8.3 pp, intervals clear of zero; `rescued` = P(metric right | CNN wrong) 15.2–17.2 %.
- Per-subject decision accuracy at W = 25, gated (`per_subject` in the JSON): metric `0 0 0 0 0 0 5 7 7 20 30 33 37 40 40 50 52 53 64 70 73 77 80 84 100 100 100 100` (6 never, 4 always, median 38.5 %); fusion `0 0 0 5 7 8 24 30 33 33 33 35 37 40 40 47 55 60 69 73 80 88 93 93 93 100 100 100` (3 never, 3 always, median 40 %). Never-identified sets: metric {002, 003, 008, 010, 016, 019}; fusion {006, 019, 023}.

---

## 2. Why 25 frames lift the metric vote by 14 pp and the CNN by 7 pp — three mechanisms, each grounded and each with a prediction

### Mechanism A — the metric vote integrates an error that is *partly* independent across frames; the sensor is not the part that matters

**Claim.** A length measured in millimetres has a per-frame error with two components: a fast component (gait-phase and segmentation jitter of the p99.5/p0.5 extremes that define `stature_mm`, arm swing inflating `w_45`/`w_60`, step width inflating `depth_extent_mm`) that decorrelates over a fraction of a second, and a slow component (the distance-coupled clipping bias, 89.5 mm/m) that is a monotone trend over the whole walk. The posterior vote averages the first and is blind to the second. Kinect sensor noise proper is a negligible third term.

**Our evidence.**
1. `noise_floor.json`: measured σ 6–12 mm. Against a 105 mm between-subject SD that is a per-frame stature d′ of roughly 10 for the sensor term alone; if sensor noise were the binding error, one frame would already separate 28 people. The single-frame gated metric accuracy is 28.86 %, so the binding per-frame error is far larger than 12 mm and is not sensor noise.
2. The gated metric curve keeps rising to 54 % at 100 frames; the **ungated** curve peaks at 30 % (W = 25) and *falls* to 26 % at W = 100 (§1.2). Same features, same classifier, same aggregation rule; the only difference is whether frames with a clipped body — whose stature error is a bias that grows monotonically as the subject approaches — are admitted. A bias correlated across the window does not average; the gate removes most of it and integration then works. This is the within-record demonstration of "systematic error does not average away; noise does" (§13.12 c).
3. **Feature averaging fails where posterior averaging succeeds.** `hiride_sequence.py::agg_windows` averages the 12 measurements over the window and classifies the mean (the RF retrained on stride-1 training windows). Its docstring predicts the opposite of what happened ("25 frames cut a stature error by five"): gated `met_agg` goes 28.86 → 23.30 (W = 25) → 16.43 (whole tracklet); ungated 18.83 → 5.37 at W = 100 (§1.1, §1.2, §13.12). If the per-frame error were i.i.d. sensor noise the window-mean feature would be the sharper measurement and `met_agg` would beat the posterior vote. It does not, so the dominant per-frame error is not i.i.d. around a stable true value; the window mean lands where no training window lies (§13.12: a test window sweeps distances a training window never does). The posterior vote survives because each frame is scored at its own distance against a classifier that saw single frames at that distance.
4. The crown-anchored `shape` block (§13.13) improves single frames (28.06 → 32.06 %) and *hurts* after aggregation (43.30 → 42.33; 50.71 → 48.57): drifting features add a correlated error that integration cannot remove. Frame-level accuracy is the wrong objective when the deployment integrates.

**Corrections to the brief.** Do not write "sensor noise ~25 mm, independent across frames, averages as 1/√N". The ≈25 mm figure is the *predicted disparity step* (`predicted_step_mm`; Khoshelham & Elberink measure the same quantity, §7) and the measured temporal noise is 6–12 mm. Write instead: the sensor term is measured and small; the per-frame error that integration removes is pose/segmentation jitter; the error it cannot remove is range-coupled clipping bias, which is why the gate and the observation budget are complementary rather than alternative fixes.

**Predictions (checkable from `metric_features.npz` + the manifest, CPU minutes).**
- P-A1. Within-recording, gated: regress `stature_mm` on `stand_dist_mm` per test recording; the residual SD (fast component) will be several times the 6–12 mm sensor σ (order 30–60 mm, a prediction not yet measured) and its lag-1 autocorrelation will be well below 1, while the fitted slope component over the window will be of the order of the residual SD or larger for windows spanning >0.5 m. The ratio of the two predicts, per window, how much of the error the vote can integrate.
- P-A2. The "i.i.d. ceiling" curve: resample W frames i.i.d. from each true class's row of the gated frame-level confusion matrix and take the plurality; plot accuracy against W. For the metric RF this curve will exceed the measured 43.3 % at W = 25 by a wide margin (with 28.9 % on the true class and the rest spread across confusers, 25 independent draws almost always recover the true class), which quantifies how correlated the metric's per-frame errors still are; the W at which the i.i.d. curve equals the measured 43.3 % is the effective number of independent frames in a 2.5 s window (expected single digits). For the CNN the i.i.d. curve will sit even further above the measured curve (see Mechanism B).
- P-A3. Restrict the gated test windows to those spanning < 0.3 m of standing distance: the metric curve should rise faster with W than on the full gated set; restrict to windows spanning > 1 m: it should flatten. The CNN curve should be indifferent to this split.

### Mechanism B — the CNN's confusions are systematic: the same wrong identity on every frame of a clip

**Claim.** The CNN's per-frame errors within a recording are the same error repeated, because the network reads the normalised outline (framing, standing pose, silhouette proportions) which is stable within a recording, and because its training and model selection are within-session (the validation tail is carved from the *training* recording; early stopping on `val_accuracy`, `hiride_train.py`), so the selected epoch is the one that best memorises recording-specific outline. Integration therefore buys little.

**Our evidence.**
1. Gap-head CNN across the whole sweep 18.40 → 20.71 % (§13.12 c); best recipe (augmented) 21.60 → 28.16 → 31.43 %, whole tracklet 29.29 (§1.1) against metric 28.86 → 43.30 → 54.29. Augmentation is the one thing that changed the CNN's character (§13.12): random framing perturbations convert some systematic error into frame-to-frame noise, which is why the augmented network gains from integration and the gap-head one does not.
2. §13.10 `rescued`: the metric model rescues one CNN-wrong frame in six; the CNN's wrong frames are a stable set.
3. Handing the network the scale it lacks changes nothing (§13.17, §14.6 wave 19) — the error is not a missing input, it is what the network chose to read.
4. Sample size (§13.17 point 3): ≈7,894 training frames per 28 classes but consecutive frames are near-duplicates; independent samples per class number in the tens against ~23 M parameters; ImageNet ConvNeXt does not rescue it (8.45 %). A model in that regime memorises recording-specific appearance, and recording-specific appearance is constant within the test recording → constant wrong answer.
5. The prior BIWI depth literature shows the *opposite* CNN behaviour under a different protocol: Karianakis et al. [V, extract lines 545–554] go from 25.4 % single-shot (CNN) to 45.7 % with CNN-LSTM + average pooling over the whole Walking sequence (50.0 % with learned RTA weighting); Haque et al. [V, extract Table 3] report 27.8 % for 3D CNN + average pooling and 45.3 % for 4D RAM multi-shot, 50 classes. Their CNNs gain ~20 pp from integration; ours gains 8–10. The difference in protocol (50 training identities, ImageNet split-rate fine-tuning, whole sequences) is exactly what wave 22 (`R4_standard_walking`, `tracklet_acc` recorded on every cell, §14.8) will measure.

**A second, mechanical contributor that must be excluded before "systematic" is claimed — the product-rule veto.** `hiride_sequence.py::sequence_acc` aggregates with `agg="geo"`: the window decision is the argmax of the mean *log* posterior (Kittler's product rule). CNN posteriors are stored as **float16** (`hiride_train.py` savez block), so any true-class probability below ≈6 × 10⁻⁸ is stored as exactly 0 and contributes log(10⁻¹²) ≈ −27.6 to the window sum, against typical per-frame log-posteriors of −1 to −5. One over-confident wrong frame can veto the true class for the whole window. RF posteriors are vote fractions in steps of 1/300 and are exposed to the same floor whenever no tree votes for the true class, but a softmax is far more prone to it. Kittler et al. [V] is the standard reference for the sum rule's robustness to exactly this kind of estimation error (content claim [UNVERIFIED this session]; DOI verified, abstract not carried by Crossref). `tracklet_scores()` in `hiride_train.py` already uses the arithmetic mean, so the two rules will be directly comparable on wave-22 cells.

**Predictions.**
- P-B1 (the decisive check, §3): within a 25-frame window in which the CNN's decision is wrong, the per-frame argmaxes concentrate on one wrong identity (modal share high, vote entropy low); for the RF they are spread (modal share low, vote entropy high) and the true class is often the runner-up.
- P-B2: re-aggregate the CNN posteriors with the arithmetic mean (`agg="mean"` — the function already supports it). If the CNN's W = 25 accuracy rises by several points, part of the flattening was the veto artefact and the manuscript must say so; if it does not move, the systematic reading stands unqualified. Report both rules for both models.
- P-B3: across the five seeds of the best-recipe cell, the modal wrong identity per test recording will be *less* stable for the CNN than for the RF. RF confusions are nearest neighbours in a fixed millimetre space and should recur across seeds; CNN confusions born of recording memorisation should vary with initialisation. (If instead they recur across seeds, the confusion is a stable outline similarity in the data — also a systematic error, and one that names the confuser.)
- P-B4 (wave 22): under the 50-class standard protocol the CNN's `tracklet_acc` − single-frame gain will move towards the 20 pp of Karianakis/Haque if the small-n memorisation account is right; if it stays ≈8–10 pp the account is wrong and the manuscript should drop it.

### Mechanism C — the two representations fail together on a hard core, so the fusion oracle is only ~8 pp above metric

**Claim.** The frame-level oracle (`either`) is close to what two *independent* classifiers would give, and slightly below it; the overlap is not "the same information" so much as "the same corrupted frames".

**Our evidence.** From §13.10 (frame level, ungated): if the two were independent, P(either) = a + b − ab would be 30.7 / 28.9 / 28.5 % for `sil_scaled` / `scale_removed` / `person_centred`; measured 27.62 / 26.18 / 27.04 — 1.5–3.1 pp *below* independence. `rescued` 15.2–17.2 % against the unconditional 18.83 % says the metric is slightly *less* likely to be right on a CNN-wrong frame than on a random frame. Both fail on 72–74 % of frames. Under the gate the fused − metric gain is +3.6 pp at W = 1, +3.7 at W = 25, 0.0 at W = 50, +1.4 at W = 100 (§1.1): the CNN's marginal contribution does not grow with integration.

**Predictions.**
- P-C1: the both-wrong set (ungated join of §13.10) is enriched for `bot_clip = 1` and for `stand_dist_mm` outside the training p05–p95 band relative to the either-right set. Both cues are stored in `metric_features.npz` block A. If confirmed, the "8 pp headroom" is largely frames where one representation was corrupted and the other was not, which is why no fixed weight or feature-level fusion captures it (§14.6 verdict 1) — the useful combination is a *validity selector*, not a mixing weight.
- P-C2: compute the window-level oracle at W = 25 (`cor_cnn ∨ cor_metric`; the per-decision arrays exist in `hiride_sequence.py` but `either` is not yet written to the JSON). If oracle − metric shrinks from ~8 pp (W = 1) towards 0 with W, the CNN's complementary frames were noise-like and the metric vote already recovers them; if it stays ~8 pp the complementarity is systematic and a frame-dependent selector is worth building.

---

## 3. The exact check on the stored per-frame posteriors

Inputs: every `cm_*.npz` of the best-recipe cell (`R4_cross_session`, depth, `alexnet/stripe/aug8/tf10`, `scale_removed`, seeds 0–4) — fields `prob` (n_test × 28, float16), `truth`, `pred`, `test_rows`, `classes`, `cm`; the manifest (`seq`, `subject`, `session`, `frame`) via `hiride_data.load_manifest`; the gate `eligible_mask(cues, feats, full_body=True)`; and the RF posteriors reproduced exactly as `hiride_sequence.py` does (`RandomForestClassifier(n_estimators=300, random_state=seed)` on `BASE_METRIC` over the same training rows). Locate the cells with `cnn_cells()` from `hiride_fuse.py`; never re-derive the cell key (§13.10 process note). No `cm_*.npz` is on the laptop — this runs on the cluster against `$SCRATCH/hiride2/runs` or the archived `runs_20260906.tar.gz`.

For each seed, each model M ∈ {CNN, RF}, each recording r (`seq|subject|session`), and each non-overlapping window of W = 25 frames in frame order (`windows()`), compute:

1. **Per-frame argmax vector** `a_t = argmax prob_t`, t = 1..25, and the window decision `D` (log-mean argmax, as in fig 7; and mean-posterior argmax, for P-B2).
2. **Modal share** `s = max_k |{t : a_t = k}| / 25` and **normalised vote entropy** `H = −Σ_k f_k log f_k / log 25`, where `f_k` is the fraction of frames voting k.
3. **Error persistence**: over windows with D wrong, the share of frames whose `a_t` equals D itself (`persist = |{t : a_t = D}| / 25`), and the lag-k same-wrong-label agreement `P(a_t = a_{t+k} ≠ truth | a_t ≠ truth)` for k = 1, 5, 10 — for independent errors this decays towards Σ_j q_j² (q the marginal wrong-label distribution); for systematic errors it stays near 1 across the recording.
4. **Runner-up statistic**: rank of the true class in the window's aggregated posterior (already produced for top-k); the fraction of wrong windows with the true class ranked 2nd or 3rd.
5. **Veto count**: number of frames per window with `prob_t[truth] == 0` in float16 (P-B2).
6. **i.i.d. ceiling** (P-A2): resample from the gated frame-level confusion rows.
7. **Seed stability of the confuser** (P-B3): for each test recording, the modal wrong identity per seed; report the fraction of recordings whose modal confuser agrees across ≥ 4 of 5 seeds, per model.

Report, per model, the distributions of `s`, `H`, `persist` split by D correct / D wrong, on the gated set (103 windows × 5 seeds) and ungated (211 × 5). **Predicted signature:** CNN-wrong windows: `s` high (well above 0.5), `H` low, `persist` high, lag agreement flat in k; RF-wrong windows: `s` lower, `H` higher, true class frequently runner-up, lag agreement decaying. If the CNN's `s` and `persist` fall to RF levels after the arithmetic-mean re-aggregation raises its accuracy, the veto artefact was material. Everything here is a re-read of stored arrays; nothing is retrained.

---

## 4. The information view: what a normalised silhouette carries versus a vector of lengths

### 4.1 Discriminability of a length

Stature: between-subject SD 105.2 mm (§13.12). Twenty-eight people at roughly ±2 SD span ≈420 mm, an average spacing of ≈15 mm. The sensor's per-frame σ is 6–12 mm (`noise_floor.json`) — so the *sensor* would resolve neighbours in one frame only marginally, and comfortably after a handful of frames (σ/√N). The reason a single frame scores 28.9 % rather than something far higher is that the effective per-frame error of a length is dominated by pose/segmentation jitter and by clipping bias (P-A1), not by the sensor. This is the correct place for the 1/√N argument: it applies to the fast component only, and only after the slow component has been gated out. Twelve lengths carry more than one: the top cues (`stature_mm`, `height_p50`, `w_75`, `w_45`, `surface_area_m2`, §12.1) are partly redundant (`height_p50` ≈ 0.5 × stature for a standing adult), so the effective dimensionality is well below 12 — and 10 of the 12 drift with range (§13.12), so the range-stable identity content is carried by `w_15`, `w_30` and, after aggregation, the stature/height family with its bias averaged rather than removed.

A principled measure exists and is cheap to compute: Adler, Youmaran & Loyka [V] define biometric feature information as the relative entropy D(p‖q) of the intra-person feature distribution p against the inter-person distribution q, with a closed form for Gaussian features. On the 12 features (gated, per frame; and on 25-frame window means) this gives bits per frame and bits per 2.5 s directly, and the ratio is the information gain of integration. **Recommendation:** compute it; it turns fig 7's accuracy axis into an information axis and it is the natural quantitative content of the phrase "biometric information" in the manuscript. (Adler et al. computed it for face features; do not quote their numbers without re-reading the paper — none are quoted here.)

Daugman's [V] principle for iris is the one to cite for *why* pixel count is not information: the identifying power of a template is the number of independent degrees of freedom in its between-person variation relative to within-person variation, estimated from the impostor score distribution, not the number of bits stored. The record already makes the depth version of that point: quantising depth to 1 bit and normalising the silhouette costs the CNN nothing (Z-precision axis, §14.6), so all the identity the CNN extracts sits in the outline — a 65,536-pixel image reduced to a few outline degrees of freedom — and 12 numbers in millimetres beat it at single-frame R4 (19.04 vs 13–17 %).

### 4.2 Silhouette versus lengths — what each can and cannot hold

- The `scale_removed`/`sil_scaled` silhouette has had *absolute size removed by construction* (bounding-box normalisation, `apply_mask_condition`). What remains is proportion: width-at-height profiles, head/shoulder/hip outline, limb outline, posture, and the framing artefact of partial clipping. `--aux dist` would have let a network restore absolute scale; it did not help (§13.17), so the CNN does not turn even proportions into a stable identity code at n = 28.
- The 12 lengths hold absolute scale (stature, surface area) *plus* six coarse proportions (`w_15..w_90`/stature). The silhouette holds finer outline detail between those bands. The two sets therefore overlap on proportions and each holds something the other lacks: scale (metric only) and fine outline (silhouette only). That is consistent with an oracle 8 pp above metric and with the both-wrong hard core (Mechanism C).
- Within a session the silhouette also carries recording-specific framing and pose — which is why R1 `sil_scaled`/`flatten` reaches 69.7 % (§12.1) and R4 does not. Lengths in a gravity-aligned millimetre frame carry none of that by construction (§12.1: "millimetres … cancel camera pose by construction"), which is the mechanistic reason their between-session transfer is better.

**Prediction P-D1.** PCA of gated `sil_scaled` test silhouettes: the between-subject variance will be concentrated in a small number of leading components and the within-subject (pose) variance will be comparable to or larger than it along those components; the count of components with between/within ratio > 1 is the silhouette's effective degrees of freedom and is predicted to be single-digit — the same order as the 12 lengths, which is the information reason the two representations are near each other in accuracy.

### 4.3 The biometric menagerie reading of "0–100 %"

Doddington et al. [V] named the sheep/goats/lambs/wolves of speaker recognition; Yager & Dunstone [V] formalised per-user classes from the joint genuine/impostor score behaviour and tested whether they are statistically real. In our closed-set setting the per-subject decision accuracy at W = 25 (§1.3) is the goat axis: 6 of 28 never identified by the metric model, 4 always; fusion moves five of the six goats off zero and drops two others (006, 023) to zero — menagerie membership is partly a property of the (person, system) pair, not of the person alone, which is the Yager–Dunstone question in miniature. Two further points already in the record belong here: at K = 4 the spread across cohorts is 58–60 pp (§13.18, §14.6) — who else is enrolled decides identifiability — and the answer-when-sure curves (metric gated: threshold 0.4 keeps 47 % of decisions at 62 %; 0.6 keeps 14 % at 76 %) show the confident decisions are the sheep's.

**Predictions.**
- P-E1: the goats' frames concentrate on a few attractor identities (column sums of the gated confusion matrix), and attractors sit near the population centroid of the 12 standardised features (large within-class spread makes the centre absorb). Check: rank correlation between a subject's Mahalanobis distance from the centroid and their confusion-column sum.
- P-E2: each metric goat has a measurable cause in millimetres — either a between-session shift in stature/height larger than the spacing to their nearest enrolled neighbour (shoes, hair, hat, posture: a *session* effect), or a nearest neighbour within one within-recording SD in the 12-D standardised space (a *cohort* effect). Compute both per goat; the manuscript can then name the cause for each of the six.
- P-E3: the CNN's never-identified set at W = 25 (not stored in the JSON; compute from `cm_*.npz`) will overlap the metric's only partially, as fusion's already does (only 019 is shared between metric-never and fusion-never).

---

## 5. Why the metric features are explainable — and what "explainable" should mean in the manuscript

The word should be operational, not the loose XAI sense. Three concrete properties are supported by the record, one is not yet, and the manuscript should claim exactly the supported ones:

1. **Every decision decomposes into physical lengths.** The classifier's input is 12 named quantities in millimetres in a gravity-aligned frame (§13.14 B), each comparable to a tape measure. A decision can be presented as the probe's 12 values beside the enrolled subject's training distribution (per-feature z-scores) — a human can check whether "1,744 mm, 480 mm at the shoulders" is that person. *Caveat for the text:* the RF's feature importances (§12.1 top cues) are global, not per-decision; a per-decision attribution (nearest training frames in standardised feature space, or per-feature contributions from the trees) has not been computed and must not be implied. Recommend computing nearest-neighbour explanations for the 103 gated decisions — cheap, and it makes the claim literal.
2. **Every frame refusal has a physical reason.** The gate is `bot_touch`/`top_touch` (the mask touches the frame edge: the feet or head are out of view), plus the eligibility rule (≥2,000 person pixels, ≤25 % of the frame) — each a statement about the frame a reader can verify by looking at it, none learned (§13.13: "a per-frame VALIDITY check rather than anything learned"). The refusal rate itself is a measurement: 74 % of walking test frames are clipped, 36.3 % lie outside the training distance band (§13.12 b). The answer-when-sure refusal (posterior threshold) is the one refusal that is *not* physical; label it as a confidence rule.
3. **Every unidentifiable person has a measurable cause.** Not yet computed, but computable in the same units (P-E2). Until it is, write "can be given" rather than "has been given".
4. **Every failure mode was found by measurement in the same units.** The drift table (§13.12), the band-position error model (`stature_error × f × width gradient`, §13.14), and the thickness finding (§13.13) are explanations of *why* features fail, expressed in millimetres per metre. The CNN offers none of this: its refusal cannot be stated in physical units, its confusions can at best be read off a saliency map, and three attempts to hand it the missing physics changed nothing (§13.10, §13.17, wave 19).

**Suggested manuscript definition.** "We call the metric pipeline explainable in a narrow, checkable sense: every quantity it consumes is a length in millimetres that a person could re-measure; every frame it declines is declined for a stated geometric condition; and every person it fails to identify can be assigned a cause in the same units. We make no claim about the interpretability of the random forest's internal structure."

---

## 6. Sentences the manuscript can carry (bounded by this lens)

- "Averaging 25 frames lifts the metric vote from 28.9 to 43.3 % and the CNN from 21.6 to 28.2 %; over 100 frames the metric vote reaches 54 % while the CNN flattens near 31 %. The asymmetry is the signature of two error types: the metric model's per-frame errors are partly independent across frames and integrate; the CNN's repeat within a recording and do not."
- "Integration removes only the independent part of the metric error. Without the full-body gate the metric curve peaks at 30 % and declines, and averaging the measurements themselves over a window *lowers* accuracy (28.9 → 23.3 % at 25 frames): the range-coupled clipping bias (89.5 mm/m) is correlated across a window and must be gated, not averaged."
- "Measured frame-to-frame sensor noise is 6–12 mm (static-background differencing), against a 105 mm between-subject stature SD; the sensor is not what limits a single frame."
- "The frame-level oracle exceeds the metric model by ~8 pp, 1.5–3 pp *below* what two independent classifiers would give; the two representations fail together on a hard core of frames." (Add P-C1's result when run.)
- "Identifiability is a property of the individual within the cohort: 6 of 28 people are never identified by the metric model in a 2.5 s window and 4 always are; fusion changes which people those are."
- Do **not** write: "sensor quantisation of ~25 mm averages as 1/√N" (the 25 mm is the predicted disparity step, not the measured noise); "the CNN gains nothing from integration" (the augmented recipe gains 7–10 pp); or "explainable" without the operational definition above.

---

## 7. Literature — verification status

| item | status | record |
|---|---|---|
| Doddington, Liggett, Martin, Przybocki, Reynolds, "SHEEP, GOATS, LAMBS and WOLVES: a statistical analysis of speaker performance in the NIST 1998 speaker recognition evaluation", ICSLP 1998 | **[V]** Crossref | DOI 10.21437/icslp.1998-244 |
| Yager & Dunstone, "The Biometric Menagerie", IEEE TPAMI 32(2):220–230, 2010 | **[V]** Crossref | DOI 10.1109/TPAMI.2008.291 (earlier: "Worms, Chameleons, Phantoms and Doves…", IEEE AutoID 2007, DOI 10.1109/autoid.2007.380583) |
| Daugman, "The importance of being random: statistical principles of iris recognition", Pattern Recognition 36(2):279–291, 2003 | **[V]** Crossref | DOI 10.1016/S0031-3203(02)00030-4. Content (degrees-of-freedom argument) paraphrased from memory; no numbers quoted |
| Adler, Youmaran, Loyka, "Towards a measure of biometric information", CCECE 2006, pp. 210–213; journal version "Towards a measure of biometric feature information", Pattern Analysis and Applications 12(3):261–270 (online 2008) | **[V]** Crossref | DOIs 10.1109/ccece.2006.277447; 10.1007/s10044-008-0120-3. Cite the journal version |
| Kittler, Hatef, Duin, Matas, "On combining classifiers", IEEE TPAMI 20(3):226–239, 1998 | **[V]** bibliographic (Crossref); the sum-vs-product robustness claim **[UNVERIFIED this session]** (Crossref carries no abstract; IEEE page not fetched) | DOI 10.1109/34.667881. Companion: Tax, van Breukelen, Duin, Kittler, "Combining multiple classifiers by averaging or by multiplying?", Pattern Recognition 33(9):1475–1485, 2000, DOI 10.1016/S0031-3203(99)00138-7 [V bibliographic] |
| Khoshelham & Elberink, "Accuracy and resolution of Kinect depth data for indoor mapping applications", Sensors 12(2):1437–1454, 2012 | **[V]** Crossref + PMC full text (PMC3304120) | DOI 10.3390/s120201437. Verified content: depth point spacing ≈2 mm at 1 m, "close to 2.5 cm" at 3 m, ≈7 cm at 5 m (§4.3, Fig. 11); random error from a few mm at 0.5 m to ≈4 cm at maximum range (§4.3, Fig. 10), with σ_Z ∝ Z² (§3.2); 11-bit disparity. Our `predicted_step_mm` (21.9 mm at 2.75 m, 30.6 mm at 3.25 m) matches their 2.5 cm at 3 m; our *measured* 6–12 mm temporal σ is below their random-error curve at 3 m, consistent with BIWI depth being post-processed (§12.2) |
| Barbosa, Cristani, Del Bue, Bazzani, Murino, "Re-identification with RGB-D sensors", ECCV 2012 Workshops, LNCS, pp. 433–442 | **[V]** Crossref + saved extract | DOI 10.1007/978-3-642-33863-2_43. Extract: 79 people, different days and clothing; height among the most relevant features by nAUC (lines 400–402) |
| Andersson & Araujo, "Person Identification Using Anthropometric and Gait Data from Kinect Sensor", AAAI 2015 (vol. 29 no. 1) | **[V]** Crossref + saved extract | DOI 10.1609/aaai.v29i1.9212. Extract: 140 individuals, KNN/SVM/MLP, accuracy vs gallery size (Fig. 4) — the precedent for the cohort-size axis |
| Karianakis et al. 2018 (RTA), BIWI depth | **[V]** saved extract lines 545–554 | CNN single-shot 25.4; CNN-LSTM + average pooling 45.7; RTA 50.0 |
| Haque et al. 2016, BIWI depth Table 3 | **[V]** saved extract lines 400–416 | 3D CNN + Avg Pooling 27.8; 4D RAM 45.3 multi-shot; single-shot 3D RAM 30.1 (Table 2) |

---

## 8. Open checks, in priority order (all re-reads of stored arrays or `metric_features.npz`; none retrains)

1. §3 consistency/entropy/persistence statistics on the five best-recipe `cm_*.npz` (P-B1) with the arithmetic-mean re-aggregation beside the product rule (P-B2) — the check that licenses the word "systematic".
2. i.i.d. ceiling from the gated confusion rows (P-A2) — gives the effective number of independent frames per 2.5 s for each model.
3. Both-wrong enrichment for `bot_clip` / distance band (P-C1) and the window-level oracle vs W (P-C2) — decides how the ~8 pp headroom is described.
4. Within-recording residual SD and slope of `stature_mm` (P-A1) — replaces the "25 mm sensor noise" sentence with measured numbers.
5. Adler relative entropy on the 12 features per frame and per 25-frame window (§4.1) — the information axis for fig 7.
6. Per-goat cause in millimetres (P-E2) and nearest-neighbour explanations for the 103 gated decisions (§5) — turns "explainable" from a promise into a table.
7. Seed stability of the confuser (P-B3) and wave 22 `tracklet_acc` (P-B4).
