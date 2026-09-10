# 09 — Adversarial check of the five proposed CNN-failure mechanisms (ERRORS lens)

Date: 2026-09-09. Analyst: Fable 5.1, working alone (no sub-agents).
Question: for each mechanism the ERRORS analyst proposed to explain why the CNN under-performed hand-computed geometry on BIWI RGBD-ID at R4, does the MEASURED RECORD support it, contradict it, or say nothing?

Sources read in full for this check: `HIRIDE_HANDOFF.md` §8.4, §12.1–12.3, §13.10, §13.12–13.14, §13.17, §13.18, §14.6, §14.8 and lines 556–600; `hiride_train.py` (`build_alexnet`, `attach_aux`, `apply_mask_condition`, `TestCurve`, `tracklet_scores`, EarlyStopping block, `cm_*.npz` writer); `hiride_sequence.py` (`agg_windows`, `windows`, `sequence_acc`, main loop); `hiride_fuse.py` (rules, gating imports); `hiride_data.py` (`eligible_mask`, `_policy_cross`, `_tail_val`, POLICIES); `hiride_metric_floor.py`; `hiride_metric.py` (ground-plane branch); `make_runs.py` WAVE22; stored artefacts `sequence_gated_best.json`, `sequence_ungated_best.json`, `fusion.json`, `noise_floor.json`, `stats_final.json`; saved extracts `karianakis.txt`, `haque2016.txt`; Crossref for the two menagerie papers.

Conventions. Every number below is copied from a named file or section; where I derive one by arithmetic I say so and label it indicative. Literature items are marked [V] (confirmed in a saved extract or on Crossref) or [UNVERIFIED]. The verdict vocabulary is the task's: SUPPORTED = a measurement in the handoff or code is direct evidence; CONTRADICTED = a measurement is evidence against; UNTESTED = nothing measured bears on it.

---

## 0. Summary of verdicts

| # | Mechanism | Verdict on the mechanism as stated | What is decisive | Covered by wave 22? |
|---|---|---|---|---|
| A | Metric vote integrates a partly independent per-frame error; sensor noise is not the binding term | **SUPPORTED** (constituent measurements); the fast/slow decomposition and its attribution to gait phase / arm swing are UNTESTED | `sequence_gated_best.json` metric 28.86 → 43.30 → 54.29 vs `met_agg` 28.86 → 23.30 → 16.43; `noise_floor.json` sigma 5.2–12.6 mm; §13.12a drift table | No |
| B | CNN confusions are systematic (same wrong identity per frame), recording memorisation via within-session early stopping; veto artefact to exclude first | **SUPPORTED** for the gap-head CNN's non-averaging error (§13.12c); "same wrong identity" and the memorisation cause are UNTESTED; the veto is UNTESTED and applies equally to the RF; three of the analyst's evidence items are wrong (see B.4–B.6) | §13.12c 18.40 → 20.71; best recipe 21.60 → 31.43; `hiride_data._tail_val`; `hiride_train.py:1126–1131, 1272` | Partly (tracklet arithmetic-mean number on every cell; 1.8× training pool but 50 classes) |
| C | Oracle only ~8 pp above metric BECAUSE both fail together on a hard core of corrupted frames | **CONTRADICTED** as a causal attribution by the record's own arithmetic; mild positive dependence is SUPPORTED (either 1.5–3 pp below a+b−ab); the clipped-frame attribution is UNTESTED and an indicative stratified estimate points against it | `fusion.json`, §13.10, §13.12b, `hiride_fuse.py` (no gate flag) | No |
| D | Biometric menagerie: identifiability is a property of the individual within the cohort and of the (person, system) pair | **SUPPORTED** as a description of the stored per-subject vectors; "property of the individual" is confounded with "property of that person's single test recording"; per-goat causes in mm UNTESTED; labels rest on 3–7 windows per subject and change with the gate | `sequence_gated_best.json` / `sequence_ungated_best.json` per_subject; §13.18 | No (wave 22 stores only a class-balanced mean) |
| E | Explainability, operationally defined | **SUPPORTED** in the hedged form the analyst already uses, with one mandatory correction: the frame is NOT gravity-aligned in the numbers that ran (camera −Y fallback, §12.3, `hiride_metric.py:151`) | `hiride_data.eligible_mask`; §13.13; §13.14; §12.3 | No |

---

## 1. Mechanism A — the metric vote integrates a partly independent per-frame error; the sensor is not the binding term

### A.1 What the record says (verified)

- `noise_floor.json` `predicted_step_mm`: 21.89 at 2750 mm, 30.57 at 3250 mm, 40.7 at 3750 mm. `noise_mm` (frame-to-frame sigma): Training 6.29 mm at 1500–2250 mm, 5.24 at 2500, 6.29 at 2750, 7.34 at 3000, 9.44 at 3250, 10.48 at 3500, 12.58 at 3750; Testing/Walking 4.19–6.29 mm at 1750–2250, 8.39–9.44 at 2500–3000, 10.48–11.53 at 3250–3750. The analyst's "6.3 mm (1.5–2.75 m) to 9.4–12.6 mm (3.0–3.75 m)" is a fair reading of the Training rows. **Do not quote the bins `Training|1250` = 691.9 mm and `Testing/Walking|1500` = 833.4 mm**; they are near-empty-bin artefacts and would be read as a sensor result.
- §12.2 records the 40–70 mm quantisation prediction as refuted by that measurement and says BIWI depth "is evidently post-processed". Verified.
- `sequence_gated_best.json` (`scale_removed`, best recipe, full-body gate): metric 0.2886 (W=1, 2,933 decisions) → 0.4330 (W=25, 103) → 0.5429 (W=100, 28) → 0.5071 (tracklet, 28); `met_agg` 0.2886 → 0.2330 (W=25) → 0.2000 (W=100) → 0.1643 (tracklet). `sequence_ungated_best.json`: metric 0.1883 → 0.3014 (W=25, 211) → 0.2585 (W=100, 41) → 0.2643 (tracklet); `met_agg` 0.1883 → 0.1564 (W=25) → 0.0537 (W=100). All of the analyst's figures reproduce.
- `hiride_sequence.py:81–92`: `sequence_acc` averages `log(prob + 1e-12)` over the window ("geo"); `agg_windows` (lines 49–71) averages the FEATURE vector then classifies with an RF trained on sliding-window means. So "posterior averaging helps, feature averaging hurts" is a measured contrast on identical test blocks (comment at lines 231–233).
- §13.12a: between-subject SD of `stature_mm` 105.2 mm, global slope 89.5 mm/m, drift over the 2.2-m walk ~197 mm; ten of twelve features have drift/sig > 1. §13.13: `metric+shape` 28.06 → 32.06 single-frame, 43.30 → 42.33 at W=25, 50.71 → 48.57 whole tracklet. Verified.

### A.2 Hostile points

1. **The fast/slow decomposition is inferred, not measured.** Nothing in the record computes the autocorrelation of the per-frame residual of any metric feature within a recording, nor a spectral peak at the step frequency for `w_45`/`w_60`. The attribution to "gait-phase and segmentation jitter of the percentile extremes, arm swing" is anatomically plausible and entirely UNTESTED. The record supports only the weaker statement: part of the per-frame error is uncorrelated enough between frames that a 25-frame vote removes it, and part is correlated with standing distance.
2. **"Posterior averaging is blind to the slow error" is too strong.** A monotone trend over the walk is partly averaged by a vote too: a window that spans a range of distances contains frames biased both ways relative to the window's mean distance. The gated curve keeps rising to W=100 (54.29 %), which is consistent with either "the slow term is small once clipped frames are gated" or "the vote partly averages it". The record cannot separate these.
3. **"Without the gate the curve peaks and falls" is inside the decision-count noise.** Ungated metric 30.14 % rests on 211 decisions (binomial SE ≈ 3.2 pp) and 25.85 % on 41 decisions (SE ≈ 6.8 pp). The 4.3-pp fall is not resolvable. The `met_agg` collapse (18.83 → 5.37 %) is resolvable in direction because it is monotone across every window, but its magnitude at W=100 (41 decisions) is not.
4. **Which "sensor noise" is meant.** `noise_floor.json` measures per-pixel temporal depth sigma. Stature is p99.5 − p0.5 of thousands of unprojected points, so per-pixel noise averages further within one frame; the frame-to-frame jitter of a percentile extreme is dominated by segmentation at the mask edge (1 px at 3 m ≈ 5 mm, §12.2), not by depth sigma. The analyst's conclusion (sensor not binding) is right, but the comparison "6–12 mm vs 105 mm SD" is the wrong pair of numbers for it; the relevant quantity — frame-to-frame sigma of `stature_mm` itself within a recording — is not in the record.
5. **A common-mode between-session term is not modelled.** §12.3 and `hiride_metric.py:151` show the "gravity-aligned" frame is the camera −Y axis on nearly every frame (ground-plane files empty on all but 130 of 28,037 frames). `hiride_metric.py:21` records a camera-pose change between sessions (background +620 mm, people ~30 cm closer). Any pitch change between the two days leaks depth extent into the height axis (my arithmetic: at 5° pitch, a 300-mm-deep body adds ≈ 26 mm to the h-extent while stature × (1 − cos 5°) removes ≈ 7 mm; net ≈ +19 mm, common to every subject in that session). This is exactly the "slow error" class and no number in the record tests it. §13.13 already suspects a height-axis SCALE error "probably the ground plane" as a third defect.

### A.3 Verdict

**SUPPORTED** for everything the analyst can point to a number for: sensor sigma is small, the vote helps, feature averaging hurts, the drift is range-coupled. **UNTESTED** for the specific decomposition (fast = gait/segmentation, slow = clipping trend) and for the between-session common-mode term. Wording ceiling for the paper: "the per-frame error of the metric features has a component that decorrelates within 25 frames and a component correlated with standing distance; the vote removes the first"; do not name gait phase or arm swing as the source without the test below.

### A.4 Cheapest settling test (stored data only)

From `metric_features.npz` plus the manifest (both already on disk; no training): for every Testing/Walking recording, (i) subtract the subject's Training median from each of the 12 features, (ii) regress the residual on `stand_dist_mm` within the recording, (iii) compute the lag-1…25 autocorrelation of what remains and the fraction of residual variance that is white at lag ≥ 5, (iv) for `w_45`/`w_60` compute the power spectrum and look for a peak near the step frequency (~1 Hz at 30 fps). Also compute the per-subject session offset of `stature_mm` after de-trending (the common-mode check for A.2 point 5). CPU minutes. **Not covered by wave 22**, which adds CNN cells and RF floors, not residual diagnostics.

---

## 2. Mechanism B — CNN confusions are systematic (same wrong identity every frame); recording memorisation via within-session early stopping; veto artefact to exclude first

### B.1 What the record says (verified)

- §13.12c (gap head): CNN 18.40 → 21.42 → 21.36 → 19.11 → 20.71 % from W=1 to whole tracklet on gated frames; metric 28.86 → 50.71. §13.12 also records that the best recipe (`alexnet/stripe/aug8/tf10`) changed this: `sequence_gated_best.json` CNN 0.2160 → 0.2816 (W=25) → 0.3143 (W=100) → 0.2929 (tracklet). §13.12's own sentence: augmentation "converts SYSTEMATIC error into noise".
- §13.10 `rescued` 15.2–17.2 % (old table); regenerated `fusion.json`: 15.60 (`sil_scaled`), 16.45 (`scale_removed`). §13.17 and §14.6 verdict 1: `--aux dist` and `--aux metric` change nothing within seed variance. ConvNeXt-tiny 8.45 % at R4 (§12.1 table, full frame).
- Code. `hiride_data._policy_cross` (line 313) carves validation with `_tail_val` (line 205) from the tail of each TRAINING recording, `val_frac = 0.15`, guard 50; `hiride_train.py:1126–1131` `EarlyStopping(monitor="val_accuracy", mode="max", restore_best_weights=True)`; line 1272 `prob=prob.astype(np.float16)`; `hiride_sequence.sequence_acc` default `agg="geo"` = mean of `log(p + 1e-12)`; `tracklet_scores` (line 788) uses the arithmetic mean. All as the analyst states.
- §13.17 item 1 and §8.4: 4 bits (16 levels over 6 m) costs nothing at R4; 2 bits ≈ 16 bits; best R4 depth input is `sil_scaled`. "It reads outline" is measured.

### B.2 Hostile points on the core claim

1. **"Systematic" is measured only operationally, for the gap head.** The flat gap-head curve is direct evidence that its errors do not average out. But the paper quotes the best recipe, whose CNN gains +9.8 pp from 1 to 100 frames (21.60 → 31.43) against the metric arm's +25.4 pp. The right sentence is quantitative — "the CNN's errors are less noise-like than the metric arm's" — not "the same wrong identity on every frame", which nobody has counted.
2. **The R3 reconciliation cuts against "too little data".** R3 (`Testing/Still` → `Testing/Walking`, same day, same clothes; `hiride_data.py:353–354`) trains on 1,821 frames, "4.3× fewer" than R4's 7,894 (handoff lines 564–566), yet the image floor scores 8.55 vs 5.35 % and the gap-head CNN on `scale_removed` scores 14.89 (R3) vs 13.54 (R4) on the byte-identical test set (§13.17 table). Four times the training frames, drawn from the other session, buys nothing. This is direct evidence that session shift dominates data quantity for the CNN as well as for the floor. It is NOT proof that data quantity is irrelevant: R3 also has an easier task (same room set-up, same clothes), so the two levers move together and the record cannot separate them. Handoff line 598–600 itself calls R3 "data-starved" and below the campaign's own viability criterion for CNN training, so quoting R3's CNN numbers needs that caveat. Verdict on "too little data" as the binding cause: **CONTRADICTED as sufficient**, untested as contributing.
3. **The memorisation mechanism is plausible and unmeasured.** The code makes early stopping select on within-session validation; whether that selected a memorising epoch is not in `stats_final.json` (no `best_epoch`, `val_acc` or test-curve fields at cell level). The `TestCurve` docstring (`hiride_train.py:767–773`) records validation saturating at 1.0 after one epoch for ImageNet ConvNeXt on RGB — not for the depth AlexNet cells in question. `cm_*.npz` stores `val_prob` and `val_truth`, so the val−test gap per cell is computable from stored data.

### B.3 The veto artefact — real, untested, and symmetric

float16 has no representable positive value below ≈ 6 × 10⁻⁸, so any softmax posterior smaller than that is stored as exactly 0 and enters the window mean as log(10⁻¹²) ≈ −27.6. Over W=25 that is −1.1 nats on the true class from one frame; a decision whose margin is under ~1 nat flips. So the concern is real as a PENALTY, not a hard veto, and nobody has counted zero-posterior frames. Two things the analyst did not say:

- **The RF arm has the same exposure, plausibly worse.** `rf.predict_proba` returns exact zeros for any class no tree voted for (300 trees, 28 classes), and `hiride_sequence.py:204, 256` applies the same `log(p + 1e-12)` to `p_rf`. The metric curve reaches 54 % under that rule. If the veto were what capped integration, it would cap the arm with more exact zeros first. This does not exonerate the CNN curve; it means the artefact cannot by itself explain the CNN–metric asymmetry.
- **The record already contains the comparator.** `tracklet_scores` (arithmetic mean) is now recorded as `tracklet_acc` on every cell trained after commit 2ffce28; for existing cells the same quantity can be recomputed offline from `cm_*.npz`. The whole-tracklet log-mean row in `sequence_*_best.json` (CNN 0.2929 gated, 0.2500 ungated) against the arithmetic-mean tracklet on the same cells is the veto test at W=0; a re-run of `hiride_sequence.py` with `agg="arith"` (a one-line change) gives it at every W.

Verdict on the veto: **UNTESTED**; must be run before "systematic" is claimed, exactly as the analyst says, but for both arms.

### B.4 Evidence item that is wrong: "~23M parameters"

The figure is copied from §13.17 item 3 and is contradicted by the record. §14.6 measured the AlexNet-gap model at **3,735,292** parameters (1 channel; 3,758,524 at 3). `build_alexnet` (`hiride_train.py:124–157`) is a 96-256-384-384-256 conv stack with a GAP or stripe head into a single `Dense(n_classes)`; the stripe head's `Flatten(H × 256) → Dense(28)` adds at most a few hundred thousand weights. The 58.5 M and "37.7M in one layer" figures in the code comment belong to the 2023 `Flatten → Dense(4096)` model, which this campaign removed. The argument (tens of independent samples per class against millions of weights) survives; the number does not. **Correct §13.17 and every downstream sentence to ≈ 3.7 M.**

### B.5 Evidence item that is wrong: "prior BIWI CNNs gain ~20 pp from whole-sequence pooling"

Verified in the saved extracts ([V], `karianakis.txt` Table 1; `haque2016.txt` Tables 2–3, BIWI column):

| model | single-shot | multi-shot | gain | what changed |
|---|---|---|---|---|
| Haque 2016 3D CNN → "3D CNN+Avg Pooling [9]" | 27.7 | 27.8 | **+0.1** | frame posteriors averaged |
| Haque 2016 3D RAM → 4D RAM | 30.1 | 45.3 | +15.2 | a different, temporal model |
| Karianakis 2018 CNN → CNN-LSTM+Avg Pooling | 25.4 | 45.7 | +20.3 | an LSTM trained on sequences |
| Karianakis 2018 CNN-LSTM → RTA | 45.7 | 50.0 | +4.3 | learned frame weighting |

The only like-for-like posterior pooling in the prior record gained 0.1 pp. The ~20-pp gains belong to recurrent models trained on sequences, which our fig-7 CNN is not. So the comparison "20 pp for them versus 8–10 pp for us" is a mismatch of operations; the honest comparator (Haque's 3D CNN, +0.1) makes our gap-head (+2.3) and best-recipe (+9.8) gains look ordinary or better, not deficient. Whether Table 3's row 8 is Table 2's row 6 averaged over frames is what the extract shows by name; the primary text should be checked before the paper says so. Karianakis's single-shot 25.4 and Haque's 3D RAM 30.1 are already in §14.8 and match the extracts.

### B.6 Evidence item to re-key: the §13.10 frame-level table

The analyst quotes cnn 14.60/12.45/11.92 and either 27.62/26.18/27.04. §14.6 says the wave-17/19 partners re-trained those seeds and the regenerated `fusion.json` reads cnn 12.97 (`sil_scaled`), 13.62 (`scale_removed`); either 26.55, 27.83; rescued 15.60, 16.45; either−metric +7.7, +9.0 with intervals clear of zero. The paper quotes only the regenerated artefacts (§14.6: "THIS section wins").

### B.7 Verdict

**SUPPORTED** for the operational core (the gap-head CNN's error does not average away; §13.12c is a direct measurement) and for "reads outline" (§8.4, §13.17). **UNTESTED**: same-wrong-class structure; memorisation via the stopping rule; the veto. **CONTRADICTED**: the ~23 M parameter count; "too little data" as the binding cause (R3 contrast); the "~20 pp pooling gain" framing of the prior literature.

### B.8 Cheapest settling tests

All but the last need only `cm_*.npz` (stored per cell: `pred`, `prob` float16, `truth`, `test_rows`, `val_prob`, `val_truth`) plus a deterministic RF refit:

1. Per test recording, the fraction of frames whose argmax equals that recording's modal prediction, for CNN and RF; and the number of distinct wrong classes per recording. This IS the "same wrong identity" test.
2. Count frames with `prob[true] == 0` per cell; recompute fig-7 with the arithmetic mean and with a floor of 10⁻⁴ instead of 10⁻¹²; compare the W=0 log-mean row with `tracklet_acc`. Do the same for `p_rf`.
3. Per cell, validation accuracy from `val_prob` against test accuracy; join `best_epoch`/`epochs_run` from the per-run JSONs (on the cluster). A best epoch of 1–2 with val ≈ 1.0 is the memorisation signature.
4. Data quantity (the only item needing GPU): subsample R4 training to 1,821 frames (~65 per subject) at the best recipe, 5 seeds, `--skip-existing` off with a new tag. Minutes per cell.

**Wave 22 coverage**: partial. It records the arithmetic-mean tracklet number on every new cell (item 2 at W=0 for those cells) and trains on 1.8× the frames — but with 50 classes and 22 distractor identities, so it is a protocol rung, not a clean data-quantity test. It leaves validation within-session, so item 3 is untouched.

---

## 3. Mechanism C — the fusion oracle is only ~8 pp above metric because both fail together on a hard core of corrupted frames

### C.1 What the record says (verified)

- §13.10 (old table) cnn/metric/either 14.60/18.83/27.62, 12.45/18.83/26.18, 11.92/18.83/27.04; independence a+b−ab = 30.7/28.9/28.5 (my arithmetic reproduces the analyst's); rescued 15.2/15.7/17.2 < 18.83; both-wrong 72–74 %. Regenerated `fusion.json`: `sil_scaled` cnn 12.97, either 26.55, independence 29.4 (−2.8); `scale_removed` cnn 13.62, either 27.83, independence 29.9 (−2.1). Mild positive dependence of correctness is a measured fact in both versions.
- `sequence_gated_best.json` geo−metric: W=1 +3.6, W=25 +3.7, W=50 0.0, W=100 +1.4 pp. Verified.
- `apply_mask_condition` → `scale_remove` normalises bounding-box scale (`hiride_train.py:437–438`); `sil_scaled` is the binary mask through the same normalisation. So "absolute scale removed by construction" is a code fact.
- Clipping: §13.12b says 25.6 % of `Testing/Walking` frames contain a whole body (74.4 % clipped); the gate retains 52.0 % of the cue-eligible R4 test frames (2,933 of 5,642; 47.8 % in `hiride_data.py:178`). Both are right; they have different denominators and must never be mixed in one sentence.

### C.2 The 1-bit evidence is mis-stated

The analyst writes that a "1-bit normalised silhouette costs the CNN nothing". §8.4 has two different rows: the 1-bit QUANTISATION row (9.51 %, R4 CI [3.93, 16.28], does NOT clear the majority rate) and the binary-silhouette row (`sil_scaled`, 14.59 then 12.97 after regeneration). §8.4 and §11.3 say explicitly: do not call the 1-bit row "the silhouette"; a global 3-m threshold cuts through the body. The claim that costs the CNN nothing is "2 bits ≈ 16 bits" and "`sil_scaled` ≈ `scale_removed`". Fix the wording.

### C.3 Hostile arithmetic: the "because" is wrong

The oracle's gain over metric is bounded by P(CNN right ∧ metric wrong) ≤ P(CNN right). With cnn = 13.6 % and metric = 18.8 % (`scale_removed`, regenerated), INDEPENDENCE would give a gain of 0.136 × (1 − 0.188) = **11.1 pp**; the record shows **9.0 pp**. Co-failure therefore removes ~2 pp of an 11-pp ceiling; the oracle is "only" ~8–9 pp above metric because the CNN is right on 13.6 % of frames, not because the two fail together. The analyst's causal clause assigns the smallness to the wrong term.

### C.4 Hostile arithmetic: does clipping explain the co-failure? (indicative)

Back-solving the clipped stratum from the gate: gap-head CNN unclipped 18.40 % (§13.12c) and ungated 13.62 % (`fusion.json`) → clipped ≈ (0.1362 × 5642 − 0.1840 × 2933)/2709 ≈ **8.4 %**; metric unclipped 28.86 %, ungated 18.83 % → clipped ≈ **8.0 %**. If the two were independent WITHIN each stratum, pooled either = 0.52 × [1 − 0.816 × 0.711] + 0.48 × [1 − 0.916 × 0.920] ≈ 0.52 × 0.4195 + 0.48 × 0.157 ≈ **29.4 %**, against 29.9 % unstratified and **27.8 % measured**. So stratifying by clipping accounts for about a quarter of the 2.1-pp shortfall; roughly 1.5 pp of positive dependence remains inside the strata. Caveats: the 18.40 % comes from an earlier set of gap-head seeds than `fusion.json`'s 13.62 %, and the two metric numbers come from different RF fits (28.86 from the sequence tool, 28.06 from the floor). This is indicative, not decisive — but it points the opposite way from "mostly the same corrupted frames". Both arms DO fail heavily on clipped frames (both ≈ 8 %); that part of the picture is right. What is unsupported is that this shared failure is what produces the sub-independence overlap.

### C.5 Verdict

**CONTRADICTED** as the causal explanation of the ~8-pp oracle headroom (C.3 is arithmetic on recorded numbers). **SUPPORTED** that a mild positive dependence exists and that clipped frames are hard for both. **UNTESTED** that the dependence lives on clipped/out-of-band frames; the indicative estimate in C.4 says most of it does not. The informational-overlap paragraph (silhouette = proportions + outline, metric = scale + six coarse proportions) is a correct description of the code but is not evidence of anything about failure overlap.

### C.6 Cheapest settling test (stored data only)

`hiride_fuse.py` imports `eligible_mask` but never passes `full_body=True` (lines 57, 159); it has no gate flag. Add one (one argument), and report cnn/metric/either/a+b−ab/rescued separately for unclipped, clipped, in-band and out-of-band strata from the stored `cm_*.npz` and `metric_features.npz`. CPU minutes. If either sits below independence within the unclipped, in-band stratum, C's attribution is dead. **Not covered by wave 22.**

---

## 4. Mechanism D — biometric menagerie: identifiability is a property of the individual within the cohort and of the (person, system) pair

### D.1 What the record says (verified)

`sequence_gated_best.json` `per_subject` (W=25, gated, 5 seeds): metric zeros {002, 003, 008, 010, 016, 019} (6), ones {017, 027, 031, 037} (4); the sorted vector matches the analyst's list exactly (28 values). Fusion (`geo`) zeros {006, 019, 023}, ones {027, 031, 037}; 002/003/008/010/016 move to 0.24/0.55/0.33/0.30/0.60 and 006/023 fall from 0.64/0.70 to 0. §14.6's fusion sentence matches. Cohort draw spread 60.0 pp at K=4 (§13.18), 58.2 pp on the 12-draw re-run (§14.6). Answer-when-sure numbers reproduce from the `coverage` arrays (metric 0.4 → 47 % at 62.0 %). All verified.

Literature: Yager & Dunstone, "The Biometric Menagerie", IEEE TPAMI 32:220–230, 2010, DOI 10.1109/tpami.2008.291 [V, Crossref]; Doddington et al., "Sheep, goats, lambs and wolves", ICSLP 1998, DOI 10.21437/icslp.1998-244 [V, Crossref]. That menagerie membership is partly algorithm-dependent is the question those papers frame; I did not read their texts here, so the attribution of the specific finding is [UNVERIFIED].

### D.2 Hostile points

1. **Each label rests on 3–7 windows from ONE recording.** The fractions in `per_subject` have denominators of 15, 20, 25, 30, 35 — five seeds times 3–7 windows — and the seeds re-score the same frames, so they are not independent observations. Under a common rate of 43.3 % the expected number of "never identified" subjects among 28 with ~3.7 windows is ≈ 3.4 and of "always" ≈ 1.3 (my arithmetic); the observed 6 and 4 exceed chance but no INDIVIDUAL label is resolvable.
2. **Membership changes with the gate, not only with the system.** `sequence_ungated_best.json` metric zeros are {000, 003, 010}; 002 → 0.029, 008 → 0.044, 016 → 0.167, 019 → 0.089. Only 003 and 010 are goats under both gatings of the same RF. So the analyst's "system-dependence" is real but stronger than stated: the labels are not even stable across a frame-selection rule.
3. **Person versus recording.** With one Testing/Walking recording per person, "identifiability is a property of the individual" cannot be separated from "a property of that person's test walk" (its distance band, its clipping fraction, the camera pose that day). The 12.5-pp lift of the gate on the metric arm is itself evidence that recording conditions move per-subject outcomes. The cohort spread (D.1) does establish the "within the cohort" half.
4. **No goat has been given a cause in millimetres.** Nothing in the record computes a between-session shift or a nearest-neighbour spacing in 12-D. The analyst's "should have a measurable cause" is a prediction, correctly hedged.

### D.3 Verdict

**SUPPORTED** as a description of stored per-subject outcomes and their dependence on the classifier (and on the gate). The "property of the individual" reading is UNTESTED and confounded with the single test recording; the millimetre causes are UNTESTED. Wording ceiling: "per-subject decision accuracy spans 0–100 % at the operating point and the set of never-identified subjects changes with the classifier and with frame selection; with one test recording per person and 3–7 decisions per subject, individual labels are not resolvable."

### D.4 Cheapest settling test (stored data only)

From `metric_features.npz`: per subject, the Training-to-Testing shift of each of the 12 features in units of that subject's Training SD, its Mahalanobis distance to the nearest other subject's Training centroid, and the rank of the true class by centroid distance; tabulate for the six metric goats and four sheep and for the fusion goats. Bootstrap the per-subject accuracies over windows to show which labels survive. CPU minutes. **Not covered by wave 22**: `tracklet_per_subject` there is a class-balanced mean, not the vector.

---

## 5. Mechanism E — explainability, operationally defined

### E.1 What the record says (verified)

- `hiride_data.eligible_mask` (line 166): `n_person_px ≥ 2000`, `frac ≤ 0.25`, and with `full_body` `top_touch ≤ 0` and `bot_touch ≤ 0`. None learned. §13.13: the gate "is a per-frame VALIDITY check rather than anything learned". §13.14: 12 named millimetre quantities and the band-position error model. §13.12a: drift table in mm/m. 36.3 % of R4 test frames outside the training p05–p95 distance band (§13.12b). Three failed attempts to hand the CNN the metric information (§13.10, §13.17, §14.6 wave 19). All verified.
- The analyst's hedges are right and must stay: per-decision attribution and per-goat causes are not computed; RF global importances are not per-decision explanations; the confidence threshold is a statistical refusal, not a geometric one.

### E.2 Hostile points

1. **The frame is not gravity-aligned in the numbers that ran.** §12.3: `_groundCoeff.txt` parsed on 130 of 28,037 frames; "the metric result above therefore used the CAMERA-Y FALLBACK". `hiride_metric.py:151–154`: `nvec = [0, −1, 0]`, `ground = 0.0`. §13.14's "GRAVITY-ALIGNED frame using the per-frame ground plane" describes the intended path, not the executed one, and §13.13 already suspects a height-axis scale defect "probably the ground plane". So "every input is a length in millimetres a person could re-measure" needs the qualifier: millimetres along the camera's vertical axis, which equals stature only if the camera is level; the `ground` column is 0 on essentially every frame and should be reported as such. This also revives a session common-mode term (A.2 point 5) that the drift model does not include.
2. **"Every failure mode was found and expressed in mm/m" is an overclaim.** The record names three defects: range-coupled clipping (expressed, 89.5 mm/m), band misplacement (expressed, `stature_error × f × gradient`), and the height-axis scale error (diagnosed by direction only, cause unconfirmed). Write "two of three".
3. **"Checkable" versus "explanatory".** The inputs are checkable; the classifier is 300 trees. Until a per-decision attribution exists, the pipeline is transparent at the input and opaque at the decision. The analyst says this; the paper must not slide from one to the other.
4. Two clipping denominators (74 % of all Testing/Walking frames; 47.8–48.0 % of cue-eligible R4 test frames). Use one and name it.

### E.3 Verdict

**SUPPORTED** in the hedged operational form, with the ground-plane correction mandatory. Wording ceiling: "every input is a length in millimetres computed by fixed arithmetic from the depth frame in a camera-vertical frame (the shipped ground plane is empty in this distribution); every refusal is a stated, un-learned geometric condition; two of the three identified failure modes are expressed as a drift per metre; per-decision attribution can be computed from the stored features but has not been."

### E.4 Cheapest settling test (stored data only)

(i) Report the `ground` column's mean over all frames (expected ≈ 0.005) in the feature-reference table. (ii) For each goat, list the two or three features whose between-session shift is largest in SD units (the D.4 computation) — this is the per-goat cause "in the same units". (iii) A per-decision attribution for the RF (path-based contributions summed over trees) on the 103 gated decisions. CPU minutes. **Not covered by wave 22.**

---

## 6. Corrections the paper must carry regardless of which mechanisms it uses

1. AlexNet parameter count is **3,735,292** (§14.6), not "~23 M" (§13.17). The handoff's own number is wrong; correct §13.17.
2. Quote the regenerated `fusion.json` (cnn 12.97/13.62; either 26.55/27.83; rescued 15.6/16.5), not §13.10's table.
3. The 1-bit quantisation row (9.51 %) is not the silhouette (§8.4); the binary-silhouette row is `sil_scaled` (12.97 % after regeneration).
4. `noise_floor.json` bins `Training|1250` and `Testing/Walking|1500` are artefacts; quote 5.2–12.6 mm.
5. The "gravity-aligned" frame is the camera −Y fallback on all but 130 frames (§12.3); say so wherever §13.14's description is reused.
6. Haque 2016's like-for-like pooling gain is +0.1 pp (3D CNN 27.7 → 3D CNN+Avg 27.8) [V]; the 15–20-pp multi-shot gains in the prior BIWI record belong to recurrent/attention models.
7. Two clipping prevalences (74 % of Testing/Walking frames; 48 % of cue-eligible R4 test frames) — pick one per sentence and name the denominator.
8. Ungated turnover of the metric curve (30.14 → 25.85 %) is on 211 vs 41 decisions and is not resolvable; do not describe the ungated curve as "peaking and falling".

## 7. Literature items touched

- Karianakis, Liu, Chen, Soatto 2018, Table 1 BIWI: CNN 25.4; CNN-LSTM+Avg Pooling 45.7; RTA 50.0; 3D RAM 30.1; 4D RAM 45.3; 3D CNN+Avg Pooling 27.8 — [V] `karianakis.txt` lines 534–554.
- Haque, Alahi, Fei-Fei 2016, Table 2 BIWI: 3D CNN 27.7, 3D RAM 30.1; Table 3 BIWI: 3D CNN+Avg Pooling 27.8, 4D RAM 45.3, GEI 21.4, GEV 25.7 — [V] `haque2016.txt` lines 326–415.
- Yager & Dunstone 2010, "The Biometric Menagerie", IEEE TPAMI 32(2):220–230, DOI 10.1109/tpami.2008.291 — [V] Crossref.
- Doddington, Liggett, Martin, Przybocki, Reynolds 1998, ICSLP, DOI 10.21437/icslp.1998-244 — [V] Crossref.
- The claim that menagerie membership is algorithm-dependent is what Yager & Dunstone test; their specific conclusions were not read here — [UNVERIFIED].
