# Adversarial check of lens L1 (REPRESENTATION): why the CNN under-performed hand-computed geometry

Check of `/Volumes/Workspace/study/hiride2-results/lit-audit/09_cnn_failure_L1_representation.md` against the measured record. Written 2026-09-09, British English, single analyst, no sub-agents. Sources read in full or in the sections named: `HIRIDE_HANDOFF.md` §0, §5.1–5.4, §8.1–8.4, §8.6, §11.1b, §12.1–12.2, §13.9, §13.10, §13.12–13.15, §13.17, §13.18, §14.6–14.8; code `hiride_train.py` (`spatial_head`, `attach_aux`, `scale_remove`, `centre_person`, `quantise_depth`, `apply_mask_condition`, `ArrayBatches._aug`, `TestCurve`, `tracklet_scores`, the `--aux` block, the EarlyStopping block), `hiride_metric.py::frame_features`, `hiride_metric_floor.py`, `hiride_sequence.py`, `hiride_fuse.py`, `hiride_data.py` (`eligible_mask`, `_tail_val`, `_policy_cross`, `POLICIES`), `make_runs.py` (WAVE19, WAVE22, the `--track-test` line); stored artefacts `stats_final.json`, `sequence_gated_best.json`, `sequence_ungated_best.json`, `fusion.json`, `cohort.json`, `range_profile.json`, `signal_diagnostic.json`, `signal_diagnostic_pc400_prep.json`, `noise_floor.json`, `tables.tex`, `collect_final_out.txt`; literature extracts `haque2016.txt`, `karianakis.txt`, `lidargait.txt`, `xiao2021.txt`.

Verdict key. SUPPORTED = a measurement in the handoff or code is direct evidence for the mechanism as an explanation of the CNN's deficit. CONTRADICTED = a measurement is evidence against it in that role. UNTESTED = nothing measured bears on it. A mechanism can be a correct DESCRIPTION of what the code does and still be CONTRADICTED as an EXPLANATION; the two are separated below. Literature: [V] = confirmed in a saved extract this session; [UNVERIFIED] = not re-fetched here (the analyst's own [V] tags for arXiv/Crossref items were not re-checked and no DOI is restated).

---

## 0. The cross-cutting problem: the deficit the lens explains is not where the record puts it

The lens is titled as an account of why the CNN "recovered the outline and not the millimetres" and under-performed the twelve metric scalars. The record locates that under-performance precisely, and it is narrower than the lens assumes.

| R4, `scale_removed`, best recipe (`alexnet/stripe/aug8/tf10`), same frames | CNN | metric RF | source |
|---|---|---|---|
| single frame, ALL 5,642 test frames | **18.36 %** | **18.83 %** | `sequence_ungated_best.json`, W = 1 |
| single frame, full-body frames only (2,933) | 21.60 % | 28.86 % | `sequence_gated_best.json`, W = 1 |
| single frame, clipped frames only (2,709, back-solved) | ≈ 14.9 % | ≈ 8–9 % | arithmetic on the two rows above; §13.12(b) gives 9.3 % for the metric RF |
| 25 frames, gated (103 decisions) | 28.16 % | 43.30 % | §14.6 |
| whole tracklet, gated (28 decisions) | 29.3 % | 50.7 % | `sequence_gated_best.json`, W = 0 |

So on a single frame drawn from the whole test set the CNN and the metric features are tied. The CNN is BETTER than the metric pipeline on clipped frames and WORSE on whole bodies; the ~15 pp headline gap is produced by (i) the full-body gate, which discards the frames where the metric measurement is invalid and the CNN happens to be more robust, and (ii) integration, from which the metric posteriors gain ~14 pp and the CNN's ~7 pp. Any representational explanation therefore has two specific things to explain: why the CNN is 7 pp behind on whole bodies, and why its per-frame errors are less independent across a tracklet. It does not need to explain a single-frame representational ceiling, because on all frames there is none to explain. Several of the analyst's evidence items compare the metric RF against the trivial-cue image FLOOR (5.35 %), which is a random forest on thirteen pixel scalars (§5.4), not the CNN; the like-for-like CNN number is 18.36 %.

The second cross-cutting fact: on every person-only representation, R3 (1,821 training frames, same day, same clothes) and R4 (7,894 frames, other day, other clothes) score alike — gap-head `scale_removed` 13.76 vs 12.45 (§8.2; regenerated 13.62), `sil_scaled` 13.71 vs 14.59, `person` 8.65 vs 7.87, metric RF 15.56 vs 19.04 (§13.9). Neither a 4.3× larger frame count nor session proximity moves the person-only CNN. Whatever binds it is the number of identities and of recordings per identity (one), which is the axis wave 22 partly moves (50 identities, still one recording each).

---

## 1. Mechanism A — depth as intensity, not geometry (the network cannot form (u−cx)·z/fx)

**What the code establishes.** `hiride_metric.py::frame_features` computes `X = (xs − CX) * z / FX`, `Y = (ys − CY) * z / FY` with FX = FY = 575.816 and never passes any of these constants to the trainer; `build_alexnet` receives only the [0,1] depth image. That a conv–pool stack is not handed the focal length is a fact of the code. That it "has no gradient pressure to learn the product" is the analyst's conjecture.

**What the record says.**
- §13.17, pre-registered: handing the network the standing distance (`--aux dist`) changes nothing (mean delta −0.58 / −0.45 pp; at R3, where the power exists, a ceiling of +2.4 pp). The handoff's own conclusion: the scale story "is wrong as a SUFFICIENT explanation".
- §14.6 verdict 1, wave 19: handing the network the FINISHED twelve scalars at the head (`attach_aux`: z-scored on train rows, Dense(32, relu), concatenated after pooling) gives +0.18 / +1.39 / +2.65 pp at R4, inside seed variance (`stats_final.json`: `stripe/aug8/tf10/auxmetric scale_removed` 0.20 vs `none` 0.18). The same twelve scalars alone, in a random forest, reach 18.83 % on the same frames and 28.86 % on whole bodies.
- The oracle headroom is real (either − metric +7.7 / +9.0 / +8.3 pp, intervals clear of zero, §13.10, §14.6), so the information is complementary and available.

**Verdict: CONTRADICTED as an explanation of the deficit.** The network was given the missing factor and then the finished product, and neither moved it; so not being able to compute the product was not what held it back. What A's evidence items actually show is that the CNN does not USE metric size, which is not the same claim. Two of the items do not bear on the CNN at all: "19.04 % vs image floor 5.35 %" compares two random forests (metric scalars vs thirteen trivial pixel scalars), and "adding `stand_dist_mm` drops R1 67.97 → 57.59" is a statement about the metric RF taking a recording shortcut (§13.9), not about convolution.

What the wave-19 null does point at is stronger and different from A: a Dense head trained jointly with the image branch fails to exploit twelve scalars that a random forest exploits without difficulty. That is a joint-training failure (the image branch fits the within-session training labels first and the auxiliary branch receives no gradient), which belongs to mechanisms E/G, not to A.

**Cheapest settling analysis.** No training. (1) From the stored `cm_*.npz` posteriors of the `auxmetric` and `none` partners and `metric_features.npz`: mean |Δstature| over confused (true, predicted) pairs, for CNN, metric RF and aux-CNN. Prediction if A were right in its strong form: CNN ≈ population mean |Δstature| and aux-CNN ≈ RF; the wave-19 null predicts aux-CNN ≈ CNN (the scalars are ignored). (2) From the stored `history` in each `auxmetric` result JSON: training accuracy at the selected epoch — if it is near 100 % the auxiliary branch was starved. Wave 22 does not touch A (no `--aux` cells).

---

## 2. Mechanism B — `scale_removed` deletes absolute size and range; clipping remaps rows

**What the code establishes (all confirmed).** `scale_remove` crops to the userMap bounding box, sets the person's median depth to `SCALE_TARGET_DEPTH` = 3000/6000, resizes the crop so its height is `SCALE_TARGET_H` = 0.70 of the frame (width capped), pastes centred on a `fill` canvas; depth `fill` is 0.0 (`load_split_arrays`, line 656). Both factors of the metric product are set to constants — an exact description. `eligible_mask` states the clipping shift, and `range_profile.json` reproduces it: weighting `bot_touch` by bin counts gives train ≈ 9.2 % and test ≈ 47.8 %, the docstring's figures.

**What the record says about it as an EXPLANATION of the CNN deficit.**
- The CNN is the LESS clipping-sensitive of the two systems: back-solved clipped-frame accuracy ≈ 14.9 % (CNN) against ≈ 8–9 % (metric RF); on whole bodies the CNN is 21.6 % against 28.9 %. So the CNN's shortfall is largest where clipping is ABSENT.
- Removing the framing shift from training as well does not help the CNN: the trained `scale_removed/full_body` cell (`stripe/aug8/tf10`, eligibility `full_body` on train and test, 5 seeds) scores 0.20 ± 0.03 (`stats_final.json`) against 21.60 % for the ungated-trained model evaluated on the same gated frames. The metric RF under the same treatment: 28.06 → 28.89 (§13.12(b)). Neither system learns anything from unclipped training; the 7 pp whole-body gap survives the removal of the row-correspondence shift.
- The whitened aligned-pixel probe (`signal_diagnostic.json`, `range_bins`) does fall to 5.4 % in the 0–2000 mm bin where `bot_touch` is 98 %, so clipping does hurt an alignment-reading model — but it also falls to 12.7 % in the 3500+ mm bin where `bot_touch` is 0 % and the training pool holds 220 frames, so range covariate shift is not only clipping.
- "RGB gains from the same normalisation because its cue is scale-free": `person` 14.46 → `scale_removed` 17.72 at R4, seed sd 2–6 pp (§14.6 verdict 3); inside noise.
- "Keeping size does not rescue depth": `person` 9.22, `person_centred` 11.29, `scale_removed` 13.62 (regenerated §14.6) — an ordering with every pairwise interval overlapping; §14.6 withdrew every R4 condition-vs-full contrast.

**Verdict: SUPPORTED as a description of what the pipeline does; CONTRADICTED as an explanation of the CNN's under-performance.** Clipping and the base-anchored `w_XX` bands are the METRIC pipeline's defect (§13.12(a), §13.14 B) — the gate exists because the metric measurement is invalid on 74 % of walking frames. Transferring that defect to the CNN inverts the measured direction of the effect.

**Cheapest settling analysis.** Already done above from stored JSONs (clipped-vs-unclipped split, trained-full-body cell). If the analyst's `metric_scaled` condition (rescale by z_median/3000 rather than to a fixed height) is still wanted, it is new code and five R4 seeds; the prediction it tests (pixel height proportional to true height helps on whole bodies) is worth one wave only after wave 22 has shown whether the 50-class protocol closes the gap by itself. Wave 22 has no `full_body` cells and does not test B.

---

## 3. Mechanism C — translation equivariance and global pooling discard arrangement

**What the code establishes.** `spatial_head`: `gap` = GlobalAveragePooling2D; `stripe` = mean over width, rows kept; `flatten` = AveragePooling2D(2) then Flatten. The docstring's "60.5 % template vs 40.9 % GAP CNN" is the R1 whitened nearest-class-mean probe (`signal_diagnostic_pc400_prep.json`: `interior_only` whitened 0.640, `scale_removed` 0.607) against the gap cell then at 40.92.

**What the record says.**
- The R1 head asymmetry is real (`stats_final.json`: gap `scale_removed` 0.43 → stripe 0.65 / flatten 0.67; `interior_only` 0.38 → 0.65 / 0.70; rgb −0.2 pp, §12.1(b)). But it is CONFOUNDED with regularisation: `aug8` — random horizontal flip plus integer translation in [−8, 8] px (`ArrayBatches._aug`) — with the GAP head KEPT lifts R1 `scale_removed` 0.43 → 0.64 (3 seeds) and `sil_scaled` 0.53 → 0.68, matching the head change. If pooling had destroyed the arrangement, translation jitter could not have restored it. The R1 asymmetry therefore cannot be read as proof that GAP was discarding correspondence; it is at least as consistent with the gap-head from-scratch network over-fitting per-recording detail that either a spatial head or jitter regularises away.
- At R4, the rung that matters, every alignment-preserving variant lands in one cluster: gap 0.14, gap+aug8 0.16, stripe 0.17 (16.69 %), flatten 0.17, stripe/aug8/tf10 0.18 — spread inside the ~6 pp MDE (§12.2). §12.1(b)'s own R4 statement is "12.5–14.6 to 16.5–18.0".
- The strongest test of C at R4 is already stored: the whitened nearest-class-mean template on 96 PCA components of the ALIGNED pixels — a model with no pooling and perfect access to arrangement — scores 15.1 % (`scale_removed`), 14.4 % (`sil_scaled`), 15.6 % (`interior_only`) (`signal_diagnostic.json`, `whitened`). That is the GAP CNN's level, below the stripe CNN, and below the metric RF (18.83 %) on the same frames. The arrangement is linearly present at about 15 %; reading it fully does not close the gap to geometry. `te_centred` ≈ `whitened` (14.9 vs 15.1), so the R4 failure is not a fixable offset either (§11.1b's reading rule).
- "Re-centring is worth nothing at R1 and everything at R4" (23.31 → 23.81; 7.87 → 11.92): regenerated R4 values are 9.22 → 11.29 (§14.6), both inside their intervals. A within-noise ordering, consistent with C but not evidence for it.

**Verdict: SUPPORTED at R1 as a description (the head matters within session, and only for depth); CONTRADICTED as an explanation of the R4 deficit against geometry.** The head change and augmentation are the two levers that moved R4 depth from ~13 to ~18 %, and that gain is real; but a pooling-free template does no better than the pooled network at R4, so pooling is not what separates the CNN from the millimetre features.

**Cheapest settling analysis.** Zero-training: the whitened probe versus the stripe/flatten cells at R4 (done above). For the R1 confound: one gap-head `aug8` cell already exists at 3 seeds; two more seeds would make the aug8-vs-stripe comparison a proper 5 × 5 contrast. Wave 22 includes `scale_removed` under the gap head AND under `stripe/aug8/tf10` at 50 classes, so it re-measures the combined head+aug effect under the standard protocol — but it has no stripe-alone or aug8-alone cell, so it cannot separate the two.

---

## 4. Mechanism D — 2 bits ≈ 16 bits and silhouette ≈ depth imply outline reading

**What the code establishes, and where the analyst's arithmetic fails.** `quantise_depth` bins the FIXED 0–6000 mm range into 2^b bin centres and leaves 0 (the depth fill) as 0. Under `scale_removed` the person's median is set to exactly 3000 mm. Because 6000/2^b divides 3000 for every b ≥ 1, the median plane is a BIN BOUNDARY at every bit depth: at 2 bits the body (median extent 419 mm, roughly 2790–3210 mm) straddles the levels [1500, 3000) and [3000, 4500); at 4 bits (375 mm) it straddles [2625, 3000) and [3000, 3375). The analyst's premise that "the whole body sits inside one level" at 2–4 bits is false. What the coarse bit depths deliver is the outline PLUS a binary sagittal relief map (nearer than / farther than the median plane), not the outline alone.

Consequently the 1-bit explanation is also not what the code does. With depth fill 0.0, a 1-bit image has background 0, near-half body 0.25, far-half body 0.75: the person/fill boundary is INTACT. The 1-bit row's deficit (9.51) cannot be attributed to a damaged outline; the candidate is a high-contrast false edge inside the body at the median plane, which moves with pose. The handoff's own sentence — a threshold that "cuts through the body" — describes an interior artefact, not an outline one.

**What the record says.**
- R4 axis: 13.62 / 12.45 / 13.02 / 11.22 / 12.17 / 9.51 for 16/8/4/3/2/1 bits; `sil_scaled` 12.97 (§14.6). Flat within noise — but the noise is wide: the 2-bit cell has the largest seed sd in the R4 depth table (0.12 ± 0.05, `stats_final.json`), and 3 bits dips 2.4 pp. The axis bounds the interior's contribution at R4 to ≲ 3–4 pp; it does not put it at zero.
- R1: 47.62 → 33.61 from 16 to 2 bits. The analyst attributes the loss to "recording/distance" in the interior — but `scale_removed` has already normalised standing distance to 3000 mm. What the R1 drop shows is that body-relative relief carries WITHIN-RECORDING information (stance, clothing folds, the recording's pose regime), which is exactly the kind of cue §13.9 says fails across recordings.
- The physical clause — "the 26 mm quantisation step at 3 m exceeds the ~15 mm relief" — is contradicted by the campaign's own measurement. `noise_floor.json`: predicted step 21.9 mm at 2750 mm and 30.6 mm at 3250 mm, MEASURED noise 7.3 mm at 3000 mm and 9.4 mm at 3250 mm; §12.2 records that the disparity-step prediction "was wrong and was corrected by direct measurement". The sensor resolves the interior. So does the metric pipeline: the crown-anchored thickness family `ht_*` is the most range-stable family in the feature set (drift 0.10–0.91) and `metric+shape` lifts full-body R4 from 28.06 to 32.06 % (§13.13). The interior is informative when measured in millimetres; the CNN does not use it. D's conclusion describes what the network reads, not what the frame holds.
- `interior_only` 38.22 vs `scale_removed` 43.06 at R1 (§14.6): removing two rim pixels costs more than the interior contributes — consistent with D. But at R4 the whitened probe gives `interior_only` 15.6 % ≥ `sil_scaled` 14.4 %: the interior alone, linearly, is worth at least as much as the outline alone across sessions. Both are small.
- "Shipped-mask silhouette 12.97–16.67": 16.67 is the 08-31 `tables.tex` row that averaged every recipe into the gap-head row — §14.8: "Quote no mechanism or bit-depth number from the 08-31 tables.tex". The clean gap cell is 12.97; stripe `sil_scaled` is 0.17 and `stripe/aug8/tf10 sil_scaled` 0.19 (`stats_final.json`). A range across recipes must be labelled per recipe.

**Verdict: SUPPORTED in its conclusion (the trained CNN's cross-session accuracy is insensitive to depth precision below the median-plane split, and a normalised silhouette matches it); CONTRADICTED in two of its stated reasons (the "one level" arithmetic and the "sensor cannot resolve it" clause) and UNVERIFIED in a third (the 1-bit outline damage).**

**Cheapest settling analysis.** (1) Zero-training: the whitened probe on `sil_scaled` vs `scale_removed` vs `interior_only` at R4 is stored (14.4 / 15.1 / 15.6) — the interior's linear contribution is ~1 pp. (2) Zero-training: recompute the 2-bit and 1-bit images for a handful of frames and count the fraction of body pixels on each side of the median plane; the arithmetic above predicts ~50/50. (3) The analyst's `rim_matched` condition (zero boundary contrast, interior intact) is the right stronger test and is new code plus five seeds. (4) Saliency needs weights, which the result files do not store. Wave 22 has no `--bits` cells and does not test D.

---

## 5. Mechanism E — texture and local-statistics bias; RGB re-ID lives on clothing depth lacks

**What the record says.**
- The room finding is measured: depth `bg_plate` 38.57 % beats `full` 34.82 % at R3 with `person` at 8.65 % (§8.2); across the session change every background-carrying condition loses ~30 pp and every person-only one is flat (§13.9). This is a shortcut-learning result and the depth analogue of Xiao et al.'s background-only accuracy [V, `xiao2021.txt` abstract: models achieve "non-trivial accuracy by relying on the background alone"]. Note §5.3's caveat that BIWI's background plates differ between Training recordings by a median of only 55 mm, so "reads the room" at R1/R3 is largely "reads the standing spot and the recording", not the room.
- "CNN errors are systematic": gap-head CNN 18.40 → 20.71 across the whole window sweep against metric 28.86 → 50.71 (§13.12(c)); best recipe 21.6 → 28.2 → 31.4 (gated fig-7 curve) against 28.9 → 43.3 → 54.3. Supported as a relative statement — the CNN's per-frame errors are less independent — and the handoff already names a measured cause: augmentation "converts SYSTEMATIC error into noise" (§13.12), i.e. framing-dependent cues. That is a framing shortcut, not a texture one.
- "A depth crop has no texture so the only local statistics are the outline's" is contradicted at R1 by the same cells cited under D: the interior carries +14 pp within session (47.62 vs 33.61) and `interior_only` reaches 0.70 with the flatten head. Local interior statistics are exploitable; they encode the recording.
- ConvNeXt replicating AlexNet's ordering (R3 36.74 vs 34.82, R4 8.45 vs 6.70) is consistent with texture bias and equally with a data limit or with the full frame being dominated by the moved room; it does not discriminate.
- Mis-citation: "RGB 96.33 % at R1 ... with the full frame" — 96.33 is R1 rgb `scale_removed`; the full-frame R1 rgb figure is 91.94 (§8.1, §8.3).
- The literature is about ImageNet-trained networks on natural RGB images. Our AlexNet is trained from scratch on a textureless single channel; carrying "texture bias" across is an analogy, not a measured mechanism. The shortcut-learning framing (the analyst's unverified Geirhos 2020) is the one the record actually supports.

**Verdict: UNTESTED as a distinct mechanism.** The measured content (room/standing-spot shortcut within session; less integrable errors; augmentation randomising a framing cue) is real and already attributed in §8.2, §13.9 and §13.12 without any appeal to texture. Nothing measured distinguishes "texture bias" from "any easier within-recording cue".

**Cheapest settling analysis.** Zero-training, from stored `cm_*.npz`: lag-1 autocorrelation of the per-frame argmax (or of the posterior on the true class) within each test recording, for the gap CNN, the aug8 CNN and the metric RF. Prediction: gap CNN ≫ aug8 CNN > RF. Wave 22 records `tracklet_acc` for every cell, which gives the frame → tracklet gain per recipe under the standard protocol but not the autocorrelation itself. The `cm_*.npz` files are in the cluster archive (`runs_20260906.tar.gz`), not on the laptop.

---

## 6. Mechanism F — ImageNet-pretrained ConvNeXt on the full frame does not escape

**What the code establishes, against the analyst's description.** `build_convnext` does NOT feed "a 3-channel copy of the normalised depth with ImageNet mean/std". It builds a 1-channel stem by channel-averaging the pretrained stem kernel, uses `include_preprocessing=False`, and its comment states that depth "gets no ImageNet statistics"; the ImageNet normalisation is applied to RGB only. The F premise about the input is wrong in its specifics. The wider point — pretrained RGB filters do not perform unprojection — is not something any cell tests.

**What the record says.**
- ConvNeXt-Tiny/ImageNet exists only on the FULL frame (`stats_final.json`: `convnext_tiny full` at R1/R3/R4, depth and rgb). At R4 the full frame is dominated by the moved room (§8.2: `bg_hole` and `bg_plate` at 5.7 / 4.9 %, `full` 6.7 %), so ConvNeXt's 8.45 % measures whether pretraining overcomes a moved background, not whether it can read body geometry from a normalised crop. Whether pretraining helps `scale_removed` or `person` is UNTESTED — there is no such cell.
- The "too little data" reconciliation. The analyst's reading — "distribution proximity beats frame count" from R3 (1,821 frames, CNN 34.82, floor 8.55) vs R4 (7,894 frames, 6.70, 5.35) — holds only for the room-carrying full frame. On person-only representations R3 ≈ R4 (§0 above), so proximity buys nothing either once the room is removed. What the R3/R4 comparison actually isolates is that neither 4× more frames nor same-day training helps a 28-class person-only model; the binding quantity is identities × recordings per identity.
- §13.17 item 3 ("a data limit and not an AlexNet one") rests on ConvNeXt's full-frame 8.45 %, which for the reason above does not test the architecture on the representation in question.

**Verdict: SUPPORTED that pretraining does not rescue the full frame at R4 (8.45 %, inside the ~6 pp MDE of AlexNet's 6.70, §12.2); UNTESTED for the person-only representations, and the analyst's account of the input is contradicted by `build_convnext`.**

**Cheapest settling analysis.** The analyst's HHA-style orthographic input is the informative experiment (new code, five seeds, plus the bit-depth control on the height channel); it is the one proposal in the lens that would distinguish representation from capacity. Cheaper first step: ConvNeXt on `scale_removed` at R4 (5 seeds, existing code path) — if it lands with the AlexNet cluster at ~13–18 %, F's architecture half is closed. Wave 22 is AlexNet-only and does not test F.

---

## 7. Mechanism G (adjacent) — within-session validation rewards within-session cues

**What the code establishes (confirmed).** `_tail_val` takes the contiguous tail (`val_frac` 0.15) of each TRAINING recording with a guard (`--cross-val-guard`, default 50); `EarlyStopping(monitor="val_accuracy", mode="max", restore_best_weights=True)`; best weights restored unconditionally afterwards. For R3/R4, model selection therefore optimises same-recording accuracy. The comment at line 767 says so: validation "is never used for stopping or model selection" refers to the test curve, and confirms that EarlyStopping "still watches val_accuracy on the training session".

**What the record says.** The analyst's evidence (the mechanism-suite ordering; R0's 98.36 %) shows that within-session cues exist and dominate; it does not show that the selection criterion is what keeps the network on them. No cell has ever been trained with a cross-recording validation set, so G's causal claim has no measurement for or against it. Two stored facts bear on its plausibility: (a) the wave-19 null (twelve informative scalars ignored by a jointly trained head) is the signature of a shortcut fitting the within-session labels first, which is what a within-session stopping rule would lock in; (b) wave 4 ran every ConvNeXt full-frame cell with `--track-test`, so those result JSONs hold `test_curve` (per-epoch test accuracy) alongside `history["val_accuracy"]`.

**Verdict: UNTESTED.** Correct as a description of the code; no measurement bears on its effect size.

**Cheapest settling analysis.** Zero-training, on the wave-4 files: for each ConvNeXt cell compute max over epochs of `test_curve` minus `test_curve[best_epoch]`. That is an oracle bound on what within-session early stopping cost at R3/R4 for the full frame. If it is a few points, G is a footnote; if it is large at R4, run three R4 `scale_removed` seeds with validation carved from `Testing/Still` of the 28 (the only change is the split builder) and compare selected epochs. Wave 22 uses the same `_tail_val` and does not test G.

---

## 8. Mechanism H — why the field's depth networks do better: representation or data

**Literature status (checked in the saved extracts).**
- Haque, Alahi, Fei-Fei 2016 [V, `haque2016.txt`]: footnote 2 "We use a tensor of size 250 × 100 × 200"; Table 2 single-shot BIWI: 3D CNN 27.7, 2D RAM 24.7, 3D RAM 30.1 (Random 2.0, Human 6.7); Table 3 multi-shot BIWI: Energy Image 21.4, Energy Volume 25.7, 4D RAM 45.3. The analyst's 30.1 / 45.3 are correct. Not used by the analyst but decisive for H: within Haque's own table, moving from a 2-D to a 3-D input at fixed model class (2D RAM → 3D RAM) is worth +5.4 pp single-shot on BIWI — the only like-for-like measurement in the literature of what unprojected input buys on this dataset.
- Karianakis, Liu, Chen, Soatto 2018 [V, `karianakis.txt`]: Table 1 BIWI single-shot "Our method (CNN)" 25.4, 3D RAM 30.1; multi-shot CNN-LSTM + average pooling 45.7, with RTA 50.0, 4D RAM 45.3. The D_gp representation: person region normalised to [1, 256 − 56] with too-near/too-far clamped, body index used as mask before range normalisation — absolute depth is discarded, as in our `scale_removed`. The analyst's numbers and description are correct.
- Shen et al., LidarGait 2023 [V, `lidargait.txt`]: Fig. 8 camera silhouette 76.12, LiDAR silhouette 64.70, LiDAR depth 86.77; Table 3 RV 86.77 overall; 1,050 subjects, 25,239 sequences. Correct.

**What the record says.**
- The analyst compares Karianakis's 25.4 % single-shot against "our ~19–21 % gated single frame". The fair pairing is our UNGATED single frame, 18.36 % at 28-way (chance 3.57 %), against their 25.4 % at 50-way (chance 2.0 %) on all Walking frames. A 2-D CNN on a size-normalised depth crop with absolute depth discarded — the analyst's own regime — scores higher than ours at a harder chance level. That is evidence against a representational ceiling for 2-D CNNs on Kinect v1 crops and for a training-regime difference (split-rate RGB → depth transfer, body-index crop, 50 training identities).
- "Our gate/integration, not representation, is what moved the number" is half right: the gate and the window moved the METRIC number (19 → 28 → 43); the CNN's move from ~13 to ~18 % single-frame came from the head and augmentation (representation-side), and its integration gain (21.6 → 28.2) came from augmentation changing the error structure (§13.12).
- Karianakis's own ablation (RTA 50.0 vs uniform pooling 45.7) shows learned frame weighting is worth ~4 pp on BIWI cross-session; the handoff §14.8 already scopes our gate against this.
- The LidarGait scoping (sensor, range, n) is required and the analyst gives it; the record's own noise floor (6–12 mm, `noise_floor.json`) weakens the "Kinect cannot resolve interior" leg of that scope, leaving n = 28 and one recording per identity as the operative difference.

**Verdict: SUPPORTED as a literature description; the inference drawn from it is CONTRADICTED in one respect** — Karianakis 2018 IS a 2-D CNN on a range-normalised Kinect v1 crop out-performing ours single-shot at 50 classes, so the literature does not leave the representational reading unchallenged. Whether the difference is protocol (50 identities, more frames) or training regime (transfer) is exactly what wave 22 measures.

**Cheapest settling analysis.** Wave 22 covers this mechanism directly and is the only one of the eight it covers: `R4_standard_walking` / `R4_standard_still` × {`full`, `person`, `scale_removed` gap, `scale_removed stripe/aug8/tf10`, rgb `scale_removed stripe/aug8/tf10`} × 5 seeds, with `tracklet_acc` recorded per cell (`tracklet_scores`, the field's multi-shot number). Read-out: if `scale_removed stripe/aug8/tf10` reaches ~25 % single-shot and ~45 % tracklet at 50 classes, the shortfall against the field was protocol and the representational lens loses its main empirical support; if it stays near 18 / 30, the remaining difference to Karianakis is transfer pretraining and the crop, which is a training-regime story, still not a proof that a 2-D CNN cannot read geometry.

---

## 9. Corrections the analyst's report needs regardless of verdicts

1. "Image floor 5.35 %" is a random forest on thirteen trivial pixel scalars (§5.4), not the CNN; the CNN's like-for-like single-frame number is 18.36 % (`sequence_ungated_best.json`).
2. "12.97–16.67" for the silhouette mixes a clean gap cell with the buggy 08-31 `tables.tex` pooled row (§14.8 forbids quoting it). Use 12.97 (gap), 0.17 (stripe), 0.19 (stripe/aug8/tf10), each labelled.
3. "At 2–4 bits the whole body sits inside one level" is arithmetically false; the median plane at 3000 mm is a bin boundary at every bit depth.
4. "1 bit damages the outline": with depth fill 0.0 the outline survives 1-bit quantisation; the artefact is interior.
5. "26 mm quantisation step vs ~15 mm relief" contradicts `noise_floor.json` (measured 7–9 mm at 3–3.25 m) and §12.2.
6. "ConvNeXt received a 3-channel copy with ImageNet mean/std": `build_convnext` uses a channel-averaged 1-channel stem with no ImageNet statistics for depth.
7. "RGB 96.33 % at R1 with the full frame": 96.33 is `scale_removed`; full frame is 91.94.
8. The head-asymmetry argument omits the gap-head `aug8` cells (R1 `scale_removed` 0.64, `sil_scaled` 0.68 at 3 seeds), which match the head change without touching pooling.
9. §13.17 item 3's "a data limit and not an AlexNet one" rests on a full-frame ConvNeXt cell; carry the qualifier.
10. Every R4 person-only contrast straddles zero (§14.6); all orderings in the lens are within-noise orderings and must be presented as such (the analyst says this in §10 of the lens but not beside each item).

---

## 10. Literature table for this check

| item | status here | what was confirmed |
|---|---|---|
| Haque, Alahi, Fei-Fei, CVPR 2016 | [V] `haque2016.txt` | 250 × 100 × 200 tensor; BIWI single-shot 3D CNN 27.7, 2D RAM 24.7, 3D RAM 30.1; multi-shot 4D RAM 45.3, GEI 21.4, GEV 25.7 |
| Karianakis, Liu, Chen, Soatto, ECCV 2018 | [V] `karianakis.txt` | Table 1 BIWI: CNN 25.4 single-shot; CNN-LSTM+avg 45.7, RTA 50.0 multi-shot; D_gp range normalisation with body-index mask |
| Shen et al., LidarGait, CVPR 2023 | [V] `lidargait.txt` | Fig. 8: 76.12 / 64.70 / 86.77; Table 3 RV 86.77; 1,050 subjects, 25,239 sequences |
| Xiao, Engstrom, Ilyas, Madry, "Noise or Signal" | [V] `xiao2021.txt` abstract | background-only accuracy non-trivial; 87.5 % adversarial-background misclassification |
| Geirhos 2019; Baker 2018; Kayhan & van Gemert 2020; Islam, Jia, Bruce 2020; Eitel 2015; Gupta 2014; Hermann & Lampinen 2020 | [UNVERIFIED] in this check | not re-fetched; the analyst's tags stand on their own record; no DOI restated here |
| Geirhos et al. 2020 "Shortcut learning" | [UNVERIFIED] | the framing the record actually supports for E; fetch before citing |
| Munaro 2014; Wu 2017 (the other BIWI cross-session numbers in §14.8) | [UNVERIFIED] in this check | quoted from §14.8 only |

---

## 11. Summary table

| mechanism | verdict | deciding source | cheapest settling step | in wave 22? |
|---|---|---|---|---|
| A depth as intensity | CONTRADICTED (as explanation) | §13.17; §14.6 verdict 1; `attach_aux` | |Δstature| over confusions from `cm_*.npz` + `metric_features.npz`; train-accuracy of `auxmetric` cells from stored `history` | no |
| B scale_removed / clipping remaps rows | description SUPPORTED; explanation CONTRADICTED | `sequence_*_best.json` back-solve (CNN ≈ 14.9 % clipped vs RF ≈ 8–9 %); trained `scale_removed/full_body` 0.20 | already done from stored JSONs; `metric_scaled` only after wave 22 | no |
| C pooling discards arrangement | R1 description SUPPORTED; R4 explanation CONTRADICTED | `signal_diagnostic.json` whitened 15.1 % vs stripe 16.7 % vs RF 18.8 %; gap+aug8 R1 0.64 | two more gap+aug8 seeds at R1; whitened probe already stored | gap vs stripe/aug8 at 50 classes, confounded |
| D bits/silhouette ⇒ outline | conclusion SUPPORTED; two reasons CONTRADICTED, one UNVERIFIED | `quantise_depth` + fill 0.0 arithmetic; `noise_floor.json`; §13.13 `ht_*` | recount body pixels each side of the median plane; `rim_matched` condition | no |
| E texture / local-statistics bias | UNTESTED as distinct mechanism | §8.2, §13.9, §13.12 attribute the same facts to room/framing shortcuts | per-recording argmax autocorrelation from `cm_*.npz` | `tracklet_acc` per cell only |
| F pretrained ConvNeXt does not escape | full frame SUPPORTED; person-only UNTESTED; input description CONTRADICTED | `stats_final.json` (convnext only on `full`); `build_convnext` | ConvNeXt on `scale_removed` R4 (5 seeds); then HHA input | no |
| G within-session validation | UNTESTED | `_tail_val`, EarlyStopping code; no cross-recording-val cell | oracle-epoch gap from wave-4 `test_curve` | no |
| H field networks: representation or data | literature SUPPORTED; inference CONTRADICTED in part | `karianakis.txt` Table 1 (2-D CNN 25.4 at 50-way); `haque2016.txt` Table 2 (2D→3D RAM +5.4) | wave 22 read-out against 25.4 / 45.7–50.0 and 30.1 / 45.3 | YES — the only mechanism it covers |

**Bottom line for the manuscript.** The lens's safe content is descriptive: the trainer never receives intrinsics; `scale_removed` sets both factors of the metric product to constants; the trained CNN's cross-session accuracy is insensitive to depth precision and matches a normalised silhouette; the head and augmentation moved R4 depth from ~13 to ~18 %. Its causal content — that the CNN under-performs geometry BECAUSE it cannot form the product, BECAUSE pooling discards arrangement, or BECAUSE clipping remaps rows — is contradicted by measurements already in the record: supplying the product changed nothing; a pooling-free template does no better; the CNN is the more clipping-robust system and is tied with the metric features on a single ungated frame. What remains to be explained is the 7 pp whole-body gap and the CNN's less integrable errors, and the record's own attribution (per-recording framing shortcuts that augmentation partly randomises, with one training recording per identity) is a training-regime account, not a representational one. Wave 22 decides whether even that gap survives the field's protocol.
