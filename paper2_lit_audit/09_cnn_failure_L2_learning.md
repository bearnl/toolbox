# 09 — Why the CNN failed, Lens 2: learning regime, shortcuts and model selection

Analyst report for the paper-2 manuscript (BIWI RGBD-ID, Kinect v1; 28 people re-recorded on another day in other clothes). Written 2026-09-09 against `toolbox/HIRIDE_HANDOFF.md` §12.1, §13.10, §13.12, §13.13, §13.15, §13.17, §13.18, §14.6, §14.7 and the code in `toolbox/hiride_train.py`, `hiride_data.py`, `hiride_metric.py`, `hiride_metric_floor.py`, `hiride_sequence.py`, `hiride_fuse.py`. Every number below is copied from those sources; where a section and §14.6 disagree, §14.6 wins (its own rule). Literature items are tagged **[V]** (confirmed today on arXiv, Crossref, PMLR or in a saved extract under `lit-audit/extracts/`) or **[UNVERIFIED]**.

The lens in one sentence: *given the training signal it actually received, the network was never asked to find cross-session body geometry, and nothing in the loss, the data or the model-selection rule would have rewarded it for doing so.* Lens 1 (the geometry lens) explains what the invariant is; this lens explains why a from-scratch classifier trained on one recording per person, validated inside that recording and stopped on within-session accuracy, has no gradient pointing at it — and why bolting the missing quantities onto the head (`--aux dist`, `--aux metric`) could not change that.

---

## 0. The training signal the network actually received (measured record)

Everything in this lens rests on five facts about how the R3/R4 cells were trained. They are read from the code, not the summary.

**F1 — One recording per identity, 28 classes, from scratch.** `hiride_data._policy_cross` (lines 313–345): for R4 the training pool is every `Training` recording of the 28 subjects who also appear in `Testing`; the test set is their `Testing/Walking` frames, byte-identical between R3 and R4. BIWI has exactly one `Training` recording per person, so each class is one room state, one standing trajectory, one outfit, one lighting, one camera pose. Training pool at R4 ≈ 7,894 frames (handoff ladder table, line 522), i.e. ≈ 282 frames per class before the validation carve (derived, 7,894/28). AlexNet is built from random initialisation (`build_alexnet`, no pretrained weights); ConvNeXt-tiny is an optional ImageNet-initialised check.

**F2 — Validation is carved from the tail of the same training recording.** `_tail_val` (lines 205–219): the last 15 % of each training recording's frames (`val_frac=0.15`) become validation, with a 50-frame guard removed from the training side (`--cross-val-guard`, default 50, `hiride_train.py` line 826: "the validation tail carved from each TRAINING recording"). So the validation set for a cross-session rung is a *within-session, same-recording, guard-50 block split* — structurally an R2-like split — while the test set is a different day and different clothes.

**F3 — Model selection maximises within-session validation accuracy.** `hiride_train.py` lines 1126–1131: `EarlyStopping(monitor="val_accuracy", mode="max", patience=12, restore_best_weights=True)`, 60 epochs, Adam, sparse cross-entropy, mixed fp16. Lines 1141–1147 restore the best-val weights unconditionally (the tf.keras 2.15 bug fix). `TestCurve` (lines 764–773) exists precisely because of the concern this lens raises — its docstring records that "the within-session validation set saturates at 1.0 after one epoch" for ImageNet ConvNeXt on RGB, so the selected epoch is the first one. Whether depth AlexNet validation also saturates is **not recorded in the sections read** (per-cell `best_epoch`, `epochs_run`, `hit_epoch_cap` are written to each `results_*.json`, lines 1211–1213, and `val_prob` is stored since wave 19, lines 1273–1276 — see §8 for the check).

**F4 — Consecutive frames are near-duplicates.** `hiride_adjacency.py` (handoff lines 919–932): at lag 1, median |Δdepth| 80.6 mm, silhouette IoU 0.954, |ΔRGB| 2.4/255. The guard sweep at matched n_train (line 572): guard 0 → 31.81 %, 25 → 26.18, 50 → 19.44, 100 → 18.70, 150 → 17.73 (trivial-cue floor). §13.17 item 3: independent samples per class "number in the tens".

**F5 — Within a session, the room and the framing carry more identity than the person.** R3 depth: background-only with the person inpainted from the recording's own plate 38.57 % beats the full frame 34.82 % while the person alone scores 8.65 % (handoff lines 310–311, 854–859). The 13-scalar detector-metadata floor reaches 89.38 % at R0 and 17.73 % at R1 g150 (line 522), with `cent_y` 0.154, `bbox_h` 0.140, `cent_x` 0.109 as the top attributions at the strictest guard (line 623) — position and apparent size in the frame, not body shape.

These five facts are the premises. The mechanisms below are consequences of them, each tied to a measurement and to a prediction that can still be checked.

---

## 1. Mechanism A — model selection on a criterion that cannot see the target

**Claim.** Early stopping on within-session `val_accuracy` selects the epoch at which the network best recognises *the tail of the same recording it trained on*. Every cue that is constant within a recording — background plate, standing trajectory, framing, clothing, lighting — is fully predictive on that validation set and vanishes or inverts at test. The selection rule therefore has no resolution for the quantity the paper cares about (cross-session accuracy), and can actively prefer epochs that have memorised within-session cues.

**Our evidence.**
- F2 + F3: the validation split is a guard-50 block inside the training recording; the monitored metric is accuracy on it.
- F5: at R3, inpainting the person *out* raises accuracy (38.57 vs 34.82). A validation set drawn from the same recording rewards exactly that behaviour.
- The R3-vs-R4 contrast on byte-identical test frames: the floor trained on 1,821 same-session frames scores 8.55 % where 7,894 cross-session frames score 5.35 % (line 564–566); the CNN on the full frame goes R3 34.82 → R4 6.70 (line 1785). "Session shift, not data quantity, is what breaks it" (line 556). A selection rule computed inside the training session is blind to the very shift that decides the number.
- ConvNeXt-tiny/ImageNet on the full frame: R1 78.02, R3 36.74, R4 8.45 (§12.1 table). Pretraining lifts every within-session rung and leaves the cross-session rung at the floor — the pretrained features make the within-session shortcut *easier* to fit and the validation set rewards that immediately (the `TestCurve` docstring's one-epoch saturation).

**Literature.** Kapoor & Narayanan 2023 **[V]** (Patterns 4(9):100804, DOI 10.1016/j.patter.2023.100804; arXiv 2207.07048): a taxonomy of eight leakage types in ML-based science including a test set not drawn from the distribution of scientific interest and non-independence between train and test samples (the specific labels L3.1–L3.3 are recalled, not re-read today — **[UNVERIFIED]** for the labels, **[V]** for the eight-type taxonomy and the model-info-sheet proposal). Hammerla & Plötz 2015 **[V]** (UbiComp 2015, DOI 10.1145/2750858.2807551, pp. 1041–1051): pairwise similarity between temporally adjacent samples biases cross-validation in activity recognition — the same failure, on wearables. Beery, Van Horn & Perona 2018 **[V]** (ECCV 2018, arXiv 1807.04975): recognition trained and validated at known camera-trap locations generalises poorly to new locations. Note what these do *not* say: none of them concerns the stopping rule; the point that model selection, not only evaluation, inherits the leak is ours to state (and it is a small one — the paper should not claim it as a contribution).

**Prediction (checkable).**
1. Tabulate `best_epoch` across the R4 depth cells from the stored `results_*.json`. If Mechanism A binds, `best_epoch` clusters early (val saturates) and is uncorrelated with the cell's test accuracy across seeds.
2. Using `--track-test` on one R4 cell per recipe (already implemented; wave 4 recorded per-epoch test curves for ConvNeXt), compare the val-selected epoch's test accuracy against the max over epochs. A gap larger than the seed SD (1–4 pp) means the stopping rule cost accuracy; a gap inside the seed SD means the regime, not the rule, is the limit. Either result is reportable; neither may be used to *pick* a number.
3. Train the same cells with no early stopping (fixed 60 epochs). Lens 2 predicts a change inside seed variance at R4 — the criterion was never informative, so removing it changes little.
4. Report the within-session validation accuracy at the selected epoch beside the R4 test accuracy in the paper (both are now computable from `val_prob`). The expected picture — validation near ceiling, test 6–19 % — is itself the cleanest single statement of Lens 2.

---

## 2. Mechanism B — shortcut learning: the easy feature fits the loss, the hard invariant is never learned

**Claim.** The training set offers several features that are perfectly predictive of identity within it and cheap to compute from pixels — the background plate, the standing position and apparent size in the frame, and the silhouette outline. The cross-session invariant, metric body geometry, is a *hard* feature: recovering millimetres from a depth image needs pixel extent × depth value ÷ focal length, a multiplicative interaction that convolution and pooling are built to be invariant to (`attach_aux` docstring, `hiride_train.py` lines 96–104). Gradient descent on cross-entropy learns the easy features first, drives the loss down with them, and thereafter supplies almost no gradient to anything else. The network does not "fail to generalise" body geometry; it never represents it.

**Our evidence.**
- *Which features it uses.* Depth-value precision does not matter at R4: 16 bits 13.62 %, 8/4/3/2 bits 12.45/13.02/11.22/12.17, 1 bit 9.51; the normalised binary silhouette 12.97–16.67 (§14.6, fig 4). Interior relief — the part of the depth image that carries 3-D shape — contributes nothing measurable; the outline does. §13.17 item 1: "It reads outline, not 3D shape." (Scoped, per ESSENCE C2, to single-frame Kinect v1 and this architecture — not a general claim.)
- *The shortcut is removable and removing it helps, in ordering.* R4 depth: full 6.70, person 9.22, person_centred 11.29, scale_removed 13.62 (§14.6). Each edit removes an easy within-session cue (plate, position, size) and the cross-session number rises. No single contrast clears zero at n = 28; the ordering across cells and rungs is the evidence (§14.6, §13.11).
- *Augmentation changes the character of the error.* Random ±8 px shift + horizontal flip on training batches only (`ArrayBatches._aug`, lines 714–728; "no scaling", deliberately). The gap-head CNN gained nothing from temporal integration (16.39 → 19.61 → 20.00 across the window sweep) while the augmented recipe rises steadily (23.40 → 32.62 → 37.78 in the earlier draw; 21.6 → 28.2 → ~31 % in the final fig-7 curves) (§13.12, §14.6). Making the network invariant to framing removes a shortcut and converts a systematic error into a noisy one — which is what shortcut removal should look like.
- *Pretraining does not rescue.* ConvNeXt-tiny/ImageNet on the full frame: 8.45 % at R4 (§12.1). A better feature extractor, fed the same training signal, takes the same shortcut.
- *The other way round: even the random forest takes the shortcut when offered.* Adding `stand_dist_mm` to the 12 metric scalars drops R1 67.97 → 57.59 and R3 15.56 → 10.76 (§12.1, `metric+nuisance`). The shortcut is a property of the training distribution, not of neural networks.

**Literature.** Geirhos et al. 2020 **[V]** (Nature Machine Intelligence, DOI 10.1038/s42256-020-00257-z; arXiv 2004.07780): shortcut learning as decision rules that fit the benchmark and "fail to transfer to more challenging testing conditions". Shah et al. 2020 **[V]** (NeurIPS 2020, arXiv 2006.07710): networks can "exclusively rely on the simplest feature" and remain invariant to complex predictive ones — the extreme simplicity bias. Pezeshki et al. 2021 **[V]** (NeurIPS 2021, arXiv 2011.09468): gradient starvation — cross-entropy is minimised by a subset of features and the others receive vanishing gradient. Hermann & Lampinen 2020 **[V arXiv 2006.12433; venue UNVERIFIED — Crossref returned only later papers by these authors]**: with redundantly predictive features the network represents the one more linearly decodable at initialisation, and easy features suppress harder ones. Arpit et al. 2017 **[V]** (ICML 2017, arXiv 1706.05394): networks "prioritise learning simple patterns first". Zhang et al. 2017 **[V]** (ICLR 2017, arXiv 1611.03530): standard CNNs "easily fit a random labeling of the training data" — capacity to memorise is not in question.

**Prediction (checkable).**
1. *The Hermann & Lampinen test on our own network.* Linear decodability of `stature_mm`, `w_15`, `w_30` (the low-drift columns of §13.12) from the penultimate features of (a) an untrained and (b) a trained R4 `scale_removed` AlexNet, on training frames. Lens 2 predicts no increase from (a) to (b) — training amplified the outline/framing features and did not amplify metric height or width. If decodability *does* rise substantially, the network represents geometry and the failure lies elsewhere (Lens 1's territory).
2. Training loss/accuracy at R4 should reach near-ceiling within a few epochs; `hist.history` is available in a re-run with logging. If training accuracy does not saturate, the gradient-starvation reading is weakened.
3. Per-class confusion at R4 should be *stable across frames of one test recording* (the same wrong identity, frame after frame) — §13.17 item 2 and the flat integration curve already say so for the gap head; the prediction is that the confused-with identity is the training subject whose recording is closest in standing position / plate to the test recording, which `cues.npz` (`p_med`, centroid) can test directly.

---

## 3. Mechanism C — why `--aux dist` and `--aux metric` changed nothing

**How the auxiliary input is wired (read from `hiride_train.py`).**
- `attach_aux` (lines 95–121): `aux = Input(aux_dim)` → `Dense(32, relu)` → `Dropout(0.2)` → `Concatenate([x, a])`, where `x` is the pooled image feature vector *after* `Dropout(0.5)` (line 154). The concatenation feeds the single `Dense(n_classes, softmax)` output (line 156). There is no auxiliary loss, no gating, no modality dropout, no separate head. The aux path holds 12×32 + 32 = 416 weights plus 32×28 into the softmax; the image path contributes 256 features (gap head) or H×256 (stripe head) into the same softmax.
- `--aux dist` (lines 1069–1077): `p_med` from `cues.npz`, standardised on TRAIN rows only, one scalar.
- `--aux metric` (lines 1078–1106): the 12 `BASE_METRIC` columns, `nanmean`/`nanstd` on TRAIN rows only, `sd == 0 → 1`, then `nan_to_num(…, 0.0)` so frames without metric features (person < 200 px) receive the train mean. Frame set unchanged relative to the `--aux none` partner — the pairing is exact.
- Shuffling permutes aux rows with the image rows (line 695–697), and `ArrayBatches.__getitem__` returns `[xb, A[idx]]` so the aux vector is a second input, not a target (the earlier bug in lines 750–753 is fixed).

**Claim.** The auxiliary branch receives gradient only in proportion to the classification residual (p − y) of the whole model. With one recording per class and near-duplicate frames (F1, F4), the image path alone fits the training labels early; the residual collapses and the aux weights stop moving. This is gradient starvation in exactly Pezeshki et al.'s sense, and it is *predicted* by Mechanism B rather than being a separate story. Two things compound it: (i) early stopping on within-session validation (F3) cannot distinguish a model that uses the aux vector from one that ignores it, because both are near-ceiling within the session, so selection never favours aux use; (ii) at test the z-scored aux vector is out of the training range — 36.3 % of R4 test frames sit outside the training pool's p05–p95 standing-distance band and `stature_mm` drifts 89.5 mm/m against a 105 mm between-subject SD (§13.12) — so even a head that had learned a small aux weight would apply it to shifted inputs. Point (ii) is secondary: the random forest generalises from the *same* 12 numbers to 19.04 % frame-level / 28.86 % gated, so the information survives the shift; it is the network that never learned to read it.

**Our evidence.**
- Wave 17, `--aux dist`, pre-registered with a stated falsifier (§13.17): mean delta −0.58 pp (size-kept) and −0.45 pp (size-removed); every interval straddles zero; at R3 the intervals are ±2.4 pp, so distance buys at most +2.4 pp against a 10 pp gap to the metric features. The predicted separation between size-kept and size-removed conditions did not occur.
- Wave 19, `--aux metric` (§14.6 Verdict 1): R4 person_centred +0.18, scale_removed +1.39, sil_scaled +2.65 pp; R3 +0.24, −0.89, +0.66 — every difference inside the 1–4 pp seed SD.
- The oracle headroom is real on the same frames: either−metric +7.7 / +9.0 / +8.3 pp with intervals clear of zero (§14.6, regenerated fusion table). So the 12 scalars *do* carry information the CNN lacks, and a model given both still does not use them.
- Three ways of handing over the same information — posterior fusion (§13.10), the distance term, the 12 scalars at the head — all fail; the common factor is that none of them changes the training signal.

**Literature.** Pezeshki et al. 2021 **[V]** (gradient starvation); Shah et al. 2020 **[V]** (the network stays invariant to the complex feature even when it is predictive); Hermann & Lampinen 2020 **[V arXiv]** (easy features suppress harder ones; representations of a model trained on both resemble those trained on the easy one alone).

**Prediction (checkable, ranked by cost).**
1. *Test-time aux ablation, no retraining if weights were kept; otherwise one cell.* Feed the trained `--aux metric` model its test images with the aux vector set to zero (the train mean). If accuracy is unchanged within seed SD, the aux branch is functionally unused. This is the direct test of the claim and costs minutes.
2. *Weight movement.* ‖W_aux‖ of `Dense(32)` at the restored best epoch versus its initialisation, compared with the same ratio for the last conv block. Starvation predicts the aux layer barely moves.
3. *Aux-only control.* Train the same head with the image input zeroed (or a 12-input MLP with the same Adam/early-stopping regime). It should approach the random forest's R4 number (19.04 % frame-level; 28.06 % gated). If it does, capacity and optimiser are not the obstacle; the presence of the image is.
4. *Modality dropout.* Blank the image on a fraction (say 0.5) of training samples so the aux path must carry the loss on those batches. Lens 2 predicts a lift at R4 toward the metric number and possibly toward the oracle; this is the one fusion variant that changes the training signal rather than the input. If it fails too, the complementarity in §13.10 is not learnable from one session — also worth knowing.
5. *Range-shift check.* Restrict the R4 test set to frames inside the training p05–p95 distance band and re-score the `--aux metric` cells. If the delta stays at zero, point (ii) above is not the cause.

---

## 4. Mechanism D — the memorisation regime: near-duplicates, capacity and a spurious correlation with no counter-examples

**Claim.** ≈ 282 frames per class with lag-1 silhouette IoU 0.954 is tens of independent images per class against millions of parameters. In that regime a network fits by memorisation of recording-specific appearance, and — this is the part Sagawa et al. formalise — when a spurious attribute (here: everything that identifies the *recording*) is perfectly correlated with the label in training, over-parameterisation makes reliance on it worse, not better. In BIWI Training the correlation is not merely strong; there is no minority group at all. No re-weighting, loss or regulariser applied to that training set can prefer body geometry over room state, because the training set contains zero examples in which the two disagree. Only a second session per identity would create them.

**Our evidence.**
- Sample size (§13.17 item 3): ≈ 7,894 frames for 28 classes, consecutive frames near-duplicates, independent samples per class in the tens. Parameter count: AlexNet-gap 3,735,292 (§14.6 latency, 1 channel); §13.17 wrote "~23M parameters" — **the two statements disagree** (the stripe/flatten heads are larger than gap, and the 2023 Flatten→Dense head was 58.5 M, so the 23 M figure may refer to a different head; state one number in the paper and derive it from `model.count_params()` for the recipe quoted).
- "Session shift, not data quantity" (line 556): R3 with 4.3× fewer training frames beats R4 on byte-identical test frames for both the floor (8.55 vs 5.35) and the CNN (full frame 34.82 vs 6.70).
- Wave 18 (§13.18): retraining per cohort, K = 7 scores 36.71 % where a 28-class model restricted to the 7 columns scores 47.66 % — "the harder task regularises it". More classes means more counter-examples to *some* shortcuts (a room state shared by two people would no longer separate them), consistent with the memorisation reading.
- The permutation null: R4 label-permuted training scores 3.92 ± 0.16 % (line 549). Whether those permuted-label runs reached high *training* accuracy (Zhang et al.'s random-label fit) is not recorded in the sections read — see prediction 1.
- Pretraining does not rescue (ConvNeXt 8.45 %): "a data limit and not an AlexNet one" (§13.17).

**Literature.** Sagawa et al. 2020 **[V]** (ICML 2020, PMLR 119:8346–8356; arXiv 2005.04345): over-parameterisation exacerbates spurious correlations; the governing quantities are the majority-to-minority ratio and the spurious feature's signal-to-noise — here the ratio is infinite. Zhang et al. 2017 **[V]**; Arpit et al. 2017 **[V]** (memorisation follows simple-pattern learning and depends on the data, not only on capacity). Beery et al. 2018 **[V]** (location-specific memorisation in camera traps — the same "one place per class" structure).

**Prediction (checkable).**
1. The permuted-label R4 cells should show near-ceiling *training* accuracy if the memorisation reading is right (Zhang et al.). Retrieve from the stored histories or re-run one cell with logging.
2. The number of *sessions* per identity, not the number of frames, should govern R4. BIWI cannot test this (one Training recording per person), which is a limitation to state; MultiGait-style corpora with three sessions could, and the paper should say what experiment would settle it rather than imply BIWI can.
3. Wave 22 (below) adds 22 identities but no sessions; Lens 2 predicts it therefore adds counter-examples to *between-person* shortcuts only where two Training recordings happen to share room state, and predicts only a modest single-shot change on the 28 probe subjects.

---

## 5. Data volume versus the field's protocol, and what wave 22 will test

**The field's numbers on the same 28 probe people (§14.7):** Munaro 2014 21–32 % single / 42.9 % multi; Wu 2017 24.47 % Walking / 30.52 % Still; Karianakis 2018 25.4 % single / 50.0 % multi; Haque 2016 30.1 % single / 45.3 % multi with a 50-class softmax (chance 2 %). Haque et al. trained on "the full training set" and "from scratch" with dropout 0.5 (saved extract `haque2016.txt`, lines 317, 375–380 **[V]**); Karianakis et al. used split-rate RGB-to-depth transfer with fine-tuning (`karianakis.txt`, lines 22–27, 361–379 **[V]**). Neither extract, in the lines searched, states how a validation set or stopping epoch was chosen (**[UNVERIFIED]**; the full extracts should be read before the paper characterises their model selection).

**What differs in our R4 cell:** 28 classes instead of 50 (no distractor identities); 15 % of every training recording removed for validation plus a 50-frame guard; a 2.5-s decision window instead of whole sequences; the full-frame systems number (6.70 %) carries the room, which the field's cropped or point-cloud inputs did not. Our single-frame numbers (CNN 6.7 % full frame, ~19 % best recipe on the person; metric 19.04 %) sit below the field's single-shot band, and our gated 2.5-s point (CNN 28.16 %, metric 43.30 %, fusion 46.99 %) sits inside its multi-shot band (§14.6, §14.7).

**Wave 22 (commit 2ffce28, approved 2026-09-09):** policies `R4_standard_walking` / `R4_standard_still` with `train_all_subjects=True` — every `Training` recording trains (50 classes, chance 2.00 %), the 28 shared subjects' Walking or Still frames test, Walking test frames byte-identical to `R4_cross_session`; `tracklet_scores()` records one decision per test recording from the mean posterior (the field's multi-shot number) on every cell; 2 policies × {full, person, scale_removed gap, scale_removed stripe/aug8/tf10, rgb scale_removed stripe/aug8/tf10} × 5 seeds = 50 cells (§14.7 "Code").

**What Lens 2 predicts for wave 22, and how to read each outcome.** The training pool grows 1.8× and the class count to 50, but every identity still has exactly one session, so the shortcut structure of §§1–4 is unchanged.
- *Predicted:* single-shot CNN accuracy on the 28 probe subjects' Walking frames stays within seed variance of the matching R4 recipe (full 6.70; scale_removed gap 13.62; stripe/aug8 in the high teens frame-level), or falls slightly because 22 distractor classes enlarge the decision space. The tracklet-mean number should rise for the augmented recipe and stay flat for the gap head, as the integration curves already show (§13.12).
- *If instead wave 22 reproduces the field's 25–30 % single-shot on the same frames*, the data-pool and decision-space difference was binding and Lens 2's "not data volume" reading is refuted for the CNN — say so.
- *If it stays in the 7–18 % band*, the remaining gap to Haque/Karianakis is sequence modelling and input representation (recurrent attention over frames, 3-D/4-D inputs, RGB-to-depth transfer), not protocol — which the paper then states as the reason its CNN is not a re-implementation of theirs and does not claim to be.
Either way the table is commensurable for the first time on one pipeline, which is what §14.7 asked for.

---

## 6. Mechanism E — the random forest on 12 features is in a different regime altogether

**What it is (read from `hiride_metric_floor.py` and `hiride_metric.py`).** `RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)` on `BASE_METRIC` = `depth_extent_mm, height_p05, height_p50, stature_mm, surface_area_m2, volume_proxy_l, w_15, w_30, w_45, w_60, w_75, w_90` (`hiride_metric.py` lines 109–111), each a millimetre, square-metre or litre quantity computed per frame by unprojecting with the Kinect intrinsics (fx = fy = 575.816, cx = 320, cy = 240) through the sensor's user mask, heights measured against the shipped ground plane where present (§12.1; `frame_features`, lines 125–186). No validation split, no early stopping, no learning-rate schedule, no scaler needed (trees are invariant to monotone transforms), default fully-grown trees. Deterministic given the seed: §14.6 notes the metric number was the only one unchanged when GPU re-draws moved every CNN cell ("the RF is deterministic").

**Why it is a different regime, point by point.**
1. *No representation to learn.* The invariance that matters — cancellation of camera pose by measuring in a gravity-aligned metric frame — is done by arithmetic before the learner sees anything (§12.1: "Millimetres in a gravity-aligned frame cancel camera pose by construction"). The learner's job is a 12-dimensional partition, not feature discovery. Mechanisms A–D are about feature discovery under a misleading loss; they do not apply to a learner that is handed the invariant.
2. *The shortcut features are absent by construction.* The 12 columns contain no background, no position in the frame, no apparent size in pixels (those are in `NUISANCE`, lines 113–115, and are excluded). When a shortcut *is* added (`stand_dist_mm`), the forest takes it and cross-recording accuracy drops (R1 67.97 → 57.59, R3 15.56 → 10.76, §12.1). The forest is not immune to shortcuts; it was not offered any.
3. *Its errors are frame-level noise plus a known systematic bias, not memorisation.* Posterior integration lifts it 28.86 → 43.30 → 50.71 % (1 → 25 → whole tracklet, gated) while the gap-head CNN moves 18.40 → 21.36 → 20.71 (§13.12). The systematic part — stature drift 89.5 mm/m against a 105 mm SD, 74 % of walking frames clipping the body — is a measurement-validity defect that the full-body gate removes (19.04 → 28.06 % frame-level) and that training on gated frames does not (28.06 → 28.89, +0.83 pp, §13.12(b)). De-trending the features against distance *costs* within-session accuracy (R1 67.97 → 44.30) and is neutral at R4 (19.04 → 19.28): within one session the distance-correlated variance is itself a cue, exactly as it is for the CNN (§13.12 "What did NOT work").
4. *Twelve monotone, physically interpretable inputs.* Each column is a length, area or volume with an ordering that means the same thing for every subject; the forest's axis-aligned thresholds are statements like "stature above 1,720 mm", which transfer across sessions because the units do. The CNN's features have no such guaranteed meaning across a camera-pose change (§12.1: background +620 mm for all 28 shared subjects, people ~30 cm closer, 34 px lower).
5. *Model selection is trivial.* Nothing is tuned; three seeds are reported with subject-cluster CIs (R4 19.04 % [14.00, 24.40]; gated 28.06 [21.29, 34.66]; §14.6 Verdict 5). The threshold sweep in §13.13 is explicitly *not* quoted because it would be selection on the test set — the discipline is the same one Mechanism A says the CNN pipeline lacks at the stopping rule.

**Literature.** Kapoor & Narayanan 2023 **[V]** (their civil-war case: complex models appeared superior to logistic regression only under leakage — the shape of our RF-beats-CNN result is the same and should be presented with the same humility: it is a statement about the protocol, not about forests versus networks). Sagawa et al. 2020 **[V]** for the contrast: the forest here is not over-parameterised relative to 12 inputs in the sense their analysis needs, and it sees no spurious attribute.

**Prediction (checkable).**
1. A 12-input MLP trained with the CNN's own regime (Adam, tail validation, early stopping on within-session `val_accuracy`) on the same 12 features should land near the forest at R4. If it does, the regime difference is the *input*, not the learner or the stopping rule; if it falls to the CNN's level, the stopping rule itself (Mechanism A) is doing more damage than this report assumes. Cheap and decisive for how much weight to put on Mechanism A.
2. A forest on the 13 detector-metadata floor scalars plus the 12 metric columns should score *below* metric-only at R4 (the `metric+nuisance` result already shows the direction for one added column). This is the cleanest demonstration that the difference between the two learners is what they were given, and it costs seconds.
3. The forest's confusions at R4 should be *unstable* across adjacent frames of one test recording (noise), where the CNN's are stable (systematic) — directly testable from the stored per-frame posteriors in `cm_*.npz` and the forest's per-frame predictions, and the reason integration helps one and not the other.

---

## 7. Where the lens stops (what it does not claim)

- It does not claim that convolutional networks cannot learn metric body geometry from depth. LidarGait, TUM-GAID and Haque et al. report depth-shape recognisers at fixed encoder that plainly do (ESSENCE C2 caveat, `lidargait.txt`, `tumgaid_preprint.txt`, `haque2016.txt` in extracts **[V as saved]**). The claim is that *this* training signal — one session per identity, within-session validation, 28 classes, near-duplicate frames — gives a from-scratch classifier no reason to.
- It does not claim that a better stopping rule would fix R4. Mechanism A predicts (§1, prediction 3) that removing early stopping changes little; the rule is uninformative rather than destructive. If prediction 3 fails, the paper says so.
- It does not resolve whether the CNN's R4 number is limited by *data volume*. The R3-vs-R4 contrast argues against volume as the primary factor; wave 22 is the direct test and the paper should wait for it.
- It relies on two things not yet read from the record: the per-cell `best_epoch` distribution and the training-accuracy curves (§0 F3, §4 prediction 1). Both are one cluster session away. Until then, "the image path fits the training labels early" is an inference from the regime, not a measurement — mark it as such in any draft.
- Parameter count: §13.17's "~23M" and §14.6's 3,735,292 disagree; derive the figure for the quoted recipe before it appears in print.

---

## 8. Literature verification table

| item | status | where verified | what it supports here |
|---|---|---|---|
| Geirhos, Jacobsen, Michaelis, Zemel, Brendel, Bethge, Wichmann, "Shortcut Learning in Deep Neural Networks", Nature Machine Intelligence 2020, DOI 10.1038/s42256-020-00257-z, arXiv 2004.07780 | **[V]** | arxiv.org/abs/2004.07780 (journal ref + DOI shown) | Mechanism B framing |
| Shah, Tamuly, Raghunathan, Jain, Netrapalli, "The Pitfalls of Simplicity Bias in Neural Networks", NeurIPS 2020, arXiv 2006.07710 | **[V]** | arxiv.org/abs/2006.07710 | Mechanism B, C |
| Pezeshki, Kaba, Bengio, Courville, Precup, Lajoie, "Gradient Starvation: A Learning Proclivity in Neural Networks", NeurIPS 2021, arXiv 2011.09468 | **[V]** | arxiv.org/abs/2011.09468 | Mechanism C (aux branch receives no gradient) |
| Arpit et al., "A Closer Look at Memorization in Deep Networks", ICML 2017, arXiv 1706.05394 | **[V]** | arxiv.org/abs/1706.05394 | Mechanism B, D |
| Sagawa, Raghunathan, Koh, Liang, "An Investigation of Why Overparameterization Exacerbates Spurious Correlations", ICML 2020, PMLR 119:8346–8356, arXiv 2005.04345 | **[V]** | proceedings.mlr.press/v119/sagawa20a.html; arXiv | Mechanism D |
| Kapoor & Narayanan, "Leakage and the reproducibility crisis in machine-learning-based science", Patterns 4(9):100804, 2023, DOI 10.1016/j.patter.2023.100804, arXiv 2207.07048 | **[V]** (taxonomy labels L3.x **[UNVERIFIED]**) | api.crossref.org; arxiv.org/abs/2207.07048 | Mechanism A, E |
| Hammerla & Plötz, "Let's (not) stick together: pairwise similarity biases cross-validation in activity recognition", UbiComp 2015, pp. 1041–1051, DOI 10.1145/2750858.2807551 | **[V]** | api.crossref.org | Mechanism A (temporal adjacency in CV) |
| Beery, Van Horn, Perona, "Recognition in Terra Incognita", ECCV 2018, arXiv 1807.04975 | **[V]** | arxiv.org/abs/1807.04975 | Mechanism A, D (one location per class) |
| Zhang, Bengio, Hardt, Recht, Vinyals, "Understanding deep learning requires rethinking generalization", ICLR 2017, arXiv 1611.03530 | **[V]** | arxiv.org/abs/1611.03530 | Mechanism D (capacity to memorise) |
| Hermann & Lampinen, "What shapes feature representations? Exploring datasets, architectures, and training", arXiv 2006.12433 | **[V arXiv]**, venue **[UNVERIFIED]** (Crossref query returned only later papers by these authors; Semantic Scholar rate-limited) | arxiv.org/abs/2006.12433 | Mechanism B, C (easy features suppress hard ones; the decodability test) |
| Haque, Alahi, Fei-Fei 2016 — trained from scratch on the full BIWI training set, dropout 0.5; 30.1 % single / 45.3 % multi | **[V]** from saved extract `haque2016.txt` (lines 317, 375–380) and §14.7 | §5 (field's protocol) |
| Karianakis et al. 2018 — split-rate RGB-to-depth transfer; 25.4 % single / 50.0 % multi | **[V]** from saved extract `karianakis.txt` and §14.7 | §5 |

No DOI or number above was invented; where a venue could not be confirmed it is marked.

---

## 9. Sentences the paper can carry (ceiling wording)

- "The cross-session cells were trained on one recording per identity and validated on the tail of that same recording; early stopping therefore selected for within-session recognition, a criterion that cannot resolve the quantity of interest."
- "Handing the network the standing distance, or the twelve metric scalars, at the classifier head changed nothing within seed variance (−0.6 to +2.7 pp). Because the image path alone fits the training labels of a single session, the auxiliary branch receives little gradient — a case of gradient starvation (Pezeshki et al. 2021) rather than of missing information: a random forest on the same twelve numbers reaches 19–28 % on the same frames."
- "A training set in which every identity is one room, one trajectory and one outfit contains no example in which body geometry and recording appearance disagree; no loss defined on it can prefer the former (cf. Sagawa et al. 2020). This is a property of the corpus, not of the architecture, and pretrained features do not escape it (ConvNeXt-tiny/ImageNet: 8.45 % at R4)."
- "The metric features are not a better learner; they are the invariant computed before learning, on inputs from which the within-session shortcuts are absent. When a shortcut is added to them (`stand_dist_mm`) the forest takes it too."
- "Whether the field's higher single-shot numbers on these subjects reflect the larger training pool and decision space or their sequence models is tested directly in Table D (wave 22)."

---

## 10. Checks to run, ranked by cost

| # | check | cost | mechanism | decisive if |
|---|---|---|---|---|
| 1 | tabulate `best_epoch` / `hit_epoch_cap` over R4 cells from `results_*.json` | minutes, laptop or cluster | A | early cluster + no correlation with test acc |
| 2 | RF on 13 floor scalars + 12 metric columns at R4 | seconds, CPU | E | drops below metric-only |
| 3 | frame-to-frame stability of confusions, CNN vs RF, from stored posteriors | minutes | B, E | CNN stable, RF unstable |
| 4 | test-time aux ablation (aux = 0) on an `--aux metric` cell | minutes if weights kept, one GPU cell otherwise | C | accuracy unchanged within seed SD |
| 5 | 12-input MLP under the CNN's regime | one CPU/GPU cell | A vs E | lands near RF → input is the regime |
| 6 | linear decodability of `stature_mm`, `w_15`, `w_30` from penultimate features, untrained vs trained | one cell + probe | B | no increase with training |
| 7 | fixed-epoch (no early stopping) R4 cells | 5 GPU cells | A | inside seed variance |
| 8 | modality dropout (image blanked on half the batches) with `--aux metric` | 5–15 GPU cells | C | lifts R4 toward metric / oracle |
| 9 | wave 22 (approved) | ~4–8 GPU-hours | D, §5 | single-shot stays in 7–18 % band |
| 10 | permuted-label training accuracy | from histories or 1 cell | D | near ceiling |
