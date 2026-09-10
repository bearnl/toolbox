# 09 — Adversarial check of Lens 2 (LEARNING): what the measured record actually says about the five mechanisms

Written 2026-09-09 against `toolbox/HIRIDE_HANDOFF.md` (§5, §8.2–8.6, §12.1–12.3, §13.10, §13.12–13.18, §14.6, §14.8), the code in `toolbox/hiride_train.py`, `hiride_data.py`, `hiride_metric.py`, `hiride_metric_floor.py`, `hiride_sequence.py`, `hiride_fuse.py`, `hiride_collate.py`, `make_runs.py`, and the laptop artefacts in `hiride2-results/` (`stats_final.json`, `sequence_gated_best.json`, `sequence_ungated_best.json`, `fusion.json`, `metric_floor.json`, `metric_check.json`, `latency.json`, `collect_final_out.txt`). The report under check is `09_cnn_failure_L2_learning.md`. Every number below is copied from those sources with its location; nothing is derived except where marked "derived".

Method. For each mechanism: the claim as the analyst stated it; what in the record bears on it, for and against; a single verdict (SUPPORTED = a measurement is direct evidence for it; CONTRADICTED = a measurement is evidence against it as stated; UNTESTED = nothing measured bears on it); the cheapest analysis that would settle it, preferring what is already stored; and whether wave 22 covers it. Literature is tagged **[V]** (confirmed in a saved extract this pass) or **[UNVERIFIED]**.

---

## 0. Two findings about the premise that reframe the whole lens

Before the mechanisms, two things in the record change what there is to explain.

### 0.1 At single-frame level the best-recipe CNN is at parity with the metric random forest

The lens sets out to explain why "the CNN under-performed hand-computed geometry". The stored numbers say where that is true and where it is not.

| operating point | CNN, best recipe (`alexnet/stripe/aug8/tf10`, `scale_removed`) | metric RF (12 scalars) | source |
|---|---|---|---|
| single frame, ungated, shared frame set (5,642) | **18.36 %** | **18.83 %** | `sequence_ungated_best.json`, W = 1 |
| single frame, ungated, each on its own set | 18.36 % [12.15, 25.24] | 19.04 % [14.00, 24.40] | `stats_final.json`; §14.6 Verdict 5 |
| single frame, full-body gated (2,933) | 21.60 % | 28.86 % | `sequence_gated_best.json`, W = 1 |
| W = 25, ungated (211 decisions) | 24.08 % | 30.14 % | `sequence_ungated_best.json` |
| W = 25, gated (103 decisions) | 28.16 % [14.8, 43.0] | 43.30 % [30.3, 56.1] | §14.6 Verdict 2 |
| whole tracklet, ungated (28) | **25.00 %** | **26.43 %** | `sequence_ungated_best.json` |
| whole tracklet, gated (28) | 29.29 % | 50.71 % | `sequence_gated_best.json` |
| gap-head default recipe, single frame | 13.62 % | 19.04 % | §14.6 |
| full frame, single frame | 6.70 % | — | §8.2, §13.15 |

`frame_acc` is genuinely single-frame: `--test-fuse` is "Reported ALONGSIDE frame accuracy, never instead of it" (`hiride_train.py` 855–860) and lands in separate `fused_*` fields (1167–1191).

So the under-performance that is real is (i) the gap-head default recipe (−5.4 pp at frame level), (ii) the full frame, and (iii) the response to the full-body gate (+3.2 pp for the CNN, +10.0 pp for the RF) and to integration under the gate (+7.7 pp versus +21.9 pp at whole tracklet). With the best recipe and no gate the two learners are indistinguishable at every window, including whole tracklet (25.00 vs 26.43 on 28 decisions).

Consequence for the lens. Mechanisms A–D are stories about single-frame learning. For the recipe the paper quotes, the single-frame gap they would explain is ~0.5 pp. The gap-head deficit they might explain (5.4 pp) was closed by the stripe head and ±8 px augmentation — changes to the representation and its invariances, made with the same validation split, the same early stopping, the same data and the same loss. What still needs a mechanism is the gate × integration asymmetry, which is a statement about the STRUCTURE of the two learners' errors (Lens 3), not about how the CNN was trained.

### 0.2 The R3-versus-R4 contrast the lens leans on is a room phenomenon

Mechanisms A and D both cite "R3 trains on 1,821 frames yet beats R4's 7,894 on byte-identical test frames" (§5) as proof that session shift dominates data quantity. That holds for inputs that carry the room and for nothing else.

| input | R3 (same day, same clothes, Still → Walking, ~65 frames/subject) | R4 (other day, other clothes, Training → Walking, ~282 frames/subject) | source |
|---|---|---|---|
| full frame (depth CNN, gap) | 34.82 | 6.70 | §8.2 |
| `bg_plate` (person inpainted out) | 38.57 | 4.92 | §8.2 |
| 13-scalar pixel-cue floor | 8.55 | 5.35 | §5 |
| `person` (gap) | 8.65 | 7.87 (§8.2) → **9.22** (§14.6 regenerated) | §8.2, §14.6 |
| `person_centred` (gap) | 9.08 | 11.92 → **11.29** | §8.2, §14.6 |
| `scale_removed` (gap) | 13.76 | 12.45 → **13.62** | §8.2, §14.6 |
| `sil_scaled` (gap) | 13.71 | 14.59 → **12.97** | §8.2, §14.6 |
| metric RF, 12 scalars | 15.56 | **19.04** | §12.1 |

Once the room is cut out, R3 ≤ R4 for the CNN in every condition and for the RF. R3 also confounds two things the lens does not mention: its training recording is `Testing/Still` (standing) while the probe is `Walking`, so the pose regime changes, and it has ~65 training frames per subject (§5 item 4: "not viable for CNN training"). "Session shift, not data quantity, is what breaks it" (§5) is established for room-carrying inputs only. On person-borne inputs the record shows no measurable effect of session-match or of frame count at n = 28, in either direction.

---

## 1. Verdict summary

| mechanism | verdict | the deciding measurement | cheapest settling test | covered by wave 22? |
|---|---|---|---|---|
| A. Within-session early stopping | **UNTESTED** (premise SUPPORTED in code; effect unmeasured) | `hiride_data.py` 205–219, 341; `hiride_train.py` 1126–1147. Against "selected immediately": TVRID `best_epoch` 19–34 for from-scratch AlexNet (`collect_final_out.txt` 840–851) | tabulate `best_epoch`/`epochs_run`/`hit_epoch_cap` and `test_curve` from archived `results_*.json` (seconds) | No — both `R4_standard_*` policies call `_tail_val` (`hiride_data.py` 341) and the same `EarlyStopping` |
| B. Shortcut learning / gradient starvation on the image path | **CONTRADICTED** in the strong form ("never represents the invariant") | R1 `scale_removed` + stripe/flatten 66.90–69.67 vs RF 67.97 (§12.1 b); R4 frame-level parity 18.36 vs 18.83 (§0.1); the metric geometry itself falls 67.97 → 19.04 (§12.1) | `history` train-accuracy curves from archived `results_*.json` (seconds) | No |
| C. Aux branch starves at the head | **UNTESTED** (premise SUPPORTED in code; the null is consistent with three stories) | `hiride_train.py` 95–121, 154–156, 1068–1106; wave 17 §13.17, wave 19 §14.6; no weights saved | 12-input MLP under the CNN regime on `metric_features.npz` (CPU minutes) | No — no `--aux` cells in the wave-22 list (§14.8) |
| D. Memorisation regime; sessions not frames | **CONTRADICTED** as stated ("sessions, not frames"); memorisation premise SUPPORTED; sessions-binding clause UNTESTED | Wave 18: retrained K = 7 36.71 vs 47.66 for the 28-class model (§13.18); person-borne R3 ≤ R4 (§0.2); head change alone +4 pp at R4 (§12.1 b) | wave 22 itself; `history` of the `[perm]` cells (seconds) | Partly — tests the frames/identities clause, not the sessions clause |
| E. The RF is in a different regime | **SUPPORTED** as a description (config, shortcut exclusion, gate response); two sub-claims CONTRADICTED (ground-plane arithmetic; "not the CNN"); causal reading UNTESTED | `hiride_metric_floor.py` 46; §12.3 ground plane empty → camera-Y fallback (`hiride_metric.py` 150–154); best-recipe CNN gains +6 to +10 pp from integration (`sequence_*_best.json`) | same 12-input MLP as C; confusion run-lengths CNN vs RF from `cm_*.npz` + re-fit RF (minutes) | RF gets 50-class rungs (`LADDER`), regime question untouched |

---

## 2. Mechanism A — model selection on a within-session criterion

**Claim as stated.** Early stopping on `val_accuracy` over the tail 15 % of each training recording (guard 50) selects the epoch that best recognises within-session cues and has no resolution for cross-session accuracy; pretrained features fit the shortcut faster and are "selected immediately".

**Premise — SUPPORTED by code.** `_policy_cross` (`hiride_data.py` 313–342) hands its training pool to `_tail_val` (205–219) with `val_frac=0.15`, `guard=50`; the validation rows are the contiguous tail of each training recording. `EarlyStopping(monitor="val_accuracy", mode="max", patience=args.patience, restore_best_weights=True)` (`hiride_train.py` 1126–1131), restored unconditionally afterwards (1141–1147). Each run records `best_epoch`, `epochs_run`, `hit_epoch_cap` (1211–1213), the full Keras `history` (1226) and, when `--track-test` was passed, `test_curve` (1227); `cm_*.npz` carries `val_prob`/`val_truth` (1275–1279).

**Against the "selected immediately" version — measured, on TVRID.** The only artefact on the laptop that tabulates `best_epoch` is the TVRID collate (`collect_final_out.txt` 840–851, column `epochs` = "mean best_epoch / mean epochs_run", `hiride_collate.py` 67–68, 98). For AlexNet from scratch on depth: R0 22.0/35.0, R1 g0 33.8/44.8 (one seed hit the cap), R1 g5 18.6/31.6, R4 19.0/32.0; RGB R4 25.4/38.4. Within-session validation kept improving for 19–34 epochs; it did not saturate at epoch 0–1. The one-epoch saturation in the `TestCurve` docstring (764–773) is stated for "ImageNet ConvNeXt on RGB", not for the depth AlexNet cells the lens is about. BIWI `best_epoch` values exist only in the per-run `results_*.json` (archived in `runs_20260906.tar.gz`, §14.6); `stats_final.json`, `stats_wave2.json`, `stats_wave23.json` do not carry the field and `collect_final_out.txt` prints the column only for TVRID.

**Non-discriminating evidence.** `bg_plate` 38.57 > full 34.82 > person 8.65 at R3, and ConvNeXt 78.02 / 36.74 / 8.45 at R1/R3/R4, show WHAT was learned. They do not show that the stopping rule chose it: the loss is minimised on the same within-session frames whether or not one stops early, so a network run to 60 epochs would learn the same room. A is distinct from B and D only if some other epoch would have scored higher at R4. The R3-versus-R4 contrast is a room phenomenon (§0.2). The floor's 8.55 vs 5.35 is a random forest with no stopping rule at all and cannot speak to A.

**The proposer's own expectation.** The L2 report's §1 prediction 3 expects that removing early stopping changes R4 "inside seed variance" because "the criterion was never informative". If that is the expectation, A is a reproducibility note about the pipeline, not a mechanism of the CNN–RF gap.

**Verdict: UNTESTED** as a mechanism of the gap. The premise is exactly as described; its effect on R4 accuracy is not measured anywhere; the one measured `best_epoch` dataset argues against the "immediate selection" form for from-scratch AlexNet.

**Cheapest settling analyses, in order.**
1. From the archived `results_*.json`: tabulate `best_epoch`, `epochs_run`, `hit_epoch_cap` for every R3/R4 BIWI cell (seconds, CPU). A is alive only if `best_epoch` clusters at 0–2 for the depth AlexNet cells.
2. Wave 4 passed `--track-test` on every ConvNeXt cell (`make_runs.py` 507), so those files hold `test_curve`. Compute `test_curve[best_epoch]` against `max(test_curve)` per seed (seconds). A gap larger than the seed SD (1–4 pp) means the rule cost accuracy in those cells; report it as a diagnostic only — never use the maximum as the number.
3. From `history`, report within-session `val_accuracy` at the selected epoch beside the R4 test accuracy for the recipe quoted (seconds); this is the sentence the paper needs regardless of the verdict.
4. Only if 2 shows a real gap: one `--track-test` run per best-recipe condition (three GPU cells, minutes each).

**Wave 22 coverage: none.** `R4_standard_walking`/`R4_standard_still` pass through the same `_tail_val` call (`hiride_data.py` 341 applies for `train_all_subjects=True` as well) and the same `EarlyStopping`.

---

## 3. Mechanism B — shortcut learning, simplicity bias, gradient starvation on the image path

**Claim as stated.** Easy, perfectly predictive within-session features drive the cross-entropy to near zero; the hard invariant (metric geometry) "receives no gradient and is never represented".

**Code premise.** Augmentation is ±8 px shift plus horizontal flip on training batches only, no scaling (`ArrayBatches._aug`, `hiride_train.py` 714–728). The multiplicative-interaction argument is in the `attach_aux` docstring (96–104); it is an argument, not a measurement.

**What the cited evidence does and does not discriminate.**
- *Z-precision axis* (§8.4; §14.6: 16 bits 13.62, 8/4/3/2 bits 12.45/13.02/11.22/12.17, 1 bit 9.51; `sil_scaled` 12.97–16.67). Shows the transferable signal is in the outline. It is equally explained without any learning story: §13.17 item 1 gives 26 mm quantisation at 3 m against ~15 mm of relief, and §12.2 measured sensor sigma at 6–12 mm. Non-discriminating between "not learned" and "not resolvable".
- *Condition ordering* (gap head, §14.6: full 6.70 < person 9.22 < person_centred 11.29 < scale_removed 13.62). Under the best recipe the ordering flattens: `person_centred` 19.85 [13.29, 26.98] ≥ `scale_removed` 18.36 [12.15, 25.24] ≈ `sil_scaled` 19.45 [13.12, 26.62] (`stats_final.json`). Removing the size cue no longer helps once the network is made shift-invariant by augmentation. Consistent with "framing is the shortcut", but the remedy was representational (head, augmentation), not a change to the loss, the data or the stopping rule.
- *Augmentation turning flat integration into rising integration* (§13.12). Consistent with B and equally with §13.12's own reading (framing invariance converts systematic error to noise). Non-discriminating.
- *ConvNeXt 8.45 % at R4.* Every ConvNeXt cell in `stats_final.json` is `condition=full`. A full-frame number says nothing about shortcut learning on person-borne inputs.
- *The RF takes `stand_dist_mm`* (R1 67.97 → 57.59, R3 15.56 → 10.76, §12.1). This is the analyst's own concession that the shortcut is a property of the data, not of gradient descent. At R4 the same column is neutral: `metric+nuisance` 19.27 vs `metric` 19.04 (`metric_floor.json`).

**Against the strong form — measured.**
1. Within a session, once framing is normalised, the network's person-borne discrimination equals hand-computed millimetres: R1 g150 depth `scale_removed` with flatten/stripe head 66.90 / 69.67, `interior_only` 69.59, `sil_scaled` 69.67, against the metric RF's 67.97 (§12.1 b). Whatever the network represents after `scale_removed`, it is not "nothing but shortcuts". (Caveat, in fairness to B: the RF's R1 number itself contains distance-correlated variance — de-trending drops it to 44.30, §13.12 — so R1 parity is parity of two partly session-specific learners. It still refutes "never represents person-borne shape".)
2. At R4, frame-level parity: 18.36 vs 18.83 on the shared frames (§0.1). If the RF's 19 % is what metric geometry buys at frame level, the CNN transfers the same amount. "Never represents the invariant" cannot be squared with equal transfer.
3. The hand-computed metric geometry ITSELF collapses R1 67.97 → R4 19.04, a 49 pp fall (§12.1), because the "invariant" is not invariant on this corpus: stature drifts 89.5 mm/m against a 105 mm between-subject SD, 74 % of walking frames clip the body (§13.12). B attributes to gradient dynamics a loss that §13.12 attributes to measurement validity and that the RF suffers in full.

**Untested part.** The premise that the training loss saturates early has no measurement in any section read — but `history` (loss, accuracy, val_loss, val_accuracy per epoch) is stored in every archived `results_*.json` (`hiride_train.py` 1226). The decodability probe (Hermann–Lampinen) needs the trained weights, which are not saved (no `model.save`/`save_weights` anywhere in `hiride_train.py`; only the results JSON and `cm_*.npz` are written, 1262–1279).

**Verdict: CONTRADICTED** in the strong form the analyst states. A weak form — "the network learns pixel-normalised outline plus session-specific appearance rather than metric body geometry" — is consistent with the record, but it is Lens 1's representation claim, and it explains the gap-head deficit that the head and augmentation already closed, not the gate/integration asymmetry that remains.

**Cheapest settling analyses.**
1. `history` train-accuracy curves for the R4 best-recipe cells (seconds). If training accuracy reaches ≥ 0.99 within a handful of epochs while validation lags, the "loss saturates early" premise stands; if it climbs over ~20 epochs (as the TVRID `best_epoch` values suggest), gradient starvation is the wrong picture.
2. Only if 1 supports B: one retrain per condition with weights saved, then linear decodability of `stature_mm`, `w_15`, `w_30` from penultimate features at initialisation versus after training (one GPU cell plus a CPU probe).

**Wave 22 coverage: none.** It changes neither the loss, the inputs, nor the validation; it adds one-session identities.

---

## 4. Mechanism C — auxiliary inputs starve for gradient at the head

**Claim as stated.** `attach_aux` wires aux → Dense(32, relu) → Dropout(0.2) → Concatenate with the Dropout(0.5) pooled image features → one softmax, z-scored on train rows, NaN → 0, with no auxiliary loss, gating or modality dropout; once the image path fits one session's labels the residual vanishes and the aux weights stop moving; distribution shift at test is secondary because the RF generalises from the same numbers.

**Premise — SUPPORTED by code.** `attach_aux` (`hiride_train.py` 95–121) and the head (154–156) are exactly as described; `--aux dist` standardises `p_med` on train rows (1069–1077); `--aux metric` z-scores the 12 `BASE_METRIC` columns on train rows with `nan_to_num(…, 0.0)` (1078–1106); there is no auxiliary loss, gate or modality dropout. Aux rows are permuted with the image rows (696–697, 738–739).

**What the null result can and cannot tell.** Wave 17 (`--aux dist`, §13.17): mean delta −0.58 / −0.45 pp, every interval straddling zero, R3 power bound ±2.4 pp. Wave 19 (`--aux metric`, §14.6 Verdict 1): R4 +0.18 / +1.39 / +2.65, R3 +0.24 / −0.89 / +0.66, seed SD 1–4 pp; `stats_final.json`: `auxmetric` `sil_scaled` 22.11 [15.83, 28.96] vs 19.45 [13.12, 26.62], `scale_removed` 19.75 vs 18.36, `person_centred` 20.04 vs 19.85. These nulls are consistent with at least three stories: (a) C — the branch is starved; (b) the branch is used but its information is redundant with what the image path already extracts at frame level — and §0.1 shows the image path alone already matches the RF at frame level, so redundancy is a live alternative the lens does not consider; (c) the branch is used in training but mis-calibrated at test under the 36.3 % standing-distance shift and the 89.5 mm/m stature drift (§13.12 b). The oracle `either − metric` +7.7 / +9.0 / +8.3 pp (§14.6) shows complementary information exists per frame at TEST; nothing shows it exists on the TRAINING frames, where the image path is near ceiling (R0 98.36 %, §12.1; within-session validation near 1.0, `TestCurve` docstring), so a frame-dependent combination has no training signal to learn from. That logical point is the strongest part of C and it is not a measurement.

**The "distribution shift is secondary" dismissal.** The RF sees raw millimetres through axis-aligned thresholds; the aux head sees z-scored values with mean-imputation, a ReLU layer and a joint softmax. That the RF transfers does not establish that this function class transfers on the same numbers. Untested.

**No weights are stored**, so neither the weight-movement check nor a test-time aux ablation can be done from the archive; both need a retrain.

**Verdict: UNTESTED.** The wiring is as described; the mechanism (vanishing residual starving the aux branch) has no measurement; the null is consistent with three explanations, one of which — redundancy at frame level — is favoured by §0.1.

**Cheapest settling analyses, ranked.**
1. Aux-only control on the laptop: 12 z-scored inputs → Dense(32, relu) → Dropout → softmax, Adam, tail validation, early stopping on `val_accuracy`, on `metric_features.npz` with the R4 split from `hiride_data.make_split` (CPU minutes). If it lands near 19 % the numbers survive the head's regime and the image path's presence is the suspect (C alive); if it lands far below, the head, the z-scoring or the imputation is the suspect and the "secondary" dismissal fails. This one experiment also bears on A and E.
2. `history` of the wave-19 cells with and without `--aux`: does training accuracy saturate early in both (C's premise)? Seconds.
3. One GPU cell retrained with weights saved: test-time aux ablation (aux = 0) and ‖ΔW_aux‖ versus ‖ΔW_conv5‖.
4. Modality dropout — new training; only if 1 and 2 support C.

**Wave 22 coverage: none** — the wave-22 cell list in §14.8 carries no `--aux` variants.

---

## 5. Mechanism D — memorisation regime, capacity, a spurious correlate with no counter-examples

**Claim as stated.** ~282 frames per class at lag-1 IoU 0.954 is tens of independent images per class against millions of parameters; recording appearance is perfectly correlated with identity so no loss or regulariser defined on this corpus can prefer body geometry over room state; "the binding quantity is sessions per identity, not frames".

**Memorisation premise — SUPPORTED.** Lag-1 silhouette IoU 0.954, |Δdepth| 80.6 mm (§8.6); 7,894 training frames for 28 classes (§5 table; §13.17 item 3); one `Training` recording per identity (`hiride_data.py` policies; §12.3). "282 per class" is derived (7,894/28) and is before the 15 % validation carve and the guard.

**Parameter count — discrepancy CONFIRMED.** `latency.json`: `alexnet/gap@1ch` 3,735,292; `alexnet/stripe@1ch` 3,778,300; `alexnet/2023head@1ch` 72,008,444; `convnext_tiny@1ch` 27,838,588. §13.17's "~23M" matches none of them. The paper should quote 3.7 M for the gap/stripe AlexNet and 27.8 M for ConvNeXt-tiny. "Millions against tens of independent samples" survives at 3.7 M.

**The "not frames" clause — CONTRADICTED by measurement.**
- Wave 18 (§13.18): retrained on K classes versus the 28-class model restricted to the same K columns, same draws, same gated frames — K = 7: 36.71 vs 47.66; K = 14: 31.94 vs 33.15; K = 21: 24.86 vs 28.09. More identities and ~4× the frames, at FIXED one session per identity, improved cross-session accuracy on the same enrolled people by ~11 pp at K = 7. Every recording still has its own plate, so no counter-example to the room–identity correlation was added; yet transfer improved. That is a measured data-quantity effect which "no loss … defined on this corpus can prefer body geometry over room state" says should not exist.
- The R3-versus-R4 contrast runs through the room (§0.2). On person-borne inputs, same-session training with 4.3× fewer frames does not beat cross-session training with more.
- Architecture moved the number with identical data: the head change alone took R4 depth "from 12.5–14.6 to 16.5–18.0" (§12.1 b) and the best recipe to 18–20 (`stats_final.json`). §13.17 item 3's "a data limit and not an AlexNet one" rests on ConvNeXt full-frame 8.45 versus AlexNet full-frame 6.70, both room-bound; there is no ConvNeXt cell on a person-borne condition at R4 (`stats_final.json`: every ConvNeXt cell is `full`). The claim that architecture is not the limit is unsupported for the inputs that matter and contradicted by a ~5 pp head effect.

**The "sessions per identity is binding" clause — UNTESTED and untestable on BIWI.** Each identity has one `Training` session; `Testing/Still` shares day and clothes with the probe, so it cannot serve as a second session. The permutation-null cells' training accuracy (Zhang-style random-label fit) is checkable from the `[perm]` cells' `history` in the archive.

**Verdict: CONTRADICTED** as stated. The record shows a frames/identities effect (wave 18) and no measurable session-match effect on person-borne inputs (R3 vs R4); the sessions-binding half is untested; the memorisation premise stands.

**Cheapest settling analyses.**
1. Wave 22 (scheduled): 50 classes, training pool 1.8× R4's, the same 28-probe Walking frames. A single-shot rise beyond seed SD on the matched recipe refutes "not frames" a second time; a flat result says wave 18's effect has saturated and D regains ground. Either way, report it.
2. `history` of the `[perm]` cells for the training-accuracy ceiling (seconds).
3. A learning-curve sub-sampling of R4 training frames (25 / 50 / 100 % of each recording, fixed recipe) is the direct frame-count test — 3 × 5 GPU cells, not stored.

**Wave 22 coverage: partial** — the frames/identities clause yes; the sessions clause no.

---

## 6. Mechanism E — the random forest on 12 metric features is in a different regime

**Claim as stated.** `RandomForestClassifier(n_estimators=300, random_state=seed)`, fully grown, no validation, no early stopping, deterministic given seed, on 12 millimetre/area/volume quantities computed by unprojection through the sensor mask against the ground plane; camera-pose invariance is arithmetic done before learning; shortcut columns excluded by construction; errors are frame-level noise plus a known systematic bias removed by the gate, so integration helps it and not the CNN; it is not immune to shortcuts (takes `stand_dist_mm`).

**Config — SUPPORTED by code.** `fit_eval` (`hiride_metric_floor.py` 41–59): `RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)`, `fit(Xtr, ytr)`, `predict(Xte)`; no validation split, sklearn defaults (fully grown). Determinism: §14.6 records the metric cells unchanged when every CNN partner was re-drawn — "the RF is deterministic".

**Shortcut exclusion — SUPPORTED.** `BASE_METRIC` (`hiride_metric.py` 109–111) excludes `NUISANCE` (113–115); `SETS["metric+nuisance"]` adds `stand_dist_mm` explicitly (`hiride_metric_floor.py` 172–173). "Takes it when offered": R1 67.97 → 57.59, R3 15.56 → 10.76 (§12.1). At R4 the column is neutral — `metric+nuisance` 19.27 vs `metric` 19.04 (`metric_floor.json`) — uninformative rather than harmful once train and test distance ranges barely overlap (36.3 % of test frames outside the train p05–p95 band, §13.12 b). State the R4 row too.

**"Camera-pose invariance is arithmetic done before learning" — PARTLY CONTRADICTED.** §12.3: `_groundCoeff.txt` is "effectively EMPTY", parsed on 130/28,037 frames, so "the metric result above therefore used the CAMERA-Y FALLBACK, not the shipped ground plane". The code path is `hiride_metric.py` 150–154: `nvec = (0, −1, 0)`, `h` re-based at its 0.5th percentile, `ground = 0`. Unprojection cancels the pixel-scale dependence on distance; it does not cancel camera pitch or roll, and the census records a moved camera/room between sessions (background +620 mm for all 28 shared subjects, people 34 px lower, §8.6). §13.13 itself names the residual height-axis defect as "probably the ground plane". §13.14's sentence "into a GRAVITY-ALIGNED frame using the per-frame ground plane" is therefore inconsistent with §12.3 for the published numbers and must not reach the paper as written. The invariance is partial and the analyst's point 1 overstates it.

**"Integration helps it and not the CNN" — SUPPORTED for the gap head, CONTRADICTED for the best recipe.** Gap head: 18.40 → 21.36 → 20.71 (§13.12). Best recipe, gated: CNN 21.60 → 28.16 (W = 25) → 31.43 (W = 100) → 29.29 whole; ungated 18.36 → 24.08 → 25.00 (`sequence_gated_best.json`, `sequence_ungated_best.json`). The augmented CNN gains +6 to +10 pp from integration. The RF gains +11 pp ungated (18.83 → 30.14 at W = 25; whole 26.43) and +22 pp gated (28.86 → 50.71). Ungated whole tracklet is at parity: RF 26.43 vs CNN 25.00. The RF's integration advantage exists only WITH the gate.

**"Errors are frame-level noise" — needs qualification from the record.** Without the gate the RF's errors are as systematic as the CNN's (parity at whole tracklet). With the gate they are not pure noise either: §13.13 — the `metric+shape` frame-level gain INVERTS under aggregation (43.30 → 42.33 at W = 25; 50.71 → 48.57 whole) because distance-driven error is "correlated across the whole window", and drift/sig > 1 for ten of the twelve `BASE_METRIC` columns (§13.12) is measured over the walk, not only over clipped frames. The analyst writes "known systematic bias" and then defaults to "noise"; the record supports "systematic, and the gate converts enough of it to noise for integration to work".

**The causal reading ("different regime" as the reason it wins) — UNTESTED.** Nothing measured separates "wins because its inputs are millimetres" from "wins because it has no early stopping / no validation split / 12 inputs". The analyst's own E-prediction 1 is the cheap test.

**Verdict: SUPPORTED** as a description of the RF (configuration, shortcut exclusion, gate response), with the ground-plane arithmetic and "not the CNN" sub-claims contradicted and the causal reading untested.

**Cheapest settling analyses.**
1. The same 12-input MLP under the CNN's regime as C-1 (CPU minutes). Near 19 % → the regime difference is the input, not the learner or the stopping rule; far below → the CNN's regime is damaging on its own and A/C gain weight.
2. Confusion run-lengths within each test recording, CNN (`cm_*.npz`: `pred`, `truth`, `test_rows`) versus RF (re-fit in seconds from `metric_features.npz` with `hiride_data.make_split`), gated and ungated. Long runs of the same wrong label = systematic; short = noise. Directly tests the characterisation and explains the gate × integration asymmetry in §0.1.
3. RF on the 13 floor scalars plus the 12 metric columns at R4 (analyst's E-2), seconds; confirms direction only.

**Wave 22 coverage.** `hiride_metric_floor.py` `LADDER` (36–38) includes both `R4_standard_*` rungs, so the RF gets 50-class numbers beside the CNN's; the regime question is untouched.

---

## 7. Inconsistencies in the record found on the way

1. **Ground plane.** §12.3 (camera-Y fallback used; `groundCoeff` parsed on 130/28,037 frames) versus §13.14 ("using the per-frame ground plane") and the analyst's E point 1. The published 12 features are in a camera-Y-aligned metric frame, not a gravity-aligned one. `metric_check.json` lists `ground` among the features of an earlier 15-column variant (17.98 %); if `ground` is constant 0 it is a dead column there.
2. **Parameter count.** §13.17 "~23M" versus `latency.json` 3,735,292 (gap) / 3,778,300 (stripe) / 27,838,588 (ConvNeXt-tiny) / 72,008,444 (2023 head).
3. **"A data limit and not an AlexNet one"** (§13.17 item 3) rests on a full-frame-only ConvNeXt comparison; the head change alone moved R4 by ~4 pp on person-borne inputs (§12.1 b).
4. **"R3 beats R4 … session shift dominates"** (§5) generalised beyond room-carrying inputs; on every person-borne condition R3 ≤ R4 (§0.2).
5. **The single-frame gap for the quoted recipe is ~0.5 pp** (§0.1); the lens's framing "why the CNN under-performed hand-computed geometry" should be scoped to the gap head, the full frame, and the gated/integrated operating point.
6. `tf10` in the recipe tag is tracklet evaluation reported alongside `frame_acc`, never in its place (`hiride_train.py` 855–860); the parity in §0.1 is genuinely single-frame.

---

## 8. What is already stored, and which cheap analysis settles what

| stored artefact | location | fields | settles |
|---|---|---|---|
| `results_*.json` (744 runs) | cluster `$SCRATCH/hiride2/runs`; archive `runs_20260906.tar.gz` (§14.6) | `best_epoch`, `epochs_run`, `hit_epoch_cap`, `history` (loss/acc/val_loss/val_acc per epoch), `test_curve` (wave-4 ConvNeXt cells), `tracklet_*` | A (1, 3), B (1), C (2), D (2) — seconds each once the tarball is open |
| `cm_*.npz` | same | `prob`, `pred`, `truth`, `test_rows`, `test_subject`, `val_prob`, `val_truth` (wave 19+) | E (2) confusion run-lengths; A (3) within-session val accuracy at the selected epoch |
| `metric_features.npz` | prep dir | 53 columns incl. `BASE_METRIC`, `NUISANCE`, `manifest_row` | C (1) / E (1) aux-only MLP; E (2) RF re-fit; E (3) |
| `metric_floor.json` | laptop | R4 `metric` 19.04, `metric+nuisance` 19.27, `metric+shape` 24.34, `shape` 20.79 | E: the R4 shortcut row is already there |
| `sequence_gated_best.json`, `sequence_ungated_best.json` | laptop | CNN/metric/geo by window, `n_decisions`, CIs, contrasts, top-k, per-subject | §0.1; E integration claims |
| `stats_final.json` | laptop | per-cell frame acc + subject CIs, incl. `auxmetric` and `full_body` cells; no epoch fields | §0.1, B ordering, C deltas, ConvNeXt scoping |
| `latency.json` | laptop | `params` per architecture/head | D parameter count |
| `collect_final_out.txt` 840–851 | laptop | TVRID `epochs` column | A ("selected immediately" is false for from-scratch AlexNet on TVRID) |
| model weights | **not saved** (`hiride_train.py` writes only JSON + `cm_*.npz`) | — | B decodability, C ablation and weight movement need a retrain |

The single most informative cheap experiment across the lens is the **12-input MLP under the CNN's own regime** (C-1 / E-1): CPU minutes, no cluster, and it separates "inputs" from "regime" for A, C and E at once. The single most informative stored-data read is the **`history`/`best_epoch` tabulation** (A-1, B-1, C-2, D-2), which tests the premise every mechanism shares — that the image path fits the training labels early.

---

## 9. Wave 22 coverage matrix

Wave 22 (§14.8, commit 2ffce28): `R4_standard_walking` / `R4_standard_still`, all 50 `Training` recordings train, the 28 shared subjects' Walking or Still frames test (Walking frames byte-identical to `R4_cross_session`), `tracklet_scores()` on every cell, 2 policies × {full, person, scale_removed gap, scale_removed stripe/aug8/tf10, rgb scale_removed stripe/aug8/tf10} × 5 seeds.

| varies | held fixed | therefore tests |
|---|---|---|
| training pool (1.8×), class count (50), decision space (chance 2 %), whole-sequence decisions | validation policy (`_tail_val`, within-session), stopping rule, aux (none), sessions per identity (1), RF regime, loss | D's "not frames/identities" clause (yes); commensurability with Haque/Karianakis/Wu/Munaro (yes); A, B, C (no); D's sessions clause (no); E's regime question (no, though the RF gets 50-class rungs) |

---

## 10. Literature status

Verified this pass in saved extracts (`lit-audit/extracts/`):
- Haque, Alahi, Fei-Fei 2016 — "we use the full training set" (`haque2016.txt` 317); "We train our model from scratch" with SGD, batch 20, lr 1e-4, momentum 0.9, weight decay 5e-4, dropout 0.5 (375–380); "trained over many epochs" (496). **[V]**. No validation split or stopping rule is stated in the extract → their model selection **[UNVERIFIED]**.
- Karianakis et al. 2018 — end-to-end training with lr "linearly decreases from 0.01 to 0.0001 in 200 epochs" up to 250 (`karianakis.txt` 522–523); lr reduced at a loss "plateau" for the embedding (517–518); named validation sequences (n05–n06, n11–n12) are stated for the 305-person corpus with 32 re-recorded subjects (618–626), and for BIWI the "Training IDs" are used for RGB-to-depth transfer (620–621). **[V]**. BIWI stopping rule **[UNVERIFIED]**.
- Wu et al. 2017 — `Training` used as gallery, Walking/Still as probes (`wu2017.txt` 759–776). **[V]**.

Not re-fetched this pass (all are ML-theory framing and no verdict above depends on them): Geirhos et al. 2020, Shah et al. 2020, Pezeshki et al. 2021, Arpit et al. 2017, Zhang et al. 2017, Sagawa et al. 2020, Hermann & Lampinen 2020, Kapoor & Narayanan 2023, Hammerla & Plötz 2015, Beery et al. 2018 — **[UNVERIFIED here]**; the L2 report's §8 table reports **[V]** on arXiv/Crossref for each, which this pass neither confirms nor disputes. No DOI or number was invented; none is quoted for these.

---

## 11. Sentences the paper can carry after this check (ceiling wording)

Replacing the L2 report's §9 sentences where the record does not support them.

- "At single-frame level and without frame selection, the augmented stripe-head CNN and the twelve metric features are indistinguishable (18.4 % vs 18.8 % on the same 5,642 frames). The metric features pull ahead only when frames are gated for a complete body and integrated over a window (43.3 % vs 28.2 % at 25 frames), so the difference between the two lies in how their errors respond to selection and integration, not in single-frame accuracy."
- "The cross-session cells were validated on the tail of each training recording and stopped on within-session accuracy; this rule cannot see cross-session accuracy. Whether it cost any is not measured; the stored per-epoch histories permit the check." (Not: "selected for within-session recognition" as an explanation of the gap.)
- "Handing the network the standing distance or the twelve metric scalars at the head changed nothing within seed variance. The result is consistent with the auxiliary branch receiving little gradient, with the information being redundant with what the image path already extracts at frame level, and with a distribution shift at test; the experiments do not separate these." (Not: "a case of gradient starvation".)
- "Training on more one-session identities improved cross-session accuracy on the same enrolled people (28-class vs 7-class, +11 pp at K = 7), so data volume is not irrelevant; whether a second session per identity would matter more cannot be tested on this corpus." (Not: "the binding quantity is sessions per identity, not frames".)
- "The metric features are computed in a camera-axis-aligned metric frame — the shipped ground plane was unavailable — so distance-dependent pixel scale is removed by arithmetic before learning while camera pitch is not; the residual stature drift of 89.5 mm/m is consistent with that." (Not: "gravity-aligned … cancel camera pose by construction".)
- "The metric features' errors are systematic across a walk (ten of twelve drift more than people differ); the full-body gate removes enough of that systematic component for posterior integration to nearly double their accuracy, whereas the augmented CNN gains 6–10 pp from the same integration." (Not: "integration helps it and not the CNN".)
