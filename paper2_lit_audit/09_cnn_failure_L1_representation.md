# Lens 1 — Representation and inductive bias: why the CNN recovered the outline and not the millimetres

Report for the BIWI RGBD-ID manuscript (28 people re-recorded on another day in other clothes, Kinect v1). Written 2026-09-09 from the measured record in `/Volumes/Workspace/study/toolbox/HIRIDE_HANDOFF.md` (§8.2–8.4, §12.1–12.2, §13.10, §13.12–13.18, §14.6) and the code in `/Volumes/Workspace/study/toolbox/` (`hiride_train.py`, `hiride_metric.py`, `hiride_metric_floor.py`, `hiride_data.py`, `hiride_sequence.py`, `hiride_fuse.py`). Every literature item is tagged [V] (confirmed on arXiv, Crossref or a saved extract under `/Volumes/Workspace/study/hiride2-results/lit-audit/extracts/`) or [UNVERIFIED]. No DOI or number below is invented; where a venue could not be reached it is said so.

---

## 0. The question and the record it must fit

A from-scratch AlexNet-style CNN and an ImageNet-pretrained ConvNeXt-Tiny were trained on Kinect v1 depth crops of 28 people and tested on a second session (R4). Twelve hand-computed lengths in millimetres, fed to a random forest, beat both. The numbers this lens must explain, all from the regenerated 2026-08-31 artefacts unless stated:

| fact | value | source |
|---|---|---|
| Gated 2.5-s operating point (W = 25, 103 decisions, 28 people, chance 3.57 %) | CNN 28.16 % [14.8, 43.0]; metric 43.30 % [30.3, 56.1]; fusion 46.99 % [32.9, 61.4] | §14.6; `sequence_gated_best.json` |
| Single-frame R4 | CNN 6.70 % (full frame), ~19–21 % best recipe on gated frames; metric 19.04 % (all frames), 28.06 % on full-body frames | §8.2, §13.12, §14.6 |
| Fig. 7 curves (gated) | CNN 21.6 → 28.2 → 31.4 % at 1 / 25 / 100 frames, flattening; metric 28.9 → 43.3 → 54.3 % | `sequence_gated_best.json` |
| Z-precision axis at R4 (`scale_removed`, fixed 0–6000 mm range) | 16 bits 13.62; 8/4/3/2 bits 12.45 / 13.02 / 11.22 / 12.17; 1 bit 9.51; normalised binary silhouette 12.97–16.67 | §8.4, §14.6, Fig. 4 |
| Background with the person inpainted out beats the full frame within session | R3 `bg_plate` 38.57 % vs `full` 34.82 %; person alone 8.65 % (floor 8.45 %) | §8.2 |
| Feeding the CNN the standing distance (`--aux dist`) | mean delta −0.58 pp (size kept) / −0.45 pp (size removed); R3 upper bound +2.4 pp | §13.17 |
| Feeding the CNN the 12 metric scalars at the head (`--aux metric`, wave 19) | +0.18 / +1.39 / +2.65 pp at R4, inside seed variance | §14.6 |
| ImageNet ConvNeXt-Tiny on the full frame | R4 8.45 %, R3 36.74 %, R1 78.02 % | §8.1 table, §12.2 |
| Metric features beat the image floor on the same frames | R4 19.04 % vs 5.35 %; R1 67.97 vs 17.73 | §12.1 |
| Adding `stand_dist_mm` as a feature | R1 67.97 → 57.59, R3 15.56 → 10.76 | §12.1 |
| Clipping | 74 % of walking test frames clip the body; gate retains 52.0 % (2,933 / 5,642); train frames clip 9.2 % vs R4 test 47.8 % | §13.12, §14.8, `hiride_data.py` |
| Stature drift | 89.5 mm/m against a 105 mm between-subject SD | §13.12 |
| Head asymmetry (R1 depth, gap → flatten/stripe) | interior_only +28.7 pp, scale_removed +19.3, sil_scaled +16.9; rgb −0.2 | §12.1(b) |
| Training pool at R4 | ≈ 7,894 frames, one recording per person; R3 trains on 1,821 frames and its image floor is 8.55 % vs R4's 5.35 % on the same test set | §5, §13.17 |
| Validation | carved from the tail of each TRAINING recording; early stopping on `val_accuracy` | `hiride_data.py::_tail_val`, `hiride_train.py` |

The regenerated record also withdrew every R4 condition-vs-full contrast whose interval cleared zero (§14.6), so everything below rests on the ORDERING across many cells and rungs, exactly as §13.11 item 2 said it must.

---

## 1. The two representations, stated exactly

**What the CNN sees.** `hiride_train.py` gives the network an S×S single-channel float image: depth clipped at 6,000 mm and divided by 6,000 (`DEPTH_CLIP_MM`), 0 as the invalid sentinel. `apply_mask_condition` then makes one controlled edit:

- `person` — pixels outside the shipped userMap set to `fill`; apparent size, image position and absolute depth are all kept.
- `person_centred` — pure integer translation so the mask's bounding-box centre sits at the frame centre; size and depth kept.
- `scale_removed` (`scale_remove`) — crop the mask's bounding box, set non-mask pixels to `fill`, shift the valid depth values so the person's MEDIAN is 3,000 mm (`SCALE_TARGET_DEPTH`), resize the crop so its height is 0.70 of the frame (`SCALE_TARGET_H`, aspect kept, width capped), paste centred. Apparent size, position and standing distance are all gone; relative interior depth is preserved as an offset, not rescaled.
- `sil_scaled` — the same geometry on the binary mask.
- `--bits b` (`quantise_depth`) — uniform quantisation to 2^b bin centres over the FIXED 0–6,000 mm range, applied after the condition.

The network is `build_alexnet`: five Conv–BN–ReLU blocks (11×11 stride 4, then 5×5, 3×3, 3×3, 3×3), max-pooling after blocks 1, 2 and 5, then a head — `gap` (GlobalAveragePooling2D), `stripe` (mean over width, keep rows) or `flatten` — dropout and a softmax Dense. The total downsampling is 4 × 2 × 2 × 2 = 32, so each final-map cell covers 32 × 32 input pixels. Nowhere does the network receive fx, fy, cx, cy, the ground plane, or the pre-normalisation median depth. `build_convnext` uses `ConvNeXtTiny(include_top=False, weights='imagenet')` with ImageNet channel normalisation on a 3-channel copy of the same image (§9 notes).

**What the metric pipeline computes.** `hiride_metric.py::frame_features` unprojects every person pixel with the Kinect intrinsics (fx = fy = 575.816, cx = 320, cy = 240):

    X = (u − cx) · z / fx,   Y = (v − cy) · z / fy,   Z = z

projects the cloud onto a gravity-aligned basis (h up, lat across, dep front-to-back — camera −Y fallback because BIWI's `_groundCoeff.txt` is effectively empty, §12.3), and measures: `stature_mm` = p99.5 − p0.5 of h; `height_p50`, `height_p05`; `depth_extent_mm` = p99 − p1 of dep; six lateral widths `w_15 … w_90` in bands at fixed FRACTIONS of stature; `surface_area_m2` = Σ (z/fx)(z/fy); `volume_proxy_l` = area × depth extent. `hiride_metric_floor.py` pins these twelve as `BASE_METRIC` and fits a 300-tree RandomForest. `stand_dist_mm` is written but excluded as NUISANCE.

Two things follow immediately and everything else in this lens is a consequence of them. (i) Every metric feature is a product of a PIXEL EXTENT and a DEPTH VALUE divided by a constant the network never sees. (ii) `scale_removed` sets the pixel extent to a constant (0.70 S) and the depth value to a constant (3,000 mm) — it deletes BOTH factors of that product before the network sees the image.

---

## 2. Mechanism A — depth treated as an intensity image, not as geometry

**Claim.** A length in millimetres is a NON-LOCAL, MULTIPLICATIVE function of pixel coordinate and pixel value: two pixels' positions (Δu), their depth values (z), and a constant (f) the network is never given. A convolutional layer is a fixed linear functional of a LOCAL intensity neighbourhood followed by a pointwise nonlinearity; the product "extent × intensity ÷ f" is not something a conv-pool stack computes by construction, it is something it would have to LEARN from data — and with tens of independent samples per class (§13.17 item 3) there is no gradient pressure to learn a hard non-local product when an easy local cue predicts the training labels.

**Our evidence.**
- On identical frames the twelve unprojected scalars reach 19.04 % at R4 against an image floor of 5.35 % — a floor built from pixel counts and pixel bounding boxes (§12.1). Pixels are not camera-invariant; the census measured the reason: background +620 mm for all 28 shared subjects, people ~30 cm closer and 34 px lower in the second session.
- The mirror-image test: adding `stand_dist_mm` back to the metric set DROPS R1 67.97 → 57.59 and R3 15.56 → 10.76 (§12.1). Distance is the term that turns a camera-invariant measurement back into a recording signature.
- The pre-registered test (§13.17): hand the CNN the one factor it cannot compute (`--aux dist`, z-scored standing distance concatenated to the pooled features). Prediction fixed in advance: it should help the size-preserving conditions (`person`, `person_centred`) and do nothing for `sil_scaled`. Result: mean delta −0.58 pp (size kept) and −0.45 pp (size removed); every interval straddles zero; at R3, where the intervals are ±2.4 pp, distance buys at most +2.4 pp. So the network was NOT sitting one multiplication short of the metric features: it was not attempting the product at all.
- Wave 19 (§14.6): concatenate the twelve metric scalars themselves at the head. +0.18 / +1.39 / +2.65 pp at R4, inside seed variance — even the finished answer, handed to a 28-class softmax trained on ~7,894 near-duplicate frames, does not get used. The oracle bound (`either − metric` +7.7 / +9.0 / +8.3 pp, intervals clear of zero, §13.10, §14.6) says the information is complementary; the network cannot learn a frame-dependent combination at this n.

**Why the twelve lengths DID work.** Unprojection makes each feature a quantity in millimetres in a gravity-aligned frame; camera translation, pitch and standing distance cancel "by construction" (`hiride_metric.py` docstring). What is left is true body size — the single most stable biometric a body has — plus the one defect the pipeline still carries (band anchoring at the visible base, mechanism B).

**Checkable prediction.** Under `scale_removed` the CNN's confusions should be statistically INDEPENDENT of stature difference between the confused pair, because the input carries no stature; the metric RF's confusions should concentrate among pairs within ~1 SD of stature (105 mm). Compute the mean |Δstature| over confused pairs for both models from the stored `cm_*.npz` posteriors and `metric_features.npz`: prediction is metric ≪ CNN ≈ the population mean |Δstature|.

---

## 3. Mechanism B — what `scale_removed` does to the metric information, and what clipping does to a row

**Claim.** Rescaling the crop to a fixed bounding-box height deletes ABSOLUTE size; shifting the median depth deletes ABSOLUTE range. What survives is scale-free shape (outline proportions, width-to-height ratios, relative interior relief). And because the bounding box is the VISIBLE body, a clipped body is rescaled as if it were a whole one: row r of the normalised crop is the chest in one frame and the waist in another, depending only on how much of the legs left the frame.

**Our evidence.**
- Depth `scale_removed` at R4 (13.62 regenerated) and `sil_scaled` (12.97) are indistinguishable (§14.6): after size and distance are normalised the interior adds nothing, i.e. the network is left with the outline. RGB under the same normalisation goes to 17.72 %, and with a self-derived mask 20.60 % (§14.6 verdict 3) — for RGB the normalisation removes nuisance and keeps its cue (clothing texture is already scale-free); for depth it removes nuisance AND the one cue that survives a session change.
- Keeping size does not rescue depth: `person_centred` (translation only) scores 11.29–11.92 at R4 and `person` 7.87–10.66 (§8.2, §14.6). Pixel size WITHOUT depth is confounded by distance — "person 30 cm closer" and "person is larger" are indistinguishable in an image (`hiride_metric.py` docstring), and in session B everyone stood ~30 cm closer. The CNN was forced to choose between a size cue poisoned by a camera move and no size cue at all; either way it cannot recover TRUE size.
- Clipping as correspondence shift: `hiride_data.py::eligible_mask` records that the body touches a frame edge in 47.8 % of R4 TEST frames but 9.2 % of TRAIN frames; the docstring's own wording is that "the same image row means a different body part in train and test", so an alignment-preserving head reads a corrupted correspondence. The gate that removes the shift lifts the CNN's single-frame accuracy from 18.36 % (ungated, 5,642 frames) to 21.60 % (gated, 2,933 frames) and the metric features from 18.83 % to 28.86 % (`sequence_ungated_best.json`, `sequence_gated_best.json`).
- The metric pipeline has the SAME defect in a different guise (§13.12a, §13.14B): the `w_XX` bands are anchored at the base of the VISIBLE body and scaled by the visible stature, so `stature_mm` drifts 89.5 mm/m — ~197 mm over the 2.2 m walk, 1.9× the between-subject SD — and `w_75` (shoulders, steepest width gradient) drifts 2.34 SD while `w_15`/`w_30` beside the anchor drift ~0. Crown-anchored bands in absolute millimetres (`hw_`, `ht_`) fix the anchor and the thickness family `ht_*` is the most range-stable in the set (0.10–0.91). Training on unclipped frames adds only +0.83 pp (§13.12b): a clipped body yields a WRONG measurement and no training regime repairs it.

**Checkable prediction.** Add a `metric_scaled` condition: rescale the crop by z_median / 3,000 instead of to a fixed 0.70 S, so pixel height at the canvas is proportional to TRUE height at a 3 m reference. Prediction: on full-body frames at R4 it beats `scale_removed` (upper bound: the accuracy of `stature_mm` alone in the RF); on clipped frames it is WORSE than `scale_removed`, because the visible box is no longer the body; and the full-body/clipped gap widens. A second, cheaper test: augmenting `scale_removed` training with random ±10 % rescales should cost the CNN nothing (no size cue to lose), while the same augmentation on `metric_scaled` should cost accuracy.

---

## 4. Mechanism C — translation equivariance and pooling versus absolute position and size

**Claim.** Convolution is translation-equivariant; global average pooling makes the network translation-INVARIANT; three stride-2 pools after a stride-4 conv leave a final map whose cells span 32 input pixels. Absolute position survives only through zero-padding boundary effects (Kayhan & van Gemert 2020 [V]; Islam, Jia & Bruce 2020 [V]), and GAP removes even that. Absolute SIZE is not an invariant of a fixed-receptive-field CNN either: it appears only as which filters fire at which scale, entangled with distance. Body geometry is exactly "which body part is at which row, how wide, how far apart" — a spatial ARRANGEMENT — and GAP answers "is this feature present anywhere?" but never "where?" (`spatial_head` docstring).

**Our evidence.**
- The head asymmetry (§12.1b): replacing `gap` with `flatten`/`stripe` lifts R1 depth `interior_only` 40.92 → 69.59 (+28.7 pp), `scale_removed` 47.62 → 66.90, `sil_scaled` 52.73 → 69.67 — and moves RGB `scale_removed` by −0.2 pp. A nearest-class-mean template matcher on 96 PCA components of the ALIGNED pixels scores 60.5 % where the GAP CNN scores 40.9 % on more frames (`spatial_head` docstring). A linear template beats a deep network only if the network is discarding the correspondence.
- Position does different jobs at different rungs (§8.2): re-centring is worth nothing within a session (R1 23.31 → 23.81) and everything across one (R4 7.87 → 11.92), because within a session each person stands in the same place — position IS identity there — and across sessions the camera moved 34 px. A model that reads position as identity is hit exactly where position moved.
- `person` at R4 (7.87–10.66) sits within a few points of the 5.35 % floor: with size and position kept and a moved camera, what the network learnt in session A does not transfer.

**Checkable prediction.** With the `stripe` head, a row-wise occlusion sweep on `scale_removed` frames should concentrate importance on the head–shoulder rows (where the outline carries proportion information) and permuting the H stripes consistently at test time should drop `stripe` accuracy to the `gap` level. Also: under `person` (size and position kept), test-time accuracy should fall monotonically with the per-frame vertical offset between the test subject's bbox centre and the training-session mean for that subject — the network's position cue read out as a nuisance.

---

## 5. Mechanism D — why 2 bits ≈ 16 bits and silhouette ≈ depth imply edge/outline features

**Claim.** `quantise_depth` bins the fixed 0–6,000 mm range; at 2 bits a level is 1,500 mm and at 4 bits 375 mm. Under `scale_removed` the person's median is shifted to 3,000 mm and the body's own depth extent has a median of 419 mm (`scale_remove` comment), so at 4 bits the whole body sits inside one level and at 2 bits inside one level with certainty: the interior relief is DELETED while the person/fill boundary is untouched. If accuracy does not move, the network was not reading the interior. Trained conv filters are edge and gradient detectors; the only place a normalised depth crop HAS gradients the sensor can resolve is the outline.

**Our evidence.**
- R4 is flat within noise from 16 to 2 bits: 13.62 / 12.45 / 13.02 / 11.22 / 12.17 (§8.4, §14.6). The 1-bit row (9.51) is lower for a legible reason the record insists on: a single threshold at 3,000 mm cuts THROUGH the body whose median is exactly there, so it damages the outline — it is a thresholded depth map, not a silhouette. The shipped-mask silhouette, which keeps the outline crisp, scores 12.97–16.67 at R4 — as high as any depth condition.
- At R1 the same axis falls 47.62 → 33.61 from 16 to 2 bits: WITHIN a session the interior does carry information — but §8.2 shows what kind: standing distance and the room (bg_hole ≈ full at R1, bg_plate > full at R3). The interior is useful when it encodes the recording, not the person.
- Physical reason the interior is empty at this range (§13.17 item 1): the quantisation step at 3 m is ~26 mm against ~15 mm of facial relief; the person spends ~7 % of the 6,000 mm input range ("near-flat plate", `scale_remove` comment). Stretching contrast 15× with `--depth-slab-mm` did nothing (§12.2) because BatchNorm absorbs a linear rescale; surface normals were neutral at R1 and harmful at R4 because derivatives amplify noise. `interior_only` (rim eroded 2 px, same normalisation) scores 38.22 at R1 against `scale_removed` 43.06 (§14.6): removing the last two pixels of outline costs more than the whole interior contributes.
- `sil_scaled` ≈ `scale_removed` at R4 (§14.6), and the R4 fusion oracle (§13.10) shows the frames the CNN gets right are not the frames the metric model gets right — consistent with outline proportions and millimetre lengths being different quantities.

**Checkable prediction.** (i) Gradient or occlusion saliency on R4 `scale_removed` and `sil_scaled` cells should localise on the boundary, with the interior contributing near zero. (ii) A `rim_matched` condition — inpaint the fill outside the mask to the rim depth so the boundary has ZERO contrast while the interior is untouched — should fall to the floor at R4 (a stronger test than `interior_only`, whose eroded mask still has an edge). (iii) The 16-vs-2-bit gap at R4 should remain zero with the `stripe` head; it should reappear only if training identities grow enough for the interior to be learnable (see §9 prediction).

---

## 6. Mechanism E — texture and local-statistics bias, and why RGB re-ID works on clothing that depth does not have

**Claim.** ImageNet-trained CNNs prefer texture to shape (Geirhos et al. 2019 [V]) and do not classify by global object shape (Baker et al. 2018 [V]); networks preferentially represent the most linearly decodable feature and partially suppress harder task-relevant ones (Hermann & Lampinen 2020 [V, arXiv]); models achieve non-trivial accuracy from image BACKGROUNDS alone (Xiao et al. 2020/2021 [V, arXiv]). Stature, shoulder width and thickness in millimetres are hard, global, non-local features; clothing colour and print are easy, local, texture features; the room is an even easier one. A depth crop has no texture — a smooth surface with one edge — so the local statistics a CNN feeds on are the outline's.

**Our evidence.**
- RGB within session: 96.33 % at R1 and 79.51 % at R3 with the full frame; across sessions 5.59 % (full) and 17.72 % (`scale_removed`) (§8.3). The clothing cue is exactly the local-texture cue, and it fails when the clothes change; seed 2 puts RGB at 1.90 %, below chance — "the clothing cue actively misleads" (§8.1). Depth never had that cue and never had that far to fall: R1 43–70 %, R4 12–14 %.
- The depth analogue of Xiao's background-only result: within session, the background with the person inpainted OUT (`bg_plate`) scores 38.57 % against the full frame's 34.82 % and the person alone 8.65 % (§8.2). The easiest feature — the room, the standing spot — is what training selects; the hard feature — body geometry — is never learnt; across a session the room moved, and the model has nothing left.
- The character of the CNN's errors is what shortcut features produce: SYSTEMATIC per recording. Integration lifts the metric model 28.86 → 50.71 % across a whole tracklet and the plain CNN only 18.40 → 20.71 % (§13.12c, §13.17 item 2): it makes the same wrong call frame after frame. Augmentation (`aug8`) converts that systematic error into noise — the augmented CNN gains 23.40 → 32.62 → 37.78 across the window sweep (§13.12) — which is the same result seen from the other side: framing-dependent cues, once randomised, stop being memorisable.
- ConvNeXt, with ImageNet's texture-biased filters, replicates AlexNet's ordering on the full frame (R3 36.74 vs 34.82, R4 8.45 vs 6.70); pretrained RGB filters supply edges and textures, not unprojection.

**Checkable prediction.** Per-frame argmax autocorrelation within a tracklet should be much higher for the plain-CNN posteriors than for the metric RF's — the direct signature of systematic vs. noisy error — and should FALL for the `aug8` cells. It can be read off the stored `cm_*.npz` predictions with no retraining.

---

## 7. Mechanism F — why the ImageNet-pretrained ConvNeXt on the full frame does not escape

**Claim.** Pretraining transfers filters tuned to RGB statistics; the RGB-D literature learnt early that a depth image must be re-ENCODED before a pretrained CNN can use it — Eitel et al. 2015 [V] colourise the depth map with a jet colour map so that ImageNet filters apply; Gupta et al. 2014 [V] replace raw depth with HHA: horizontal disparity, HEIGHT ABOVE GROUND and ANGLE WITH GRAVITY per pixel. HHA is the metric content our features compute, placed in the input channels precisely because the network cannot compute it. We gave ConvNeXt neither: a three-channel copy of the same normalised depth, ImageNet mean/std applied (§9).

**Our evidence.** ConvNeXt-Tiny/ImageNet on the full frame at R4: 8.45 % [floor 5.35], R3 36.74, R1 78.02 (§8.1 table). The gain over AlexNet is +2–5 pp, inside the ~6 pp minimum detectable effect (§12.2); it reads the same room at R3 and collapses the same way at R4. Data is the other half: ~7,894 near-duplicate frames from one recording per person against a network with tens of millions of parameters (§13.17 item 3) — and R3, trained on 1,821 frames of the SAME day and clothes, beats R4's 7,894 frames from another day on the same test set (CNN 34.82 vs 6.70; image floor 8.55 vs 5.35), so distribution proximity matters more than frame count.

**Checkable prediction.** An HHA-style input built from `hiride_metric.py`'s unprojection — channel 1 = height above ground in mm (camera-Y fallback), channel 2 = lateral offset, channel 3 = depth — resampled to a fixed mm-per-pixel orthographic grid, should beat `scale_removed` at R4 on full-body frames. The decisive control is bit depth on the HEIGHT channel: quantising it to 2 bits over 0–2 m (500 mm levels, five times the 105 mm stature SD) should erase the gain, while ~6 bits (≈ 31 mm) should keep it — the opposite of the flat axis measured on the intensity image. If that pattern appears, the network can use millimetres when they are put in a channel; the failure was representation, not capacity.

---

## 8. Mechanism G (adjacent) — the training and validation regime rewards within-session cues

**Claim.** `_tail_val` carves validation from the contiguous tail of each TRAINING recording and `EarlyStopping` watches `val_accuracy` there (`hiride_train.py`). Model selection therefore rewards whatever predicts identity WITHIN the training session — room, standing spot, apparent size — which is exactly the cue that fails at R4. This does not create the representational limit, but it removes any pressure to overcome it.

**Our evidence.** The mechanism suite ordering (§8.2): every condition that removes a within-session cue (person, centring, rescaling) LOWERS R1 and RAISES R4. A selection criterion computed within session cannot see that trade-off. The 2023 frame-random policy (R0) is the extreme case: 98.36 % depth, 89.38 % from thirteen trivial scalars (§0).

**Checkable prediction.** Early-stop on a cross-recording validation set (hold out `Testing/Still` frames of the 28 when training on `Training`) and the selected epoch should move earlier and R4 should change by a small, positive, measurable amount; the effect should be larger for `person` than for `scale_removed`, because `person` has more within-session cue to over-fit.

---

## 9. Why the field's depth networks that do better differ in input representation or training data

- **Haque, Alahi & Fei-Fei, CVPR 2016 [V]** (`haque2016.txt`; Crossref DOI 10.1109/cvpr.2016.138). Their input is a 4-D tensor of size 250 × 100 × 200 (x, y, z voxels plus time), i.e. a POINT CLOUD "constructed from depth images" — unprojection is done before the network, and body size is a voxel count in metric units. A 4-D glimpse encoder plus recurrent attention reaches 30.1 % single-shot (3D RAM) and 45.3 % multi-shot (4D RAM) on BIWI at 50 classes (their Tables 2–3). That is the geometric representation our CNN never saw; it is not evidence that a 2-D CNN on a depth image can recover geometry.
- **Karianakis, Liu, Chen & Soatto, ECCV 2018 [V]** (`karianakis.txt`; Crossref DOI 10.1007/978-3-030-01228-1_44, Springer LNCS, 2018). Body-index crop from the skeleton tracker, range-normalised grey representation D_gp in [1, 256 − 56], split-rate RGB→depth transfer that shares the bottom layers of an ImageNet-scale RGB model without fine-tuning, and reinforced temporal attention that WEIGHTS frames. BIWI: 25.4 % single-shot (CNN), 50.0 % with RTA (their Table). Two points matter for us: their range normalisation also discards absolute depth, so their frame-level CNN is in our regime (25.4 % against our ~19–21 % gated single-frame); and their gain comes from transfer plus learned frame weighting — a gate — which is the same lever that moved our number (§13.12), not a representational fix.
- **Shen et al., LidarGait, CVPR 2023 [V]** (`lidargait.txt`; Crossref DOI 10.1109/cvpr52729.2023.00108). At a FIXED encoder (GaitBase) the range-view depth projection beats the LiDAR silhouette 86.77 % vs 64.70 % rank-1 on SUSTech1K (their Fig. 8). Three differences from our regime: 1,050 subjects and 25,239 sequences (three orders of magnitude more identities than 28); LiDAR range values with centimetre-class precision projected by the authors with known geometry; and normalised-then-colourised depth fed to a gait backbone trained for it. The audit (§14.7) already flags this as contradicting "outline, not 3-D shape" at fixed encoder — the honest scope is: with Kinect v1 at 2–4 m (26 mm step), 28 subjects and one training recording each, the depth CNN does not learn the interior; with LiDAR precision and a thousand identities it can.

**Checkable prediction (data vs. sensor).** If the flat bits axis is a DATA limit, it should break as training identities grow: on TVRID (88 identities) or under the standard 50-class protocol (wave 22), a 16-vs-2-bit gap should begin to open. If it stays flat there too, the sensor-resolution explanation (26 mm step vs 15 mm relief) dominates. Either outcome is informative and the run is cheap.

---

## 10. What this lens does not claim

- It does not claim a CNN CANNOT compute millimetres — a universal approximator can represent the product; the claim is that its inductive bias (local, translation-equivariant, size-and-position-invariant after normalisation) gives it no reason to, and the pre-registered tests of giving it the missing term (§13.17) and the finished features (§14.6) both came back null. That is an argument about what the network DOES at this n, not what it could do.
- The CNN's 28 % at W = 25 and its complementarity with the metric model (oracle +7.7–9.0 pp, §13.10, §14.6) show the outline carries real identity: scale-free proportions are a biometric too, only a weaker and less integrable one than millimetres.
- Every R4 single-contrast interval straddles zero at n = 28 (§14.6). The lens rests on the ORDERING of cells across rungs, heads, bit depths and gates, which no single interval controls.
- The metric pipeline's own defect (base-anchored bands, 89.5 mm/m drift) is a representation failure of the SAME kind — a pixel-space anchor imported into a metric feature — and is what the full-body gate repairs. The point is not that hand features are immune; it is that in millimetres the defect could be diagnosed and gated, whereas in the CNN it is invisible.

---

## 11. Literature, verified

| item | status | what was confirmed | use in the paper |
|---|---|---|---|
| Geirhos, Rubisch, Michaelis, Bethge, Wichmann, Brendel — "ImageNet-trained CNNs are biased towards texture; increasing shape bias improves accuracy and robustness" — arXiv 1811.12231, ICLR 2019 (oral) | [V] arXiv abs page (comments: ICLR 2019 oral) | title, authors, venue | mechanism E |
| Baker, Lu, Erlikhman, Kellman — "Deep convolutional networks do not classify based on global object shape" — PLOS Computational Biology 2018, DOI 10.1371/journal.pcbi.1006613 | [V] Crossref | title, authors, journal, year, DOI | mechanism E |
| Kayhan & van Gemert — "On Translation Invariance in CNNs: Convolutional Layers can Exploit Absolute Spatial Location" — arXiv 2003.07064, CVPR 2020 | [V] arXiv abs page | title, authors, venue; claim: boundary effects let conv layers exploit absolute position | mechanism C |
| Islam, Jia, Bruce — "How Much Position Information Do Convolutional Neural Networks Encode?" — arXiv 2001.08248, ICLR 2020 | [V] arXiv abs page (comments: accepted to ICLR 2020) | title, authors, venue; claim: CNNs implicitly encode absolute position | mechanism C |
| Eitel, Springenberg, Spinello, Riedmiller, Burgard — "Multimodal Deep Learning for Robust RGB-D Object Recognition" — arXiv 1507.06821, IROS 2015 | [V] arXiv abs (comments: final version submitted to IROS 2015) and PDF text: depth "colorizing" and "apply a jet colormap" to reuse ImageNet-pretrained weights | title, authors, venue, encoding | mechanism F |
| Gupta, Girshick, Arbeláez, Malik — "Learning Rich Features from RGB-D Images for Object Detection and Segmentation" — arXiv 1407.5736, ECCV 2014 | [V] arXiv abs page | title, authors, venue; HHA = disparity, height above ground, angle with gravity | mechanism F |
| Xiao, Engstrom, Ilyas, Madry — "Noise or Signal: The Role of Image Backgrounds in Object Recognition" — arXiv 2006.09994 (v1 June 2020) | [V] arXiv abs page and saved extract `xiao2021.txt` (title, authors, abstract). ICLR 2021 venue **[UNVERIFIED]** — Semantic Scholar returned 429, dblp blocked, Crossref does not index it | background-only accuracy is non-trivial; misclassification up to 87.5 % with adversarial backgrounds | mechanism E |
| Hermann & Lampinen — "What shapes feature representations? Exploring datasets, architectures, and training" — arXiv 2006.12433 | [V] arXiv abs page (title, authors, claim). NeurIPS 2020 venue **[UNVERIFIED]** — same index failures | networks prefer linearly decodable features; task-irrelevant ones partially suppressed | mechanism E |
| Haque, Alahi, Fei-Fei — "Recurrent Attention Models for Depth-Based Person Identification" — CVPR 2016, DOI 10.1109/cvpr.2016.138 | [V] Crossref + saved extract `haque2016.txt` (4-D tensor 250×100×200; BIWI 30.1 single / 45.3 multi) | | §9 |
| Karianakis, Liu, Chen, Soatto — "Reinforced Temporal Attention and Split-Rate Transfer for Depth-Based Person Re-identification" — ECCV 2018, DOI 10.1007/978-3-030-01228-1_44 | [V] Crossref DOI record + saved extract `karianakis.txt` (D_gp representation; BIWI 25.4 / 50.0) | | §9 |
| Shen, Fan, Wu, Wang, Huang, Yu — "LidarGait: Benchmarking 3D Gait Recognition with Point Clouds" — CVPR 2023, DOI 10.1109/cvpr52729.2023.00108 | [V] Crossref + saved extract `lidargait.txt` (Fig. 8: 64.70 → 86.77 at fixed GaitBase encoder; 1,050 subjects, 25,239 sequences) | | §9 |

Not cited because not verified in this session: Geirhos et al. 2020 "Shortcut learning" (would fit mechanism E; fetch before use).

---

## 12. Sentences the manuscript can carry

1. "A length in millimetres is the product of a pixel extent and a depth value divided by a focal length the network is never given; convolution and pooling are built to be invariant to exactly the two factors of that product, and the normalisation that made the CNN's cross-session number non-trivial (`scale_removed`) sets both factors to constants before the network sees the image."
2. "Handing the network the missing factor (standing distance, pre-registered) or the finished features (twelve scalars at the head) changed nothing within seed variance; the network was not one multiplication short of the metric features, it was not attempting the multiplication."
3. "Accuracy is flat from 16 to 2 bits of depth precision and a normalised binary silhouette matches normalised depth: at Kinect v1 range the interior is below the sensor's step and the trained filters are reading the outline."
4. "The field's depth networks that do better either unproject before the network (Haque et al. 2016, a 4-D point-cloud tensor), add learned frame weighting and RGB transfer (Karianakis et al. 2018), or train a thousand identities on centimetre-precise LiDAR range views (Shen et al. 2023); none shows a 2-D CNN recovering millimetres from a Kinect v1 crop at n = 28."
5. Scope sentence for the LidarGait contradiction: "with this sensor, at this range, with one training recording per person, the depth CNN learns outline; with LiDAR precision and a thousand identities the same encoder family learns interior depth — the difference is representation and data, not the encoder."
