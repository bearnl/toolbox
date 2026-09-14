# Paper 2 — the essence (rev 3, 2026-09-11: the author's reframe; venue T-BIOM)

Approved direction (author, 2026-09-10/11): not "depth is not private" but **what carries discriminative information in depth video** — approached from two sides that converge on a result that is both usable and explainable. Depth *is* opaque to human observers (Haque et al. 2016 measured people at 6.7 % on BIWI depth, chance 2 %); that is the premise, cited, never our finding. Every number below is from the regenerated artifacts (handoff §14.6, §14.10, §14.11); every novelty sentence is at the ceiling the refutation passes set (`08_refute_*`, `09_cnn_failure_refute_*`).

## Working title

**What Carries Identity in Consumer Depth Video? A Two-Sided Evaluation of Person Identification with HI-RIDE**

## Thesis, one paragraph

Depth cameras are installed in care rooms, lavatories and clinical corridors because a depth frame shows a person no human can recognise. What a machine can extract from it, from where in the frame, and under which conditions, has been reported only at the two ends — near-perfect within a session (98 % here; 100 % in MultiGait; 97.8 % frame-level on BIWI in Patruno 2019) and 21–50 % across a change of day and clothes (Munaro, Haque, Wu, Karianakis, 2014–2018; TUM-GAID 2014; MultiGait 2026) — never resolved in between. We resolve it from both sides. **From the top**, a near-perfect depth classifier (our own 2023 model) is taken apart one shortcut at a time — frame adjacency, the recording, the session, then the room, the position, the size, the depth values — until what remains is a normalised outline worth ~13 % on a single frame of 28 people. **From the bottom**, an identifier is built from nothing but named quantities — twelve body measurements in millimetres, a validity rule for frames in which the body is clipped, and time — until it reaches 43 % from 2.5 s and 41–47 % from a whole walk under the field's own protocol. The two sides meet exactly: with the best recipe the network and the measurements extract the same identity from a single frame (18.4 vs 18.8 %), the network carrying it as an outline (accuracy flat from 16 to 2 bits of depth), the measurements as lengths. The measurements pull ahead for one stated reason — a measurement can be refused when it cannot be taken, and averaged when it can; without the gate the two are tied at every window (23.6 vs 23.6 % per recording). That is the operational content of "explainable", and it is what the field's 45–50 % multi-shot numbers buy with recurrent attention and LSTM frame weighting.

## Side A — decomposition (C1 + C2)

The ladder, one model, one dataset, only the split changes: R0 frame-random 98.4 → R1 block + *within-recording guard sweep at matched training size* 62.3 → R3 cross-recording 34.8 → R4 cross-session 6.7 % (RGB 99.99 → 91.9 → 79.5 → 5.6), read against a *detector-metadata-only floor* (13 bounding-box scalars: 89.4 → 17.7 → 8.5 → 5.4 %) and a permutation null at every rung; replicated on three corpora and three sensor technologies (TVRID depth 62.6 → 13.5 → 8.4 → 6.8 % on 86/88 identities), 5–9× inflation everywhere. Then single controlled edits at R4: person 9.2, re-centred 11.3, size+position removed 13.6, normalised silhouette 13.0 %; the room's share flips sign across the session change (R3: plate 38.6 > full 34.8 > person 8.8 %); 2 bits ≈ 16 bits; the interior contributes ≲ 3–4 pp. *Written as*: same-session vs cross-session inflation is known (HumanID 2005; TUM-GAID 2014; MultiGait 2026; and in touch, ECG, EEG, MRI, wildlife re-ID); graded ladders exist elsewhere; background/person controls are standard in RGB (Xiao 2021; Tian 2018), inpainting as a leakage control was used for wildlife (Rueda-Toicen 2026), and RealGait swept intensity precision from the silhouette up. To our knowledge new: the within-recording guard-band sweep at matched n, the metadata-only floor, the exact own-plate complement resolved along the ladder, the depth-*value* precision axis, background-only exceeding the full frame, and the full ladder on one depth pipeline across three sensors. Not claimed: the leak, the block split, "time", the first graded hierarchy in vision, "the depth field never saw its inflated number", any general "CNNs read outline not 3-D shape" (scoped to Kinect v1 at 2–4 m, one recording per person; LidarGait shows the opposite at a thousand identities).

## Side B — reconstruction (C3 + C4)

Twelve lengths in camera-frame millimetres from unprojected depth through the sensor's runtime mask — stature, two height percentiles, thickness, six widths at fixed fractions of stature, area, a volume proxy — no skeleton tracker (the ground plane parsed on 0.5 % of frames; camera pitch is a disclosed residual). Random forest. Single frame R4 19.0 % [14.0, 24.4] vs the ladder CNN 6.7 % (full frame) / 13.6 % (normalised person). 74 % of walking frames clip the body; stature drifts 89.5 mm/m against a 105 mm between-person SD; a *validity gate* (mask touches neither frame edge) → 28.1 % on whole bodies. 2.5 s → **43.3 % [30.3, 56.1]**, fusion 47.0 % (fusion − metric +3.7 [−13.8, +20.4], not resolvable; fusion − CNN +18.8 [+9.0, +29.4]); top-3 81 %. Per-person 0–100 % (metric: 6 of 28 never, 4 always). Cohort K = 4 → 28: 83.9 → 43.3 %, 58-pp spread between draws at K = 4. **Commensurability (wave 22, the field's protocol — all 50 Training recordings, 28 probes, whole Walking recording):** per-frame CNN averaged 25.0 % (Haque's plain 3-D CNN + averaging 27.8); gate + measurements averaged **41.4 %**; fusion **47.1 %** — inside the published 43–50 % (Munaro 42.9; 4D-RAM 45.3; CNN-LSTM 45.7; RTA 50.0), every one of which, like ours, rests on 28 decisions (±9 pp). Ungated, CNN and measurements tie at every window (23.6/23.6). On standing whole bodies (Still probe) the measurements lead from the first frame (36.8 vs 23.9 %; whole recording 43.6 %, above Wu's 39.4 five-shot and Munaro's 32.5) and the network gains nothing from averaging. *Written as*: anthropometric soft biometrics from consumer depth are 2012–2018 practice (Barbosa; Munaro; Andersson; Paolanti) — not our discovery; gates were applied since Barbosa 2012 and Karianakis 2018 learned frame weights; observation-length curves are standard in gait; cohort curves and the menagerie are known. To our knowledge new: the first quantification of clipping prevalence and height drift, an explicit mechanism-derived gate ablated across sessions, depth-image accuracy against a wall-clock budget, the first cross-session depth cohort curve with draw spread and per-person distribution, and the side-by-side of a learned model and a measured one on identical frames under both the ladder and the field's protocol. **C4**: RGB's edge is clothing — +19.5 / +38.7 / +33.4 / +1.3 pp by rung; on Still under the standard protocol depth beats RGB by 4.0 pp in every seed; RGB with its own segmenter equals RGB with the sensor mask (a control with no found precedent). Replication of Hafner 2022, Karianakis 2018, Castro 2020, PRCC/LTCC, extended.

## The convergence, and why the network stopped where it did

Single-frame parity (18.4 vs 18.8 %) is the meeting point. What the record supports about the network: it reads the outline (bits axis; silhouette ≈ depth), it does not *use* size even when handed it (`--aux dist` pre-registered null; `--aux metric` null), and the gap-head deficit was closed by the head and augmentation under the same training regime — so "cannot compute millimetres" and "gradient starvation" are *not* the explanation and will not be written. What separates the two afterwards is measurement discipline: the gate (+10 pp for the measurements, +3 for the network, which is the more clipping-robust of the two) and integration (+14 vs +7 at 2.5 s). "Systematic errors" is written only after the arithmetic-mean re-aggregation excludes the float16 product-rule veto (stored posteriors; minutes). Fusion's contribution lives on walking frames and is never resolvable; on standing bodies it is nil — the useful combination is frame validity, not a mixing weight.

## Negative results (one table, kept)

Score-, distance- and feature-level fusion fail to realise a +7.7–9.0 pp oracle; feature-averaging over a window *lowers* accuracy (28.9 → 23.3 % at 25 frames) because the clipping bias is correlated across a window — gate, do not average; the crown-anchored shape block helps single frames and hurts after integration; gait cycles could not be segmented under *our* protocol (Haque 2016 did get GEI/GEV on whole sequences); per-identity duplication → drop correlation refuted (room confound); 2023 efficiency claims arithmetically false; label-free drift selection cannot work here (the bias is out of the training distribution by construction). Two pre-registered hypotheses refuted.

## Structure — T-BIOM regular paper, 10 pages including references, supplementary for the rest

I Introduction (premise; the two ends; the two sides; contributions, audit-bounded) · II Related work (depth/RGB-D ID on BIWI and beyond; split leakage and attribution controls; anthropometric soft biometrics; MultiGait and TUM-GAID as the endpoints) · III Data, protocol and measurement model (three corpora; ladder R0–R4 and the field's rung; floor and null; eligibility and the validity gate; the network and the single-edit conditions; the twelve measurements — camera-frame; aggregation; statistics in ISO/IEC 19795 terms; pre-registration) · IV Taking the classifier apart (Table 1, Figs 1–2; attribution Fig 3 + one table; bits Fig 4) · V Building the identifier (Fig 5 clipping; Fig 7 operating point; Fig 8 cohort; the commensurability table) · VI The convergence: what the network read, what the measurements measured, where they part · VII What did not work · VIII Discussion (what "explainable" means operationally; what a deployment should know; the modality contrast; limits: n = 28, one sensor generation for the main corpus, camera-frame mm, no perception study) · IX Conclusion. Supplementary: full ladder tables for all corpora, guard sweep, all mechanism cells, per-subject tables, the pre-registration document, reproducibility manifest.

## What will not be claimed

SOTA; open-set re-ID; skeletons (paper 1); "depth is under-explored"; model size or speed; "depth is not private" or "as identifying as RGB"; "depth identifies across sessions" as a discovery; "first to show the split leaks"; "cannot compute millimetres"; "gradient starvation"; "systematic errors" before the veto check; "gravity-aligned"; "13-dimensional descriptor" (it is 12); any mechanism or bit-depth number from the 2026-08-31 `tables.tex`.

## Venue — decided 2026-09-11: IEEE T-BIOM

Biometric evaluation study is the venue's native genre (Hou 2023; Rosberg 2026); "reproducibility is highly valued"; reviewers speak ISO 19795. Fallbacks in order: CVIU (scope invites "insights that differ from predominant views"; Hafner 2022 on BIWI; no page limit), Pattern Recognition (Zhang 2026 silhouette-disentanglement precedent), IEEE Access (Negative Result article type). Not pursued: TIFS, PoPETs (under this frame), IMWUT, npj Digital Medicine, IEEE T-Privacy (rejected the 2023 paper).

**Raising the odds at T-BIOM:** (1) one modern encoder at fixed protocol before submission — **wave 23**: ConvNeXt-Tiny/ImageNet on `scale_removed`, R4 and both standard rungs, gap head and the best recipe, 5 seeds; (2) the commensurability table in the body with the 28-decision caveat on the published numbers too; (3) ISO 19795 reporting throughout (checklist 05); (4) novelty sentences at the refutation ceiling; (5) artefacts: split manifests, plates, seeds, code, pre-registration; (6) length discipline: mechanism suite as one figure + one table, negatives as one table, everything else supplementary.

## Author's decisions on record

2026-09-09: IEEE T-Privacy rejected the 2023 paper; wave 22 approved and run; keep "HI-RIDE"; in-house participants all consented; IAS-Lab no response → TVRID closes external validity. 2026-09-10: reframe to "what carries discriminative information", two sides converging, depth privacy-preserving for human eyes as premise. 2026-09-11: T-BIOM primary; wave 23 approved; `main.tex` started on IEEEtran in `papers/paper-depth/submissions-TBIOM-2026/`.

## Addendum 2026-09-13 — wave 23 changes the meeting point (rev 4 pending)

The modern-encoder control did not land in the AlexNet cluster: ConvNeXt-Tiny/ImageNet on the
normalised person reaches 25.9 % [19.2, 33.0] single-frame and 49.3 % whole-recording at R4,
and 47.3 % whole-walk under the field's protocol — the published band, with no temporal
model and no gate. "Single-frame parity" and "the measurements pull ahead only via the gate"
hold for the from-scratch network; against the pretrained one the measurements are behind on
all frames, level after gating, ahead on standing whole bodies (36.8 vs 31.4), and far behind
on clipped walking frames. Rev 4 will restate the convergence as: both sides reach the band;
the network is the clipping-robust reader, the measurements the precise and explainable one;
whether the pretrained encoder reads the outline or the interior is wave 24's question; the
ConvNeXt gated / fusion numbers (`submit_sequence.sh`) set the operating point. Handoff §14.13.

## Rev 4 — 2026-09-13, after waves 23/24, the ConvNeXt operating point and the error-structure check

**The meeting point, restated.** Both sides reach the field's band, and each carries identity a
different way. The from-scratch network and the twelve measurements extract the same identity
from a single frame (18.4 vs 18.8 %); an ImageNet-pretrained ConvNeXt on the same normalised
person extracts more (25.9 %), and on whole bodies it and the measurements are level again
(29.6 vs 28.9 %). What the pretrained encoder reads is still, dominantly, the outline: 2-bit
depth and the binary silhouette both give 23.3 %, the interior alone 23.8 %, full-precision
depth 25.9 % — a seed-consistent 2.5 pp from the depth values that the subject-level
intervals cannot resolve. Under the ladder's R4, gated 2.5 s: ConvNeXt 48.2 % [33.2, 63.2],
measurements 43.3 % (48.2 % under the arithmetic-mean rule), fusion **58.6 % [43.5, 73.2]**,
top-3 84 %; whole recording 55.7 / 50.7 (54.3) / 62.1 (72.1). Under the field's protocol,
gated whole walk: fusion 59.8 %, above the published 43–50 %; ungated ConvNeXt alone 45.5 %.
On standing whole bodies the measurements lead the pretrained network (41.5 vs 31.2 % at
2.5 s); on walking frames the network is the more clipping-robust reader; fusion of the two is
a real gain (+15 pp over the measurements, resolvable in two of three settings) where the
AlexNet fusion was not.

**Two corrections the error-structure check forces.** (1) The product-rule aggregation with
float16 posteriors vetoed the *measurements'* windows: on identical decisions the arithmetic
mean gives 48.2 % instead of 43.3 % at 2.5 s gated and 42.9 % instead of 26.4 % whole-recording
ungated. Every aggregated measurement number in the record is conservative by the rule; the
"ungated curve peaks and falls" reading was the veto, not the bias. Recommendation: switch all
aggregation to the sum rule (Kittler 1998) and regenerate figs 7–8. (2) "Systematic errors" in
the strong sense is not supported: wrong network windows are no more internally consistent than
wrong measurement windows. What is supported: the from-scratch network's confusions are
concentrated on a few identities (its i.i.d. plurality ceiling is only 31 %) and change with
initialisation (a stable confuser in 29–36 % of recordings vs 86–89 % for the measurements);
the measurements' errors are within-walk correlated by the distance bias (measured 8 pp below
their i.i.d. ceiling). Early stopping on within-session validation cost the pretrained network
1.7 pp. Both models fail together on clipped and out-of-range frames (54 vs 32 %).

**Closing sentence of the paper.** Usable is reached by both readers — 48 % from the network,
43–48 % from twelve lengths, 59 % together, from 2.5 s of depth on 28 people. Explainable is the
measurements' property alone: every input a length, every refusal a stated geometric condition,
every unidentified person assignable a cause in millimetres. What carries identity in consumer
depth video is body geometry seen as an outline; the depth values add a little to a pretrained
network and everything to a tape measure.


## Rev 5 — 2026-09-13, the sum rule adopted; draft 4 written

Author decision: every aggregated number is the sum rule (arithmetic mean of posteriors,
Kittler et al. 1998); the product rule appears once, as the check that found the float16
veto. Numbers that move (R4, gated, 2.5 s = 103 decisions): measurements **48.2 % [33.6, 61.9]**,
ConvNeXt 47.8 % [31.7, 63.7] — parity, −0.4 pp [−18.4, +17.4] — fusion **61.4 % [45.7, 76.2]**,
top-3 84 %, top-5 88 %; whole recording 54.3 / 55.0 / 72.1. Field's protocol, gated whole walk:
measurements 46.4, ConvNeXt 44.3, fusion **62.1** against the published 43–50; Still probe
whole 46.4 for the measurements. Ungated R4 measurements 35.0 → 42.9, so the gate is worth
+13 / +11 pp and the Rev-3 "tie at every ungated window" was the veto. Per-subject at 2.5 s:
measurements 8 never / 7 always (median 48 %); ConvNeXt fusion 3 never / 4 always (median 80 %).

Withdrawn from Rev 4: "the measurements' errors are within-walk correlated (8 pp below their
i.i.d. ceiling)". Under the sum rule both readers sit ~3 pp below their plurality ceilings
(network 30.9 → 27.6, measurements 51.5 → 48.2, ConvNeXt 49.4 → 47.8) and their wrong frames
agree with their neighbours at similar rates (lag-1 0.48 vs 0.45). What limits a reader after
2.5 s is its frame-level confusion matrix; what distinguishes the readers is whose confusions
those are — stable nearest neighbours in millimetres (86–89 % of recordings keep their
confuser across seeds) against confusions of the fit (29–36 % AlexNet, 57–61 % ConvNeXt).

Closing sentence of the paper, as drafted: 48 % from twelve lengths, 48 % from the pretrained
network, 61 % together from 2.5 s of gated depth on 28 people, 62 % over a whole walk under the
field's protocol. Explainable is the measurements' property alone; what carries identity in
consumer depth video is body geometry seen as an outline.
