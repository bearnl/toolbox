# 07 — Deep read: MultiGait (Todt et al. 2026) and Delécluse et al. (FG 2025 / arXiv 2026)

Auditor: single agent, no sub-agents. Sources: primary texts only. MultiGait read from the arXiv HTML (saved as `extracts/multigait_2609.01036.html` and `.txt`, 511 KB / 2,199 lines, fetched 9 Sep 2026); Delécluse read from `extracts/delecluse2026.txt` (arXiv PDF text). Metadata checked on the arXiv abs pages and Crossref. Legend: [V] = confirmed in the primary text or on a metadata page; [UNVERIFIED] = not stated in the text or inferred by me. All quotations are under 15 words.

---

## 0. Bottom line

1. **MultiGait never evaluates depth values.** Its "depth" numbers are binary-silhouette gait recognition (OpenGait models) on silhouettes obtained by subtracting an empty-room image, with height/width normalisation [V, §5.2, §4.5.2]. The authors say so themselves: the recognisers "only use a subset of the information present in the recordings: the extracted silhouettes" [V, §5.4]. This means MultiGait's depth result is a *silhouette* result. The manuscript's C2 attribution (outline vs 3-D shape vs room vs bit-depth) and C3 metric anthropometry are not touched, and MultiGait's finding that "the actual sensing technology has little impact on recognition accuracy" once silhouettes exist [V, §5.4] is direct corroboration of the manuscript's "the outline leaks" thesis.
2. **Exact MultiGait depth numbers** (Tables 7 and 8, GaitBase, four perspectives centre-low / centre-high / left / right): single-session 100.0 / 100.0 / 100.0 / 100.0 %; multi-session 76.7 / 72.5 / 65.0 / 65.3 % [V]. The handoff's "60–80 % cross-session" should be read as "65–77 % with the best model; 27–67 % with the others". The multi-session probe set is at most 64 people (those recorded twice or more), classified against all 199 training identities [V, §5.3]; the phrase "on 199 people" must not be used for the cross-session figure.
3. **MultiGait single-session is not frame-random.** The split is 70/10/20 *within identity over walk samples* (one sample = one full traversal of the walkway) within one randomly chosen session [V, §5.3]. On the manuscript's ladder it sits closest to R3 (cross-recording, same session), not R0. MultiGait has no frame-random split, no block/guard sweep, no trivial-cue floor, no temporal-leakage analysis for identity (it does show a within-identity vs across-identity split effect, but only for *activity* inference, Tables 3 vs 4 [V]).
4. **None of the manuscript's methodological items appear in MultiGait**: no clip-length or observation-budget analysis, no validity gating, no background-only / person-masked / inpainted ablation, no bit-depth axis, no per-person identifiability distribution, no fusion of hand-crafted and learned features, no anthropometric features (height/weight are *predicted attributes*, Table 9), no care-setting threat model (smart-city framing throughout). A cohort-size subset test exists only for radar/CSI systems (Table 5), not for depth.
5. **Delécluse et al. is a published IEEE FG 2025 paper** (DOI 10.1109/FG61629.2025.11099215, Tampa/Clearwater, published 26 May 2025 [V, Crossref]); the arXiv preprint 2606.23230v1 appeared 22 Jun 2026 [V]. Cite it as FG 2025 with the arXiv as the preprint, not as "June 2026 work".
6. **Delécluse's BIWI protocol is under-specified.** The text never says which two BIWI sequences form query and gallery, never gives frames per decision, and never gives the BIWI training set for Table II. It misdescribes BIWI as "78 individuals across 106 sequences" [V, §IV-A-3] (BIWI has 50 people; 28 re-recorded; 50+28+28 = 106 sequences [V, Wu 2017 §V-B; Karianakis 2018 §4]) and labels the Kinect v1 "Time of flight" in Table I (it is structured light [V, Barbosa 2012; Patruno]). Every BIWI CMC value in Tables II and IV is an exact multiple of 1/28 (17/28 = 60.7 %, 27/28 = 96.4 %, 7/28 = 25.0 %, 21/28 = 75.0 %, 19/28 = 67.9 %, 24/28 = 85.7 %), so the test set is almost certainly the 28 re-recorded subjects in a one-to-one pairing [UNVERIFIED — arithmetic inference].
7. **Delécluse's privacy claim is the strongest form of the rebutted premise**: it claims depth "inherently preserves privacy" (§I), calls the data "anonymized depth data" (§VI) and uses that to justify that the study "did not undergo formal ethical review board approval" (§VI) — in a paper whose contribution is re-identifying 17 of 28 people from that data [V].
8. **Nothing in either paper pre-empts the repositioned framing** (protocol ladder, attribution, operating point under a 2.5-s budget, pre-registered negative results). The one item to cite as methodological precedent is the same group's Hanisch, Todt, Patino, Evans, Strufe, "A False Sense of Privacy: Towards a Reliable Evaluation Methodology for the Anonymization of Biometric Data", PoPETs 2024(1), DOI 10.56553/popets-2024-0008 [V, MultiGait reference list] — the manuscript should cite it alongside MultiGait.

---

## 1. MultiGait — Todt, Morsbach, Dissert, Strufe (KIT / KASTEL)

### 1.1 Bibliographic record
- Title: "MultiGait: A Multi-Sensor Multi-Perspective Multi-Session Biometric Inference Benchmark and its Dataset" [V].
- arXiv:2609.01036v1, submitted Tue 1 Sep 2026 10:35 UTC; primary cs.CR, cross-list cs.CV; licence CC BY 4.0; DOI 10.48550/arXiv.2609.01036; no "Comments" field; no journal-ref [V, abs page].
- Manuscript is in ACM template with placeholder "DOI: XXXXXXX.XXXXXXX", "Conference: arXiv '26" and a template artefact "Received 5 June 2009" [V, HTML header]. Target venue unknown [UNVERIFIED]. Faces in figures "redacted to preserve double-blindness during peer-review" [V, Fig. 1 caption] — i.e. under review somewhere.
- Uses the SEBA framework (Todt, Hanisch, Strufe, arXiv:2407.06648) for experiments; code "will be made publicly available with the publication" [V, §5.2].

### 1.2 Cohort and sessions (§4.3 "Study Participants", "Time of recording")
- N = 199 participants, recruited from a student panel (ORSEE); over 18, able to walk unaided, German-speaking; instructed to avoid loose clothing and heels [V].
- Demographics: 58.8 % male, 40 % female; age 23.4 ± 3.4 y; height 177 ± 9.7 cm; weight 73.8 ± 14.5 kg; 73.4 % German nationality (Table 6) [V].
- Sessions: three. Session 1 over two weeks in November 2024 (all 199). Sessions 2 and 3 in February and March 2025 respectively. "135 participants once, 36 participants twice and 28 participants three times" [V]. Hence 64 people have two or more sessions. Inter-session gaps: roughly three months (Nov→Feb) and one month (Feb→Mar); exact dates not given [V for months; days UNVERIFIED].
- Per participant per session: 120 walk samples (normal walking 20 repetitions back-and-forth; fast, backpack, bottle crate, turnstile 10 each) plus five poses × 5 repetitions [V, §4, §4.3]. Each walk sample is "one sample for each time the participant walked the length of our setup" [V, §4.4]. Walkway length and sample duration are not stated in the text [UNVERIFIED].
- Ethics: university IRB approval; GDPR Art. 9 biometric data disclosed; 15 EUR / 10 EUR compensation; 183/199 (92 %) consented to data sharing [V].

### 1.3 Sensors and the depth sensor (§4.1, Table 2)
- Eight sensors on four boards: video (Azure Kinect, 1920×1080, 30 fps), **depth (Microsoft Azure Kinect, 640×576, 30 fps)**, NIR (modified Logitech C920, 1280×720, 10 fps), LWIR (Seek Thermal S314SPX, 320×240, 23 fps), lidar (Livox HAP TX, ~45,000 pts/s), radar (TI AWR2944, ~250 pts/s), CSI and BFI (Intel AX210) [V].
- Depth is explicitly introduced as "a commonly proposed sensor for “privacy-friendly” surveillance" with citations to Kepski & Kwolek 2014, Rougier et al. 2011, Villamizar et al. 2018 and **Mucha & Kampel 2022** [V, §4.1].
- Perspectives (§4.2): centre-low (baseline, perpendicular to the walking line, "around hip height"); centre-high (directly above centre-low, 2.2 m); left and right (2.2 m, angled). All sensors synchronised to millisecond accuracy [V]. Lighting: six 6500 K flood lights, indirect [V].

### 1.4 Pre-processing (§5.2 "Sensor-Specific Pre-Processing") — the crucial detail
- "all imaging-based sensors required silhouette extraction for all chosen recognition systems" [V].
- Video: DeepLabv3 segmentation. **Depth: "we simply subtract an image of the empty room for every frame."** NIR/LWIR: MOG2 background subtraction. Then thresholding and denoising [V].
- Lidar: points outside a manual bounding box removed; silhouettes rendered as a depth map from the centre-low viewpoint [V].
- Silhouettes are size-normalised: the authors attribute weak height/weight inference to "pre-processing that normalizes height and width of the extracted silhouettes" [V, §4.5.2]. The silhouette resolution (OpenGait default 64×44) is not stated [UNVERIFIED].
- Consequence: the depth stream contributes only the segmentation mask. No experiment uses depth values, surface geometry or metric size.

### 1.5 Models (§5.1)
- Imaging sensors (video, depth, NIR, LWIR): GaitBase, DeepGaitv2, GaitPart, GaitSet, GaitGL (OpenGait implementations) and GEINet (own implementation) [V].
- Lidar: LidarGait, LidarGait++, GEINet. Radar: mmGaitNet, SRPNet, mID. CSI: BFId, LW-WiID, CAUTION, FreeSense. BFI: BFId [V].

### 1.6 Evaluation protocol (§5.3)
- Metric: "identification accuracy" (top-1 classification against the training identities); mean ± SD over five repeats [V]. No rank-k curves, no CMC, no mAP, no open-set, no verification (the words "gallery", "probe", "rank", "open-set", "closed-set" do not occur in the methods) [V by grep].
- **Single-session**: one session chosen at random per participant; per identity, 70 % of walk samples train / 10 % validation / 20 % test ("within identities") [V]. Hyper-parameters tuned by Optuna on centre-low, up to 50 iterations or until >99 % validation accuracy; then reused for all perspectives and for multi-session [V].
- **Multi-session**: session 2 is the test set; "all recordings from the other weeks are used for training" — i.e. sessions 1 and (where present) 3, so for the 28 triple-session participants the training data bracket the test session in time [V, inference on bracketing]. "no session related information of the testing session could have been learned" [V]. Training contains 199 identities; the probe population is at most the 64 multi-session people (the paper does not state how many of the 64 have a session-2 recording) [V/UNVERIFIED]. Chance level for a 199-way classifier ≈ 0.5 % (my arithmetic; the paper states chance only for activity and attributes).
- Unit of decision: one walk sample (whole traversal). No frame-level or window-level results; no aggregation study [V by absence].

### 1.7 Results — depth (Appendix B, Tables 7 and 8) [V]

Single-session (Table 7), depth, % accuracy (centre-low / centre-high / left / right):
- GaitBase 100.0 / 100.0 / 100.0 / 100.0
- DeepGaitv2 100.0 / 99.9 / 99.9 / 99.9
- GaitPart 100.0 / 99.8 / 99.7 / 99.8
- GaitGL 100.0 / 99.6 / 99.9 / 99.9
- GEINet 98.8 / 96.1 / 98.5 / 97.8
- GaitSet 71.8 / 74.4 / 86.5 / 86.2

Multi-session (Table 8), depth:
- GaitBase 76.7 ± 0.5 / 72.5 ± 0.7 / 65.0 ± 0.4 / 65.3 ± 1.4
- DeepGaitv2 67.1 / 60.0 / 48.9 / 51.5
- GaitGL 67.1 / 60.4 / 47.6 / 57.4
- GaitPart 65.0 / 54.2 / 47.9 / 57.1
- GEINet 48.4 / 45.8 / 43.8 / 41.1
- GaitSet 30.9 / 27.4 / 30.5 / 29.7

Context from the other sensors (best model, Table 8 multi-session): video GaitBase 58.6 / 59.3 / 73.2 / 64.1; LWIR GaitBase 71.7 / 45.5 / 65.2 / 62.6; NIR DeepGaitv2 53.8 / 50.4 / 48.1 / 41.4; lidar LidarGait 74.5 / 77.4 / 74.3 / 78.4; radar ≤ 9.1; CSI ≤ 2.1; BFI ≤ 1.2 [V]. Headline sentences: "We find no accuracy higher than 80%" [V, §5.4]; "video cameras are actually out-performed by depth by often more than 10 percentage-points" [V, §5.4]; "alternative sensors are not necessarily more privacy-friendly by design than video cameras" [V, §5.4].

Caution for the manuscript's C4: MultiGait's video pipeline is also silhouette-only (DeepLabv3 masks), so its video-vs-depth gap is a segmentation-quality comparison, not an appearance-vs-shape comparison. It cannot be cited for or against "RGB's edge is clothing".

Attribute inference from depth silhouettes (Table 9, balanced accuracy, 5 classes for numeric attributes): gender 99.2 / 94.8 / 98.1 / 99.7; height 44.8 / 47.7 / 46.7 / 52.2; weight 45.0 / 37.4 / 35.8 / 38.5; age ≈ chance [V]. Activity (Table 3, identity-disjoint split): depth GaitBase 96.2 / 96.0 / 93.1 / 96.5 [V].

### 1.8 Checklist — does MultiGait report any of the manuscript's items?

| Item | Verdict | Evidence |
|---|---|---|
| Frame-random vs session-disjoint comparison | **NO** for identity. Single-session split is per walk sample within identity (§5.3), never per frame. **PARTIAL** for activity only: Table 3 (identity-disjoint) vs Table 4 (within-identity) shows the within-identity split inflates radar/CSI/BFI activity accuracy (e.g. BFI right 36.3 → 81.5 %) [V]. | §4.5.1, §5.3 |
| Temporal-leakage / adjacency analysis | **NO** | not found |
| Trivial-cue / position floor | **NO** | not found |
| Background-only, person-masked or inpainted ablations | **NO**. Background is removed once (empty-room subtraction) and never studied as a signal | §5.2 |
| Precision / bit-depth reduction | **NO** ("bit", "precision", "quantiz" absent from methods) | grep |
| Silhouette-only input | **YES, but not as an ablation**: every imaging result *is* silhouette-only; there is no raw-depth comparison | §5.2, §5.4 |
| Short observation budget (≤ 3 s) or accuracy-vs-clip-length | **NO**. Decision unit is a whole traversal; length unstated | §4.4 |
| Validity gating of frames | **NO**. Only sample cutting via empty-room similarity, with manual fixes | §4.4 |
| Per-person identifiability distribution | **NO** | not found |
| Cohort-size curve | **PARTIAL**: Table 5 subsamples 10/20/12/6 identities to reproduce mmGaitNet, mID and FreeSense claims (radar/CSI only); they "simply do not scale to the size of MultiGait". Nothing for depth | §5.4, Table 5 |
| Fusion of hand-crafted and learned features | **NO** (multi-modal fusion listed as future work) | §6.3 |
| Anthropometric features as identifiers | **NO**. Height/weight are prediction *targets* (Table 9), and size normalisation removes them from the input | §4.5.2 |
| Threat model for ambient sensing in care settings | **NO**. Framing is smart-city / public-space surveillance; care-related depth work is cited only as an example of "privacy-friendly" claims (Kepski & Kwolek fall detection; Mucha & Kampel) | §1, §4.1 |
| Open-set / verification | **NO** | grep |
| Rank-k / CMC | **NO** (top-1 accuracy only) | §5.3 |

### 1.9 Stated limitations and future work (§6.2, §6.3) [V]
- Some video gait datasets have more subjects; larger multi-sensor multi-session collection "is outside of the scope of academic research".
- Lab environment chosen deliberately: "a best-case scenario for multi-session experiments".
- Anonymisation evaluation "does not require datasets with a large number of individuals" (citing Hanisch et al. 2024).
- Population "limited to able-bodied individuals and biased in terms of ethnicity and age" (18–34); homogeneity may make identification harder, so risk may be underestimated.
- Not every recogniser tested.
- Open question stated in §6: "It is unclear to which extent this is a limitation of current recognition systems" or of the sensed biometrics' stability — exactly the gap a mechanistic study can address.
- Future work: privacy-enhancing technologies / anonymisation for these sensors; representation-based recognisers for radar/CSI/BFI; cross-sensor recognition and multi-modal fusion; informed choice of "(combinations of) sensors, their resolution, their field-of-view" for privacy by design at the sensor.

### 1.10 Datasets compared or cited (Table 1, §3) [V]
Table 1 lists humanID, CASIA B, CASIA C, Tianjin, OU-ISIR (Large, C, LP, LP-Age, MVLP), PAVIS (Barbosa 2012), WOSAG, **TUM-GAID (Hofmann 2014: 305 subjects, 2 sessions, video+depth+audio)**, UFPEL (Andersson & Araujo 2015), mID, mmGait, LW-WiID, FVG, Gait3D, SUSTech1K, CASIA-E, BFId, GREW. GRIDDS (Nunes 2019) is cited in text. **BIWI RGBD-ID is not mentioned; Munaro is not cited; TVRID is not mentioned** [V by grep]. No numerical comparison with any external dataset is made; the dataset is used only for its own benchmark.

### 1.11 Citation check [V by grep of reference list]
- Mucha & Kampel 2022 (ICCHP, "Addressing Privacy Concerns in Depth Sensors"): **YES**, cited as a privacy-friendly-depth proposal (§4.1).
- Haque et al. 2016: **NO**. Wu, Zheng, Lai TIP 2017: **NO** (the only "Wu" is Wei Wu, LidarGait author). Karianakis et al. 2018: **NO**. Munaro / BIWI: **NO**. Sivapalan 2011: **NO**.
- Hanisch et al. PoPETs 2023 ("Understanding Person Identification through Gait") and Hanisch et al. PoPETs 2024 ("A False Sense of Privacy") are cited [V].
- Kepski & Kwolek 2014 (VISAPP, ceiling-mounted depth fall detection), Rougier 2011, Villamizar 2018, Stone & Skubic 2011, Planinc & Kampel 2013 cited as depth-for-privacy proposals [V].

### 1.12 Release and licence
- Paper: CC BY 4.0 [V, abs page].
- Dataset: raw recordings of the 183 consenting participants, all sensors/perspectives/sessions, to be published "with this paper"; access via a signed release agreement (Appendix C): academic research only, no redistribution, no modification, no commercial use, mandatory citation, KIT indemnity; students need a permanent staff co-signature [V]. Whether the data are already downloadable is not verifiable from the text [UNVERIFIED].

### 1.13 Points to note when citing
- Multi-session probes ≤ 64 people; gallery 199 identities; training may include a session recorded *after* the test session (session 3) — state this rather than "199 people cross-session".
- The depth result is a silhouette result (§1.4 above).
- Hyper-parameters were tuned on single-session data and reused for multi-session without retuning (except a check on five combinations) [V, §5.3]; the cross-session figures are therefore not an upper bound.
- Radar/CSI/BFI results near chance cross-session; the authors attribute this to softmax classifiers not disentangling session from identity [V, §5.4] — a leakage-style hypothesis they do not test.

---

## 2. Delécluse, Wannous, Guimas — "Privacy-Preserving Person Re-Identification from Temporal Sequences with Transformer and Hungarian Optimization"

### 2.1 Bibliographic record
- Published: 2025 IEEE 19th International Conference on Automatic Face and Gesture Recognition (FG), Tampa/Clearwater, FL, USA; pages 1–10; date parts [2025, 5, 26]; DOI 10.1109/FG61629.2025.11099215 [V, Crossref]. Author string on Crossref: Raphaël Delecluse, Hazem Wannous, Laurent Guimas.
- Preprint: arXiv:2606.23230v1, 22 Jun 2026, cs.CV, CC BY 4.0; Comments: "Published at 2025 19th International Conference on Automatic Face and Gesture Recognition (FG)" [V, abs page]. The PDF footnote reads "authors' preprint version ... will appear in the FG 2025 proceedings" [V].
- Affiliations: IMT Nord Europe / Univ. Lille / CNRS CRIStAL, and the company Explain (CIFRE fellowship) [V].
- Code: https://github.com/RaphaelDel/PrivacyPreserving-ReID — README refers to a `preprocess` folder that is not present; no BIWI protocol, no frame counts, no hyper-parameters documented [V, repo fetch 9 Sep 2026].

### 2.2 Task framing and protocol (§III-A, §III-D)
- Hypothesis: "each individual captured by the system passes under the camera twice" (entry and exit); each identity therefore has exactly two sequences; query = incoming set, gallery = outgoing set [V].
- Matching: Euclidean distance between L2-normalised sequence embeddings; nearest neighbour gives CMC/mAP; the Hungarian algorithm then solves a one-to-one assignment over the whole distance matrix and "assumes a one-to-one correspondence between the query and gallery sets" [V, §III-D(c)]. "Precision" in their tables is the fraction of correct assignments after Hungarian matching; it exploits a closed bijection (every gallery identity appears exactly once), a constraint unavailable in deployment.
- Closed-set, one probe sequence per identity, whole-sequence decision (mean pooling over all N frame embeddings, §III-B). **Frames per sequence N: not stated anywhere** [UNVERIFIED]. No sampling/stride described.

### 2.3 BIWI specifics
- Their description (§IV-A-3, Table I): BIWI "captures 78 individuals across 106 sequences", "split between training and testing sets", "standing still or walking", frontal view, Kinect 1, "Time of flight" [V]. Errors: BIWI has 50 subjects, of whom 28 were re-recorded on another day in different clothes; 50 Training + 28 Still + 28 Walking = 106 sequences [V, Wu et al. TIP 2017 §V-B: "three groups of sequences “Training”, “Still” and “Walking” captured from 50 different people"; Karianakis 2018: "50 individuals ... 28 of them are re-recorded in a different room with new clothes"]. Kinect v1 is structured light, not time-of-flight [V, Barbosa 2012; Patruno].
- **Which sequences are query/gallery: not stated** (the words "Still", "Walking" as sequence names never appear) [V by grep → UNVERIFIED pairing]. Possible readings: Still↔Walking (same day, same clothes: an R3-type cross-recording test) or Training↔Walking/Still (different day, different clothes: an R4-type cross-session test). The paper calls BIWI "long-term" but never analyses clothing or day change.
- Test-set size: all BIWI CMC and precision values in Tables II and IV are multiples of 1/28 (Table II depth: 17/28 = 60.7, 27/28 = 96.4, 28/28 = 100; RGB-D 27/28 = 96.4; Table IV depth: 7/28 = 25.0, 15/28 = 53.6, 21/28 = 75.0, 14/28 = 50.0; RGB-D 19/28 = 67.9, 26/28 = 92.9, 27/28 = 96.4, 24/28 = 85.7) — consistent with 28 query sequences, i.e. the 28 re-recorded subjects [UNVERIFIED — arithmetic inference, not stated].
- Training data for the in-dataset BIWI result (Table II): not stated; presumably BIWI's own training portion, since Table IV is explicitly "trained only on TVPR2" [UNVERIFIED].
- Single-shot vs multi-shot: one sequence per side, so single-shot at sequence level, multi-frame within the sequence [V from §III-A].

### 2.4 Model and training (§III-B, §III-C, Fig. 1)
- Input: sequence of RoIs (bounding-box crops) of the person, extracted by a "robust blob detection algorithm" on depth [V, §III-A]. RoIs are crops, not silhouettes; whether background pixels inside the box are zeroed is not stated [UNVERIFIED].
- Depth encoder: "a smaller variation of ResNet", not pretrained, single input channel, 128-d embedding per frame. RGB encoder: ResNet-50 ImageNet-pretrained, 128-d. Per-frame embeddings concatenated (256-d for RGB-D) → transformer encoder → mean pooling → one embedding per sequence [V, Fig. 1].
- Loss: batch-hard triplet (Hermans et al.) applied to transformer output, depth encoder output and RGB encoder output; summed (eq. 2) [V]. No epochs, learning rate, batch size or margin given [V by grep → UNVERIFIED hyper-parameters].
- Training data: per-dataset training (Table II) and TVPR2-only training (Table IV) [V].

### 2.5 Results (Tables II–IV) [V]
- Table II (in-dataset), BIWI: RGB-D precision 100.0, Rank-1 96.4, Rank-5 100.0, Rank-10 100.0, mAP 97.4; **Depth precision 100.0, Rank-1 60.7, Rank-5 96.4, Rank-10 100.0, mAP 56.6**. TVPR2 depth Rank-1 88.4; GODPR depth Rank-1 78.6.
- Table III (SOTA, RGB-D only) on BIWI: HRN 47.1, Distillation (Hafner) 40.4, SimMC 41.7, Hi-MPC 47.5, TranSG 68.7, CMOT 77.8, Ours 96.4 Rank-1.
- Table IV (trained on TVPR2 only): BIWI RGB-D precision 85.7 / Rank-1 67.9 / Rank-5 92.8 / Rank-10 96.4 / mAP 70.6; **Depth 50.0 / 25.0 / 53.5 / 75.0 / 22.0**.
- Internal inconsistency: §IV-B-2 text says the depth-only BIWI rank-1 is "60.4%" while Table II says 60.7 % [V].
- Fig. 5 (TVPR2 test subset, depth-only): CMC-1 falls from ~100 % to ~86 % as the test-set size grows, while Hungarian precision stays ~96–100 % [V]. This is a gallery-size dependence observation, within-session, top-view — the only ablation in the paper.
- Acknowledged limits (§IV-B-2): depth-only models "face inherent limitations in differentiating individuals with similar body shapes or movement patterns" and struggle transferring from top-view to frontal BIWI [V].

### 2.6 Privacy sentences (verbatim fragments, with section) [V]
- Abstract: depth "inherently obscures facial and other identifiable features, making it a privacy-preserving solution".
- §I para 2: depth "does not capture identifiable facial features, making it a more privacy-preserving option".
- §I contribution 1: "Depth data inherently preserves privacy by omitting facial or other identifiable features".
- §II-B: depth is "invariant to changes in lighting and clothing" (asserted, never tested).
- §IV-A-1 (TVPR2): "The top-view configuration ensures privacy preservation by avoiding facial details".
- §V Conclusion: depth "inherently obscures facial features and other identifiable characteristics".
- §VI Ethical Impact Statement: depth imaging obscures faces "ensuring that individuals cannot be easily recognized"; "depth data is less prone to revealing detailed identity characteristics"; the study "did not undergo formal ethical review board approval" because it does not involve "the collection of personally identifiable information"; "the research relies solely on anonymized depth data".

### 2.7 What Delécluse does not evaluate [V by absence]
- No cross-session, cross-day or clothing-change analysis on any dataset ("cross-session" absent; "clothing" appears only in invariance claims). TVPR2's eight recording days are mentioned but not exploited.
- No frame-random split anywhere; no per-frame results; no accuracy-vs-sequence-length curve.
- No background/person ablation; no bit-depth; no gating; no per-person analysis; no threat model beyond "surveillance implications" boilerplate.

---

## 3. Comparability of the manuscript's 43–47 % (28 people, 2.5 s, closed-set, sensor mask, Kinect v1)

| Axis | Manuscript | MultiGait depth (multi-session) | Delécluse depth on BIWI |
|---|---|---|---|
| Sensor | Kinect v1 structured light, 640×480 | Azure Kinect ToF, 640×576 | Kinect v1 (mislabelled ToF) |
| Viewpoint | Frontal living-room walk (BIWI) | Hip-height side view + three 2.2 m elevated views, lab walkway | Frontal (BIWI) |
| Input representation | Person-masked depth in mm; metric anthropometry; also CNNs on depth/silhouette variants | Size-normalised binary silhouettes only | Bounding-box RoI crops of depth |
| Decision unit | 2.5 s aggregation (also single frame) | One whole walkway traversal (length unstated) | One whole sequence, mean-pooled (frames unstated) |
| Identities in gallery | 28 (chance 3.6 %) | 199-way classifier (chance ≈ 0.5 %) | 28 (chance 3.6 %) |
| Probe population | 28 re-recorded on a different day, new clothes | ≤ 64 with ≥ 2 sessions; session 2 probes | 28 (inferred) |
| Training per identity | One ~300-frame recording from another day | 120 samples/session from sessions 1 (+3), bracketing the test session | Unstated |
| Split regime | R4 cross-session (day, clothes, room) | Session-disjoint (Nov/Mar → Feb) | Unknown (Still↔Walking or day-crossing) |
| Metric | Rank-1, top-3 | Top-1 accuracy | Rank-1/5/10, mAP, Hungarian precision |
| Headline | 43–47 % (top-3 81 %); single frame 19 % | 65.0–76.7 % (GaitBase), 27–67 % others | 60.7 % rank-1 (17/28); 100 % Hungarian |

What CAN be said:
- All three show depth-derived data identifying people far above chance under conditions the sensor was supposed to defeat; MultiGait does so at scale, with modern gait models, across a 1–3 month gap; ours does so with 2.5 s, one training recording and hand-crafted metric features.
- MultiGait's within→cross-session collapse (100 → 65–77 % for the best model, 100 → 27–31 % for GaitSet) is qualitatively the same phenomenon as the manuscript's ladder; MultiGait observes it, the manuscript dissects it.
- MultiGait's statement that once silhouettes exist "the actual sensing technology has little impact" supports C2's "the outline leaks" and "best cross-session input = framing-normalised binary silhouette".
- Delécluse's 17/28 rank-1 on BIWI is on the same 28-person cohort as ours, so it is the closest external number — but only as evidence that the premise is contradicted by its own proponents.

What CANNOT be said:
- Do not rank 43–47 % against 65–77 % or 60.7 %: different chance levels (3.6 % vs 0.5 %), decision lengths (2.5 s vs whole clips), training volumes (one recording vs 120–240 samples), input representations and sensors. "Conservative" is defensible as a qualitative remark only if these differences are named.
- Do not say MultiGait measured depth cross-session "on 199 people": probes ≤ 64, gallery 199.
- Do not say MultiGait tested depth *values*; it tested depth-derived silhouettes.
- Do not place Delécluse's 60.7 % on the manuscript's ladder (R3 vs R4 unknown), and do not equate its 100 % Hungarian precision with identification accuracy.
- Do not cite MultiGait's depth > video result as support for C4 ("RGB's edge is clothing"): both streams are silhouettes there.

---

## 4. Positioning text (ready to adapt)

MultiGait as the scale result, the manuscript as the mechanistic complement:
1. MultiGait (Todt et al., arXiv:2609.01036) supplies the scale result the field lacked: on 199 people, eight sensors, four viewpoints and three sessions, silhouette-based gait recognisers reach ≥ 98 % within-session and 65–77 % across a one-to-three-month gap from Azure Kinect depth alone, refuting the assumption that depth is privacy-friendly by construction.
2. MultiGait reports these numbers but does not dissect them: it evaluates a single input representation (size-normalised silhouettes), a single split per regime, whole-walk clips and no attribution, precision or observation-budget ablations, and its authors leave open whether the cross-session drop reflects the sensors or the recognisers.
3. This paper is the mechanistic and evaluation-methodology complement: on the 28-person cross-day BIWI cohort we show how the split protocol alone manufactures near-ceiling within-session accuracy, which cue carries identity (outline versus room versus 3-D shape versus bit-depth), and what a validity-gated 2.5-s observation actually yields (43–47 % rank-1, 81 % top-3) together with its per-person and cohort-size dependence.

Delécluse as the freshest instance of the rebutted premise:
1. Delécluse, Wannous and Guimas (IEEE FG 2025; arXiv:2606.23230, June 2026) re-identify 17 of 28 BIWI subjects at rank-1 — and all 28 under one-to-one Hungarian assignment — from depth alone while asserting that depth "inherently preserves privacy" and waiving ethics review because the data are "anonymized".
2. Their evaluation is exactly what our protocol ladder is designed to expose: unspecified gallery/probe pairing, unspecified frames per decision, a bijection constraint unavailable in deployment, and no cross-session, clothing or leakage analysis.

---

## 5. Does anything in these two papers pre-empt the repositioned framing?

- Protocol ladder / frame-random leakage: **No.** MultiGait's only leakage-type observation is for activity (Tables 3 vs 4). Delécluse has no split analysis.
- Attribution (room / person / outline / bit-depth): **No.** MultiGait removes background once and never studies it; Delécluse crops RoIs and never ablates.
- Operating point under a budget with gating, top-k, per-person distribution: **No** in either.
- Cohort-size dependence: **Partial.** Delécluse Fig. 5 (TVPR2, within-session, top-view) shows CMC-1 falling with test-set size; MultiGait Table 5 subsamples identities for radar/CSI. Neither does it for depth cross-session with draw spread. Cite Delécluse Fig. 5 as a precedent and differentiate.
- Fusion negative results: **No**; MultiGait lists fusion as future work.
- Methodology precedent to cite: Hanisch, Todt, Patino, Evans, Strufe, PoPETs 2024(1), DOI 10.56553/popets-2024-0008 ("A False Sense of Privacy ..."), cited by MultiGait as the evaluation-methodology reference for anonymisation [V, reference list].

## 6. Corrections to the handoff summary of MultiGait
- "depth > 98 % within-session → 60–80 % cross-session": use 100 % (GaitBase, all perspectives) → 65.0–76.7 % (GaitBase), 27–67 % for other models.
- "on 199 people": gallery of 199 identities; cross-session probes ≤ 64 people; training includes session 3 for the 28 triple-session participants.
- "depth" = depth-derived, size-normalised silhouettes.
- Delécluse: FG 2025 (May 2025) publication; arXiv posting June 2026.

## 7. Unverified items
- MultiGait walkway length and walk-sample duration; silhouette resolution; how many of the 64 multi-session participants have a session-2 recording; target venue; whether the dataset is already downloadable.
- Delécluse: which BIWI sequence pair forms query/gallery; frames per sequence; training set for Table II BIWI; hyper-parameters; whether RoI backgrounds are zeroed; the 28-query inference (arithmetic only).
- Live resolution of DOI 10.1109/FG61629.2025.11099215 was confirmed through Crossref metadata, not by opening IEEE Xplore.
