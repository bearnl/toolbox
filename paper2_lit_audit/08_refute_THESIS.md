# 08 — ADVERSARIAL REFUTATION of the thesis-level novelty claims (COMPLETE, 2026-09-09)

Role: hostile reviewer. For each of the eight "first / to our knowledge" claims, find a prior work that already did the same measurement (any modality first, then depth), or the nearest adjacent work that forces a qualifier. Judged against the REPOSITIONED framing (manuscript = mechanistic and evaluation-methodology complement to MultiGait).

Method (single agent, no sub-agents, ~40 tool calls): read `briefs/00_shared_context.md`, `README_HANDOFF.md` and the completed audits 01, 02, 03, 04, 04b, 05 (precedents), 06, 07; grepped the local extracts (`tumgaid_preprint.txt`, `mucha_icchp.txt`, `hanisch.txt`, `barbosa.txt`, `andersson.txt`, `karianakis.txt`, `haque2016.txt`, `wu2017.txt`, `luo18a.txt`, `haque17a.txt`, the Li et al. TPAMI text, and the regulator extracts `aiact.txt`, `edpb2019.txt`, `wp193.txt`, `cnil2022.txt`, `cnil2020.html`, `jo2021.txt`, `ep2021.txt`, `hhs_deid*.txt`); then arXiv API (5 queries incl. id_list), Crossref (7 queries), Europe PMC (3 queries). WebSearch not used. One fetch failed (cnil.fr, HTTP 403). No DOI or number below is typed from memory; each is from a record, a page, or an extract.

Legend: **[V]** = citation and the cited content confirmed on a DOI/arXiv/venue record or read in an extract; **[V-rec]** = bibliographic record confirmed, cited content NOT read; **[UNVERIFIED]** = neither.

Quotes are under 15 words.

---

## 0. Summary table

| # | Claim (short) | Verdict | Decisive prior work | Key qualifier |
|---|---|---|---|---|
| 1 | First mechanistic decomposition (ladder R0–R4, 13-scalar floor, guard sweep at matched n_train, 3 corpora) | **WEAKENED** | Hofmann et al. 2014 TUM-GAID Table 6 (depth same-session 94–99 % → elapsed-time 28–50 %) [V]; MultiGait 2026 [V]; Miller et al. 2024 (train–test delay and duration varied, VR) [V]; Patruno 2019 (BIWI frame-level 97.84 %) [V]; Li 2021 / Saeb 2017 / Hammerla 2015 / SeaTurtleID 2024 / Young 2025 / Woo 2026 / ECG-bench 2026 [V] | The two endpoints for depth were published in 2014 and 2026; the within-recording guard sweep at matched n_train and the metadata floor are unprecedented |
| 2 | First depth-specific test of precision reduction and person-masking as privacy controls | **WEAKENED** | Mucha & Kampel ICCHP 2022 (depth face down-scaling vs FR: fine at 40 % scale, collapses <10 %) [V]; Liu et al. Sensors 2025 (depth spatial resolution 640×480 → 16×12, ID 99.5 → 96.8 %) [V]; MultiGait (the 1-bit endpoint — depth silhouettes — 100 → 65–77 % cross-session) [V]; Nair 2024 (precision, VR) [V]; Xiao 2021 (hole leak, RGB) [V] | Depth *resolution* reduction has been tested against identification twice; the Z-*precision* (bit-depth) axis and the exact-complement/hole masking in depth have not |
| 3 | Deployment-facing operating point (2.5 s, validity gate, top-k, per-person 0–100 %, cohort K=4→28) | **WEAKENED** | Andersson & Araujo AAAI 2015 (accuracy vs gallery size, 10 random galleries per size, 95 % CIs, Kinect) [V]; Miller et al. 2024 (training duration and delay) [V]; Nair 2023 (10 s vs 100 s) [V]; GaitSet 7-frame / LidarGait 1-frame ablations [V]; Barbosa 2012 / Munaro 2014 / Karianakis 2018 gates (un-ablated) [V]; Doddington 1998 / Yager–Dunstone 2010 [V]; Friedman 2022 [V] | Each component has an analogue; none is quantified for cross-session depth ID, and no one has quantified FOV clipping or the gate's effect |
| 4 | Metric anthropometry in mm from depth as the identifying mechanism | **REFUTED** (as a mechanism discovery) | Barbosa et al. 2012 (height/floor–head/geodesic distances from Kinect depth, "different days", height most informative) [V]; Munaro 2014 (limb lengths, cross-day BIWI) [V-sec]; Andersson 2015 (20 anthropometric attributes, 140 people, 85 %) [V]; Dubois & Bresciani 2015 (height + gait from depth at home) [V]; Cătrună 2024 (height shortcut in gait) [V] | What survives: anthropometry from *unprojected depth pixels under the sensor mask, without a skeleton SDK*, with a measured drift model, compared with the manuscript's own CNNs |
| 5 | Pre-registered negative results (fusion, normals, rescaling, multi-frame, gait untestable, room confound) | **WEAKENED** | Haque 2016 evaluated gait (GEI 21.4 %, GEV 25.7 %) on BIWI [V]; Hofmann 2014 and Castro 2020 report positive depth+RGB fusion on TUM-GAID [V]; Wu 2017 positive shape+skeleton fusion [V]; Karianakis 2018 uses range normalisation as a step [V] | "Gait untestable on BIWI" contradicts published BIWI gait numbers unless scoped to the 2.5-s / clipped-frame regime; "fusion fails" must be reported as a discrepancy with prior positive fusion results |
| 6 | Legal reading: biometric status attaches to processing, not sensor; instruments assembled | **WEAKENED** | Jasserand EDPL 2016 [V-rec]; Kindt CLSR 2018 [V-rec]; Bygrave & Tosoni, OUP GDPR Commentary Art. 4(14) 2020 [V-rec]; Mucha & Kampel 2022 §2.3 legal aspects of depth FR [V]; Gerke, Yeung, Cohen JAMA 2020 [V-rec]; Martinez-Martin et al. Lancet Digit. Health 2021 [V-rec]; MultiGait "same legal framework" [V] | The processing-based definition is settled scholarship; the *application* to depth body sensing with the named instruments is not found |
| 7 | Premise-in-the-wild catalogue (16 systems) with internal contradictions | **WEAKENED** | Mucha & Kampel 2022 §2 already catalogues contradictory depth-privacy publications ("more such contradictory examples") [V]; MultiGait §4.1 lists depth-for-privacy proposals [V]; Baldassini 2025, Amin 2026, Momin 2022, Kröger 2019 (field-level versions) [V/V-cite] | The 16-system verified catalogue and the *same-institution* contradictions (Stanford, Missouri) are new |
| 8 | No regulator text examined mentions depth, 3D, thermal or LiDAR sensing | **REFUTED** (as worded) | The extracts themselves: CNIL 2022 footnote 11 cites CNIL's own 2020 "caméras thermiques" guidance and line 620 lists "capteurs infrarouges" [V]; WP193 mentions near-infrared vein cameras and "3D imaging" anti-spoofing [V]; EP 2021 mentions a "facial and thermal analysis tool" and cites 3D face recognition [V] | The substantive point survives: none addresses depth/3D *body* sensing as a modality, and none declares any sensor non-identifying |

Nothing found pre-empts the repositioned framing as a whole. Three items force wording changes beyond the completed audits: Hofmann 2014 Table 6 (claim 1), Mucha 2022's resolution test and catalogue (claims 2, 7), and the regulator-extract hits (claim 8).

---

## 1. Claim 1 — "First mechanistic decomposition of why depth identification looks near-perfect"

### Verdict: WEAKENED

### Refuting / adjacent works

**1a. Hofmann, Geiger, Bachmann, Schuller, Rigoll, "The TUM Gait from Audio, Image and Depth (GAID) database: Multimodal recognition of subjects and traits", J. Vis. Commun. Image Represent. 25(1), 2014, DOI 10.1016/j.jvcir.2013.02.006 [V DOI via audit 02; Table 6 read in `extracts/tumgaid_preprint.txt`].**
What they did: 305 subjects; 32 re-recorded ~3 months later "wearing significantly different clothes". Gallery = recordings N1–N4 (session 1); probes N5–N6 (same session, different recordings = the manuscript's R3), and TN5–TN6 / TB / TS (second session = R4). Same 155-class gallery, chance 0.6 %. Depth features on the *same corpus*: depth-GEI 96.8 % (N) → 28 % (TN), 0 % (TB), 22 % (TS); GEV 94.2 → 41 / 0 / 31; DGHEI 99.0 → 50 / 0 / 44; RGB GEI 99.4 → 44 / 6 / 9. Note: TN/TB/TS are evaluated on 16 people, N on 155.
How it matches: a depth paper reporting the within-session and cross-session numbers **on the same data** in 2014 — audit 01's statement that "only MultiGait" does so is wrong, and the manuscript cannot say the depth field "never saw" the within/across gap. The gap (96.8 → 28) is as large as the manuscript's R3 → R4 on BIWI.
How it does not match: no frame-random rung, no block/guard sweep, no matched n_train, no trivial-cue floor, no attribution; the authors attribute the drop to appearance change, not to the split.

**1b. Todt, Morsbach, Dissert, Strufe, MultiGait, arXiv:2609.01036, 1 Sep 2026 [V extract].** Depth (as silhouettes) 100 % single-session → 65–77 % multi-session (GaitBase). Single-session split is per walk sample within identity (≈R3), not per frame. No dissection (audit 07 §1.8).

**1c. Miller, Nair, Han, DeVeaux, Rack, Wang, Huang, Latoschik, O'Brien, Bailenson, "Effect of Duration and Delay on the Identifiability of VR Motion", arXiv:2407.18380, 25 Jul 2024 (SePAR@WoWMoM 2024 per audit 04b) [V arXiv abstract].** They vary "training data duration and train-test delay" and find "minimal train-test delay leads to very high accuracy" and that "train-test delay should be controlled in future experiments". This is a *swept temporal gap between training and test* for person identification — at session scale (weeks), in VR telemetry, without matched n_train. It is the closest published sweep to the manuscript's guard band and must be cited; the manuscript's contribution is the *within-recording* sweep (seconds) at matched n_train.

**1d. Patruno et al., Pattern Recognition 89, 2019, DOI 10.1016/j.patcog.2019.01.003 [V via audit 03].** 97.84 % on BIWI from 10-fold CV over 11,780 frame instances of session-1 videos only. A *published* R0-type BIWI number: the manuscript's retracted 0.99 is not the only one, so "the field never saw its own inflated number" is false; say instead that the R0 number exists in the literature (Patruno 2019) but was never placed beside the cross-day number on the same pipeline.

**1e. The frame-level/within-subject leakage genre (all [V] in audits 01/05):** Hammerla & Plötz UbiComp 2015 (adjacent samples not independent; meta-segmented CV); Saeb et al. GigaScience 2017 (record-wise vs subject-wise); Li et al. TPAMI 2021 (block design; "close temporal proximity" of trials — read in extract line 757; arbitrary-label relabelling test); Georgiev et al. ASIA CCS 2022 (random vs contiguous, 3.8 pp EER); Adam et al. SeaTurtleID2022 WACV 2024 (random split "performance overestimation"; per-part 18–54 pp per audit 01); Young et al. Diagnostics 2025 (95–99 % vs 66–90 % subject-wise); Woo arXiv 2605.06359 May 2026 (frame → temporal → scene split hierarchy); Parvan, ECG-biometrics-bench, arXiv 2605.01548 May 2026 ("Random Split Fallacy"; intra-session protocols "artificially inflate performance") [V arXiv abstract, re-confirmed this pass].

**1f. Not found (searched this pass):** any paper inserting a swept temporal guard band *within a recording* at matched training size for identity (arXiv query "frame-level OR temporal leakage OR session-disjoint OR cross-session" × "person identification OR re-identification OR biometric" × "leakage OR inflat OR overestimat OR pitfall", 20 most recent: only ECG-biometrics-bench flagged); any bounding-box-metadata-only identity floor (audit 01 §0b, re-checked against Cătrună 2024 and Xiao 2021's bbox-size leak of ~23 % for objects).

### What can still be claimed
The graded ladder with a **within-recording temporal guard sweep at matched n_train**, the **13-scalar bounding-box floor (89 % under frame-random)**, the **cross-recording rung distinct from the cross-session rung** on the same pipeline, and replication on three corpora/three depth technologies. Not: the existence of the within/across gap for depth (Hofmann 2014; MultiGait 2026), the existence of frame-level leakage (1e), or a swept train–test gap per se (Miller 2024).

### Safe wording
"Within-session and cross-session depth identification numbers on one corpus exist since TUM-GAID (Hofmann et al., 2014: 97–99 % → 28–50 %) and at scale in MultiGait (2026: 100 % → 65–77 %), and a swept train–test delay has been reported for VR telemetry (Miller et al., 2024). What has not been done is to resolve the within-session number itself: we insert a temporal guard band inside the recording at matched training size, measure a 13-scalar bounding-box floor that alone reaches 89 % under a frame-random split, separate the cross-recording from the cross-session rung, and replicate on three corpora and three depth technologies — to our knowledge the first such decomposition for person identification in any modality."

---

## 2. Claim 2 — "First depth-specific test of two would-be privacy controls (precision reduction; person-masking)"

### Verdict: WEAKENED

### Refuting / adjacent works

**2a. Mucha & Kampel, "Addressing Privacy Concerns in Depth Sensors", ICCHP-AAATE 2022, LNCS, DOI 10.1007/978-3-031-08645-8_62 [V full text `mucha_icchp.txt`].** To "find the border conditions of correct identification", faces from HRRFaceD are scaled down and depth FR re-run: original 118×153 px; at 40 % (47×61 px) FR still works; "worst FR results are seen below 10% (11x14 pixels)". Their Table 1 also tabulates depth FR accuracy against sensor *precision in mm* across datasets (BU-3DFE 0.2 mm precision 99.3 %, etc.). Conclusion: identity disclosure possible when "less than 100" individuals and "high precision sensors".
How it matches: a **depth-specific** test of a resolution-reduction control against identification, with sensor precision explicitly named as a factor. This is exactly the genre the claim says does not exist.
How it does not match: face scale, near range, *spatial* resolution not Z-precision, cross-dataset precision comparison rather than a controlled bit-depth sweep, no body-scale, no cross-session, no masking.

**2b. Liu, Bouazizi, Xing, Ohtsuki, Sensors 25(1):271, 2025, DOI 10.3390/s25010271 [V via audit 02 extract `liu2025.xml`].** RealSense L515 depth down to 16×12: ResNet34 identification 99.54 → 96.78 % (6 subjects, predefined paths, no background control). A depth spatial-resolution axis with an identification read-out — degradation does not remove identity.

**2c. MultiGait [V extract].** Every "depth" result is a size-normalised **binary silhouette** result (audit 07 §1.4), i.e. the manuscript's 1-bit limit, tested cross-session on ≤64 probes / 199 gallery identities at 65–77 %. The 1-bit endpoint of the manuscript's axis is therefore already published at scale, and MultiGait's remark that "the actual sensing technology has little impact" once silhouettes exist is the same conclusion as "the outline leaks". The silhouette-gait literature (CASIA-B, OU-MVLP, GaitSet/GaitBase) is the general prior that silhouettes identify.

**2d. Nair, Miller, Wang, Huang, Rack, Latoschik, O'Brien, "Effect of Data Degradation on Motion Re-Identification", arXiv:2407.18378, 2024 [V arXiv abstract].** "added noise, reduced framerate, reduced precision, and reduced dimensionality"; attacks "still achieve near-perfect accuracy for each of these degradations". Precision reduction as a failed privacy control — VR telemetry.

**2e. Xiao, Engstrom, Ilyas, Mądry, ICLR 2021 [V via audit 02]:** No-FG (hole kept) beats Only-BG-B by ~13 pp because "the hole keeps the object shape"; Tian et al. CVPR 2018 mean-filled person hole; HAT NeurIPS 2022 video-inpaints the human out. Person-masking-with-hole leaks shape — RGB objects/actions.

**2f. Not found (searched this pass):** any work that quantises depth *values* (bit depth / Z precision at fixed range) and measures identification (arXiv query "depth × (quantization OR bit depth OR precision) × privacy × (identification OR re-identification OR anonymization)": only Delécluse 2026 and noise); any depth person-ID work reporting an inpainted exact-complement or a hole condition (consistent with audits 02, 04, 06).

### What can still be claimed
The **Z-precision axis 16 → 1 bit at fixed metric range** in depth (2-bit ≈ 16-bit); **person-masking with the recording's own plate (exact complement) and with a hole** in depth; both reported along the leakage ladder. Not: "no depth identity-leakage evaluation of a degradation control exists" (Mucha 2022; Liu 2025), nor that silhouettes identify across sessions (MultiGait; gait literature).

### Safe wording
"Two degradation controls have been tested against identification in depth before, both along the *spatial-resolution* axis: Mucha and Kampel (2022) found depth face recognition survives down-scaling to 40 % and fails below 10 %, and Liu et al. (2025) found 16×12 depth still identifies six subjects; MultiGait (2026) shows the 1-bit limit — the silhouette — identifies across sessions at scale. We add the two controls that practitioners actually reach for and that have not been tested in depth: reducing the *value precision* of each pixel (16 → 1 bit at fixed range; 2-bit matches 16-bit) and masking the person out (pixel-exact complement from the recording's own plate, or a hole). Neither is a privacy control."

---

## 3. Claim 3 — "Deployment-facing operating point: 2.5-s budget, validity gate, top-k, per-person identifiability, cohort curve"

### Verdict: WEAKENED

### Refuting / adjacent works (component by component)

**Cohort curve with draw spread.** Andersson & Araujo, AAAI 2015, DOI 10.1609/aaai.v29i1.9212 [V extract `andersson.txt`]: Kinect skeleton anthropometrics + gait, 140 subjects; Figures 4–5 plot accuracy vs gallery size where each point "is the average over 10 galleries with the same size and randomly drawn", with 95 % confidence-interval error bars; ~98 % for very small galleries declining to ~87 % at 140. Same measurement (random sub-galleries of increasing K with spread), single session, skeleton not depth pixels. Also Delécluse et al. FG 2025 Fig. 5 (CMC-1 vs test-set size, within-session, depth) [V via audit 07]; Friedman et al. 2022 log-linear law [V via audit 03]; MultiGait Table 5 identity subsets for radar/CSI only [V].

**Observation budget / duration.** Miller et al. 2024 [V, §1c above] vary *training* duration and delay; Nair et al. USENIX Sec 2023 [V via 04b]: 94.33 % from 100 s vs 73.20 % from 10 s of VR motion; GaitSet (arXiv 1811.06186) "82.5 % on CASIA-B with only 7 frames" and LidarGait 1-frame → n-frame ablation [V arXiv / audit 03]; MilliTRACE-IR "95 % in <20 s" (radar) [V-cite via 04b]. No BIWI/depth work fixes a seconds budget with a gate.

**Validity gate.** Barbosa 2012 chose one frame per acquisition "not cropped by the sensors fields of view" [V extract]; Karianakis 2018 removed frames with a person "heavily occluded from the image boundaries or too far from the sensor" [V extract line 473–474]; Munaro 2014 selected by face detection [V-sec]. Gates applied, never ablated or quantified (74 % clipping, 89.5 mm/m drift, 28 vs 9 % are new).

**Per-person identifiability 0–100 %.** Doddington et al. 1998; Yager & Dunstone TPAMI 2010 (biometric menagerie) [V via audit 03]. Crossref query "biometric menagerie gait" this pass returned only keystroke (Migdal 2021), fusion (Ross 2009) and encyclopaedia entries — **no gait/body-shape menagerie study found** (Crossref only; S2/dblp unavailable) [absence UNVERIFIED].

**Top-k.** Standard CMC reporting (Wu 2017, Haque 2016 report nAUC; Delécluse rank-5/10) [V].

**The EP 2021 "less unique / less stable" and Mucha "<100 individuals" hooks** are the manuscript's own framing; Mucha's residual condition [V extract lines 292–296] has not been tested by anyone (04 §B2 confirms no 2023–2026 follow-up).

### What can still be claimed
The first **quantification of the FOV-clipping mechanism and of the gate's effect** in depth ID; a **fixed 2.5-s probe budget with a gate, top-k and per-person distribution, cross-session, in depth**; the cohort curve **across sessions with draw spread exceeding the K effect**. Not: cohort-size curves with random draws (Andersson 2015), duration/delay dependence (Miller 2024; Nair 2023), frame-count ablations (GaitSet; LidarGait), the menagerie concept, or frame gating as such.

### Safe wording
"Gallery-size curves with random re-draws (Andersson and Araujo, 2015), observation-length and train–test-delay dependence (Nair et al., 2023; Miller et al., 2024) and frame-quality gates (Barbosa et al., 2012; Karianakis et al., 2018) all exist; none has been measured for cross-session depth identification, and no one has quantified how often a walking body leaves the field of view (74 % of frames) or what that does to height-based features (89.5 mm per metre, against a 105 mm between-subject SD). We report the operating point an adversary with a 2.5-s clip actually obtains — rank-1, top-3, per-person identifiability from 0 to 100 %, and its dependence on who, not only how many, is enrolled — which is the empirical content of the regulators' 'less unique, less stable' category and the direct test of Mucha and Kampel's residual condition."

---

## 4. Claim 4 — "Metric anthropometry in mm from unprojected depth as the identifying mechanism"

### Verdict: REFUTED (as a mechanism discovery); the specific formulation survives with qualifiers

### Refuting works

**4a. Barbosa, Cristani, Del Bue, Bazzani, Murino, "Re-identification with RGB-D sensors", ECCV Workshops 2012, DOI 10.1007/978-3-642-33863-2_43 [V extract `barbosa.txt`].** 79 people "captured in different days and with different clothing"; ten soft biometrics from Kinect depth: skeleton-based Euclidean distances (d1 floor–head, d3 height estimate, d4 floor–neck, …) and surface-based geodesic distances on the depth point cloud (d8–d10); features from height "were the most reliable"; nAUC ≈ 88–92 % cross-day (audit 03). Same axis (metric body distances from consumer depth), same control (different day, different clothes), same conclusion (height-type measures carry identity).

**4b. Munaro, Fossati, Basso, Menegatti, Van Gool, 2014, DOI 10.1007/978-1-4471-6296-4_8 [V DOI; numbers V-sec via Haque/Karianakis tables].** Skeleton limb lengths and ratios on BIWI cross-day: 21.1–26.6 % single-shot, 39.3 % multi-frame. Metric anthropometry as a cross-session BIWI identifier, 2014.

**4c. Andersson & Araujo, AAAI 2015 [V extract].** 20 anthropometric + gait attributes from Kinect skeleton; anthropometric alone 84.7–85.4 % on 140 people vs gait alone 59–63 % — anthropometry, not motion, is the dominant cue.

**4d. Dubois & Bresciani, EMBC 2015, DOI 10.1109/embc.2015.7319514, PMID 26737414 [V Europe PMC abstract].** Identifies 10 known + 2 unknown people at home from "height and gait pattern" modelled from depth sequences (HMMs). Height from depth as an identifier in the AAL setting.

**4e. Cătrună, Cosma, Rădoi, arXiv:2402.08320, 2024 [V arXiv abstract].** Skeleton gait models rely on "implicit anthropometric information"; removing height causes "notable performance degradation"; a temporal-free spatial model is "unreasonably good". Zhao, IEEE SSCI 2021 [V via 04b]: >90 % from a single skeleton frame.

**4f. Uniqueness linkage.** Godil, Grother, Ressler 2003 [V] and Lucas & Henneberg 2016 [V] are cited by nobody in the depth-ID line as far as the audits found; the *linkage* is new but is a discussion point, not a measurement. Adjeroh et al. WIFS 2010 (predictability of body measurements) [V DOI].

### What can still be claimed
Anthropometry recovered **directly from unprojected depth pixels under the sensor mask, without a skeleton SDK**; the **drift/validity model** (clipping, mm/m); the **comparison against the manuscript's own CNNs on the same frames and protocol** (19 % single-frame cross-session vs 7 %); the **explicit link to the anthropometric-uniqueness literature** in a privacy argument. Not: that metric anthropometry from consumer depth identifies people across days (Barbosa 2012; Munaro 2014), that it dominates gait (Andersson 2015; Cătrună 2024), or that it "beats every CNN" (Haque 2016 3D RAM 30.1 % and Karianakis 2018 25.4 % single-shot exceed 19 %; Wu 2017 hand-crafted 24–30 % likewise — audit 03).

### Safe wording
"That body measurements recovered from a consumer depth camera identify people across days is a 2012–2015 result (Barbosa et al.; Munaro et al.; Andersson and Araujo; Dubois and Bresciani), and skeleton-gait models are now known to lean on height (Cătrună et al., 2024). We recover the measurements in millimetres from the unprojected depth pixels themselves, with no skeleton tracker, show that this hand-crafted vector outperforms the CNNs we train on the same frames and protocol (19 % vs 7 % single-frame, cross-session), quantify why it degrades (field-of-view clipping; 89.5 mm/m drift), and connect it to the forensic uniqueness literature (Godil et al., 2003; Lucas and Henneberg, 2016) to explain why a sensor that discards colour and texture still discloses identity."

---

## 5. Claim 5 — "Pre-registered negative results"

### Verdict: WEAKENED

### Refuting / adjacent works

**Gait "untestable on BIWI".** Haque, Alahi, Fei-Fei CVPR 2016 Table 3 [V extract lines 402–403]: Gait Energy Image 21.4 % and Gait Energy Volume 25.7 % rank-1 multi-shot on BIWI Training → Walking (chance 2 %); Sivapalan et al. IJCB 2011 GEV [V DOI via audit 03]; Munaro's "Walking" set is a gait set by design. Gait features have therefore been tested on BIWI and score above chance. "Untestable" is false as an unqualified statement; it can only mean "within our 2.5-s, 74 %-clipped-frame, single-view regime the gait cycle cannot be segmented".

**Fusion.** Hofmann 2014 Table 6 [V extract]: score-level fusion DGHEI + GEI raises TN 50 → 66 %, B 40.3 → 51.3 %, average 74.1 → 77.9 %; audio + DGHEI gain "significant at a 0.002 level". Castro et al. 2020 [V via audit 02]: multimodal CNN fusion helps on TUM-GAID. Wu 2017 [V extract]: ED + SKL > ED alone (30.52 vs 28.98 % Still). So depth-with-other-modality fusion has published *positive* results; the manuscript's null must be presented as a discrepancy with an explanation (cross-session single-frame regime; the 8-pp oracle gap; 28-way cohort).

**Range normalisation / multi-frame input.** Karianakis 2018 [V extract lines 182–205]: fixed-range normalisation to [1, 256] with an offset after background subtraction is a *design step* (not ablated); multi-shot via late fusion (avg / soft attention / RTA: 45.7 / 46.4 / 50.0 %) rather than early frame stacking. A null on early multi-frame fusion is consistent with, not contradicted by, that literature.

**Normals.** Not searched further (no person-ID work using surface normals located in the audits); [UNVERIFIED absence].

**Pre-registration.** Precedent formats: NeurIPS pre-registration workshops PMLR 148/181 [V via audit 05]; no pre-registered depth-biometrics study located [UNVERIFIED absence].

### Safe wording
"Gait energy features do identify BIWI subjects across days when whole sequences are available (Haque et al., 2016: 21–26 %), and depth–RGB fusion helps on TUM-GAID (Hofmann et al., 2014; Castro et al., 2020). Under our pre-registered protocol — single 2.5-s probes in which 74 % of frames clip the body — gait cycles could not be segmented, and score-, feature- and auxiliary-level fusion failed to realise an 8-pp oracle complementarity; surface normals, dynamic-range rescaling and early multi-frame stacking were null. We report these as pre-registered negative results with their falsifiers and minimum detectable effects, and note where they diverge from whole-sequence results."

---

## 6. Claim 6 — "Legal reading: biometric status attaches to processing and identifying capability, not the sensor"

### Verdict: WEAKENED

### Refuting / adjacent works

**6a. Legal scholarship on the definition.** Jasserand, "Legal Nature of Biometric Data: From 'Generic' Personal Data to Sensitive Data", EDPL 2(3):297–311, 2016, DOI 10.21552/edpl/2016/3/6 [V-rec Crossref]; Kindt, "Having yes, using no? About the new legal regime for biometric data", CLSR 34(3):523–538, 2018, DOI 10.1016/j.clsr.2017.11.004 [V-rec]; Bygrave & Tosoni, "Article 4(14). Biometric data", in *The EU GDPR: A Commentary* (OUP 2020), DOI 10.1093/oso/9780198826491.003.0020 [V-rec]; Kindt 2013 monograph chapters DOI 10.1007/978-94-007-7522-0_8 and _9 [V-rec]. These are the standard analyses of Art. 4(14)'s "specific technical processing" wording — the processing-based reading is settled doctrine, not a finding of this manuscript. (Content not read here; the Art. 4(14) wording itself is [V] via EDPB 3/2019 §74 extract.)

**6b. Depth-specific legal discussion exists.** Mucha & Kampel ICCHP 2022 §2.3 "Legal aspects of privacy and FR" (ECHR Art. 8; GDPR) and conclusion: publications "are not consistent with the privacy of depth images which is problematic with respect to legal regulations" [V extract] — for face recognition in depth. Gerke, Yeung, Cohen, "Ethical and Legal Aspects of Ambient Intelligence in Hospitals", JAMA 323(7):601, 2020, DOI 10.1001/jama.2019.21699 [V-rec; Viewpoint, content unread]; Martinez-Martin, Luo, Kaushal, Adeli, Haque, …, Fei-Fei, Schulman, Milstein, "Ethical issues in using ambient intelligence in health-care settings", Lancet Digit. Health 2021, DOI 10.1016/s2589-7500(20)30275-2, PMID 33358138 [V-rec; abstract lists "privacy, data management, bias and fairness, and informed consent"; depth/HIPAA content unread]. Both are by the Stanford group whose depth systems are in the premise catalogue; a reviewer will expect them cited and their treatment of identifiability checked.

**6c. MultiGait** draws the "same legal framework as any other personal, biometric data" conclusion [V via audit 04]; Hanisch 2023 PoPETs mentions legal access constraints [V extract line 664].

**6d. Instruments.** EDPB 3/2019 §74/§76, Recital 51, AI Act Recital 15 ("body shape … gait, posture"), JO 5/2021 §32, CNIL 2022 reasonable-means test, WP193, EP 2021, BIPA closed list, Zellmer v. Meta, HIPAA Safe Harbor (Q)/(R) — all [V] in audit 04 extracts. No prior work assembling these for depth body sensing was found (Crossref query "gait recognition biometric data GDPR legal definition body shape": only Bygrave & Tosoni and Jasserand relevant).

### Safe wording
"That 'biometric data' is defined by the processing and its identifying capability rather than by the capture device is settled in the legal literature (Jasserand, 2016; Kindt, 2018; Bygrave and Tosoni, 2020) and in supervisory guidance (EDPB 3/2019 §§74–76; Recital 51). Legal commentary on ambient depth sensing in hospitals exists (Gerke et al., 2020; Martinez-Martin et al., 2021) and Mucha and Kampel (2022) raised the problem for depth faces. We do not offer a new legal theory; we apply the settled reading to depth body sensing and assemble the instruments that name the traits our experiments recover — AI Act Recital 15 ('body shape … gait, posture'), the EDPB–EDPS call to ban gait recognition, CNIL's reasonable-means test, BIPA's exclusion of height and weight and Zellmer's capability test, HIPAA Safe Harbor (Q)/(R) — and show where capability and statute diverge."

---

## 7. Claim 7 — "Premise-in-the-wild catalogue (16 depth systems) with documented internal contradictions"

### Verdict: WEAKENED

### Refuting / adjacent works

**7a. Mucha & Kampel ICCHP 2022 §2 [V extract lines 78–112].** Already catalogues the premise and its contradictions in depth AAL: Arulselvi et al. fall detection where "the sensor's privacy is taken for granted"; Planinc & Kampel "considers depth data as private"; Chou et al. "address depth sensors as not privacy-preserving due to existing works performing FR"; Banerjee et al. presenting depth to patients as more private; Ballester et al. toileting assistance accepted by 8 of 13; explicitly: "There are more such contradictory examples of publications." A depth-privacy premise catalogue with contradictions exists, four years earlier, from the group MultiGait cites.

**7b. MultiGait §4.1 [V via audit 07]** lists depth as "a commonly proposed sensor for 'privacy-friendly' surveillance" citing Kepski & Kwolek, Rougier, Villamizar, Mucha & Kampel; Stone & Skubic and Planinc & Kampel also cited.

**7c. Field-level catalogues:** Baldassini et al. NeurIPS 2025 position paper (privacy-preserving HPE never measures privacy) [V-cite]; Amin et al. arXiv 2608.04501 (10 % of privacy-preserving AR papers adopt a formal definition) [V via 04b]; Momin et al. Sensors 2022 review ("depth or thermal imagery is preferable … due to privacy") [V]; Kröger 2019 [V].

**7d. Same-institution contradictions.** Stanford (Haque CVPR 2016 vs Haque MLHC 2017 "de-identified depth images" [V extract `haque17a.txt` line 110] / Luo MLHC 2018 "prevent identification of the person in the images" [V extract `luo18a.txt` line 127]) and Missouri (Banerjee EMBC 2012 resident identification vs Banerjee EMBC 2014 / Rantz 2014 [V via audit 04]) — not found in any prior catalogue; Mucha 2022 cites Banerjee's *acceptance* work, not the 2012 identification paper.

### Safe wording
"Mucha and Kampel (2022) first documented that depth-sensing publications contradict one another on privacy, and position papers now show that 'privacy-preserving' sensing is rarely evaluated for privacy (Baldassini et al., 2025; Amin et al., 2026). We extend the record to sixteen verified depth deployments spanning lavatories, care rooms, ICUs and transport hubs (2011–2026), and note two cases in which the same institution published depth-based identification of the same population it elsewhere described as de-identified (Stanford: Haque et al., 2016 vs 2017 and Luo et al., 2018; Missouri: Banerjee et al., 2012 vs 2014 and Rantz et al., 2014)."

---

## 8. Claim 8 — "No regulator text examined mentions depth, 3D, thermal or LiDAR sensing"

### Verdict: REFUTED as worded (the extracts contain the terms); the substantive point survives

### Evidence (grep of the regulator extracts, this pass) [V]

| Text | Hit | Context |
|---|---|---|
| CNIL, *Caméras dites « intelligentes » ou « augmentées »* (July 2022), `cnil2022.txt` | "caméras thermiques" ×2 (lines 153, 155); "capteurs infrarouges" (line 620); "silhouettes" (line 176) | Footnote 11 cites CNIL's own 2020 publications "La CNIL appelle à la vigilance … caméras thermiques" and "Caméras dites « intelligentes » et caméras thermiques : les points de vigilance de la CNIL et les règles à respecter"; line 620 gives infrared sensors as an example of non-image sensors; line 176 says the software layer recognises "des objets ou des silhouettes". So a **regulator text on thermal cameras exists** (CNIL, June 2020; page fetch returned HTTP 403, so its content is [UNVERIFIED] here — from the title it concerns fever screening, not identification). |
| Article 29 WP, WP193 (2012), `wp193.txt` | "infrared" ×2 (lines 835, 868); "3D" (line 1099) | Near-infrared cameras for vein recognition; "3D imaging" as an anti-spoofing technique. |
| EP JURI/PETI study 2021, `ep2021.txt` | "thermal" (line 774); "3D" (line 606) | iBorderCtrl "facial and thermal analysis tool to detect lies"; a reference to "Fundamentals and Advances in 3D Face Recognition". |
| AI Act (OJ 2024), `aiact.txt` | "depth" only as "in-depth understanding" (line 6743); no thermal/3D/LiDAR | Recital 15 lists "body shape … gait, posture" (line 271). |
| EDPB 3/2019, JO 5/2021, HHS 2012 | 0 hits for thermal/3D/depth/LiDAR (HHS "depth" is "further depth in Section 2.6") | — |

Consequence: the sentence "no regulator text examined mentions depth, 3D, thermal or LiDAR sensing" is false for three of the seven texts (CNIL 2022, WP193, EP 2021), and CNIL issued a dedicated thermal-camera text in 2020. What remains true, and is the point worth making: none of the texts addresses **depth or 3D body sensing as a modality**, none discusses LiDAR, the thermal mentions concern fever screening and deception detection (not identification), the 3D mentions concern face spoofing and 3D face recognition, and **no text states that any sensor modality is non-identifying**.

### Safe wording
"None of the regulatory texts examined (AI Act; EDPB 3/2019; EDPB–EDPS JO 5/2021; CNIL 2022; WP193; the EP 2021 study; HHS 2012) addresses depth or 3D body sensing as a modality, and none states that any sensor is non-identifying: thermal cameras appear only in the context of fever screening and deception detection (CNIL 2020/2022; EP 2021), '3D' only as anti-spoofing or 3D face recognition (WP193; EP 2021), and LiDAR nowhere. The premise that depth is non-identifying therefore has no regulatory author; it lives in engineering practice and in an inference from HIPAA Safe Harbor's silence."

---

## 9. Does anything pre-empt the REPOSITIONED framing?

- **No single work does.** The three closest are MultiGait (endpoints for depth silhouettes, false-sense framing), Hofmann 2014 (depth endpoints on TUM-GAID, twelve years earlier) and Miller et al. 2024 (swept duration and train–test delay for identification, VR).
- **Wording that must change beyond audits 01–07:** (i) do not say the depth field "never saw" the within-vs-across gap (Hofmann 2014 Table 6; Patruno 2019 R0 number); (ii) do not say "no depth identity-leakage evaluation of a degradation control exists" (Mucha 2022; Liu 2025) — say "no *value-precision* or *person-masking* test"; (iii) do not say the premise catalogue is the first (Mucha 2022 §2); (iv) reword claim 8 as in §8; (v) scope "gait untestable" and "fusion fails" as in §5; (vi) present metric anthropometry as a re-measurement with a new recovery path and drift model, not as the discovery of the mechanism.
- **Citations a hostile reviewer will expect that the current audits lack:** Hofmann 2014 Table 6 (with numbers); Miller et al. 2024 (2407.18380); Jasserand 2016; Kindt 2018; Bygrave & Tosoni 2020; Gerke, Yeung, Cohen JAMA 2020; Martinez-Martin et al. Lancet DH 2021; CNIL 2020 thermal-camera guidance; Dubois & Bresciani 2015 (already in 04).

---

## 10. Searches that returned nothing relevant (for the record)
- arXiv API: "frame-level / temporal leakage / session-disjoint / cross-session" × "person identification / re-identification / biometric" × "leakage / inflat / overestimat / pitfall", 20 newest — only ECG-biometrics-bench (May 2026) relevant; nothing in vision person-ID.
- arXiv API: depth × (quantization / bit depth / precision) × privacy × (identification / re-identification / anonymization) — no depth value-precision leakage study.
- arXiv API: gait × (number of frames / sequence length / observation time / gait cycles) — frame-count ablations only (GaitSet 7 frames; partial-cycle reconstruction), no seconds budget with a gate.
- Crossref: "biometric menagerie gait" — no gait/body-shape menagerie study.
- Crossref: "gait recognition biometric data GDPR legal definition body shape" — general Art. 4(14) commentary only; no gait/body-shape-specific legal analysis surfaced.
- arXiv API: Siskind group — 2004.06046 (randomized EEG trials) abstract does not report accuracy vs temporal distance; Li et al. TPAMI text mentions "close temporal proximity" without a swept gap.

## 11. Unverified items (do not cite content without checking)
- CNIL, June 2020 thermal-camera page — title [V via CNIL 2022 footnote], content not fetched (HTTP 403).
- Gerke, Yeung, Cohen JAMA 2020 — record [V]; whether it treats depth as de-identified: unread.
- Martinez-Martin et al. 2021 — record and abstract [V]; depth/HIPAA content: unread.
- Jasserand 2016; Kindt 2018; Bygrave & Tosoni 2020 — records [V]; texts unread.
- Pala et al. TCSVT 26(4):788–799, 2016, DOI 10.1109/tcsvt.2015.2424056 — record [V]; anthropometric content unread.
- Zhang et al., Pattern Recognition 179 (2026), DOI 10.1016/j.patcog.2026.113619 — record [V]; abstract unavailable; possible overlap with outline-vs-shape (flagged by audit 05) unresolved.
- Padilla-López et al. ESWA 2015 — record [V]; silhouette-as-privacy-filter content unread.
- Munaro 2014 numbers — secondary only (Haque/Karianakis tables).
- Absence of a gait/body-shape biometric-menagerie study and of any surface-normal person-ID work — Crossref/arXiv only.
- Hofmann 2014: TN/TB/TS evaluated on 16 people while N on 155 — stated in the preprint caption [V]; whether the published JVCIR version differs: unchecked.
