# 08 — ADVERSARIAL REFUTATION of C3/C4 novelty claims — COMPLETE (2026-09-09)

Auditor: single agent, no sub-agents, 44 tool calls. Task: for each of seven "first / to our knowledge" claims, find a prior work that already did the same measurement (any modality first, then depth), and give the strongest sentence the manuscript can still write.

Method: (i) re-read the completed reports 03, 07, 02, 04b for context; (ii) full-text search of the saved primary texts in `extracts/` (barbosa, karianakis, wu2017, haque2016, andersson, hanisch, friedman, sgela [Rao et al. TPAMI], rfreid [Fan et al. RF-ReID], lidargait, gait3d, tvrid/tvrid2026, delecluse2026, multigait_2609.01036, sdas2021) with a context-window script; (iii) external checks: arXiv API (5 queries), Crossref (9 queries incl. DOI records), Europe PMC (Paolanti 2018 full text), OpenAlex (5), Semantic Scholar (DOI look-ups worked; the search endpoint returned 429 after two calls), GaitSet PDF from arXiv (text-extracted locally). MDPI and SpringerLink pages were blocked (403 / login redirect). WebSearch was not used.

Legend: [V] = citation and numbers confirmed in the primary text (extract or fetched page) or on Crossref/arXiv/OpenAlex/Europe PMC; [V-sec] = number confirmed only in a named secondary source; [UNVERIFIED] = existence confirmed but content not read, or not confirmed at all. Quotations are under 15 words.

Verdict rules applied: REFUTED only if a [V] work makes essentially the same measurement (same axis, same kind of control), any modality; WEAKENED if adjacent work forces a qualifier; STANDS if a genuine search found nothing. Default to WEAKENED when uncertain.

---

## 0. Summary table

| # | Claim (short) | Verdict | Closest prior work | Qualifier forced |
|---|---|---|---|---|
| 1 | First quantification of FOV clipping (74 % of frames), height drift (89.5 mm/m), full-body gate ablation (28 vs 9 %) | **WEAKENED** | Karianakis 2018 ablated a learned frame-weighting on BIWI cross-session (RTA 50.0 vs uniform 45.7) [V]; Barbosa 2012 FOV-aware frame selection [V]; Andersson 2015 removed FOV distortion by design and discarded >2 SD measurements [V]; occluded/partial-silhouette gait literature (Roy 2011, Shaikh 2014, Uddin 2019) detects and reconstructs incomplete silhouettes [V DOIs, content UNVERIFIED] | "first to quantify clipping prevalence and metric height drift, and to ablate an explicit full-body validity gate"; not "first to show frame selection matters" |
| 2 | Metric (mm) anthropometry from unprojected depth + sensor mask as a cross-session baseline beating own CNNs | **WEAKENED** | Metric anthropometry from depth for re-ID is 2012–2018 practice: Barbosa 2012 (floor–head, height, via RANSAC floor plane) [V]; Munaro 2014 skeleton limb lengths [V-sec]; Andersson 2015 (140 people, 85 %) [V]; Paolanti 2018 / Liciotti 2017 TVPR: skeleton-free floor–head and floor–shoulder distances, head/shoulder circumference from top-view depth, 100 people, 23 sessions over 8 days [V]; John 2013 height-based gait from a colour-depth camera, 75 people, >80 % [V abstract] | "skeleton-free, frontal-view, cross-session, with a measured drift model"; "beats the CNNs we train, not published ones" |
| 3 | Fixed 2.5-s observation budget with top-k; "no BIWI work reports accuracy vs observation time" | **WEAKENED** (the BIWI sub-claim is false for skeleton input) | Rao et al. TPAMI 2022 Fig. 8: rank-1 on BIWI for skeleton sequence length f = 6/8/10 [V]; GaitSet AAAI 2019 Fig. 5: rank-1 vs number of silhouettes on CASIA-B, 82.5 % at 7 frames [V]; LidarGait Fig. 9 / §C, Gait3D sequence-length ablations [V]; Karianakis used PAVIS 5-frame walks for "shorter video sequences" [V]; RF-ReID fixed 3-s (90-frame) segments [V]; Nair 2023 10 s vs 100 s [V via 04b] | "no BIWI work reports depth-image accuracy against a wall-clock budget; skeleton work ablates frame count" |
| 4 | Cohort curve K = 4→28 with draw spread AND per-subject identifiability 0–100 % on cross-session depth; "who is enrolled matters more than how many" | **WEAKENED** | Andersson 2015 Fig. 4: average over 10 randomly drawn galleries per size, Kinect anthropometrics, within-session [V]; Delécluse FG 2025 Fig. 5: depth CMC-1 vs test-set size, within-session top-view [V via 07]; MultiGait Table 5 subsets (radar/CSI) [V via 07]; Friedman 2022 log-linear law [V]; Doddington 1998 / Yager & Dunstone 2010 per-subject vocabulary [V]. No gait/body/depth menagerie study found (arXiv, Crossref, OpenAlex) | "first cross-session depth-ID cohort curve with draw spread and per-subject distribution"; the concept is the menagerie |
| 5 | RGB with its own DeepLabV3 segmenter = RGB with the sensor mask, as a control | **WEAKENED** (no direct precedent; adjacent practice forces a qualifier) | Karianakis 2018 ran "Body RGB" and "Body Depth" through the same body region on TUM-GAID [V; whether the Kinect mask was applied to RGB is UNVERIFIED]; MultiGait uses DeepLabv3 on video but empty-room subtraction on depth (segmenter differs by sensor, not equalised) [V]; GaitEdge ECCV 2022 treats segmentation noise as a shortcut [V arXiv]; QAGait 2024 silhouette quality [V arXiv]; LTCC parsing input; TVRID background erasing [V via 03] | "to our knowledge no prior modality comparison equalised the mask source with a second, image-native segmenter" |
| 6 | Same-pipeline RGB vs depth across sessions on three sensor technologies, as replication/extension of Hafner 2022 and Karianakis 2018 | **WEAKENED** (already framed as replication; one further precedent) | Karianakis TUM-GAID Body RGB 41.8 vs Body Depth 48.0 (ss), 50.0 vs 59.4 (ms) [V]; Castro 2020 same 3D-CNN gray vs depth cross-session TUM-GAID [V via 02]; Hofmann 2014 GEI vs depth-GEI [V via 02]; Delécluse FG 2025: one pipeline, depth vs RGB-D on Asus Xtion (structured light), Kinect 2 (ToF) + RealSense D435 (active stereo), Kinect 1 — three depth technologies, within-session, RGB-D not RGB [V]; MultiGait video vs depth cross-session, silhouettes only [V] | "RGB-only vs depth-only under one pipeline, across sessions, on three depth technologies"; cite Castro 2020 and Delécluse 2025 |
| 7 | Human single-shot 6.7 % on BIWI depth (Haque 2016) vs machine 30–50 % | **STANDS** (as a citation; verified with caveats) | Haque 2016 Table 2 row 2 "Human Performance": BIWI 6.7, IAS-A 21.2, IAS-B 15.1, PAVIS 1.7, DPI-T 19.2; Random 2.0; "Four humans manually performed the identification task" with "full access to the training data" [V] | n = 4 observers, trials unstated; compare with Haque's own 30.1 (ss) / 45.3 (ms), not with the manuscript's 28-way numbers; humans were above chance (6.7 vs 2.0) |

Nothing in MultiGait (arXiv:2609.01036) pre-empts any of the seven items (checked in report 07 §1.8; re-confirmed by grep of the saved HTML text: no gating, drift, budget, per-subject, cohort-for-depth, or mask-source control).

---

## 1. Claim 1 — FOV-clipping mechanism (74 % of frames; 89.5 mm/m drift vs 105 mm between-subject SD; gate 28 vs 9 %)

**Verdict: WEAKENED.**

What was searched: extracts for "field of view / cropped / truncated / image boundaries / too far / partially visible" (Barbosa, Karianakis, Andersson, Haque, Wu, TVRID, MultiGait, SDAS); arXiv API for gait + silhouette + (truncated | partial body | field of view | cropped) → 0 results; arXiv API gait + occlusion + silhouette → 14 papers (2021–2026), none quantifying the fraction of affected frames; Crossref for partial-silhouette / occlusion-reconstruction gait; Crossref and OpenAlex for Kinect height-measurement error vs distance (nothing relevant found).

Evidence, closest first:

1. **Karianakis, Liu, Chen, Soatto, ECCV 2018, DOI 10.1007/978-3-030-01228-1_44 [V].** Two relevant facts. (a) Frame removal: from BIWI "we remove the frames with no person or a person heavily occluded from the image boundaries or too far from the sensor" — an un-ablated hard gate [V, karianakis.txt]. (b) A *quantified* frame-weighting effect on BIWI cross-session: Reinforced Temporal Attention (per-frame Bernoulli weights) 50.0 % vs uniform averaging over the CNN-LSTM 45.7 % vs soft attention 46.4 % (Table 1) [V via report 03, primary text]. This is an ablation of "which frames count" on exactly the manuscript's probe set. It differs from the manuscript's in kind — learned weights, no named mechanism, no clipping statistic, no metric drift — but a reviewer will say frame selection has been ablated on BIWI. The manuscript must position the full-body gate as an *interpretable, mechanism-derived* gate whose effect (28 vs 9 %) is larger and explained.
2. **Barbosa, Cristani, Del Bue, Bazzani, Murino, ECCV-W 2012, DOI 10.1007/978-3-642-33863-2_43 [V].** Frame selection: the frame with best joint confidence, "closest to the camera and it was not cropped by the sensors fields of view", typically "approximately 2.5 meters away" [V, barbosa.txt]. The FOV-aware gate is 14 years old; no prevalence count and no ablation.
3. **Andersson & Araujo, AAAI 2015, DOI 10.1609/aaai.v29i1.9212 [V].** Designed the capture so that gait cycles were recorded "without distortions caused by subjects moving in or out of the sensor's field of view" (pan-following Kinect), and discarded measurements "beyond two standard deviations from the mean" as noise [V, andersson.txt]. FOV clipping was recognised as a measurement hazard for Kinect anthropometrics and removed by design rather than measured.
4. **Occluded / partial-silhouette gait literature (RGB silhouettes):** Roy, Sural, Mukherjee, "Occlusion detection and gait silhouette reconstruction from degraded scenes", SIViP 2011, DOI 10.1007/s11760-011-0245-5 [V DOI]; Shaikh, Saeed, Chaki, "Gait recognition using partial silhouette-based approach", SPIN 2014, DOI 10.1109/SPIN.2014.6776930 [V DOI] and IJBM 2017, DOI 10.1504/ijbm.2017.084130 [V DOI]; Uddin et al., "Spatio-temporal silhouette sequence reconstruction for gait recognition against occlusion", IPSJ T-CVA 2019, DOI 10.1186/s41074-019-0061-3 [V DOI]; Hasan et al., IEEE Access 2024, DOI 10.1109/ACCESS.2024.3482430 [V DOI]. These detect incomplete silhouettes and reconstruct or use partial ones; whether any quantifies the prevalence of incomplete frames or ablates a completeness gate is [UNVERIFIED] (abstracts elided; Springer pages blocked). They establish that "incomplete body in frame degrades gait ID" is known; none concerns depth or metric height.
5. **Kinect occlusion benchmark:** Li et al., "A Benchmark for Gait Recognition under Occlusion Collected by Multi-Kinect SDAS", arXiv:2107.08990 (2021) [V] — Azure Kinect, occlusion types, cameras placed so their depth FOVs "completely cover the subject's walking area" [V, sdas2021.txt]; no FOV-clipping statistic.
6. **Depth error vs range:** Khoshelham & Elberink, Sensors 2012, DOI 10.3390/s120201437 [V DOI via 03] — random depth error grows with range. This is a noise model, not the systematic height-under-estimation from clipping that the manuscript reports; no paper measuring a *height-feature drift per metre* on Kinect was found (Crossref ×3, OpenAlex ×1).

Why not REFUTED: no work measures (i) the fraction of walking frames in which the body is clipped, (ii) a metric height drift per metre of standing distance against between-subject SD, or (iii) cross-session accuracy with vs without an explicit full-body gate. Karianakis's RTA ablation is the same *axis* (frame inclusion) but not the same *kind of control* (learned weights vs explicit validity gate) and offers no mechanism.

Safe wording: "Frame-quality gating has been applied to BIWI-style depth data since Barbosa et al. (2012) and Karianakis et al. (2018) showed that learned per-frame weights raise cross-session accuracy by 4.3 pp; to our knowledge no work has measured how often the sensor's field of view clips the walking body (74 % of frames here), how far height features drift with standing distance as a result (89.5 mm per metre, against a 105 mm between-subject SD), or what an explicit full-body gate does to cross-session accuracy (28 % vs 9 %)."

---

## 2. Claim 2 — Metric anthropometry from unprojected depth + sensor mask as the cross-session baseline that beats the manuscript's own CNNs

**Verdict: WEAKENED.**

What was searched: extracts (Barbosa, Andersson, Munaro rows in Haque/Karianakis, Wu, TVRID, Delécluse); Crossref for Paolanti 2018 and Liciotti 2017 (TVPR); Europe PMC full text of Paolanti 2018 (PMC6210929); OpenAlex records for John 2013, Munsell 2012, Munaro 2014; Crossref for BenAbdelkader 2002 (RGB height + stride).

Evidence:

1. **Paolanti, Romeo, Liciotti, Pietrini, Cenci, Frontoni, Zingaretti, Sensors 18(10):3471, 2018, DOI 10.3390/s18103471, PMC6210929 [V, full text].** Skeleton-free anthropometric features from top-view depth: d1 "distance between floor and head", d2 "distance between floor and shoulders", d3 head surface area, d4 head circumference, d5 shoulder circumference, d6 shoulder breadth, d7 thoracic anteroposterior depth (units not stated explicitly; camera "4 m above the floor", so these are physical distances). TVPR dataset (Liciotti et al. 2017, DOI 10.1007/978-3-319-56687-0_1 [V DOI]): 100 people, "23 sessions" over "eight days", Asus Xtion Pro Live, ceiling-mounted; protocol: "the first passage under the camera for training and using the others as the testing set" — the passages are the out-and-back walk of one session, so this is within-session (same clothes) [inference from the recording description; UNVERIFIED that no cross-day pairs exist]. This pre-empts "metric, skeleton-free anthropometry from a depth mask" as a re-ID representation, in the top-view setting and within session.
2. **Barbosa et al. 2012 [V].** Ten soft biometrics in physical units from depth + skeleton + RANSAC floor plane: d1 floor–head distance, d3 "height estimate", d4 floor–neck, torso/legs ratio, geodesics on the mesh [V, barbosa.txt]; 79 people, cross-day with clothing change; height-related features most informative (nAUC 88.10 % for d1) [V via 03]. Metric anthropometry across sessions from a consumer depth sensor is therefore 2012 work.
3. **Munaro et al. 2014, DOI 10.1007/978-1-4471-6296-4_8 [V Crossref; S2 137 citations; abstract not retrievable].** Skeleton descriptor of limb lengths and ratios; BIWI Walking single-shot 21.1 % (NN), multi-frame 39.3 % [V-sec: Haque Tables 2–3; Karianakis Table 1].
4. **Andersson & Araujo 2015 [V].** Twenty anthropometric and gait attributes from Kinect skeletons; anthropometric alone 84.7–85.4 % on 140 people; anthropometric > gait attributes [V, andersson.txt] — a hand-crafted anthropometric baseline outperforming a competing representation, within session.
5. **John, Englebienne, Kröse, "Person re-identification using height-based gait in colour depth camera", ICIP 2013, DOI 10.1109/ICIP.2013.6738689 [V DOI; abstract via OpenAlex].** Height time series from a colour-depth camera; TUM-GAID and studio data; "over 80% accuracy for 75 people" with combined features, "significantly better than standard colour-based features" [V abstract]. Height from depth beating colour appearance is in print since 2013.
6. **Munsell, Temlyakov, Qu, Wang, ECCV-W 2012, DOI 10.1007/978-3-642-33885-4_10 [V DOI; content UNVERIFIED]** — anthropometric + motion biometrics from Kinect.
7. **Adverse to "hand-crafted beats CNN":** Karianakis 2018 and Haque 2016 both state the opposite on BIWI (3D RAM 30.1 > skeleton NN 21.1; CNN 25.4 > skeleton 21.1) [V via 03]; Wu 2017 ED+SKL 24.47 % Walking ≈ Karianakis CNN 25.4 % [V]. No prior depth paper reports hand-crafted anthropometry beating a CNN at matched protocol cross-session; the manuscript's result is specific to its own CNNs, data regime (matched n_train, full-frame inputs) and 28-way closed set.

Why not REFUTED: no prior work recovers millimetre anthropometry from *unprojected frontal depth through the sensor's person mask without a skeleton*, characterises its drift, and uses it as the reference against which learned models on the same frames are judged across sessions. But "metric anthropometry from depth for re-ID" and "skeleton-free anthropometry from a depth mask" are both pre-empted (Barbosa; Paolanti/TVPR).

Safe wording: "Anthropometric soft biometrics from consumer depth sensors are established (Barbosa et al., 2012; Munaro et al., 2014; Andersson and Araujo, 2015), including skeleton-free floor-referenced distances from top-view depth (Paolanti et al., 2018). We recover them in millimetres from unprojected frontal depth through the sensor mask, model their distance-dependent drift, and find that this 13-dimensional descriptor outperforms every CNN we train on the same frames and protocol across sessions (19 % single-frame); published depth networks evaluated on whole sequences under the standard Training→Walking protocol remain higher (Haque et al., 2016: 30.1 %; Karianakis et al., 2018: 25.4 % single-shot), as do hand-crafted depth descriptors (Wu et al., 2017: 24.47 %; Munaro et al., 2014: 22.4 %), for reasons we set out in §X."

---

## 3. Claim 3 — Fixed 2.5-s observation budget with top-k; "no BIWI work reports accuracy versus observation time"

**Verdict: WEAKENED** — the general practice is standard in several modalities, and the BIWI sub-claim is false for skeleton input.

What was searched: extracts (Karianakis, Wu, Haque, sgela [Rao TPAMI], lidargait, gait3d, rfreid, andersson); GaitSet PDF from arXiv (text-extracted); arXiv API "observation time" + gait/re-ID → 0; Crossref DOI record for Rao TPAMI; S2 search on Kinect frames-vs-accuracy (429).

Evidence:

1. **Rao, Wang, Hu, Tan, Guo, Cheng, Liu, Hu, "A Self-Supervised Gait Encoding Approach With Locality-Awareness for 3D Skeleton Based Person Re-Identification", IEEE TPAMI 2022, DOI 10.1109/TPAMI.2021.3092833 [V Crossref]; arXiv:2009.03671 [V].** Fig. 8: "Rank-1 accuracy on different datasets ... f denotes the sequence length", for f = 6, 8, 10 skeleton frames, datasets including BIWI (training set vs walking test set, i.e. the cross-session protocol) [V, sgela.txt]. This is accuracy versus observation length on BIWI, albeit in frames (0.2–0.33 s at 30 fps), for skeletons, and as a hyper-parameter sweep rather than a deployment budget. It falsifies the flat statement "no BIWI work reports accuracy versus observation time".
2. **Chao, He, Zhang, Feng, GaitSet, AAAI 2019, DOI 10.1609/aaai.v33i01.33018126 [V via 02]; arXiv:1811.06186 [V PDF].** Fig. 5 "Average rank-1 accuracies with constraints of silhouette volume on CASIA-B": random-frame curve 44.1 → 94.8 % as the number of silhouettes grows; "82% accuracy with only 7 silhouettes"; "close to the best performance when the samples contain more than 25 silhouettes" [V]. The accuracy-vs-frame-count curve is a standard gait ablation.
3. **LidarGait, CVPR 2023, DOI 10.1109/CVPR52729.2023.00108 [V via 02]**: Fig. 9 "performance ... on used frame number in inference", 1 frame → 25.82 % rank-1 [V, lidargait.txt]. **Gait3D, CVPR 2022, DOI 10.1109/CVPR52688.2022.01959 [V via 02]**: sequence-length analysis [V, gait3d.txt].
4. **Karianakis 2018 [V]**: "To evaluate our method when shorter video sequences are available, we use IIT PAVIS", 5-frame walks — a short-observation regime, on PAVIS rather than BIWI [V].
5. **Fan et al., RF-ReID, ECCV 2020 (extract rfreid.txt; DOI [UNVERIFIED])**: decision unit = "RF segments of 3 seconds (90 frames)" sampled from tracklets, with cross-day clothing change [V extract]. A fixed short observation unit in a privacy-framed cross-day re-ID paper.
6. **Nair et al., USENIX Security 2023, arXiv:2302.08927 [V via 04b]**: 94.33 % from 100 s vs 73.20 % from 10 s of VR telemetry — accuracy vs observation time in seconds, multi-session.
7. **BIWI depth-image work**: Wu 2017 varies the number of *gallery* images (1 vs 5) [V]; Haque 2016 and Karianakis 2018 pool whole sequences [V]; Delécluse 2025 mean-pools whole sequences with frame counts unstated [V via 07]. None reports depth-image accuracy against clip duration.

Why not REFUTED: no BIWI *depth-image* work fixes a wall-clock budget and reports rank-1/top-k under it with a validity gate; Rao's f-sweep is frames, skeletons and a hyper-parameter, not an operating point. But the axis (accuracy vs observation length) is thoroughly explored in gait, LiDAR, RF and VR, and on BIWI itself for skeletons.

Safe wording: "Accuracy as a function of observation length is a standard ablation in gait recognition (GaitSet: 82 % from 7 silhouettes; LidarGait; Gait3D) and in VR telemetry (Nair et al., 2023: 10 s vs 100 s), and Rao et al. (2022) sweep skeleton sequence length on BIWI; depth-image work on BIWI has instead reported single-shot and whole-sequence numbers (Haque et al., 2016; Karianakis et al., 2018; Delécluse et al., 2025) or varied the number of gallery images (Wu et al., 2017). We fix a deployment-motivated 2.5-s budget and report rank-1 and top-3 under it with a validity gate."

---

## 4. Claim 4 — Cohort curve K = 4→28 with draw spread AND per-subject identifiability; "which people are enrolled matters more than how many"

**Verdict: WEAKENED.**

What was searched: extracts (andersson, friedman, hanisch, multigait, delecluse via report 07); arXiv API "biometric menagerie / Doddington zoo / gait per-subject" → only Popescu-Bodorin iris papers; Crossref "biometric menagerie gait sheep goats lambs wolves" → Doddington 1998 and Mason's book chapters (not menagerie studies); OpenAlex "biometric menagerie gait recognition" → no gait/body/depth menagerie study; S2 search 429.

Evidence:

1. **Andersson & Araujo 2015 [V].** Fig. 4: accuracy vs gallery size where "each point ... is the average over 10 galleries with the same size and randomly drawn"; ~98 % for very small galleries converging to ~87 % at 140; KNN vs SVM ordering flips with gallery size; Fig. 5 by attribute set [V, andersson.txt]. This is a cohort curve with random-draw averaging (and, per report 03, 95 % CIs) on Kinect anthropometrics — within session, no per-subject analysis.
2. **Delécluse, Wannous, Guimas, IEEE FG 2025, DOI 10.1109/FG61629.2025.11099215 [V via 07].** Fig. 5: depth-only CMC-1 falls from ~100 % to ~86 % as the test-set size grows (TVPR2, within-session, top-view) [V via 07]. A depth-specific gallery-size dependence, no draw spread, no per-subject.
3. **MultiGait, arXiv:2609.01036 [V].** Table 5 re-runs radar/CSI systems on identity subsets (10/20/12/6) matched to the original papers — a cohort-size control, single-session, not for depth [V via 07].
4. **Friedman, Stern, Prokopenko, Djanian, Applied Sciences 2022, DOI 10.3390/app122111144 [V].** Rank-1 IR "linear in log(Gallery Size)"; EER stable across gallery size; random subsets of subjects [V, friedman.txt]. Grother & Phillips CVPR 2004, DOI 10.1109/CVPR.2004.1315146 [V via 03]; Bolle et al. 2005, DOI 10.1109/AUTOID.2005.48 [V via 03].
5. **Per-subject heterogeneity:** Doddington et al., ICSLP 1998, DOI 10.21437/ICSLP.1998-244 [V Crossref]; Yager & Dunstone, TPAMI 2010, DOI 10.1109/TPAMI.2008.291 [V via 03]. Hanisch et al., PoPETs 2023, DOI 10.56553/popets-2023-0011 [V via 04b] analyse which *features* make gait identifiable, not which *people*, and propose human-observer comparison as future work [V, hanisch.txt]. No gait, body-shape, skeleton or depth menagerie study was located.
6. **The sentence "which people are enrolled matters more than how many"** was not found anywhere. Friedman's framing is that gallery size drives rank-1 IR; Andersson's that accuracy declines smoothly with K. The manuscript's observation that the between-draw spread at fixed K (58 pp) exceeds the K effect is therefore new as an empirical statement in this domain, but it is the menagerie's prediction.

Why not REFUTED: no prior work reports, on cross-session depth identification, a K-sweep with draw spread together with a per-subject identifiability distribution.

Safe wording: "Gallery-size decline follows known laws (Grother and Phillips, 2004; Friedman et al., 2022) and has been measured for Kinect anthropometrics within a session with random galleries (Andersson and Araujo, 2015) and for top-view depth within a session (Delécluse et al., 2025); per-subject heterogeneity is the biometric menagerie (Doddington et al., 1998; Yager and Dunstone, 2010). Across sessions on depth we find that the spread between random cohorts of the same size (58 pp at K = 4→28) exceeds the effect of size itself and that per-subject identifiability spans 0–100 %: in this regime who is enrolled matters more than how many, which a privacy assessment based on cohort size alone would miss."

---

## 5. Claim 5 — RGB with its own DeepLabV3 segmenter equals RGB with the sensor mask, as a control

**Verdict: WEAKENED** (no direct precedent found; adjacent practice forces a "to our knowledge" and a citation to Karianakis).

What was searched: extracts (multigait, karianakis, lidargait, gait3d, hafner, liu2025, tumgaid_preprint) for DeepLab / mask quality / segmentation quality / body index; arXiv API gait + segmentation + silhouette + (quality | ground truth) → QAGait, It Takes Two, GaitSTR, wildlife gait (none compare mask sources as a control); arXiv API GaitEdge; S2 search for predicted-vs-ground-truth mask ablations in re-ID (429).

Evidence:

1. **Karianakis 2018 [V].** Table 2 on TUM-GAID compares "Body RGB (ss) [82] 41.8" with "Body Depth (ss) 48.0" — both restricted to the body region; Fig. 2 shows "the cropped color image" alongside the body-index-masked depth [V, karianakis.txt]. Whether the RGB arm was masked by the Kinect body index or only bounding-box cropped is not stated in the text read [UNVERIFIED]. Either way, feeding sensor-derived person regions to an RGB pipeline for a modality comparison is precedent; the manuscript's addition is the second arm (an RGB-native segmenter) that tests whether the mask source explains the modality gap.
2. **MultiGait [V].** Silhouettes come from DeepLabv3 for video and from empty-room subtraction for depth [V, §5.2]; the video-vs-depth comparison is therefore confounded by segmenter, and the paper does not run a segmenter-equalised control (report 07 §1.7 caution). This is the closest *unaddressed* version of the confound the manuscript controls.
3. **Liang et al., GaitEdge, ECCV 2022, arXiv:2203.03972 [V arXiv].** End-to-end gait models "suffer from the gait-irrelevant noises, i.e., low-level texture and colourful information" leaking through segmentation; GaitEdge fixes silhouette interiors and learns only edges to limit what segmentation passes to recognition [V abstract]. Establishes that segmentation is a leakage channel; not a mask-source control for a modality comparison.
4. **Wang et al., QAGait, arXiv:2401.13531 (2024) [V arXiv]** — silhouette quality assessment; no source comparison.
5. **LTCC "ResNet-50 (Parsing)" input baseline and TVRID "background erasing" augmentation** [V via 03] — related uses of image-derived masks, not controls.

Why not REFUTED: nothing found that runs RGB through (a) an image-native segmenter and (b) the depth sensor's mask at fixed pipeline to show equality, thereby attributing a modality gap to appearance rather than to mask quality.

Safe wording: "Because the depth arm uses the sensor's person mask, we also run RGB through an image-native DeepLabV3 segmenter; the two RGB conditions agree within X pp at every rung, so the within-session RGB advantage is not a segmentation artefact. Prior RGB-vs-depth comparisons either applied sensor-derived body regions to both modalities without this control (Karianakis et al., 2018) or used a different segmenter per sensor (Todt et al., 2026); to our knowledge the mask-source control has not been reported."

---

## 6. Claim 6 — Same-pipeline RGB vs depth across sessions on three sensor technologies (replication of Hafner 2022 within-session, Karianakis 2018 across clothing), with the segmenter control attributing the within-session edge to clothing

**Verdict: WEAKENED** (the claim is already framed as replication; two further precedents must be named).

What was searched: extracts (karianakis TUM-GAID section; delecluse2026 sensor table; multigait; hafner; liu2025); reports 02/03/07.

Evidence:

1. **Karianakis 2018, TUM-GAID new-clothes scenario (session 2, three months later, 32 people) [V]**: Body RGB 41.8 vs Body Depth 48.0 (single-shot), 50.0 vs 59.4 (multi-shot, LSTM+RTA); "re-identification from body depth is more robust than from body RGB" [V, karianakis.txt]. Same-pipeline (CNN-LSTM) RGB-vs-depth across sessions on Kinect v1.
2. **Castro, Marín-Jiménez, Guil, Pérez de la Blanca, Neural Comput. Appl. 2020, DOI 10.1007/s00521-020-04811-z [V via 02]**: same 3D-CNN, TUM-GAID cross-session: gray TN 18.8 / TB 21.9 / TS 12.5 vs depth 62.0 / 53.1 / 78.1 [V via 02]. A second same-pipeline cross-session RGB-vs-depth result on the same sensor.
3. **Hofmann et al., JVCIR 2014, DOI 10.1016/j.jvcir.2013.02.006 [V via 02]**: GEI (RGB silhouettes) 44 % vs depth-GEI 28 % vs DGHEI 50 % cross-session — an early same-representation comparison where plain depth-GEI lost to RGB-GEI.
4. **Hafner et al., CVIU 2022, DOI 10.1016/j.cviu.2021.103352 [V]**: within-instance BIWI RGB 89.0 vs depth 44.5 (44.5 pp) [V via 03] — the manuscript's +45 pp within-session replicates this.
5. **Delécluse et al., FG 2025 [V]**: one pipeline evaluated on TVPR2 (Asus Xtion Pro Live, structured light), GODPR (Kinect 2 ToF and RealSense D435 active stereo) and BIWI (Kinect 1): depth-only vs RGB-D on each [V, delecluse2026.txt Table I/II]. This is "one pipeline, three depth technologies, depth vs colour-bearing input" — within-session (or protocol-unspecified for BIWI), RGB-D rather than RGB-only, and with no attribution. It forces the manuscript to say "RGB-only vs depth-only, across sessions" rather than "three sensor technologies under one pipeline" tout court.
6. **MultiGait [V]**: video vs depth across sessions with eight sensors, but both as silhouettes — cannot bear on "RGB's edge is clothing" (report 07 §1.7).
7. **Liu et al., Sensors 2025, DOI 10.3390/s25010271 [V via 03]**: RGB/thermal/depth same ResNet34/ViT pipeline, Grad-CAM shows RGB on clothing and background; 6 subjects; no cross-day.

Why not REFUTED: no prior work runs RGB-only and depth-only through one pipeline across sessions on three depth technologies and attributes the within-session gap to clothing with a mask-source control.

Safe wording: "That depth outlasts RGB across a clothing change at fixed pipeline was shown on TUM-GAID (Karianakis et al., 2018; Castro et al., 2020), and RGB's within-session advantage on BIWI (~45 pp) was reported by Hafner et al. (2022); one-pipeline depth-versus-RGB-D comparisons across structured-light, time-of-flight and active-stereo sensors exist within session (Delécluse et al., 2025). We replicate the within-session gap and its disappearance across sessions on three depth technologies with RGB-only and depth-only arms, and, using the mask-source control of §Y, attribute the within-session edge to clothing appearance."

---

## 7. Claim 7 — Human single-shot 6.7 % on BIWI depth (Haque 2016) vs machine 30–50 %

**Verdict: STANDS** (as citable support; the number and protocol are verified, with caveats that must appear in the text).

Verified from haque2016.txt (Haque, Alahi, Fei-Fei, CVPR 2016, DOI 10.1109/CVPR.2016.138 [V via 03]):
- Table 2 (single-shot), row 2, modality Depth, "Human Performance": BIWI 6.7, IAS-A 21.2, IAS-B 15.1, PAVIS 1.7, DPI-T 19.2; row 1 "Random": BIWI 2.0, IAS-A 9.1, IAS-B 8.1, PAVIS 1.3, DPI-T 8.3 [V].
- Protocol: "Four humans manually performed the identification task. Each human was shown a single test input and was given full access to the training data." [V]. Number of trials per human, presentation format (depth image vs point cloud rendering), and time allowed are not stated [V by absence].
- Machine comparators on the same protocol (BIWI, Training→Walking, 50-way): 3D RAM 30.1 (single-shot), 4D RAM 45.3 (multi-shot) [V]; Karianakis CNN 25.4 / RTA 50.0 [V].

Adversarial caveats a reviewer will raise:
1. n = 4 observers, unspecified trial count — a weak human baseline; on PAVIS humans were at chance (1.7 vs 1.3), on BIWI 3.3× chance (6.7 vs 2.0), so "illegible" should read "near chance" or "far below machine".
2. The machine figure in the sentence should be Haque's own 30.1/45.3 (or Karianakis 25.4/50.0) on the same 50-way protocol, not the manuscript's 28-way 2.5-s numbers (chance 3.6 %).
3. Other human-vs-machine precedents to cite for the thesis: Kumawat & Nagahara, BDQ, ECCV 2022, arXiv:2208.02459 [V via 04b] (machine attacker vs human-observer study on degraded RGB video); Hanisch et al. 2023 propose "experiments with human observers" as future work [V, hanisch.txt] — i.e. the human/machine gap is recognised as under-measured for gait.

Safe wording: "On BIWI depth, four human observers given the full enrolment set identified single frames at 6.7 % (chance 2.0 %), whereas the machine baselines of the same study reached 30.1 % single-shot and 45.3 % over sequences (Haque et al., 2016): imagery that is nearly illegible to people is not anonymous to models."

---

## 8. What the C3/C4 text must NOT say (consolidated)

- "First to show frame selection or gating matters on BIWI" (Karianakis's RTA ablation, 50.0 vs 45.7).
- "First metric anthropometry from depth" or "first skeleton-free anthropometry from a depth mask" (Barbosa 2012; Paolanti 2018).
- "No BIWI work reports accuracy versus observation length" (Rao et al. 2022 sweep skeleton sequence length on BIWI) — restrict to depth-image input and wall-clock budgets.
- "First cohort-size curve for depth/Kinect identification" (Andersson 2015; Delécluse 2025 Fig. 5).
- "Three sensor technologies under one pipeline" without "RGB-only vs depth-only, across sessions" (Delécluse 2025 did depth vs RGB-D on three technologies).
- "Humans cannot identify people from depth" — they were above chance; say near-chance / far below machine.
- Any comparison of 43–47 % with MultiGait 65–77 % or Delécluse 60.7 % without the protocol caveats of report 07 §3.

---

## 9. Unverified / not located

- Munaro 2014 chapter: abstract and any frames-vs-accuracy figure (Crossref, S2 and OpenAlex all lack the abstract; Springer blocked).
- Whether Karianakis's "Body RGB" arm on TUM-GAID was masked by the Kinect body index or bounding-box cropped.
- Whether Roy 2011 / Shaikh 2014, 2017 / Uddin 2019 / Hasan 2024 quantify the prevalence of incomplete silhouettes or ablate a completeness gate (abstracts elided; Springer redirect).
- Whether any TVPR test passage pairs a person across different days (Paolanti's protocol reads as within-session).
- Munsell 2012 and John 2013 feature units and protocols (abstract-level only).
- Any Kinect study of height-measurement drift versus standing distance (none found on Crossref ×3, OpenAlex ×1).
- Any gait / body-shape / skeleton / depth biometric-menagerie study (none found on arXiv, Crossref, OpenAlex; S2 search unavailable).
- RF-ReID (Fan et al., ECCV 2020) DOI — not checked.
- Semantic Scholar and dblp search coverage was unavailable (429 / bot wall), so absence claims for items 4 and 5 rest on arXiv, Crossref and OpenAlex only.
