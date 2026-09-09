# 08 — Adversarial refutation of contribution C1 (protocol ladder) — COMPLETE, 2026-09-09

Role: hostile reviewer. For each of the six "first / to our knowledge" claims that audit 01 §B cleared, try to find a prior work that already did the same measurement — in any modality first, then in depth. Single agent, no sub-agents. About 45 tool calls.

Method: (i) local extracts first (`extracts/` — MultiGait HTML text, Delécluse 2026, TUM-GAID preprint, Wu 2017, Haque 2016, Hafner 2022, TVRID 2026, Li et al. TPAMI full text, Andersson 2015); (ii) arXiv abstract pages and the arXiv API; (iii) Crossref and Europe PMC records; (iv) four WebSearch probes (quota was still available). No DOI below was typed from memory: each is from a Crossref/Europe PMC record, an arXiv page, or a saved extract. Legend: **[V]** = confirmed on the DOI/arXiv/venue page, in an API record, or read in an extract; **[V in 01]** = verified by audit 01, not re-checked here; **[UNVERIFIED]** = not confirmed. Quotes are under 15 words.

Strategic frame: manuscript = mechanistic and evaluation-methodology complement to MultiGait (Todt, Morsbach, Dissert, Strufe, arXiv:2609.01036, 1 Sep 2026). Novelty judged against that framing.

---

## 0. Headline

No claim is cleanly REFUTED, but **none STANDS as worded**. All six are WEAKENED, and two of them (claims 4 and 5, and the "MultiGait supplies the only two endpoints" clause of claim 3) are one hostile citation away from refutation because of a depth corpus the manuscript's C1 text ignores: **TUM-GAID (Hofmann, Geiger, Bachmann, Schuller, Rigoll, JVCIR 2014)**, which reported same-session and three-months-later identification rates for depth-based gait features on the same 155-class gallery twelve years before MultiGait. The three new facts a referee is most likely to raise:

1. **TUM-GAID already published same-session vs cross-session depth numbers** [V, `extracts/tumgaid_preprint.txt`, Table 6]: rank-1 identification, gallery N1–N4 (first session, 155 classes), probe N5–N6 (same session) vs TN1–TN6 (second session, three months later, different clothes): GEI (RGB) 99.4 → 44 %; depth-GEI 96.8 → 28 %; GEV 94.2 → 41 %; DGHEI 99.0 → 50 %. Chance 0.6 %. Caveat that saves the manuscript: the TN probe set is only 16 persons × 2 recordings, the N probe set 155 × 2 (the gallery is the same 155 classes); and the same-session split is *sequence-disjoint*, not frame-random.
2. **MultiGait's single-session split is sequence-level, not frame-level** [V, `extracts/multigait_2609.01036.txt`, lines 719 and 903]: a sample is "one sample for each time the participant walked the length of our setup", and the 70/10/20 split is "within identities" over those samples. So MultiGait's 100 % is an R2-type (same session, disjoint walks) number, not the manuscript's R0 frame-random number. This *helps* the manuscript (nobody has published R0 for depth) but the manuscript must stop describing MultiGait's endpoints as "frame-random vs session-disjoint".
3. **A temporal gap sweep at constant gallery size already exists for person identification** — in ECG, at the year scale: Scagnetto, "ECG Biometrics with ArcFace-Inception: External Validation on MIMIC and HEEDB", arXiv 2604.04485, 6 Apr 2026 [V]: "temporal stress test at constant gallery size", Rank@1 0.7853 → 0.6433 (MIMIC) and 0.6864 → 0.5560 (HEEDB) from 1 to 5 years. The manuscript's guard-band sweep is a different axis (an excluded buffer *inside one recording*, seconds to minutes), but "first … in any modality" cannot stand without that qualifier.

---

## 1. Claim-by-claim

### Claim 1 — "First temporal guard-band sweep at matched training size for person identification in any modality"

**Verdict: WEAKENED.**

What was searched: arXiv API (abs:"temporal gap" AND identification/re-identification/biometric/gait AND split/leakage/evaluation; abs:"guard band"/"embargo" variants returned nothing relevant in audit 01); WebSearch ("temporal gap" OR "guard band" between training and test frames person identification); WebSearch (activity recognition / video "temporal distance" between training and test windows); Li et al. TPAMI 2021 full text grepped for temporal-distance/proximity analyses; MultiGait text grepped for gap/matched/buffer; the elapsed-time literature listed in audit 01.

Closest prior work:

| Work | What they did | Same axis? | Same control? |
|---|---|---|---|
| Scagnetto, arXiv 2604.04485 (Apr 2026) **[V]** | ECG identification; enrolment–probe interval swept 1→5 years at constant gallery size; Rank@1 0.79→0.64, 0.69→0.56 | Time gap between train and test, swept | Constant *gallery* size (not n_train) |
| Best-Rowden & Jain, TPAMI 2018, DOI 10.1109/TPAMI.2017.2652466 **[V in 01]** | Face genuine scores vs elapsed time (years) | Yes, session scale | No matched-n control stated |
| Matovski et al., TIFS 2012, DOI 10.1109/TIFS.2011.2176118 **[V in 01]** | Gait CCR vs elapsed time with clothing/environment controlled | Yes, session scale | No |
| Hammerla & Plötz, UbiComp 2015, DOI 10.1145/2750858.2807551 **[V in 01]** | Adjacent HAR segments not independent; meta-segmented CV | Within-recording adjacency, but no gap sweep found (full text not read) [UNVERIFIED] | No |
| Li et al., TPAMI 2021, DOI 10.1109/TPAMI.2020.2973153 **[V extract]** | Block-design EEG; trials "in close temporal proximity" leak; re-split across blocks; relabelling | Within-recording, but binary (same block / other block), no gap sweep | No |
| Woo, arXiv 2605.06359 (May 2026) **[V in 01]** | Frame-level → temporal → scene-level splits, intrinsic decomposition | Graded rungs, no gap width sweep, not identity | No |
| h-block / hv-block / blocked CV (Burman 1994; Racine 2000; Bergmeir & Benítez 2012) and spatial dead zones (Pohjankukka 2017; Roberts 2017) **[V in 01]** | Gap h between train and test defined as a construct | Formal construct; sweeping of h not confirmed [UNVERIFIED] | n/a |

Why not REFUTED: no found work inserts and *sweeps* an excluded buffer between training and test frames drawn from the same recording while holding n_train fixed, for person identification. The ECG stress test sweeps a *longitudinal* interval between recordings at constant gallery size; that is the same family of measurement (accuracy vs time separation, size held fixed) on a different axis (between recordings, years) and with a different fixed quantity (gallery, not training set). A hostile referee will call it a precedent for "accuracy-versus-gap at fixed size"; the manuscript must pre-empt by naming it.

Safe wording: "To our knowledge this is the first *within-recording* guard-band sweep for person identification — an excluded temporal buffer between training and test frames of the same recording, widened from zero to the recording length at fixed n_train. Elapsed-time studies (face, gait, EEG, ECG — e.g. a constant-gallery 1–5-year stress test for ECG) sweep the interval *between* recordings; blocked and purged cross-validation define a gap but do not sweep it."

---

### Claim 2 — "First metadata-only identity floor (13 bounding-box scalars, 89 % under frame-random split)"

**Verdict: WEAKENED.**

What was searched: arXiv API (abs:"bounding box" AND identification/re-identification AND shortcut/leakage/trivial/metadata — 4 hits, none relevant [V]); WebSearch (re-ID "bounding box" position/size-only classifier, trajectory metadata); WebSearch (top-view depth anthropometric-only re-ID, TVPR/Liciotti); arXiv abs 2502.10195 (camera bias) and 2604.09690 (jaguar background/foreground diagnostic) read; MultiGait, Tian 2018, Wu 2017, TVRID extracts grepped for "bounding".

Closest prior work:

| Work | What they did | Match to "bbox-metadata-only floor"? |
|---|---|---|
| Liciotti et al., "Person Re-identification Dataset with RGB-D Camera in a Top-View Configuration", 2017, DOI 10.1007/978-3-319-56687-0_1 **[V DOI from Springer link; chapter not read]**; Paolanti et al., Sensors 18(10):3471, 2018, DOI 10.3390/s18103471 **[V Europe PMC]** | Top-view depth re-ID from hand-crafted anthropometric scalars (height, head/shoulder geometry) plus colour; Paolanti's abstract: features "derived from both depth and color images" | Scalar geometric features as the *primary* identifier, not a null floor; features use depth values, not detector metadata only |
| TVRID competition, ICPR 2026, DOI 10.1007/978-3-032-31936-4_16 **[V extract]** | Notes "height-related" cues and that models may exploit "simple height cues" | Speculation, not measured |
| Wang et al., "Spatial-Temporal Person Re-identification", AAAI 2019, arXiv 1812.03282 **[V in 01]** | Camera + timestamp prior lifts Market-1501 rank-1 91.2 → 98.1 % | Non-appearance metadata carries identity, but as a *prior* on top of appearance; no metadata-only accuracy isolated |
| Tian et al., CVPR 2018, DOI 10.1109/CVPR.2018.00607 **[V extract]**; Rueda-Toicen et al., arXiv 2604.09690 (Apr 2026) **[V]** | Background-only re-ID (37.1 % top-1 CUHK03); "background/foreground" leakage-controlled context ratio from inpainted images (jaguars) | Pixel-content floors, not metadata |
| Miller et al., Sci Rep 2020, DOI 10.1038/s41598-020-74486-y, PMC7567078 **[V]** | VR telemetry, 511 participants; full 18-DOF tracking 95.3 %; "only headset rotation (3DOF)" 19.5 % | A reduced-channel floor for identification — same *logic* (how much identity survives in a trivial sub-signal), different signal |
| Cătrună et al., arXiv 2402.08320 **[V in 01]** | Skeleton gait models exploit height | Shortcut diagnosis, no floor |
| Li et al. 2021 relabelling; Yagis et al. 2021 random-label null, DOI 10.1038/s41598-021-01681-w **[V in 01]** | Label-independence nulls | Null of a different kind (label permutation) |
| Madden & Piccardi, "Height Measurement as a Session-Based Biometric …" **[UNVERIFIED — Crossref query did not return the record]** | Height as a single-scalar biometric across disjoint cameras | If verified: a one-scalar identity floor from 2005, RGB |
| "Bounding-Box Trajectories Matter for Video Anomaly Detection", arXiv 2605.21957 **[V title only, via search]** | Bbox kinematics as features | Anomaly detection, not identity |

Why not REFUTED: nothing found trains a classifier on detector bounding-box geometry alone (no pixels, no depth values) and reports it as the floor under the same split ladder. The anthropometric-scalar lineage (Liciotti/Paolanti; height biometrics) is the genuine ancestor and must be cited, otherwise a depth referee will say "top-view depth re-ID has used a handful of scalars since 2017".

Safe wording: "We report what we believe is the first *detector-metadata-only* floor for person identification: thirteen bounding-box scalars, with no pixel or depth content, reach 89 % under a frame-random split. Hand-crafted anthropometric scalars have long been used as depth re-ID features (top-view height/head geometry), and spatio-temporal priors, background-only and reduced-channel baselines have shown that non-appearance signals carry identity; the floor differs in using nothing but framing metadata and in being measured on every rung of the ladder."

---

### Claim 3 — "First depth study to report every rung R0–R4 on the same data across three corpora and three sensor technologies with a cross-recording (R3) rung; MultiGait supplies only two endpoints on one own-sensor corpus without matched n_train"

**Verdict: WEAKENED** (the "every rung on the same data" core stands; two subordinate clauses are pre-empted).

Evidence:

- **Delécluse et al. 2026** (arXiv 2606.23230; `extracts/delecluse2026.txt`) **[V extract]** already evaluate depth-only re-ID on three corpora with three-plus depth technologies: TVPR2 (Asus Xtion Pro Live, structured light), GODPR ("Kinect V2 (time-of-flight) and Intel RealSense D435 (active stereo)"), BIWI (Kinect 1). Depth rank-1: TVPR2 88.4 %, GODPR 78.6 %, BIWI 60.7 %. Single protocol per corpus, no rungs — but "three corpora and three sensor technologies" is not itself a first for depth re-ID.
- **TUM-GAID (Hofmann et al. 2014, JVCIR 25:195–206, DOI 10.1016/j.jvcir.2013.02.006 [V Crossref]; numbers from `extracts/tumgaid_preprint.txt` Table 6 [V])**: depth-based gait (Kinect v1) with same-session (N) and cross-session (TN, three months, different clothes) probes against the same 155-class gallery — DGHEI 99.0 → 50 %, depth-GEI 96.8 → 28 %, GEV 94.2 → 41 %. So MultiGait is *not* the only depth work with two endpoints; TUM-GAID had them in 2014, with a gallery of fixed size (155 classes) though with a probe population that shrinks from 155 to 16 persons.
- **MultiGait's granularity** [V extract, lines 719, 903]: samples are whole walks; the single-session split is walk-level within identities. Its single-session number is therefore an R2-type number, not R0. Its matched-n construct (line 943) matches *identity count* to prior papers, not n_train across rungs.
- BIWI lineage (Wu 2017; Haque 2016; Karianakis 2018; Hafner 2022; Delécluse 2026) and TVRID 2026 [V extracts]: single-protocol each; no rungs.

No depth paper found reports a frame-random rung, a block/guard rung, a cross-recording rung and a cross-session rung on the same data. That core stands.

Safe wording: "This is, to our knowledge, the first depth study to report the full ladder — frame-random, block with guard sweep at matched n_train, cross-recording and cross-session — on the same data, replicated on three corpora spanning structured-light, time-of-flight and active-stereo sensors. Prior depth work reports one protocol per corpus (BIWI: cross-day; TVRID: same passage; Delécluse et al.: three corpora, one protocol each) or two endpoints (TUM-GAID 2014: same-session vs three-month probes; MultiGait 2026: walk-level within-session vs held-out session); none has a frame-random rung, a swept guard band, or a cross-recording rung with matched n_train."

Must not write: "MultiGait supplies the only two endpoints for depth" or "three corpora and three sensor technologies" as a stand-alone first.

---

### Claim 4 — "Quantified frame-random vs session-disjoint inflation for vision-based HUMAN identification; prior quantifications are animals, touch, ECG, PPG, EEG, MRI, HAR; the audit-06 gap stands"

**Verdict: WEAKENED — close to REFUTED for the enumerated "prior quantifications" list.**

Evidence:

- **Hofmann et al. 2014 (TUM-GAID)** [V extract, Table 6]: vision-based human gait identification — RGB GEI 99.4 % same-session → 44 % after three months; depth DGHEI 99.0 → 50 %. This is a published same-data quantification of same-session vs cross-session accuracy for vision-based human identification, twelve years before MultiGait, and it is absent from the claim's list. It is *sequence-disjoint* within session, not frame-random.
- **Sarkar et al., TPAMI 2005, DOI 10.1109/TPAMI.2005.39** [V in 01]: HumanID gait challenge, 78 % → 3 % across covariates including time — again vision, human, same data; not frame-random.
- **MultiGait 2026** [V extract]: walk-level within-session 100 % → held-out session 65–77 % (GaitBase, silhouettes from depth). Vision-based human identification, not frame-random.
- **Glazner et al., "Find the Leak, Fix the Split", arXiv 2511.13944, ICSEE 2026** [V arXiv]: frame-random splits of video-derived datasets leak through "high spatial and temporal correlations among consecutive frames"; proposes cluster-based frame selection; domain and numbers not in the abstract [UNVERIFIED].
- **Varga, arXiv 2604.14161 (Mar 2026)** [V arXiv]: subject-level leakage in a human gesture-recognition video study ("near-perfect accuracy"), no corrected numbers in abstract.
- **Miller et al. 2020** [V PMC7567078]: VR telemetry, all data "collected in a single day", session-held-out CV within that day → 95.3 %; not vision, no cross-day rung.

What remains unquantified in the literature searched: the *frame-random* (R0) rung against a session-disjoint rung for human identification from video, in any vision modality. That is narrower than the claim as worded.

Safe wording: "Same-session versus cross-session inflation has been quantified for vision-based human identification since the gait covariate studies (HumanID 2005; TUM-GAID 2014: RGB 99 → 44 %, depth 99 → 50 % after three months) and most recently by MultiGait (100 → 65–77 %, walk-level split). What has not been quantified for human vision is the *frame-random* rung — the split most video-identification papers actually use — against block, cross-recording and cross-session rungs on the same data; the closest quantifications of frame- or record-level leakage are in animal re-ID, touch, ECG, PPG, EEG, MRI and HAR."

Correction to audit 06/01 wording: "no 2020+ vision paper quantifying frame-random vs session-disjoint inflation for human identification" is defensible only with "frame-random" emphasised; drop any phrasing implying that same-session vs cross-session comparisons are absent from vision.

---

### Claim 5 — "The BIWI/TVRID depth literature never saw its own inflated number because protocols were cross-day or same-passage by construction; the retracted 0.99 is that number on the same data"

**Verdict: WEAKENED.**

Evidence for the claim (protocol statements read in primary texts):

- Wu, Zheng, Lai, TIP 2017 [V extract]: BIWI "Training" as gallery, "Still"/"Walking" as probe, "collected in a different day and in a different scene".
- Haque et al., CVPR 2016 [V extract]: identification "across days".
- Karianakis et al., ECCV 2018 [V extract, per audit 01]: train on Training, test on Walking.
- Rao et al., IJCAI 2020 (arXiv 2008.09435) [V PDF text]: "For BIWI, we use the full training set and the Walking" testing set — the skeleton line also inherits the cross-day protocol. (KGBD, single-session, is split by leaving "one skeleton video of each person" out — sequence-level, not frame-level.)
- Delécluse et al. 2026 [V extract]: BIWI treated as a cross-dataset evaluation; TVPR2/GODPR gallery = outgoing, probe = incoming.
- TVRID 2026 [V extract]: "queries from the IN direction are matched against OUT samples from the same camera".

Evidence against the claim as generalised:

- **TUM-GAID** is a Kinect-depth corpus whose authors *did* publish the same-session and the cross-session numbers side by side in 2014 [V extract]. The claim is true only for the BIWI and TVRID *protocols*; it is false for "the depth literature".
- Completeness cannot be proven: Munaro et al. 2014's own tables were not read here [UNVERIFIED]; Hafner 2022's exact split follows a cited division [UNVERIFIED]; dozens of skeleton-based BIWI papers exist and were not enumerated. A single BIWI paper with a within-recording random split would falsify "never".

Safe wording: "Under the BIWI protocol defined by Munaro et al. and followed by Wu et al., Haque et al., Karianakis et al., Rao et al., Hafner et al. and Delécluse et al., gallery and probe come from different days; under the TVRID protocol they come from opposite directions of the same passage. Neither protocol contains a frame-random within-recording split, so, to our knowledge, the inflated number has not been published for these corpora — our earlier 0.99 is that number. (TUM-GAID, by contrast, reported same-session and three-month probes side by side in 2014.)"

---

### Claim 6 — "A ladder that operationalises Kapoor & Narayanan's temporal-leakage/non-independence categories and extends Saeb's and Georgiev's two-way contrasts into a graded instrument with a cue floor"

**Verdict: WEAKENED.**

Graded (more-than-two-rung) leakage/protocol ladders already exist in several fields:

| Work | Rungs | Domain |
|---|---|---|
| Hofmann et al. 2014, DOI 10.1016/j.jvcir.2013.02.006 **[V]** | Six standardised probe configurations N/B/S/TN/TB/TS against one gallery — a covariate-graded protocol including time | Human gait, RGB + depth + audio |
| Del Pup et al., arXiv 2505.13021 (May 2025) **[V]** | "five cross-validation settings", >100,000 models; subject-based and nested schemes vs the rest; "overestimated performance claims" | EEG (clinical, not identity) |
| Parvan, arXiv 2605.01548 (May 2026) **[V]** | "Random Split Fallacy"; progressively realistic protocols — intra-session, cross-session, long-term, closed-set, open-set | ECG biometrics |
| Woo, arXiv 2605.06359 (May 2026) **[V in 01]** | Frame-level → temporal → scene-level | Vision, non-identity |
| Adam et al., SeaTurtleID2022, WACV 2024, DOI 10.1109/WACV57701.2024.00699 **[V in 01]** | Random → time-aware closed → time-aware open | Animal re-ID |
| Li et al. 2021 **[V extract]**; Yagis et al. 2021; Miller et al. 2020 (3DOF-only 19.5 %) **[V]** | Null/floor constructs: block relabelling, random labels, reduced-channel floor | EEG, MRI, VR |
| Kapoor & Narayanan 2023, DOI 10.1016/j.patter.2023.100804; Saeb et al. 2017, DOI 10.1093/gigascience/gix019; Georgiev et al. 2022, DOI 10.1145/3488932.3517388 **[V in 01]** | Taxonomy; record-/subject-wise; contiguous/random | Framing works the claim already cites |

Why not REFUTED: none of these combines a graded temporal ladder with a swept guard band at matched n_train *and* a metadata-only floor, for identification from video. But "graded instrument" and "operationalises the taxonomy" are not novel moves in themselves; ECG-biometrics-bench (May 2026) already operationalises a random-split fallacy into a protocol ladder for a biometric.

Safe wording: "Graded evaluation ladders have been proposed for EEG (five partitionings), ECG biometrics (intra-session → cross-session → long-term, closed/open set), intrinsic decomposition (frame → temporal → scene) and wildlife re-ID (random → time-aware), and TUM-GAID standardised a six-configuration probe protocol for gait in 2014. Ours is the identity-from-video instance, and adds two elements those ladders lack: a guard band swept at matched training size between the frame-random and session rungs, and a detector-metadata-only floor measured on every rung."

---

## 2. Search ledger (what was actually run)

Local: `ls extracts/`; grep of all `.txt/.xml` for guard-band/embargo/dead-zone/purged terms (hits only in unrelated PMC XMLs) and for "bounding box"; `tumgaid_preprint.txt` (Table 6, §5.1 protocol, session dates); `multigait_2609.01036.txt` (lines 719, 895–915, 943); `delecluse2026.txt` (datasets, sensors, Table I); `wu2017.txt`, `haque2016.txt`, `hafner.txt`, `tvrid2026.txt`, `andersson.txt`; Li et al. TPAMI full text (`webfetch-1788734689790-b9m8m9.txt`) grepped for temporal-distance analyses; Rao et al. 2020 PDF converted with pdftotext.

Web: WebSearch ×4 (guard band/temporal gap for person ID; bbox-only re-ID; frame-level vs subject-level split human identification; HAR temporal distance between train/test windows). arXiv API ×2 (bounding box + identification + shortcut/metadata; temporal gap + identification + split). arXiv abs pages: 2604.04485, 2604.09690, 2505.13021, 2605.01548, 2511.13944, 2607.23542, 2502.10195, 2201.02110, 2604.14161, 2008.09435 (+PDF). Crossref ×2 (TUM-GAID JVCIR record; Madden & Piccardi — not found). Europe PMC ×3 (Miller 2020 search + full text PMC7567078; Paolanti 2018). Semantic Scholar ×2 — HTTP 429, not retried.

Not found / not run (would be the next hostile probes if budget allowed): a BIWI paper using a within-recording random split (would falsify claim 5's "never"); a within-recording accuracy-vs-gap curve in HAR or endoscopy video; Pohjankukka's dead-zone radius sweep; Lohr & Komogortsev's test-retest-interval analysis (EKYT abstract silent [UNVERIFIED]); Madden & Piccardi height biometric DOI.

## 3. Unverified items (do not cite without checking)

- Madden & Piccardi, height as session-based biometric — DOI not resolved.
- DGHEI original paper (Hofmann, Bachmann, Rigoll, BTAS 2012) — `dghei2012.pdf` in extracts is not a valid PDF; numbers above are from the JVCIR 2014 preprint text.
- Hammerla & Plötz 2015 — whether any figure plots accuracy against temporal offset.
- Glazner et al. 2511.13944 — domain and inflation numbers.
- Lohr & Komogortsev 2201.02110 — test-retest interval analysis.
- Munaro et al. 2014 own BIWI tables; Hafner 2022 exact BIWI split.
- TUM-GAID: whether any later paper reports a *frame-random* split on it (not searched).

## 4. Notes for the orchestrator

- Add Hofmann et al. 2014 (TUM-GAID) to the C1 related-work paragraph and to Table "prior two-endpoint comparisons"; it is a depth corpus, it is human, it is vision, and it has been public since 2012. Karianakis 2018 already uses TUM-GAID [V in 01], so a depth referee will know it.
- Re-describe MultiGait's single-session number as walk-level within-session (R2-type), not frame-random; this strengthens the manuscript's R0 novelty but changes the sentence "MultiGait supplies the two endpoints".
- Cite Scagnetto 2026 (ECG constant-gallery temporal stress test) and Parvan 2026 (ECG-biometrics-bench) next to the guard-band sweep so "first in any modality" is replaced by "first within-recording".
- Cite Liciotti 2017 / Paolanti 2018 (top-view anthropometric scalars) next to the 13-scalar floor.
- Keep every "first" scoped: within-recording; detector-metadata-only; frame-random rung; BIWI/TVRID protocols (not "depth literature").
