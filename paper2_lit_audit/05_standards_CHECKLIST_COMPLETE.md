# Standards / reporting-guideline verification — COMPLETE (sub-agent report, verbatim, 2026-09-06)

Conventions: "Verified via" lists only URLs actually opened or API records retrieved. Anything not opened is flagged UNVERIFIED. (Agent note: web-search budget exhausted mid-task; later verification via publisher pages, Crossref/Europe PMC/OpenAlex/Semantic Scholar APIs and official style files.)

## 1. ISO/IEC 19795 family and its NPL ancestor

**1a. ISO/IEC 19795-1:2021** | Biometric performance testing and reporting — Part 1: Principles and framework. 2nd ed., 2021-05 (corrected 2024-09). Replaces 19795-1:2006.
- Verified via iTeh preview PDF (foreword, scope, TOC, terms 3.1–3.6) https://cdn.standards.iteh.ai/samples/73515/b592cb9773984da49a8d6b46a64696f0/ISO-IEC-19795-1-2021.pdf ; EVS catalogue https://www.evs.ee/en/iso-iec-19795-1-2021 .
- Relevant structure: 3.1 test subject; 3.2 test crew; 3.3 target population; 7.4 test subject selection; 7.5 test size (7.5.2 multiple transactions per subject; 7.5.3 requirements on test size); 8.4.11 offline non-mated trials when references are dependent; 9.6.6 closed-set identification; 9.7.1 longitudinal analyses; 9.11 uncertainty of estimates; 11 record keeping; 12 reporting (12.1–12.8); Annex A evaluation types; Annex B test size and random uncertainty; Annex C factors influencing performance.
- Rule of 3 / Rule of 30: verified indirectly via NIST (Iyer 2015) quoting 19795-1.
- Methods sentence: "Following ISO/IEC 19795-1:2021 (cl. 7.4–7.5, 9.11, 12; Annex B), we treat the test subject as the sampling unit, report test-crew size and composition, size the test against the Rule of 3/Rule of 30, attach uncertainty to every error-rate estimate, and report protocol details sufficient for reproduction."
- UNVERIFIED: body text on visit/time separation, habituation, demographics, Annex B CI methods.

**1b. ISO/IEC 19795-2:2007** | Part 2: technology and scenario evaluation. Verified via https://www.evs.ee/en/iso-iec-19795-2-2007 . Methods sentence: "Our BIWI/TVRID/in-house experiments constitute technology evaluations in the sense of ISO/IEC 19795-2:2007."

**1c. ISO/IEC 19795-10:2024** | Part 10: performance variation across demographic groups (2024-10-04). Verified via https://webstore.iec.ch/en/publication/101876 . Methods sentence: "Per ISO/IEC 19795-10:2024 we report the demographic composition of each corpus and note that the crews are too small for group-wise error estimates."

**1d. Mansfield & Wayman 2002**, Best Practices in Testing and Reporting Performance of Biometric Devices, NPL CMSC 14/02 v2.01. Verified record https://eprintspublications.npl.co.uk/2460/ (PDF is image-only). UNVERIFIED specifics; safe to cite for the Rule of 30 via Iyer.

## 2. Doddington et al. 2000 — origin of the Rule of 30
Doddington, Przybocki, Martin, Reynolds, "The NIST speaker recognition evaluation – Overview, methodology, systems, results, perspective," Speech Communication 31(2–3):225–254, 2000, DOI 10.1016/S0167-6393(99)00080-1. Rule statement verified via NIST slides https://www.nist.gov/system/files/documents/forensics/Iyer-Presentation.pdf : with 90 % confidence the true rate lies within ±30 % of the observed rate once ≥30 errors are observed (90%/30% → 30 errors; 95%/30% → 50; 90%/10% → 276); Rule of 3: zero errors in N iid trials → 95 % confidence error ≤ 3/N; iid is the key assumption. Methods sentence: "By Doddington's Rule of 30, an error rate is estimated to within ±30 % (90 % CI) only once ≥30 errors are observed; with 28 subjects this bounds the precision of any subject-level contrast."

## 3. Biometric subject-level bootstraps
- **Bolle, Ratha, Pankanti 2004**, "Error analysis of pattern recognition systems—the subsets bootstrap," CVIU 93(1):1–33, DOI 10.1016/j.cviu.2003.08.002 (Crossref/S2 verified). Resampling at the level of identities. Methods sentence: "Following Bolle et al. (2004), confidence intervals are formed by resampling subjects (the 'subsets bootstrap'), not individual frames, to respect within-subject dependence."
- **Bolle, Connell, Pankanti, Ratha, Senior 2005**, "The relation between the ROC curve and the CMC," AutoID'05, DOI 10.1109/AUTOID.2005.48.
- **Schuckers 2010**, Computational Methods in Biometric Authentication, Springer, DOI 10.1007/978-1-84996-202-5 (chapters 2–5).
- **Poh & Bengio 2007**, joint bootstrap, ICASSP 2007 DOI 10.1109/ICASSP.2007.366191; Poh, Martin, Bengio TPAMI 29(3):492–498, DOI 10.1109/TPAMI.2007.55.
- **Fogliato, Patil, Perona 2024**, "Confidence Intervals for Error Rates in 1:1 Matching Tasks," IJCV 132(11):5346–5371, DOI 10.1007/s11263-024-02078-8 — recommends Wilson with dependence-adjusted variance (or vertex/double-or-nothing bootstrap) for 1:1 all-pairs metrics and advises against naive Wilson, subsets and two-level bootstraps there; limited to 1:1, CIs for 1:N open. Relevance: our per-frame closed-set accuracy is one-way clustered by subject, so the cluster bootstrap stands; cite to show awareness.

## 4. Cluster bootstrap statistics
- **Field & Welsh 2007**, "Bootstrapping clustered data," JRSS-B 69(3):369–390, DOI 10.1111/j.1467-9868.2007.00593.x . Methods sentence: "All 95 % CIs are cluster (case) bootstrap intervals resampling subjects with replacement (Field & Welsh, 2007)."
- **Davison & Hinkley 1997**, Bootstrap Methods and their Application, CUP, DOI 10.1017/CBO9780511802843.

## 5. NIST FRVT / FRTE
- NISTIR 8271 (FRVT Part 2: Identification, 2019, DOI 10.6028/NIST.IR.8271); NISTIR 8238 (2018, DOI 10.6028/NIST.IR.8238); NISTIR 8280 (Part 3: Demographic Effects, 2019, DOI 10.6028/NIST.IR.8280). UNVERIFIED: time-lapse language; cite 8271 for the 1:N protocol generally.

## 6. Pineau et al. 2021
"Improving Reproducibility in Machine Learning Research (A Report from the NeurIPS 2019 Reproducibility Program)," JMLR 22(164):1–20. https://www.jmlr.org/papers/v22/20-303.html ; checklist v2.0 https://www.cs.mcgill.ca/~jpineau/ReproducibilityChecklist-v2.0.pdf . Methods sentence: "In line with the ML Reproducibility Checklist (Pineau et al., 2021), we state the exact number of runs (5 seeds per cell), the statistic reported, its variation, the splits, and the compute used."

## 7. NeurIPS Paper Checklist (2024→2026)
Live page https://neurips.cc/public/guides/PaperChecklist ; item 7 (statistical significance) text stable 2024–2026: error bars/CIs; state factors of variability; method (closed form/bootstrap); assumptions; SD vs SEM; no symmetric bars out of range. 2025 added item 16 (LLM usage). Methods sentence: "Following NeurIPS checklist item 7, every interval states what variability it captures (subjects via cluster bootstrap; seeds via 5 runs), how it was computed, and is asymmetric where the estimate is near a boundary."

## 8. Pre-registration in ML
NeurIPS 2020 Workshop on Pre-registration in ML → PMLR 148 (2021); NeurIPS 2021 workshop → PMLR 181 (2022); https://preregister.science/ . Methods sentence: "Hypotheses, falsifiers and the minimum detectable effect were fixed before the confirmatory runs, following the pre-registration format of the NeurIPS Pre-registration Workshops (PMLR 148, 181)."

## 9. REFORMS 2024
Kapoor, Cantrell, Peng, Pham, Bail, et al., "REFORMS: Consensus-based Recommendations for Machine-learning-based Science," Science Advances 10(18):eadk3452, DOI 10.1126/sciadv.adk3452 (PMC11092361). Items 1a, 2a–2e, 5c, 5f, 6a–6c, 7a–7c, 8a–8b. Methods sentence: "Reporting follows REFORMS items 5c, 6a–6c, 7a–7c and 8a–8b; tables are regenerated from stored per-frame predictions by an archived script (item 2e)."

## 10. Kapoor & Narayanan 2023 — leakage taxonomy
Patterns 4(9):100804, DOI 10.1016/j.patter.2023.100804 (PMC10499856). L1 (no clean separation; L1.4 duplicates), L2 illegitimate features, L3 (L3.1 temporal leakage; L3.2 non-independence — train/test from the same people/units; L3.3 sampling bias); 329 affected papers, 17 fields; model info sheets. Methods sentence: "Our split ladder targets Kapoor & Narayanan's L3.2 (same subject in train and test) and L3.1 (session/temporal) leakage; the leak-free rung is the only one about which claims are made."

## 11. Dietterich 1998
"Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms," Neural Computation 10(7):1895–1923, DOI 10.1162/089976698300017197. McNemar acceptable Type I error on a single test set. Methods sentence: "Paired contrasts on identical test frames use McNemar's test (Dietterich, 1998), complemented by a subject-level bootstrap of the accuracy difference."

## 12. Ojala & Garriga 2010
"Permutation Tests for Studying Classifier Performance," JMLR 11:1833–1863. Methods sentence: "Per-rung permutation nulls follow Ojala & Garriga (2010): identity labels are permuted at the split's grouping level and the pipeline is re-run."

## 13. Saeb et al. 2017 and Little et al. 2017
Saeb, Lonini, Jayaraman, Mohr, Kording, "The need to approximate the use-case in clinical machine learning," GigaScience 6(5):gix019, DOI 10.1093/gigascience/gix019 — ~45 % of reviewed papers used record-wise CV; median error 5.6 % (record-wise) vs 13.0 % (subject-wise). Little et al., "Using and understanding cross-validation strategies," GigaScience 6(5):gix020, DOI 10.1093/gigascience/gix020. Methods sentence: "Frame-random splits are reported only as the inflated reference rung (Saeb et al., 2017); claims rest on subject/session-level splits that match the deployment use case (Little et al., 2017)."

## 14. Chaibub Neto et al. 2019
"Detecting the impact of subject characteristics on machine learning-based diagnostic applications," npj Digital Medicine 2:99, DOI 10.1038/s41746-019-0178-x (PMC6789029). Subject-level restricted permutation: all records of a subject get the same shuffled label while record-wise splits are kept; a null far from chance shows the classifier identifies subjects. Methods sentence: "Identity confounding at the frame-random rung is quantified with the subject-level restricted permutation of Chaibub Neto et al. (2019)."

## 15. Agarwal et al. 2021; Bouthillier et al. 2021
Agarwal, Schwarzer, Castro, Courville, Bellemare, "Deep RL at the Edge of the Statistical Precipice," NeurIPS 2021 (bootstrap CIs over point estimates when runs are few). Bouthillier et al., "Accounting for Variance in ML Benchmarks," MLSys 2021 (randomise all sources of variation). Methods sentence: "With 5 seeds per cell we report bootstrap intervals rather than point estimates (Agarwal et al., 2021) and vary seeds jointly with split assignment (Bouthillier et al., 2021)."

## 16. Card et al. 2020
"With Little Power Comes Great Responsibility," EMNLP 2020, DOI 10.18653/v1/2020.emnlp-main.745. Methods sentence: "The ~8 pp minimum detectable effect was fixed a priori by a power analysis in the spirit of Card et al. (2020); smaller contrasts are reported but not claimed."

## 17. Demšar 2006
"Statistical Comparisons of Classifiers over Multiple Data Sets," JMLR 7:1–30.

## 18. 2024–2026 updates
ISO/IEC 19795-10:2024; NeurIPS 2025 checklist item 16; Fogliato IJCV 2024; TRIPOD+AI (Collins et al., BMJ 385:e078378, 2024, DOI 10.1136/bmj-2023-078378 — items 10, 12a, 12d, 14, 18c–f, 23a); CLAIM 2024 (Tejani et al., Radiology: AI 6(4):e240300, DOI 10.1148/ryai.240300; wording UNVERIFIED); Varoquaux & Cheplygina 2022 (npj Digit. Med. 5:48, DOI 10.1038/s41746-022-00592-y — repeated measures of one individual across train/test make the model recognise the individual); PROBAST+AI (Moons et al., BMJ 388:e082505, 2025). UNVERIFIED/not found: 2024–2026 NIST FRTE specifics; a 2025–2026 biometrics-specific leakage guideline.

## Priority citations for the Methods section
1. ISO/IEC 19795-1:2021 (+19795-2:2007 for "technology evaluation"). 2. Kapoor & Narayanan 2023 + REFORMS 2024. 3. Field & Welsh 2007 (+Davison & Hinkley) with Bolle 2004 as biometrics precedent and Fogliato 2024 as caveat. 4. Saeb 2017 + Chaibub Neto 2019. 5. Dietterich 1998. 6. Card 2020 + NeurIPS item 7. 7. Pre-registration workshops PMLR 148/181. 8. Doddington 2000 (Rule of 30).
