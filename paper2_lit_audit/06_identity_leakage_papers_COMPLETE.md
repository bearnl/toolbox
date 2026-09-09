# Depth/thermal/radar identity-leakage papers + split-leakage evidence (COMPLETE sub-agent list, verbatim, 2026-09-06)

Method: WebSearch budget exhausted; discovery via OpenAlex/Crossref/arXiv; every DOI resolved via doi.org and confirmed against the Crossref record; publisher pages read where not bot-blocked. No DOI guessed.

## Core list (14)
DEPTH / RGB-D
1. Rao, Wang, et al., "A Self-Supervised Gait Encoding Approach With Locality-Awareness for 3D Skeleton Based Person Re-Identification," IEEE TPAMI 44(10), 2022, 10.1109/TPAMI.2021.3092833 — skeleton-gait re-ID on BIWI/IAS-Lab/KS20/KGBD; identity recoverable from joint trajectories alone.
2. Hafner, Bhuiyan, et al., "Cross-modal distillation for RGB-depth person re-identification," CVIU 216, 2022, 10.1016/j.cviu.2021.103352 — depth probes matched to RGB galleries (BIWI, RobotPKU); depth captures linkable to colour identities.
3. Hanisch, Muschter, et al., "Understanding Person Identification Through Gait," PoPETs 2023(1), 10.56553/popets-2023-0011 — 57 subjects full-body motion: identification 100 %; legs alone 100 %, head alone 60 %; heavy perturbation leaves >90 %.
4. Cao, Liu, et al., "Cross Vision-RF Gait Re-identification with Low-cost RGB-D Cameras and mmWave Radars," IMWUT 6(3), 2022, 10.1145/3550325 — ~92.5 % top-1 among 56.
THERMAL
5. Di, Riggan, et al., "Multi-Scale Thermal to Visible Face Verification via Attribute Guided Synthesis," T-BIOM 3(2), 2021, 10.1109/TBIOM.2021.3060641.
6. Jiang, Ye, et al., "WarmGait: Thermal Array-Based Gait Recognition for Privacy-Preserving Person Re-ID," IEEE TMC 25(2), 2026 (early access 2025), 10.1109/TMC.2025.3608447 — low-resolution thermal arrays marketed as privacy-preserving: 87.3 % recognition.
RADAR / RF
7. Fan, Li, et al., "Learning Longterm Representations for Person Re-Identification Using Radio Signals," CVPR 2020, 10.1109/CVPR42600.2020.01071 — RF-ReID: 100 identities over 15 days, mAP 59.5 % vs 41.3 % RGB; across clothing changes; RF framed as "more privacy-preserving".
8. Han, Xu, et al., "TWReID: Through-Wall and Free-Walking Person Re-identification Based on MIMO Radar," IEEE TIFS early access 2026, 10.1109/TIFS.2026.3729523 — mAP 86.8 %, CMC-1 96.5 % on 14 people.
LiDAR
9. Shen, Fan, et al., "LidarGait," CVPR 2023, 10.1109/CVPR52729.2023.00108 — 1,050 subjects.
EXTREMELY LOW RESOLUTION
10. Chai, Ng, et al., "Recognizability Embedding Enhancement for Very Low-Resolution Face Recognition and Quality Estimation," CVPR 2023, 10.1109/CVPR52729.2023.00960 — TinyFace (avg 20×16 px): rank-1 73.06 %.
EVALUATION PITFALLS / SPLIT LEAKAGE
11. **Georgiev, Eberz, et al., "Common Evaluation Pitfalls in Touch-Based Authentication Systems," ASIA CCS 2022, 10.1145/3488932.3517388** — all 30 surveyed papers overlook ≥1 pitfall; on 515 users/31 days, randomised (non-contiguous) training splits shift EER by 3.8 pp; combined pitfalls 8.9 pp.
12. **Melzi, Tolosana, et al., "ECG Biometric Recognition: Review, System Proposal, and Benchmark Evaluation," IEEE Access 11, 2023, 10.1109/ACCESS.2023.3244651** — same system: EER 0.14 % single-session vs 2.06 % multi-session — ~15× optimism from session-shared splits.
DEPTH IN HOSPITALS / BATHROOMS / AAL CLAIMING PRIVACY
13. Haque, Milstein, et al., "Illuminating the dark spaces of healthcare with ambient intelligence," Nature 585:193–202, 2020, 10.1038/s41586-020-2669-y — promotes depth/thermal/radio sensing in ICUs, ORs, elder homes as "privacy-protected computer vision" (body wording unverified, paywalled).
14. **Ballester, Gall, et al., "Depth-based interactive assistive system for dementia care," J. Ambient Intell. Humaniz. Comput. 15, 2024, 10.1007/s12652-024-04865-0** — ToiletHelp guides dementia patients in the lavatory; claims privacy is maintained "by sensing only depth maps"; 60 participants. (A direct premise-in-the-wild exemplar.)

## Additional verified candidates
- Meng, Fu, et al., mmWave multi-person gait, AAAI 2020, 10.1609/aaai.v34i01.5430 — 95 volunteers, 90 %.
- Cheng, Liu, "Person Reidentification Based on Automotive Radar Point Clouds," IEEE TGRS 60, 2022, 10.1109/TGRS.2021.3073664 — 98 %/91 % identification of 15/40; claims radar "preserves privacy".
- Pegoraro, Rossi, IEEE Access 9, 2021, 10.1109/ACCESS.2021.3083980 — 91.62 % identifying 3 of 8 walkers, unseen room.
- Guo, Pan, et al., "LiDAR-Based Person Re-Identification," CVPR 2024, 10.1109/CVPR52733.2024.01651 — rank-1 94.0 %.
- Hwang, Taha, et al., "Evaluation of the Time Stability and Uniqueness in PPG-Based Biometric System," IEEE TIFS 16, 2021, 10.1109/TIFS.2020.3006313 — 98 % single-session vs 87.1 % two-session.
- Bari, Gavrilova, "KinectGaitNet," Sensors 22(7), 2022, 10.3390/s22072631 — 96.91 % / 99.33 % from Kinect skeletons.
- Delécluse, Wannous, et al., TVRID competition, LNCS (ICPR 2026), 10.1007/978-3-032-31936-4_16.
- Zhu, Hong, et al., "Privacy-Protected Contactless Sleep Parameters Measurement Using a Defocused Camera," JBHI 28, 2024, 10.1109/JBHI.2024.3396397 — asserts defocus makes identification impossible.
- **Baldassini, Pistolesi, et al., "Don't call it privacy-preserving or human-centric pose estimation if you don't measure privacy," NeurIPS 38 (2025), 10.52202/085713-4346** — position paper: "privacy-preserving" HPE is assessed only by accuracy; privacy is never measured. (Frames exactly our complaint.)

## Coverage gaps flagged by the agent — **these are C1's gap**
- No 2020+ peer-reviewed paper found that specifically quantifies frame-random vs subject-disjoint inflation for VISION-based person identification (split-leakage evidence above is from touch, ECG and PPG biometrics).
- No empirical paper found that attacks identity from depth captured IN a bathroom or care-home deployment (#14 ToiletHelp and TVRID are the closest).
