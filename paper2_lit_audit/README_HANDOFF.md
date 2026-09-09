# Paper-2 literature audit — STATE SAVED 2026-09-06 (session switching to Opus 5)

Purpose: before rewriting `paper/paper-depth/main.tex`, verify that the four
contributions (C1 protocol ladder, C2 attribution, C3 measurement validity /
operating point, C4 modality) and the thesis ("illegible to humans ≠ anonymous to
machines") are journal-level against the literature, then write the paper's
"essence" (thesis, defensible contributions, structure) for the author's approval.
The author asked for the audit FIRST, then the essence, then the rewrite.

## STATUS UPDATE 2026-09-09 — audit COMPLETE, essence written, decisions taken

Resumed under Fable 5.1 (the 09-06 session's last message was the model switch). Every row
below marked RELAUNCH has been relaunched and completed: `01_protocol_AUDIT_COMPLETE.md`,
`03_depthid_AUDIT_COMPLETE.md`, `04_privacy_thesis_AUDIT_COMPLETE.md`,
`04c_venue_tprivacy_COMPLETE.md` (IEEE T-Privacy ranked fifth; PoPETs first),
`05_standards_precedents_COMPLETE.md`, plus `07_multigait_delecluse_DEEPREAD.md` (primary-text
read: MultiGait's "depth" is a size-normalised silhouette, its single-session split is per
walk ≈ R3, 100 → 65–77 %, probes ≤ 64) and four adversarial refutation passes
`08_refute_{C1,C2,C3C4,THESIS}.md` whose "safe wording" lines are BINDING for every
"first / to our knowledge" sentence in the manuscript. **`ESSENCE.md`** is the one-page
statement written for the author; the author answered its five decisions on 2026-09-09
(PoPETs — T-Privacy rejected the 2023 paper; wave 22 approved; keep HI-RIDE; in-house
consented; IAS-Lab no response) — see ESSENCE.md §"Author's decisions" and HIRIDE_HANDOFF
§14.8. Next: run wave 22 on Nibi (`submit_wave22.sh`), then the main.tex rewrite on the
PoPETs template. The `extracts/` and `transcripts/` bulk stays in
`/Volumes/Workspace/study/hiride2-results/lit-audit/` (not in this repo).

## Status

| audit | brief | status | deliverable |
|---|---|---|---|
| C1 protocol leakage | briefs/01 | STOPPED mid-search (75 web actions) | 01_protocol_PARTIAL_notes.md (sources visited) — RELAUNCH |
| C2 attribution / background bias / quantisation | briefs/02 | **COMPLETE** | 02_attribution_AUDIT_COMPLETE.md (verbatim) |
| C3/C4 prior BIWI numbers, anthropometry, cohort | briefs/03 | STOPPED mid-search (94 web actions; had extracted several primary PDFs) | 03_depthid_PARTIAL_notes.md — RELAUNCH |
| thesis / premise-in-the-wild / legal / IEEE T-Privacy fit | briefs/04 | STOPPED mid-search (107 web actions) | 04_privacy_PARTIAL_notes.md — RELAUNCH |
| standards / precedents / objections | briefs/05 | v1 over-delegated and stalled (147 actions); v2 relaunch stopped at start | 05_standards_v1_PARTIAL_notes.md — RELAUNCH (self-contained brief) |
| venue fit (TIFS, T-BIOM, TOPS, PoPETs, S&P) — a sub-agent of v1 | — | **COMPLETE** | venue_fit_audit.md (verbatim) |
| standards / reporting checklist — a sub-agent of v1 | — | **COMPLETE** | 05_standards_CHECKLIST_COMPLETE.md (ISO 19795, Rule of 30, cluster bootstrap, REFORMS, leakage taxonomy, Chaibub Neto permutation, McNemar, MDE — each with a Methods sentence) |
| non-depth modalities prior art — a sub-agent of the thesis audit | — | **COMPLETE** | 04b_nondepth_modalities_COMPLETE.md — **contains the MultiGait pre-emption (read first)** |
| venue fit B (PR, CVIU, JBHI, IMWUT, Access, NMI, npj) — sub-agent | — | **COMPLETE** | venue_fit_audit_B.md (IMWUT best of these; IEEE Access explicitly welcomes negative results; npj has the Nebbia/Engelmann split-artefact precedent but excludes small cohorts) |
| identity-leakage paper list + split-leakage evidence — sub-agent | — | **COMPLETE** | 06_identity_leakage_papers_COMPLETE.md — **states the C1 gap: no 2020+ paper quantifies frame-random vs subject-disjoint inflation for VISION person-ID; Georgiev ASIA CCS 2022 / Melzi 2023 / Hwang 2021 do it for touch/ECG/PPG** |
| premise-in-the-wild, legal framing, IEEE T-Privacy fit, 2026 venue scan | briefs/04 | STOPPED (grandchildren killed) | 04_privacy_PARTIAL_notes.md; premise exemplars already in hand: Ballester 2024 ToiletHelp ('sensing only depth maps'), Haque Nature 2020, Zhu JBHI 2024 defocus, Basile 2026 LiDAR 'anonymized', Zhu SenSys 2021 thermal array, Ryoo AAAI 2017 |

`transcripts/` holds the raw JSONL of every agent (do not load whole into a model
context); `extracts/` holds the pages/PDFs/text the agents fetched (e.g. tian2018.txt,
xiao2021.txt, hat2022.txt, wu2017.txt, lidargait.txt, tumgaid_preprint.txt,
marin2018.txt, liu2025.xml, tvrid2026.txt, delecluse2026.txt) — reuse them instead of
re-fetching.

## READ FIRST — the strategic finding

**MultiGait (Todt, Morsbach, Dissert, Strufe; arXiv:2609.01036, posted 1 Sep 2026 — Strufe's
group, the PoPETs "False Sense of Privacy" authors).** 199 people, 8 sensors incl. depth,
thermal, LiDAR, radar, WiFi; 4 viewpoints; 3 sessions; session-disjoint evaluation; depth
>98 % within-session → 60–80 % cross-session; explicit "false sense of privacy" framing.
It PRE-EMPTS the broad thesis ("privacy-friendly sensors still identify people across
sessions") at a scale we cannot match. What it does NOT do — and what therefore becomes
the paper's spine — is the MECHANISM and the DEPLOYMENT-FACING measurement: (i) the
protocol decomposition (the ladder + trivial-cue floor + adjacency measurement explaining
WHY within-session numbers are near-ceiling, which MultiGait observes but does not
dissect); (ii) attribution — room vs person, outline vs 3D shape, the inpainted exact
complement, the 16→1-bit precision axis, "masking the person out is not a control";
(iii) the operating point under a 2.5-s observation budget with a validity gate,
top-k, per-person identifiability (0–100 %), cohort-size dependence; (iv) the
pre-registered negative results on fusion. Reframe the essence accordingly: cite
MultiGait prominently as the scale result and position this paper as the mechanistic
and evaluation-methodology complement, not as the discovery that depth identifies.
Their cross-session 60–80 % on 199 people with modern models makes our 43–47 % on 28
look conservative, not wrong. Also cite: Nair et al. 2024 (reduced precision does not
stop VR-motion re-ID), Wu et al. ECCV 2018 (only cropping the whole body removes
identity), Hammerla & Plötz UbiComp 2015 (adjacent-sample CV inflation in HAR),
Chaibub Neto 2019 (subject-level permutation for identity confounding).

## Key findings so far (verified by the completed audits)

**C2 (attribution) — survives, narrowed.** Background-only / person-only / hole
controls are established in RGB (Xiao et al. ICLR 2021 "Noise or Signal" incl. the
hole-shape-leak finding; Tian et al. CVPR 2018 background-bias re-ID; HAT NeurIPS 2022
inpaints humans out of video). What remains ours: the first DEPTH person-ID study
reporting the pixel-exact complement (person inpainted from the recording's OWN plate)
at fixed model/protocol, resolved along a leakage ladder (background share of accuracy
flips sign across the session change); the Z-precision axis 16→1 bit at fixed range
with the silhouette as the 1-bit limit (not found anywhere — say "to our knowledge");
that the RGB cross-session loss is PERSON-borne; two negative privacy-control results
(hole ≠ control, precision reduction ≠ control). MUST NOT claim: novelty of background
ablations or of "depth is clothing-invariant" (Wu TIP 2017: depth 30.5 % vs RGB 9 %
cross-clothing on BIWI; Karianakis ECCV 2018; Castro 2020 on TUM-GAID), nor a general
"CNN reads outline not 3D shape" — contradicted at fixed encoder by LidarGait CVPR 2023
(+22 pp depth over silhouette), Hofmann 2014, Haque CVPR 2016 (4D RAM 45.3 % vs GEI
21.4 % on BIWI). Scope that finding to single-frame Kinect v1 / this architecture.

**Prior BIWI cross-session numbers a reviewer will compare against** (from the C2
report; the C3 audit was to complete this table): Wu 2017 depth shape+skeleton 30.52 %
Still / 24.47 % Walking single-shot (RGB LOMO ~9 %); Karianakis 2018 25.4 % single /
50.0 % multi-shot; Haque 2016 3D-RAM 30.1 % single, 4D-RAM 45.3 % multi; Munaro 2014
PCM+skeleton 27.4 / 42.9 %; **Delécluse et al. arXiv:2606.23230 (June 2026): depth-only
transformer 60.7 % rank-1 on BIWI, claiming depth "inherently preserves privacy"** —
the freshest instance of the claim the paper rebuts. Our ~19 % single-frame / 43–47 %
at 2.5 s (closed-set, 28-way) sits inside that range once protocol differences
(open-set gallery/probe, multi-shot) are stated.

**Venue.** PoPETs is the best fit (scope "measurement of privacy in real-world
systems"; fee-free; ~2-month decisions; 12-page body; precedents Hanisch et al. 2023
"Understanding Person Identification Through Gait" and 2024 "A False Sense of Privacy"
— an evaluation-methodology critique for biometric anonymisation, the closest analogue
to C1). T-BIOM second (reproducibility valued; Hou et al. 2023 silhouette-gait
evaluation study; Rosberg et al. 2026 identity leakage). TIFS in scope, no hook. ACM
TOPS excludes biometrics. IEEE Transactions on Privacy (the current template) NOT yet
assessed — part of the stopped brief 04.

## How to resume (Opus 5)

1. Relaunch briefs 01, 03, 04, 05 (files in `briefs/`, each prefixed by
   `00_shared_context.md`). Instruct each agent: no sub-agents; WebSearch quota can run
   out — fall back to WebFetch on known URLs and Crossref/arXiv; reuse `extracts/`; the
   `*_PARTIAL_notes.md` list sources already visited; deliver a complete structured
   report within ~50 tool calls even if some items stay [UNVERIFIED].
2. With all five reports + venue_fit_audit.md, write the ESSENCE (one page): thesis;
   contributions in their audit-narrowed form; structure; what will and will not be
   claimed; venue recommendation. Get the author's approval BEFORE touching main.tex.
3. Then the rewrite per HIRIDE_HANDOFF §3 / §11.3 / §14.6, tables from
   `hiride2-results/tables.tex`, caption sentence §14.2 on every mechanism table.
