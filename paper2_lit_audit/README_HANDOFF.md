# Paper-2 literature audit — STATE SAVED 2026-09-06 (session switching to Opus 5)

Purpose: before rewriting `paper/paper-depth/main.tex`, verify that the four
contributions (C1 protocol ladder, C2 attribution, C3 measurement validity /
operating point, C4 modality) and the thesis ("illegible to humans ≠ anonymous to
machines") are journal-level against the literature, then write the paper's
"essence" (thesis, defensible contributions, structure) for the author's approval.
The author asked for the audit FIRST, then the essence, then the rewrite.

## Status

| audit | brief | status | deliverable |
|---|---|---|---|
| C1 protocol leakage | briefs/01 | STOPPED mid-search (75 web actions) | 01_protocol_PARTIAL_notes.md (sources visited) — RELAUNCH |
| C2 attribution / background bias / quantisation | briefs/02 | **COMPLETE** | 02_attribution_AUDIT_COMPLETE.md (verbatim) |
| C3/C4 prior BIWI numbers, anthropometry, cohort | briefs/03 | STOPPED mid-search (94 web actions; had extracted several primary PDFs) | 03_depthid_PARTIAL_notes.md — RELAUNCH |
| thesis / premise-in-the-wild / legal / IEEE T-Privacy fit | briefs/04 | STOPPED mid-search (107 web actions) | 04_privacy_PARTIAL_notes.md — RELAUNCH |
| standards / precedents / objections | briefs/05 | v1 over-delegated and stalled (147 actions); v2 relaunch stopped at start | 05_standards_v1_PARTIAL_notes.md — RELAUNCH (self-contained brief) |
| venue fit (TIFS, T-BIOM, TOPS, PoPETs, S&P) — a sub-agent of v1 | — | **COMPLETE** | venue_fit_audit.md (verbatim) |

`transcripts/` holds the raw JSONL of every agent (do not load whole into a model
context); `extracts/` holds the pages/PDFs/text the agents fetched (e.g. tian2018.txt,
xiao2021.txt, hat2022.txt, wu2017.txt, lidargait.txt, tumgaid_preprint.txt,
marin2018.txt, liu2025.xml, tvrid2026.txt, delecluse2026.txt) — reuse them instead of
re-fetching.

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
