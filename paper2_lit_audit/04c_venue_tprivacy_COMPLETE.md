# 04c — IEEE Transactions on Privacy assessment and FINAL venue recommendation (COMPLETE, 2026-09-09)

Method. WebSearch not used (quota assumed exhausted). Sources: saved extracts (`extracts/cs_cfp.html` = IEEE Computer Society calls-for-papers listing containing the TP call, saved 6 Sep 2026; `extracts/tp_works.json` = Crossref works filtered on container-title; `extracts/cfp27.html` = PoPETs 2027 CFP), plus live WebFetch on Crossref (works + journals), OpenAlex (source + works), Semantic Scholar (two DOIs), the ISSN portal, DOAJ, IEEE open-access APC page, IEEE Computer Society author-resources page, PoPETs 2027 authors page, Rutgers faculty page, arXiv abs 2609.01036. Blocked: IEEE Xplore (HTTP 418 on the about-journal page, the REST metadata endpoint and the DOI redirect), computer.org CSDL journal pages (JavaScript shell), SCImago (403), Clarivate Master Journal List (JS shell), dblp (bot-wall), IEEE Template Selector (418). Legend: [V] = read on the official/registry page or in a saved extract; [UNVERIFIED] = not confirmable within budget. Prior completed rankings (`venue_fit_audit.md`, `venue_fit_audit_B.md`) are taken as given and not re-verified here; their items are marked [V-prior].

---

## 1. IEEE Transactions on Privacy (TP) — fact sheet

### 1.1 Identity
- ISSN 2836-208X (electronic only). Publisher IEEE; ISSN portal lists "IEEE Computer Society"; start year 2024; frequency recorded as "Annual" (one volume per calendar year, continuous publication: vol 1 = 2024, vol 2 = 2025, vol 3 = 2026); Xplore punumber 10167710; archived in Portico (2024–2026). [V — Crossref `/journals` record; portal.issn.org record]
- IEEE Computer Society (CS) describes TP as "our second" Gold Open Access journal; CS author page: "The Computer Society publishes two fully gold Open Access (OA) journals, IEEE Open Journal of the Computer Society and IEEE Transactions on Privacy." [V — computer.org/publications/author-resources/peer-review/journals and /authors]

### 1.2 Official scope statement (verbatim, from the TP call for papers on the CS CFP listing) [V — extracts/cs_cfp.html]
- "TP provides a multidisciplinary forum for theoretical, methodological, engineering, and applications aspects of privacy and data protection."
- Privacy "is defined as the freedom from unauthorized intrusion in its broadest sense, arising from any activity in information collection, information processing, information dissemination, or invasion."
- Invited topics: "Data protection specification, design, implementation, testing, and validation"; "Information collection, processing, and dissemination"; "Significant advances in theoretical models"; "Engineering tools"; "Design frameworks and languages"; "Architectures"; "Infrastructures"; "Model-based approaches"; "Study cases"; "Standards"; "Contributions to solving privacy problems in various application domains such as healthcare are also accepted."
- Exclusion: "Purely theoretical papers with no potential application or purely developmental work without any methodological contribution or generalizability would not be in scope."
- Reading for this manuscript: a study that tests whether a deployed data-protection design (depth sensing "on an anonymity premise") actually protects, with a methodological contribution (protocol ladder, attribution) and a healthcare setting, is inside the letter of the scope ("testing, and validation"; "Study cases"; "healthcare"). Scope is not the problem; the published record is (Section 2).

### 1.3 Editor-in-Chief
- The inaugural "Editorial Introduction to the IEEE Transactions on Privacy" (vol 1, pp. 1–2, created 2024-03-11, DOI 10.1109/TP.2024.3359328) has a single author, Jaideep Vaidya, affiliation string "IEEE Computer Society, Newark, NJ, USA", ORCID 0000-0002-7420-6947. [V — Crossref work record]
- Rutgers Business School faculty page: Distinguished Professor of Computer Information Systems and Vice Dean; states he "proposed the IEEE Transactions on Privacy". [V — business.rutgers.edu]
- Inference: Vaidya is the founding EiC. The title "Editor-in-Chief" itself was not read on any reachable page → [UNVERIFIED as a title; strong inference].
- "Welcome to IEEE Transactions on Privacy!" (vol 1, pp. i–ii, created 2024-10-10, DOI 10.1109/TP.2024.3463068) is by Chris Clifton, affiliation "IEEE Transactions on Privacy, IEEE Computer Society". [V — Crossref] His role (steering committee / CS publications board) [UNVERIFIED].

### 1.4 Open access, licence, APC
- "TP is 100% open access ... All articles are published under Creative Commons licenses (either CCBY or CCBY-NC-ND), and the author retains copyright." [V — TP CFP]
- APC: IEEE's fully-OA journals list a 2026 APC of US$2,160 with "Some exceptions apply for certain titles"; TP is not itemised there. Discounts: 20 % for IEEE society members, 5 % for other IEEE members; none for students. [V — open.ieee.org APC page] OpenAlex records `apc_usd` = 1,050 for TP in 2024 and 2025 [V as an OpenAlex field; provenance of that figure unknown]. TP-specific APC therefore [UNVERIFIED]; plan for US$1,050–2,160.
- Not in DOAJ: DOAJ API returns zero journals for ISSN 2836-208X; OpenAlex `is_in_doaj` = false. [V]

### 1.5 Review model and timeline
- TP CFP: "Rapid and Vigorous Peer Review ... a peer review turnaround of 10 weeks for most accepted papers." [V — TP CFP] No independent confirmation (received/accepted dates live on Xplore, which blocked). Treat "10 weeks" as the publisher's claim.
- CS general rule: "Unless a double-anonymous review is requested, each article undergoes a single-anonymous peer review process." [V — CS authors page]
- Rolling submission via the IEEE Author Portal; no deadlines. [V — TP CFP]

### 1.6 Length and format
- CS general rule for Transactions: "The regular paper page length limit is defined at 12 formatted pages for Transactions ... including references and author biographies"; "Any pages or fraction thereof exceeding this limit are charged $220 per page." [V — CS authors page] Whether TP applies exactly this (or a TP-specific limit) [UNVERIFIED — TP author-information page unreachable]. Assume 12 double-column pages INCLUDING references.
- Template: CS says "Using an article template is required for journal submissions" and points to the IEEE Template Selector, which blocked (418). Which template TP mandates → [UNVERIFIED]. The class currently in the repo, `ieeetj.cls` (2022/11/09 V1.1), self-describes as "IEEE Open Journals ... created to match the design layout of journals resembling IEEE TMLCN" [V — file header read]. It is the IEEE fully-open-access-journal layout, so it is a plausible guess for a Gold-OA IEEE title, but nothing reachable confirms TP uses it.
- Abstract: CS rule "Regular/special-issue paper – 100 to 200 words". [V — CS authors page]

### 1.7 Article types; attitude to negative results and measurement studies
- The TP CFP names no article types beyond general submissions; no "negative results", "replication", "SoK" or "comments" category appears. [V — absence in TP CFP] CS-wide: journals publish peer-reviewed content only; IEEE DataPort and Code Ocean are offered for data/code. [V — CS authors page]
- Measurement studies ARE published: Verna et al. 2025 (Topics API crawl over "tens of thousands of websites") and Pham et al. 2025 (PII leakage in SSO flows, "30.1% of first-party domains leak PII"). [V — OpenAlex abstracts] So an empirical "does this control work in practice" paper is not alien to the journal.
- Negative results as a genre: nothing stated either way. [V — absence]

### 1.8 Indexing and metrics
- Not in DOAJ [V]. ISSN portal coverage lines: Crossref, EZB, OpenAlex, ROAD, Portico (and a "PubMed" line that is almost certainly a registry artefact) [V — portal.issn.org]. Scopus / Web of Science: no reachable evidence (SCImago 403; Clarivate MJL shell) → [UNVERIFIED]. A journal that started in 2024 with ~23 research articles cannot yet hold a Journal Impact Factor (a JIF needs at least two full indexed years) — this is reasoning, not a read statement.
- OpenAlex: 34 works, 43 citations total, two-year mean citedness 1.34, h-index 3, i10-index 1. [V — api.openalex.org/sources]

### 1.9 Output volume (Crossref, 9 Sep 2026) [V]
- 33 DOIs in total. Excluding covers, tables of contents, reviewer lists and the two editorials: 5 research articles in vol 1 (2024), 12 in vol 2 (2025), 6 in vol 3 (2026 to date) + 1 early-access item (Zhang et al., "Collaborative Log Anomaly Detection using LLM with Privacy-Constrained Retrieval", created 2026-09-07) → 24 research articles in ~30 months.

---

## 2. Has TP published empirical sensing / biometric-privacy / re-identification-risk work?

Short answer: no. All 24 research titles were scanned; the corpus is differential privacy, federated learning, cryptographic/e-voting protocols, k-anonymity, web measurement, GDPR/IoT policy enforcement, and HCI. Zero papers on cameras, depth, gait, body shape, silhouettes, biometric evaluation protocols, or identification attacks on sensor data. The six closest, with DOIs and years [all V — Crossref titles/years; abstracts via OpenAlex or Semantic Scholar where stated]:

1. Aldahiri, Stephanie, Khalil, Ng — "Privacy-Preserving Vertical Split Federated Learning for Face Recognition", vol 3, 2026, 10.1109/TP.2026.3693243. Biometric domain, but a training-architecture paper (VFL + split learning + P-SMPC + DP); abstract reports no identification-leakage measurement. [V abstract — S2/OpenAlex]
2. Barreteau, Le Carpentier, Regnier-Coudert, Moussaoui, Gourraud — "Utility and Privacy-Preserving ECG Anonymization via Stochastic Linear Mixing", vol 3, 2026, 10.1109/TP.2026.3689006. Synthetic "avatar" ECG database; "privacy metrics confirm that the avatar dataset is anonymous"; no re-identification attack is described. [V abstract — S2]
3. Wang, Wang, Xu, Wang — "Preserving Location Privacy for Buildings in Street View Images", vol 3, 2026, 10.1109/TP.2026.3666140. GAN obfuscation of images evaluated "against location inference attacks" — image privacy, non-biometric. [V abstract — OpenAlex]
4. Barezzani, De Capitani di Vimercati, Foresti, Ghirimoldi — "TA_DA: Target-Aware Data Anonymization", vol 2, 2025, 10.1109/TP.2025.3527461. Tabular k-anonymity / ℓ-diversity. [V abstract — OpenAlex]
5. Verna, Jha, Trevisan, Mellia — "Understanding Topics API in the Wild: Dubious Usage and Stale Adoption", vol 2, 2025, 10.1109/TP.2025.3615120. Web measurement study (precedent that TP accepts empirical "in the wild" measurement). [V abstract — OpenAlex]
6. Pham, Vo, Dao, Fukuda — "I Never Willingly Consented to This! Investigate PII Leakage via SSO Logins", vol 2, 2025, 10.1109/TP.2025.3558463. Web measurement of leakage (precedent for "leakage" framing, but web, not sensors). [V abstract — OpenAlex]

Consequences: (a) no in-journal precedent a reviewer or editor can anchor on; (b) the associate-editor / reviewer pool visible through the published record is DP/FL/crypto/web-privacy, not computer vision or biometrics — a paper whose core is split-protocol leakage, inpainting attribution and anthropometric validity would likely be reviewed by non-specialists (risk: superficial "why not a modern backbone / larger cohort" reviews with no appreciation of the ISO 19795 framing); (c) nothing in TP pre-empts the repositioned (mechanistic, evaluation-methodology) framing. The "false sense of privacy" line of work that the paper must position against lives at PoPETs (Hanisch 2024 [V-prior]) and arXiv (MultiGait, cs.CR + cs.CV [V]).

---

## 3. Ranking for THIS paper — PoPETs vs IEEE T-Privacy vs T-BIOM vs TIFS vs IMWUT

Paper profile used for judging: journal-length mechanistic audit + pre-registered negative results + privacy framing; closed-set ID; n = 28 cross-session (BIWI) with two supporting corpora (10 ids Azure Kinect; 88 ids TVRID); one obsolete CNN backbone at fixed protocol; a deployment-facing operating point (validity gate + 2.5-s aggregation, top-k, per-subject identifiability, cohort curve); explicitly NOT SOTA / open-set / skeletons. Positioned as the mechanistic and evaluation-methodology complement to MultiGait.

| Criterion | PoPETs | IEEE T-Privacy | T-BIOM | TIFS | IMWUT |
|---|---|---|---|---|---|
| Scope fit for "privacy control fails" audit | High — "Measurement of privacy in real-world systems", "Surveillance", IoT privacy [V-prior] | High on the letter of the scope [V]; none in practice (0/24) [V] | Medium — biometric evaluation study, privacy is a secondary frame [V-prior] | Medium — biometrics + surveillance in scope, no hook [V-prior] | Medium–high — sensing + "societal impact" [V-prior] |
| In-venue precedent for evaluation-methodology critique / identity leakage | Hanisch 2023, 2024 "A False Sense of Privacy" [V-prior] — closest analogue to C1 | None [V] | Hou 2023 (silhouette-gait evaluation), Rosberg 2026 (identity leakage) [V-prior] | Hirose 2022, TWReID 2026 [V-prior]; method papers | Cao 2022 RGB-D↔radar gait re-ID; RDGait 2024 [V-prior] |
| Tolerance for n = 28, old backbone, no SOTA | Good if framed as measurement + threat model; reviewers value threat model over SOTA | Unknown; non-CV reviewers may ask for scale | Low–medium: biometric reviewers will compare against Wu 2017 / Haque 2016 / Delécluse 2026 numbers and expect a modern encoder | Low: TIFS expects method novelty | Medium: UbiComp expects a system or new sensing; a retrospective audit of Kinect v1 data is an unusual IMWUT paper |
| Negative results welcome | Not stated; SoK rubric invites "challenges to commonly held assumptions" [V]; open-science section mandatory [V] | Not stated [V] | Not stated; "Reproducibility is highly valued" [V-prior] | Not stated [V-prior] | Not stated [V-prior] |
| Length | 12 main-body pp + 1 after Revise; refs, appendices, ethics/open-science/AI-use sections excluded [V] | ~12 pp incl. refs & bios (CS rule), $220/page over [V; TP-specific UNVERIFIED] | 10 pp regular, MOPC beyond [V-prior] | 13 pp initial / 16 revised [V-prior] | No limit, ~8–10k words appropriate [V-prior] |
| Artefacts / reproducibility | Artifact review + badges; open-science section mandatory [V] | IEEE DataPort / Code Ocean offered [V] | Artefacts encouraged, datasets expected available [V-prior] | Reproducibility encouraged [V-prior] | Not stated |
| Speed | ~2 months to decision; 4 deadlines/yr; rejected → skip one issue [V] | "10 weeks for most accepted papers" (publisher claim) [V] | Not stated | Not stated | ~8 weeks initial; deadlines 1 Feb / 1 May / 1 Nov [V-prior] |
| Cost / OA | Fee-free, CC BY 4.0 [V] | 100 % OA, APC US$1,050–2,160 [V/UNVERIFIED] | Hybrid; OA US$2,800 [V-prior] | Hybrid; OA US$2,800 [V-prior] | Fully OA; APC US$350 / 250 member [V-prior] |
| Indexing / visibility | Established (2015–), read by the exact community that argues "false sense of privacy" | Not in DOAJ [V]; Scopus/WoS UNVERIFIED; no IF yet; 43 citations to date [V] | Established IEEE title | Top IEEE title | Established ACM title |
| MultiGait collision | Highest: Strufe's group publishes at PoPETs; MultiGait posted 1 Sep 2026, the day after the PoPETs Issue 2 deadline (31 Aug 2026) [V dates; the inference that it is under review there is speculation] | None | Low | Low | Low |

Verdict.

1. **Best fit: PoPETs (Proceedings on Privacy Enhancing Technologies).** It is the only venue where (i) the exact argument the paper makes already has a published methodological precedent (Hanisch et al. 2024, "A False Sense of Privacy", an evaluation-methodology critique for biometric anonymisation [V-prior]); (ii) an empirical audit is an accepted genre ("Measurement of privacy in real-world systems" [V-prior]); (iii) length is decided on the main body only, with references and appendices outside the cap [V]; (iv) an artifact track and a mandatory open-science section reward exactly the pre-registration / split-manifest / inpainting-plate artefacts the paper has; (v) no APC. The MultiGait collision is manageable and arguably an asset: the same reviewer pool will already know the scale result and will read a paper that explains WHY within-session numbers are near-ceiling and WHAT the outline leaks as the natural complement — provided MultiGait is cited prominently and the paper never claims discovery of cross-session depth identification.
2. **Fallback: IEEE T-BIOM.** If PoPETs rejects (or if the authors prefer an IEEE Transactions), T-BIOM has the reproducibility stance and two evaluation-protocol precedents [V-prior]. Cost: reframe as a biometric technology-evaluation study (ISO 19795 vocabulary; closed-set identification with cluster-bootstrap CIs; MultiGait and Delécluse 2026 as the comparison points), compress to 10 pages incl. references (MOPC beyond), move the privacy thesis to Discussion, and expect a demand to add one modern encoder at fixed protocol (which would also strengthen the paper).
3. **IMWUT** third: the RGB-D/mmWave gait re-ID precedents and the societal-impact hook fit, no page limit, cheap OA, next deadline 1 Nov 2026 [V-prior]; but a retrospective audit of a 2014 Kinect v1 corpus without a new system or deployment is atypical for UbiComp reviewers.
4. **TIFS** fourth: in scope, longest budget (13/16 pp), but no evaluation-methodology hook and a method-novelty expectation the paper deliberately does not meet.
5. **IEEE Transactions on Privacy** fifth for THIS paper. Its scope text fits and it is fast and OA, but: zero biometric/vision/sensing precedent in 24 research articles [V]; the visible reviewer pool is DP/FL/crypto/web; not in DOAJ and Scopus/WoS unverified [V/UNVERIFIED] — a young venue where a negative-results audit would gain little visibility among the people who deploy depth sensors "for privacy"; a tighter budget than PoPETs once references count against the 12 pages; and an APC the paper does not need to pay. It remains a legitimate last-resort IEEE OA option if speed is the overriding constraint. (IEEE Access, from audit B, is the other safety net: the only venue with an explicit "Negative Result" article type [V-prior], at a prestige cost.)

---

## 4. Template and folder: stay in `submissions-IEEE-on-Privacy` or move?

**Move.** Concretely:

- `papers/paper-depth/submissions-IEEE-on-Privacy/` (main.zip, title.tex/.docx/.zip, all dated 15 Mar 2025) is the package of the OLD paper — the one whose 0.99 frame-random accuracy the new manuscript explicitly retracts. Keep the folder as an archive; do not build the new paper in it. Whether that package was actually submitted to TP in March 2025 is not recorded in the files [UNVERIFIED]; if it was, resubmitting a paper that reverses its own headline number to the same journal is an avoidable complication, and one more reason to change venue.
- `ieeetj.cls` is the IEEE Open Journals class ("journals resembling IEEE TMLCN") [V]. It is not the PoPETs template (mandatory `\documentclass[sigconf]{acmart}` PoPETs 2027 build, "mandatory use or desk reject" [V]) and not the T-BIOM template (IEEEtran journal). Even for TP its applicability is unverified. Do not carry it forward.
- Create `papers/paper-depth/submissions-PoPETs-2027/` with the PoPETs 2027 LaTeX template; write the new `main.tex` there (the current `main.tex`, 7.1k words incl. markup, 10 figures, 0 tables, 94 BibTeX entries, is the old paper and is being rewritten anyway). Keep an IEEEtran-journal build script ready for the T-BIOM fallback; the body text ports with minor changes (reference style, figure widths).

---

## 5. Concrete consequences of the primary choice (PoPETs), with the fallback noted

### 5.1 Length
- Cap: 12 typeset main-body pages at submission, 13 after a Revise decision; references, acknowledgements, clearly marked appendices, and the mandatory ethics / open-science / AI-use sections do not count [V]. In acmart sigconf two-column, 12 pages with roughly eight display items is on the order of 8,000–9,000 words of body text (rough estimate from typical acmart density; not a measured figure).
- What must go to appendices (unlimited but not guaranteed to be read) or the artefact: full ladder tables for all three corpora; the complete guard-sweep at matched n_train; per-subject identifiability table; the 16→1-bit quantisation sweep; all fusion negative results beyond a one-table summary; the 13-scalar trivial-cue floor details; sensor-clipping statistics behind the 74 % figure.
- What stays in the body: one ladder figure (depth 98→62→35→7 %, RGB 99.99→92→80→6 %, floor 89 %), one attribution figure (complement / hole / silhouette / normalised, within- vs cross-session), one operating-point figure (single frame 19 % → gate 28 % → gate + 2.5 s 43–47 %, top-3 81 %, cohort curve K = 4→28), one modality table (clothing +45 pp within, ~0 across), one negative-results table.
- Fallback T-BIOM: 10 pages INCLUDING references [V-prior] → roughly 30 % less body than PoPETs; supplementary material carries the appendices.

### 5.2 Structure (PoPETs conventions)
1. Introduction: the premise in the wild (depth "for privacy" in bathrooms/care rooms; exemplars from brief 04), MultiGait cited up front as the scale result; contribution list framed as mechanism + evaluation methodology, never as discovery.
2. Threat model and setting (closed-set, cooperative-free, 2.5-s budget; what is NOT modelled: open-set, skeletons).
3. Protocol ladder R0→R4 (C1) with the trivial-cue floor and the adjacency/leakage explanation; ISO 19795 / Kapoor–Narayanan vocabulary from checklist 05.
4. Attribution (C2): exact complement, hole, silhouette, normalisation, Z-precision axis; "masking the person out is not a control".
5. Measurement validity and operating point (C3): metric anthropometry, clipping, validity gate, aggregation, top-k, per-subject identifiability, cohort curve.
6. Modality (C4): clothing accounts for RGB's within-session edge; segmenter equivalence.
7. Pre-registered negative results (fusion, multi-frame, normals, range rescaling, gait untestable, duplication-drop confound) — one section, one table, hypotheses and falsifiers stated as pre-registered.
8. Discussion: illegible ≠ anonymous; what a depth deployment should now claim; limits (n = 28, one backbone, Kinect v1).
9. Mandatory sections: Ethical considerations (BIWI RGBD-ID and TVRID are public research corpora — cite their licences; the in-house Azure Kinect corpus: consent and ethics approval status must be stated; state whether an IRB/REB reviewed the work [V requirement]); Open science (what is released, where, licence); AI use.
- Double-blind: strip authors, affiliations, funding, and self-identifying phrases ("our earlier version reported 0.99" must be written as "an earlier version of this study" and the self-citation anonymised) [V requirement].

### 5.3 Reproducibility artefacts (for the PoPETs artifact track)
- Split manifests for R0–R4 on all three corpora (frame indices, block boundaries, guard widths, recording/session ids); seeds; the matched-n_train guard-sweep script.
- The inpainted background plates and the code that produced the exact complement; the quantisation and normalisation transforms; the 13-scalar trivial-cue extractor.
- Metric anthropometry code (unprojection + sensor mask), the full-body gate, and the aggregation/top-k evaluation.
- Trained weights for the fixed backbone at each rung; cluster-bootstrap CI code (checklist 05).
- The pre-registration document with timestamps (hypotheses, falsifiers, minimum detectable effect) — this is the single most persuasive artefact for the negative-results section.
- Deposit with a DOI (e.g. Zenodo) and submit to artifact review for the Available/Functional/Reproduced badges; the artefact chairs' process is described on the PETS 2027 site [V].

### 5.4 Calendar (today 2026-09-09)
- PoPETs 2027 Issue 3: submission 30 Nov 2026 (firm); rebuttal 12–18 Jan 2027; notification 1 Feb 2027; revision 1 Mar 2027; camera-ready 15 Mar 2027. Issue 4: 28 Feb 2027 → notification 1 May 2027. PETS 2027, Delft, 19–24 July 2027. [V — cfp27]
- A reject forces skipping one issue (Issue 3 reject → earliest resubmission Issue 5 = PETS 2028 Issue 1) [V rule]; a Revise gives up to four months and one extra page [V].
- Fallbacks: IMWUT 1 Nov 2026 [V-prior] (earlier than PoPETs Issue 3 — only if the authors decide against PoPETs now); T-BIOM / TIFS / TP rolling.

---

## 6. Novelty / pre-emption check against the repositioned framing (venue-relevant)
- Nothing in the 24 TP research articles touches depth, gait, silhouettes, protocol leakage or attribution → no pre-emption at TP. [V]
- PoPETs holds the closest methodological precedent (Hanisch 2024) — a precedent to build on, not a pre-emption: it critiques anonymisation evaluation, not sensor-modality identification protocols. [V-prior]
- MultiGait (arXiv 2609.01036, 1 Sep 2026; cs.CR + cs.CV; abstract notes "poor cross-session generalisation of existing methods") [V] pre-empts the broad thesis, not the mechanism. If it is under review at PoPETs (speculation from dates), the manuscript must cite it as the scale result in the first page and state the complement explicitly; a reviewer who has both in hand should see two different contributions.

---

## 7. Verification ledger
[V]: ISSN 2836-208X; start 2024; publisher IEEE/CS; Xplore punumber 10167710; scope text and exclusion clause; "10 weeks" review claim; 100 % OA with CC BY / CC BY-NC-ND; CS "second Gold OA journal"; IEEE fully-OA APC US$2,160 (2026) with exceptions and member discounts; OpenAlex apc_usd 1,050 (as a record); not in DOAJ; ISSN-portal coverage list; OpenAlex counts (34 works, 43 citations, mean citedness 1.34, h = 3); Crossref 33 DOIs and per-volume research-article counts; the six closest TP papers (titles, years, DOIs, abstracts as stated); Vaidya as sole author of the editorial with CS affiliation and "proposed the IEEE Transactions on Privacy"; Clifton as author of the welcome note; CS 12-page-including-references rule and $220/page; CS single-anonymous default; PoPETs 2027 page rule, template rule, double-blind, ethics/open-science/AI-use sections, resubmission rule, all four issue calendars; ieeetj.cls header; contents and dates of `submissions-IEEE-on-Privacy`; MultiGait arXiv metadata.
[UNVERIFIED]: TP's own page-limit statement; TP's mandated template; TP-specific APC amount; Vaidya's formal title "Editor-in-Chief"; Clifton's role; Scopus / Web of Science indexing of TP; actual observed review times at TP; whether the March 2025 package was ever submitted to TP; whether MultiGait is under review at PoPETs; the word-count equivalents of page budgets (estimates only).
