# Venue-fit audit (completed 2026-09-06 by a research sub-agent; verbatim save)

Method note from the agent: WebSearch quota was exhausted in its session, so venue
pages were reached at known official URLs (WebFetch / browser pane), candidate papers
located via Crossref filtered by each journal's ISSN, and every DOI verified by
loading https://doi.org/<DOI> and matching the title.

## 1. IEEE TIFS
- Scope: "information forensics, information security, biometrics, surveillance and
  systems applications" (https://ieeexplore.ieee.org/xpl/aboutJournal.jsp?punumber=10206).
- Negative results / replication: not stated; reproducibility encouraged (SPS page).
- Length: initial <= 13 double-column pages, revised <= 16; overlength $220/page beyond 10.
- OA: hybrid; IEEE 2026 hybrid APC $2,800.
- Review timeline: not stated.
- Verified relevant papers: Hirose et al., "Anonymization of Human Gait in Video Based on
  Silhouette Deformation and Texture Transfer", TIFS 17, 2022, 10.1109/TIFS.2022.3206422;
  Han et al., "TWReID: Through-Wall and Free-Walking Person Re-identification Based on
  MIMO Radar", TIFS early access 2026, 10.1109/TIFS.2026.3729523.
  Lead [UNVERIFIED]: "SecureReID: Privacy-Preserving Anonymization for Person
  Re-Identification", 2024, 10.1109/TIFS.2024.3356233.

## 2. IEEE T-BIOM
- Scope: biometrics incl. theory, applications, systems, surveys; position papers.
- "Reproducibility is highly valued"; artifacts encouraged; datasets expected available.
- Length: regular <= 10 pages, short <= 6, survey <= 16, position <= 8; MOPC beyond;
  optional double-anonymous.
- OA: hybrid (IEEE 2026 APC $2,800). Review timeline: not stated.
- Verified: Hou et al., "A Comprehensive Study on the Evaluation of Silhouette-Based Gait
  Recognition", T-BIOM 5(2) 2023, 10.1109/TBIOM.2022.3216857; Rosberg et al.,
  "Adversarial Attacks and Identity Leakage in De-Identification Systems: An Empirical
  Study", T-BIOM 8(2) 2026, 10.1109/TBIOM.2025.3596069.
  Lead [UNVERIFIED]: "Are Synthetic Datasets Reliable for Benchmarking Generalizable
  Person Re-Identification?", 2025, 10.1109/TBIOM.2024.3459828.

## 3. ACM TOPS
- Author guidelines: "Submissions on biometrics, steganography, and watermarking are
  currently out of scope" -> EXCLUDE. (Fully OA since 2026-01-01; <= 35 pages; 3-4 month
  first round.)

## 4. PoPETs / PETS
- Scope (CFP 2027): PETs and contexts; topics incl. "Measurement of privacy in real-world
  systems", "Machine learning and privacy", "Internet of Things privacy", "Surveillance".
  SoK papers solicited ("challenges to commonly held assumptions"). Mandatory open-science
  section; artifact submission requested.
- Length: <= 12 typeset pages main body (+1 after Revise); PoPETs LaTeX template
  (acmart sigconf); double-blind. Fee-free, CC-BY 4.0.
- Timeline: four deadlines/year (31 May, 31 Aug, 30 Nov 2026; 28 Feb 2027); decisions
  ~2 months after submission; accept/revise/reject; rejected papers skip one issue.
- Verified: Hanisch et al., "Understanding Person Identification Through Gait", PoPETs
  2023(1), 10.56553/popets-2023-0011; Hanisch et al., "A False Sense of Privacy: Towards a
  Reliable Evaluation Methodology for the Anonymization of Biometric Data", PoPETs 2024(1),
  10.56553/popets-2024-0008 (evaluation-methodology critique -- the closest analogue to
  our protocol argument).
  Leads [UNVERIFIED]: "Fantomas: Understanding Face Anonymization Reversibility" 2024
  (10.56553/popets-2024-0105); "BehaVR: User Identification Based on VR Sensor Data" 2025
  (10.56553/popets-2025-0022); "SoK: (Un)usable Privacy: the Lack of Overlap between
  Privacy-Aware Sensing and Usable Privacy Research" 2025 (10.56553/popets-2025-0026).

## 5. IEEE Security & Privacy (magazine)
- "please do not submit research papers"; 4,900-7,200 words, <= 15 references -> not for
  the study itself; a later perspective piece only. Verified: Maxion, "Reproducibility:
  Buy Low, Sell High", S&P 18(6) 2020, 10.1109/MSEC.2020.3005077; Shi et al., "Your
  Headset Is Listening", S&P 23(6) 2025, 10.1109/MSEC.2025.3603109.

## Caveats
No venue explicitly welcomes negative results/replications on its official pages;
PoPETs (SoK, open-science section) and T-BIOM ("Reproducibility is highly valued") come
closest. IEEE Transactions on Privacy (the authors' current template) was NOT covered by
this sub-agent; it was in the brief of the privacy-thesis auditor (stopped before report).

## Best fit (agent's verdict)
1) PoPETs -- empirical evaluation showing supposed privacy controls fail; matches
   "measurement of privacy in real-world systems"/AAL; publishes the closest precedents;
   fee-free; ~2-month decisions; 12-page body.
2) T-BIOM -- if framed as a biometric evaluation-protocol study; explicit reproducibility
   expectations and precedent (Hou 2023, Rosberg 2026). TIFS in scope but no hook.
3) Exclude TOPS and S&P magazine.
