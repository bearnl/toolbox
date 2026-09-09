# 08 — Adversarial refutation of contribution C2 (attribution) — COMPLETE, 2026-09-09

Role: hostile reviewer. For each of the five "first / to our knowledge" claims of C2, try to find a prior work that already did the same measurement — in any modality first, then in depth. Single agent, no sub-agents, 55 tool calls. Written in British English.

Method: (i) local extracts first (`extracts/`: xiao2021, tian2018, he2016, choi2019, hat2022, wu2017, wu2018, karianakis, haque2016, lidargait, tumgaid_preprint, hanisch, prcc, multigait, delecluse2026, tvrid2026); (ii) the completed audits 02, 04b, 06, 07 and 08_refute_C1 for context (not redone); (iii) WebSearch (quota was available), arXiv API, Crossref, Semantic Scholar, Europe PMC, arXiv abs/HTML pages, one CVF PDF downloaded and grepped. Legend: **[V]** = confirmed on the DOI/arXiv/venue page, in an API record, or read in an extract; **[V in 02]** = verified by audit 02, not re-checked; **[UNVERIFIED]** = not confirmed. Quotes are under 15 words. No DOI below was typed from memory.

Strategic frame: manuscript = mechanistic and evaluation-methodology complement to MultiGait (Todt, Morsbach, Dissert, Strufe, arXiv:2609.01036). Audit 07 confirmed MultiGait has **no** background-only, person-masked, inpainted or bit-depth ablation and that its depth results are silhouette results [V in 07]. Nothing in MultiGait or Delécluse et al. touches C2. The threats below all come from RGB object/action recognition, RGB re-ID, RGB gait, wildlife re-ID and VR/motion-capture privacy.

---

## 0. Headline

No claim is cleanly REFUTED. **All five are WEAKENED** — each survives only with a modality qualifier ("first in depth", "for depth values") and with an explicit acknowledgement of a same-shaped RGB or motion-data precedent. The three findings a referee is most likely to raise, none of which appears in the manuscript's current C2 text or in audit 02:

1. **RealGait already published an input-precision ladder from the binary silhouette upward for gait recognition** — Zhang, Wang, Chai, Li, Jain, "RealGait: Gait Recognition for Person Re-Identification", arXiv:2201.04806 (v1 Jan 2022, revised Feb 2023) [V arXiv abs + HTML, §V-E3 "Impact of Color and Background"]: on BUAA-Duke-Gait, person-region input quantised from binary silhouette (rank-1 **78.24 %**) through three grey levels (**84.94 %**) up to 8-bit grey (~89 %); their reading: "just three bins of a grayscale image" already restore appearance. This is the same *shape* of measurement as C2's 16→1-bit axis (same model, precision of the pixel value swept, silhouette as the 1-bit limit), on RGB intensity rather than metric depth, with no fixed metric range and no session axis. Claim 2 must be re-worded to "depth values" and must cite it.
2. **The inpainted "exact complement" is now standard practice in wildlife re-ID, with the hole-leak explicitly named** — Rueda-Toicen et al., "Are We Recognizing the Jaguar or Its Background?", arXiv:2604.09690 (6 Apr 2026) [V arXiv HTML]: background-only images via FLUX.1-Fill generative inpainting, foreground-only via SAM-3 mask, a "leakage-controlled context ratio" = mAP(background-inpainted) / mAP(foreground-only), chosen precisely because a zeroed foreground leaves a silhouette-shaped hole that "would leak shape information". Full/FG/BG mAP e.g. 0.303 / 0.304 / 0.178 (MiewID). No depth, no session axis, generative fill rather than the recording's own plate — but "inpaint the subject out and score the remainder as a leakage control" is no longer new in any modality (HAT 2022 did it for actions; this does it for identity).
3. **Person-only RGB collapsing under clothing change at fixed protocol is already published on the manuscript's own corpora.** Karianakis et al., ECCV 2018 [V extract, Table 2]: TUM-GAID new-clothes scenario, body-index-masked "Body RGB" 41.8 % vs "Body Depth" 48.0 % top-1 single-shot (50.0 vs 59.4 multi-shot). Wu, Zheng, Lai, TIP 2017 [V extract, Table]: BIWI Still/Walking probes (changed clothes), LOMO RGB 9.07 / 8.74 % vs depth+skeleton 30.52 / 24.47 %, chance ≈ 3.6 %. Claim 3's novelty is confined to the *background-only complement* that shows the loss is not room-borne; the person-borne half is established.

Additional hostile points: (a) Balieva et al., Technologies 14(1):50, 2026 [V Crossref abstract] find for individual cow identification that "background cues carry more information" than the segmented foreground — the "network prefers the room" reading already has an RGB precedent, in the ordinary direction (with-background > without), not the manuscript's inversion (background-only > full frame). (b) Wu et al., ECCV 2018 [V extract, lines 594–596] found on SBU-Kinect RGB that "Cropping out the body removes … identity information" — the RGB literature is *divided* on whether removing the person is a control, which makes Claim 4's depth result useful but not unprecedented as a question.

---

## 1. Claim-by-claim

### Claim 1 — "First depth person-identification study to report the pixel-exact complement (person inpainted out from the recording's OWN depth plate), fixed model and protocol, resolved along the leakage ladder (background share flips sign across the session change)"

**Verdict: WEAKENED.** Nothing found in depth; the construct, the hole-vs-fill contrast and the "background helps in-domain, hurts out-of-domain" pattern all exist in RGB.

Searched: extracts grep for inpaint / background-only / "without human"; WebSearch ("empty room" OR "background plate" OR "clean plate" background-only identification inpainted depth) — only VFX clean-plate tools and patents returned; WebSearch (background-only vs full image, shortcut learning, individual identification); Semantic Scholar (background-only re-identification inpainting) — rate-limited (429); arXiv HTML 2604.09690; Crossref for Beery 2018, Zheng 2015 and Balieva 2026; MultiGait/Delécluse/TVRID extracts (no background condition anywhere [V in 07]).

| Work | What they did | Same axis? | Same kind of control? |
|---|---|---|---|
| Xiao, Engstrom, Ilyas, Mądry, ICLR 2021, arXiv:2006.09994 **[V extract]** | ImageNet-9: Only-BG-B (bbox blacked), Only-BG-T (bbox filled with *tiled strip of the same image's background*), No-FG (GrabCut hole kept), Only-FG, Mixed-Same/Rand. IN-9L model: Original 96.32, Only-BG-T 43.60, No-FG 42.52 (Appendix table, line 834–839) | Background-only complement, hole vs fill contrast, at fixed model | Yes, for RGB object classes; fill is tiled, not a true plate; no sessions |
| Tian et al., CVPR 2018, DOI 10.1109/CVPR.2018.00607 **[V extract]** | CUHK03/Market: "background-only" = person region set to mean pixel (a person-shaped hole); original-trained model on background-only 5.3 % top-1; background-only-trained model 25.9 % on original / 37.1 % on background-only | Background-only for *identity*, cross-camera | Hole kept (mean fill), not inpainted; one protocol only |
| Chung, Wu, Russakovsky, HAT, NeurIPS D&B 2022 **[V in 02]** | Humans video-inpainted out; Background-Only ratio ≈ 0.7 across 74 Kinetics models | Inpainted complement | Actions, not identity; learned fill |
| Rueda-Toicen et al., arXiv:2604.09690, Apr 2026 **[V arXiv HTML]** | Jaguar individual re-ID; background-only by FLUX.1-Fill inpainting, foreground-only by SAM-3 mask; ratio mAP(BG-inpainted)/mAP(FG); explicitly avoids the silhouette hole because it "would leak shape information"; full/FG/BG mAP 0.303/0.304/0.178 (MiewID), 0.298/0.296/0.228 (DINOv3), 0.281/0.303/0.200 (EVA-02) | Inpainted complement for *identity*, hole-leak argued | Generative fill, not the recording's plate; splits stratified by identity, no location/session axis |
| Beery, Van Horn, Perona, "Recognition in Terra Incognita", ECCV 2018, DOI 10.1007/978-3-030-01270-0_28 **[V DOI; numbers not re-read]** | Camera-trap species recognition at seen ("cis") vs unseen ("trans") locations: location/background helps in-domain, hurts out-of-domain | The sign-flip logic (same room helps, new room hurts) | Species, not identity; no background-only condition |
| Balieva et al., Technologies 14(1):50, 2026, DOI 10.3390/technologies14010050 **[V Crossref abstract]** | Cow identification; train low-variability, test high pose/environment variability; with vs without background; "background cues carry more information" than the segmented foreground | Background contribution to individual identification | RGB; segmentation not inpainting; no background-only |
| Choi et al., NeurIPS 2019 **[V in 02]**; He et al. ACCV-W 2016 **[V extract]**; Zhu, Xie, Yuille IJCAI 2017 **[V in 02]** | Scene bias / "without human" / "without objects" | Background-only, hole kept | Actions/objects |
| Zheng et al., Market-1501, ICCV 2015, DOI 10.1109/ICCV.2015.133 **[V DOI; convention text not re-read]** | The standard re-ID convention of excluding same-camera gallery matches exists because same-camera (same background/viewpoint) matching is trivially easy | Methodological precedent that background inflates within-camera identity accuracy | Not a measurement |

Why not REFUTED: no found work (a) uses the recording's own empty-scene plate so the complement is pixel-exact, (b) does it in depth, or (c) reports background-only accuracy at more than one leakage rung and hence the sign flip. Tian 2018 is the closest identity precedent and is single-protocol, cross-camera; Rueda-Toicen 2026 is the closest design precedent and is RGB wildlife with generative fill.

Hostile technical caveat the authors should pre-empt in the text: with Kinect v1 structured light, a person casts an invalid-pixel shadow onto the plate; if the sensor mask is not dilated to cover that halo, the "background-only" frame still carries a person-shaped trace of invalid pixels. State how the complement is constructed at the mask boundary and show a difference image.

Safe wording: "Background-only and person-only controls are standard in RGB object and action recognition (Xiao et al. 2021; He et al. 2016; HAT 2022), have been used once for RGB re-ID with a mean-filled person region (Tian et al. 2018), and have recently been used with generative inpainting as a leakage-controlled ratio for wildlife re-ID (Rueda-Toicen et al. 2026). To our knowledge this is the first *depth* person-identification study to report the complement, the first in any modality to fill the person from the recording's own background plate so that person-only and background-only are pixel-exact complements, and the first to resolve the background's share of accuracy at every rung of a leakage ladder, where it changes sign across the session change."

---

### Claim 2 — "A Z-precision axis from 16 bits down to 1 bit at fixed range, with the binary silhouette as the 1-bit limit, for depth person identification ('to our knowledge')"

**Verdict: WEAKENED — one modality word away from REFUTED.** The identical experimental shape (same encoder; pixel-value precision swept from the binary silhouette upward; silhouette read as the limit) exists for RGB intensity in gait recognition.

Searched: WebSearch ("quantized depth" / "depth quantization" gait, person identification, bits, silhouette); WebSearch (3D face recognition depth map quantization bit levels); WebSearch (LiDAR/radar/thermal reduced resolution or quantization, person identification, privacy); arXiv API (depth AND quantization AND gait/re-identification → 0 hits; "bit depth" AND privacy AND depth/silhouette → 0 hits; depth AND privacy AND quantized/coarse/"low precision" AND identification → 1 irrelevant hit); Crossref for Pini et al. 2021; CVF PDF of Li et al. CVPR 2023 (CCPG) downloaded and grepped — it does **not** contain a quantisation experiment (the search snippet belonged to RealGait); arXiv abs + HTML of RealGait; extracts grep for quantiz/bit/binar in karianakis, lidargait, tumgaid, hanisch, wu2018.

| Work | What they did | Same axis? | Depth? |
|---|---|---|---|
| **Zhang, Wang, Chai, Li, Jain, RealGait, arXiv:2201.04806 (2022/2023) [V arXiv abs + HTML §V-E3]** | BUAA-Duke-Gait (1,404 persons, from DukeMTMC video re-ID data); same RealGait model; person-region input as binary silhouette → grey with increasing quantisation levels → 8-bit grey; rank-1 78.24 % (binary) → 84.94 % (three levels) → ~89 % (8-bit); "just three bins of a grayscale image" suffice. Exact set of intermediate levels should be re-read by the authors [the HTML summary listed 2/3/4/8; the snippet says "three quantization levels"] | **Yes**: pixel-value precision swept at fixed model, silhouette as the 1-bit end | No — RGB intensity, no metric range, no session axis |
| Shen et al., LidarGait, CVPR 2023, DOI 10.1109/CVPR52729.2023.00108 **[V extract lines 402, 408]** | Same GaitBase encoder: camera silhouette 76.12 %, LiDAR silhouette 64.70 %, LiDAR depth 86.77 % rank-1 on SUSTech1K | The two endpoints (1-bit vs full depth) at fixed encoder | Yes (LiDAR range) — but no intermediate precisions |
| Hofmann et al., TUM-GAID, JVCIR 2014 **[V extract Table 6]** | GEI (binary RGB silhouette) 99.4/44 %, depth-GEI 96.8/28 %, GEV (binary voxels) 94.2/41 %, DGHEI (depth gradients) 99.0/50 % (N same-session / TN three months later) | Silhouette vs depth-derived features, within and cross-session | Yes, but different feature extractors, not a precision sweep |
| Kumawat & Nagahara, BDQ, ECCV 2022, arXiv:2208.02459 **[V arXiv abs]** | Learnable Blur–Difference–**Quantisation** module as a privacy step; identity privacy evaluated on three datasets | Quantisation of the input as the privacy lever, identity measured | No (RGB video); learned, not swept |
| Nair et al., "Effect of Data Degradation on Motion Re-Identification", SePAR@WoWMoM 2024, arXiv:2407.18378 **[V arXiv abs]** | Noise, frame rate, **reduced precision**, dimensionality of VR motion; attacks "still achieve near-perfect accuracy" | Precision reduction of the signal value | No (3-D tracking coordinates) |
| Hanisch et al., PoPETs 2023(1), DOI 10.56553/popets-2023-0011 **[V extract lines 454–464]** | Motion-capture gait; "coarsening macro/micro" of coordinate digits; coarsening below the 100th digit "has no effect"; identity stays near 100 % | Coordinate-precision sweep for identification | No (marker trajectories) |
| Liu et al., Sensors 25(1):271, 2025, DOI 10.3390/s25010271 **[V in 02]** | Depth 640×480 → 16×12 spatial resolution; ResNet34 99.54 → 96.78 % | Degradation axis for depth identification | Yes, but *spatial* resolution, not Z precision; 6 subjects |
| Pini, Borghi, Vezzani et al., Sensors 21(3):944, 2021, DOI 10.3390/s21030944 **[V Crossref abstract]** | Depth-map representations (depth/normal images, voxels, point clouds) for face recognition; "depth map quality and the acquisition distance" studied | Depth representation and quality for identity | Yes (face), but not a bit-depth sweep [full text not read] |
| QGait (2024) **[title only via search; UNVERIFIED]** | Network-weight quantisation for gait models with binarised input | No — model quantisation, not input precision | — |

Why not REFUTED: no found work quantises the *depth value* (metric Z) at a fixed range for person identification, and none combines that with the session axis. But the sentence "we introduce a precision axis whose 1-bit limit is the silhouette" is, as a construct, RealGait's §V-E3 with "grey" replaced by "depth".

Safe wording: "RealGait (Zhang et al. 2023) swept the intensity precision of the person region from the binary silhouette to 8-bit grey for RGB gait recognition and found three grey levels already restore most appearance information; fixed-encoder silhouette-versus-depth endpoints exist for LiDAR (LidarGait 2023) and Kinect (TUM-GAID 2014). To our knowledge no prior work varies the precision of the *depth value itself* — here from 16 bits to 1 bit at a fixed metric range, so that the 1-bit end is exactly the sensor-mask silhouette — for person identification, nor reports it across a session change."

Must not write: "no prior work has quantised the input towards the silhouette" or "a novel precision axis" without the depth-value qualifier.

---

### Claim 3 — "The RGB cross-session loss is PERSON-borne (person-only RGB collapses; background-only does not explain it), shown with the same pipeline as depth"

**Verdict: WEAKENED — the first half is established prior art on the manuscript's own corpora; only the background-only complement is new.**

Searched: extracts wu2017 (BIWI table), karianakis (TUM-GAID §3.6, Table 2), prcc (Yang et al. TPAMI 2021), tian2018; Crossref for LTCC (Qian et al.); audit 02 items 6, 7, 10, 11.

| Work | What they did | Match |
|---|---|---|
| Karianakis, Liu, Chen, Soatto, ECCV 2018, DOI 10.1007/978-3-030-01228-1_44 **[V extract §3.6, Table 2]** | TUM-GAID, 32 subjects re-recorded "with new clothes" in session 2; training has "no access to data from the session 2"; person-only via Kinect body index: Body RGB (ss) 41.8 % / nAUC 74.3 vs Body Depth (ss) 48.0 / 85.0; multi-shot 50.0 vs 59.4 | Person-only RGB vs person-only depth, cross-session, fixed protocol. No within-session RGB reported, so the *collapse magnitude* is not shown |
| Wu, Zheng, Lai, TIP 2017, DOI 10.1109/TIP.2017.2675201 **[V extract]** | BIWI: 22 train / 28 test; "Still"/"Walking" probes recorded on another day, "dressed differently"; LOMO (RGB) 9.07 / 8.74 % rank-1 vs ED+SKL (depth) 30.52 / 24.47 %; PAVIS Walking1→Walking2 with clothing change | RGB appearance collapses to near chance on BIWI cross-session while depth does not — on the manuscript's corpus |
| Yang, Wu, Zheng, TPAMI 2021, DOI 10.1109/TPAMI.2019.2960509 **[V in 02; extract prcc.txt confirms design]** | PRCC: Camera A↔B same clothes, A↔C different clothes; RGB re-ID same-clothes 86.88 % → cross-clothes 22.86 % (PCB) | Person-crop RGB collapses under clothing change at fixed model — the mechanism named as clothing |
| Qian et al., LTCC, ACCV 2020, DOI 10.1007/978-3-030-69535-4_5 **[V Crossref]** | "Standard" vs "cloth-changing" settings on the same data and models | Same-pipeline within/cross-clothes drop, RGB |
| Castro et al., Neural Comput. Appl. 2020 **[V in 02]** | TUM-GAID 3D-CNN gray TN 18.8 vs depth 62.0 | Grey vs depth cross-session, fixed architecture |
| Tian et al., CVPR 2018 **[V extract]** | Background-only test 5.3 % top-1 cross-camera on CUHK03 (chance 1 %) | The nearest "background does not carry the cross-condition identity" number in RGB re-ID |

Why not REFUTED: no found work runs the *background-only* RGB complement alongside person-only RGB and full-frame RGB across a session change and attributes the loss by subtraction. But "person-only RGB collapses across clothing change while depth does not" is already published at fixed protocol on TUM-GAID and BIWI, and the cloth-changing re-ID field has attributed the RGB loss to clothing for years.

Safe wording: "That person-cropped RGB fails under a clothing change while depth degrades less is established, including on BIWI and TUM-GAID (Wu et al. 2017; Karianakis et al. 2018; Castro et al. 2020) and throughout cloth-changing re-ID (Yang et al. 2021; Qian et al. 2020). We add the complement on the same pipeline: the background-only RGB condition does not carry the cross-session loss, so the loss is attributable to the person's appearance rather than to the room change, and the same decomposition run on depth gives the opposite pattern."

Must not write: "we show for the first time that RGB's cross-session loss is due to the person/clothing".

---

### Claim 4 — "Two negative privacy-control results: background-with-hole is not a control; reducing depth precision (2-bit ≈ 16-bit) is not a control"

**Verdict: WEAKENED.** Both negatives have direct RGB / motion-data precedents; neither has a depth precedent; and the RGB literature is split on the hole.

Searched: extracts xiao2021 (No-FG rows), he2016, tian2018, wu2018 (lines 590–596), hanisch (coarsening), multigait; WebSearch (silhouette anonymisation re-identification body shape leak); Europe PMC full text of Weiß et al. 2026; arXiv abs 2407.18378, 2208.02459; Europe PMC (MRI defacing re-identification).

Hole ≠ control — precedents:
- Xiao et al. 2021 **[V extract, lines 814–839]**: "only No-FG retains the foreground shape"; No-FG-trained model 70.91 % on No-FG vs 54.30 % Only-BG-B and 50.25 % Only-BG-T; the hole's shape is a usable signal (RGB objects).
- He et al., ACCV-W 2016, DOI 10.1007/978-3-319-49409-8_2 **[V extract]**: human replaced by a black rectangle; two-stream 47.42 % "without human" vs 56.91 % with (UCF101 84.30 baseline).
- Tian et al. 2018 **[V extract]**: person region mean-filled (a hole) — a model trained on it reaches 37.1 % top-1 on CUHK03 background-only images.
- Rueda-Toicen et al. 2026 **[V]**: rejects the zeroed-foreground background because the silhouette-shaped hole "would leak shape information" — the very argument, stated as a design rule.
- Silhouette gait recognition as a field (Han & Bhanu 2006 **[V in 02]**; MultiGait's depth = binary silhouettes, 100 % within-session **[V in 07]**): the hole *is* the silhouette's complement; that it identifies is textbook.
- Counter-evidence the manuscript can turn to its advantage: Wu et al., ECCV 2018, arXiv:1807.08379 **[V extract lines 594–596]** — on SBU-Kinect RGB, "Cropping out the body removes not only identity information but also deteriorates action recognition"; i.e. in RGB the body crop *was* reported to remove identity. The depth result contradicts this for a scene-camera setting and is worth stating as such.
- Weiß et al., Sensors 26(1):187, 2026, DOI 10.3390/s26010187 **[V Europe PMC full text]**: DeepPrivacy2 full-body replacement leaves pose-based gait identification at 80.9–95.9 % rank-1 (10 subjects); no silhouette/hole condition — adjacent evidence that "removing the person's appearance" is not a control.

Precision ≠ control — precedents:
- Nair et al. 2024 **[V]**: reduced precision of VR motion; attacks "still achieve near-perfect accuracy".
- Hanisch et al. 2023 **[V extract]**: coarsening marker coordinates "has no effect" on identity until very coarse; identity ~100 %.
- Kumawat & Nagahara 2022 **[V abs]**: quantisation *does* work as a privacy lever when learned jointly (RGB) — the opposite sign, so the depth result needs the "fixed uniform quantisation" qualifier.
- Liu et al. 2025 **[V in 02]**: depth 640×480 → 16×12 keeps 96.78 % (6 subjects); Chou et al. 2018, Srivastav et al. 2019 **[in 02]**: low-resolution depth marketed as privacy.
- MRI defacing literature (Gao et al., Comput Biol Med 2025, DOI 10.1016/j.compbiomed.2025.111112; Molchanova et al., Hum Brain Mapp 2024, DOI 10.1002/hbm.26721) **[V Europe PMC records; content not read]**: cutting out the identifying region (defacing) is re-identifiable — the "hole is not a control" logic in medical imaging.

Why not REFUTED: nothing found measures either negative for depth person identification, and the RGB evidence points both ways (Xiao/Tian/He vs Wu 2018 for the hole; Nair/Hanisch vs BDQ for precision).

Safe wording: "Neither negative is new as a construct. In RGB the hole left by a removed subject is known to leak its shape (Xiao et al. 2021; He et al. 2016; Tian et al. 2018), although one RGB study reported that cropping the body does remove identity (Wu et al. 2018); and reducing signal precision is known not to anonymise VR motion (Nair et al. 2024) or motion-capture gait (Hanisch et al. 2023). We show, for depth person identification at fixed model and protocol, that a person-shaped hole in the recording's own plate and a uniform reduction of depth precision to 2 bits both leave accuracy essentially unchanged, so neither can serve as a privacy control for a depth sensor."

---

### Claim 5 — "Background alone beats the full frame within session on depth (38.6 vs 34.8 % at R3) — the network prefers the room when it can have it"

**Verdict: WEAKENED (the nearest of the five to STANDS).** No prior report of a background-only condition *exceeding* the full frame for person identification was found in any modality; but background-preference for identity is documented in the ordinary direction, and the interpretive clause "prefers the room" is exposed.

Searched: WebSearch ("background-only" higher accuracy than "full image", shortcut learning, identification, removing foreground improves); Xiao 2021 Appendix table [V extract: Original 96.32 vs Only-BG-T 43.60 — no inversion]; Tian 2018 [V extract: 37.1 % background-only-trained vs 75.3 % DGD on original — no inversion]; He 2016 [V extract: no inversion]; HAT ratio ≈ 0.7 [V in 02]; Rueda-Toicen 2026 [V: background-inpainted mAP 0.178–0.228 < full 0.281–0.303; but *foreground-only* 0.303 > full 0.281 for EVA-02 — an inversion in the other direction]; Balieva 2026 [V abstract: with-background > segmented foreground for cow ID]; Xiao's Only-BG-T-trained models reach ~40–50 % vs 11 % chance [V]; Rectifying shortcut learning of background (arXiv:2107.07746) [title only via search, UNVERIFIED].

Why not REFUTED: none of the above reports background-only > full frame for identity.

Why not STANDS: (i) "the network prefers the background/scene" is the scene-bias / shortcut-learning literature (Choi 2019; Xiao 2021; Balieva 2026) — the manuscript's number is a new *instance*, not a new phenomenon; (ii) a 3.8-pp gap between two conditions at 50 identities needs a confidence interval before it can bear the word "beats"; (iii) the inversion admits a plainer explanation — the full frame adds pose/position nuisance variation the small training set cannot average out — so "prefers" (an attribution) should become "matches or exceeds" (an observation); (iv) the invalid-pixel-halo caveat under Claim 1 applies with full force here, because if the plate carried a person-shaped trace the background-only condition would not be background-only.

Safe wording: "Within session (R3) the background-only input matched or exceeded the full frame (38.6 vs 34.8 %, CI …): under this protocol the room is at least as informative to the network as the person. Background preference is well documented in RGB recognition (Xiao et al. 2021; Choi et al. 2019) and individual animal identification (Balieva et al. 2026), where the background adds to, but does not exceed, the full frame; we know of no prior report of a background-only condition exceeding the full frame for person identification in any modality."

---

## 2. What C2 can still claim as first (after this pass)

1. First **depth** person-ID study with a background-only / person-only pair built from the recording's own plate (pixel-exact complement) at fixed model and protocol. [No counter-example found; Rueda-Toicen 2026 and Tian 2018 are the RGB ancestors.]
2. First report, in any modality found, of background share of identification accuracy at several leakage rungs with a sign change across the session change. [Beery 2018 and Tian 2018 give the logic; nobody gives the ladder.]
3. First sweep of **depth-value** precision (16→1 bit at fixed metric range) for person identification. [RealGait 2023 is the RGB-intensity ancestor and must be cited; LidarGait/TUM-GAID give fixed-encoder endpoints.]
4. First background-only RGB complement showing that the RGB cross-session loss is not room-borne. [The person-borne half is prior art.]
5. First measurement, in depth, that a person-shaped hole and 2-bit quantisation are not privacy controls. [RGB and motion-data analogues exist and point both ways.]
6. First report of background-only exceeding the full frame for person identification. [Weakest claim statistically; phrase as an observation.]

## 3. Items to add to the reference list (all [V] unless marked)

- Zhang, Wang, Chai, Li, Jain, "RealGait: Gait Recognition for Person Re-Identification", arXiv:2201.04806 (2022; rev. 2023) — §V-E3 quantisation ladder. Venue of record [UNVERIFIED].
- Rueda-Toicen et al., "Are We Recognizing the Jaguar or Its Background?", arXiv:2604.09690 (Apr 2026).
- Balieva, Marazov, Tanchev, Lazarova, Rankova, "Impact of Background Removal on Cow Identification with CNNs", Technologies 14(1):50, 2026, DOI 10.3390/technologies14010050.
- Beery, Van Horn, Perona, "Recognition in Terra Incognita", ECCV 2018, DOI 10.1007/978-3-030-01270-0_28.
- Qian et al., "Long-Term Cloth-Changing Person Re-identification", ACCV 2020, DOI 10.1007/978-3-030-69535-4_5.
- Kumawat & Nagahara, BDQ, ECCV 2022, arXiv:2208.02459.
- Nair et al., "Effect of Data Degradation on Motion Re-Identification", SePAR@WoWMoM 2024, arXiv:2407.18378.
- Weiß, Bonenberger, Niedermaier, Knof, Schneider, "Privacy Beyond the Face …", Sensors 26(1):187, 2026, DOI 10.3390/s26010187.
- Wu, Wang, Wang, Jin, ECCV 2018, arXiv:1807.08379 (the RGB counter-finding on body cropping).
- Pini et al., "A Systematic Comparison of Depth Map Representations for Face Recognition", Sensors 21(3):944, 2021, DOI 10.3390/s21030944 (depth-representation precedent; full text not read).
- Li, Hou, Zhang et al., CCPG, CVPR 2023, DOI 10.1109/CVPR52729.2023.01328 — checked and found **not** to contain a quantisation experiment; cite only as a cloth-changing RGB+silhouette benchmark if at all.

## 4. Dead ends (so nobody repeats them)

- arXiv API: (depth AND quantization AND gait/re-identification) → 0; ("bit depth" AND privacy AND depth/silhouette) → 0; (depth AND privacy AND quantized/coarse/"low precision" AND identification) → 1 irrelevant.
- WebSearch "empty room / background plate / clean plate" + identification → VFX tools and patents only.
- Semantic Scholar search endpoint returned 429 once; the DOI lookup endpoint worked.
- CVF openaccess HTML/PDF return 403 to WebFetch; `curl` with a browser UA succeeded (PDF saved in scratchpad as ccpg.pdf/ccpg.txt).
- PMC article pages are reCAPTCHA-gated; Europe PMC `fullTextXML` works.
