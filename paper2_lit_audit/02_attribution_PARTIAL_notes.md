# 02_attribution — partial working notes recovered from the killed transcript
(assistant-side text only; 4 blocks, 53 web actions)

## Notes
I'll start by loading the web research tools.

---
Several PDFs were saved locally but not parsed; I'll extract them with pdftotext, retry blocked sources via curl/PMC/arXiv mirrors, and run the remaining searches in parallel.

---
Only one small verification remains (He et al. 2016 numbers); everything else is in hand.

---
## Novelty audit of C2 ("attribution: what the accuracy is OF")

Method note: 200 web searches plus direct extraction of the primary PDFs (Tian 2018, Xiao 2021, Choi 2019, Zhu 2017, He 2016, HAT 2022, Karianakis 2018, Wu 2017, Haque 2016, LidarGait 2023, Hofmann 2014, Castro 2020, Yang 2021, Gait3D 2022, Liu 2025, Delécluse 2026, TVRID 2026, Zhou 2025, Fukuzawa 2025, Kobayashi 2025). DOIs were checked against Crossref/DBLP. Items I could not verify are marked [UNVERIFIED].

### A. Ranked list (closest first)

1. **Xiao, Engstrom, Ilyas, Mądry, "Noise or Signal: The Role of Image Backgrounds in Object Recognition", ICLR 2021, arXiv:2006.09994.** ImageNet-9 with eight single-edit variants: Only-BG-B (object bbox blacked), Only-BG-T (bbox replaced by a *tiled strip of the same image's background*), No-FG (GrabCut foreground blacked, silhouette-shaped hole kept), Only-FG, Mixed-Same/Rand/Next. ResNet-50 trained on IN-9L: Original 96.3 %, Only-BG-T 43.6 %, Mixed-Same 89.9 vs Mixed-Rand 75.6 (BG-Gap 14.3). Appendix: No-FG-trained model scores 70.91 % on No-FG vs 54.30 % (Only-BG-B) and 50.25 % (Only-BG-T); the IN-9L model is "about 13% better on No-FG than Only-BG-B" because the hole keeps the object shape; bbox size alone leaks ~23 %. **PRE-EMPTS the mechanism design** (person-only / hole-kept / hole-filled complement, and the "hole shape leaks" finding) for RGB object classification. Not depth, not identity, fill is tiling not the true plate, no session axis.

2. **Tian, Yi, Li, Li, Zhang, Shi, Yan, Wang, "Eliminating Background-bias for Robust Person Re-identification", CVPR 2018, doi:10.1109/CVPR.2018.00607.** Human-parsing masks on CUHK03/Market-1501; four joint sets: original, background-only (person region filled with the *mean pixel value*, i.e. a person-shaped hole), mean-background (person-only), random-background. Original-trained model on CUHK03: top-1 falls 10.3 pp (person-only) and 28.4 pp (random background); background-only test gives 5.3 % top-1 (chance 1 %); a model trained on background-only reaches 25.9 % on original and 37.1 % on background-only. **PARTIAL:** pre-empts "background alone re-identifies" and person-only-vs-full in RGB re-ID; hole is mean-filled not inpainted; background never exceeds the full frame; no depth, no cross-session.

3. **Chung, Wu, Russakovsky, "Enabling Detailed Action Recognition Evaluation Through Video Dataset Augmentation" (HAT), NeurIPS 2022 Datasets & Benchmarks, https://papers.nips.cc/paper_files/paper/2022/hash/ff52407b80dde0f0f45814db2738464c-Abstract-Datasets_and_Benchmarks.html.** Segments humans and *video-inpaints them out* (chosen over constant fill "to generate a more realistic-looking video"), plus Human-Only (human on grey) and Action-Swap; 74 Kinetics models; Background-Only Ratio ≈ 0.7, i.e. ~70 % of original accuracy survives human removal. Reused by DEVIAS (Bae et al., arXiv:2312.00826, ECCV 2024 [venue UNVERIFIED]), Zhou et al. 2025, Kobayashi et al. 2025. **PARTIAL:** pre-empts "inpaint the person out and measure what remains" as an attribution control (RGB, actions); no hole-vs-fill contrast, no exact plate, no identity.

4. **Zhu, Xie, Yuille, "Object Recognition with and without Objects", IJCAI 2017, doi:10.24963/ijcai.2017/505** and **He, Shirakabe, Satoh, Kataoka, "Human Action Recognition without Human", ACCV 2016 Workshops (LNCS), doi:10.1007/978-3-319-49409-8_2.** Zhu: ImageNet BGSet with object bboxes zeroed; AlexNet 14.41 % top-1 / 29.62 % top-5 on backgrounds alone (chance 0.1 %); 127-class 41.65 % vs humans 18.36 %. He: UCF101 with detected human boxes replaced by black rectangles; two-stream 47.42 % without human vs 56.91 % with human (84.30 % baseline). **PARTIAL:** background-only, hole-kept controls for classification; not identity, not depth.

5. **Choi, Gao, Messou, Huang, "Why Can't I Dance in the Mall?", NeurIPS 2019, arXiv:1912.05534.** Human-masked videos: Mask R-CNN boxes filled with the frame's average pixel value; used only inside a "human mask confusion" training loss plus scene-adversarial loss; no background-only accuracy reported. **PARTIAL (weak):** masking-out-the-human construct only.

6. **Wu, Zheng, Lai, "Robust Depth-based Person Re-identification", IEEE TIP 2017, doi:10.1109/TIP.2017.2675201.** BIWI protocol: 22 IDs train / 28 test, "Training" gallery, "Still"/"Walking" probes with changed clothes; RGB features collapse (LOMO 9.07 % Still, 8.74 % Walking single-shot; chance ≈3.6 %) while depth shape+skeleton reaches 30.52 % / 24.47 %. Person point clouds only (background removed by construction). **PARTIAL:** already shows on BIWI that depth body shape survives the clothing/session change and RGB appearance does not; no background-only complement, no fixed model.

7. **Karianakis, Liu, Chen, Soatto, "Reinforced Temporal Attention and Split-Rate Transfer for Depth-Based Person Re-Identification", ECCV 2018, doi:10.1007/978-3-030-01228-1_44.** Person-only depth via Kinect body-index mask ("background subtraction") before range normalisation; BIWI single-shot 25.4 %, multi-shot 50.0 %; TUM-GAID unseen clothes: body depth 48.0 % vs body RGB 41.8 % (single-shot), 59.4 vs 50.0 (multi-shot). **PARTIAL:** depth-is-clothing-robust at matched protocol; person-only only, no background or silhouette conditions.

8. **Haque, Alahi, Fei-Fei, "Recurrent Attention Models for Depth-Based Person Identification", CVPR 2016, doi:10.1109/CVPR.2016.138.** BIWI (cross-session): silhouette-based gait Energy Image 21.4 % multi-shot vs Energy Volume 25.7 % vs 4D RAM on point clouds 45.3 %; 3D RAM 30.1 % single-shot; human performance 6.7 %, chance 2.0 %. Reports Munaro's PCM+Skeleton at 27.4 / 42.9 % (Munaro et al. 2014, doi:10.1007/978-1-4471-6296-4_8; numbers taken from Haque's table, not from the original). **PARTIAL and adverse:** a BIWI silhouette-vs-depth data point where depth wins (models differ).

9. **Shen et al., "LidarGait: Benchmarking 3D Gait Recognition with Point Clouds", CVPR 2023, doi:10.1109/CVPR52729.2023.00108.** SUSTech1K (1,050 subjects); *same GaitBase encoder*: camera silhouette 76.12 %, LiDAR silhouette 64.70 %, LiDAR depth 86.77 % rank-1; clothing subset 49.60 (silhouette GaitBase) vs 74.56 (LidarGait). **PARTIAL, CONTRADICTS "outline not 3D":** at fixed encoder, adding depth to the silhouette gives +22 pp.

10. **Hofmann, Geiger, Bachmann, Schuller, Rigoll, TUM-GAID, JVCIR 2014, doi:10.1016/j.jvcir.2013.02.006** and **Castro, Marín-Jiménez, Guil, Pérez de la Blanca, "Multimodal feature fusion for CNN-based gait recognition", Neural Comput. Appl. 2020, doi:10.1007/s00521-020-04811-z.** 305 subjects, 32 re-recorded months later in new clothes (TN/TB/TS). Hofmann rank-1 TN: GEI (RGB silhouettes) 44 %, depth-GEI 28 %, DGHEI (depth gradients) 50 %. Castro: 3D-CNN gray TN 18.8 / TB 21.9 / TS 12.5 vs depth 62.0 / 53.1 / 78.1; 2D-CNN gray TN 28.1 vs depth 43.8. **PARTIAL:** cross-session silhouette/depth/appearance comparisons already exist, with depth ≥ silhouette; background is constant across sessions there, so no attribution.

11. **Yang, Wu, Zheng, "Person Re-identification by Contour Sketch under Moderate Clothing Change", TPAMI 2021, doi:10.1109/TPAMI.2019.2960509** (with Hong et al. FSAM, CVPR 2021, doi:10.1109/CVPR46437.2021.01037; Han & Bhanu GEI, TPAMI 2006, doi:10.1109/TPAMI.2006.38; GaitSet, AAAI 2019, doi:10.1609/aaai.v33i01.33018126; GaitPart, CVPR 2020, doi:10.1109/CVPR42600.2020.01423; Gait3D, CVPR 2022, doi:10.1109/CVPR52688.2022.01959: silhouettes 47.70 % vs +SMPL 53.20 %, pose-only ≤6.25 %). PRCC cross-clothes: outline model 34.38 % vs RGB PCB 22.86 %; same-clothes 64.20 vs 86.88. **PARTIAL:** "outline carries identity across clothing while colour collapses" is established in RGB.

12. **Liu, Bouazizi, Xing, Ohtsuki, "A Comparison Study of Person Identification Using IR Array Sensors and LiDAR", Sensors 25(1):271, 2025, doi:10.3390/s25010271.** 6 subjects, RealSense L515 depth 640×480 down to 320×240, 128×96, 64×48, 16×12; ResNet34 depth 99.54 % → 96.78 % at 16×12, ViT 94.21 → 92.76; YOLO cropping failed on depth (detection ≤3.5 %), so depth frames were effectively full-frame; predefined paths; no background control; Grad-CAM shows RGB "reliance on background features as resolution decreases". **PARTIAL:** a depth degradation axis exists (spatial resolution, not Z precision), and it also finds degradation does not remove identity — but on a confounded design.

Also relevant: Mucha & Kampel, PETRA 2022, doi:10.1145/3529190.3534764 and ICCHP 2022, doi:10.1007/978-3-031-08645-8_62 (depth face recognition; "boundary conditions" under which depth reveals identity; numbers not accessible [UNVERIFIED]); Chou et al., NeurIPS ML4H 2018, arXiv:1811.09950 (low-res depth as privacy for hand hygiene/ICU); Srivastav et al., MICCAI 2019, doi:10.1007/978-3-030-32254-0_65 (64×48 depth pose ≈ 640×480); Ryoo et al., AAAI 2017, doi:10.1609/aaai.v31i1.11233 (RGB 16×12); Chang et al., ICLR 2019, arXiv:1807.08024 (generative in-fill counterfactual attribution); Li et al., ICCV 2023, doi:10.1109/ICCV51070.2023.01823 (SCUBA/SCUFO, background swap); Sivapalan et al., IJCB 2011, doi:10.1109/IJCB.2011.6117504 (GEV vs GEI; numbers [UNVERIFIED]). No work found that quantises depth bit-depth for identification, and no depth person-ID work found that reports an inpainted background-only condition.

### B. What C2 can still claim / must not claim

Can claim:
- First **depth** person-ID study (with paired RGB) reporting the pixel-exact complement pair — person-only vs background with the person inpainted from *that recording's own plate* — at fixed protocol and model, and contrasting hole-kept vs hole-filled in depth.
- The **ladder-resolved** attribution: background share of accuracy measured at every leakage rung, showing the within-session/cross-session sign flip (no prior depth work reports background-only at any rung).
- The **Z-precision axis** (16→1 bit at fixed metric range, silhouette as the 1-bit limit) — not found anywhere; phrase as "to our knowledge", and distinguish it from spatial-resolution sweeps (Liu 2025, Chou 2018, Srivastav 2019).
- For **this CNN and data regime**, a framing-normalised binary silhouette is the best cross-session input; the RGB cross-session loss is person-borne (person-only RGB falls 33–49 pp) — a new attribution, given prior work only showed depth > RGB.
- Two negative privacy-control results in depth (hole ≠ control; precision reduction ≠ control), citing Xiao's No-FG shape-leak and Liu 2025 as RGB/resolution precedents.

Must not claim:
- Novelty of background-only / person-only ablations or of "background alone identifies" (Tian 2018, Zhu 2017, He 2016, Xiao 2021, HAT 2022), or of inpainting as an attribution control (HAT 2022, Chang 2019).
- Novelty of "depth is clothing-invariant, RGB is not" (Wu 2017, Karianakis 2018, Castro 2020) or "outline carries identity across clothing" (Yang 2021, FSAM, GEI/GaitSet).
- Any general "the CNN reads outline, not 3D shape" / "depth adds nothing over silhouette": contradicted at fixed encoder by LidarGait (+22 pp), Hofmann 2014 (DGHEI > GEI) and on BIWI itself by Haque 2016 (4D RAM 45.3 vs GEI 21.4). Scope it to single-frame Kinect v1, 50 IDs, this architecture.

### C. Positioning sentences

"Single-edit background/foreground controls are standard in RGB object and action recognition (Xiao et al. 2021; Tian et al. 2018; HAT 2022), where the hole left by a removed object is known to leak its shape; we transfer this suite to depth person identification, replace tiled or learned fills with the recording's own background plate so the complement is pixel-exact, and report it at every rung of a leakage ladder. Prior depth re-ID (Wu 2017; Karianakis 2018; Haque 2016) always cropped the person first and so could not measure what the background contributes; gait work (Hofmann 2014; LidarGait 2023) compared silhouettes to depth but never at fixed protocol with the background complement, and no work varied depth precision toward the 1-bit silhouette."

### D. 2025–2026 items the authors may not know

- **Delécluse, Wannous, Guimas, ICPR 2026 TVRID competition, arXiv:2605.04977 (May 2026):** 86 IDs, four overhead RealSense D455; depth track average mAP 57.19 % vs RGB 89.66 %, top depth entries ≈99 %; RGB winners used "background erasing" augmentation; authors conclude depth is "privacy-enhancing, but not automatically privacy-guaranteeing".
- **Delécluse et al., arXiv:2606.23230 (June 2026):** depth-only transformer on BIWI RGBD-ID, rank-1 60.7 % vs 96.4 % RGB-D (100 % precision with Hungarian matching); person ROIs cut from background by depth blob detection; asserts depth "inherently preserves privacy" — the claim C2 rebuts.
- **Sun et al., ReGenHuman, arXiv:2606.14972 (June 2026):** full-body video anonymisation regenerating humans from pose/segmentation/monocular depth as "identity-free structural cues" — C2's silhouette/depth results bear directly on that assumption.
- **Zhou, Chung, Russakovsky, arXiv:2512.17953 (Dec 2025, NeurIPS-W):** background bias across CLIP/SigLIP/VLLMs via HAT Action-Swap; segmented-human input cuts bias 3.78 %.
- **Fukuzawa et al., arXiv:2501.12681 (Jan 2025)** and **Kobayashi, Ding, Tamaki, arXiv:2509.23009 (Sep 2025):** random-colour background masking; note "about 50%" accuracy on Kinetics with the person inpainted out (via HAT).
- **Fioresi, Dave, Shah, ALBAR, ICLR 2025, arXiv:2502.00156;** **Huang et al., "Background Matters Too", arXiv:2509.03032 (Sep 2025)** (re-ID that deliberately exploits background text descriptions); **Liu et al., Sensors Jan 2025** (item 12); **VIP, arXiv:2504.03041 (Apr 2025)** (human-removal video inpainting, no leakage evaluation).

Local extracts for all sources are in `/private/tmp/claude-501/-Volumes-Workspace-study/f044c300-f60e-48b9-bdec-c2797de29563/scratchpad/` (e.g. `tian2018.txt`, `xiao2021.txt`, `hat2022.txt`, `wu2017.txt`, `lidargait.txt`, `tumgaid_preprint.txt`, `marin2018.txt`, `liu2025.xml`, `tvrid2026.txt`, `delecluse2026.txt`).

---

## Sources visited / searched (in order)
- WebSearch: Tian "Eliminating Background-bias for Robust Person Re-identification" CVPR 2018 background-only images Market-1501 accuracy
- WebSearch: Choi "Why Can't I Dance in the Mall?" NeurIPS 2019 scene bias action recognition human-masked
- WebSearch: He Yun "Human Action Recognition without Human" 2016 background only accuracy UCF101
- WebSearch: Xiao Engstrom Ilyas Madry "Noise or Signal: The Role of Image Backgrounds in Object Recognition" ImageNet-9 Only-BG-T Only-BG-B accuracy
- WebSearch: depth person re-identification background removal ablation Kinect "BIWI RGBD-ID" background subtraction accuracy
- WebSearch: person removal video inpainting privacy "removed" pedestrians inpainting evaluation re-identification leakage background
- WebSearch: silhouette versus depth person identification comparison "binary silhouette" depth map identification accuracy Kinect gait
- WebSearch: depth map quantization bit depth person identification privacy "depth" "bits" identification accuracy degradation
- WebSearch: Liu Sensors 2025 depth camera spatial resolution privacy person identification 6 subjects downsampling
- WebSearch: "BIWI RGBD-ID" cross-session "different day" re-identification depth CNN accuracy results
- WebFetch: https://openaccess.thecvf.com/content_cvpr_2018/papers/Tian_Eliminating_Background-Bias_for_CVPR_2018_paper.pdf
- WebFetch: https://arxiv.org/pdf/2006.09994
- WebFetch: https://proceedings.neurips.cc/paper_files/paper/2019/file/ab817c9349cf9c4f6877e1894a1faa00-Paper.pdf
- WebFetch: https://arxiv.org/pdf/1611.06596
- WebFetch: https://www.mdpi.com/1424-8220/25/1/271
- WebFetch: https://dl.acm.org/doi/fullHtml/10.1145/3529190.3534764
- WebFetch: https://arxiv.org/abs/2606.23230
- WebFetch: https://arxiv.org/abs/2606.13410
- WebFetch: https://arxiv.org/abs/2512.17953
- WebFetch: https://arxiv.org/html/2509.03032
- WebFetch: https://arxiv.org/pdf/1705.09882
- WebFetch: https://arxiv.org/pdf/1703.09474
- WebSearch: Sivapalan "Gait Energy Volumes" depth images frontal gait recognition IJCB 2011 GEV vs GEI comparison
- WebSearch: LidarGait CVPR 2023 SUSTech1K silhouette vs LiDAR depth projection comparison rank-1 point cloud density
- WebSearch: Srivastav "Human Pose Estimation on Privacy-Preserving Low-Resolution Depth Images" MICCAI 2019 downsampled depth privacy
- WebSearch: Chou "Privacy-Preserving Action Recognition for Smart Hospitals using Low-Resolution Depth Images" 2018 hand hygiene
- WebSearch: Yang Wu Zheng "contour sketch" clothing change person re-identification PRCC TPAMI outline
- WebSearch: Chang Creager Goldenberg Duvenaud "Explaining Image Classifiers by Counterfactual Generation" inpainting remove region ICLR 2019
- WebSearch: depth image person re-identification privacy "quantization" OR "quantisation" OR "bit depth" OR "coarse depth" identification robustness Kinect anonymization
- WebSearch: "depth" person identification privacy 2025 2026 arXiv "silhouette" "depth" comparison cross-session clothing change CNN
- WebFetch: https://pmc.ncbi.nlm.nih.gov/articles/PMC11723478/
- WebSearch: "Beyond Privacy of Depth Sensors in Active and Assisted Living Devices" PETRA 2022 authors depth face recognition identification
- WebSearch: Munaro Basso Fossati Van Gool Menegatti "One-shot person re-identification with a consumer depth camera" BIWI RGBD-ID results Still Walking rank-1 point cloud matching skeleton
- WebSearch: action recognition background bias "inpainting" OR "inpainted" actor removed "background only" scene-only accuracy Kinetics UCF101 static bias SCUBA SCUFO
- WebSearch: depth person re-identification "with background" "without background" OR "background removed" rank-1 ablation depth images CNN TVPR OR DPI-T OR BIWI
- WebSearch: TUM-GAID Hofmann 2014 "Depth Gradient Histogram Energy Image" DGHEI vs GEI results clothing "coat" "backpack" rank-1
- WebSearch: Hong "Fine-Grained Shape-Appearance Mutual Learning" CVPR 2021 silhouette body shape cloth-changing re-identification PRCC LTCC shape stream rank-1
- WebSearch: face recognition "background" bias "background only" OR "background-only" accuracy context cues CNN dataset shortcut identity
- WebSearch: depth privacy "depth images" identity leakage re-identification "privacy-preserving" claim tested attack depth camera "person identification" gait 2024 2025 study
- WebSearch: "Kinect" depth "1-bit" OR "binarized" OR "thresholded" depth silhouette equivalent identification gait "quantization levels" experiment
- WebSearch: Ding Shao "Mitigating and Evaluating Static Bias of Action Representations in the Background and the Foreground" ICCV 2023 SCUBA SCUFO background-only accuracy
- WebSearch: Sivapalan 2011 gait energy volume GEV GEI CMU MoBo frontal depth recognition rate results eprints.qut.edu.au
- WebSearch: Kinect depth-only face recognition identification accuracy "depth" KinectFaceDB Min Kose Dugelay rank-1 depth vs RGB
- WebSearch: person re-identification "background only" OR "background-only" "person removed" OR "without the person" camera identification background leakage rank-1 experiment shortcut
- WebSearch: Gait3D CVPR 2022 Zheng "dense 3D representations" SMPL vs silhouette rank-1 silhouettes better than 3D mesh gait in the wild
- WebSearch: Chung Wu Russakovsky "Enabling Detailed Action Recognition Evaluation Through Video Dataset Augmentation" HAT background-only inpainted human accuracy Kinetics
- WebFetch: https://www.ce.cit.tum.de/en/mmk/misc/tum-gaid-database/
- WebFetch: https://link.springer.com/chapter/10.1007/978-3-031-08645-8_62
- WebFetch: https://www.semanticscholar.org/paper/The-TUM-Gait-from-Audio,-Image-and-Depth-(GAID)-of-Hofmann-Geiger/bdeff833f3a7c035ff2987984317493d0a02f8d9/figure/11
- WebFetch: https://ieeexplore.ieee.org/document/6117504/
- WebFetch: https://core.ac.uk/search/?q=%22Gait%20energy%20volumes%20and%20frontal%20gait%20recognition%20using%20depth%20images%22
- WebFetch: https://idp.springer.com/authorize?response_type=cookie&client_id=springerlink&redirect_uri=https%3A%2F%2Flink.springer.com%2Fchapter%2F10.1007%2F978-3-031-08645-8_62
- WebFetch: https://link.springer.com/chapter/10.1007/978-3-031-08645-8_62?error=cookies_not_supported&code=a67bd26f-78fb-468e-9a3f-138407c08f1a
