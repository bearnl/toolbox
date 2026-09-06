# 03_depthid — partial working notes recovered from the killed transcript
(assistant-side text only; 4 blocks, 94 web actions)

## Notes
I'll start by loading the web tools, then run the literature searches.

---
Now I'll pull the primary sources for the exact numbers, plus a second wave of targeted searches.

---
The summary looks unreliable (it mentions silhouettes, which this skeleton paper doesn't use), so I'll read the saved PDF text directly and grep the protocol passages myself.

---
The grep came back empty, which suggests odd character spacing in the extraction; I'll inspect the file size and search for the experiments section with looser patterns.

---

## Sources visited / searched (in order)
- WebSearch: Munaro "One-shot person re-identification with a consumer depth camera" BIWI RGBD-ID rank-1 Still Walking skeleton point cloud matching
- WebSearch: Wu Zheng Lai "Robust Depth-Based Person Re-Identification" IEEE Transactions on Image Processing 2017 BIWI Walking Still rank-1 Eigen-depth
- WebSearch: Karianakis "Reinforced Temporal Attention and Split-Rate Transfer for Depth-Based Person Re-Identification" BIWI top-1 results
- WebSearch: Hafner "cross-modal distillation" RGB-depth person re-identification BIWI RobotPKU results depth rank-1
- WebSearch: Haque Alahi Fei-Fei "Recurrent Attention Models for Depth-Based Person Identification" BIWI top-1 4D RAM
- WebSearch: Patruno "skeleton standard posture" people re-identification RGB-D BIWI Pattern Recognition 2019 results
- WebSearch: Imani Soltanizadeh RGB-D person re-identification anthropometric measures Kinect BIWI results
- WebSearch: "feature funnel" RGB-D person re-identification depth 2017
- WebSearch: Pala Serra Marras Fumera "Enhanced skeleton and face 3D data for person re-identification from depth cameras" results dataset
- WebSearch: Sensors 2023 survey "Person Re-Identification with RGB-D and RGB-IR Sensors" BIWI comparison table rank-1
- WebSearch: Ren Lu Feng Zhou "Multi-modal uniform deep learning for RGB-D person re-identification" Pattern Recognition BIWI results
- WebSearch: Barbosa Cristani Del Bue Bazzani Murino "Re-identification with RGB-D sensors" 2012 soft biometrics skeleton anthropometric PAVIS results
- WebSearch: Yu Yang Tian "3D point cloud based body recognition" ICME 2016 Kinect anthropometric identification
- WebSearch: Kinect height estimation accuracy depends on distance field of view body clipped stature measurement depth camera walking
- WebSearch: "biometric menagerie" gait recognition sheep goats lambs wolves Doddington Yager Dunstone gait analysis
- WebSearch: Grother Phillips "Models of large population recognition performance" identification rate versus gallery size CMC
- WebSearch: RGB versus depth person re-identification clothing change same pipeline depth robust RGB drops comparison BIWI IIT PAVIS
- WebSearch: Liu Sensors 2025 RGB-D person re-identification clothing change depth
- WebSearch: depth-based person re-identification 2025 2026 BIWI RGBD-ID new method
- WebSearch: gait recognition Kinect frames excluded "entire body" visible full body in field of view frame selection depth
- WebFetch: https://arxiv.org/pdf/1705.09882
- WebFetch: https://arxiv.org/pdf/1703.09474
- WebFetch: https://arxiv.org/pdf/1810.11641
- WebFetch: https://arxiv.org/pdf/1611.07212
- WebFetch: https://www.mdpi.com/1424-8220/23/3/1504
- WebFetch: https://www.sciencedirect.com/science/article/pii/S2468232217300215
- WebFetch: https://iris.cnr.it/retrieve/e0d2d9d0-4747-46ca-87c7-4ce0e9bd0111/prod_399791-doc_185851.pdf
- WebFetch: https://lorisbaz.github.io/papers/proceedings/Barbosaetal_REID12.pdf
- WebFetch: https://www.mdpi.com/2076-3417/12/21/11144/htm
- WebFetch: https://www.mdpi.com/1424-8220/25/1/271
- WebSearch: Munaro Fossati Basso Menegatti Van Gool one-shot re-identification depth camera pdf BIWI results table "Still" "Walking" rank-1 dei.unipd.it
- WebSearch: Andersson Araujo "Person identification using anthropometric and gait data from Kinect sensor" AAAI 2015 results accuracy anthropometric attributes
- WebSearch: Yu Yang Tian ICME 2016 "body recognition" point cloud Kinect person identification anthropometric
- WebSearch: "Doddington zoo" OR "biometric menagerie" gait recognition subjects never recognized "goats" gait study
- WebSearch: PRCC cloth-changing re-identification ResNet-50 baseline rank-1 same-clothes versus cross-clothes drop Yang Wu Zheng TPAMI 2019 contour sketch numbers
- WebSearch: Uddin Bhuiyan "Fusion in dissimilarity space" RGB-D skeleton person re-identification BIWI rank-1 cross-clothing results 2021
- WebSearch: Godil Grother Ressler "human identification from body shape" CAESAR anthropometric identification; Adjeroh Cao Piccirilli Ross "predictability and correlation in human metrology" identification
- WebSearch: person re-identification "number of frames" accuracy versus sequence length depth video identification "frames needed" gait identification time budget seconds
- WebSearch: Kinect anthropometric identification "standing distance" OR "distance from the sensor" height measurement bias body partially outside field of view top of head cut off depth
- WebSearch: arXiv 2025 depth person identification privacy-preserving Kinect OR "depth video" OR LiDAR re-identification clothing change body shape 2025 2026
- WebFetch: https://arxiv.org/html/2606.23230
- WebFetch: https://arxiv.org/pdf/2605.04977
- WebSearch: "Enhancing person re-identification on RGB-D data with noise free pose-regularized color and skeleton distance features" journal 2024 BIWI Still Walking recognition rate
- WebSearch: Munaro thesis OR ICRA 2014 "3D reconstruction of freely moving persons for re-identification with a depth sensor" BIWI rank-1 "multi-frame" "single-frame" Still Walking results
- WebFetch: https://ojs.aaai.org/index.php/AAAI/article/view/9212
- WebFetch: https://www.researchgate.net/publication/4040365_Human_identification_from_body_shape
- WebSearch: "Biometric Performance as a Function of Gallery Size" Applied Sciences 2022 authors doi rank-1 decline log gallery size
- WebSearch: Sensors 2025 Kai Liu person identification RGB thermal depth "infrared array" LiDAR ViT ResNet34 resolution 640 × 480 privacy
- WebSearch: "Long-Term People Reidentification using Anthropometric Signature" 2016 Nambiar OR authors Kinect skeleton anthropometric results
- WebSearch: Munsell Temlyakov Qu Wang "Person Identification Using Full-Body Motion and Anthropometric Biometrics from Kinect Videos" ECCV workshop 2012 results accuracy
- WebFetch: https://www.mdpi.com/2076-3417/12/6/2835
- WebSearch: Pala Marras Fumera Roli "Multimodal person reidentification using RGB-D cameras" TCSVT 2016 anthropometric clothing appearance clothing change results
- WebSearch: "point cloud" "body recognition" OR "body identification" ICME 2016 Tian Kinect person identification 3D shape "Yu"
- WebSearch: human parsing mask versus depth mask ablation person re-identification "segmentation mask" RGB "background" control clothing change silhouette from RGB segmentation vs Kinect user mask comparison
- WebSearch: gallery composition effect small gallery identification accuracy variance "which subjects" enrolled random subsets subjects rank-1 spread face OR gait OR re-identification
- WebSearch: Haque Alahi Fei-Fei DPI-T depth person identification BIWI "top-1" table "Skeleton" "Energy Volume" "3D CNN" "4D RAM" results
- WebFetch: https://ojs.aaai.org/index.php/AAAI/article/view/9212/9071
- WebFetch: https://arxiv.org/pdf/1906.06272
- WebFetch: https://arxiv.org/pdf/2002.02295
- WebFetch: https://arxiv.org/html/2312.03033v2
- WebSearch: Pala Satta Fumera Roli 2016 "Multimodal person reidentification" RGB-D anthropometric measures "clothing" appearance rank-1 results "different clothes" dataset re-identification
- WebSearch: Munaro "BIWI RGBD-ID" "Table 3" OR "rank-1" "multi-frame" skeleton descriptor 13 features "Still" "Walking" one-shot re-identification book chapter numbers 26.6 OR 32.5 OR 21.1
- WebSearch: Nambiar Bernardino Nascimento "anthropometric" gait Kinect re-identification KS20 dataset "view-point invariant" results 2017 anthropometric features accuracy
- WebSearch: Wayman OR "Bolle" "identification" small gallery "closed-set" accuracy depends "subject composition" OR "gallery membership" bootstrap subjects variance biometric evaluation
- WebSearch: depth video person re-identification "body shape" anthropometric "hand-crafted" outperform CNN OR "deep" small dataset cross-session Kinect 2022 OR 2023 OR 2024
- WebSearch: "pointcloud" OR "point cloud" re-identification Kinect "body measurements" height "shoulder width" "chest" volume identification accuracy 2020 OR 2021 OR 2022 OR 2023 OR 2024 OR 2025
- WebSearch: re-identification risk "identification accuracy" versus "number of people" OR "population size" cohort curve privacy "k people" gait OR "body shape" OR depth sensor uniqueness
- WebFetch: https://robotics.dei.unipd.it/images/Papers/Conferences/MunaroICRA14a.pdf
- WebFetch: https://robotics.dei.unipd.it/images/Papers/Conferences/MunaroICRA14b.pdf
- WebSearch: LTCC long-term cloth-changing re-identification Qian ACCV 2020 ResNet-50 baseline rank-1 standard setting versus cloth-changing setting numbers
- WebSearch: Kinect skeleton gait identification "tracked" joints "inferred" frames discarded OR removed OR excluded "not tracked" person identification anthropometric preprocessing
- WebSearch: Hasan Babaguchi BTAS 2016 "anthropometric signature" long-term reidentification results rank-1 accuracy dataset Kinect clothes
- WebSearch: "Engineering Research Express" 2024 person re-identification RGB-D "pose-regularized" BIWI Bhattacharya OR authors doi
- WebFetch: https://arxiv.org/pdf/2005.12633
- WebFetch: https://pmc.ncbi.nlm.nih.gov/articles/PMC11723478/
- WebFetch: https://arxiv.org/abs/2307.14578
- WebFetch: https://robotics.dei.unipd.it/reid/index.php/8-dataset
- WebFetch: https://link.springer.com/chapter/10.1007/978-1-4471-6296-4_8
- WebFetch: https://ar5iv.labs.arxiv.org/html/2005.12633
- WebFetch: https://www.academia.edu/25117360/Human_identification_from_body_shape
- WebFetch: https://ieeexplore.ieee.org/document/7791184/
- WebFetch: https://ui.adsabs.harvard.edu/abs/2024ERExp...6a5205B/abstract
- WebFetch: https://arxiv.org/pdf/2307.14578
- WebFetch: https://arxiv.org/html/2401.15296v2
- WebFetch: https://www.nist.gov/publications/human-identification-body-shape
- WebFetch: https://ouci.dntb.gov.ua/en/works/lxbMmrG7/
- WebFetch: https://linkinghub.elsevier.com/retrieve/pii/S2590005621000369
- WebFetch: https://ar5iv.labs.arxiv.org/html/2303.06819
- WebFetch: https://ar5iv.labs.arxiv.org/html/2009.03671
- WebFetch: https://ar5iv.labs.arxiv.org/html/2204.09826
- WebFetch: https://ar5iv.labs.arxiv.org/html/2008.09435
- WebFetch: https://robotics.dei.unipd.it/reid/index.php/downloads
- WebFetch: https://arxiv.org/pdf/2009.03671
- WebFetch: https://ar5iv.labs.arxiv.org/html/2009.03671
