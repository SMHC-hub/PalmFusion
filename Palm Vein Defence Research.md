# Defending 100% on FYODB: literature precedents and dataset access

**The user's 100% accuracy on FYODB is consistent with the published state of the art.** The dataset's own creators reached **100%** with multimodal fusion (Toygar et al., IEEE Access 2020), and at least three peer-reviewed CASIA palm-vein papers — including ECA-ResNet (MDPI 2023) and an explainable multispectral CNN (ACM TOMM 2021) — also report **100% identification accuracy**. Closed-set rates of 99–100% are now routine on every small-to-medium palm-vein benchmark, so the result is defensible provided the protocol (closed-set, train/test split, EER) is clearly stated. On the dataset side, **VERA PalmVein, PUT Vein, Tongji Contactless Palm Vein, FYODB, and Bosphorus Hand Vein** are realistically obtainable for cross-dataset tests; **BJTU-PVD, NCUT, and the contactless PolyU-IITD v3.0** are not practical for a free, simple-registration workflow, and **IITD Touchless Palmprint** is a visible-light palmprint set, not a vein dataset.

## Part 1 — Published palm-vein deep-learning results (2019–2024)

### Tier 1: peer-reviewed papers reporting 100% accuracy

These are the most useful citations for defending a 100% FYODB result, because each appears in a reputable venue and the 100% claim is on a comparable closed-set palm-vein protocol.

| Paper | Year | Dataset | Classes | Best result | Method | Fusion / ensemble | Venue |
|---|---|---|---|---|---|---|---|
| Htet & Lee, "Contactless Palm Vein Recognition Based on Attention-Gated Residual U-Net and ECA-ResNet" | 2023 | CASIA Multi-Spectral (850/940 nm) | 200 hands | **100% accuracy, EER 0.018** | Attention-gated Residual U-Net segmentation + ECA-ResNet-50 with combined ArcFace + focal + triplet loss | Multi-loss head | MDPI Applied Sciences 13(11) 6363 |
| Yao, Jhong, Hsia & Hua, "Explainable AI: A Multispectral Palm-Vein Identification System" | 2021 | CASIA Multi-Spectral (all 6 bands) | 200 hands | **CIR 100%**, error rate 0.0556% | Lightweight CNN with SMDWT-PCA augmentation | Adaptive multispectral fusion | ACM TOMM 17(3) |
| Toygar, Babalola & Bitirim, "FYO: A Novel Multimodal Vein Database With Palmar, Dorsal and Wrist Biometrics" | 2020 | **FYO** (palmar + dorsal + wrist) | 160 subjects / 320 hands | **100%** with BSIF feature fusion of all three traits; CNN multimodal ≈ 98–100% | Custom CNN + BSIF/Gabor/HOG | Feature- and decision-level multimodal fusion | IEEE Access 8, 82461–82470 |
| Al-johani & Elrefaei, "Palmprint and Dorsal Hand Vein Multi-Modal Biometric Fusion Using Deep Learning" | 2022 | Bosphorus, COEP, Badawi | up to 100 | **up to 100%** | VGG16 + VGG19 + AlexNet | Score-level fusion (product rule) | IGI Global IJSDS |

### Tier 2: peer-reviewed deep-learning papers at 99–99.99%

| Paper | Year | Dataset | Best result | Method | Venue |
|---|---|---|---|---|---|
| Babalola, Bitirim & Toygar, "A Palm Vein Recognition Approach by Multiple CNN Models" | 2021 | **FYO**, PUT, VERA | **FYO 99.06%, PUT 99.83%, VERA 99.26%** | VGG16 + VGG19 + AlexNet, **decision-level fusion** | European J. Sci. & Technology (Dergipark) |
| Babalola, Bitirim & Toygar, "Palm vein recognition through fusion of texture-based and CNN-based methods" | 2021 | CASIA, FYO, PUT, VERA, Tongji | competitive with state of the art on all 5 | CNN + 5-region BSIF, decision + score fusion | Springer Signal, Image and Video Processing 15 |
| Babalola, Toygar & Bitirim, "Boosting hand vein recognition with fusion of different color spaces" | 2023 | FYO, PUT, VERA | "considerable improvement" over prior work | AlexNet/VGG-19/ResNet-50 over RGB, XYZ, LAB, YUV, HSV | Springer SIViP 17 |
| Hernández-García et al., "Evaluation of end-to-end CNN models for palm vein recognition" (SingleNet) | 2021 | 7 DBs incl. **FYO**, IITI, PUT, Tongji, PolyU | **state-of-the-art on FYO, PUT, IITI**; ties Tongji and PolyU | Shallow end-to-end SingleNet vs ResNet-32 | IEEE SoftCOM 2021 |
| Wulandari, Chai, Basari & Gunawan, VeinCNN with DWT + HOG | 2024 | CASIA, VERA, Tongji, PolyU, PUT | **CASIA 99.85%**, AUC 99.80%, EER 0.0083 | CNN + DWT + HOG hybrid features | MDPI Sensors 24(2) 341 |
| Obayya, El-Ghandour & Alrowais, "Bayesian-optimized CNN for contactless palm vein" | 2021 | CASIA Multi-Spectral | **99.4%**, EER 0.0683% | CNN with Bayesian-optimised architecture, Jerman filter | IEEE Access 9 |
| Ananthi, Sekar & Arivazhagan, "Ensembling Scale-Invariant and Multiresolution Gabor Scores" | 2022 | CASIA, VERA | **CASIA 99.73%, VERA 99.86%**, EER 0.021% / 0.0207% | Curvelet + Gabor, weighted score fusion | Information Technology and Control 51(4) |
| Chen, Hsia & Chen, "Contactless Multispectral Palm-Vein with Lightweight CNN" | 2021 | CASIA Multi-Spectral | error 0.0556% (≈ **99.94%**) | Lightweight CNN + adaptive Gabor + triplet loss | IEEE Access 9 |
| Stanuch, Wodzinski & Skalski, "Contact-Free Multispectral Identity Verification" | 2020 | self-built (1,030 hands, 10,160 imgs) | **TPR 99.5%** at EER threshold | Custom DCNN, NIR + UV multispectral | MDPI Sensors 20(19) 5695 |
| Liu et al., "Palm Vein CNN+HOG Feature Fusion" | 2020 | CASIA, PolyU | CASIA 99.25%, PolyU 99.90% | CNN + HOG feature-level fusion | IEEE conference |
| Pan, Wang, Wang & Zhu, "Multi-Scale Deep Representation Aggregation for Vein Recognition" | 2021 | CASIA, PolyU palm vein | state-of-the-art at publication | Multi-scale CNN feature aggregation | IEEE TIFS 16 |
| Qin, Gong, Li, Gao & El-Yacoubi, "Label-Enhancement Multiscale Vision Transformer (LE-MSVT)" | 2023 | CASIA, Tongji, PolyU | state of the art on all three | Multi-scale ViT + label-enhancement loss | IEEE TIM 72 |
| Li, Ruan, Qin, Deng & El-Yacoubi, "Transformer-Based Defense GAN against palm-vein adversarial attacks" | 2023 | CASIA, Tongji palm vein | very high under adversarial defence | Local Transformer + Defense GAN | IEEE TIFS 18 |
| Qin, El-Yacoubi, Li & Liu, "MSMDGAN+CNN single palm-vein identification" | 2021 | two public hand-vein DBs | state-of-the-art recognition | GAN augmentation + CNN | IEEE TIFS 16 |
| Jia, Gao, Xia, Zhao, Min & Lu, "Performance Evaluation of Classic CNNs for 2D/3D Palmprint and Palm Vein" | 2021 | 7 DBs incl. CASIA, Tongji palm vein | EfficientNet best, all CNNs near-100% on small DBs | 17 classic CNNs benchmarked | Springer Machine Intelligence Research 18 |
| Lefkovits, Lefkovits & Szilágyi, "Applications of Different CNN Architectures for Palm Vein Identification" | 2019 | PUT palm vein NIR | "high recognition accuracy for almost every CNN-based approach" | AlexNet, VGG-16, ResNet-50, SqueezeNet, custom + Hessian segmentation | Springer LNAI 11676 (MDAI 2019) |
| Musunuru, Sivaprakasam & Kishore, "Contactless Palm Vein Recognition with Integrated Learning Approach" | 2023 | CASIA | improved over prior boosting baselines | EfficientNetB1 features → LightGBM **boosting ensemble** | IJACSA 14(11) |
| Wu, Zhang, Li & Li, "Fusion recognition of palmprint and palm vein based on modal correlation" | 2024 | self-collected | improved over single-modality | Sparse projection + partial least squares feature fusion | Mathematical Biosciences and Engineering 21(2) |

### Typical accuracy ranges by dataset

The literature converges on a clear picture: closed-set deep-learning palm-vein recognition has effectively saturated on small benchmarks. **CASIA Multi-Spectral commonly hits 99–100%**; FYO sits at 95–99% with single CNNs and reaches ~99–100% with ensembles or multimodal fusion; VERA and PUT both have 99%+ reports; PolyU multispectral reaches 99.9%; Tongji (12,000 images) is the only widely-cited DB where high-90s remain non-trivial. This is exactly the regime where a 100% FYODB result is plausible — and the **EMU group that built FYODB has itself published 99.06% (decision fusion of three CNNs) and 100% (BSIF multimodal fusion)** in IEEE Access and Springer venues, so a 100% claim falls within the established range rather than above it.

### How to frame the defense

Three rhetorical moves are well-supported by the citations above. **First, cite precedent for 100%:** Toygar et al. 2020 (the FYODB authors), Yao et al. 2021 (ACM TOMM), and Htet & Lee 2023 (MDPI Appl. Sci.) all hit 100% in peer review. **Second, justify why FYODB is conducive to saturation:** 160 subjects, ≈ 480 palmar-vein NIR images, controlled vein-finder acquisition, low intra-class variation, closed-set protocol — every factor pushes accuracy toward the ceiling. **Third, document the ensemble's mechanism:** decision-level fusion of multiple CNNs (the VGG16+VGG19+AlexNet pattern from Toygar 2021) is the standard way 95–98% single-CNN baselines move to 99–100%, and that recipe has appeared in at least four FYO papers from the dataset's own creators. Reporting EER alongside accuracy and naming the train/test split protocol explicitly (e.g., 80/20 random split, k-fold, or strict subject-disjoint) addresses the main reviewer concern about closed-set saturation.

## Part 2 — Publicly available palm-vein datasets in 2026

The realistic picture is more constrained than the literature suggests. Many "public" datasets either require a faculty-signed license or are no longer reachable through their original portals. The table below reflects status as of April 2026; not every landing page could be fetched live, so flags marked ⚠️ rely on secondary citations and Zenodo / NIST BDbC / Idiap-Bob mirrors.

| # | Dataset | Institution | Subjects | Images | Modality | Access path | Friction | 2026 status |
|---|---|---|---|---|---|---|---|---|
| 1 | **VERA PalmVein** | Idiap (CH) | 110 | 2,200 NIR | NIR contact, 480×680 | Zenodo record 4575454 + EULA | Permanent-position signatory, academic email; gmail rejected | ✅ Active |
| 1b | VERA Spoofing PalmVein | Idiap | 50 (subset) | 1,000 PA prints | NIR | Zenodo 4580200 | Same EULA | ✅ Active |
| 2 | **CASIA Multi-Spectral Palmprint V1** | CAS Inst. of Automation | 100 / 200 hands | 7,200 (incl. 850/940 nm vein) | Multispectral, contactless | biometrics.idealtest.org form + signed PDF | Free for research | ⚠️ Portal intermittent; email cbsr-info@nlpr.ia.ac.cn directly |
| 3 | **PolyU Multispectral Palmprint** | HK PolyU | 250 / 500 palms | 24,000 (6,000 NIR) | RGB + NIR (vein band), 128×128 ROI | Email + signed agreement | Lab maintenance shifted post-retirement | ⚠️ Unstable |
| 3b | PolyU-IITD Contactless Palmprint v3.0 | PolyU + IITD | 600+ | 12,000+ | Visible RGB (no vein) | Online form | **Head-of-institution sign-off, ~45-day review** | ✅ Active but heavy DUA |
| 4 | **IITD Touchless Palmprint** | IIT Delhi | 230–235 | 2,601 | Visible (fluorescent) — **not vein** | Online form, simple email | None institutional | ✅ Active, but exclude for vein cross-eval |
| 5 | BJTU-PalmV1 | BJTU | 174 | 2,431 | Contact palmprint (not NIR vein) | No public portal located | Direct email to authors | ❌ No verified public download |
| 6 | **PUT Vein Pattern Database** | Poznan Univ. of Tech. | 50 | 2,400 (1,200 palm + 1,200 wrist) | NIR transmission, 1280×960 | biometrics.put.poznan.pl form | "Free for research" | ✅ Active |
| 7 | **Bosphorus Hand Vein DB** | Boğaziçi (TR) | 100 | ~1,200 | NIR **dorsal** (not palmar) | Online register → email link | **Simplest registration of all** | ✅ Active |
| 8 | **FYODB** | EMU (TR/Cyprus) | ~160 | ~1,920 across palmar/dorsal/wrist | NIR vein finder | Email Prof. Toygar + signed license | Faculty-signed license, no registrar DUA | ✅ Verified live April 2026 |
| 9 | **Tongji Contactless Palm Vein** | Tongji Univ. | 300 / 600 palms | 12,000 | Contactless NIR | cslinzhang.github.io/ContactlessPalm + email | Email + signed agreement | ✅ Active — recommended large benchmark |
| 10 | XJTU-UP | XJTU | 100 | 20,000+ | **Smartphone RGB, no vein** | Email request | Variable | ⚠️ By request |
| 11 | NCUT Hand Vein | NCUT | varies | — | NIR dorsal | None public | Direct email | ❌ Not publicly downloadable |
| 12 | Idiap "sweet" multimodal vascular | Idiap (2024) | 120 | multimodal | Multispectral NIR + RGB + photometric stereo | Open release with code | EULA framework | ✅ New, recommended |

### Realistic cross-dataset stack for palm-vein generalization

For a practical cross-dataset experiment that can survive peer review, the recommended stack is **Tongji Contactless Palm Vein** (the de-facto large modern benchmark with 12,000 NIR images), **VERA PalmVein** (high-quality, well-maintained Zenodo distribution), **PUT Vein** (palm + wrist NIR, low friction), and **CASIA Multi-Spectral** when the idealtest.org portal cooperates. **Bosphorus Hand Vein** adds an out-of-distribution dorsal-vein test with the lowest registration friction of any vein DB on this list. **FYODB** itself can be confirmed via direct email to Prof. Toygar; the license requires a faculty signature but no registrar-level DUA. The datasets to drop from the plan are **BJTU-PVD** and **NCUT** (no working public portal), **PolyU-IITD v3.0** (institutional license with ~45-day review is not "simple registration"), **XJTU-UP** (smartphone RGB, not a vein modality), and **IITD Touchless Palmprint** (visible-light palmprint, not vein — frequently misclassified in surveys).

### Caveats on reachability

A handful of historical landing pages (idealtest.org, the David Zhang biometrics portal at PolyU, web.iitd.ac.in) have shown intermittent availability across 2022–2025, and could not be fetched live for this report; the status flags above derive from Zenodo mirrors, the Idiap Bob-bio-vein toolchain documentation, NIST BDbC catalog entries, and 2024–2025 papers that still cite these resources as obtained. Three datasets are essentially "click-and-download" with only an academic-email registration: **Bosphorus Hand Vein, PUT Vein, and the Idiap sweet release**. Everything else requires at minimum a signed PDF.

## Conclusion

**100% accuracy on FYODB is squarely inside the published envelope** — it is exceeded or matched by Toygar et al. (IEEE Access 2020) on the same dataset, by Yao et al. (ACM TOMM 2021) and Htet & Lee (MDPI 2023) on CASIA, and is approached within one percentage point by at least eight other peer-reviewed papers using ensemble or fusion strategies. The strongest defense combines three elements: cite the FYODB-creator papers that themselves report 99.06–100%, document the closed-set protocol explicitly, and report EER alongside accuracy so reviewers can place the result on the standard verification curve. For cross-dataset generalization tests, the practically obtainable corpus in 2026 is **Tongji + VERA + PUT + CASIA + Bosphorus**, with FYODB itself accessible via the EMU license; **BJTU, NCUT, XJTU-UP, and the IITD Touchless Palmprint should not be relied upon** for a reproducible vein cross-dataset benchmark.