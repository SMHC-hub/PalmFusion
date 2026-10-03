<div align="center">

# Palm Vein Biometric Authentication
### Multi-Model Score-Level Fusion · PyTorch · FYODB · 100% Closed-Set Accuracy

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![CUDA](https://img.shields.io/badge/CUDA-12.6-76B900?style=for-the-badge&logo=nvidia&logoColor=white)](https://developer.nvidia.com/cuda-toolkit)
[![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)](LICENSE)

*A research-grade palm vein recognition system achieving 100% closed-set accuracy on the FYO dataset through score-level fusion of 11 CNN architectures, CLAHE infrared preprocessing, and Test-Time Augmentation.*

</div>

---

## Overview

This repository contains a complete implementation of a **palm vein biometric authentication system** developed as a Final Year Project (FYP). The system uses **near-infrared (NIR) palm vein imaging** and an ensemble of 11 deep CNN models to achieve state-of-the-art recognition accuracy.

### Key Results

| Metric | Value |
|--------|-------|
| **Closed-Set Accuracy (FYO)** | **100%** |
| **EfficientNetV2-S Val Accuracy** | 99.83% |
| **ResNet50 / ResNet101** | 100.00% |
| **DenseNet121** | 100.00% |
| **MobileNetV3-Large** | 100.00% |
| **VGG16 / VGG19** | 100.00% |
| **Dataset** | FYODB — 150 classes, 6,000 images |
| **External Validation** | PLUSVein (HandVein dataset) |
| **EER Analysis** | Ensemble EER computed across all fusion strategies |

> These results align with the state-of-the-art. The FYODB creators (Toygar et al.) themselves published 99.06% with a 3-CNN decision fusion and 100% with BSIF multimodal fusion in IEEE Access and Springer SIVIP.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Input: NIR Palm Image                    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│            CLAHE Preprocessing (Contrast Enhancement)       │
│  RGB → Grayscale → CLAHE → 3-Channel Replication           │
└─────────────────────────┬───────────────────────────────────┘
                          │
            ┌─────────────┼─────────────┐
            ▼             ▼             ▼
     EfficientNetV2   ResNet50/101   DenseNet121
     MobileNetV3      VGG16/VGG19    ConvNeXt-Small
     EfficientNet-B4  RegNetY-8GF    Custom PVCNN
            │             │             │
            └─────────────┼─────────────┘
                          ▼
┌─────────────────────────────────────────────────────────────┐
│          Score-Level Fusion (Softmax Probability Avg)       │
│               + Test-Time Augmentation (TTA)                │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
                  Identity Prediction
```

### Model Zoo (11 Models)

| Model | Architecture Family | Val Accuracy |
|-------|--------------------:|-------------:|
| EfficientNetV2-S | Compound Scaling | 99.83% |
| ResNet50 | Deep Residual | 100.00% |
| ResNet101 | Deep Residual (deeper) | 100.00% |
| DenseNet121 | Dense Connections | 100.00% |
| MobileNetV3-Large | Lightweight + SE Attention | 100.00% |
| VGG16 | Classic Deep Conv | 100.00% |
| VGG19 | Classic Deep Conv (deeper) | — |
| EfficientNet-B4 | Compound Scaling | — |
| ConvNeXt-Small | Modern Conv | — |
| RegNetY-8GF | Scalable Arch | — |
| Custom PVCNN | NIR-Specific Design | — |

---

## Dataset

### FYODB (FYO Palm Vein Database)
- **Source:** Eastern Mediterranean University (EMU), Cyprus
- **Subjects:** 150 individuals
- **Images:** 6,000 NIR palm vein images (40 per subject)
- **Acquisition:** Near-infrared vein finder device
- **Split:** 70% Train / 10% Val / 20% Test

> **Dataset not included** in this repository due to size. Request access from the dataset authors or use your own NIR palm vein dataset in the same folder structure.

### External Validation: PLUSVein
- **Source:** PLUSVein Hand Vein Database
- See `Model/PLUSVEIN_Eval.ipynb` for cross-dataset evaluation details.

---

## Project Structure

```
palm-vein-multimodel/
├── Model/
│   ├── PalmVein_V2.ipynb          # Main training notebook (11 models + fusion)
│   ├── PLUSVEIN_Eval.ipynb        # External dataset cross-validation
│   ├── Ensemble_EER.ipynb         # Equal Error Rate analysis
│   ├── Seed_Validation.ipynb      # Reproducibility / seed validation
│   ├── PLUSVein_Stats.ipynb       # Statistical distribution evaluation
│   └── plusvein_eval.py           # Standalone PLUSVein evaluation script
├── docs/
│   ├── figures/                   # Training curves, ROC, ablation, and Grad-CAM maps
│   └── reports/                   # Performance summary and ablation CSVs
├── Palm Vein Defence Research.md  # Comprehensive literature review (20+ papers)
├── LICENSE                        # MIT License
├── requirements.txt
├── .gitignore
└── README.md
```

> **Not in repo (too large):** `Dataset FYO/`, `Dataset FYO Preprocessed/`, `External/`, `Model/Saved Models/`, `Model/Outputs/`

---

## Setup & Usage

### Prerequisites
- Python 3.10+
- NVIDIA GPU with CUDA 12.x (tested on RTX 5070 — 12GB VRAM recommended)

### 1. Clone the Repository

```bash
git clone https://github.com/SMHC-hub/palm-vein-multimodel.git
cd palm-vein-multimodel
```

### 2. Install PyTorch (CUDA)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

> For other CUDA versions, visit [pytorch.org/get-started](https://pytorch.org/get-started/locally/)

### 3. Install Remaining Dependencies

```bash
pip install -r requirements.txt
```

### 4. Prepare Your Dataset

Place your NIR palm vein dataset in the following structure:

```
Dataset FYO/
├── 1/          ← subject 1 images
│   ├── img001.png
│   └── ...
├── 2/
│   └── ...
└── 150/
```

### 5. Run the Training Notebook

```bash
jupyter notebook Model/PalmVein_V2.ipynb
```

Run cells in order:
1. **Cell 1** — Install deps & detect GPU *(run once, then restart kernel)*
2. **Cell 2** — Imports & GPU config
3. **Cell 3** — CLAHE preprocessing *(creates `Dataset FYO Preprocessed/`)*
4. **Cell 4** — Dataset & DataLoaders
5. **Cell 5** — Training utilities
6. **Cells 6–16** — Train each model *(skip-if-trained: re-running loads checkpoints)*
7. **Final cells** — Score-level fusion & TTA evaluation

---

## Training Details

| Hyperparameter | Value |
|---------------|-------|
| Image Size | 224×224 |
| Batch Size | 32 |
| Optimizer | AdamW |
| LR Schedule | OneCycleLR (warm-up + cosine decay) |
| Mixed Precision | AMP float16 |
| Early Stopping | patience=7 |
| Label Smoothing | 0.1 |
| Augmentation | RandomFlip, RandomRotation±15°, ColorJitter, RandomErasing |

---

## Evaluation & Analysis

### EER Analysis
Run `Model/Ensemble_EER.ipynb` to compute Equal Error Rate across individual models and fusion strategies.

### External Validation
Run `Model/PLUSVEIN_Eval.ipynb` or `Model/plusvein_eval.py` to evaluate trained models on the PLUSVein dataset.

### Seed Reproducibility
Run `Model/Seed_Validation.ipynb` to verify result reproducibility across different random seeds.

---

## Research Background

The `Palm Vein Defence Research.md` file contains a comprehensive literature review of 20+ peer-reviewed papers on palm vein recognition (2019–2024), including:

- Benchmark accuracy ranges by dataset (FYODB, CASIA, VERA, PolyU, Tongji)
- Justification for 100% accuracy on FYODB (literature precedent)
- Publicly available dataset access status (2026)
- Defense strategy for FYP presentation

Key references include papers from IEEE TIFS, IEEE Access, Springer SIVIP, and MDPI Sensors.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

<div align="center">
  <em>Final Year Project · Palm Vein Biometric Authentication · PyTorch · FYODB</em>
</div>
