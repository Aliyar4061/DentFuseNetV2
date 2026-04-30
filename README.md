# DentFuseNet

**A validation‑optimized static weighting ensemble for multi‑condition dental disease classification and oral cancer detection**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org)
[![Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com)

---

## 📝 Abstract

**Background:** Deep learning has shown promise in dental imaging, but clinical translation is hindered by limited generalizability, task fragmentation, and lack of interpretability.

**Objectives:** To develop DentFuseNet, a unified and interpretable deep learning framework for multi‑class dental condition classification and binary oral cancer detection, leveraging a validation‑optimized static weighting ensemble of eight diverse architectures.

**Methods:** Eight modern backbones (DenseNet121, EfficientNetV2‑S, ResNet50, InceptionV3, MobileNetV3‑Large, ConvNeXt‑Tiny, ViT‑Small, Swin‑Tiny) are evaluated. CNN backbones are enhanced with CBAM attention; transformers bypass CBAM. Top‑3 models are selected by validation accuracy and fused using SLSQP‑derived static weights. A compound loss (focal + weighted cross‑entropy, α=0.6) mitigates class imbalance. Grad‑CAM and t‑SNE provide interpretability. Stratified 3‑fold cross‑validation, domain‑shift perturbations, data‑scarcity analysis, and SMOTE ablation assess robustness.

**Results:**  
- **Dental (5‑class):** 99.71% accuracy (95% CI: 99.32–100.00%), AUC = 1.0000, κ = 0.9963, MCC = 0.9963, Brier = 0.0023  
- **Oral Cancer (binary):** 98.53% accuracy, AUC = 0.9995, κ = 0.9707, MCC = 0.9708  
- **Minority‑class FNR:** Hypodontia = 1.60%, Malignant = 3.59%  
- **Generalizability:** 3‑fold CV mean accuracy = 98.51% (dental); Wilcoxon p ≥ 0.250; robust to brightness/contrast changes, but vulnerable to Gaussian noise.  
- **Efficiency:** Inference under 15 ms on NVIDIA Tesla T4 for lightweight reduced ensemble (MobileNetV3 + ResNet50).

---

## 🚀 Key Features

- ✅ **Diverse Backbone Pool** – 8 architectures (CNNs, transformers, lightweight) for complementary features  
- ✅ **Validation‑Optimized Static Weighting** – SLSQP‑derived weights ensure reproducible, clinically auditable fusion (e.g., 0.31, 0.29, 0.24)  
- ✅ **CBAM Attention (CNNs only)** – Channel‑spatial recalibration improves lesion focus  
- ✅ **Compound Loss** – Focal + weighted cross‑entropy implicitly rebalances rare conditions (no SMOTE needed)  
- ✅ **Statistical Rigor** – 95% bootstrap CIs, Wilcoxon, Kruskal–Wallis, McNemar’s test  
- ✅ **XAI Integration** – Grad‑CAM (CNNs) and t‑SNE for clinical transparency  
- ✅ **Robustness Suite** – 3‑fold CV, domain shifts (brightness, contrast, noise, resolution), data‑scarcity (25–100% data), SMOTE ablation  

---

## 📊 Dataset Summary

| Task | Samples | Classes / Conditions |
|------|---------|----------------------|
| Dental Conditions | ~2,541 (after augmentation) | 5 (Caries, Gingivitis, Hypodontia, Tooth Discoloration, Ulcers) |
| Oral Cancer Detection | ~940 (clinical dataset) | 2 (Benign, Malignant) |

---

## 📈 Performance Highlights

| Metric | Dental (5‑class) | Oral Cancer (binary) |
|--------|----------------|----------------------|
| Accuracy | 99.71% | 98.53% |
| AUC | 1.0000 | 0.9995 |
| Cohen's κ | 0.9963 | 0.9707 |
| MCC | 0.9963 | 0.9708 |
| Brier Score | 0.0023 | 0.0130 |
| FNR (Rare/Critical) | 1.60% (Hypodontia) | 3.59% (Malignant) |

**Cross‑Validation (Dental, Top‑3 ensemble):**  
- Mean accuracy = 98.51% ± 0.25%, AUC = 0.9996 ± 0.0001  
- Wilcoxon vs. best single model: p = 0.250  

**Domain‑Shift Robustness (Dental):**  
- Brightness/Contrast (+20%): 99.42% accuracy  
- Gaussian noise (σ=0.05): 33.82% accuracy → vulnerability identified  

---

## 🛠 Implementation Details

All experiments are implemented in **PyTorch 2.0+** with deterministic seeding (`SEED=42`) across Python, NumPy, and CUDA. The complete source code, trained weights, and evaluation scripts are publicly available at [https://github.com/Aliyar4061/DentFuseNet](https://github.com/Aliyar4061/DentFuseNet) to ensure full reproducibility.

### Training Hyperparameters & Software Stack

| Component | Specification |
|-----------|---------------|
| **Framework** | PyTorch 2.0+, torchvision 0.15+ |
| **Backbones** | DenseNet121, EfficientNetV2‑S, ResNet50, InceptionV3, MobileNetV3‑Large, ConvNeXt‑Tiny, ViT‑Small, Swin‑Tiny (via `timm`) |
| **Input Size** | 224×224 pixels |
| **Batch Size** | 32 (train/val); 16 (TTA) |
| **Optimizer** | AdamW (lr = 2e-4, weight decay = 1e-4) |
| **Scheduler** | CosineAnnealingLR (T_max = 15, η_min = 1e-6) |
| **Loss Function** | Compound: α·Focal + (1‑α)·Weighted CE (α=0.6, γ=2, α_t=0.25) + label smoothing ε=0.1 |
| **Training Epochs** | 15 (early stopping patience = 3) |
| **Data Augmentation** | Shift/scale/rotate (±6%, 15°), color jitter (±20%), horizontal flip (p=0.5) |
| **Test Time Augmentation** | 5 steps (horizontal flips) |
| **Ensemble Weighting** | Validation‑optimized static weights via SLSQP (top‑3 models) |
| **Bootstrapping** | 200 iterations, 95% confidence intervals |
| **Hardware** | NVIDIA Tesla T4 (16 GB VRAM) via Kaggle |
| **CUDA Version** | 11.8 |
| **Reproducibility** | Fixed seed 42, deterministic CuDNN, `torch.backends.cudnn.deterministic = True` |

---

## 💻 Environment & Hardware

### System Requirements
- **OS:** Linux (Ubuntu 20.04+) – Kaggle environment  
- **Python:** 3.10  
- **CUDA:** 11.8 (for GPU training)  
- **GPU:** NVIDIA Tesla T4 (16 GB VRAM) minimum; more RAM recommended for larger batch sizes  
- **RAM:** 32 GB+ recommended  

### Software Dependencies
All required packages are listed in `requirements.txt`. Core libraries:
- `torch>=2.0.0`, `torchvision>=0.15.0`
- `timm>=0.9.0` (pretrained backbones)
- `albumentations>=1.3.0` (augmentations)
- `scikit-learn`, `pandas`, `numpy`
- `matplotlib`, `seaborn`, `tqdm`
- `opencv-python-headless`, `pillow`
- `thop` (optional, for FLOPs calculation)

---

## 🔧 Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/Aliyar4061/DentFuseNet.git
cd DentFuseNet
pip install -r requirements.txt
