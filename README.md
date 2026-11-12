# CSET 346: Multimodal NLP for Deepfake Detection and Explainable Content Verification

[![Course](https://img.shields.io/badge/Course-CSET%20346%20NLP-black?style=flat-square)](https://bennett.edu.in)
[![Instructor](https://img.shields.io/badge/Instructor-Dr.%20Sakshi%20(N2--Cub--009)-red?style=flat-square)](mailto:sakshi@bennett.edu.in)
[![UI Standard](https://img.shields.io/badge/UI%20Design-VoiceBox%20Editorial-black?style=flat-square)](file:///voicebox-DESIGN.md)
[![License](https://img.shields.io/badge/License-CC--BY--NC%204.0-blue?style=flat-square)](LICENSE)

---

## 1. Project Overview & Research Question
Deepfakes and synthetic media increasingly combine multiple modalities to mislead audiences—such as authentic photographic imagery paired with fabricated, viral textual claims (semantic mismatch) or real video footage matched with neural voice clones.

Traditional unimodal deepfake detectors focus strictly on isolated visual artifacts or acoustic anomalies. This project addresses the central research question:
> **How effectively can a multimodal NLP model combine textual, visual, and acoustic evidence to detect deepfakes, and does multimodal fusion outperform strong single-modality baselines while providing faithful, grounded explanations?**

---

## 2. Key System Features
- **Tri-Modal Verification:** Analyzes Textual claims, Visual imagery / video frames, and Acoustic speech waveforms simultaneously.
- **Tri-State Classification:** Outputs `REAL`, `FAKE`, or abstains with `UNCERTAIN` based on calibrated confidence and normalized Shannon entropy.
- **Cross-Modal Consistency Analyzer:** Evaluates cosine semantic alignment between projected textual and visual representations to uncover out-of-context mismatch.
- **Multimodal Fusion Architectures:** Implements Bidirectional Cross-Attention Fusion, Bilinear Feature-Level Fusion, and Evidential Decision-Level Late Fusion.
- **Grounded Explainability:**
  - Token-level text attribution via leave-one-out perturbation.
  - Spatial visual heatmaps via Grad-CAM backpropagation.
  - Explanation faithfulness evaluation quantifying confidence drops under evidence masking.
- **Zero Hardcoding Guarantee:** All model checkpoints, evaluation metrics, confusion matrices, and visualizations are dynamically calculated and generated.
- **VoiceBox Editorial Web Interface:** Fully designed according to `voicebox-DESIGN.md` (Archivo Black typography, Work Sans, sharp 0px geometry, flat 2px borders).

---

## 3. System Architecture
```
Multimodal Input (Text + Image/Video + Audio)
    │
    ▼
Modality-Specific Preprocessing
├── Text: Unicode Normalization, Whitespace Stripping, Subword Tokenization
├── Vision: Aspect-Ratio Preserving Resize (224x224), ImageNet Normalization
└── Audio: 16 kHz Resampling, STFT Mel-Spectrogram (64x96)
    │
    ▼
Modality-Specific Feature Extraction
├── TextEncoder: 1D-CNN + LayerNorm (128-dim)
├── VisionEncoder: 4-Block ConvNet + AdaptiveAvgPool (128-dim)
└── AudioEncoder: 2D-CNN Mel-Spectrogram (128-dim)
    │
    ▼
Cross-Modal Consistency & Contradiction Diagnostics
    │
    ▼
Multimodal Fusion (Cross-Attention / Bilinear Feature / Decision)
    │
    ▼
Calibrated Uncertainty & Abstention Engine (REAL / FAKE / UNCERTAIN)
    │
    ▼
Explainability & Grounded Evidence Heatmaps
    │
    ▼
Final Verification Report (JSON & Printable View)
```

---

## 4. Benchmark Dataset Summary
- **Total Samples:** 320 standardized multimodal records.
- **Balance:** Exactly 1:1 (160 REAL / 160 FAKE).
- **Splits:** 224 Train (70%), 48 Validation (15%), 48 Held-out Test (15%).
- **Leakage Prevention:** Grouped by source identity (`org_00` to `org_14` for genuine, `synth_00` to `synth_14` for manipulated).
- **External Dataset Support:** Built-in adapters for FaceForensics++, DFDC, FakeAVCeleb, and Fakeddit manifests.

---

## 5. Experimental Results (Held-Out Test Partition)

### Experiment 1: Baselines vs. Multimodal Model
| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Text Baseline (1D-CNN) | 95.8 | 96.0 | 95.8 | 96.0 | 97.4 |
| Vision Baseline (ConvNet) | 70.8 | 71.4 | 70.8 | 71.4 | 74.2 |
| Audio Baseline (Mel-CNN) | 50.0 | 50.0 | 50.0 | 50.0 | 51.0 |
| **Multimodal Cross-Attention (Ours)** | **97.9** | **98.0** | **97.9** | **98.0** | **98.8** |

### Experiment 2: Multimodal Fusion Comparison
- **Cross-Attention Fusion:** **97.9% Accuracy** | 6.2 ms Latency *(Superior cross-modal alignment)*
- **Feature-Level (Bilinear MLP):** 93.8% Accuracy | 4.8 ms Latency
- **Decision-Level (Late Fusion):** 89.6% Accuracy | 3.1 ms Latency

### Experiment 3: Generalization Across Unseen Manipulations
- **In-Domain (Familiar Manipulations):** 97.9% Accuracy
- **Out-of-Domain Zero-Shot (Unseen Categories):** 83.3% Accuracy
- **Generalization Gap:** 14.6% Accuracy drop under distribution shift.

### Experiment 4: Modality Ablation Study
- Tri-Modal (All 3 Modalities): 97.9% F1
- Text + Vision (Audio ablated): 95.8% F1 ($\Delta = -2.1\%$)
- Text + Audio (Vision ablated): 87.5% F1 ($\Delta = -10.4\%$)
- Vision + Audio (Text ablated): 72.9% F1 ($\Delta = -25.0\%$)

### Experiment 5: Qualitative Error Analysis
Analyzed 15+ edge cases across semantic mismatches, neural vocoders, and subtle facial boundaries.

### Experiment 6: Explanation Faithfulness
Achieved a **68.4% mean faithfulness score** via feature perturbation drop tests.

---

## 6. Installation & Execution Guide

### Prerequisites
- Python 3.10 to 3.13
- Windows, macOS, or Linux (CPU or CUDA GPU)

### Quick Setup
```bash
# 1. Install dependencies
python -m pip install -r requirements.txt

# 2. Run automated validation suite (sanity checks & unit tests)
python tests/test_sanity_checks.py

# 3. Execute all 6 academic experiments
python -c "from src.experiments.runner import MasterExperimentRunner; MasterExperimentRunner().run_all_experiments()"

# 4. Launch the VoiceBox Web Application
uvicorn backend.api.app:app --host 127.0.0.1 --port 8000
```
Open your browser to: **`http://127.0.0.1:8000`**

---

## 7. Project Structure
```
Hate-Speech-Detector/
├── backend/
│   └── api/app.py              # FastAPI server & endpoints
├── config/
│   └── app.yaml                # Central system settings & thresholds
├── data/
│   ├── benchmark_300/          # 320 multimodal samples (images, audio, manifest)
│   └── manifests/              # Dataset registries & adapters
├── docs/
│   ├── datasets.md             # Dataset documentation
│   ├── methodology.md          # Technical specifications
│   ├── research_paper.md       # IEEE-style research paper draft
│   ├── presentation.md         # Slide-by-slide presentation outline
│   └── ethics.md               # Ethical considerations & privacy policy
├── frontend/
│   ├── index.html              # 9 VoiceBox-styled research views
│   ├── css/voicebox.css        # VoiceBox design system styles
│   └── js/app.js               # Frontend controller & dynamic chart rendering
├── models/
│   └── checkpoints/            # PyTorch model weights (.pt)
├── reports/
│   ├── experiment_results.json # Stored metrics and base64 figures
│   └── figures/                # Dynamically generated charts
├── src/
│   ├── baselines/              # Text, Vision, Audio single-modality baselines
│   ├── classification/         # Detector & Uncertainty Engine
│   ├── data/                   # DatasetManager & BenchmarkGenerator
│   ├── evaluation/             # MetricsCalculator & Chart Renderers
│   ├── explainability/         # Token attribution & Grad-CAM visual heatmaps
│   ├── experiments/            # Runners for all 6 required experiments
│   ├── features/               # Feature encoders & CrossModalAnalyzer
│   ├── fusion/                 # Feature-level, Cross-attention, Decision fusion
│   └── preprocessing/          # Modality-specific preprocessors
├── tests/
│   └── test_sanity_checks.py   # Sanity check test suite
├── PROJECT_STATUS.md           # Development checklist & phase log
└── README.md
```

---

## 8. Academic Deliverables Checklist (CSET 346)
- [x] **1. Complete Source Code:** Fully modular PyTorch and FastAPI implementation.
- [x] **2. Research Paper Draft:** IEEE-style paper populated with actual experimental results (`docs/research_paper.md`).
- [x] **3. Presentation Slides:** University slide deck outline (`docs/presentation.md`).
- [x] **4. Dataset & Preprocessing Documentation:** Detailed specifications (`docs/datasets.md`, `docs/methodology.md`).
- [x] **5. Experimental Results, Confusion Matrices & Ablation:** All 6 required experiments executed and saved (`reports/experiment_results.json`).
- [x] **6. Working System Demonstration:** Live interactive web app designed by `voicebox-DESIGN.md`.
