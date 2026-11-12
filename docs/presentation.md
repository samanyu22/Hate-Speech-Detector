# University Presentation Outline: Multimodal Deepfake Verification
## CSET 346 – Natural Language Processing | Instructor: Dr. Sakshi

### Slide 1: Title & Overview
- **Title:** Multimodal NLP for Deepfake Detection & Explainable Content Verification
- **Course:** CSET 346 – Natural Language Processing
- **Faculty Guide:** Dr. Sakshi (N2-Cub-009)
- **Key Proposition:** Moving beyond visual-only forensics to detect cross-modal inconsistencies (semantic mismatches, voice clones, out-of-context captions).

---

### Slide 2: Motivation & Problem Statement
- Deepfakes are no longer just faceswaps; they are multimodal misinformation packages.
- Real-world threats:
  * Authentic images paired with fabricated, viral captions.
  * AI-cloned voices endorsing fraudulent statements.
  * Deepfake videos with mismatched audio-visual speech cadence.
- **Core Research Question:** Does multimodal fusion outperform strong single-modality baselines while generating grounded, faithful explanations?

---

### Slide 3: System Architecture
- **Input Channels:** Text claims, Visual imagery / video frames, Acoustic speech waveforms.
- **Modality Encoders:**
  * Text: 1D-CNN + LayerNorm (128-dim).
  * Vision: Deep ConvNet + Adaptive Pooling (128-dim).
  * Audio: STFT Mel-Spectrogram 2D-CNN (128-dim).
- **Cross-Modal Consistency Analyzer:** Measures semantic alignment cosine similarity.
- **Multimodal Fusion Engine:** Cross-Attention Transformer querying inter-modal evidence.
- **Uncertainty Engine:** Predicts `REAL`, `FAKE`, or abstains with `UNCERTAIN`.

---

### Slide 4: Dataset & Provenance
- 320 standardized samples with exact 1:1 class balance (160 REAL / 160 FAKE).
- Covers 5 diverse manipulation categories:
  * Semantic Mismatch, Face Manipulation, Synthetic Voice, AI Diffusion, Misleading Captions.
- Identity-grouped stratified train/val/test split (70% / 15% / 15%) preventing data leakage.
- Zero missing files; verified format compliance.

---

### Slide 5: Experiment 1 – Baselines vs. Multimodal Model
- Text Baseline: 95.8% Accuracy | 96.0% F1
- Vision Baseline: 70.8% Accuracy | 71.4% F1
- Audio Baseline: 50.0% Accuracy | 50.0% F1
- **Multimodal Cross-Attention (Ours): 97.9% Accuracy | 98.0% F1**
- **Conclusion:** Combining cross-modal representations yields higher accuracy and robustness over any single modality.

---

### Slide 6: Experiment 2 – Multimodal Fusion Comparison
- **Feature-Level (Bilinear MLP):** 93.8% Accuracy (4.8 ms latency)
- **Cross-Attention Fusion:** 97.9% Accuracy (6.2 ms latency) — *Optimal synergy*
- **Decision-Level (Late Fusion):** 89.6% Accuracy (3.1 ms latency)

---

### Slide 7: Experiments 3 & 4 – Generalization & Modality Ablation
- **Generalization:** Model retains 83.3% accuracy on completely unseen zero-shot manipulation types.
- **Ablation:** Text modality provides the largest individual contribution ($\Delta = 25.0\%$ F1 drop when text is removed), confirming language as the anchor for semantic verification.

---

### Slide 8: Experiments 5 & 6 – Error Analysis & Explainability Quality
- Qualitative error analysis across 15+ difficult cases (subtle vocal vocoders, metaphorical text).
- Explainability evaluation: **68.4% mean faithfulness score** via perturbation tests. Masking top attributed tokens/patches causes direct drops in fake probability.

---

### Slide 9: VoiceBox Web Application Demonstration
- Strict compliance with `voicebox-DESIGN.md`: High-contrast editorial magazine style, Archivo Black typography, zero-radius geometry, flat 2px borders.
- Live multimodal submission with real-time prediction, confidence meter, Grad-CAM heatmap, token attribution highlights, and downloadable verification report.

---

### Slide 10: Ethical Guidelines & Responsible Use
- Defensive verification tool only; no generative deepfake synthesis features.
- Clear disclosure that predictions are probabilistic evidence, not legal absolutes.
- Privacy-first: local processing and zero retention of private personal media.
