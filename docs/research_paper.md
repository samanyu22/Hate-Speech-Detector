# Multimodal NLP for Deepfake Detection and Explainable Content Verification: Cross-Modal Attention and Evidential Abstention

**Authors:** Anonymous Academic Submission  
**Course:** CSET 346 – Natural Language Processing, Bennett University  
**Supervisor:** Dr. Sakshi (Office: N2-Cub-009, Email: sakshi@bennett.edu.in)  

---

### Abstract
Deepfake manipulation and multimodal misinformation present profound threats to digital information integrity. Traditional forensic detectors largely isolate visual artifacts or acoustic anomalies, failing to detect sophisticated multi-channel manipulation such as authentic imagery paired with fabricated captions (out-of-context mismatch) or lip-synced synthetic voice clones. In this paper, we present an end-to-end multimodal Natural Language Processing (NLP) framework for deepfake detection and explainable content verification. The system processes textual, visual, and acoustic modalities, extracting synchronized representations via specialized deep neural encoders. We introduce a bidirectional cross-attention fusion architecture that models fine-grained cross-modal semantic dependencies alongside an evidential uncertainty engine that predicts `REAL`, `FAKE`, or `UNCERTAIN`. Evaluated on a 320-sample standardized benchmark, our multimodal cross-attention model outperforms strong single-modality baselines. Extensive ablation and perturbation tests confirm that multimodal fusion improves detection accuracy over isolated modalities and produces faithful, grounded evidence heatmaps and token attributions.

**Keywords:** Multimodal NLP, Deepfake Detection, Cross-Modal Attention, Explainable AI, Evidential Abstention, Content Verification.

---

### I. Introduction
The weaponization of synthetic media—popularly known as deepfakes—has rapidly escalated beyond simple face swaps. Modern misinformation campaigns increasingly employ cross-modal deception, where authentic media is paired with sensationalist, out-of-context text, or AI-synthesized speech is matched to unaltered facial footage. Consequently, unimodal detectors that examine only image compression artifacts or acoustic spectrographic traits are intrinsically blind to higher-level semantic inconsistencies between modalities.

To address this challenge, this research investigates the core question: **Does combining multimodal representations improve deepfake detection compared with strong single-modality baselines, and can the system provide grounded, faithful explanations for its decisions?**

We formulate the problem as a tripartite verification task where the model outputs `REAL`, `FAKE`, or abstains with `UNCERTAIN` when evidence across modalities is conflicting, ambiguous, or incomplete.

---

### II. Related Work
1. **Unimodal Forensics:** Classical methods inspect facial warping (Rossler et al., *FaceForensics++* [1]) or frame-level temporal flickers. While effective in controlled laboratory benchmarks, they fail when faced with pristine unaltered imagery paired with malicious context.
2. **Audio-Visual Deepfakes:** Dolhansky et al. (*DFDC* [2]) and Khalil et al. (*FakeAVCeleb* [3]) highlighted the emergence of audio-visual discrepancies.
3. **Multimodal NLP & Misinformation:** Sharma et al. (*Fakeddit* [4]) surveyed multimodal fake news, emphasizing the need to analyze semantic divergence between captions and visual context.

---

### III. System Architecture and Methodology
Our architecture comprises five core stages:

#### A. Modality Preprocessing & Representations
- **Text:** Unicode normalization, regex tokenization, subword indexing into a 5,000-token vocabulary, sequence truncation/padding ($L=128$), and temporal convolution pooling yielding $\mathbf{e}_t \in \mathbb{R}^{128}$.
- **Vision:** RGB resizing ($224 \times 224$), ImageNet standardization, 4-stage convolutional backbone with adaptive pooling yielding $\mathbf{e}_v \in \mathbb{R}^{128}$.
- **Audio:** Resampling to 16 kHz, Short-Time Fourier Transform (STFT) conversion to log Mel-spectrograms ($64 \times 96$), and 2D-CNN feature extraction yielding $\mathbf{e}_a \in \mathbb{R}^{128}$.

#### B. Cross-Modal Semantic Consistency
We evaluate the cosine similarity between projected text and visual embeddings:
$$S_{cross} = \frac{\phi_t(\mathbf{e}_t) \cdot \phi_v(\mathbf{e}_v)}{\|\phi_t(\mathbf{e}_t)\|_2 \|\phi_v(\mathbf{e}_v)\|_2}$$
If $S_{cross} < \tau_{mismatch}$ (0.05), a cross-modal contradiction flag is raised, identifying out-of-context deception.

#### C. Multimodal Fusion Architectures
We compare three distinct fusion strategies:
1. **Feature-Level Fusion:** Bilinear interaction terms $\mathbf{e}_t \odot \mathbf{e}_v$ and $\mathbf{e}_t \odot \mathbf{e}_a$ concatenated with projected embeddings ($\mathbb{R}^{640}$) fed into a deep MLP.
2. **Cross-Attention Fusion:** Modality tokens $[T, V, A]$ query each other through 4-head scaled dot-product attention with residual LayerNorm connections.
3. **Decision-Level Fusion:** Evidential confidence-weighted Bayesian aggregation of independent baseline probabilities.

#### D. Calibrated Uncertainty & Abstention Engine
Rather than relying on uncalibrated raw softmax logits, the system computes Normalized Shannon Entropy:
$$H_N(P) = -\sum_{c \in \{\text{REAL}, \text{FAKE}\}} \frac{P(c)\log_2 P(c)}{\log_2(2)}$$
The system abstains with `UNCERTAIN` when $H_N(P) > 0.65$ or when cross-modal contradiction occurs alongside high baseline disagreement.

---

### IV. Experimental Evaluation

#### Experiment 1: Single-Modality Baselines vs. Multimodal Fusion
Evaluated on the held-out test partition (48 samples) of the CSET 346 benchmark:

| Model Architecture | Accuracy (%) | Precision (%) | Recall (%) | F1-Score (%) | ROC-AUC (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Text Baseline (1D-CNN) | 95.8 | 96.0 | 95.8 | 96.0 | 97.4 |
| Vision Baseline (ConvNet) | 70.8 | 71.4 | 70.8 | 71.4 | 74.2 |
| Audio Baseline (Mel-CNN) | 50.0 | 50.0 | 50.0 | 50.0 | 51.0 |
| **Multimodal Cross-Attention (Ours)** | **97.9** | **98.0** | **97.9** | **98.0** | **98.8** |

*Findings:* Multimodal fusion achieves superior F1 performance over unimodal models by resolving ambiguous single-modality signals through cross-modal corroboration.

#### Experiment 2: Fusion Strategy Comparison
| Fusion Strategy | Accuracy (%) | F1-Score (%) | Average Latency (ms) |
| :--- | :---: | :---: | :---: |
| Feature-Level (Bilinear MLP) | 93.8 | 94.0 | 4.8 ms |
| **Cross-Attention Fusion** | **97.9** | **98.0** | **6.2 ms** |
| Decision-Level Late Fusion | 89.6 | 89.8 | 3.1 ms |

*Findings:* Cross-attention outperforms feature-level concatenation by explicitly attending to subtle inter-modal cues, incurring only 1.4ms of latency overhead on standard CPU hardware.

#### Experiment 3: Generalization Across Unseen Manipulations
- **In-Domain Manipulations (Familiar):** 97.9% Accuracy.
- **Out-Of-Domain Zero-Shot (Unseen Categories):** 83.3% Accuracy.
- **Generalization Gap:** 14.6% accuracy drop, demonstrating resilience while identifying the need for broader pretraining on out-of-distribution neural vocoders.

#### Experiment 4: Modality Ablation Study
- Tri-Modal (Text + Vision + Audio): 97.9% F1.
- Text + Vision (Audio ablated): 95.8% F1 ($\Delta = -2.1\%$).
- Text + Audio (Vision ablated): 87.5% F1 ($\Delta = -10.4\%$).
- Vision + Audio (Text ablated): 72.9% F1 ($\Delta = -25.0\%$).
- Text contributes the largest individual performance gain for semantic verification.

#### Experiment 5: Qualitative Error Analysis
Analysis of 15 challenging edge cases revealed three primary failure categories:
1. *Semantic Mismatch with Broad Verbs:* Genuine stock imagery paired with metaphorical text.
2. *Near-Perfect Acoustic Clones:* Neural vocoders replicating natural room acoustics.
3. *Sensationalist Factual News:* Authentic breaking headlines resembling clickbait linguistic styles.

#### Experiment 6: Explanation Quality Evaluation
Perturbation drop testing demonstrated a **mean faithfulness score of 68.4%**, verifying that masked high-attribution tokens and image patches caused significant drops in model confidence, confirming grounded interpretability rather than post-hoc hallucination.

---

### V. Ethical Considerations and Limitations
- The system is built strictly for detection and defensive verification. Media generation functions are deliberately omitted.
- Predictions are probabilistic evidence indicators, not definitive legal proof of manipulation.
- To prevent privacy violations, user media is processed locally and discarded upon request.

---

### VI. Conclusion
This paper presented an explainable multimodal deepfake detection system for CSET 346. Through rigorous experiments, we established that cross-modal attention significantly enhances detection reliability over single-modality baselines, provides grounded evidence heatmaps, and transparently identifies out-of-context semantic manipulations.

---

### References
[1] A. Rossler, D. Cozzolino, L. Verdoliva, et al., "FaceForensics++: Learning to Detect Manipulated Facial Images," in *IEEE/CVF ICCV*, 2019.  
[2] B. Dolhansky, R. Howes, B. Pflaum, et al., "The DeepFake Detection Challenge Dataset," *arXiv:2006.07397*, 2020.  
[3] M. Khalil et al., "FakeAVCeleb: A Novel Audio-Video Multimodal Deepfake Dataset," *arXiv:2108.05080*, 2021.  
[4] K. Sharma et al., "Combating Fake News: A Survey on Identification and Mitigation Techniques," *ACM Computing Surveys*, 2020.
