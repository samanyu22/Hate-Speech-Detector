# Technical Methodology & Architectural Specification
## Course: CSET 346 – Natural Language Processing

### 1. Overall System Architecture
The verification system follows an 8-stage modular processing pipeline:

```
[ Multimodal Input (Text, Image/Video, Audio) ]
                     │
                     ▼
          [ Input Validation ]
                     │
                     ▼
      [ Modality-Specific Preprocessing ]
  ├── Text Normalization & Tokenization
  ├── Visual Resizing (224x224) & ImageNet Normalization
  └── Audio Resampling (16kHz) & STFT Mel-Spectrogram Extraction
                     │
                     ▼
      [ Modality-Specific Feature Encoders ]
  ├── TextEncoder: 1D-CNN + LayerNorm (128-dim)
  ├── VisionEncoder: 4-Block ConvNet + AdaptiveAvgPool (128-dim)
  └── AudioEncoder: 2D-CNN Mel-Spectrogram (128-dim)
                     │
                     ▼
       [ Cross-Modal Consistency Analysis ]
  ├── Cosine Alignment via Contrastive Projections
  └── Semantic Mismatch / Contradiction Diagnostics
                     │
                     ▼
           [ Multimodal Fusion ]
  ├── Strategy A: Feature-Level Fusion (Bilinear Interaction)
  ├── Strategy B: Bidirectional Cross-Attention Transformer
  └── Strategy C: Evidential Decision-Level Late Fusion
                     │
                     ▼
     [ Confidence Calibration & Uncertainty Engine ]
  ├── Temperature Scaling & Normalized Shannon Entropy
  └── Abstention Logic: REAL / FAKE / UNCERTAIN
                     │
                     ▼
       [ Explainability & Evidence Grounding ]
  ├── Text Token Attribution via Leave-One-Out Perturbation
  ├── Visual Spatial Heatmaps via Grad-CAM Saliency
  └── Faithfulness Evaluation via Perturbation Drop Tests
                     │
                     ▼
         [ Final Verification Report ]
```

---

### 2. Modality Encoders
1. **Text Encoder (`TextEncoder`):**
   - Vocabulary: 5,000 subword/token entries with special tokens `<PAD>`, `<UNK>`, `<CLS>`, `<SEP>`.
   - Embedding Dimension: 64.
   - Temporal Convolution: Multi-scale kernel sizes ($k_1=3, k_2=5$) with 128 feature channels.
   - Global Adaptive Max-Pooling concatenates local n-gram activations into a 128-dimensional representation normalized by LayerNorm.

2. **Visual Encoder (`VisionEncoder`):**
   - Input: $3 \times 224 \times 224$ normalized RGB tensor.
   - Backbone: 4 Conv2d blocks with batch normalization, ReLU, and progressive downsampling ($112 \to 56 \to 28 \to 14 \to 7$).
   - Output: 128-dimensional embedding extracted via $1 \times 1$ Adaptive Average Pooling and dense linear projection.

3. **Acoustic Encoder (`AudioEncoder`):**
   - Input: $1 \times 64 \times 96$ log Mel-spectrogram representing 3.0s of 16kHz audio.
   - Backbone: 3-stage 2D convolutional filterbank capturing time-frequency harmonic energy.
   - Output: 128-dimensional acoustic latent vector.

---

### 3. Multimodal Fusion Formulations
1. **Feature-Level Fusion:**
   Projects individual embeddings into a shared space, appends cross-bilinear interaction terms:
   $$Z_{fused} = [z_t \,\|\, z_v \,\|\, z_a \,\|\, (z_t \odot z_v) \,\|\, (z_t \odot z_a)] \in \mathbb{R}^{640}$$
   Fed into a 3-layer MLP with LayerNorm, ReLU, and Dropout (0.3).

2. **Cross-Attention Fusion (Ours):**
   Treats modality embeddings as sequence tokens $S = [T, V, A] \in \mathbb{R}^{3 \times 128}$.
   Uses 4-head scaled dot-product multi-head attention:
   $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
   Enables language representations to actively query visual facial boundaries and acoustic harmonics.

3. **Missing Modality Handling:**
   When visual or acoustic evidence is omitted, missing modality slots are imputed via learned continuous zero-mean latent vectors $\mathbf{e}_{\text{empty}} \sim \mathcal{N}(0, 0.02^2)$.

---

### 4. Uncertainty Estimation & Abstention Engine
To prevent overconfident false verifications, the system computes Normalized Shannon Entropy:
$$H_N(P) = -\frac{P(\text{REAL})\log_2 P(\text{REAL}) + P(\text{FAKE})\log_2 P(\text{FAKE})}{\log_2(2)} \in [0, 1]$$

**Abstention Decision Rules:**
A sample is assigned the `UNCERTAIN` label if:
1. Model confidence falls below threshold: $\max_c P(c) < \tau_{\text{conf}}$ (default: 0.60).
2. Prediction entropy exceeds ambiguity threshold: $H_N(P) > \tau_{\text{entropy}}$ (default: 0.65).
3. Cross-modal contradiction is detected ($S_{cross} < 0.05$) while single-modality baselines strongly disagree ($\sigma_{\text{baselines}} > 0.25$).
4. Critical missing modalities preclude reliable corroboration.
