# System Architecture
## Hate Speech Detection through Explainable AI and Reward Optimized LLMs

```
+-----------------------------------------------------------------------------------+
|                         SOCIAL MEDIA INPUT: x = {T, V, C}                         |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                           INPUT VALIDATION & MODALITY DETECTION                   |
|  - Script Breakdown: Devanagari [\u0900-\u097F] vs Latin [A-Za-z]                 |
|  - Hinglish Romanized Lexicon Identification                                      |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                         MODALITY-SPECIFIC PREPROCESSING                           |
|  - Text: Unicode NFKC, Hashtag Segmentation, Anonymization                        |
|  - Visual: EasyOCR Typography Extraction & Spatial Normalization (224x224)       |
|  - Context: Thread History Anonymization & Demographic Neutralization             |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                     FEATURE EXTRACTION & CROSS-ATTENTION FUSION                   |
|  - Text Feature Projection (128-dim)                                              |
|  - Visual Convolutional Backbone (128-dim)                                        |
|  - Scaled Dot-Product Cross-Attention: Attn(Q=W_q*t, K=W_k*v, V=W_v*v)            |
|  - Adaptive Gated Residual Fusion: g = sigmoid(W_g [t; Attn])                     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                      HATE SPEECH CLASSIFICATION ENGINE (f_theta)                  |
|  - Binary Objective: 0 = Non-Hateful, 1 = Hateful                                 |
|  - Calibrated Softmax Probabilities & Temperature Scaling                         |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                            EXPLAINABILITY (XAI) ENGINE                            |
|  - Leave-One-Out Token Perturbation Attribution                                   |
|  - Positive Evidence (Hostility) vs Negative Evidence (Mitigation)                |
|  - AOPC Faithfulness Verification via Top-Token Ablation                          |
|  - Visual Grounding mapped to EasyOCR Bounding Boxes                              |
|  - 6-Point Natural Language Evidence Synthesis                                    |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       HUMAN FEEDBACK & AUDIT COLLECTION                           |
|  - Reviewer Accuracy Confirmation, Corrected Labels, Rating (1-5), Slang Tags     |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                       MULTI-OBJECTIVE REWARD CALCULATION                          |
|  R = alpha*R_cls + beta*R_exp + gamma*R_ctx + delta*R_fb - lambda_err*R_err       |
+-----------------------------------------------------------------------------------+
                                          |
                                          v
+-----------------------------------------------------------------------------------+
|                    REWARD OPTIMIZATION ENGINE (DPO / SFT)                         |
|  - Preference Dataset Construction: (x, y_w, y_l)                                 |
|  - Direct Preference Optimization (DPO) Loss Update                               |
|  - Versioned Checkpoint Promotion with Held-Out Test Anti-Regression Guard        |
+-----------------------------------------------------------------------------------+
```

