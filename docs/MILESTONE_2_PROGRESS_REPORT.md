# MILESTONE-2 PROGRESS REPORT
## CSET-346: Natural Language Processing — Mini Project
**Project Title:** Hate Speech Detection through Explainable AI and Reward-Optimized LLMs  
**Academic Supervisor:** Dr. Pradeepika Verma  
**Evaluation Dates:** 25th – 26th November 2026 | **Venue:** P2-CUB-096  
**Repository:** `https://github.com/samanyu22/Hate-Speech-Detector`  

---

### 1. Project Overview & Formal Problem Formulation
Online hate speech propagation across multilingual social platforms poses severe offline safety and social cohesion risks. Existing NLP moderation systems suffer from three critical bottlenecks: (1) **Context Blindness**, failing to separate direct abuse from counterspeech or quotation critiques; (2) **Linguistic Fragility**, collapsing on code-mixed vernacular (Hinglish) and non-Latin scripts (Devanagari Hindi); and (3) **Unverifiable Black-Box Predictions**, lacking mathematical evidence attribution.

We formulate hate speech detection as a multimodal, context-aware classification objective:
$$\mathcal{F}_\theta(x_i) \to y_i \in \{0, 1\}, \quad \text{where } x_i = \{T_i, V_i, C_i\}$$
where $T_i$ is text in English, Hindi, or Hinglish; $V_i \in \mathbb{R}^{H \times W \times 3}$ is visual meme content; $C_i$ represents conversation thread history; and $y_i \in \{0, 1\}$ denotes Non-Hateful ($0$) vs. Hateful ($1$) speech.

---

### 2. Dataset Description & Empirical Statistics
To ensure robust generalization across linguistic boundaries and modalities, our framework aggregates and harmonizes three standard benchmark corpora with custom structural stress-tests:
1. **HASOC Hi-En (Indo-European Languages):** Real-world social media posts containing native Devanagari Hindi, Romanized Hindi (Hinglish), and English.
2. **HateXplain Benchmark:** Fine-grained, token-annotated rationales with target identity community labels (religion, race, caste, gender).
3. **Multimodal Social Memes Dataset:** Paired meme imagery and typography with spatial OCR annotations targeting multimodal semantic dissonance.
4. **Subtle Structural Benchmark:** Adversarial edge-cases including sarcasm, quotation critiques, and benevolent demographic statements.

#### Table 1: Comprehensive Dataset Statistics & Subgroup Distribution
| Corpus / Partition | Total Samples | Non-Hateful ($y=0$) | Hateful ($y=1$) | Language Breakdown | Modality |
| :--- | :---: | :---: | :---: | :--- | :---: |
| **HASOC Hi-En** | 840 | 490 (58.3%) | 350 (41.7%) | English: 511, Hinglish: 182, Hindi: 147 | Text ($T$) |
| **HateXplain Benchmark** | 210 | 105 (50.0%) | 105 (50.0%) | English (Annotated Targets) | Text ($T$) |
| **Multimodal Memes** | 50 | 25 (50.0%) | 25 (50.0%) | English + Visual Meme Array | Multimodal ($T, V$) |
| **Structural Benchmark** | 210 | 105 (50.0%) | 105 (50.0%) | English / Code-Mixed Counterspeech | Text + Context ($T, C$) |
| **Total Benchmark Suite** | **1,310** | **725 (55.3%)** | **585 (44.7%)** | Stratified 70 / 15 / 15 Train / Val / Test | Multi-Stream |

---

### 3. Current Methodology & Technical Implementation
Our end-to-end pipeline unifies four modular subsystems operating with strict zero-hardcoding:

```
[Input: T, V, C] ──> [Text Normalizer & Script Detector] ──> [BiLSTM + Multi-Head Self-Attention]
                 ──> [EasyOCR & Conv2D Visual Backbone]  ──> [Cross-Modal Cross-Attention Fusion] ──> Logits / y
                 ──> [Leave-One-Out Perturbation XAI]    ──> [Directional AOPC Faithfulness Check]
                 ──> [Multi-Objective Reward Vector]     ──> [Direct Preference Optimization (DPO)]
```

1. **Preprocessing & Script Detection:** Applies Unicode NFKC normalization, URL/mention sanitization, and regex-based Devanagari-to-Latin script ratio detection to route inputs dynamically without error-prone machine translation.
2. **Contextual Attention Transformer:** Tokens are mapped to 128-dim embeddings, encoded via a Bidirectional LSTM ($2d_h = 256$), and processed through Multi-Head Scaled Dot-Product Self-Attention with masked average pooling to prevent padding bias.
3. **Multimodal Fusion:** `EasyOCR` extracts spatial bounding boxes and meme text, while visual feature maps from a convolutional backbone are merged using a gated cross-attention layer: $h_{\text{fused}} = g \odot h_{\text{text}} + (1 - g) \odot h_{\text{attn}}$.
4. **Grounded Explainable AI (XAI):** Mathematically rigorous Leave-One-Out (LOO) perturbation attribution:
   $$\Delta_j = \mathcal{F}_\theta(x) - \mathcal{F}_\theta(x_{\backslash j})$$
   Tokens with $\Delta_j > 0$ represent positive evidence for hate, whereas $\Delta_j < 0$ represent mitigating civil evidence. Faithfulness is scored via Area Over the Perturbation Curve (AOPC).
5. **Reward Optimization & DPO Loop:** Moderator corrections are weighted via a multi-objective reward function:
   $$R(x, y, e, f) = 0.35 R_{\text{cls}} + 0.25 R_{\text{exp}} + 0.15 R_{\text{ctx}} + 0.25 R_{\text{fb}} - 0.50 R_{\text{err}}$$
   Winning vs. losing pairs $(x, y_w, y_l)$ undergo Direct Preference Optimization (DPO) with automated anti-regression rollback guards.

---

### 4. Experimental Results & Performance Analysis
The models were evaluated on the held-out test split (128 stratified samples) and compared across 9 formal research experiments (EXP-1 through EXP-9):

#### Table 2: Comparative Architecture Benchmarks (Held-out Test Evaluation)
| Architecture Backbone | Accuracy | Macro Precision | Macro Recall | Macro F1 | Disparity (EO Gap) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical TF-IDF + Logistic Reg.** | 98.7% | 0.988 | 0.987 | 0.987 | 0.042 | **1.8 ms** (CPU) |
| **Multilingual Attention Transformer** | **100.0%** | **1.000** | **1.000** | **1.000** | **0.000** | **8.4 ms** (CPU) |
| **Multimodal Cross-Attention Fusion** | **100.0%** | **1.000** | **1.000** | **1.000** | **0.000** | **18.8 ms** (CPU) |
| **LLM Structured Reasoner** | 98.1% | 0.982 | 0.980 | 0.981 | 0.015 | 12.1 ms (CPU) |

#### Table 3: Linguistic Fairness & Subgroup Breakdown (EXP-8)
| Subgroup Language | Sample Count | False Positive Rate (FPR) | False Negative Rate (FNR) | Subgroup F1 |
| :--- | :---: | :---: | :---: | :---: |
| **English (Standard Latin)** | 83 | 0.0000 | 0.0000 | 1.0000 |
| **Hindi (Native Devanagari)** | 16 | 0.0000 | 0.0000 | 1.0000 |
| **Hinglish (Code-Mixed Latin)** | 29 | 0.0000 | 0.0000 | 1.0000 |

* **Key Finding:** Classical TF-IDF exhibits false positives on interpersonal insults (e.g. *"You are stupid"* $\to$ 51.7% Hateful) due to bag-of-words pronoun correlation (`you`, `are`). The Multilingual Transformer correctly identifies this as Non-Hateful (81.8%) by verifying the absence of protected identity targeting.
* **XAI Faithfulness:** Achieved an **88.0% - 100.0%** AOPC prediction drop ratio when ablated rationales were removed.

---

### 5. Challenges Encountered & Engineering Solutions
1. **Script Mixing & OOV Collapse:** Code-mixed Hinglish generated high OOV rates in standard tokenizers.  
   *Solution:* Implemented a dual subword character n-gram ($2 \le n \le 5$) and phonetic Hindi lexicon normalizer.
2. **Context Disambiguation (Counterspeech vs. Hate):** Quoting hate speech to condemn it triggered naive toxic keywords.  
   *Solution:* Integrated a structured context gating adapter ($C_i$) isolating thread context from input post text.
3. **Meme Semantic Dissonance:** Civil text paired with hostile imagery bypassed text-only classifiers.  
   *Solution:* Built an intermediate cross-attention layer linking EasyOCR token embeddings with CNN visual feature maps.
4. **Softmax Attribution Saturation:** At extreme confidence ($P \to 1.0$), probability perturbation yielded flat zeros.  
   *Solution:* Introduced logit-space sensitivity fallback when $\Delta P < 0.0005$, preserving faithful attribution heatmaps.

---

### 6. Team Contributions & Responsibilities
* **Member 1 (Model Architecture & Optimization):** Built the PyTorch BiLSTM + Multi-Head Self-Attention network, DPO training harness, and anti-regression validation suite.
* **Member 2 (Multimodal Processing & Datasets):** Curated HASOC and HateXplain corpora, integrated EasyOCR spatial text extraction, and constructed the CNN cross-attention fusion layer.
* **Member 3 (Explainability & Fairness Auditing):** Engineered Leave-One-Out perturbation engine, AOPC faithfulness metric, Equalized Odds fairness testing, and error analysis.
* **Member 4 (Backend API & ThoughtStream UI):** Implemented FastAPI REST routes, SQLite database audit persistence, and the ThoughtStream interactive research dashboard.

---

### 7. Future Work Planned Before Final Evaluation
1. **Lightweight IndicBERT / MuRIL Fine-Tuning:** Benchmark fine-tuned MuRIL transformer weights against current recurrent self-attention backbones.
2. **Streaming Feedback Ingestion:** Enable real-time background worker updates during interactive DPO retraining without service restart.
3. **Cross-Platform Context Scraping:** Extend context adapter to parse Reddit/X multi-hop conversational trees automatically.
4. **Formal ACL Manuscript Finalization:** Complete camera-ready formatting in `paper/main.tex` with final ablation figures.
