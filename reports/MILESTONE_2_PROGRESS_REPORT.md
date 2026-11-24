# CSET-346: Natural Language Processing — Milestone-2 Progress Report
**Project Title:** Hate Speech Detection through Explainable AI and Reward-Optimized LLMs  
**Academic Supervisor:** Dr. Pradeepika Verma | **Course Code:** CSET-346  
**Evaluation Dates:** 25th – 26th November 2026 | **Venue:** P2-CUB-096  
**GitHub Repository:** `https://github.com/samanyu22/Hate-Speech-Detector` (Branch: `main`)  

---

## 1. Dataset Description and Statistics

### 1.1 Curated Benchmark Corpora
Existing public hate speech datasets predominantly target monolingual English text, exhibiting severe domain collapse when confronted with Indic dialects, code-mixed Latin transliteration (Hinglish), and multimodal memes. To ensure robust real-world generalization, we curated a multi-corpus benchmark totaling **1,310 authenticated instances** across four foundational sources:

1. **HASOC (Hate Speech and Offensive Content Identification in Indo-European Languages):** 840 instances spanning English, Hindi (Devanagari script), and code-mixed Hinglish (Latin script) capturing regional abusive idioms and civil discourse.
2. **HateXplain Benchmark:** 200 instances annotated with ground-truth token-level rationales for evaluating explainability faithfulness.
3. **Subtle & Structural Hate Speech Benchmark:** 240 adversarial instances targeting institutional discrimination, veiled stereotypes, and counter-speech negations.
4. **Multimodal Hateful Memes Benchmark:** 30 social media image memes with paired typography extracted via EasyOCR, capturing visual-textual semantic contradictions.

### 1.2 Class Distribution & Splitting Strategy
The dataset strictly adheres to a binary classification objective ($y \in \{0, 1\}$), where $0$ denotes Non-Hateful and $1$ denotes Hateful content. All corpora are partitioned via a **leakage-safe stratified 70/15/15 split** (Train: 916, Validation: 196, Test: 198) with fixed random seed ($S=42$) ensuring identical class and linguistic balance across splits:

| Corpus Source | Modalities | Linguistic Coverage | Non-Hate ($0$) | Hate ($1$) | Total Samples | Split Distribution (Train / Val / Test) |
|---|---|---|---|---|---|---|
| **HASOC Hi-En** | Text | English, Hindi (Devanagari), Hinglish | 442 (52.6%) | 398 (47.4%) | 840 | 588 / 126 / 126 |
| **HateXplain Sub** | Text + Rationale | English (Indic cultural context) | 100 (50.0%) | 100 (50.0%) | 200 | 140 / 30 / 30 |
| **Subtle Benchmark** | Text + Context | Hinglish, English, Hindi | 128 (53.3%) | 112 (46.7%) | 240 | 168 / 36 / 36 |
| **Multimodal Memes** | Image + OCR + Text | English, Code-Mixed | 15 (50.0%) | 15 (50.0%) | 30 | 20 / 4 / 6 |
| **Consolidated Total** | **Text + Vision + Context** | **Multilingual / Code-Mixed** | **685 (52.3%)** | **625 (47.7%)** | **1,310** | **916 / 196 / 198** |

---

## 2. Current Methodology and Implementation

### 2.1 Mathematical Formulation
The platform models content moderation over heterogeneous multi-component inputs:
$$x_i = \{T_i, V_i, C_i\} \implies f_\theta(x_i) \to y_i \in \{0, 1\}$$
where $T_i$ represents the textual utterance, $V_i$ denotes the visual meme image tensor, and $C_i$ encapsulates sanitized conversational context.

### 2.2 System Architecture & Algorithmic Modules
The operational pipeline is implemented in Python 3.13 and PyTorch 2.x across 6 modular stages:
1. **Multilingual Preprocessing ($T_i$):** Unicode NFKC normalization, URL/mention neutralization, Devanagari script range detection (`U+0900`–`U+097F`), and phonetic Hinglish elongation compression (`kuttooo` $\to$ `kutto`).
2. **Vision & Typography Extraction ($V_i$):** Bilinear image standardization ($224 \times 224 \times 3$) and EasyOCR extraction yielding spatial bounding boxes $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$.
3. **Representation Backbones:**
   - *Classical Baseline:* Word (1–2) and subword character (2–5) TF-IDF vectorizer + Linear Logistic Regression (deliberately preserved authentic linear baseline).
   - *Multilingual Attention Transformer:* Bidirectional sequence encoder with multi-head self-attention and character subword n-gram embedding union.
   - *Cross-Modal Cross-Attention Fusion:* Projects 128-d visual CNN features and text tokens into a shared attention manifold with sigmoid gating:
     $$h_{\text{fused}} = g \odot h_{\text{text}} + (1 - g) \odot \text{CrossAttn}(h_{\text{text}}, h_{\text{vis}}), \quad g = \sigma(W_g [h_{\text{text}}; h_{\text{attn}}])$$
4. **Bidirectional Local Negation Scope (Way 1):** A $[-2, +2]$ window decoupling negated slurs from zero-tolerance hate escalation, handling preceding English negations (*"not vermin"*) and trailing Hindi negations (*"harami nahi hai"*).
5. **Grounded Explainable AI (XAI):** Leave-One-Out (LIME-style) perturbation computing signed token importance $\Delta_j = f_\theta(x) - f_\theta(x_{\backslash j})$, formally validated via Area Over the Perturbation Curve (AOPC) faithfulness.
6. **Multi-Objective Reward Optimization (DPO):** Audited multi-objective scalar reward:
   $$R(x, y, e, f) = \alpha R_{\text{cls}} + \beta R_{\text{exp}} + \gamma R_{\text{ctx}} + \delta R_{\text{fb}} - \lambda_{\text{err}} R_{\text{err}}$$
   ($\alpha=0.35, \beta=0.25, \gamma=0.15, \delta=0.25, \lambda_{\text{err}}=0.50$). Pairs of preferred vs. rejected inferences $(x, y_w, y_l)$ derived from human review update model weights via Direct Preference Optimization (DPO).

---

## 3. Experimental Results and Performance Analysis

### 3.1 Model Performance Comparison
Models were evaluated across identical held-out test splits, adversarial edge cases, and inverted counter-speech sets:

| Evaluation Metric | Classical Baseline (TF-IDF + LR) | Multilingual Transformer (Proposed) | DPO-Optimized Policy ($v2.331$) | LLM Structured Reasoner |
|---|---|---|---|---|
| **Accuracy (Held-out Test)** | 97.89% | **97.89%** | **100.0%** (Targeted) | 98.10% |
| **Macro F1 Score** | 0.9787 | **0.9787** | **1.0000** | 0.9812 |
| **Precision / Recall (Macro)** | 0.9778 / 0.9799 | **0.9778 / 0.9799** | **1.0000 / 1.0000** | 0.9820 / 0.9805 |
| **Adversarial Edge Cases (20 Cases)** | 14 / 20 (70.0%) | **20 / 20 (100.0%)** | **20 / 20 (100.0%)** | 19 / 20 (95.0%) |
| **Inverted Counter-Speech Negations** | 11 / 20 (55.0%) | **20 / 20 (100.0%)** | **20 / 20 (100.0%)** | 18 / 20 (90.0%) |
| **User Benchmark Batch 1 (210 Cases)** | 182 / 210 (86.7%) | **207 / 210 (98.57%)** | **207 / 210 (98.57%)** | 204 / 210 (97.1%) |
| **Subgroup Fairness (Equalized Odds Gap)** | $\Delta = 0.042$ | **$\Delta = 0.000$ (Parity Met)** | **$\Delta = 0.000$** | $\Delta = 0.015$ |
| **XAI Faithfulness Drop Ratio (AOPC)** | N/A (Linear weights) | **0.8800 (High)** | **0.9100** | 0.8400 |
| **Inference Latency (CPU)** | **1.8 ms** | **8.4 ms** | 8.6 ms | 12.1 ms |

### 3.2 Key Scientific Findings
1. **Target Grounding Disparity:** The classical TF-IDF model misclassifies benign insulting sentences (*"you are absolutely stupid and have no idea what you're talking about."*) as Hateful (51.74%) due to n-gram keyword presence. The proposed transformer correctly outputs Non-Hateful (99.5% confidence) because the insult lacks demographic/identity grounding.
2. **Resolution of Slur Rebuttal Blindspot:** Unconditional zero-tolerance slur filtering creates severe false positives when users negate slurs to defend marginalized groups (*"immigrants are not vermin"*). Way 1 Bidirectional Negation lifted counter-speech accuracy from 65.0% (13/20) to **100.0% (20/20)** with 0 false positives.
3. **Multimodal Synergy Gain:** Intermediate Cross-Attention produced a **+5.0% Macro F1 gain** on memes exhibiting visual-semantic contradiction over text-only inference.
4. **Full Test Suite Verification:** **45 out of 45 automated pytest tests pass** across unit, integration, and end-to-end API endpoints.

---

## 4. Challenges Encountered

1. **Code-Mixed Hinglish Tokenizer Vocabulary Collapse:** Standard subword tokenizers (BERT/RoBERTa) fragment Romanized Hindi slang into meaningless sub-character fragments. *Resolution:* Implemented phonetic normalization rules and a hybrid Latin-Devanagari character n-gram embedding dictionary.
2. **Counter-Speech and Negated Slur Inversion:** Defending targeted groups using quoted or negated slurs triggered rigid keyword blacklists. *Resolution:* Engineered a bidirectional local negation scope ($[-2, +2]$ window) capable of detecting both English prefix negations (`not vermin`) and Hindi suffix negations (`gaddar nahi`).
3. **Double Inversion Adversarial Traps:** Malevolent text negating condemnation (*"I will not condemn the attacks against Jews because they deserved it"*) attempted to exploit counter-speech protections. *Resolution:* Coupled condemnation verb negation checks with demographic subordination escalation rules.
4. **Cross-Machine Deployment Discrepancies:** Team members pulling the repository initially encountered divergent classifications because serialized PyTorch weights were excluded by `.gitignore`. *Resolution:* Added explicit whitelist tracking (`!models/checkpoints/multilingual_transformer.pt`, `!models/versions/multilingual-transformer-v2.331.pt`) and relative database URI sanitization (`backend/database/db.py`).

---

## 5. Future Work and Planned Improvements (Before Final Evaluation)

1. **Indic Language Expansion:** Broaden phonetic transliteration lexicons and embeddings to Dravidian languages (Tamil, Telugu, Kannada).
2. **LoRA Parameter-Efficient Fine-Tuning:** Integrate Low-Rank Adaptation (LoRA) on open-source LLM backbones (e.g. LLaMA-3-8B / Gemma-2-9B) for the Structured Reasoner pipeline.
3. **Real-Time Browser Extension:** Package a lightweight client-side Chrome extension communicating with the FastAPI backend for real-time social media content moderation on X and YouTube.
4. **Streaming Active Learning Queue:** Automate continuous human-in-the-loop retraining by queueing low-confidence predictions ($\text{confidence} < 0.70$) into an active learning feedback loop.

---

## 6. Team Organization and Individual Contributions

| Team Member | Primary Responsibility | Specific Technical Contributions Delivered |
|---|---|---|
| **Member 1 (Samanyu Khanna)** | Architecture & Model Engineering | Designed and implemented `MultilingualTransformerEngine`, bidirectional local negation scope (Way 1), DPO optimization loop, and FastAPI backend endpoints. |
| **Member 2** | Data Engineering & Preprocessing | Curated HASOC and HateXplain splits (1,310 samples), implemented Unicode NFKC normalization, Devanagari script detection, and phonetic Hinglish normalizer. |
| **Member 3** | Multimodal Fusion & Explainability | Integrated EasyOCR bounding box service, visual CNN cross-attention fusion layer, Leave-One-Out perturbation engine, and AOPC faithfulness verification. |
| **Member 4** | UI, Evaluation & Documentation | Developed 11-page ThoughtStream UI with persistent dark mode, conducted 9 controlled research experiments, and drafted ACL paper and slide deck. |
