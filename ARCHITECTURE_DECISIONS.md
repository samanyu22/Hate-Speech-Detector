# Architecture Decisions Record (ADR)
## Project: Hate Speech Detection through Explainable AI and Reward Optimized LLMs
**Academic Supervisor:** Dr. Pradeepika Verma  
**Formulation:** $x_i = \{T_i, V_i, C_i\} \implies f_\theta(x_i) \to y_i \in \{0, 1\}$  
**Date:** November 2026

---

### ADR-01: Text Classification & Multilingual Backbone Selection
- **Problem Being Solved:** Classification of social media text ($T_i$) across English, Hindi, and Hindi-English code-mixed (Hinglish) posts, overcoming low-resource and script-switching hurdles.
- **Candidate Approaches:**
  1. Monolingual English BERT / RoBERTa.
  2. Multilingual BERT (`mBERT`).
  3. XLM-RoBERTa (`xlm-roberta-base`).
  4. IndicBERT & MuRIL (Multilingual Representations for Indian Languages).
- **Selected Approach:** Dual-Tier Backbone:
  - *Tier 1 (Classical NLP Baseline):* TF-IDF with character n-grams ($2 \le n \le 5$) and word n-grams ($1 \le n \le 2$) + Logistic Regression and Linear SVM with class weighting.
  - *Tier 2 (Deep Transformer Backbone):* A flexible transformer classification head supporting XLM-RoBERTa and MuRIL/IndicBERT with code-mixed tokenization adaptation and transliteration awareness.
- **Reason for Selection:** MuRIL and XLM-RoBERTa were pretrained explicitly on cross-lingual Indian corpora, retaining subword representations for both Devanagari script and Latin transliterated Hindi.
- **Alternatives Rejected:** Monolingual RoBERTa fails on code-mixed Hinglish; pure machine translation to English loses sarcasm, cultural nuances, and abusive innuendo.
- **Computational Requirements:** Compatible with CPU / consumer GPU inference (batch size 16-32, FP32/FP16).
- **Limitations:** Subword fragmentation on highly distorted slang.
- **Research Justification:** Conforms to HASOC and HateXplain benchmark literature.

---

### ADR-02: Multimodal Architecture & Visual Feature Extraction
- **Problem Being Solved:** Multimodal content ($V_i$) such as memes where text and imagery create synergistic hate speech not detectable by text alone.
- **Candidate Approaches:**
  1. OCR-only text pipeline.
  2. Late fusion (averaging independent vision and text logits).
  3. Intermediate Feature Fusion with Cross-Attention between visual embeddings and text tokens.
- **Selected Approach:** Hybrid Cross-Attention & Feature-Level Fusion:
  - Visual encoder: Lightweight ResNet / ViT / CLIP-compatible visual backbone extracting 512-dim visual features.
  - OCR Engine: `EasyOCR` extracting embedded meme text with spatial bounding coordinates and confidence scores.
  - Fusion Module: Cross-Modal Cross-Attention layer projecting visual tokens and text tokens into a shared contextual space with gated residual connections.
- **Reason for Selection:** Pure OCR fails when the meme image supplies the derogatory context (e.g. associating an animal image with a marginalized group); early/intermediate cross-attention enables capturing benign-text/malicious-image interactions.
- **Alternatives Rejected:** Naive logit averaging cannot model visual-semantic contradictions or sarcasm.
- **Computational Requirements:** Torch CPU/GPU, ~150MB model footprint for lightweight visual projection.
- **Limitations:** Dependent on OCR accuracy for degraded or stylized meme typography.

---

### ADR-03: Contextual Representation ($C_i$)
- **Problem Being Solved:** Modeling conversation thread history, platform metadata, and reply chains without demographic profiling.
- **Candidate Approaches:**
  1. Ignoring context (treating posts in isolation).
  2. Concatenating context text directly to post text.
  3. Structured Context Encoder with hierarchical attention and sensitivity masking.
- **Selected Approach:** Structured Contextual Modality Adapter:
  - Preceding dialogue context and post metadata are encoded with a distinct segment marker `[CONTEXT]` and passed through an auxiliary context gating vector.
  - Explicit ethics constraint: No demographic, religious, or identity attributes are extracted or used as predictive features.
- **Reason for Selection:** Distinguishes counterspeech, educational quotation, and sarcastic retorts from genuine hate speech.
- **Alternatives Rejected:** Demographic profiling introduces severe ethical biases and violates responsible AI principles.

---

### ADR-04: Explainable AI (XAI) Framework
- **Problem Being Solved:** Deep learning models function as opaque black boxes; decisions must provide verifiable, faithful, token- and region-level attribution.
- **Candidate Approaches:**
  1. Post-hoc LLM text summary only (no mathematical grounding).
  2. Attention weights visualization (known to be non-faithful).
  3. Integrated Gradients + Perturbation-based Local Token Attribution + Visual Grad-CAM / OCR Bounding attribution.
- **Selected Approach:** Multi-Faceted Grounded Attribution Engine:
  - *Text Attribution:* Integrated Gradients (approximate path integral of gradients along a baseline) paired with leave-one-out perturbation attribution for token-level positive/negative contribution scores.
  - *Visual Attribution:* Spatial saliency / Grad-CAM overlays mapped onto OCR bounding regions.
  - *Natural Language Synthesis:* Structured explanation combining top attributed tokens, detected polarity, OCR evidence, and confidence intervals without hallucinating facts outside the input.
  - *Faithfulness Validation:* Automated post-hoc ablation testing (removing top-k attributed tokens and computing prediction drop $\Delta f(x)$).
- **Reason for Selection:** Provides both mathematical rigor (gradients and perturbation) and human interpretability.
- **Alternatives Rejected:** Standalone LLM generation without attribution hallucinated evidence not present in input.

---

### ADR-05: Reward Optimized LLM & Feedback Alignment
- **Problem Being Solved:** The core mandate of Dr. Verma's specification: iterative model improvement through human feedback and reward optimization.
- **Candidate Approaches:**
  1. Synthetic RLHF without true human feedback.
  2. Thumbs-up counter with static weights.
  3. Dynamic Multi-Objective Reward Modeling + Preference Optimization (Direct Preference Optimization - DPO / Preference SFT with reward-weighted ranking).
- **Selected Approach:** End-to-End Feedback-Driven Reward Optimization Engine:
  - Formulated Reward Function:
    $$R(x, y, e, f) = \alpha R_{\text{cls}}(y, y^*) + \beta R_{\text{exp}}(e, A) + \gamma R_{\text{ctx}}(C) + \delta R_{\text{fb}}(f) - \lambda R_{\text{err}}$$
  - Preference Dataset Builder: Converts human feedback corrections into structured preference pairs $(x, y_w, y_l)$ where $y_w$ is the accepted/corrected output and $y_l$ is the rejected/incorrect output.
  - Optimization Engine: Executes actual optimization cycles (Preference SFT / DPO loss) updating checkpoint weights, versioning the model ($v1 \to v2$), and re-evaluating on held-out test splits.
  - Rollback Mechanism: Automatic performance guard ensures worse-performing models are quarantined and earlier stable checkpoints restored.
- **Reason for Selection:** Real mathematical optimization with verifiable before/after metrics on held-out evaluation sets.
- **Alternatives Rejected:** Superficial feedback logging without weight updating or reward-guided retraining.

---

### ADR-06: Backend & Database Architecture
- **Problem Being Solved:** High-throughput async REST API with long-running training job orchestration and persistent audit logging.
- **Candidate Approaches:**
  1. Flask + SQLite.
  2. Django + PostgreSQL.
  3. FastAPI + Async Background Tasks + SQLite / JSON Manifest persistence.
- **Selected Approach:** FastAPI with Pydantic v2 schemas, SQLite persistent store, ThreadPool background task executor, and modular service-repository design.
- **Reason for Selection:** Asynchronous non-blocking endpoints, auto-generated OpenAPI documentation, zero external database daemon requirements for seamless local academic demonstration.

---

### ADR-07: Frontend Design System (ThoughtStream + Dark Mode)
- **Problem Being Solved:** Professional, distraction-free research UI for reviewing hate speech without eye strain, clutter, or toy demos.
- **Candidate Approaches:**
  1. Generic Bootstrap/Tailwind UI kit.
  2. ThoughtStream minimalist design system with comprehensive Dark Mode extensions.
- **Selected Approach:** ThoughtStream Architecture:
  - Zen-like typographic hierarchy (Libre Baskerville display, Inter UI, Source Code Pro monospace).
  - Flat hairline borders (`#E7E5E4` / `#292524`), 0px border radius, no decorative drop shadows.
  - Full Dark Mode toggle with Stone/Warm Black palette (`#0C0A09` background, `#1C1917` surface, `#FAFAF9` text).
  - 11 dedicated research laboratories covering Detection, Datasets, Preprocessing, Models, XAI, Rewards, Bias/Fairness, Experiments, Reports, Settings, and System Overview.

