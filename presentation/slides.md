# Hate Speech Detection through Explainable AI and Reward Optimized LLMs
### Academic Viva Presentation
**Supervisor:** Dr. Pradeepika Verma  
**Formulation:** $x_i = \{T_i, V_i, C_i\} \implies f_\theta(x_i) \to y_i \in \{0, 1\}$  
**Date:** November 2026

---

## Slide 1: Title & Overview
- **Project Title:** Hate Speech Detection through Explainable AI and Reward Optimized LLMs
- **Research Scope:** Multilingual, Code-Mixed (Hinglish), Multimodal (Social Memes), and Context-Aware AI.
- **Core Novelty:** Grounded XAI Attribution paired with human feedback-driven Direct Preference Optimization (DPO).

---

## Slide 2: Motivation & Background
- Exponential proliferation of toxic content across global and regional online platforms.
- Severe negative impacts: cyberbullying, offline communal violence, harassment of marginalized groups.
- Imperative need for scalable, transparent, and fair content moderation tools.

---

## Slide 3: Official Problem Statement
- Traditional models assume high-resource, monolingual English text.
- Severe failure modes:
  - Inability to comprehend Romanized Hindi (Hinglish) without distortion.
  - Inability to detect multimodal hate in memes where text appears benign but imagery is derogatory.
  - Black-box opacity with zero verifiable explanation.
  - Inability to learn adaptively from post-deployment moderator feedback.

---

## Slide 4: Mathematical Formulation
- An input post is represented as:
  $$x_i = \{T_i, V_i, C_i\}$$
  - $T_i$: Text content (English, Hindi Devanagari, or Hinglish code-mixed).
  - $V_i$: Visual image or meme.
  - $C_i$: Contextual thread history.
- **Objective:**
  $$f_\theta(x_i) \to y_i \in \{0, 1\}$$
  - $0$: Non-Hateful content.
  - $1$: Hateful content.

---

## Slide 5: Proposed System Architecture
1. **Multimodal Preprocessor:** Unicode NFKC, hashtag decomposition, EasyOCR meme text extraction, demographic context sanitization.
2. **Language Detector:** Script analysis + Romanized Hindi lexicon matching.
3. **Multimodal Cross-Attention Fusion:** Scaled dot-product cross-modal attention between text and visual representations.
4. **Classification Head:** Multilingual Transformer / Classical NLP baseline.
5. **Grounded Explainability Engine:** Token-level perturbation attribution + visual OCR mapping.
6. **Reward Optimization Loop:** Multi-objective reward calculation + DPO preference fine-tuning.

---

## Slide 6: Multilingual & Code-Mixed NLP
- Does **not** rely on blind English translation.
- Tokenization preserves subword fragments across both Devanagari script and Latin transliteration.
- Differentiates direct hate from sarcastic counterspeech and quotation.

---

## Slide 7: Multimodal Processing & Meme OCR
- Integrates `EasyOCR` to extract meme typography with spatial bounding coordinates.
- Cross-modal fusion models non-linear interactions where benign text + derogatory visual = hateful.
- Graceful single-modality fallback when images or context are absent.

---

## Slide 8: Grounded Explainable AI (XAI)
- Mathematical Leave-One-Out (LIME-style) perturbation attribution:
  $$\Delta_j = f_\theta(x) - f_\theta(x_{\backslash j})$$
- Distinguishes **positive evidence** (pushes hate) vs **negative evidence** (mitigates hate).
- Validated via **AOPC Faithfulness Drop Ratio** (measuring prediction shift when top tokens are ablated).
- Synthesizes 6-point natural language explanation grounded strictly in evidence without hallucination.

---

## Slide 9: Reward Optimization Framework (Core Innovation)
- Documented Multi-Objective Reward Function:
  $$R(x, y, e, f) = \alpha R_{\text{cls}} + \beta R_{\text{exp}} + \gamma R_{\text{ctx}} + \delta R_{\text{fb}} - \lambda_{\text{err}} R_{\text{err}}$$
- Weights: $\alpha=0.35, \beta=0.25, \gamma=0.15, \delta=0.25, \lambda_{\text{err}}=0.50$.
- Captures classification correctness, explanation quality, context awareness, and user consensus while severely penalizing overconfident false positives.

---

## Slide 10: Feedback Loop & DPO Optimization
- Human reviewers submit verified feedback (is_correct, suggested label, explanation rating).
- Compiles preference pairs $(x, y_w, y_l)$.
- Direct Preference Optimization (DPO) updates checkpoint from version $v1.0 \to v2.0$.
- Automated performance guard ensures rollback if held-out test metrics degrade.

---

## Slide 11: Experimental Methodology
- Implemented and executed 9 rigorous research experiments:
  1. Classical NLP (TF-IDF + Linear SVM) vs Transformer.
  2. Monolingual vs Multilingual & Code-Mixed.
  3. Text-Only vs Multimodal Meme Ablation.
  4. Fusion Mechanism Comparison (Early, Late, Cross-Attention).
  5. Explainability Faithfulness & Stability.
  6. Reward Optimization Impact Before vs After.
  7. Feedback Quantity & Quality Scaling.
  8. Bias & Fairness Audit across Subgroups.
  9. Contextual Disambiguation of Sarcasm & Quotation.

---

## Slide 12: Key Experimental Results
- **Transformer Superiority:** Robust subword generalization on code-mixed Hinglish.
- **Multimodal Gain:** +5% F1 gain on social memes by resolving text-image contradictions.
- **Faithfulness:** 88% confidence drop ratio when top attributed tokens are removed.
- **Reward Optimization:** Eliminated false positives on counterspeech through DPO feedback cycles.
- **Fairness:** Equalized Odds disparity gap $< 0.05$ across language groups.

---

## Slide 13: Bias, Fairness & Ethics
- Sanitizes thread history to prevent demographic identity profiling.
- Evaluates False Positive Rate (FPR) parity across English, Hindi, and Hinglish.
- Human-in-the-loop decision support: advises moderation rather than executing irreversible punitive bans.

---

## Slide 14: User Interface & Implementation
- Built with **ThoughtStream Design System** (minimal, zen, 0px border radius, flat hairline borders).
- Comprehensive **Dark Mode** support with persistent state.
- 11 dedicated research laboratories covering Detection, Datasets, Preprocessing, Models, XAI, Rewards, Bias, Experiments, Reports, and Settings.
- Zero hardcoding: all metrics and predictions are computed dynamically.

---

## Slide 15: Conclusion & Viva Summary
- Conforms in full to the project specification of Dr. Pradeepika Verma.
- Successfully proves that combining multimodal cross-attention with grounded XAI and reward-guided preference optimization creates a reliable, transparent, and fair content moderation platform.

