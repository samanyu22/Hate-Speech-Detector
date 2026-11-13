# Requirements Traceability Matrix (RTM)
## Project: Hate Speech Detection through Explainable AI and Reward Optimized LLMs
**Instructor / Supervisor:** Dr. Pradeepika Verma

| Ref ID | Official Requirement from Project Specification | Implementation Component | Source File(s) | Verification Method | Status |
|---|---|---|---|---|---|
| **REQ-01** | Binary Classification Target ($f_\theta(x_i) \to y_i \in \{0,1\}$ where 0=Non-hateful, 1=Hateful) | Classification Engine | `src/classification/`, `backend/schemas/` | Automated unit & integration tests on binary loss & thresholding | VERIFIED |
| **REQ-02** | Multilingual & Code-Mixed Text Processing ($T_i$: English, Hindi, Hinglish, low-resource) | Text Preprocessor & Language Detector | `src/preprocessing/text/`, `src/language/` | Script detection tests, transliteration handling, character n-gram tokenization | VERIFIED |
| **REQ-03** | Visual & Meme Input Processing ($V_i$: images, memes, OCR text) | Multimodal Engine & OCR Service | `src/preprocessing/image/`, `src/multimodal/ocr.py` | EasyOCR text extraction and image tensor validation | VERIFIED |
| **REQ-04** | Contextual Information Modeling ($C_i$: conversational context, thread history, without demographic profiling) | Context Preprocessor & Modality Fusion | `src/preprocessing/context/`, `src/multimodal/fusion.py` | Multi-input forward pass with and without context gating | VERIFIED |
| **REQ-05** | Multimodal Fusion (Joint consideration of text, OCR, visual tokens, and context) | Cross-Attention & Feature-Level Fusion | `src/multimodal/fusion.py`, `src/models/multimodal_detector.py` | Ablation experiments comparing text-only vs multimodal inputs | VERIFIED |
| **REQ-06** | Explainable AI (XAI) Grounding (Token-level attribution, visual grounding, natural language explanation) | XAI Engine | `src/explainability/` | Integrated gradients / perturbation attribution, faithfulness perturbation test ($\Delta f(x)$) | VERIFIED |
| **REQ-07** | Human Feedback Collection & Validation | Feedback Manager | `src/feedback/`, `backend/api/feedback.py` | Persistent SQLite feedback storage and validation schemas | VERIFIED |
| **REQ-08** | Documented Reward Function $R(x, y, e, f) = \alpha R_{cls} + \beta R_{exp} + \gamma R_{ctx} + \delta R_{fb} - \lambda R_{err}$ | Reward Engine | `src/rewards/reward_function.py` | Unit tests computing multi-objective reward vectors across test instances | VERIFIED |
| **REQ-09** | Feedback-Guided Reward Optimization (Model fine-tuning / preference optimization, versioning, rollback) | Optimization Engine | `src/optimization/`, `backend/api/optimization.py` | Versioned checkpoints ($v1 \to v2$), held-out test evaluation before/after | VERIFIED |
| **REQ-10** | Bias, Fairness, and Responsible Content Moderation (Disaggregated group metrics, FPR parity, counterspeech) | Fairness & Ethics Engine | `src/fairness/`, `src/evaluation/` | Disaggregated demographic & language parity evaluation | VERIFIED |
| **REQ-11** | 9 Controlled Research Experiments (Classical vs Transformer, Monolingual vs Multilingual, Multimodal vs Text, Fusion, XAI, Rewards, Scaling, Fairness, Sarcasm) | Experiment Runner | `src/experiments/`, `backend/api/experiments.py` | Automated reproducible execution with metric persistence | VERIFIED |
| **REQ-12** | ThoughtStream Design System with Dark Mode (11 dedicated research pages, 0px border-radius, zen aesthetic) | Frontend Architecture | `frontend/`, `thoughtstream-DESIGN.md` | Interactive UI rendering across all 11 pages with persistent dark mode | VERIFIED |
| **REQ-13** | Academic Deliverables: Source Code, ACL Research Paper, and Presentation Slides | Academic Paper & Slides | `paper/main.tex`, `paper/references.bib`, `presentation/` | Valid LaTeX document and interactive presentation slide deck | VERIFIED |

