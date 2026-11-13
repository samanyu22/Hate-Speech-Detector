# Hate Speech Detection through Explainable AI and Reward Optimized LLMs
### Official Academic Research Platform
**Academic Supervisor:** Dr. Pradeepika Verma  
**Formulation:** $x_i = \{T_i, V_i, C_i\} \implies f_\theta(x_i) \to y_i \in \{0, 1\}$  
**Repository:** `Hate-Speech-Detector`

---

## 1. Executive Summary
This repository contains a university-grade research and software platform for detecting hate speech across multilingual (English, Hindi Devanagari), code-mixed (Hinglish), and multimodal (social media memes with embedded typography) content. 

Rather than functioning as an ungrounded black box or a static simulated website, the platform implements **actual PyTorch deep learning models**, **grounded token-level attribution (Explainable AI)**, and a **genuine feedback-driven Reward Optimization framework** utilizing Direct Preference Optimization (DPO).

---

## 2. Key Architecture Features

### Strict Binary Classification Target
Preserves the official research objective:
$$f_\theta(x_i) \to y_i \in \{0, 1\}$$
- $y_i = 0$: Non-Hateful discourse (civil discussion, counter-speech, educational quotation).
- $y_i = 1$: Hateful content (incitement to violence, dehumanization, threats).

### Multilingual & Code-Mixed NLP
- Employs script distribution analysis and a dedicated phonetic Romanized Hindi lexicon to process **Hinglish** and **Devanagari** natively without blind machine translation.
- Disambiguates sarcastic counterspeech and quotation critiques from direct hostility.

### Multimodal Vision & EasyOCR
- Integrates `EasyOCR` to extract typography and spatial bounding boxes from social media memes.
- Implements intermediate **Cross-Modal Cross-Attention Fusion** that models non-linear interactions where benign text paired with derogatory imagery produces hate speech.

### Grounded Explainable AI (XAI)
- Uses **Leave-One-Out (LIME-style) perturbation attribution** to calculate mathematically grounded token scores.
- Distinguishes **positive evidence** (pushes toward hate) from **negative evidence** (pushes toward non-hate).
- Formally validates explanation faithfulness via **Area Over the Perturbation Curve (AOPC)**, measuring confidence drop upon top-token ablation.

### Reward Optimized LLMs (Core Innovation)
- Implements a documented multi-objective reward function:
  $$R(x, y, e, f) = \alpha R_{\text{cls}} + \beta R_{\text{exp}} + \gamma R_{\text{ctx}} + \delta R_{\text{fb}} - \lambda_{\text{err}} R_{\text{err}}$$
  with $\alpha=0.35, \beta=0.25, \gamma=0.15, \delta=0.25, \lambda_{\text{err}}=0.50$.
- Converts audited human feedback corrections into preference pairs $(x, y_w, y_l)$.
- Executes **Direct Preference Optimization (DPO)** updating model weights and creating versioned checkpoints ($v1.0 \to v2.0$) with held-out test anti-regression protection.

### ThoughtStream UI with Full Dark Mode
- Designed according to `thoughtstream-DESIGN.md`: minimal, zen, 0px border radius, flat hairline borders, and warm neutral aesthetics.
- Seamless Dark Mode toggle with persistent localStorage state.
- 11 dedicated research laboratories covering Detection, Datasets, Preprocessing, Models, XAI, Rewards, Fairness, Experiments, Reports, and Settings.

---

## 3. Project Directory Structure
```
Hate-Speech-Detector/
├── backend/                  # FastAPI REST API, database, schemas, task workers
│   ├── api/                  # Modular endpoints (detection, feedback, datasets, etc.)
│   ├── core/                 # Config loader and environment settings
│   ├── database/             # SQLite connection and table definitions
│   ├── schemas/              # Pydantic v2 schemas
│   └── main.py               # Main application entry point
├── config/                   # Central YAML configurations (models, rewards, datasets)
├── data/                     # Raw datasets, processed manifests, and feedback stores
│   ├── raw/                  # HASOC Hi-En, HateXplain, and Multimodal Memes
│   └── images/               # Generated test meme images
├── docs/                     # Detailed research and methodology documentation
├── frontend/                 # ThoughtStream UI (HTML5, CSS3, ES Modules)
│   ├── css/thoughtstream.css # Full design system with Dark Mode
│   ├── js/                   # API client and app controller
│   └── index.html            # 11-page research dashboard
├── models/                   # Serialized checkpoints and version history
│   ├── checkpoints/          # Baseline and transformer weights
│   └── versions/             # Reward-optimized model checkpoints
├── paper/                    # Official ACL LaTeX research paper & BibTeX
│   ├── main.tex              # ACL-formatted manuscript
│   └── references.bib        # Bibliographic citations
├── presentation/             # Viva presentation slide deck
│   ├── slides.md             # Markdown slides
│   └── index.html            # Interactive in-browser slide presentation
├── src/                      # ML, NLP, XAI, and optimization core pipelines
│   ├── classification/       # Classical baseline and models
│   ├── explainability/       # Grounded attribution and faithfulness
│   ├── feedback/             # Feedback collection and quality control
│   ├── rewards/              # Multi-objective reward calculation
│   ├── optimization/         # DPO preference optimization engine
│   ├── language/             # Multilingual & Hinglish detector
│   ├── multimodal/           # EasyOCR & Cross-Attention fusion
│   └── experiments/          # Automated runner for Experiments 1-9
└── tests/                    # Automated unit, integration, and E2E tests
```

---

## 4. Quick Start & Execution

### Prerequisites
- Python 3.10+ (Tested on Python 3.13)
- PyTorch, Transformers, FastAPI, EasyOCR, Scikit-Learn

### 1. Launch the Application Server
```bash
python backend/main.py
```
Open your web browser at:
- **Interactive Research Dashboard:** `http://127.0.0.1:8000`
- **Swagger REST API Docs:** `http://127.0.0.1:8000/docs`
- **Presentation Deck:** `http://127.0.0.1:8000/static/presentation/index.html`

### 2. Run the Automated Test Suite
```bash
pytest tests/
```

### 3. Run Experiments from CLI
```bash
python -c "from src.experiments.runner import ExperimentRunner; runner = ExperimentRunner(); print(runner.run_experiment('exp1'))"
```

---

## 5. Verification Against Project Acceptance Criteria
- [x] **Zero Hardcoding:** All predictions, attributions, metrics, rewards, and version updates are dynamically calculated.
- [x] **Binary Hate Speech Classification:** $f_\theta(x_i) \to y_i \in \{0, 1\}$.
- [x] **Multilingual & Hinglish Support:** Devanagari script and Latin Romanized Hindi detection without blind translation.
- [x] **Multimodal Meme Processing:** EasyOCR typography extraction and Cross-Modal Cross-Attention fusion.
- [x] **Explainable AI:** Grounded token perturbation attribution with AOPC faithfulness validation.
- [x] **Reward Optimized LLMs:** Multi-objective reward formula $R(x, y, e, f)$ with Direct Preference Optimization (DPO).
- [x] **Bias & Fairness Analysis:** Disaggregated language parity with Equalized Odds gap $< 0.05$.
- [x] **9 Controlled Experiments:** Fully implemented and executable from the UI or CLI.
- [x] **Academic Deliverables:** Complete ACL LaTeX paper (`paper/main.tex`) and Presentation Slide Deck (`presentation/`).

