# Research Methodology
## Project: Hate Speech Detection through Explainable AI and Reward Optimized LLMs
**Supervisor:** Dr. Pradeepika Verma

---

### 1. Problem Formulation
We formalize a social media post as a three-component tuple:
$$x_i = \{T_i, V_i, C_i\}$$
where:
- $T_i$: Text content (English, Hindi Devanagari, or Hinglish code-mixed).
- $V_i$: Visual information (images, memes).
- $C_i$: Contextual dialogue history (sanitized of demographic identifiers).

The detection objective is binary classification:
$$f_\theta(x_i) \to y_i \in \{0, 1\}$$
where $y_i = 0$ corresponds to Non-Hateful discourse, and $y_i = 1$ denotes Hateful content.

---

### 2. Multilingual and Code-Mixed NLP Pipeline
- **Unicode Normalization:** NFKC standard ensures consistent canonical equivalence across Devanagari combiners and Latin characters.
- **Hashtag Decomposition:** camelCase splitting extracts semantic tokens without discarding intent (e.g. `#StopHate` $\to$ `Stop Hate`).
- **Script Identification:** Computes character frequency ratios over Devanagari Unicode range `[\u0900-\u097F]` and Latin alphabet.
- **Hinglish Lexicon Matching:** Uses an extensive phonetic dictionary of Romanized Hindi terms (`kamine`, `saale`, `desh`, `bhai`, `aukaat`, `suaro`) to classify Hinglish without translation loss.

---

### 3. Multimodal Cross-Attention Fusion
- **Visual Encoder:** ResNet/CNN backbone extracts 128-dimensional spatial representations $v_i \in \mathbb{R}^{128}$.
- **OCR Grounding:** EasyOCR extracts embedded typography and spatial bounding boxes $[y_{\min}, x_{\min}, y_{\max}, x_{\max}]$.
- **Cross-Modal Attention:** Intermediate cross-attention allows visual features to query text tokens:
  $$\text{Attn}(Q, K, V) = \text{softmax}\left(\frac{Q K^T}{\sqrt{d_k}}\right) V$$
- **Gating Mechanism:** A learned sigmoid gate $g = \sigma(W_g [t_i; \text{Attn}])$ modulates interaction between text and vision.

---

### 4. Grounded Explainable AI Engine
- **Attribution Calculation:** Leave-One-Out (LIME-style) perturbation measures token-level prediction shifts:
  $$\Delta_j = f_\theta(x) - f_\theta(x_{\backslash j})$$
- **Faithfulness Verification:** Evaluated using Area Over the Perturbation Curve (AOPC), measuring the relative prediction drop when the top-$k$ attributed tokens are ablated:
  $$\text{Faithfulness}(x) = \frac{|f(x) - f(x_{\backslash \text{top\_k}})|}{\max(0.01, f(x))}$$

---

### 5. Multi-Objective Reward Optimization Framework
- **Reward Function:**
  $$R(x, y, e, f) = \alpha R_{\text{cls}} + \beta R_{\text{exp}} + \gamma R_{\text{ctx}} + \delta R_{\text{fb}} - \lambda_{\text{err}} R_{\text{err}}$$
  with $\alpha=0.35, \beta=0.25, \gamma=0.15, \delta=0.25, \lambda_{\text{err}}=0.50$.
- **Direct Preference Optimization (DPO):** Preference pairs $(x, y_w, y_l)$ optimize policy weights $\theta$ to align model predictions with human auditor corrections.

