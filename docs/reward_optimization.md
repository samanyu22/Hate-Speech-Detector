# Reward Optimization Framework
## Hate Speech Detection through Explainable AI and Reward Optimized LLMs

---

### 1. Theoretical Rationale
Supervised fine-tuning (SFT) on static benchmarks cannot adapt to post-deployment linguistic shifts, novel abusive slang, or subtle sarcastic counterspeech. The reward optimization framework provides an adaptive alignment mechanism that continuously optimizes model weights using structured human feedback.

---

### 2. Multi-Objective Reward Formula
$$R(x, y, e, f) = \alpha R_{\text{cls}} + \beta R_{\text{exp}} + \gamma R_{\text{ctx}} + \delta R_{\text{fb}} - \lambda_{\text{err}} R_{\text{err}}$$

#### Component Definitions:
1. **$R_{\text{cls}}$ (Classification Correctness):**
   - Evaluates agreement with validated human audit labels:
     $$R_{\text{cls}} = \begin{cases} 1.0 & \text{if } y = y^* \\ 0.0 & \text{if } y \neq y^* \end{cases}$$
2. **$R_{\text{exp}}$ (Explanation Faithfulness):**
   - Assesses faithfulness via the token perturbation drop ratio:
     $$R_{\text{exp}} = \min\left(1.0, \frac{|f(x) - f(x_{\backslash \text{top\_k}})|}{f(x)}\right)$$
3. **$R_{\text{ctx}}$ (Context Appropriateness):**
   - Rewards appropriate utilization of thread history without demographic profiling:
     $$R_{\text{ctx}} = \begin{cases} 0.85 & \text{if contextual disambiguation active} \\ 0.50 & \text{otherwise} \end{cases}$$
4. **$R_{\text{fb}}$ (Human Feedback Consensus):**
   - Normalizes 1-to-5 reviewer rating:
     $$R_{\text{fb}} = \frac{\text{Rating} - 1}{4} \in [0.0, 1.0]$$
5. **$R_{\text{err}}$ (Error Penalty):**
   - Penalizes confident misclassifications and biased false positives:
     $$R_{\text{err}} = \text{Confidence} \quad (\text{when prediction is erroneous})$$

---

### 3. Preference Optimization Loss (DPO)
Rather than training a separate complex reward network prone to reward hacking, we employ Direct Preference Optimization (DPO). The objective maximizes the margin between preferred responses $y_w$ and rejected responses $y_l$:
$$\mathcal{L}_{\text{DPO}}(\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x, y_w, y_l)} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

---

### 4. Rollback and Anti-Regression Safeguard
Every optimization cycle is versioned in the database (`multilingual-transformer-v1.0` $\to$ `v2.0`). Prior to promotion, the candidate model is re-evaluated on a held-out test split. If Macro F1 degrades by more than 5%, the candidate is quarantined, and the stable checkpoint is retained.

