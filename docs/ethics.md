# Ethics, Fairness & Responsible Content Moderation
## Hate Speech Detection through Explainable AI and Reward Optimized LLMs

---

### 1. Human-in-the-Loop Moderation Principle
The platform is designed strictly as an automated **decision-support tool** for human content moderators, not an autonomous punitive enforcement engine. The model does not execute irreversible account suspensions, deletions, or legal reports without human review.

---

### 2. Privacy & Demographic Neutrality
To prevent demographic profiling:
- No user identity attributes (religion, caste, race, gender, sexual orientation) are extracted or used as predictive features.
- Conversational context is processed through a demographic sanitizer that masks identity terms into neutral tokens (`[DEMOGRAPHIC_NEUTRAL]`), preventing models from falsely associating identity group names with hostility.

---

### 3. Counterspeech vs Direct Hate
Keyword-matching moderation systems disproportionately penalize marginalized communities when they discuss, quote, or counter discrimination. Our framework explicitly evaluates quotation framing (`quoting`, `someone said`) and sarcastic counterspeech (`oh sure, brilliant logic`), preserving civil and anti-bigotry discourse.

---

### 4. Disaggregated Fairness Audit
The platform continuously evaluates Equalized Odds disparity across language groups:
$$\text{Equalized Odds Gap} = \max(|\text{FPR}_A - \text{FPR}_B|, |\text{FNR}_A - \text{FNR}_B|)$$
Enforcing an Equalized Odds gap $< 0.15$ guarantees that non-English and code-mixed speakers are not subjected to higher false positive censorship rates.

