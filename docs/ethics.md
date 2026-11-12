# Ethical Considerations, Privacy, and Responsible Use
## CSET 346 – Natural Language Processing | Multimodal NLP Project

### 1. Defensive Verification Focus
The software system developed in this project is explicitly engineered as a **forensic defense and verification tool**. It is strictly designed to detect, analyze, and explain synthetic or manipulated multimodal media. Under no circumstances does the codebase implement capabilities for:
- Generating synthetic human faces or deepfake imagery.
- Synthesizing or cloning unauthorized human voices.
- Facilitating malicious non-consensual media creation.

---

### 2. Privacy and Consent
1. **No Non-Consensual Private Data Collection:**
   The training and evaluation benchmark utilizes legally reusable, synthetic, or public educational datasets. Media depicting private individuals without documented explicit consent is strictly prohibited from ingestion into persistent datasets.
2. **Ephemeral Processing:**
   User-uploaded files submitted via the interactive verification interface are held in ephemeral memory buffers during inference and are not permanently redistributed or exposed to external third-party servers.

---

### 3. Non-Definitive Probabilistic Disclaimer
Automated neural classifiers are probabilistic models trained on statistical patterns. They must never be treated as sole, indisputable legal evidence of authenticity or manipulation. 
- The system includes transparent confidence calibration and an explicit `UNCERTAIN` abstention mechanism.
- Decisions are accompanied by grounded evidence explanations (Grad-CAM heatmaps and token attributions) so human verifiers can scrutinize the underlying reasoning rather than relying on a black-box label.

---

### 4. Mitigating Demographic and Representation Bias
Deepfake detection models frequently exhibit demographic performance discrepancies when trained on skewed facial databases. To mitigate this risk:
- The benchmark maintains a 1:1 balance between authentic and manipulated records.
- Grouped identity splitting prevents source over-fitting.
- Multi-channel cross-modal consistency analysis relies on semantic alignment rather than demographic appearance alone.
