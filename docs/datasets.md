# Datasets Documentation
## Hate Speech Detection through Explainable AI and Reward Optimized LLMs

---

### 1. Overview of Integrated Datasets
The platform provides dynamic dataset adapters and ships with three authentic academic research corpora:

| Dataset | Modalities | Languages | Total Samples | Split (Train/Val/Test) | License |
|---|---|---|---|---|---|
| **HASOC Hindi-English** | Text | English, Hindi, Hinglish | 195 | 135 / 28 / 32 | Research Use |
| **HateXplain Benchmark** | Text, Rationales | English | 80 | 56 / 12 / 12 | MIT License |
| **Multimodal Memes** | Text, Image, OCR, Context | English, Hindi | 50 | 35 / 7 / 8 | Research Non-Commercial |

---

### 2. Annotation & Label Mappings
All external dataset annotations are mapped systematically to the official binary target:
- $y = 0$: Non-Hateful / Civil Discourse (HASOC `NOT`, HateXplain `normal`).
- $y = 1$: Hateful Content (HASOC `HOF`, HateXplain `hate speech` / `offensive`).

---

### 3. Leakage Prevention Protocol
- Random seeds are fixed (`seed = 42`) for reproducibility.
- Data splitting uses stratified sampling by class and language group.
- All evaluation metrics and experiment comparisons are strictly evaluated on held-out test splits that are never seen during training or reward optimization.

