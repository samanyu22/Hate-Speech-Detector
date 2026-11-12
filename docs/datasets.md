# Dataset Documentation: Multimodal Deepfake & Content Verification Benchmark
## Course: CSET 346 – Natural Language Processing | Instructor: Dr. Sakshi

### 1. Dataset Overview & Provenance
The **CSET 346 Multimodal Benchmark Dataset** is an academically constructed, legally compliant research benchmark designed for cross-modal deepfake detection and explainable content verification. The dataset comprises **320 standardized multimodal samples**, balancing genuine and manipulated content across three modalities: **Text, Visual (Images/Video frames), and Acoustic (Audio waveforms)**.

- **Total Samples:** 320
- **Class Balance:** Exactly 1:1 (160 REAL / 160 FAKE)
- **License:** Creative Commons Attribution-NonCommercial 4.0 International (CC-BY-NC 4.0)
- **Research Use Conditions:** Exclusively for academic verification, defense research, and educational evaluation under CSET 346. No personally identifiable private individuals' media without prior consent.

---

### 2. Modality Specifications
Every sample in the benchmark contains structured data across three distinct channels:

1. **Textual Modality:**
   - Real news headlines, official summit briefings, press conferences, science disclosures, or manipulated/sensationalist claims.
   - Word count: 10 to 35 tokens per record.
   - Encoding: UTF-8 Unicode normalized text.

2. **Visual Modality (`.jpg`):**
   - Resolution: 256x256 pixels, RGB color space.
   - Authentic images: Continuous gradient transitions, organic photographic luminance distributions.
   - Manipulated images: Face swap seam discontinuities, boundary blurring, high-frequency spatial noise grids characteristic of diffusion/generative models.

3. **Acoustic Modality (`.wav`):**
   - Format: Single-channel (Mono) Linear PCM WAV.
   - Sample Rate: 16,000 Hz, 16-bit signed integer.
   - Duration: 2.5 seconds.
   - Authentic audio: Natural formant sweeps (120-220 Hz fundamental frequency) with natural breathing amplitude modulation.
   - Synthetic audio: Neural vocoder flatline pitch (180 Hz monotonic), high-frequency odd harmonics, phase discontinuities.

---

### 3. Manipulation Taxonomy
The 160 FAKE samples represent five diverse real-world manipulation styles:

| Manipulation Category | Count | Modality Target | Description |
| :--- | :--- | :--- | :--- |
| `authentic` (REAL) | 160 | None | Genuine multi-source synchronized factual reports. |
| `semantic_mismatch` | 32 | Cross-Modal | Authentic image/audio paired with contradictory, out-of-context text. |
| `face_manipulation` | 32 | Visual + Audio | Facial boundary alteration paired with synthetic speech endorsement. |
| `synthetic_speech` | 32 | Acoustic | Neural voice cloning mimicking executive or official instructions. |
| `ai_generated_image` | 32 | Visual | Diffusion-generated photorealistic synthetic imagery of fabricated events. |
| `misleading_caption` | 32 | Textual | Sensationalist or exaggerated claims misrepresenting underlying footage. |

---

### 4. Data Splitting & Leakage Prevention Strategy
To ensure strict academic validity, data splitting follows a **stratified, identity-grouped** partition:

- **Train Split:** 224 samples (70%) — 112 REAL, 112 FAKE.
- **Validation Split:** 48 samples (15%) — 24 REAL, 24 FAKE.
- **Test Split:** 48 samples (15%) — 24 REAL, 24 FAKE.

**Data Leakage Safeguards:**
- Grouped by source identity group (`org_00` to `org_14` for genuine, `synth_00` to `synth_14` for manipulated).
- No identical source identities or generated seed images span across both train and test splits.

---

### 5. Compatibility with External Research Datasets
The system includes built-in adapters (`src/data/dataset_adapter.py`) to import and normalize industry benchmarks:
- **FaceForensics++:** Facial reenactment and deepfake video manifests.
- **DFDC (Deepfake Detection Challenge):** Kaggle competition JSON manifests.
- **FakeAVCeleb:** Audio-video multimodal deepfake dataset.
- **Fakeddit:** Large-scale multimodal text-image misinformation dataset.
