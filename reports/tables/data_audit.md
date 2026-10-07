# SKYSHIELD — Phase 1 & 4 Data Audit Report

**Date:** 2026-10-06  
**Pipeline Seed:** 42  
**Mission:** Classify airborne objects entering defense perimeters as Bird / Drone / Other, prioritizing Drone Recall.

---

## 1. Modality 1: Radar Micro-Doppler Time-Series (D2 - Astra / mithula05)

- **Source File:** `data/raw/d2_radar/astra_dataset.csv`
- **Native Shape:** 2,800 samples $\times$ 100 time-steps $\times$ 3 channels (Velocity / Amplitude / Frequency)
- **Data Quality:** NaN/Inf count = 0, Exact duplicate signals = 0.
- **Outlier Handling:** Winsorized clipping to 0.5th and 99.5th percentiles computed solely on the training split.

### Class Distribution (4-class to 3-class mapping)
| Original Class | Target 3-Class Mapping | Sample Count | Percentage |
| :--- | :--- | :--- | :--- |
| Bird | **Bird (0)** | 700 | 25.0% |
| Drone | **Drone (1)** | 700 | 25.0% |
| Aircraft | **Other (2)** | 700 | 25.0% |
| Stealth UAV | **Other (2)** | 700 | 25.0% |
| **Total** | | **2,800** | **100.0%** |

### Stratified Split (70 / 15 / 15)
- **Train (70%):** 1,960 samples (Bird: 490, Drone: 490, Other: 980)
- **Validation (15%):** 420 samples (Bird: 105, Drone: 105, Other: 210)
- **Test (15%):** 420 samples (Bird: 105, Drone: 105, Other: 210) — *Strictly held out until Phase 7*

---

## 2. Modality 2: Aerial Electro-Optical Images (D1 + D3)

- **Source 1 (D1):** Bird vs Drone (`harshwalia`) — 826 full-frame images (398 Bird, 428 Drone)
- **Source 2 (D3):** AOD-4 Aerial Object Detection (`Mendeley`) — 22,516 total images with YOLOv8 bounding boxes
- **Extraction Protocol:** Fast selective sampling with bounding box cropping (+15% contextual padding; boxes < 32x32 px filtered) to curate a balanced target pool of ~7,500 objects.
- **De-duplication:** Perceptual hashing (`imagehash.phash`, hash size 8, Hamming distance $\le 4$) across all merged candidates. 128 near-duplicates and repeated video frames removed.

### Combined & Cleaned Class Distribution
| Object Class | Source Contribution | Retained Clean Count | Balance Ratio |
| :--- | :--- | :--- | :--- |
| **Bird (0)** | D1 (398) + AOD-4 (2,061) | 2,459 | 33.36% |
| **Drone (1)** | D1 (428) + AOD-4 (2,025) | 2,453 | 33.27% |
| **Other (2)** | AOD-4 Airplane (1,235) + Helicopter (1,225) | 2,460 | 33.37% |
| **Total** | | **7,372** | **100.0%** |

### Stratified Split (70 / 15 / 15, stratified by class and source dataset)
- **Train (70%):** 5,160 images (Bird: 1,721, Drone: 1,717, Other: 1,722)
- **Validation (15%):** 1,106 images (Bird: 369, Drone: 368, Other: 369)
- **Test (15%):** 1,106 images (Bird: 369, Drone: 368, Other: 369) — *Strictly held out until Phase 7*

### Held-Out External Stress Set
- **Directory:** `data/raw/internet_stress_set/`
- **Count:** 20 heavily perturbed challenge images (severe Gaussian blur, underexposure, distance reduction, contrast degradation) held out exclusively for Phase 7 external robustness verification.
