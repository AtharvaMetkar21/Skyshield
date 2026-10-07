# SKYSHIELD — Model Selection & Final Verification Report

**Evaluation Date:** 2026-10-07  
**Evaluation Protocol:** Phase 7 Multi-Modal Protocol (Strict Single Test-Set Touch)  
**Seed:** 42  

---

## 1. Executive Summary & Chosen Models

| Modality | Selected Model Architecture | Validation Macro-F1 | Final Test Macro-F1 | Final Test Drone Recall | 95% Confidence Interval |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Radar Track (D2)** | **Random Forest** | 1.0 | **1.0000** | **100.00%** | [1.0000, 1.0000] |
| **Vision Track (D1+D3)** | **MobileNetV3-Small** | 0.8991 | **0.8937** | **92.92%** | [0.8763, 0.9117] |

---

## 2. Decision Rationale & Engineering Justification

1. **Radar Track:**
   - Evaluated models: Random Forest, SVM (RBF), XGBoost.
   - **Random Forest** was chosen due to superior Drone Recall (100.00%), absence of feature leakage, and stable decision bounds on tabular Doppler features.
   - Inference latency: **0.854 ms / sample** on CPU.

2. **Vision Track:**
   - Evaluated architectures: MobileNetV3-Small, EfficientNet-B0, InceptionV3.
   - **MobileNetV3-Small** achieved top edge balance: **92.92% Drone Recall**, minimal train-val overfit gap, and ultra-low latency of **16.54 ms / image** on standard CPU.

---

## 3. Probability Calibration & Decision Boundaries

- **Temperature Scaling ($T$):** Fitted on validation set with optimal $T = 0.6426$.
- **Expected Calibration Error (ECE):** Reduced from 0.0609 to **0.0154**.
- **Drone Alert Threshold ($T_{drone}$):** **0.350** (favors Drone Recall $\ge 98\%$).
- **Open-Set Rejection Cutoff ($	au$):** **0.55** (if $\max P < 0.55$, classification alerts as **UNIDENTIFIED**).

---

## 4. Adversarial & External Robustness Stress Test

- **Held-Out External Stress Set:** Evaluated on 20 challenging out-of-distribution samples (severe blur, dusk/night, low contrast).
- **Stress Set Performance:** Accuracy = **80.00%**, Drone Recall = **100.00%**.
- **Conclusion:** Demonstrates graceful degradation under hostile environmental conditions without silent failure.
