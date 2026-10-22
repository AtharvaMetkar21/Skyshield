# SKYSHIELD — Execution Plan v4 (for Antigravity agent)

**Mission:** Classify airborne objects that enter radar range as **Bird / Drone / Other**, so a hostile drone is never mistaken for a bird at a sensitive site (army base, government facility).
**Core rule:** A missed drone (false negative) is far more costly than a false alarm. Every design, metric and threshold decision below favours **Drone recall**, without letting precision collapse.

> **Agent instructions (read first)**
> 1. Work phase by phase. Do **not** start a phase until the previous phase's *acceptance criteria* are met and logged.
> 2. Every random operation uses `SEED = 42`. Every run is logged (params, metrics, plots) to `reports/`.
> 3. The **test set is touched exactly once per modality**, at the very end of model selection. Never tune on it.
> 4. If any metric looks "too good" (> 99% on test), STOP and run the leakage audit in Phase 6 before continuing.
> 5. Never fabricate numbers. All figures in reports must come from saved run logs.
> 6. **All training is done in Jupyter notebooks (.ipynb), one step per cell, written in simple student-style code** (see Section 1B). Do not hide the training inside big scripts or classes.

---

## 0. Datasets and honest framing

| # | Dataset | Modality | Original classes | Role |
|---|---|---|---|---|
| D1 | Bird vs Drone (Kaggle, harshwalia) | Images | Bird, Drone | CNN training (Bird/Drone) |
| D2 | Micro-Doppler Aerial Classification (Kaggle, mithula05) | Radar time-series `.npy`, 2,800 samples x 100 steps x 3 features | Bird, Drone, Aircraft, Stealth UAV | Radar model |
| D3 | AOD-4 (Mendeley) | Images, 22,516 | Airplane, Helicopter, Drone, Bird | CNN "Other" class + extra Bird/Drone |

**Label mapping (3 classes):** Bird = 0, Drone = 1, Other = 2
- D2: Aircraft + Stealth UAV -> Other
- D3: Airplane + Helicopter -> Other
- D1: no Other class (comes only from D3)

**Two independent tracks (important):** the datasets are *not paired*. No radar sample corresponds to any image. So we build:
- **Radar track:** D2 only -> 3 models -> best radar model
- **Vision track:** D1 + D3 merged -> 3 CNNs -> best vision model
- **Fusion:** done at decision level (Phase 9), clearly documented as a *simulated pairing* in the demo. Do not claim real sensor fusion accuracy, because there is no paired ground truth to measure it.

"Train 3 models on each dataset, choose the best": implemented as 3 models per track, plus a **per-dataset breakdown** of the vision models (evaluate on D1-test and D3-test separately) to expose dataset bias.

---

## 1. Repository and environment

```
skyshield/
  data/{raw,interim,processed}/       # never commit raw data
  notebooks/                          # MAIN WORK HAPPENS HERE (see Section 1B)
    01_radar_audit_and_preprocessing.ipynb
    02_radar_model_training.ipynb
    03_image_audit_and_preprocessing.ipynb
    04_cnn_model_training.ipynb
    05_model_selection_and_final_test.ipynb
    06_fusion_and_export.ipynb
  models_store/                       # saved best models + scalers + split files
  reports/{figures,tables}/           # plots and tables saved from the notebooks
  app/
    backend/ (FastAPI)  frontend/ (HTML + JS)
  requirements.txt  README.md
```
- Python 3.10+, `scikit-learn`, `xgboost`, `tensorflow`/`keras` (or PyTorch + timm; pick one and stay consistent), `numpy`, `pandas`, `scipy`, `imagehash`, `opencv-python`, `matplotlib`, `seaborn`, `shap`, `joblib`, `fastapi`, `uvicorn`.
- Pin versions in `requirements.txt`. Use `sklearn.pipeline.Pipeline` so preprocessing is always fit on train only.

---

## 1B. Notebook rules and coding style (IMPORTANT)

The code must look like it was written by a student doing a normal ML project: simple, readable, step by step. It must **not** look like generated "framework" code.

### Coding style rules
1. **One step = one cell.** Never put loading, cleaning, splitting and training in the same cell.
2. **A short markdown cell before every code cell** explaining in plain English what we are about to do and why (2-3 lines, e.g. "Now we split the data into train, validation and test. We do this before scaling so the test data never leaks into training.").
3. **Simple code only:** plain variables (`X_train`, `y_train`, `X_val`, `model`, `y_pred`), basic loops, standard sklearn / keras calls. No custom classes, no decorators, no type hints, no clever one-liners, no heavy abstraction. Small helper functions are allowed only when the same code is used 3+ times (e.g., `show_confusion_matrix`).
4. **Student-style comments** on most lines, e.g. `# check how many samples are in each class`, `# fit the scaler only on training data`.
5. **Print or plot something after every step** (shape, head, value counts, a graph) so the output proves the step worked.
6. **Every notebook starts with the same 3 cells:** imports, `SEED = 42` + seed setting, and path / folder setup. Every notebook ends with a "Summary and what we learned" markdown cell.
7. **Run top to bottom** with "Restart and Run All" and it must finish with no errors. Outputs (plots, tables) stay visible in the saved notebook.
8. **Save results to disk** at the end of each notebook (split indices, scaler, model, metrics CSV, figures) so the next notebook loads them instead of recomputing.
9. Use clear names for figures (`reports/figures/radar_rf_confusion_matrix.png`) and always add title, axis labels and class names.

### Epochs and training length (what each model really uses)
| Model | Epochs? | Training length used |
|---|---|---|
| Random Forest | No epochs (not iterative) | `n_estimators` (number of trees) searched in 200-800 |
| SVM (RBF) | No epochs (solved in one go) | Tuned `C` and `gamma` only |
| XGBoost | No epochs, uses boosting rounds | Up to 1000 rounds, **early stopping after 30 rounds with no improvement**; usually stops around 150-400 |
| MobileNetV3Small / EfficientNetB0 / InceptionV3 | Yes | **Stage A (head only): max 15 epochs. Stage B (fine-tuning): max 25 epochs.** Early stopping (patience 5) so real training usually ends around 10-20 epochs in A and 8-15 in B |
| Optional 1D-CNN on raw radar (bonus) | Yes | max 60 epochs, patience 8 |

Rule: the number of epochs is **never fixed by hand to "look good"**. We set a maximum and let early stopping on validation loss decide, then plot the curves to prove it.

---

## 1C. Notebook-by-notebook, cell-by-cell steps

Each numbered item below is **one cell** (with its markdown explanation above it). Details of each step are in Phases 1-9 further below; this is the order they must appear in.

### Notebook 01 — Radar data audit and preprocessing (`D2`)
1. Title + goal of the notebook (markdown)
2. Imports; set `SEED = 42`; set folder paths
3. Load the `.npy` data and label file; print shapes and dtypes
4. Check missing values (NaN/Inf) and duplicates
5. Explain what the 3 features are (markdown, from dataset page)
6. Show summary statistics per feature, per class
7. Plot 5 sample signals for each class
8. Plot class-wise mean +/- std
9. Plot the class distribution (4 classes)
10. Map labels to 3 classes (Bird 0, Drone 1, Other 2); plot the new distribution
11. Split into train / validation / test (70/15/15, stratified); print sizes and class counts for each
12. Save the split indices to disk
13. Handle missing values and outliers (clip using **train** percentiles only)
14. Build features (time-domain) with a simple loop; show the first rows as a table
15. Add frequency-domain features (FFT, spectral entropy, etc.)
16. Combine into a feature table; check the shape and feature names
17. Remove near-constant and highly correlated features (decided on train only); show the correlation heatmap
18. Scale features with `StandardScaler` (fit on train, transform val and test)
19. Save processed arrays and the scaler
20. Summary markdown

### Notebook 02 — Radar model training (3 models)
1. Title + the plan for 3 models (markdown); imports; seed
2. Load processed train and validation data
3. **Baseline:** majority-class and logistic regression scores (so we know what "too low" means)
4. **Random Forest:** train a first simple model with default parameters; print train and val scores
5. Random Forest: hyperparameter search (`RandomizedSearchCV`, 5-fold stratified CV, scoring macro-F1); print best params
6. Random Forest: retrain with best params; classification report and confusion matrix on validation
7. Random Forest: learning curve plot and train vs validation gap check (overfit / underfit comment in markdown)
8. Random Forest: feature importance plot; save model
9. **SVM:** default model, then hyperparameter search for `C` and `gamma`; best params
10. SVM: retrain, classification report, confusion matrix on validation
11. SVM: learning curve + validation curve for `C` and `gamma`; gap check; save model
12. **XGBoost:** default model, then hyperparameter search
13. XGBoost: retrain with early stopping; plot the training vs validation log-loss per round (shows when it stops)
14. XGBoost: classification report, confusion matrix, learning curve, gap check; save model
15. Shuffled-label test for the final model of each type (accuracy must drop to chance level)
16. Comparison table of all 3 models (accuracy, macro-F1, Drone recall, train-val gap, CV std, training time)
17. Summary markdown (which one looks best on validation and why; **no test data used here**)

### Notebook 03 — Image data audit and preprocessing (`D1 + D3`)
1. Title, imports, seed, paths
2. Count images per class for D1 and D3 (bar charts)
3. Check for corrupt files and remove them
4. Show 10 random images per class (grid) and look for wrong labels
5. Check image sizes and aspect ratios (histograms)
6. Map labels to Bird / Drone / Other and build one manifest table (path, dataset, label)
7. Find duplicate and near-duplicate images with perceptual hashing; remove them
8. Split into train / val / test (stratified by class and source dataset); show counts
9. Collect the internet stress set (manual folder) and confirm it is separate
10. Compute class weights from the training split
11. Build `tf.data` loaders (resize, batch, cache) for 224x224 and 299x299
12. Define augmentation layers; show the same image augmented 8 times
13. Save the manifest with split column and class weights
14. Summary markdown

### Notebook 04 — CNN model training (3 models)
1. Title + plan (markdown); imports; seed; load manifest and loaders
2. **MobileNetV3Small:** build the model (frozen backbone + our head); `model.summary()`
3. Compile (Adam 1e-3, label smoothing 0.05) and set callbacks (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint)
4. Stage A training: `epochs = 15` max; keep the history
5. Plot loss and accuracy curves for stage A
6. Unfreeze the top layers (BatchNorm stays frozen); recompile with a low learning rate
7. Stage B fine-tuning: `epochs = 25` max
8. Plot loss and accuracy curves for stage B; markdown comment on over/underfitting
9. Validation results: classification report, confusion matrix, ROC curves, Drone recall
10. Show 12 validation images with predictions (6 correct, 6 wrong) + Grad-CAM
11. Save the model and the history
12. **EfficientNetB0:** repeat the same cells 2-11 in the same order
13. **InceptionV3 (299x299):** repeat the same cells 2-11 in the same order
14. Repeat the best config with 2 more seeds for each model (mean +/- std table)
15. Robustness check on validation images (blur, noise, low-res, brightness)
16. Comparison table of all 3 CNNs (accuracy, macro-F1, Drone recall, params, size, inference time per image, gap)
17. Summary markdown (**no test data used here**)

### Notebook 05 — Model selection and final test
1. Title and the selection rules from Phase 7 (markdown); imports; load saved models
2. Load the validation comparison tables for radar and vision
3. Apply the selection rules; pick one radar model and one CNN (write the reason in markdown)
4. Calibrate probabilities (temperature scaling) on **validation**; reliability diagram before and after
5. Choose the Drone-recall threshold and the "UNIDENTIFIED" cutoff on validation
6. Paired bootstrap / McNemar's test between the top 2 candidates
7. **Final test, radar:** run the chosen model on the test set once; report, confusion matrix, 95% confidence intervals
8. **Final test, vision:** run the chosen CNN on the test set once; same outputs
9. Per-dataset results (D1-test vs D3-test) and the internet stress-set results
10. Error analysis: show the worst mistakes and explain them
11. Final summary table and limitations (markdown)

### Notebook 06 — Fusion and export
1. Load the two chosen models, calibrators and thresholds
2. Implement the simple fusion rule (product of probabilities + safety OR-rule)
3. Test the fusion on simulated pairs from the test sets; show the table
4. Export final models, scalers, thresholds and class names to `models_store/` for the web app
5. Quick prediction test: one image + one radar sample end to end

> Phase 9 (the web simulation) is the only part that is **not** a notebook: it is the FastAPI + HTML/JS app in `app/`.

---

## 2. PHASE 1 — Data audit (both modalities)

### 2.1 Radar (D2)
1. Load `.npy`; print shape, dtype, NaN/Inf count, per-feature min/max/mean/std **per class**.
2. **Identify what the 3 features are** (e.g., velocity / amplitude / frequency, or I/Q/magnitude) from the dataset page and document it. Everything downstream depends on this.
3. Plot 5 random samples per class (all 3 features over time). Plot class-wise mean +/- std bands.
4. Check class balance (4-class and after 3-class mapping). Expect **Other ~50%** after merging; record it.
5. Check **duplicate / near-duplicate samples** (exact hash + correlation > 0.999). Remove or group them.
6. Check whether samples were generated by a simulator (very smooth, identical noise patterns). If so, note in the report that real-world generalisation is *not* proven.

### 2.2 Images (D1 + D3)
1. Count images per class per dataset; image size distribution; corrupt-file scan (try to open every file; quarantine failures).
2. **Cross-dataset de-duplication** with perceptual hash (`imagehash.phash`, Hamming distance <= 5). D1 and D3 may share images, and near-duplicate video frames inside one dataset are common. Duplicates across train/test are the #1 cause of fake 99% scores.
3. Inspect label noise: view 50 random images per class, check obvious mislabels (e.g., a bird in a "drone" folder).
4. Record resolution, aspect ratio, background type (sky vs. cluttered) per dataset to document **domain differences**.
5. Check whether D1's pre-split train/test is trustworthy (leakage between its own splits). **Ignore the provided split and re-split yourself** after de-duplication.

**Acceptance:** `reports/tables/data_audit.md` with all counts, duplicates removed, and the final per-class counts.

---

## 3. PHASE 2 — Radar preprocessing and feature engineering

### 3.1 Splitting (before any fitting)
- Stratified split **70 / 15 / 15** (train / val / test) on the 3-class label, `random_state=42`.
- If samples have a track/scenario ID, use `StratifiedGroupKFold` so one track never spans train and test.
- Save split indices to disk. Test indices are never reloaded until Phase 7.

### 3.2 Cleaning
- Impute NaN (median of train only) or drop sample if > 5% missing.
- Outliers: clip to train 0.5th/99.5th percentile per feature (do **not** drop; real radar has spikes). Log how many were clipped.

### 3.3 Two input representations
1. **Engineered features** (for XGBoost / RF / SVM). Per sample, per channel compute:
   - Time domain: mean, std, min, max, range, median, IQR, skewness, kurtosis, RMS, zero-crossing rate, mean absolute diff, peak count
   - Frequency domain (FFT/Welch): dominant frequency, spectral centroid, spectral bandwidth, spectral entropy, band energies (low/mid/high), spectral flatness
   - Cross-channel: correlation between channels
   Result ~ 3 x 25 = ~75 features. Store feature names.
2. **Raw sequence** (100 x 3) — optional bonus for a 1D-CNN/LSTM comparison model (not part of the required 3).

### 3.4 Scaling and selection (inside the Pipeline)
- `StandardScaler` (fit on train only) — required for SVM, harmless for trees.
- Remove near-zero-variance and highly correlated features (|r| > 0.95) — fit on train only.
- Optional: mutual-information or tree-importance feature selection, inside CV folds.

### 3.5 Imbalance handling
- `class_weight="balanced"` (RF, SVM); `sample_weight` computed from class frequency (XGBoost, multi-class).
- Do **not** oversample before splitting. If SMOTE is tried, put it inside the CV pipeline (`imblearn.Pipeline`) and only compare against class weights.

---

## 4. PHASE 3 — Radar model training (3 models)

| Model | Why | Hyperparameter search space |
|---|---|---|
| **Random Forest** | Robust baseline, interpretable importances | `n_estimators` 200-800, `max_depth` 6-30/None, `min_samples_leaf` 1-10, `max_features` sqrt/log2/0.3 |
| **XGBoost** | Strong on tabular features | `n_estimators` 200-1000, `learning_rate` 0.01-0.2, `max_depth` 3-8, `subsample` 0.6-1, `colsample_bytree` 0.5-1, `reg_lambda` 1-10, `min_child_weight` 1-10 |
| **SVM (RBF)** | Classic micro-Doppler classifier | `C` 0.1-1000 (log), `gamma` scale / 1e-4-1e-1 (log), `probability=True` |

**Protocol for each model (identical for all three):**
1. Pipeline = `[Imputer -> Scaler -> VarianceThreshold -> CorrFilter -> Classifier]`.
2. Hyperparameter search: `RandomizedSearchCV` (n_iter 40-60) with **5-fold StratifiedKFold on the train set only**, scoring = **macro-F1** (primary), refit on macro-F1.
3. XGBoost: use early stopping on a held-out fold of train (not test) — `early_stopping_rounds=30`.
4. After the best params are chosen, retrain on full train, evaluate on **val** (not test). Log: accuracy, macro-F1, per-class precision/recall/F1, Drone recall, confusion matrix, ROC-AUC (OvR), log-loss.
5. Save the fitted pipeline with `joblib` + the exact feature list.

**Optional rigour:** nested CV (outer 5, inner 3) on the final 2-3 candidates to get an unbiased performance estimate.

---

## 5. PHASE 4 — Image preprocessing

### 5.1 Merge and split
- Build one manifest CSV: `path, source_dataset, original_label, label3`.
- De-duplicated, then **stratified split 70 / 15 / 15** by `label3` **and** `source_dataset`, so each split has all datasets in proportion. Save the manifest with a `split` column.
- Additionally hold out an **external stress set**: 100-200 *manually collected internet images* (birds, drones, planes, helicopters in varied conditions: far away, cloudy, night, partial, small, blurred). Never used for training or tuning. This approximates the real demo scenario.

### 5.2 Class balance
- Expected: Other and Bird/Drone imbalance from D3. Use **class weights** in the loss. Cap any class at ~3x the smallest through undersampling of the *training split only* if the imbalance is severe.

### 5.3 Preprocessing per model (do not mix these up)
| Model | Input size | Normalisation |
|---|---|---|
| MobileNetV3Small | 224x224 | Use the model's built-in preprocessing (Keras variant includes rescaling inside the model; do not rescale twice) |
| EfficientNetB0 | 224x224 | Expects raw 0-255 input (Keras version has internal normalisation); use its own `preprocess_input` |
| InceptionV3 | 299x299 | `tf.keras.applications.inception_v3.preprocess_input` (scales to [-1, 1]) |

Always use `tf.data` pipelines with caching and prefetch. Keep aspect ratio: resize-with-pad rather than stretch (drones and planes distort badly).

### 5.4 Augmentation (training split only)
Horizontal flip, rotation +/- 15 deg, zoom 0.8-1.2, brightness/contrast jitter, random crop, Gaussian blur, slight Gaussian noise, JPEG compression. These simulate distance, weather and camera quality. **No** vertical flip, **no** heavy colour shifts that change object identity. Validation and test: resize/normalise only.

---

## 6. PHASE 5 — CNN training (3 models, transfer learning)

Models: **MobileNetV3Small, EfficientNetB0, InceptionV3** (ImageNet weights).

Head (identical for all): `GlobalAveragePooling2D -> Dropout(0.3) -> Dense(128, relu, L2=1e-4) -> Dropout(0.3) -> Dense(3, softmax)`.

**Two-stage training (same for all):**
1. **Stage A (feature extraction):** freeze backbone (keep BatchNorm layers in inference mode), train head only. Adam LR 1e-3, **max 15 epochs**, batch 32.
2. **Stage B (fine-tuning):** unfreeze the top ~20-30% of backbone layers (**keep BatchNorm frozen**), Adam LR 1e-5 to 3e-5, **max 25 epochs** (early stopping decides the real number).

**Loss / regularisation:** categorical cross-entropy with `label_smoothing=0.05` and class weights.
**Callbacks:** `EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)`, `ReduceLROnPlateau(factor=0.3, patience=2)`, `ModelCheckpoint(best val_macro_f1 or val_loss)`.
**Small search (per model, on val):** dropout {0.2, 0.3, 0.5}, fine-tune depth {20%, 40%}, LR {1e-5, 3e-5}. Keep it small; log every trial.
**Reproducibility:** `tf.keras.utils.set_random_seed(42)`, deterministic ops enabled; run the final config with **3 different seeds** and report mean +/- std.

**Per-model outputs:** training curves (loss/acc per epoch, both stages), val confusion matrix, per-class P/R/F1, Drone recall, ROC/PR curves, **Grad-CAM** on 12 val images (6 correct, 6 wrong) to check the model looks at the object, not the sky/background.

---

## 7. PHASE 6 — Over/underfitting and leakage controls (mandatory for ALL 6 models)

| Check | How | Pass condition |
|---|---|---|
| **Underfitting** | Train metrics vs. a trivial baseline (majority class, logistic regression) | Train macro-F1 clearly above the baseline and above ~0.90 for CNNs; if not, increase capacity / unfreeze more / train longer |
| **Overfitting gap** | Train vs. CV/val macro-F1 | Gap <= ~3-5 points. Larger gap -> more regularisation, more augmentation, shallower trees, smaller C |
| **Learning curves** | `learning_curve` (radar) with 5 train fractions; epoch curves (CNN) | Val curve plateaus close to the train curve; val loss does not rise while train loss falls |
| **Validation curves** | e.g., `max_depth`, `C`, `gamma` | Chosen value is in a stable region, not at a spike |
| **CV stability** | Std of fold scores | Std <= ~1-2 points; large variance signals instability or tiny classes |
| **Seed stability** | 3 seeds | Std of macro-F1 small |
| **Leakage audit** | (a) Hash-check train vs. test overlap; (b) shuffle-label test: train on shuffled labels and confirm accuracy drops to ~chance; (c) check that no scaler/selector/augmentation saw val/test data | All pass. A shuffled-label score far above chance = leakage |
| **Feature sanity (radar)** | Permutation importance / SHAP | Importances are physically plausible (e.g., Doppler spread, spectral features); no single ID-like feature dominates |
| **Calibration** | Reliability diagram, ECE; temperature scaling fit **on val** | ECE < ~0.05 after calibration |
| **Robustness** | Evaluate on perturbed val images (blur, noise, low-res, brightness) and radar with added Gaussian noise (SNR sweeps) | Graceful degradation, report the numbers |
| **External validation** | Internet stress set (images); cross-dataset test (train D1+D3-part, test on the other) | Report honestly, even if lower |

---

## 8. PHASE 7 — Model selection protocol (identical for radar and vision tracks)

1. **Candidates:** 3 models per track, each already tuned in Phases 3 / 5 using train + val only.
2. **Primary criteria (in order):**
   1. Drone recall on **val** >= 0.97 (hard requirement; set threshold if needed, see below)
   2. Macro-F1 (val, mean across CV/seeds)
   3. Small train-val gap and low variance (Phase 6 checks pass)
   4. Calibration (ECE) and robustness drop
   5. Inference latency and model size (tie-breaker; critical for edge/real-time)
3. **Decision threshold tuning:** on val, choose the threshold on P(Drone) that achieves target Drone recall (e.g., >= 0.97) at maximum precision. Fix it, then freeze.
4. **Statistical comparison:** paired bootstrap (1,000 resamples) of macro-F1 difference or McNemar's test on val predictions. If two models are statistically tied, pick the smaller/faster/better-calibrated one.
5. **Final evaluation (once):** run the single chosen model per track on the **test set**. Report accuracy, macro-F1, per-class P/R/F1, confusion matrix, ROC-AUC, 95% bootstrap CIs, plus the external stress-set score. **Do not re-tune after seeing test results.** If test is much worse than val, investigate leakage/shift and document it, don't quietly retune.
6. **Selection report:** `reports/model_selection.md` with the comparison table (all 6 models), the chosen two, and the reasons.

**Open-set / unknown handling:** if max calibrated probability < `tau` (choose on val, e.g., 0.60), output **"UNIDENTIFIED"** — treated as a potential threat (amber alert), never silently as Bird.

---

## 9. PHASE 8 — Fusion and alert logic

Because no paired radar+image data exists, fusion is **decision-level**:
- `P_fused = normalize( P_radar^w_r * P_vision^w_v )` (product of experts) with weights `w_r, w_v` from each model's validation macro-F1 (normalised), using the **calibrated** probabilities.
- **Safety override:** if *either* sensor gives P(Drone) above its own tuned threshold -> final = Drone alert (OR-rule, recall-first). Document the false-alarm cost of this rule.
- Outputs: `{label, confidence, per_sensor_probs, status: DRONE | OTHER | BIRD | UNIDENTIFIED}`.

| Status | UI colour | Sound |
|---|---|---|
| DRONE | Red, blinking | Continuous siren (alert) |
| UNIDENTIFIED / low confidence | Amber | Short double-beep |
| OTHER (aircraft/helicopter) | Blue | Soft single tone (optional) |
| BIRD | Green | None |

---

## 10. PHASE 9 — Live simulation web app

### 10.1 Backend (FastAPI)
- `POST /predict/image` — accepts an uploaded file **or** an image URL.
  - URL mode: fetch server-side (avoids browser CORS). **Security:** allow only http/https, block private/loopback IPs (SSRF protection), 10 MB size cap, 5 s timeout, verify `Content-Type` is an image, re-encode with Pillow before inference.
  - Returns calibrated probabilities, label, status, Grad-CAM (optional), latency.
- `GET /radar/sample?scenario=bird|drone|other` — returns a held-out **test-set** radar sample and its model prediction, for the simulated-pairing demo.
- `POST /predict/fused` — fuses both outputs.
- Load models once at startup. Health endpoint `/health`.

### 10.2 Frontend
- **Radar screen (HTML canvas):** dark green PPI display, range rings, rotating sweep line, compass bearings, blip trail.
- **Flow:** user uploads/pastes an image -> image appears as a blip at the edge, moves toward the centre with a trail -> when it crosses the **detection ring** it freezes, a "TARGET LOCKED" box appears -> the image is sent to the backend -> result card shows label, confidence bar chart (3 classes), status colour, and (optional) Grad-CAM overlay.
- **Alert audio:** generate the siren with the **Web Audio API** (oscillator sweeping 600-1200 Hz), so no audio file is needed. Browsers block audio until a user gesture, so add an **"ARM SYSTEM"** button that unlocks the `AudioContext`; show a mute toggle and "Acknowledge" button to stop the siren.
- **Log panel:** timestamped detections, exportable as CSV.
- **Mode switch:** (A) *Vision-only* (default) and (B) *Fusion demo*: operator selects a scenario, the backend pairs the uploaded image with a held-out radar sample, and the UI shows both sensors plus the fused verdict, **labelled "simulated pairing"**.
- Responsive layout, loading states, error toasts (invalid image, network error).

---

## 11. PHASE 10 — Testing, QA and deliverables

- **Unit tests:** preprocessing shapes and ranges, label mapping, split-leakage test (asserts zero hash overlap), fusion math.
- **API tests:** valid image, corrupt file, huge file, non-image URL, private-IP URL (must be rejected).
- **Demo set:** 20 unseen internet images (5 per type incl. tricky ones: bird at distance, drone silhouette at dusk, kite, helicopter). Report the results table, **including failures**.
- **Deliverables:** (1) trained models + preprocessing artifacts, (2) `reports/` (audit, 6-model comparison, selection, diagnostics plots, final test metrics), (3) web app + `README` with run instructions, (4) a "Limitations" section.

---

## 12. Known limitations to state openly in the report

1. Radar and image data are unpaired; fusion is decision-level and demonstrated via simulated pairing, not validated on real co-registered sensors.
2. D2 may be simulated/curated; strong scores do not guarantee field performance against real clutter, weather, or adversarial drones.
3. Internet photos are close-range optical images, whereas a real radar sees small, distant returns; the image model is a visual confirmation (EO/IR camera) stage, not a substitute for radar classification.
4. "Other" merges very different objects (aircraft, helicopters, stealth UAV); report per-source recall.
5. Literature numbers (e.g., "SVM 99.59%" or "90-95% RF accuracy") should be cited with the exact paper and verified before they go into any report.

---

## 13. Definition of done (checklist)

- [ ] Data audit done, duplicates removed, splits frozen and saved
- [ ] 3 radar models + 3 CNN models trained with identical protocol
- [ ] All Phase 6 checks pass (or failures documented with fixes)
- [ ] One best model per track chosen via the Phase 7 rules, test evaluated once
- [ ] Calibrated probabilities, Drone-recall threshold and unknown-rejection `tau` frozen
- [ ] Web simulation works end-to-end: upload -> approach radar -> classify -> siren on Drone
- [ ] Tests pass; limitations documented
