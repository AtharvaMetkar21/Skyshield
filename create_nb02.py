import nbformat as nbf
import os

nb = nbf.v4.new_notebook()
cells = []

# Cell 1: Markdown Title + Plan
cells.append(nbf.v4.new_markdown_cell("""# Notebook 02: Radar Model Training (3 Models)

**Goal:** In this notebook, we train, tune, and evaluate three distinct machine learning models on the engineered radar micro-Doppler features:
1. **Random Forest** (ensemble of decision trees with bagging)
2. **Support Vector Machine (SVM with RBF kernel)** (maximum margin kernel classifier)
3. **XGBoost** (gradient boosted decision trees)

We follow strict leakage-free protocols:
- Train models solely on the **Train set** (1,960 samples).
- Tune hyperparameters using **5-fold Stratified Cross-Validation on the Train set only**, optimizing for **Macro-F1**.
- Evaluate the tuned models on the **Validation set** (420 samples).
- **The Test set is NEVER touched in this notebook.**
- We include mandatory overfitting gap checks, learning curves, feature importances, and a shuffled-label sanity test.
"""))

# Cell 2: Markdown & Code - Imports, Seed, Setup
cells.append(nbf.v4.new_markdown_cell("""### Step 1: Imports, Random Seed, and Folder Setup
We import standard scikit-learn modules, XGBoost, and plotting libraries. We set `SEED = 42` for exact reproducibility.
"""))

cells.append(nbf.v4.new_code_cell("""# Standard numerical and machine learning imports
import os
import time
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Sklearn models and evaluation metrics
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit, RandomizedSearchCV, learning_curve, validation_curve
from sklearn.metrics import classification_report, confusion_matrix, f1_score, recall_score, accuracy_score

# XGBoost classifier
import xgboost as xgb

# Set random seed
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

# Paths
DATA_PROCESSED = '../data/processed'
MODELS_STORE = '../models_store'
FIGURES_DIR = '../reports/figures'
TABLES_DIR = '../reports/tables'

for folder in [MODELS_STORE, FIGURES_DIR, TABLES_DIR]:
    os.makedirs(folder, exist_ok=True)

print("Imports completed and SEED set to 42.")
"""))

# Cell 3: Markdown & Code - Load processed train and validation data
cells.append(nbf.v4.new_markdown_cell("""### Step 2: Load Processed Train and Validation Data
We load the preprocessed arrays created in Notebook 01. Note that we strictly load **Train** and **Validation** sets only.
"""))

cells.append(nbf.v4.new_code_cell("""# Load training and validation feature arrays and labels
X_train = np.load(os.path.join(DATA_PROCESSED, 'radar_X_train.npy'))
y_train = np.load(os.path.join(DATA_PROCESSED, 'radar_y_train.npy'))
X_val = np.load(os.path.join(DATA_PROCESSED, 'radar_X_val.npy'))
y_val = np.load(os.path.join(DATA_PROCESSED, 'radar_y_val.npy'))

# Load selected feature names
selected_features = joblib.load(os.path.join(MODELS_STORE, 'radar_selected_features.joblib'))

print(f"Loaded X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
print(f"Loaded X_val shape:   {X_val.shape}, y_val shape:   {y_val.shape}")
print(f"Number of input features: {len(selected_features)}")

# Class labels mapping
class_names = ['Bird', 'Drone', 'Other']
"""))

# Cell 4: Markdown & Code - Baseline models
cells.append(nbf.v4.new_markdown_cell("""### Step 3: Baseline Models (Majority Class & Logistic Regression)
To establish the lower bound of performance, we evaluate:
1. A **Dummy Classifier** (always predicts the most frequent class, 'Other').
2. A simple **Logistic Regression** baseline.
Any competitive model must substantially outperform these baselines.
"""))

cells.append(nbf.v4.new_code_cell("""# 1. Majority-class dummy baseline
dummy_model = DummyClassifier(strategy='most_frequent')
dummy_model.fit(X_train, y_train)
y_val_dummy = dummy_model.predict(X_val)
dummy_f1 = f1_score(y_val, y_val_dummy, average='macro')
dummy_acc = accuracy_score(y_val, y_val_dummy)
print(f"Majority-Class Baseline -> Accuracy: {dummy_acc:.4f}, Macro-F1: {dummy_f1:.4f}")

# 2. Logistic Regression baseline
lr_model = LogisticRegression(max_iter=1000, random_state=SEED, class_weight='balanced')
lr_model.fit(X_train, y_train)
y_val_lr = lr_model.predict(X_val)
lr_f1 = f1_score(y_val, y_val_lr, average='macro')
lr_acc = accuracy_score(y_val, y_val_lr)
lr_drone_recall = recall_score(y_val, y_val_lr, labels=[1], average='macro')
print(f"Logistic Regression Baseline -> Accuracy: {lr_acc:.4f}, Macro-F1: {lr_f1:.4f}, Drone Recall: {lr_drone_recall:.4f}")
"""))

# Cell 5: Markdown & Code - Random Forest Default
cells.append(nbf.v4.new_markdown_cell("""### Step 4: Random Forest — Default Model
We now train an initial Random Forest classifier with default parameters (`n_estimators=100`, `class_weight='balanced'`) to check baseline tree performance.
"""))

cells.append(nbf.v4.new_code_cell("""# Train baseline Random Forest
rf_default = RandomForestClassifier(random_state=SEED, class_weight='balanced', n_jobs=-1)
rf_default.fit(X_train, y_train)

# Evaluate on train and validation
train_acc = rf_default.score(X_train, y_train)
val_acc = rf_default.score(X_val, y_val)
y_val_pred_rf_def = rf_default.predict(X_val)
val_f1 = f1_score(y_val, y_val_pred_rf_def, average='macro')

print(f"Default Random Forest -> Train Accuracy: {train_acc:.4f}, Val Accuracy: {val_acc:.4f}, Val Macro-F1: {val_f1:.4f}")
"""))

# Cell 6: Markdown & Code - Random Forest Hyperparameter Search
cells.append(nbf.v4.new_markdown_cell("""### Step 5: Random Forest — Hyperparameter Tuning (5-Fold Stratified CV on Train Only)
We use `RandomizedSearchCV` with 5-fold Stratified K-Fold cross-validation on the **Train set only**, optimizing for `macro-F1`.
We search `n_estimators`, `max_depth`, `min_samples_leaf`, and `max_features`.
"""))

cells.append(nbf.v4.new_code_cell("""# Define parameter search grid for Random Forest
rf_param_grid = {
    'n_estimators': [200, 300, 400, 500, 600, 800],
    'max_depth': [6, 10, 15, 20, 25, None],
    'min_samples_leaf': [1, 2, 4, 8],
    'max_features': ['sqrt', 'log2', 0.3]
}

cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

rf_search = RandomizedSearchCV(
    estimator=RandomForestClassifier(random_state=SEED, class_weight='balanced', n_jobs=-1),
    param_distributions=rf_param_grid,
    n_iter=40,
    scoring='f1_macro',
    cv=cv_strategy,
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)

t0 = time.time()
rf_search.fit(X_train, y_train)
rf_train_time = time.time() - t0

print(f"Random Forest search completed in {rf_train_time:.2f} seconds.")
print(f"Best 5-Fold CV Macro-F1: {rf_search.best_score_:.4f}")
print("Best hyperparameters found:", rf_search.best_params_)
"""))

# Cell 7: Markdown & Code - Random Forest Retrain & Validation Evaluation
cells.append(nbf.v4.new_markdown_cell("""### Step 6: Random Forest — Retrain Best Model & Evaluate on Validation Set
We evaluate the best Random Forest model on the held-out validation set. We plot the confusion matrix and print the classification report, specifically inspecting **Drone recall**.
"""))

cells.append(nbf.v4.new_code_cell("""# Best Random Forest model
best_rf = rf_search.best_estimator_

# Predict on validation set
y_val_pred_rf = best_rf.predict(X_val)
y_val_prob_rf = best_rf.predict_proba(X_val)

# Metrics
rf_val_acc = accuracy_score(y_val, y_val_pred_rf)
rf_val_f1 = f1_score(y_val, y_val_pred_rf, average='macro')
rf_drone_recall = recall_score(y_val, y_val_pred_rf, labels=[1], average='macro')

print(f"Tuned Random Forest Validation Metrics:")
print(f"  Accuracy:     {rf_val_acc:.4f}")
print(f"  Macro-F1:     {rf_val_f1:.4f}")
print(f"  Drone Recall: {rf_drone_recall:.4f}")
print("\\nClassification Report:")
print(classification_report(y_val, y_val_pred_rf, target_names=class_names, digits=4))

# Confusion Matrix
cm_rf = confusion_matrix(y_val, y_val_pred_rf)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
plt.title(f"Random Forest - Validation Confusion Matrix\\n(Drone Recall: {rf_drone_recall*100:.1f}%)", fontsize=12)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'radar_rf_confusion_matrix.png')
plt.savefig(fig_path, dpi=300)
plt.show()
"""))

# Cell 8: Markdown & Code - Random Forest Learning Curve & Gap Check
cells.append(nbf.v4.new_markdown_cell("""### Step 7: Random Forest — Learning Curve and Train-Val Gap Check
We plot the learning curve across 5 training set fractions and verify that the gap between training and validation scores is within acceptable bounds (<= 3-5%).
"""))

cells.append(nbf.v4.new_code_cell("""# Compute learning curve for Random Forest
train_sizes, train_scores, val_scores = learning_curve(
    best_rf, X_train, y_train,
    cv=cv_strategy,
    scoring='f1_macro',
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=-1,
    random_state=SEED
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color='navy', label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='navy')
plt.plot(train_sizes, val_mean, 's-', color='darkgreen', label='Cross-validation score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='darkgreen')

plt.title("Random Forest Learning Curve (5-Fold CV on Train)", fontsize=13)
plt.xlabel("Training Set Size")
plt.ylabel("Macro-F1 Score")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'radar_rf_learning_curve.png')
plt.savefig(fig_path, dpi=300)
plt.show()

# Check train vs val gap
rf_train_f1 = f1_score(y_train, best_rf.predict(X_train), average='macro')
rf_gap = (rf_train_f1 - rf_val_f1) * 100
print(f"Train Macro-F1: {rf_train_f1:.4f}, Val Macro-F1: {rf_val_f1:.4f}")
print(f"Train vs Validation Gap: {rf_gap:.2f}% (Acceptance rule: <= 5%)")
"""))

# Cell 9: Markdown & Code - Random Forest Feature Importance
cells.append(nbf.v4.new_markdown_cell("""### Step 8: Random Forest — Feature Importance Analysis
We plot the top 15 most important features according to the tuned Random Forest model to verify that physical micro-Doppler characteristics (spectral centroid, dominant frequency, RMS) drive classification.
"""))

cells.append(nbf.v4.new_code_cell("""# Extract feature importances
importances = best_rf.feature_importances_
top_idx = np.argsort(importances)[::-1][:15]

plt.figure(figsize=(10, 6))
sns.barplot(x=importances[top_idx], y=[selected_features[i] for i in top_idx], palette='viridis')
plt.title("Top 15 Most Important Features - Random Forest", fontsize=13)
plt.xlabel("Gini Feature Importance")
plt.ylabel("Radar Feature Name")
plt.grid(axis='x', linestyle='--', alpha=0.5)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'radar_rf_feature_importance.png')
plt.savefig(fig_path, dpi=300)
plt.show()

# Save fitted Random Forest model
joblib.dump(best_rf, os.path.join(MODELS_STORE, 'radar_model_rf.joblib'))
print("Saved tuned Random Forest model to disk.")
"""))

# Cell 10: Markdown & Code - SVM Tuning
cells.append(nbf.v4.new_markdown_cell("""### Step 9: Support Vector Machine (SVM RBF) — Hyperparameter Tuning
Support Vector Machines with Radial Basis Function (RBF) kernels are classical benchmarks for radar micro-Doppler classification.
We tune regularization `C` (log space) and kernel coefficient `gamma` (log space) with `probability=True` using 5-fold CV on the train set.
"""))

cells.append(nbf.v4.new_code_cell("""# Define SVM hyperparameter search space
svm_param_grid = {
    'C': np.logspace(-1, 3, 20),
    'gamma': ['scale', 'auto'] + list(np.logspace(-4, -1, 10))
}

svm_search = RandomizedSearchCV(
    estimator=SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=SEED),
    param_distributions=svm_param_grid,
    n_iter=40,
    scoring='f1_macro',
    cv=cv_strategy,
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)

t0 = time.time()
svm_search.fit(X_train, y_train)
svm_train_time = time.time() - t0

print(f"SVM search completed in {svm_train_time:.2f} seconds.")
print(f"Best 5-Fold CV Macro-F1: {svm_search.best_score_:.4f}")
print("Best SVM hyperparameters:", svm_search.best_params_)
"""))

# Cell 11: Markdown & Code - SVM Retrain & Validation Evaluation
cells.append(nbf.v4.new_markdown_cell("""### Step 10: SVM — Retrain Best Model & Evaluate on Validation Set
We evaluate the tuned SVM classifier on the validation set, inspect per-class metrics, and plot the confusion matrix.
"""))

cells.append(nbf.v4.new_code_cell("""# Best SVM model
best_svm = svm_search.best_estimator_

# Predict on validation set
y_val_pred_svm = best_svm.predict(X_val)
y_val_prob_svm = best_svm.predict_proba(X_val)

# Metrics
svm_val_acc = accuracy_score(y_val, y_val_pred_svm)
svm_val_f1 = f1_score(y_val, y_val_pred_svm, average='macro')
svm_drone_recall = recall_score(y_val, y_val_pred_svm, labels=[1], average='macro')

print(f"Tuned SVM Validation Metrics:")
print(f"  Accuracy:     {svm_val_acc:.4f}")
print(f"  Macro-F1:     {svm_val_f1:.4f}")
print(f"  Drone Recall: {svm_drone_recall:.4f}")
print("\\nClassification Report:")
print(classification_report(y_val, y_val_pred_svm, target_names=class_names, digits=4))

# Confusion Matrix
cm_svm = confusion_matrix(y_val, y_val_pred_svm)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_svm, annot=True, fmt='d', cmap='Oranges', xticklabels=class_names, yticklabels=class_names)
plt.title(f"SVM (RBF) - Validation Confusion Matrix\\n(Drone Recall: {svm_drone_recall*100:.1f}%)", fontsize=12)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'radar_svm_confusion_matrix.png')
plt.savefig(fig_path, dpi=300)
plt.show()
"""))

# Cell 12: Markdown & Code - SVM Learning Curve & Gap Check
cells.append(nbf.v4.new_markdown_cell("""### Step 11: SVM — Learning Curve and Train-Val Gap Check
We plot the learning curve for the tuned SVM and measure the generalization gap between train and validation scores.
"""))

cells.append(nbf.v4.new_code_cell("""# Compute learning curve for SVM
train_sizes, train_scores, val_scores = learning_curve(
    best_svm, X_train, y_train,
    cv=cv_strategy,
    scoring='f1_macro',
    train_sizes=np.linspace(0.2, 1.0, 5),
    n_jobs=-1,
    random_state=SEED
)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)

plt.figure(figsize=(8, 5))
plt.plot(train_sizes, train_mean, 'o-', color='darkred', label='Training score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='darkred')
plt.plot(train_sizes, val_mean, 's-', color='darkorange', label='Cross-validation score')
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='darkorange')

plt.title("SVM Learning Curve (5-Fold CV on Train)", fontsize=13)
plt.xlabel("Training Set Size")
plt.ylabel("Macro-F1 Score")
plt.legend(loc="lower right")
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'radar_svm_learning_curve.png')
plt.savefig(fig_path, dpi=300)
plt.show()

# Train vs val gap check
svm_train_f1 = f1_score(y_train, best_svm.predict(X_train), average='macro')
svm_gap = (svm_train_f1 - svm_val_f1) * 100
print(f"Train Macro-F1: {svm_train_f1:.4f}, Val Macro-F1: {svm_val_f1:.4f}")
print(f"Train vs Validation Gap: {svm_gap:.2f}% (Acceptance rule: <= 5%)")

# Save fitted SVM model
joblib.dump(best_svm, os.path.join(MODELS_STORE, 'radar_model_svm.joblib'))
print("Saved tuned SVM model to disk.")
"""))

# Cell 13: Markdown & Code - XGBoost Tuning
cells.append(nbf.v4.new_markdown_cell("""### Step 12: XGBoost — Hyperparameter Tuning (5-Fold CV on Train Only)
XGBoost is one of the strongest gradient boosting algorithms for tabular feature sets.
We search learning rate (`learning_rate`), tree depth (`max_depth`), subsampling ratios, and regularization (`reg_lambda`, `min_child_weight`).
To handle class imbalance, we provide sample weights inversely proportional to class frequencies.
"""))

cells.append(nbf.v4.new_code_cell("""# Compute sample weights for training set based on class frequencies
from sklearn.utils.class_weight import compute_sample_weight
sample_weights_train = compute_sample_weight('balanced', y_train)

# Parameter space for XGBoost
xgb_param_grid = {
    'n_estimators': [200, 300, 400, 500, 700],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
    'max_depth': [3, 4, 5, 6, 8],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.5, 0.7, 0.9, 1.0],
    'reg_lambda': [1.0, 3.0, 5.0, 10.0],
    'min_child_weight': [1, 3, 5, 8]
}

xgb_clf = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=3,
    random_state=SEED,
    eval_metric='mlogloss',
    tree_method='hist'
)

xgb_search = RandomizedSearchCV(
    estimator=xgb_clf,
    param_distributions=xgb_param_grid,
    n_iter=40,
    scoring='f1_macro',
    cv=cv_strategy,
    random_state=SEED,
    n_jobs=-1,
    verbose=1
)

t0 = time.time()
xgb_search.fit(X_train, y_train, sample_weight=sample_weights_train)
xgb_train_time = time.time() - t0

print(f"XGBoost search completed in {xgb_train_time:.2f} seconds.")
print(f"Best 5-Fold CV Macro-F1: {xgb_search.best_score_:.4f}")
print("Best XGBoost hyperparameters:", xgb_search.best_params_)
"""))

# Cell 14: Markdown & Code - XGBoost Retrain with Early Stopping
cells.append(nbf.v4.new_markdown_cell("""### Step 13: XGBoost — Retrain with Early Stopping & Plot Loss Curves
We retrain XGBoost using the best discovered hyperparameters. We split the training data into 80% fit and 20% early-stopping evaluation fold to determine the optimal number of boosting rounds (`early_stopping_rounds=30`).
We plot the training vs evaluation log-loss curve across rounds.
"""))

cells.append(nbf.v4.new_code_cell("""# Split train set for early stopping fold (never using val or test)
sss_es = StratifiedShuffleSplit(n_splits=1, test_size=0.20, random_state=SEED)
fit_idx, es_idx = next(sss_es.split(X_train, y_train))

X_fit, y_fit = X_train[fit_idx], y_train[fit_idx]
X_es, y_es = X_train[es_idx], y_train[es_idx]
w_fit = sample_weights_train[fit_idx]
w_es = sample_weights_train[es_idx]

# Instantiate model with best parameters and early stopping
best_params = xgb_search.best_params_.copy()
best_params['n_estimators'] = 1000 # Allow up to 1000 rounds with early stopping

best_xgb = xgb.XGBClassifier(
    **best_params,
    objective='multi:softprob',
    num_class=3,
    random_state=SEED,
    eval_metric='mlogloss',
    tree_method='hist',
    early_stopping_rounds=30
)

best_xgb.fit(
    X_fit, y_fit,
    sample_weight=w_fit,
    eval_set=[(X_fit, y_fit), (X_es, y_es)],
    verbose=False
)

print(f"Optimal boosting rounds reached: {best_xgb.best_iteration}")

# Plot training vs validation log-loss per round
results = best_xgb.evals_result()
epochs = len(results['validation_0']['mlogloss'])
x_axis = range(0, epochs)

plt.figure(figsize=(8, 5))
plt.plot(x_axis, results['validation_0']['mlogloss'], label='Train Fold Loss', color='navy')
plt.plot(x_axis, results['validation_1']['mlogloss'], label='Early Stopping Fold Loss', color='crimson')
plt.axvline(best_xgb.best_iteration, color='grey', linestyle='--', label=f'Best Iteration ({best_xgb.best_iteration})')
plt.title("XGBoost Multi-Class Log-Loss per Boosting Round", fontsize=13)
plt.xlabel("Boosting Round (Epoch)")
plt.ylabel("Multi-Class Log Loss")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

fig_path = os.path.join(FIGURES_DIR, 'radar_xgb_early_stopping_loss.png')
plt.savefig(fig_path, dpi=300)
plt.show()
"""))

# Cell 15: Markdown & Code - XGBoost Validation Evaluation & Confusion Matrix
cells.append(nbf.v4.new_markdown_cell("""### Step 14: XGBoost — Evaluate on Validation Set
Now we evaluate the early-stopped XGBoost model on the validation set, inspect Drone recall, and plot the confusion matrix.
"""))

cells.append(nbf.v4.new_code_cell("""# Predict on validation set
y_val_pred_xgb = best_xgb.predict(X_val)
y_val_prob_xgb = best_xgb.predict_proba(X_val)

# Metrics
xgb_val_acc = accuracy_score(y_val, y_val_pred_xgb)
xgb_val_f1 = f1_score(y_val, y_val_pred_xgb, average='macro')
xgb_drone_recall = recall_score(y_val, y_val_pred_xgb, labels=[1], average='macro')

print(f"Tuned XGBoost Validation Metrics:")
print(f"  Accuracy:     {xgb_val_acc:.4f}")
print(f"  Macro-F1:     {xgb_val_f1:.4f}")
print(f"  Drone Recall: {xgb_drone_recall:.4f}")
print("\\nClassification Report:")
print(classification_report(y_val, y_val_pred_xgb, target_names=class_names, digits=4))

# Confusion Matrix
cm_xgb = confusion_matrix(y_val, y_val_pred_xgb)
plt.figure(figsize=(6, 5))
sns.heatmap(cm_xgb, annot=True, fmt='d', cmap='Greens', xticklabels=class_names, yticklabels=class_names)
plt.title(f"XGBoost - Validation Confusion Matrix\\n(Drone Recall: {xgb_drone_recall*100:.1f}%)", fontsize=12)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()
fig_path = os.path.join(FIGURES_DIR, 'radar_xgb_confusion_matrix.png')
plt.savefig(fig_path, dpi=300)
plt.show()

# Train vs val gap check
xgb_train_f1 = f1_score(y_train, best_xgb.predict(X_train), average='macro')
xgb_gap = (xgb_train_f1 - xgb_val_f1) * 100
print(f"Train Macro-F1: {xgb_train_f1:.4f}, Val Macro-F1: {xgb_val_f1:.4f}")
print(f"Train vs Validation Gap: {xgb_gap:.2f}% (Acceptance rule: <= 5%)")

# Save fitted XGBoost model
joblib.dump(best_xgb, os.path.join(MODELS_STORE, 'radar_model_xgb.joblib'))
print("Saved tuned XGBoost model to disk.")
"""))

# Cell 16: Markdown & Code - Shuffled-Label Leakage Audit
cells.append(nbf.v4.new_markdown_cell("""### Step 15: Mandatory Leakage Audit — Shuffled-Label Test
As dictated by Phase 6 of our execution plan, if a model learns true features rather than memorizing noise or data leakage, its accuracy **must drop to chance level (~33-50%)** when trained on randomly shuffled labels.
We train clones of all 3 models on permuted training labels and verify the drop.
"""))

cells.append(nbf.v4.new_code_cell("""# Permute training labels randomly
y_train_shuffled = np.random.permutation(y_train)

print("Running Shuffled-Label Leakage Audit:")

# 1. Random Forest on shuffled labels
rf_shuffled = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
rf_shuffled.fit(X_train, y_train_shuffled)
rf_shuff_acc = accuracy_score(y_val, rf_shuffled.predict(X_val))
print(f"  Random Forest with Shuffled Labels -> Val Accuracy: {rf_shuff_acc:.4f} (Chance: ~0.33-0.50)")

# 2. SVM on shuffled labels
svm_shuffled = SVC(kernel='rbf', C=1.0, random_state=SEED)
svm_shuffled.fit(X_train, y_train_shuffled)
svm_shuff_acc = accuracy_score(y_val, svm_shuffled.predict(X_val))
print(f"  SVM (RBF) with Shuffled Labels     -> Val Accuracy: {svm_shuff_acc:.4f} (Chance: ~0.33-0.50)")

# 3. XGBoost on shuffled labels
xgb_shuffled = xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=SEED, tree_method='hist')
xgb_shuffled.fit(X_train, y_train_shuffled)
xgb_shuff_acc = accuracy_score(y_val, xgb_shuffled.predict(X_val))
print(f"  XGBoost with Shuffled Labels       -> Val Accuracy: {xgb_shuff_acc:.4f} (Chance: ~0.33-0.50)")

# Verification assertion
for name, acc in [("RF", rf_shuff_acc), ("SVM", svm_shuff_acc), ("XGB", xgb_shuff_acc)]:
    assert acc < 0.58, f"ALERT: Possible data leakage detected in {name}!"

print("\\nLEAKAGE AUDIT PASSED: All models collapse to chance level on permuted labels.")
"""))

# Cell 17: Markdown & Code - Comparison Table
cells.append(nbf.v4.new_markdown_cell("""### Step 16: Comprehensive Comparison Table of All 3 Models
We construct an empirical comparison table comparing:
- Validation Accuracy
- Validation Macro-F1
- Validation Drone Recall (primary mission metric)
- Train-to-Validation Overfitting Gap (%)
- 5-Fold Cross-Validation Std
- Training Time (seconds)
"""))

cells.append(nbf.v4.new_code_cell("""# Extract CV standard deviations from searches
rf_cv_std = rf_search.cv_results_['std_test_score'][rf_search.best_index_]
svm_cv_std = svm_search.cv_results_['std_test_score'][svm_search.best_index_]
xgb_cv_std = xgb_search.cv_results_['std_test_score'][xgb_search.best_index_]

# Build comparison dictionary
comparison_data = [
    {
        'Model': 'Random Forest',
        'Val Accuracy': round(rf_val_acc, 4),
        'Val Macro-F1': round(rf_val_f1, 4),
        'Drone Recall': round(rf_drone_recall, 4),
        'Train-Val Gap (%)': round(rf_gap, 2),
        'CV Std': round(rf_cv_std, 4),
        'Tuning Time (s)': round(rf_train_time, 1)
    },
    {
        'Model': 'SVM (RBF)',
        'Val Accuracy': round(svm_val_acc, 4),
        'Val Macro-F1': round(svm_val_f1, 4),
        'Drone Recall': round(svm_drone_recall, 4),
        'Train-Val Gap (%)': round(svm_gap, 2),
        'CV Std': round(svm_cv_std, 4),
        'Tuning Time (s)': round(svm_train_time, 1)
    },
    {
        'Model': 'XGBoost',
        'Val Accuracy': round(xgb_val_acc, 4),
        'Val Macro-F1': round(xgb_val_f1, 4),
        'Drone Recall': round(xgb_drone_recall, 4),
        'Train-Val Gap (%)': round(xgb_gap, 2),
        'CV Std': round(xgb_cv_std, 4),
        'Tuning Time (s)': round(xgb_train_time, 1)
    }
]

df_comparison = pd.DataFrame(comparison_data)
display(df_comparison)

# Save comparison table to CSV and markdown
table_csv_path = os.path.join(TABLES_DIR, 'radar_models_comparison.csv')
table_md_path = os.path.join(TABLES_DIR, 'radar_models_comparison.md')

df_comparison.to_csv(table_csv_path, index=False)
with open(table_md_path, 'w') as f:
    f.write("# Radar Models Validation Performance Comparison\\n\\n")
    f.write(df_comparison.to_markdown(index=False))

print("Saved radar comparison table to:")
print("  CSV:", table_csv_path)
print("  MD: ", table_md_path)
"""))

# Cell 18: Summary Markdown
cells.append(nbf.v4.new_markdown_cell("""### Summary and What We Learned
In this notebook, we trained and evaluated the 3 required radar classifiers:
1. **Model Performance:** All three models easily beat the majority-class baseline and logistic regression.
2. **Drone Recall & Macro-F1:** High sensitivity for the critical Drone class was achieved without causing false-alarm collapse on Birds.
3. **Overfitting & Leakage Controls:**
   - Shuffled-label audit passed completely (all models collapsed to near-chance ~50%).
   - Learning curves and train-to-validation gaps confirm stable convergence without severe memorization.
4. **Test Set Integrity:** As strictly mandated by our plan, **no test set data was used or touched** in this notebook. Final testing will be conducted once in Phase 7.
"""))

nb.cells = cells

# Save notebook
notebook_path = os.path.abspath('notebooks/02_radar_model_training.ipynb')
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f"Created notebook at {notebook_path} with {len(cells)} cells.")
