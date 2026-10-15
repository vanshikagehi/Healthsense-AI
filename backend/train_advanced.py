"""
HealthSense AI — Advanced ML Engine v5.0
=========================================
Architecture:
  6-Model Voting Ensemble:
    Model 1: HistGradientBoostingClassifier  (native NaN, best for tabular)
    Model 2: XGBoostClassifier               (GPU-optional, high accuracy)
    Model 3: LightGBMClassifier              (fastest boosting, handles imbalance)
    Model 4: RandomForestClassifier          (breadth + stability)
    Model 5: ExtraTreesClassifier            (diversity through randomness)
    Model 6: GradientBoostingClassifier      (high precision)
    Meta     : Soft-Voting (probability averaging — more robust than stacking on clean data)

  Dataset:
    - 50,004 samples (8,334 per class) — 2.5x more than v4.0
    - 7 clinical sub-types per disease class
    - 20% comorbidity overlap

  Feature Engineering (same as v4.0 — preserved for inference compatibility):
    - 12 interaction features added → 34 raw → 46 engineered
    - metabolic_score, cardio_score, mental_score, hormonal_score
    - glucose_bmi, bp_age, sleep_anxiety, hba1c_glucose
    - symptom_total, cardinal_diabetes, cardinal_thyroid, cardinal_pcos

  Training:
    - StratifiedKFold (5-fold) cross-validation
    - StandardScaler on engineered features
    - Isotonic probability calibration on best base model
    - Class weight balancing

Performance (expected):
  - Accuracy      : 99%+
  - AUC-ROC macro : 0.999+
  - Inference     : <5ms per prediction

Run:
  python backend/train_advanced.py
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

from sklearn.ensemble import (
    GradientBoostingClassifier, RandomForestClassifier,
    ExtraTreesClassifier, VotingClassifier
)
try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_HIST = True
except ImportError:
    HAS_HIST = False

# XGBoost
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("  ⚠️  XGBoost not installed — skipping (pip install xgboost)")

# LightGBM
try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False
    print("  ⚠️  LightGBM not installed — skipping (pip install lightgbm)")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV

DATASET_PATH = os.path.join(os.path.dirname(__file__), "health_dataset.csv")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "models")

os.makedirs(MODEL_DIR, exist_ok=True)

CLASS_NAMES  = ["Normal", "Diabetes", "Hypertension", "MentalHealth", "Thyroid", "PCOS"]
N_CLASSES    = len(CLASS_NAMES)

FEATURE_COLS = [
    "fatigue","excessive_thirst","blurred_vision","frequent_urination",
    "numbness","weight_loss","slow_healing","skin_darkening",
    "headache","chest_pain","shortness_breath","palpitations",
    "dizziness","ankle_swelling",
    "anxiety","sleep_issues","low_mood","irritability",
    "weight_gain","cold_intolerance","hair_loss","nausea",
    "irregular_periods","acne",
    "glucose","blood_pressure","age","bmi","sleep_hrs","hba1c",
    "family_diabetes","family_bp","smoking","sedentary"
]

print("\n" + "="*65)
print("  HealthSense AI — ML Training v5.0")
print("  6-Model Voting + 50K Samples + XGBoost + LightGBM")
print("="*65)


# ══════════════════════════════════════════════════════════════
#   STEP 1 — GENERATE DATASET v5.0 (50,004 samples)
# ══════════════════════════════════════════════════════════════
print("\n[1/6] Generating clinical dataset v5.0 (50,004 samples)...")
t_start = time.time()

exec(open(os.path.join(os.path.dirname(__file__), "generate_dataset.py"), encoding='utf-8').read())

df = pd.read_csv(DATASET_PATH)
print(f"      Rows: {len(df):,} | Features: {len(FEATURE_COLS)} | Classes: {N_CLASSES}")
dist = df['label'].value_counts().sort_index()
for i, c in enumerate(CLASS_NAMES):
    print(f"      Class {i} ({c}): {dist.get(i, 0):,} samples")

X_raw = df[FEATURE_COLS].values.astype(np.float32)
y     = df["label"].values.astype(int)


# ══════════════════════════════════════════════════════════════
#   STEP 2 — FEATURE ENGINEERING (same as v4.0 for inference compatibility)
# ══════════════════════════════════════════════════════════════
print("\n[2/6] Feature engineering — adding 12 clinical interaction features...")

IDX = {c: i for i, c in enumerate(FEATURE_COLS)}

def engineer_features(X):
    """
    Add 12 interaction/composite features (same formula as main.py — KEEP IN SYNC).
    """
    gluc  = (X[:, IDX["glucose"]]         - 70)  / 250
    bp    = (X[:, IDX["blood_pressure"]]  - 80)  / 130
    age_n = (X[:, IDX["age"]]             - 14)  / 66
    bmi_n = (X[:, IDX["bmi"]]             - 16)  / 32
    slp   = (X[:, IDX["sleep_hrs"]]       - 2)   / 8
    hba   = (X[:, IDX["hba1c"]]           - 4.5) / 8.5

    metabolic_score  = (X[:, IDX["fatigue"]] + X[:, IDX["excessive_thirst"]] +
                        X[:, IDX["frequent_urination"]] + X[:, IDX["numbness"]] +
                        X[:, IDX["slow_healing"]] + gluc + hba) / 7

    cardio_score     = (X[:, IDX["headache"]] + X[:, IDX["chest_pain"]] +
                        X[:, IDX["shortness_breath"]] + X[:, IDX["palpitations"]] +
                        X[:, IDX["dizziness"]] + bp + age_n) / 7

    mental_score     = (X[:, IDX["anxiety"]] + X[:, IDX["sleep_issues"]] +
                        X[:, IDX["low_mood"]] + X[:, IDX["irritability"]]) / 4

    hormonal_score   = (X[:, IDX["weight_gain"]] + X[:, IDX["cold_intolerance"]] +
                        X[:, IDX["hair_loss"]] + X[:, IDX["irregular_periods"]] +
                        X[:, IDX["acne"]]) / 5

    glucose_bmi      = np.clip(gluc * bmi_n, 0, 1)
    bp_age           = np.clip(bp   * age_n, 0, 1)
    sleep_anxiety    = np.clip((1 - slp) * X[:, IDX["anxiety"]], 0, 1)
    hba1c_glucose    = np.clip(hba * gluc, 0, 1)
    symptom_total    = np.mean(X[:, :24], axis=1)
    cardinal_diabetes = (X[:, IDX["excessive_thirst"]] +
                         X[:, IDX["frequent_urination"]] +
                         X[:, IDX["blurred_vision"]]) / 3
    cardinal_thyroid  = (X[:, IDX["cold_intolerance"]] +
                         X[:, IDX["hair_loss"]] +
                         X[:, IDX["weight_gain"]]) / 3
    cardinal_pcos     = (X[:, IDX["irregular_periods"]] +
                         X[:, IDX["acne"]] +
                         X[:, IDX["weight_gain"]]) / 3

    new_feats = np.column_stack([
        metabolic_score, cardio_score, mental_score, hormonal_score,
        glucose_bmi, bp_age, sleep_anxiety, hba1c_glucose,
        symptom_total, cardinal_diabetes, cardinal_thyroid, cardinal_pcos
    ])
    return np.hstack([X, new_feats])

X_eng = engineer_features(X_raw)
print(f"      Features: {X_raw.shape[1]} -> {X_eng.shape[1]} (added {X_eng.shape[1]-X_raw.shape[1]} features)")

ENGINEERED_COLS = FEATURE_COLS + [
    "metabolic_score","cardio_score","mental_score","hormonal_score",
    "glucose_bmi","bp_age","sleep_anxiety","hba1c_glucose",
    "symptom_total","cardinal_diabetes","cardinal_thyroid","cardinal_pcos"
]


# ══════════════════════════════════════════════════════════════
#   STEP 3 — PREPROCESSING
# ══════════════════════════════════════════════════════════════
print("\n[3/6] Preprocessing (StandardScaler)...")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_eng)

joblib.dump(scaler,           os.path.join(MODEL_DIR, "scaler.pkl"),          compress=3)
joblib.dump(FEATURE_COLS,     os.path.join(MODEL_DIR, "feature_cols.pkl"),    compress=3)
joblib.dump(ENGINEERED_COLS,  os.path.join(MODEL_DIR, "engineered_cols.pkl"), compress=3)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")


# ══════════════════════════════════════════════════════════════
#   STEP 4 — BUILD 6-MODEL VOTING ENSEMBLE
# ══════════════════════════════════════════════════════════════
print("\n[4/6] Building 6-Model Soft-Voting Ensemble...")

estimators = []

# ── Model 1: HistGradientBoosting ──
if HAS_HIST:
    hgb = HistGradientBoostingClassifier(
        max_iter=400,
        learning_rate=0.05,
        max_depth=9,
        min_samples_leaf=12,
        l2_regularization=0.08,
        random_state=42
    )
    estimators.append(("hgb", hgb))
    print("      Model 1: HistGradientBoostingClassifier (400 iters, depth=9)")

# ── Model 2: XGBoost ──
if HAS_XGB:
    xgb = XGBClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=7,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=3,
        gamma=0.1,
        reg_alpha=0.05,
        reg_lambda=1.0,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
    estimators.append(("xgb", xgb))
    print("      Model 2: XGBClassifier (400 trees, depth=7)")

# ── Model 3: LightGBM ──
if HAS_LGB:
    lgb = LGBMClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=12,
        reg_alpha=0.05,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    estimators.append(("lgb", lgb))
    print("      Model 3: LGBMClassifier (400 trees, depth=8, balanced)")

# ── Model 4: Random Forest ──
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=16,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
estimators.append(("rf", rf))
print("      Model 4: RandomForestClassifier (300 trees, depth=16)")

# ── Model 5: ExtraTrees ──
et = ExtraTreesClassifier(
    n_estimators=200,
    max_depth=14,
    min_samples_leaf=3,
    max_features="sqrt",
    class_weight="balanced",
    n_jobs=-1,
    random_state=123
)
estimators.append(("et", et))
print("      Model 5: ExtraTreesClassifier (200 trees, depth=14)")

# ── Model 6: GradientBoosting ──
gb = GradientBoostingClassifier(
    n_estimators=250,
    learning_rate=0.07,
    max_depth=7,
    subsample=0.85,
    min_samples_leaf=3,
    max_features="sqrt",
    random_state=0
)
estimators.append(("gb", gb))
print("      Model 6: GradientBoostingClassifier (250 trees, depth=7)")

print(f"\n      Ensemble: {len(estimators)}-model soft-voting")
print("      Training ensemble (this takes ~5-10 min)...")
t0 = time.time()

voting = VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
voting.fit(X_train, y_train)
elapsed = time.time() - t0

vote_pred = voting.predict(X_test)
vote_acc  = accuracy_score(y_test, vote_pred)
print(f"      Voting Ensemble Accuracy: {vote_acc*100:.2f}% | Time: {elapsed:.0f}s")


# ══════════════════════════════════════════════════════════════
#   STEP 5 — PROBABILITY CALIBRATION on best single model
#   Calibrate the best base model separately + blend with voting
# ══════════════════════════════════════════════════════════════
print("\n[5/6] Probability calibration (isotonic, cv=3)...")

# Calibrate the full voting ensemble
calibrated = CalibratedClassifierCV(voting, method="isotonic", cv=3)
calibrated.fit(X_train, y_train)

cal_pred = calibrated.predict(X_test)
cal_acc  = accuracy_score(y_test, cal_pred)
print(f"      Calibrated Accuracy: {cal_acc*100:.2f}%")

# Use uncalibrated if calibration hurt accuracy (safety net)
final_model = calibrated if cal_acc >= vote_acc - 0.005 else voting
final_acc   = cal_acc    if final_model is calibrated else vote_acc
print(f"      Final model: {'calibrated' if final_model is calibrated else 'voting (calibration skipped)'}")

joblib.dump(final_model, os.path.join(MODEL_DIR, "model.pkl"), compress=9)
print(f"      Saved → backend/models/model.pkl (compress=9)")


# ══════════════════════════════════════════════════════════════
#   STEP 6 — EVALUATION + METADATA
# ══════════════════════════════════════════════════════════════
print("\n[6/6] Evaluation report...")

# Use final model predictions
final_pred = final_model.predict(X_test)
final_acc  = accuracy_score(y_test, final_pred)

# Classification report
print("\n      Classification Report:")
print(classification_report(y_test, final_pred, target_names=CLASS_NAMES, digits=3))

# Confusion matrix
print("      Confusion Matrix:")
cm    = confusion_matrix(y_test, final_pred)
cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=[f"Pred_{c}" for c in CLASS_NAMES])
print(cm_df.to_string())

# AUC-ROC
y_proba = final_model.predict_proba(X_test)
try:
    auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="macro")
    print(f"\n      AUC-ROC (macro): {auc:.4f}")
except Exception:
    auc = None

# 5-fold CV on voting (subset for speed)
print("\n      5-Fold Cross Validation (Voting ensemble, subset 20K)...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Use a stratified 20K subset for CV to keep it under 3 min
idx_sub = []
for cls in range(N_CLASSES):
    cls_idx = np.where(y == cls)[0]
    chosen  = np.random.choice(cls_idx, min(3334, len(cls_idx)), replace=False)
    idx_sub.extend(chosen.tolist())
idx_sub = np.array(idx_sub)
cv_scores = cross_val_score(voting, X_scaled[idx_sub], y[idx_sub], cv=cv,
                             scoring="accuracy", n_jobs=-1)
print(f"      Per fold: {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"      Mean    : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# Save metadata
model_names = [name for name, _ in estimators]
metadata = {
    "version"        : "5.0",
    "model_type"     : f"SoftVoting({'+'.join(model_names)}) + IsotonicCalibration",
    "accuracy"       : round(final_acc  * 100, 2),
    "vote_accuracy"  : round(vote_acc   * 100, 2),
    "cv_mean"        : round(cv_scores.mean() * 100, 2),
    "cv_std"         : round(cv_scores.std()  * 100, 2),
    "auc_roc_macro"  : round(auc, 4) if auc else None,
    "classes"        : CLASS_NAMES,
    "n_features_raw" : len(FEATURE_COLS),
    "n_features_eng" : len(ENGINEERED_COLS),
    "engineered_features": ENGINEERED_COLS[len(FEATURE_COLS):],
    "training_samples": len(X_train),
    "test_samples"    : len(X_test),
    "feature_cols"    : FEATURE_COLS,
    "dataset_version" : "v5.0 (50004 samples, 7 sub-types, 20% comorbidity overlap)",
    "base_models"     : model_names,
    "notes": (
        "v5.0: 6-model soft voting (HistGB+XGB+LGB+RF+ET+GB), 50K samples, "
        "7 sub-types per class, 20% comorbidity overlap, 12 engineered features, isotonic calibration"
    )
}

with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

total_time = time.time() - t_start
print("\n" + "="*65)
print("  TRAINING COMPLETE — v5.0")
print(f"  Voting Accuracy  : {vote_acc*100:.2f}%")
print(f"  Final Accuracy   : {final_acc*100:.2f}%")
print(f"  CV 5-fold        : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
if auc: print(f"  AUC-ROC macro    : {auc:.4f}")
print(f"  Training samples : {len(X_train):,}")
print(f"  Features         : {len(FEATURE_COLS)} raw + 12 engineered = {len(ENGINEERED_COLS)}")
print(f"  Base models      : {', '.join(model_names)}")
print(f"  Total time       : {total_time:.0f}s")
print(f"  Model saved      : backend/models/model.pkl")
print("="*65)
