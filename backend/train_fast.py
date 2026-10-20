"""
HealthSense AI — Fast ML Engine v5.1
=====================================
Approach: Single best model (LightGBM, fallback → HistGradientBoostingClassifier)
          instead of 6-model voting ensemble.

Why faster:
  - 20,004 samples (3,334 per class) instead of 50,004
  - 1 model instead of 6-model voting ensemble
  - No isotonic calibration (cv=2 only if needed)
  - No heavy 5-fold CV on full set — quick 3-fold on 10K subset

Expected training time: ~45s–90s (vs 5–10 min for v5.0)
Expected accuracy: 97–99% (synthetic data separates cleanly)

Run:
  python backend/train_fast.py
"""

import os, json, time, warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings('ignore')

from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

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

print("\n" + "="*60)
print("  HealthSense AI — Fast ML Training v5.1")
print("  Single Best Model + 20K Samples (~1-2 min)")
print("="*60)

# ══════════════════════════════════════════════════════
#   STEP 1 — GENERATE SMALL DATASET (20,004 samples)
# ══════════════════════════════════════════════════════
print("\n[1/5] Generating fast dataset (20,004 samples)...")
t_start = time.time()

# Temporarily override SAMPLES_PER_CLASS for speed
import importlib, sys

_gen_path = os.path.join(os.path.dirname(__file__), "generate_dataset.py")
_gen_src  = open(_gen_path, encoding="utf-8").read()
# Override samples per class: 3334 × 6 = 20,004
_gen_src  = _gen_src.replace("SAMPLES_PER_CLASS = 8334", "SAMPLES_PER_CLASS = 3334")
exec(_gen_src, {"__file__": os.path.join(os.path.dirname(__file__), "generate_dataset.py")})

df = pd.read_csv(DATASET_PATH)
print(f"      Rows: {len(df):,} | Features: {len(FEATURE_COLS)} | Classes: {N_CLASSES}")
dist = df['label'].value_counts().sort_index()
for i, c in enumerate(CLASS_NAMES):
    print(f"      Class {i} ({c}): {dist.get(i, 0):,} samples")

X_raw = df[FEATURE_COLS].values.astype(np.float32)
y     = df["label"].values.astype(int)


# ══════════════════════════════════════════════════════
#   STEP 2 — FEATURE ENGINEERING (same as v5.0)
# ══════════════════════════════════════════════════════
print("\n[2/5] Feature engineering (12 interaction features)...")

IDX = {c: i for i, c in enumerate(FEATURE_COLS)}

def engineer_features(X):
    """Mirror of train_advanced.py — KEEP IN SYNC with main.py."""
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
print(f"      Features: {X_raw.shape[1]} -> {X_eng.shape[1]}")

ENGINEERED_COLS = FEATURE_COLS + [
    "metabolic_score","cardio_score","mental_score","hormonal_score",
    "glucose_bmi","bp_age","sleep_anxiety","hba1c_glucose",
    "symptom_total","cardinal_diabetes","cardinal_thyroid","cardinal_pcos"
]


# ══════════════════════════════════════════════════════
#   STEP 3 — PREPROCESSING
# ══════════════════════════════════════════════════════
print("\n[3/5] Preprocessing (StandardScaler)...")

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_eng)

joblib.dump(scaler,           os.path.join(MODEL_DIR, "scaler.pkl"),          compress=3)
joblib.dump(FEATURE_COLS,     os.path.join(MODEL_DIR, "feature_cols.pkl"),    compress=3)
joblib.dump(ENGINEERED_COLS,  os.path.join(MODEL_DIR, "engineered_cols.pkl"), compress=3)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.15, random_state=42, stratify=y
)
print(f"      Train: {len(X_train):,} | Test: {len(X_test):,}")


# ══════════════════════════════════════════════════════
#   STEP 4 — PICK + TRAIN SINGLE BEST MODEL
#   Priority: LightGBM > HistGBM > RandomForest
# ══════════════════════════════════════════════════════
print("\n[4/5] Training single best model (fastest approach)...")

t0 = time.time()

if HAS_LGB:
    model = LGBMClassifier(
        n_estimators=300,
        learning_rate=0.08,
        max_depth=7,
        num_leaves=63,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=10,
        reg_alpha=0.05,
        reg_lambda=0.1,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1
    )
    model_name = "LGBMClassifier"
else:
    model = HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.08,
        max_depth=8,
        min_samples_leaf=10,
        l2_regularization=0.08,
        random_state=42
    )
    model_name = "HistGradientBoostingClassifier"

print(f"      Using: {model_name}")
model.fit(X_train, y_train)
elapsed = time.time() - t0

pred = model.predict(X_test)
acc  = accuracy_score(y_test, pred)
print(f"      Accuracy: {acc*100:.2f}% | Train time: {elapsed:.0f}s")

# Save model
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"), compress=6)
print("      Saved -> backend/models/model.pkl")


# ══════════════════════════════════════════════════════
#   STEP 5 — QUICK EVALUATION
# ══════════════════════════════════════════════════════
print("\n[5/5] Evaluation...")

print("\n      Classification Report:")
print(classification_report(y_test, pred, target_names=CLASS_NAMES, digits=3))

# AUC
proba = model.predict_proba(X_test)
try:
    auc = roc_auc_score(y_test, proba, multi_class="ovr", average="macro")
    print(f"      AUC-ROC (macro): {auc:.4f}")
except Exception:
    auc = None

# Quick 3-fold CV on 10K subset
print("\n      3-Fold CV (10K subset)...")
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
idx_sub = []
for cls in range(N_CLASSES):
    cls_idx = np.where(y == cls)[0]
    chosen  = np.random.choice(cls_idx, min(1667, len(cls_idx)), replace=False)
    idx_sub.extend(chosen.tolist())
idx_sub = np.array(idx_sub)
cv_scores = cross_val_score(model, X_scaled[idx_sub], y[idx_sub],
                             cv=cv, scoring="accuracy", n_jobs=-1)
print(f"      Per fold: {[f'{s*100:.1f}%' for s in cv_scores]}")
print(f"      Mean    : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

# Save metadata
metadata = {
    "version"         : "5.1-fast",
    "model_type"      : model_name,
    "accuracy"        : round(acc * 100, 2),
    "cv_mean"         : round(cv_scores.mean() * 100, 2),
    "cv_std"          : round(cv_scores.std()  * 100, 2),
    "auc_roc_macro"   : round(auc, 4) if auc else None,
    "classes"         : CLASS_NAMES,
    "n_features_raw"  : len(FEATURE_COLS),
    "n_features_eng"  : len(ENGINEERED_COLS),
    "engineered_features": ENGINEERED_COLS[len(FEATURE_COLS):],
    "training_samples": len(X_train),
    "test_samples"    : len(X_test),
    "feature_cols"    : FEATURE_COLS,
    "dataset_version" : "v5.1-fast (20004 samples, 7 sub-types, 20% comorbidity)",
    "base_models"     : [model_name],
    "notes": (
        f"v5.1-fast: single {model_name}, 20K samples, 12 engineered features. "
        "~5-10x faster than v5.0 6-model ensemble. Accuracy comparable on synthetic data."
    )
}

with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

total_time = time.time() - t_start
print("\n" + "="*60)
print("  FAST TRAINING COMPLETE — v5.1")
print(f"  Model       : {model_name}")
print(f"  Accuracy    : {acc*100:.2f}%")
print(f"  CV 3-fold   : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")
if auc: print(f"  AUC-ROC     : {auc:.4f}")
print(f"  Samples     : {len(X_train):,} train / {len(X_test):,} test")
print(f"  Features    : {len(FEATURE_COLS)} raw + 12 eng = {len(ENGINEERED_COLS)}")
print(f"  Total time  : {total_time:.0f}s  (v5.0 was ~5-10 min)")
print("  Saved       : backend/models/model.pkl")
print("="*60)
