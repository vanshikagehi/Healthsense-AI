"""Pre-deploy system check — run before git push"""
import sys, os, json
sys.path.insert(0, 'backend')

errors = []

# ── CHECK 1: model files exist ──
print("[CHECK 1] Model files exist...")
required = [
    "backend/models/model.pkl",
    "backend/models/scaler.pkl",
    "backend/models/feature_cols.pkl",
    "backend/models/engineered_cols.pkl",
    "backend/models/model_metadata.json",
]
for f in required:
    if os.path.exists(f):
        size = os.path.getsize(f)
        print(f"  OK  {f} ({size//1024} KB)")
    else:
        print(f"  MISSING: {f}")
        errors.append(f"Missing: {f}")

# ── CHECK 2: model loads correctly ──
print("\n[CHECK 2] Model + scaler load...")
import joblib, numpy as np
try:
    model  = joblib.load("backend/models/model.pkl")
    scaler = joblib.load("backend/models/scaler.pkl")
    feat   = joblib.load("backend/models/feature_cols.pkl")
    meta   = json.load(open("backend/models/model_metadata.json"))
    print(f"  OK  model_type={meta['model_type']}")
    print(f"  OK  accuracy={meta['accuracy']}%  auc={meta.get('auc_roc_macro','?')}")
    print(f"  OK  features={len(feat)} raw  version={meta['version']}")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append(str(e))
    sys.exit(1)   # Can't continue without model

# ── feature list + engineer_features inlined (no fastapi needed) ──
_FEATURE_COLS_ORDER = [
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
_IDX = {c: i for i, c in enumerate(_FEATURE_COLS_ORDER)}

def _engineer_features(X):
    gluc  = (X[:, _IDX["glucose"]]         - 70)  / 250
    bp    = (X[:, _IDX["blood_pressure"]]  - 80)  / 130
    age_n = (X[:, _IDX["age"]]             - 14)  / 66
    bmi_n = (X[:, _IDX["bmi"]]             - 16)  / 32
    slp   = (X[:, _IDX["sleep_hrs"]]       - 2)   / 8
    hba   = (X[:, _IDX["hba1c"]]           - 4.5) / 8.5
    metabolic_score  = (X[:, _IDX["fatigue"]] + X[:, _IDX["excessive_thirst"]] +
                        X[:, _IDX["frequent_urination"]] + X[:, _IDX["numbness"]] +
                        X[:, _IDX["slow_healing"]] + gluc + hba) / 7
    cardio_score     = (X[:, _IDX["headache"]] + X[:, _IDX["chest_pain"]] +
                        X[:, _IDX["shortness_breath"]] + X[:, _IDX["palpitations"]] +
                        X[:, _IDX["dizziness"]] + bp + age_n) / 7
    mental_score     = (X[:, _IDX["anxiety"]] + X[:, _IDX["sleep_issues"]] +
                        X[:, _IDX["low_mood"]] + X[:, _IDX["irritability"]]) / 4
    hormonal_score   = (X[:, _IDX["weight_gain"]] + X[:, _IDX["cold_intolerance"]] +
                        X[:, _IDX["hair_loss"]] + X[:, _IDX["irregular_periods"]] +
                        X[:, _IDX["acne"]]) / 5
    glucose_bmi      = np.clip(gluc * bmi_n, 0, 1)
    bp_age           = np.clip(bp   * age_n, 0, 1)
    sleep_anxiety    = np.clip((1 - slp) * X[:, _IDX["anxiety"]], 0, 1)
    hba1c_glucose    = np.clip(hba * gluc, 0, 1)
    symptom_total    = np.mean(X[:, :24], axis=1)
    cardinal_diabetes = (X[:, _IDX["excessive_thirst"]] +
                         X[:, _IDX["frequent_urination"]] +
                         X[:, _IDX["blurred_vision"]]) / 3
    cardinal_thyroid  = (X[:, _IDX["cold_intolerance"]] +
                         X[:, _IDX["hair_loss"]] +
                         X[:, _IDX["weight_gain"]]) / 3
    cardinal_pcos     = (X[:, _IDX["irregular_periods"]] +
                         X[:, _IDX["acne"]] +
                         X[:, _IDX["weight_gain"]]) / 3
    new_feats = np.column_stack([
        metabolic_score, cardio_score, mental_score, hormonal_score,
        glucose_bmi, bp_age, sleep_anxiety, hba1c_glucose,
        symptom_total, cardinal_diabetes, cardinal_thyroid, cardinal_pcos
    ])
    return np.hstack([X, new_feats])

# ── CHECK 3: feature engineering pipeline ──
print("\n[CHECK 3] Feature engineering pipeline...")
try:
    assert len(_FEATURE_COLS_ORDER) == 34, f"Expected 34 raw cols, got {len(_FEATURE_COLS_ORDER)}"
    dummy = np.zeros((1, 34), dtype=np.float32)
    dummy[0, _IDX["glucose"]]        = 90.0
    dummy[0, _IDX["blood_pressure"]] = 110.0
    dummy[0, _IDX["age"]]            = 35.0
    dummy[0, _IDX["bmi"]]            = 26.0
    dummy[0, _IDX["sleep_hrs"]]      = 7.0
    dummy[0, _IDX["hba1c"]]          = 5.0
    X_eng = _engineer_features(dummy)
    assert X_eng.shape == (1, 46), f"Expected (1,46), got {X_eng.shape}"
    print(f"  OK  34 raw -> 46 engineered features")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append(str(e))

# ── CHECK 4: scaler shape matches engineered features ──
print("\n[CHECK 4] Scaler n_features check...")
try:
    n_sc = scaler.n_features_in_
    assert n_sc == 46, f"Scaler expects {n_sc} features, pipeline produces 46"
    print(f"  OK  scaler.n_features_in_ = {n_sc}")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append(str(e))

# ── CHECK 5: end-to-end prediction (diabetes test case) ──
print("\n[CHECK 5] End-to-end prediction — Diabetes test case...")
try:
    vec = [0.0] * 34
    vec[_IDX["fatigue"]]            = 1.0
    vec[_IDX["excessive_thirst"]]   = 1.0
    vec[_IDX["frequent_urination"]] = 1.0
    vec[_IDX["glucose"]]            = 185.0
    vec[_IDX["blood_pressure"]]     = 128.0
    vec[_IDX["age"]]                = 48.0
    vec[_IDX["bmi"]]                = 31.0
    vec[_IDX["sleep_hrs"]]          = 7.0
    vec[_IDX["hba1c"]]              = 8.2

    X_raw = np.array([vec], dtype=np.float32)
    X_sc  = scaler.transform(_engineer_features(X_raw))
    proba = model.predict_proba(X_sc)[0]
    NAMES = ["Normal","Diabetes","Hypertension","MentalHealth","Thyroid","PCOS"]
    top   = NAMES[int(np.argmax(proba))]
    conf  = round(max(proba) * 100, 1)
    print(f"  OK  Prediction: {top} ({conf}%)")
    print(f"  Proba: { {n: round(p*100,1) for n,p in zip(NAMES,proba)} }")
    if top != "Diabetes":
        print(f"  WARN: Expected Diabetes for this test case, got {top}")
        errors.append(f"Prediction sanity: expected Diabetes got {top}")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append(str(e))

# ── CHECK 6: end-to-end prediction (normal healthy case) ──
print("\n[CHECK 6] End-to-end prediction — Normal (healthy) test case...")
try:
    vec2 = [0.0] * 34
    vec2[_IDX["glucose"]]        = 88.0
    vec2[_IDX["blood_pressure"]] = 108.0
    vec2[_IDX["age"]]            = 30.0
    vec2[_IDX["bmi"]]            = 23.0
    vec2[_IDX["sleep_hrs"]]      = 8.0
    vec2[_IDX["hba1c"]]          = 5.1

    X2    = np.array([vec2], dtype=np.float32)
    X2_sc = scaler.transform(_engineer_features(X2))
    proba2 = model.predict_proba(X2_sc)[0]
    top2   = NAMES[int(np.argmax(proba2))]
    conf2  = round(max(proba2) * 100, 1)
    print(f"  OK  Prediction: {top2} ({conf2}%)")
    if top2 != "Normal":
        print(f"  WARN: Expected Normal for healthy case, got {top2}")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append(str(e))

# ── CHECK 7: metadata version match ──
print("\n[CHECK 7] Metadata version + key fields...")
try:
    assert "version" in meta
    assert "model_type" in meta
    assert "accuracy" in meta
    assert meta["accuracy"] >= 95.0, f"Accuracy {meta['accuracy']}% seems too low"
    assert meta["n_features_raw"] == 34
    assert meta["n_features_eng"] == 46
    print(f"  OK  version={meta['version']}  accuracy={meta['accuracy']}%")
    print(f"  OK  features: {meta['n_features_raw']} raw + 12 eng = {meta['n_features_eng']}")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append(str(e))

# ── CHECK 8: render.yaml uses train_fast.py ──
print("\n[CHECK 8] render.yaml deployment config...")
try:
    with open("render.yaml", encoding="utf-8") as f:
        render_content = f.read()
    assert "train_fast.py" in render_content, "render.yaml should call train_fast.py"
    assert "healthCheckPath" in render_content
    assert "gunicorn" in render_content
    print(f"  OK  render.yaml uses train_fast.py as build fallback")
    print(f"  OK  health check path configured")
    print(f"  OK  gunicorn start command present")
except Exception as e:
    print(f"  FAIL: {e}")
    errors.append(str(e))

# ── SUMMARY ──
print()
print("="*55)
if errors:
    print(f"  DEPLOY CHECK FAILED -- {len(errors)} error(s):")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print("  ALL 8 CHECKS PASSED -- safe to git push + deploy")
print("="*55)
