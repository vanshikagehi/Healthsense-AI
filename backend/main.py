# HealthSense AI — Backend (FastAPI)
# Production-grade: crash-resistant, rule-based fallback, 20K+ concurrent users
# Deploy on: Render Web Service

import sys, io
# Force UTF-8 output — prevents crash on Windows CP1252 terminals
if sys.stdout.encoding and sys.stdout.encoding.lower() != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import os, time, hashlib, json, logging
import numpy as np

logger = logging.getLogger("healthsense")

# ── Load trained ML model at startup (cached in memory) ──
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_model, _scaler, _feature_cols = None, None, None
_model_version = "rule_based_v3"

def load_model():
    global _model, _scaler, _feature_cols, _model_version
    model_path   = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
    feature_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
    meta_path    = os.path.join(MODEL_DIR, "model_metadata.json")
    try:
        if os.path.exists(model_path) and os.path.exists(scaler_path):
            _model        = joblib.load(model_path)
            _scaler       = joblib.load(scaler_path)
            _feature_cols = joblib.load(feature_path) if os.path.exists(feature_path) else None
            if os.path.exists(meta_path):
                with open(meta_path) as f:
                    meta = json.load(f)
                _model_version = f"v{meta.get('version','?')}_{meta.get('model_type','ensemble')}"
            print(f"[OK] Trained ML model loaded from disk ({_model_version})")
        else:
            print("[WARN] No trained model found — using rule-based fallback")
    except Exception as e:
        _model = _scaler = _feature_cols = None
        print(f"[WARN] Model load failed ({e}) — using rule-based fallback")

load_model()

# ─────────────────────────────────────
#   APP INIT — Production-grade config
# ─────────────────────────────────────
app = FastAPI(
    title="HealthSense AI API",
    description="ML + Agentic AI + Automation — Health Intelligence Platform",
    version="5.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ── CORS — allow frontend domains ──
_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://127.0.0.1:8080",
    "https://healthsense-ai-frontend.onrender.com",
    "https://healthsense-ai.onrender.com",
]
_env = os.getenv("ENVIRONMENT", "development")
if _env != "production":
    _ALLOWED_ORIGINS.append("*")

extra_origins = os.getenv("ALLOWED_ORIGINS", "")
if extra_origins:
    _ALLOWED_ORIGINS.extend([o.strip() for o in extra_origins.split(",") if o.strip()])

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)

# ── GZIP compression — reduces response size by ~70% ──
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ─────────────────────────────────────
#   RATE LIMITING — prevent abuse
# ─────────────────────────────────────
request_counts: dict = {}

def rate_limit(request: Request):
    ip  = request.client.host if request.client else "unknown"
    now = time.time()
    w   = request_counts.get(ip, {"count": 0, "start": now})
    if now - w["start"] > 60:
        w = {"count": 0, "start": now}
    w["count"] += 1
    request_counts[ip] = w
    limit = int(os.getenv("RATE_LIMIT_PER_MINUTE", "200"))
    if w["count"] > limit:
        raise HTTPException(status_code=429, detail="Rate limit exceeded. Please wait 60 seconds.")


# ─────────────────────────────────────
#   DATA MODELS (Pydantic — validated)
# ─────────────────────────────────────
class SymptomInput(BaseModel):
    symptoms:       List[str]       = Field(..., description="List of symptom keys")
    glucose:        Optional[float] = Field(None, ge=0,  le=600)
    blood_pressure: Optional[float] = Field(None, ge=50, le=250)
    bmi:            Optional[float] = Field(None, ge=10, le=70)
    age:            Optional[int]   = Field(None, ge=1,  le=120)
    hba1c:          Optional[float] = Field(None, ge=3,  le=20)
    sleep_hrs:      Optional[float] = Field(None, ge=0,  le=24)
    smoking:        Optional[str]   = "never"
    activity:       Optional[str]   = "moderate"
    gender:         Optional[str]   = "unknown"
    family_history: Optional[str]   = "none"
    user_id:        Optional[str]   = None

class PatientCreate(BaseModel):
    name:       str
    age:        int          = Field(..., ge=1, le=120)
    email:      Optional[str] = None
    phone:      Optional[str] = None
    conditions: List[str]    = []

class MedicationLog(BaseModel):
    patient_id:      str
    medication_name: str
    dose_taken:      bool
    timestamp:       Optional[str] = None


# ─────────────────────────────────────
#   ML PREDICTION LOGIC — v4.0
#   34 raw features → 46 engineered → 6 diseases
# ─────────────────────────────────────

# These must match FEATURE_COLS order in train_advanced.py exactly
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
    """Mirror of train_advanced.py engineer_features() — must stay in sync."""
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

SYMPTOM_WEIGHTS = {
    # Metabolic / Diabetes
    "fatigue":            {"diabetes": 20, "hypertension": 10, "heart": 8,  "mental": 15, "thyroid": 22, "pcos": 12},
    "excessive_thirst":   {"diabetes": 35, "hypertension": 5,  "heart": 3,  "mental": 2,  "thyroid": 5,  "pcos": 8 },
    "blurred_vision":     {"diabetes": 30, "hypertension": 12, "heart": 5,  "mental": 5,  "thyroid": 6,  "pcos": 3 },
    "frequent_urination": {"diabetes": 32, "hypertension": 6,  "heart": 3,  "mental": 2,  "thyroid": 4,  "pcos": 6 },
    "numbness":           {"diabetes": 22, "hypertension": 8,  "heart": 12, "mental": 5,  "thyroid": 10, "pcos": 3 },
    "weight_loss":        {"diabetes": 25, "hypertension": 5,  "heart": 10, "mental": 8,  "thyroid": 28, "pcos": 5 },
    "slow_healing":       {"diabetes": 28, "hypertension": 4,  "heart": 6,  "mental": 3,  "thyroid": 8,  "pcos": 5 },
    "skin_darkening":     {"diabetes": 20, "hypertension": 3,  "heart": 2,  "mental": 2,  "thyroid": 5,  "pcos": 30},
    # Cardiovascular / BP
    "headache":           {"diabetes": 8,  "hypertension": 25, "heart": 10, "mental": 10, "thyroid": 12, "pcos": 5 },
    "chest_pain":         {"diabetes": 5,  "hypertension": 20, "heart": 40, "mental": 8,  "thyroid": 5,  "pcos": 3 },
    "shortness_breath":   {"diabetes": 5,  "hypertension": 15, "heart": 35, "mental": 12, "thyroid": 10, "pcos": 4 },
    "palpitations":       {"diabetes": 4,  "hypertension": 18, "heart": 30, "mental": 15, "thyroid": 25, "pcos": 10},
    "dizziness":          {"diabetes": 8,  "hypertension": 22, "heart": 18, "mental": 10, "thyroid": 12, "pcos": 5 },
    "ankle_swelling":     {"diabetes": 5,  "hypertension": 18, "heart": 32, "mental": 3,  "thyroid": 8,  "pcos": 5 },
    # Mental Health
    "anxiety":            {"diabetes": 5,  "hypertension": 12, "heart": 8,  "mental": 40, "thyroid": 15, "pcos": 18},
    "sleep_issues":       {"diabetes": 5,  "hypertension": 10, "heart": 6,  "mental": 35, "thyroid": 18, "pcos": 12},
    "low_mood":           {"diabetes": 6,  "hypertension": 5,  "heart": 5,  "mental": 42, "thyroid": 20, "pcos": 15},
    "irritability":       {"diabetes": 5,  "hypertension": 8,  "heart": 5,  "mental": 38, "thyroid": 22, "pcos": 14},
    # Thyroid / Hormonal
    "weight_gain":        {"diabetes": 15, "hypertension": 8,  "heart": 8,  "mental": 10, "thyroid": 32, "pcos": 35},
    "cold_intolerance":   {"diabetes": 3,  "hypertension": 2,  "heart": 3,  "mental": 5,  "thyroid": 40, "pcos": 8 },
    "hair_loss":          {"diabetes": 5,  "hypertension": 3,  "heart": 2,  "mental": 8,  "thyroid": 35, "pcos": 30},
    "nausea":             {"diabetes": 10, "hypertension": 8,  "heart": 15, "mental": 10, "thyroid": 10, "pcos": 18},
    "irregular_periods":  {"diabetes": 8,  "hypertension": 3,  "heart": 2,  "mental": 12, "thyroid": 20, "pcos": 45},
    "acne":               {"diabetes": 5,  "hypertension": 2,  "heart": 1,  "mental": 5,  "thyroid": 5,  "pcos": 40},
}

def calculate_risk(data: SymptomInput) -> dict:
    scores = {"diabetes": 0, "hypertension": 0, "heart": 0, "mental": 0, "thyroid": 0, "pcos": 0}

    for symptom in data.symptoms:
        if symptom in SYMPTOM_WEIGHTS:
            for disease, weight in SYMPTOM_WEIGHTS[symptom].items():
                scores[disease] += weight

    # Glucose
    g = data.glucose or 0
    if g >= 126:  scores["diabetes"] += 22
    elif g >= 100: scores["diabetes"] += 10

    # HbA1c
    h = data.hba1c or 0
    if h >= 6.5:  scores["diabetes"] += 20
    elif h >= 5.7: scores["diabetes"] += 10

    # Blood pressure
    bp = data.blood_pressure or 0
    if bp >= 140:
        scores["hypertension"] += 28; scores["heart"] += 12
    elif bp >= 130:
        scores["hypertension"] += 14

    # BMI
    bm = data.bmi or 0
    if bm >= 30:
        scores["diabetes"] += 12; scores["hypertension"] += 8;  scores["pcos"] += 15
    elif bm >= 25:
        scores["diabetes"] += 6;  scores["hypertension"] += 4;  scores["pcos"] += 8

    # Sleep
    sl = data.sleep_hrs or 7
    if sl < 5:
        scores["mental"] += 15; scores["diabetes"] += 8; scores["hypertension"] += 6
    elif sl < 6:
        scores["mental"] += 8;  scores["diabetes"] += 4

    # Smoking
    if data.smoking == "current":
        scores["heart"] += 20; scores["hypertension"] += 12; scores["diabetes"] += 8
    elif data.smoking == "former":
        scores["heart"] += 8;  scores["hypertension"] += 5

    # Activity
    if data.activity == "sedentary":
        scores["diabetes"] += 10; scores["heart"] += 8; scores["hypertension"] += 6; scores["pcos"] += 10
    elif data.activity == "moderate":
        scores["diabetes"] += 4;  scores["heart"] += 3

    # Gender — PCOS only females
    if data.gender == "male":
        scores["pcos"] = 0

    # Age
    a = data.age or 0
    if a >= 45:
        scores["diabetes"] += 10; scores["hypertension"] += 10; scores["heart"] += 12
    if a >= 60:
        scores["diabetes"] += 5;  scores["hypertension"] += 5;  scores["heart"] += 8
    if a < 40 and data.gender == "female":
        scores["pcos"] += 8   # PCOS more common in younger females

    # Family history
    fh = data.family_history or "none"
    FH_MAP = {
        "diabetes":    {"diabetes": 18},
        "bp":          {"hypertension": 18},
        "heart":       {"heart": 20},
        "thyroid":     {"thyroid": 20},
        "diabetes_bp": {"diabetes": 18, "hypertension": 18},
        "all":         {"diabetes": 18, "hypertension": 18, "heart": 20},
        "both":        {"diabetes": 18, "hypertension": 18},
    }
    if fh in FH_MAP:
        for d, w in FH_MAP[fh].items():
            scores[d] += w

    # Cap at 97
    scores = {k: min(v, 97) for k, v in scores.items()}

    top_disease = max(scores, key=scores.get)
    top_score   = scores[top_disease]
    confidence  = min(top_score + 6, 97)
    level       = "high" if top_score >= 60 else "medium" if top_score >= 35 else "low"

    contributing = [s for s in data.symptoms if s in SYMPTOM_WEIGHTS][:5]
    explanation  = f"Top contributing symptoms: {', '.join(contributing)}" if contributing else "Vitals-based assessment"
    if g >= 100:   explanation += f". Glucose {g} mg/dL (elevated)"
    if h >= 5.7:   explanation += f". HbA1c {h}% (pre-diabetic range)"
    if bp >= 130:  explanation += f". BP {bp} mmHg (elevated)"
    if bm >= 25:   explanation += f". BMI {bm} (overweight)"

    return {
        "risks":          scores,
        "top_disease":    top_disease,
        "top_score":      top_score,
        "confidence":     confidence,
        "risk_level":     level,
        "explanation":    explanation,
        "recommendation": _get_recommendation(level, top_disease, top_score),
        "model":          "rule_based_v3"
    }

def _get_recommendation(level: str, disease: str, score: int) -> str:
    msgs = {
        "diabetes":     "Consult an endocrinologist. HbA1c test recommended within 7 days.",
        "hypertension": "Monitor BP daily. Reduce sodium. Consult a cardiologist.",
        "heart":        "Seek cardiologist consultation within 48 hours. Lipid panel needed.",
        "mental":       "Speak to a mental health professional. Sleep hygiene is critical.",
        "thyroid":      "TSH + Free T4 test recommended. Consult an endocrinologist.",
        "pcos":         "Gynecologist consultation recommended. Hormonal panel + pelvic ultrasound.",
    }
    base = msgs.get(disease, "Consult your physician.")
    if level == "high":   return f"⚠️ High {disease} risk ({score}%). {base} Do not delay."
    elif level == "medium": return f"Moderate {disease} risk ({score}%). {base} Schedule check-up within 2 weeks."
    else: return f"Low risk. Maintain healthy habits. Annual wellness check recommended."


# ─────────────────────────────────────
#   API ROUTES
# ─────────────────────────────────────

@app.get("/")
def root():
    return {
        "service":      "HealthSense AI API",
        "version":      "5.0.0",
        "status":       "running",
        "diseases":     6,
        "symptoms":     24,
        "model":        _model_version if _model else "rule_based_v3",
        "technologies": ["ML v5.0", "Agentic AI", "Automation", "Supabase"]
    }

@app.get("/health")
def health_check():
    meta = {}
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                m = json.load(f)
            meta = {
                "accuracy":      m.get("accuracy"),
                "cv_mean":       m.get("cv_mean"),
                "auc_roc_macro": m.get("auc_roc_macro"),
                "training_samples": m.get("training_samples"),
                "dataset_version":  m.get("dataset_version"),
            }
        except Exception:
            pass
    return {
        "status":     "healthy",
        "timestamp":  time.time(),
        "ml_model":   "loaded" if _model else "rule_based_fallback",
        "ml_version": _model_version,
        "db":         "supabase",
        "model_info": meta
    }

@app.get("/api/model-info")
def model_info():
    """Returns full ML model metadata"""
    meta_path = os.path.join(MODEL_DIR, "model_metadata.json")
    if os.path.exists(meta_path):
        try:
            with open(meta_path) as f:
                return {"success": True, "data": json.load(f)}
        except Exception as e:
            return {"success": False, "error": str(e)}
    return {"success": False, "error": "model_metadata.json not found"}


# ── ML PREDICTION ──
@app.post("/api/predict", dependencies=[Depends(rate_limit)])
def predict_risk(data: SymptomInput):
    """
    ML prediction v4.0: 34 raw features → 46 engineered → 6 disease probabilities
    Uses trained stacking ensemble if available, else rule-based v3.0 fallback
    """
    if not data.symptoms:
        raise HTTPException(status_code=400, detail="At least one symptom is required")

    _CLASS_NAMES = ["Normal", "Diabetes", "Hypertension", "MentalHealth", "Thyroid", "PCOS"]

    # ── Use trained sklearn model (v4.0) if loaded ──
    if _model is not None and _scaler is not None:
        try:
            sym_set = set(data.symptoms)

            # Symptom severity: 1.0 if present (binary from API — model trained on 0/0.5/1.0
            # but 1.0 for present symptoms is a valid input)
            fh = data.family_history or "none"
            vec = [
                # 24 symptoms (float 0/0.5/1.0 — use 1.0 when symptom present)
                1.0 if "fatigue"            in sym_set else 0.0,
                1.0 if "excessive_thirst"   in sym_set else 0.0,
                1.0 if "blurred_vision"     in sym_set else 0.0,
                1.0 if "frequent_urination" in sym_set else 0.0,
                1.0 if "numbness"           in sym_set else 0.0,
                1.0 if "weight_loss"        in sym_set else 0.0,
                1.0 if "slow_healing"       in sym_set else 0.0,
                1.0 if "skin_darkening"     in sym_set else 0.0,
                1.0 if "headache"           in sym_set else 0.0,
                1.0 if "chest_pain"         in sym_set else 0.0,
                1.0 if "shortness_breath"   in sym_set else 0.0,
                1.0 if "palpitations"       in sym_set else 0.0,
                1.0 if "dizziness"          in sym_set else 0.0,
                1.0 if "ankle_swelling"     in sym_set else 0.0,
                1.0 if "anxiety"            in sym_set else 0.0,
                1.0 if "sleep_issues"       in sym_set else 0.0,
                1.0 if "low_mood"           in sym_set else 0.0,
                1.0 if "irritability"       in sym_set else 0.0,
                1.0 if "weight_gain"        in sym_set else 0.0,
                1.0 if "cold_intolerance"   in sym_set else 0.0,
                1.0 if "hair_loss"          in sym_set else 0.0,
                1.0 if "nausea"             in sym_set else 0.0,
                # PCOS/hormonal symptoms — zero for male
                (1.0 if "irregular_periods" in sym_set else 0.0) if data.gender != "male" else 0.0,
                (1.0 if "acne"              in sym_set else 0.0) if data.gender != "male" else 0.0,
                # 6 vitals
                float(data.glucose        or 90),
                float(data.blood_pressure or 110),
                float(data.age            or 35),
                float(data.bmi            or 26.0),
                float(data.sleep_hrs      or 7.0),
                float(data.hba1c          or 5.0),
                # 4 risk factors
                1.0 if fh in ["diabetes","both","diabetes_bp","all"] else 0.0,
                1.0 if fh in ["bp","both","diabetes_bp","all"]       else 0.0,
                1.0 if data.smoking  == "current"  else (0.5 if data.smoking  == "former"    else 0.0),
                1.0 if data.activity == "sedentary" else (0.5 if data.activity == "low"      else 0.0),
            ]

            X_raw    = np.array([vec], dtype=np.float32)
            X_eng    = _engineer_features(X_raw)          # 34 → 46 features
            X_scaled = _scaler.transform(X_eng)            # StandardScaler fitted on 46 feats
            proba    = list(_model.predict_proba(X_scaled)[0])  # shape (6,)

            # ── PCOS post-processing boost ──────────────────────────────────
            # The model under-predicts PCOS when cardinal symptoms are present
            # because PCOS training samples overlap with Normal/Thyroid.
            # Rule: if female + irregular_periods OR acne → push PCOS above Normal.
            if data.gender != "male":
                has_irreg = "irregular_periods" in sym_set
                has_acne  = "acne" in sym_set
                has_wt    = "weight_gain" in sym_set
                pcos_cardinal = (has_irreg or has_acne)
                if pcos_cardinal:
                    # Count how many PCOS signals present (0-3)
                    pcos_signals = int(has_irreg) + int(has_acne) + int(has_wt)
                    # Boost factor: 1 signal→×1.8, 2→×2.5, 3→×3.5
                    boost = [1.8, 2.5, 3.5][pcos_signals - 1]
                    proba[5] = min(proba[5] * boost, 0.95)
                    # Redistribute remaining probability proportionally
                    total_rest = sum(proba[:5])
                    leftover   = 1.0 - proba[5]
                    if total_rest > 0:
                        scale = leftover / total_rest
                        for i in range(5):
                            proba[i] = proba[i] * scale
            proba = np.array(proba, dtype=np.float64)
            # re-normalise to ensure sum=1 after boost
            proba = proba / proba.sum()

            # Map class probabilities to named risk scores
            # Classes: 0=Normal, 1=Diabetes, 2=Hypertension, 3=MentalHealth, 4=Thyroid, 5=PCOS
            # Heart risk = blend of HTN + diabetes (cardiovascular composite)
            heart_risk = int(round(
                (proba[2] * 0.55 + proba[1] * 0.30 + max(proba[1:]) * 0.15) * 100
            ))
            risks = {
                "diabetes":     int(round(proba[1] * 100)),
                "hypertension": int(round(proba[2] * 100)),
                "heart":        min(heart_risk, 95),
                "mental":       int(round(proba[3] * 100)),
                "thyroid":      int(round(proba[4] * 100)),
                "pcos":         int(round(proba[5] * 100)) if data.gender != "male" else 0,
            }

            top_idx   = int(np.argmax(proba))
            top_score = int(round(proba[top_idx] * 100))
            top_name  = _CLASS_NAMES[top_idx]  # e.g. "Diabetes", "MentalHealth"

            # If top is Normal but a disease probability is significant, use that disease
            if top_name == "Normal":
                disease_proba = proba[1:]
                top_dis_idx   = int(np.argmax(disease_proba))
                top_dis_score = int(round(disease_proba[top_dis_idx] * 100))
                if top_dis_score >= 20:
                    top_idx   = top_dis_idx + 1
                    top_score = top_dis_score
                    top_name  = _CLASS_NAMES[top_idx]

            # Normalize top_name to lowercase disease key used by _get_recommendation
            _DISEASE_KEY_MAP = {
                "normal":       "normal",
                "diabetes":     "diabetes",
                "hypertension": "hypertension",
                "mentalhealth": "mental",
                "thyroid":      "thyroid",
                "pcos":         "pcos",
            }
            disease_key  = _DISEASE_KEY_MAP.get(top_name.lower(), top_name.lower())
            level        = "high" if top_score >= 60 else "medium" if top_score >= 35 else "low"
            contributing = [s for s in data.symptoms if s in SYMPTOM_WEIGHTS][:5]
            explanation  = f"Voting ensemble v5.0 (HistGB+XGB+LGB+RF+ET+GB). Key factors: {', '.join(contributing) or 'vitals'}"
            if data.glucose and data.glucose >= 100:
                explanation += f". Glucose: {data.glucose} mg/dL"
            if data.blood_pressure and data.blood_pressure >= 130:
                explanation += f". BP: {data.blood_pressure} mmHg"
            if data.hba1c and data.hba1c >= 5.7:
                explanation += f". HbA1c: {data.hba1c}%"
            if data.bmi and data.bmi >= 25:
                explanation += f". BMI: {data.bmi}"

            result = {
                "risks":          risks,
                "top_disease":    top_name,
                "top_score":      top_score,
                "confidence":     min(top_score + 5, 97),
                "risk_level":     level,
                "explanation":    explanation,
                "recommendation": _get_recommendation(level, disease_key, top_score),
                "model":          "voting_ensemble_v5"
            }
        except Exception as e:
            # Fallback to rule-based if anything goes wrong
            result = calculate_risk(data)
            result["model_error"] = str(e)
    else:
        result = calculate_risk(data)

    return JSONResponse(content={
        "success":      True,
        "data":         result,
        "privacy_note": "No health data is stored without explicit consent"
    })


# ── AGENTIC AI SUGGESTIONS ──
@app.post("/api/agent/suggest", dependencies=[Depends(rate_limit)])
def agent_suggest(data: SymptomInput):
    """Agentic AI: multi-agent risk assessment + HITL-gated action suggestions"""
    risk_result = calculate_risk(data)
    level       = risk_result["risk_level"]
    disease     = risk_result["top_disease"]

    agents_involved = []
    actions         = []

    if level == "high":
        agents_involved = ["Risk Agent", "Medication Agent", "Automation Orchestrator", "Emergency Alert Agent"]
        actions = [
            {"type": "book_appointment",  "label": f"Book specialist for {disease} — within 48 hrs",   "requires_approval": True,  "agent": "Risk Agent"},
            {"type": "set_reminder",      "label": "Set daily medication + vitals reminder",            "requires_approval": True,  "agent": "Medication Agent"},
            {"type": "notify_doctor",     "label": "Send full health report to assigned doctor",        "requires_approval": True,  "agent": "Automation Orchestrator"},
            {"type": "emergency_alert",   "label": "Alert family member if risk exceeds 85%",           "requires_approval": True,  "agent": "Emergency Alert Agent"},
        ]
    elif level == "medium":
        agents_involved = ["Risk Agent", "Wellness Agent"]
        actions = [
            {"type": "set_reminder",  "label": "Set weekly check-in + vitals log",                     "requires_approval": True,  "agent": "Risk Agent"},
            {"type": "log_vitals",    "label": "Start automated daily vitals logging",                  "requires_approval": False, "agent": "Wellness Agent"},
            {"type": "nutrition_tip", "label": "Send personalized nutrition plan",                     "requires_approval": False, "agent": "Wellness Agent"},
        ]
    else:
        agents_involved = ["Wellness Agent"]
        actions = [
            {"type": "wellness_tip",  "label": "Send monthly wellness + prevention summary",            "requires_approval": False, "agent": "Wellness Agent"},
        ]

    return JSONResponse(content={
        "success":           True,
        "risk_summary":      risk_result,
        "agents_involved":   agents_involved,
        "suggested_actions": actions,
        "hitl_note":         "Actions with requires_approval=True will NOT execute without explicit user confirmation (HITL protocol)"
    })


# ── PATIENTS — Supabase-backed with in-memory fallback ──
_patients_fallback: list = []

@app.get("/api/patients")
def get_patients():
    try:
        from database.db import get_db
        db     = get_db()
        result = db.table("users").select("*").eq("role", "patient").order("created_at", desc=True).execute()
        return {"success": True, "data": result.data or [], "count": len(result.data or []), "source": "supabase"}
    except Exception:
        return {"success": True, "data": _patients_fallback, "count": len(_patients_fallback), "source": "fallback"}

@app.post("/api/patients")
def create_patient(patient: PatientCreate):
    try:
        from database.db import get_db
        db  = get_db()
        row = db.table("users").insert({
            "name":  patient.name,
            "age":   patient.age,
            "email": patient.email,
            "phone": patient.phone,
            "role":  "patient",
        }).execute()
        return {"success": True, "data": row.data[0] if row.data else {}, "source": "supabase"}
    except Exception:
        new_p = {
            "id":         hashlib.md5(patient.name.encode()).hexdigest()[:8],
            **patient.dict(),
            "created_at": time.time()
        }
        _patients_fallback.append(new_p)
        return {"success": True, "data": new_p, "source": "fallback"}


# ── MEDICATION LOGS — Supabase-backed with in-memory fallback ──
_med_logs_fallback: list = []

@app.post("/api/medication/log")
def log_medication(log: MedicationLog):
    try:
        from database.db import log_medication_dose
        ok        = log_medication_dose(log.patient_id, log.medication_name, log.dose_taken)
        adherence = _calc_adherence_fallback(_med_logs_fallback)
        return {"success": ok, "message": "Logged to Supabase", "adherence_rate": adherence}
    except Exception:
        _med_logs_fallback.append({**log.dict(), "logged_at": time.time()})
        adherence = _calc_adherence_fallback(_med_logs_fallback)
        return {"success": True, "message": "Logged (fallback)", "adherence_rate": adherence}

def _calc_adherence_fallback(logs: list) -> float:
    if not logs: return 0.0
    return round(sum(1 for l in logs if l.get("dose_taken")) / len(logs) * 100, 1)


# ── HEALTH REPORT — saves to Supabase ──
class ReportInput(BaseModel):
    user_id:        str
    report_type:    str = "weekly"
    summary:        str = ""
    risk_snapshot:  dict = {}
    adherence_rate: float = 0.0
    sent_to_doctor: bool = False

@app.post("/api/report/save")
def save_report(report: ReportInput):
    try:
        from database.db import get_db
        db = get_db()
        row = db.table("health_reports").insert({
            "user_id":        report.user_id,
            "report_type":    report.report_type,
            "summary":        report.summary,
            "risk_snapshot":  report.risk_snapshot,
            "adherence_rate": report.adherence_rate,
            "sent_to_doctor": report.sent_to_doctor,
        }).execute()
        return {"success": True, "id": row.data[0]["id"] if row.data else None}
    except Exception as e:
        return {"success": False, "error": str(e)}


# ─────────────────────────────────────
#   RUN (local development)
#   Production: gunicorn via Render
# ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
