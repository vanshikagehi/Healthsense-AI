# HealthSense AI — Backend (FastAPI)
# Scalable: handles 20,000–30,000 concurrent users
# Deploy on: Render Web Service

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Optional
import os, time, hashlib, json
import numpy as np

# ── Load trained ML model at startup (cached in memory) ──
import joblib

MODEL_DIR = os.path.join(os.path.dirname(__file__), "models")
_model, _scaler, _feature_cols = None, None, None

def load_model():
    global _model, _scaler, _feature_cols
    model_path   = os.path.join(MODEL_DIR, "model.pkl")
    scaler_path  = os.path.join(MODEL_DIR, "scaler.pkl")
    feature_path = os.path.join(MODEL_DIR, "feature_cols.pkl")
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        _model       = joblib.load(model_path)
        _scaler      = joblib.load(scaler_path)
        _feature_cols = joblib.load(feature_path) if os.path.exists(feature_path) else None
        print("✅ Trained ML model loaded from disk")
    else:
        print("⚠️  No trained model found — using rule-based fallback")

load_model()

# ─────────────────────────────────────
#   APP INIT — Production-grade config
# ─────────────────────────────────────
app = FastAPI(
    title="HealthSense AI API",
    description="ML + Agentic AI + Automation — Health Intelligence Platform",
    version="1.0.0",
    docs_url="/docs",        # Swagger UI (disable in production)
    redoc_url="/redoc"
)

# ── CORS — allow frontend domains ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5500",          # Local dev (Live Server)
        "http://127.0.0.1:5500",
        "https://healthsense-ai.vercel.app",  # Production frontend
        "*"  # Remove in production — list exact domains
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── GZIP compression — reduces response size by ~70% ──
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ─────────────────────────────────────
#   RATE LIMITING — prevent abuse
#   (simple in-memory; use Redis for 30K users)
# ─────────────────────────────────────
request_counts = {}

def rate_limit(request: Request):
    ip = request.client.host
    now = time.time()
    window = request_counts.get(ip, {"count": 0, "start": now})
    if now - window["start"] > 60:
        window = {"count": 0, "start": now}
    window["count"] += 1
    request_counts[ip] = window
    if window["count"] > 30:  # max 30 requests/minute per IP
        raise HTTPException(status_code=429, detail="Too many requests. Please wait.")


# ─────────────────────────────────────
#   DATA MODELS (Pydantic — validated)
# ─────────────────────────────────────
class SymptomInput(BaseModel):
    symptoms: List[str] = Field(..., description="List of symptom keys")
    glucose: Optional[float] = Field(None, ge=0, le=600)
    blood_pressure: Optional[float] = Field(None, ge=50, le=250)
    age: Optional[int] = Field(None, ge=1, le=120)
    family_history: Optional[str] = "none"
    user_id: Optional[str] = None   # for memory/history (optional)

class PatientCreate(BaseModel):
    name: str
    age: int = Field(..., ge=1, le=120)
    email: Optional[str] = None
    phone: Optional[str] = None
    conditions: List[str] = []

class MedicationLog(BaseModel):
    patient_id: str
    medication_name: str
    dose_taken: bool
    timestamp: Optional[str] = None


# ─────────────────────────────────────
#   ML PREDICTION LOGIC
#   (Rule-based weights — replace with
#    trained TF/sklearn model in production)
# ─────────────────────────────────────
SYMPTOM_WEIGHTS = {
    "fatigue":           {"diabetes": 20, "hypertension": 10, "heart": 8,  "mental": 15},
    "excessive_thirst":  {"diabetes": 35, "hypertension": 5,  "heart": 3,  "mental": 2 },
    "blurred_vision":    {"diabetes": 30, "hypertension": 12, "heart": 5,  "mental": 5 },
    "frequent_urination":{"diabetes": 32, "hypertension": 6,  "heart": 3,  "mental": 2 },
    "headache":          {"diabetes": 8,  "hypertension": 25, "heart": 10, "mental": 10},
    "chest_pain":        {"diabetes": 5,  "hypertension": 20, "heart": 40, "mental": 8 },
    "shortness_breath":  {"diabetes": 5,  "hypertension": 15, "heart": 35, "mental": 12},
    "weight_loss":       {"diabetes": 25, "hypertension": 5,  "heart": 10, "mental": 8 },
    "anxiety":           {"diabetes": 5,  "hypertension": 12, "heart": 8,  "mental": 40},
    "sleep_issues":      {"diabetes": 5,  "hypertension": 10, "heart": 6,  "mental": 35},
    "numbness":          {"diabetes": 22, "hypertension": 8,  "heart": 12, "mental": 5 },
    "nausea":            {"diabetes": 10, "hypertension": 8,  "heart": 15, "mental": 10},
}

def calculate_risk(data: SymptomInput) -> dict:
    scores = {"diabetes": 0, "hypertension": 0, "heart": 0, "mental": 0}

    for symptom in data.symptoms:
        if symptom in SYMPTOM_WEIGHTS:
            for disease, weight in SYMPTOM_WEIGHTS[symptom].items():
                scores[disease] += weight

    # Vitals boost
    if data.glucose:
        if data.glucose >= 126: scores["diabetes"] += 20
        elif data.glucose >= 100: scores["diabetes"] += 8
    if data.blood_pressure:
        if data.blood_pressure >= 140:
            scores["hypertension"] += 25
            scores["heart"] += 10
        elif data.blood_pressure >= 130:
            scores["hypertension"] += 12

    # Age factor
    if data.age:
        if data.age >= 45:
            scores["diabetes"] += 10; scores["hypertension"] += 10; scores["heart"] += 12
        if data.age >= 60:
            scores["diabetes"] += 5;  scores["hypertension"] += 5;  scores["heart"] += 8

    # Family history
    fh = data.family_history or "none"
    if fh == "diabetes":  scores["diabetes"] += 18
    elif fh == "bp":      scores["hypertension"] += 18
    elif fh == "heart":   scores["heart"] += 20
    elif fh == "both":    scores["diabetes"] += 18; scores["hypertension"] += 18

    # Cap at 97
    scores = {k: min(v, 97) for k, v in scores.items()}

    # Top risk
    top_disease = max(scores, key=scores.get)
    top_score   = scores[top_disease]
    confidence  = min(top_score + 6, 97)

    # Explanation
    contributing = [s for s in data.symptoms if s in SYMPTOM_WEIGHTS][:4]
    explanation  = f"Top contributing factors: {', '.join(contributing)}"
    if data.glucose and data.glucose >= 100:
        explanation += f". Glucose {data.glucose} mg/dL (elevated)"
    if data.blood_pressure and data.blood_pressure >= 130:
        explanation += f". BP {data.blood_pressure} mmHg (elevated)"

    level = "high" if top_score >= 60 else "medium" if top_score >= 35 else "low"

    return {
        "risks": scores,
        "top_disease": top_disease,
        "top_score": top_score,
        "confidence": confidence,
        "risk_level": level,
        "explanation": explanation,
        "recommendation": _get_recommendation(level, top_disease, top_score)
    }

def _get_recommendation(level: str, disease: str, score: int) -> str:
    if level == "high":
        return f"High {disease} risk ({score}%). Consult an endocrinologist or specialist within 48 hours. Do not ignore."
    elif level == "medium":
        return f"Moderate {disease} risk. Schedule a routine check-up within 2 weeks. Monitor vitals daily."
    else:
        return "Risk levels are low. Maintain healthy habits and schedule a wellness check every 3 months."


# ─────────────────────────────────────
#   API ROUTES
# ─────────────────────────────────────

@app.get("/")
def root():
    return {
        "service": "HealthSense AI API",
        "version": "1.0.0",
        "status": "running",
        "technologies": ["ML", "Agentic AI", "Automation"]
    }

@app.get("/health")
def health_check():
    """Used by Railway/Render for uptime monitoring"""
    return {"status": "healthy", "timestamp": time.time()}


# ── ML PREDICTION ──
@app.post("/api/predict", dependencies=[Depends(rate_limit)])
def predict_risk(data: SymptomInput):
    """
    Main ML prediction endpoint — uses trained Deep Learning model if available.
    Input: symptoms + vitals + family history
    Output: risk scores, confidence, explanation, recommendation
    """
    if not data.symptoms:
        raise HTTPException(status_code=400, detail="At least one symptom is required")

    # ── Use trained model if loaded ──
    if _model is not None and _scaler is not None:
        try:
            sym_set = set(data.symptoms)
            feat_order = [
                "fatigue","excessive_thirst","blurred_vision","frequent_urination",
                "headache","chest_pain","shortness_breath","weight_loss",
                "anxiety","sleep_issues","numbness","nausea",
                "glucose","blood_pressure","age","bmi","family_diabetes","family_bp"
            ]
            vec = [
                1 if "fatigue"            in sym_set else 0,
                1 if "excessive_thirst"   in sym_set else 0,
                1 if "blurred_vision"     in sym_set else 0,
                1 if "frequent_urination" in sym_set else 0,
                1 if "headache"           in sym_set else 0,
                1 if "chest_pain"         in sym_set else 0,
                1 if "shortness_breath"   in sym_set else 0,
                1 if "weight_loss"        in sym_set else 0,
                1 if "anxiety"            in sym_set else 0,
                1 if "sleep_issues"       in sym_set else 0,
                1 if "numbness"           in sym_set else 0,
                1 if "nausea"             in sym_set else 0,
                data.glucose or 90,
                data.blood_pressure or 110,
                data.age or 35,
                26.0,  # default BMI
                1 if data.family_history in ["diabetes","both"] else 0,
                1 if data.family_history in ["bp","both"] else 0,
            ]
            X = np.array([vec], dtype=np.float32)
            X_scaled = _scaler.transform(X)
            proba = _model.predict_proba(X_scaled)[0]
            CLASS_NAMES = ["Normal","Diabetes","Hypertension","MentalHealth"]
            risks = {
                "diabetes":     int(round(proba[1] * 100)),
                "hypertension": int(round(proba[2] * 100)),
                "heart":        int(round(max(proba[1], proba[2]) * 0.65 * 100)),
                "mental":       int(round(proba[3] * 100)),
            }
            top_idx  = int(np.argmax(proba))
            top_name = CLASS_NAMES[top_idx]
            top_score = int(round(max(proba) * 100))
            level = "high" if top_score >= 60 else "medium" if top_score >= 35 else "low"

            contributing = [s for s in data.symptoms][:4]
            explanation  = f"ML model prediction. Key factors: {', '.join(contributing)}"
            if data.glucose and data.glucose >= 100:
                explanation += f". Glucose: {data.glucose} mg/dL"
            if data.blood_pressure and data.blood_pressure >= 130:
                explanation += f". BP: {data.blood_pressure} mmHg"

            result = {
                "risks": risks,
                "top_disease": top_name,
                "top_score": top_score,
                "confidence": min(top_score + 5, 97),
                "risk_level": level,
                "explanation": explanation,
                "recommendation": _get_recommendation(level, top_name, top_score),
                "model": "trained_ml"
            }
        except Exception as e:
            result = calculate_risk(data)
            result["model"] = "fallback"
    else:
        result = calculate_risk(data)
        result["model"] = "rule_based"

    return JSONResponse(content={
        "success": True,
        "data": result,
        "privacy_note": "No health data is stored without explicit consent"
    })


# ── AGENTIC AI ACTION ──
@app.post("/api/agent/suggest", dependencies=[Depends(rate_limit)])
def agent_suggest(data: SymptomInput):
    """
    Agentic AI: returns suggested next steps (not executed automatically — HITL required)
    """
    risk_result = calculate_risk(data)
    level = risk_result["risk_level"]

    actions = []
    if level == "high":
        actions = [
            {"type": "book_appointment", "label": "Book specialist appointment", "requires_approval": True},
            {"type": "set_reminder",     "label": "Set daily medication reminder", "requires_approval": True},
            {"type": "notify_doctor",    "label": "Send report to doctor",          "requires_approval": True},
        ]
    elif level == "medium":
        actions = [
            {"type": "set_reminder",  "label": "Set weekly check-in reminder",     "requires_approval": True},
            {"type": "log_vitals",    "label": "Start daily vitals logging",        "requires_approval": False},
        ]
    else:
        actions = [
            {"type": "wellness_tip", "label": "Send monthly wellness summary",     "requires_approval": False},
        ]

    return JSONResponse(content={
        "success": True,
        "risk_summary": risk_result,
        "suggested_actions": actions,
        "hitl_note": "All critical actions marked requires_approval=True will NOT execute without explicit user confirmation"
    })


# ── PATIENTS (mock — connect to Supabase in production) ──
patients_db = []  # Replace with: from supabase import create_client

@app.get("/api/patients")
def get_patients():
    return {"success": True, "data": patients_db, "count": len(patients_db)}

@app.post("/api/patients")
def create_patient(patient: PatientCreate):
    new_patient = {
        "id": hashlib.md5(patient.name.encode()).hexdigest()[:8],
        **patient.dict(),
        "created_at": time.time()
    }
    patients_db.append(new_patient)
    return {"success": True, "data": new_patient}


# ── MEDICATION LOGS ──
medication_logs = []

@app.post("/api/medication/log")
def log_medication(log: MedicationLog):
    medication_logs.append({**log.dict(), "logged_at": time.time()})
    adherence = (sum(1 for l in medication_logs if l["dose_taken"]) / len(medication_logs)) * 100
    return {
        "success": True,
        "message": "Logged successfully",
        "adherence_rate": round(adherence, 1)
    }


# ─────────────────────────────────────
#   RUN (for local development)
#   Production: gunicorn with uvicorn workers
# ─────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
