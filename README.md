# HealthSense AI 🏥

**AI-powered multi-disease health risk prediction with Agentic AI workflows**

> Live Demo: [https://healthsense-ai-backend.onrender.com](https://healthsense-ai-backend.onrender.com)  
> Frontend: Deployed via Render Static Site  
> Backend API: `https://healthsense-ai-backend.onrender.com`

---

## Overview

HealthSense AI is a full-stack health intelligence platform that combines a **production-grade ML ensemble model** with **Agentic AI workflows** to predict risk for 6 chronic conditions:

| Class | Disease |
|-------|---------|
| 0 | Normal (Healthy) |
| 1 | Diabetes / Pre-Diabetes |
| 2 | Hypertension / Cardiovascular |
| 3 | Mental Health (Anxiety / Depression / Burnout) |
| 4 | Thyroid Disorder (Hypo/Hyper) |
| 5 | PCOS / Hormonal Imbalance |

---

## ML Model — v5.0 (Voting Ensemble)

### Architecture
- **Model type**: Soft Voting Ensemble + Isotonic Calibration
- **Base models**: HistGradientBoosting + XGBoost + LightGBM + RandomForest + ExtraTrees + GradientBoosting
- **Training samples**: 50,004 (8,334 per class)
- **Features**: 34 raw → 46 engineered features
- **Accuracy**: 98.73%
- **AUC-ROC (macro)**: 0.9996

### Feature Engineering (34 → 46)
```
Raw features (34):
  Symptoms (24): fatigue, excessive_thirst, blurred_vision, frequent_urination,
                 numbness, weight_loss, slow_healing, skin_darkening,
                 headache, chest_pain, shortness_breath, palpitations, dizziness,
                 ankle_swelling, anxiety, sleep_issues, low_mood, irritability,
                 weight_gain, cold_intolerance, hair_loss, nausea,
                 irregular_periods, acne
  Vitals (6):    glucose, blood_pressure, age, bmi, sleep_hrs, hba1c
  Risk (4):      family_diabetes, family_bp, smoking, sedentary

Engineered features (+12):
  metabolic_score, cardio_score, mental_score, hormonal_score,
  glucose×bmi, bp×age, anxiety×sleep_deprivation, hba1c×glucose,
  mean_symptom_burden, diabetes_triad, thyroid_triad, pcos_triad
```

### Dataset v5.0
- **50,004 samples** — 2.5× more than v4.0
- **7 sub-types per class** (e.g. PCOS: classical, lean, adolescent, insulin-resistant, hidden, comorbid, metabolic)
- **20% comorbidity overlap** — realistic co-occurring conditions
- **Age-stratified noise** curves for clinical realism
- **3-level symptom severity**: absent (0) / mild (0.5) / moderate-severe (1.0)

### Post-processing
- **PCOS boost**: If female + `irregular_periods` or `acne` present → probability amplified by 1.8×–3.5× depending on number of cardinal symptoms. Ensures clinical PCOS cases are not masked by Normal baseline.
- **PCOS gender gate**: Male patients always receive 0% PCOS probability.
- **Normal override**: If top class is Normal but a disease probability ≥ 20%, that disease is reported instead.

### Test Results (8/8 passing)
```
[OK] Diabetes (male, glucose=180, HbA1c=8.0)      -> Diabetes       100%
[OK] Hypertension (male, BP=165)                   -> Hypertension   100%
[OK] Mental Health (female, anxiety+low_mood)      -> MentalHealth   100%
[OK] Thyroid (female, cold_intolerance+hair_loss)  -> Thyroid        100%
[OK] PCOS (female, irregular_periods+acne)         -> PCOS            95%
[OK] Normal (male, all vitals healthy)             -> Normal         100%
[OK] Mental (no PCOS symptoms, female)             -> MentalHealth    61%
[OK] Chest pain only (male)                        -> Normal         100%
```

---

## Agentic AI Features

### 1. Symptom Intelligence Agent (`/api/agent/suggest`)
- Accepts a partial symptom list + demographics
- Returns **next most likely symptoms to check** based on disease probability landscape
- Uses decision-tree traversal over trained model feature importances

### 2. Medication Logging Agent (`/api/medication/log`)
- Tracks patient medications with structured storage
- AI-assisted drug interaction warnings (rule-based + symptom correlation)

### 3. Patient History Agent (`/api/patients`)
- CRUD operations for patient records
- Temporal risk tracking — compare predictions over time

### 4. Risk Explanation Engine
- Every prediction includes a natural-language explanation:  
  `"Voting ensemble v5.0. Key factors: irregular_periods, acne, weight_gain. BMI: 27"`
- Contributing symptoms extracted and ranked

---

## Architecture

```
Frontend (Static HTML/CSS/JS)
  └── frontend/pages/ml-demo.html   — Main prediction UI
  └── frontend/pages/dashboard.html — Patient dashboard
  └── frontend/js/main.js           — Shared utilities

Backend (FastAPI + Python)
  └── backend/main.py               — API server (crash-resistant)
  └── backend/models/
      ├── model.pkl                  — Trained ensemble (166 MB, Git LFS)
      ├── scaler.pkl                 — StandardScaler
      ├── feature_cols.pkl           — Feature column names
      ├── engineered_cols.pkl        — Engineered feature names
      └── model_metadata.json        — v5.0 metadata
  └── backend/train_advanced.py     — Training script
  └── backend/generate_dataset.py   — Dataset generator v5.0

Infrastructure
  └── render.yaml                   — Render Blueprint (auto-deploy)
  └── requirements.txt              — Python dependencies
  └── .gitattributes                — Git LFS for *.pkl
```

---

## API Reference

### `POST /api/predict`
```json
{
  "symptoms": ["irregular_periods", "acne", "weight_gain"],
  "gender": "female",
  "age": 22,
  "glucose": 95,
  "blood_pressure": 115,
  "bmi": 27.5,
  "sleep_hrs": 6,
  "hba1c": 5.2,
  "family_history": "none",
  "smoking": "never",
  "activity": "low"
}
```
**Response**:
```json
{
  "top_disease": "PCOS",
  "top_score": 95,
  "confidence": 97,
  "risk_level": "high",
  "risks": { "pcos": 95, "diabetes": 2, "hypertension": 1, "mental": 1, "thyroid": 1, "heart": 2 },
  "explanation": "Voting ensemble v5.0. Key factors: irregular_periods, acne, weight_gain. BMI: 27.5",
  "recommendation": { ... },
  "model": "voting_ensemble_v5"
}
```

### `GET /health`
Returns model status, version, accuracy, and uptime.

### `GET /api/model-info`
Returns full model metadata including training samples, accuracy, AUC, base models.

### `POST /api/agent/suggest`
Agentic symptom suggestion based on current symptom profile.

### `GET /api/patients` / `POST /api/patients`
Patient CRUD operations.

### `POST /api/medication/log`
Log medications with AI-assisted interaction checks.

---

## Deploy on Render

### Automatic Deploy (Blueprint)
1. Fork this repo
2. Go to [Render Dashboard](https://dashboard.render.com) → New → Blueprint
3. Connect `vanshikagehi/Healthsense-AI`, branch `main`
4. Render auto-detects `render.yaml` and deploys both services

### Manual Deploy
```bash
# Backend
cd backend
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000

# Model training (optional — model.pkl already included via Git LFS)
python generate_dataset.py
python train_advanced.py
```

### Environment
- Python 3.10+
- Dependencies: `fastapi uvicorn scikit-learn xgboost lightgbm joblib numpy pydantic`
- Model file tracked via **Git LFS** (166 MB `.pkl`)

---

## Crash Prevention

The backend is hardened against crashes:

1. **UTF-8 encoding fix** — print statements use safe ASCII, no emoji in server logs
2. **Graceful model fallback** — if `model.pkl` fails to load, rule-based prediction runs
3. **try/except everywhere** — no unhandled exceptions crash the server
4. **PCOS gender gate** — male patients never trigger PCOS path (no division by zero risk)
5. **`render.yaml` skip-training** — model not re-trained on every deploy (avoids 10-min timeout)

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML5, CSS3, Vanilla JS |
| Backend | FastAPI (Python) |
| ML | scikit-learn, XGBoost, LightGBM |
| Model | Soft Voting Ensemble + Isotonic Calibration |
| Deploy | Render.com (Blueprint) |
| Model Storage | Git LFS |
| API | REST + JSON |

---

## Project Structure

```
BOB PROJECT/
├── backend/
│   ├── main.py                    # FastAPI server
│   ├── train_advanced.py          # ML training (v5.0)
│   ├── generate_dataset.py        # Dataset generator (v5.0)
│   ├── requirements.txt
│   └── models/
│       ├── model.pkl              # Trained model (Git LFS)
│       ├── scaler.pkl
│       ├── feature_cols.pkl
│       ├── engineered_cols.pkl
│       └── model_metadata.json
├── frontend/
│   ├── index.html
│   └── pages/
│       ├── ml-demo.html           # Main prediction UI
│       └── dashboard.html
├── render.yaml                    # Render deployment config
├── .gitattributes                 # Git LFS config
└── README.md
```

---

*HealthSense AI — Built for clinical-grade ML accuracy with production-ready deployment*
