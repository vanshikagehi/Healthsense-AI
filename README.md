# HealthSense AI 🏥

> **AI-powered health intelligence platform** — ML + Agentic AI + Automation  
> Built for SDG Goal 3: Good Health & Well-Being

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-brightgreen)](https://healthsense-ai-frontend.onrender.com)
[![Backend API](https://img.shields.io/badge/Backend%20API-FastAPI-blue)](https://healthsense-ai-backend.onrender.com/docs)
[![GitHub](https://img.shields.io/badge/GitHub-vanshikagehi-black)](https://github.com/vanshikagehi/Healthsense-AI)

---

## 🚀 What It Does

HealthSense AI combines **Machine Learning**, **Agentic AI**, and **Automation** to deliver proactive, explainable health intelligence:

- 🧠 **Deep Learning v3.0** — 4-layer neural network predicts **6 diseases**: Diabetes, Hypertension, Heart Risk, Mental Health, Thyroid Risk, PCOS
- 🤖 **Agentic AI v2.0** — **8 specialized agents** collaborate with full reasoning chains, session memory, and proactive alerts
- ⚡ **Automation** — Medication reminders, BP tracking, report generation — all with Human-in-the-Loop approval
- 🧑‍💼 **HITL (Human-in-the-Loop)** — AI never acts without explicit user approval
- 🔍 **Explainability Layer** — Layer-by-layer feature attribution shows exactly why each prediction was made
- 💬 **Hindi + English AI Chat** — Patients can type symptoms in Hindi or English for free-text prediction

---

## 🧠 ML Engine v3.0 — What's New

| Feature | Previous | Now |
|---|---|---|
| **Diseases predicted** | 4 | **6** (Thyroid + PCOS added) |
| **Symptoms supported** | 12 chips | **24 chips + free text box** |
| **Input features** | 18 | **36 features** |
| **Neural Net layers** | 3 | **4 layers** |
| **Vitals inputs** | Glucose, BP, Age | + **BMI, HbA1c, Sleep hrs** |
| **Risk factors** | Family history | + **Smoking, Activity, Gender** |
| **Interaction features** | 4 | **14 patterns** |
| **Training data** | 2,400 records | **4,800 records** |

### 6 Diseases Detected:
1. 🩸 **Diabetes** — ADA 2024 thresholds + Polyuria-Polydipsia triad detection
2. 🫀 **Hypertension** — JNC8 guidelines + 14-day BP trend analysis
3. ❤️ **Heart Risk** — Framingham score + cardiac combo detection
4. 🧘 **Mental Health** — Depression triad + anxiety-sleep-fatigue compound
5. 🔬 **Thyroid Risk** — Hypothyroid triad (weight gain + cold + hair loss)
6. 🌸 **PCOS Risk** — Rotterdam criteria (irregular periods + acne + weight gain)

### Free-Text Symptom Input:
Patients can type symptoms in **Hindi or English**:
> *"chakkar aa raha hai, baal bahut gir rahe hain, thakaan mehsoos hoti hai"*

AI auto-detects: ✅ Dizziness · ✅ Hair Loss · ✅ Fatigue — and merges with chip selections.

---

## 🤖 Agentic AI v2.0 — What's New

### 8 Specialized Agents (was 3):

| Agent | Role |
|---|---|
| 🧠 Risk Agent | Layer-by-layer diabetes reasoning chain |
| 🥗 Nutrition Agent | Chrono-nutrition + evidence-based meal plan |
| 💊 Medication Agent | Drug interaction scan + adherence dashboard |
| ⚡ Agentic AI (Orchestrator) | Doctor matching + 5-step automation queue |
| 🫀 Cardio Agent | BP trend + Framingham heart risk |
| 🧬 Lab Intelligence Agent | Priority test queue + thyroid risk flag |
| 🚨 Emergency Alert Agent | Emergency guide + hypoglycemia first aid |
| ⚡ Automation Orchestrator | **Proactive alerts** after conversation patterns detected |

### New AI Features:
- **Session Memory** — Tracks conversation topics, flags risk patterns across turns
- **Proactive Alerts** — After 3 messages, AI automatically flags compound risk signals
- **Memory Sync** — After 6 turns, AI confirms what it saved for next session
- **Cross-agent reasoning** — Agents share findings (e.g. Lab Agent flags thyroid when Mental Agent detects fatigue)
- **12 Quick Topics** — BP, Lab tests, Heart risk, Sleep & stress, Thyroid, Emergency

---

## 📱 Pages

| Page | Description |
|------|-------------|
| `/` | Landing page — product overview |
| `/pages/auth.html` | Sign In / Sign Up |
| `/pages/patient-dashboard.html` | Patient health dashboard with 8-agent AI chat |
| `/pages/doctor-dashboard.html` | Doctor dashboard with patient management |
| `/pages/ml-demo.html` | Live ML prediction demo — 6 diseases, free-text input |

---

## 🏗️ Tech Stack

### Frontend
- Vanilla HTML5 + CSS3 + JavaScript (no framework)
- TensorFlow.js — in-browser 4-layer deep learning
- Responsive design (mobile + desktop)
- Hindi + English NLP keyword parser (custom)

### Backend
- **FastAPI** (Python) — async, production-grade
- **Gunicorn** — 4 workers for 20K–30K user scalability
- **scikit-learn** — GradientBoosting + RandomForest ensemble
- **sentence-transformers** — NLP symptom analysis
- **TensorFlow** — Deep learning model training

### Database
- **Supabase** (PostgreSQL) — 9 tables with Row Level Security
- Patient data, predictions, medication logs, automation actions

### Deployment
- **Render** — Frontend (static) + Backend (web service)
- **GitHub** — CI/CD via render.yaml

---

## 📁 Project Structure

```
healthsense-ai/
├── frontend/
│   ├── index.html              # Landing page
│   ├── style.css               # Global styles
│   ├── js/main.js              # Shared JS (HITL, toast, confidence bar)
│   └── pages/
│       ├── auth.html           # Authentication
│       ├── patient-dashboard.html  # 8-agent AI chat + patient dashboard
│       ├── doctor-dashboard.html   # Doctor dashboard with AI alerts
│       └── ml-demo.html            # ML v3.0 — 6 diseases, free-text input
│
├── backend/
│   ├── main.py                 # FastAPI app (scalable)
│   ├── train_advanced.py       # ML training script
│   ├── generate_dataset.py     # Synthetic dataset generator (4,800 records)
│   ├── requirements.txt
│   └── database/
│       ├── schema.sql          # Supabase schema (9 tables)
│       └── db.py               # Supabase client helpers
│
├── render.yaml                 # Render deployment config
├── Procfile                    # Gunicorn startup
├── .env.example                # Environment variables template
└── SECURITY.md                 # Security guidelines
```

---

## ⚙️ Setup & Run Locally

### Frontend
```bash
cd frontend
python -m http.server 3000
# Visit: http://localhost:3000
```

### Backend
```bash
cd backend
pip install -r requirements.txt

# Setup environment
cp ../.env.example ../.env
# Fill in SUPABASE_URL, SUPABASE_ANON_KEY, etc.

# Generate training data + train model
python generate_dataset.py
python train_advanced.py

# Start server
uvicorn main:app --reload --port 8000
```

---

## 🔐 Environment Variables

Copy `.env.example` to `.env` and fill in:

```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key
SUPABASE_SERVICE_KEY=your-service-key
JWT_SECRET=your-random-32-char-secret
ENVIRONMENT=development
```

> ⚠️ **Never commit `.env`** — it's in `.gitignore`

---

## 📊 ML Model Details

- **Architecture:** 4-layer neural network + GradientBoosting ensemble
- **Training data:** 4,800 synthetic patient records (800 per class)
- **Classes:** Diabetes · Hypertension · Heart Risk · Mental Health · Thyroid · PCOS
- **Input features:** 36 (24 symptoms + 6 vitals + 4 family history + 2 lifestyle)
- **Calibration:** Isotonic regression (realistic 22–94% probability range)
- **Clinical standards:** ADA 2024 (glucose) · JNC8 (blood pressure) · WHO (BMI) · Rotterdam (PCOS)
- **Interaction features:** 14 non-linear patterns (e.g. Polyuria-Polydipsia triad, Cardiac combo, Hypothyroid triad)
- **Runs in:** Browser (TensorFlow.js) — zero data sent to server

---

## 🔑 8 Unique Features

1. **6-Disease ML Prediction** — Diabetes, Hypertension, Heart, Mental Health, Thyroid, PCOS — all in one engine
2. **Free-Text Symptom Input** — Type in Hindi or English, AI auto-parses and maps to features
3. **Layer-by-Layer Explainability** — Feature attribution per neural net layer shows WHY each prediction was made
4. **8-Agent Agentic AI** — Specialized agents collaborate with cross-agent reasoning and session memory
5. **Proactive Health Alerts** — AI detects compound risk patterns across conversation turns
6. **Health Memory** — AI remembers patient history, no need to re-explain each session
7. **Confidence Score** — Isotonic-calibrated probability, not just binary output
8. **Human-in-the-Loop** — Every AI action requires explicit user approval before execution

---

*Built with ❤️ for good health and well-being*
