# HealthSense AI 🏥

> **AI-powered health intelligence platform** — ML + Agentic AI + Automation  
> Built for SDG Goal 3: Good Health & Well-Being

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Render-brightgreen)](https://healthsense-ai-frontend.onrender.com)
[![Backend API](https://img.shields.io/badge/Backend%20API-FastAPI-blue)](https://healthsense-ai-backend.onrender.com/docs)

---

## 🚀 What It Does

HealthSense AI combines **Machine Learning**, **Agentic AI**, and **Automation** to deliver proactive, explainable health intelligence:

- 🧠 **Deep Learning** — 3-layer neural network with isotonic calibration predicts Diabetes, Hypertension, Heart Risk, Mental Health
- 🤖 **Multi-Agent AI** — Risk Agent, Nutrition Agent, Medication Agent, Mental Health Agent collaborate with full reasoning chains
- ⚡ **Automation** — Medication reminders, BP tracking, report generation — all with Human-in-the-Loop approval
- 🧑‍💼 **HITL (Human-in-the-Loop)** — AI never acts without explicit user approval
- 🔍 **Explainability Layer** — Feature attribution shows exactly why each prediction was made

---

## 📱 Pages

| Page | Description |
|------|-------------|
| `/` | Landing page — product overview |
| `/pages/auth.html` | Sign In / Sign Up |
| `/pages/patient-dashboard.html` | Patient health dashboard with AI chat |
| `/pages/doctor-dashboard.html` | Doctor dashboard with patient management |
| `/pages/ml-demo.html` | Live ML prediction demo |

---

## 🏗️ Tech Stack

### Frontend
- Vanilla HTML5 + CSS3 + JavaScript (no framework)
- TensorFlow.js — in-browser deep learning
- Responsive design (mobile + desktop)

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
│       ├── patient-dashboard.html
│       ├── doctor-dashboard.html
│       └── ml-demo.html
│
├── backend/
│   ├── main.py                 # FastAPI app (scalable)
│   ├── train_advanced.py       # ML training script
│   ├── generate_dataset.py     # Synthetic dataset generator
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
# Just open in browser
cd frontend
python -m http.server 3000
# Visit: http://localhost:3000
```

### Backend
```bash
# Install dependencies
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

- **Training data:** 2,400 synthetic patient records (600 per class)
- **Classes:** Normal, Diabetes, Hypertension, Mental Health
- **Features:** 12 symptoms + glucose + BP + age + BMI + family history
- **Architecture:** 3-layer neural network + GradientBoosting ensemble
- **Calibration:** Isotonic regression (realistic 30–95% probability range)
- **Clinical thresholds:** ADA 2024 (glucose), JNC8 (blood pressure)

---

## 🔑 5 Unique Features

1. **Explainability Layer** — Feature attribution shows why each prediction was made
2. **Health Memory** — AI remembers patient history across sessions
3. **Multi-Agent Collaboration** — 4 specialized agents work together
4. **Confidence Score** — Calibrated probability, not just binary output
5. **Human-in-the-Loop** — Every AI action requires explicit user approval

---

*Built with ❤️ for good health and well-being*
