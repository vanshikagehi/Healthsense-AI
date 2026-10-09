"""
HealthSense AI — Supabase Database Connection
Handles all DB operations with connection pooling for 20K+ users
"""

import os
from supabase import create_client, Client

# ── Load from environment variables (set in Render dashboard) ──
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY", "")

_client: Client = None

def get_db() -> Client:
    """Returns singleton Supabase client (cached — not recreated per request)"""
    global _client
    if _client is None:
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise RuntimeError(
                "SUPABASE_URL and SUPABASE_ANON_KEY must be set in environment variables"
            )
        _client = create_client(SUPABASE_URL, SUPABASE_KEY)
    return _client


# ── Helper functions used by main.py API routes ──

def save_prediction(user_id: str, symptom_data: dict, result: dict) -> str:
    """Save ML prediction to database and return prediction ID"""
    db = get_db()
    try:
        # Save symptom log first
        log = db.table("symptom_logs").insert({
            "user_id":         user_id,
            "symptoms":        symptom_data.get("symptoms", []),
            "glucose":         symptom_data.get("glucose"),
            "blood_pressure":  symptom_data.get("blood_pressure"),
            "notes":           symptom_data.get("notes", "")
        }).execute()

        log_id = log.data[0]["id"] if log.data else None

        # Save prediction
        pred = db.table("ml_predictions").insert({
            "user_id":           user_id,
            "symptom_log_id":    log_id,
            "diabetes_risk":     result["risks"].get("diabetes", 0),
            "hypertension_risk": result["risks"].get("hypertension", 0),
            "heart_risk":        result["risks"].get("heart", 0),
            "mental_risk":       result["risks"].get("mental", 0),
            "top_disease":       result.get("top_disease", ""),
            "confidence":        result.get("confidence", 0),
            "risk_level":        result.get("risk_level", "low"),
            "explanation":       result.get("explanation", ""),
        }).execute()

        return pred.data[0]["id"] if pred.data else None

    except Exception as e:
        print(f"DB save_prediction error: {e}")
        return None


def get_patient_history(user_id: str, limit: int = 10) -> list:
    """Fetch last N predictions for a patient (agent memory)"""
    db = get_db()
    try:
        result = db.table("ml_predictions") \
            .select("*") \
            .eq("user_id", user_id) \
            .order("predicted_at", desc=True) \
            .limit(limit) \
            .execute()
        return result.data or []
    except Exception as e:
        print(f"DB get_patient_history error: {e}")
        return []


def save_agent_action(user_id: str, prediction_id: str, action_type: str, action_label: str) -> str:
    """Save an AI agent suggestion (pending HITL approval)"""
    db = get_db()
    try:
        result = db.table("agent_actions").insert({
            "user_id":           user_id,
            "prediction_id":     prediction_id,
            "action_type":       action_type,
            "action_label":      action_label,
            "requires_approval": True,
            "approved":          None,  # pending
            "executed":          False
        }).execute()
        return result.data[0]["id"] if result.data else None
    except Exception as e:
        print(f"DB save_agent_action error: {e}")
        return None


def approve_action(action_id: str, user_id: str, approved: bool) -> bool:
    """User approves/rejects an AI action (HITL)"""
    db = get_db()
    try:
        from datetime import datetime, timezone
        db.table("agent_actions") \
            .update({
                "approved":    approved,
                "approved_at": datetime.now(timezone.utc).isoformat()
            }) \
            .eq("id", action_id) \
            .eq("user_id", user_id) \
            .execute()
        return True
    except Exception as e:
        print(f"DB approve_action error: {e}")
        return False


def get_medication_adherence(user_id: str) -> float:
    """Get adherence percentage from the adherence view"""
    db = get_db()
    try:
        result = db.table("medication_adherence") \
            .select("adherence_pct") \
            .eq("user_id", user_id) \
            .execute()
        if result.data:
            return float(result.data[0]["adherence_pct"] or 0)
        return 0.0
    except Exception as e:
        print(f"DB get_medication_adherence error: {e}")
        return 0.0


def log_medication_dose(user_id: str, medication_id: str, taken: bool, scheduled_time: str = None) -> bool:
    """Log whether a medication dose was taken"""
    db = get_db()
    try:
        from datetime import datetime, timezone
        db.table("medication_logs").insert({
            "user_id":        user_id,
            "medication_id":  medication_id,
            "scheduled_time": scheduled_time,
            "taken":          taken,
            "taken_at":       datetime.now(timezone.utc).isoformat() if taken else None
        }).execute()
        return True
    except Exception as e:
        print(f"DB log_medication_dose error: {e}")
        return False
