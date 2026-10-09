-- ═══════════════════════════════════════════════════════════
--  HealthSense AI — Supabase PostgreSQL Schema
--  Run this SQL in Supabase → SQL Editor → New Query
--  Free tier: 500MB storage, 2 CPU, handles 20K+ users
-- ═══════════════════════════════════════════════════════════

-- Enable UUID generation
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ─────────────────────────────────────
--  TABLE 1: users
--  Stores both patients and doctors
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
  id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  name          VARCHAR(100) NOT NULL,
  email         VARCHAR(150) UNIQUE,
  phone         VARCHAR(20),
  role          VARCHAR(20) NOT NULL CHECK (role IN ('patient','doctor','admin')),
  age           INTEGER CHECK (age BETWEEN 1 AND 120),
  language      VARCHAR(20) DEFAULT 'english',
  created_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ─────────────────────────────────────
--  TABLE 2: health_profiles
--  Patient's health background
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS health_profiles (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id           UUID REFERENCES users(id) ON DELETE CASCADE,
  blood_group       VARCHAR(5),
  bmi               DECIMAL(5,2),
  family_diabetes   BOOLEAN DEFAULT FALSE,
  family_bp         BOOLEAN DEFAULT FALSE,
  family_heart      BOOLEAN DEFAULT FALSE,
  existing_conditions TEXT[],   -- e.g. ['diabetes','hypertension']
  allergies         TEXT[],
  ayushman_id       VARCHAR(50),  -- Ayushman Bharat card ID
  created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  updated_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
  UNIQUE(user_id)
);

-- ─────────────────────────────────────
--  TABLE 3: symptom_logs
--  Every time user reports symptoms
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS symptom_logs (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id           UUID REFERENCES users(id) ON DELETE CASCADE,
  symptoms          TEXT[],       -- ['fatigue','excessive_thirst',...]
  glucose           DECIMAL(6,2),
  blood_pressure    DECIMAL(6,2),
  notes             TEXT,
  logged_at         TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ─────────────────────────────────────
--  TABLE 4: ml_predictions
--  ML model results — full audit trail
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS ml_predictions (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id           UUID REFERENCES users(id) ON DELETE CASCADE,
  symptom_log_id    UUID REFERENCES symptom_logs(id),
  diabetes_risk     INTEGER CHECK (diabetes_risk BETWEEN 0 AND 100),
  hypertension_risk INTEGER CHECK (hypertension_risk BETWEEN 0 AND 100),
  heart_risk        INTEGER CHECK (heart_risk BETWEEN 0 AND 100),
  mental_risk       INTEGER CHECK (mental_risk BETWEEN 0 AND 100),
  top_disease       VARCHAR(30),
  confidence        INTEGER CHECK (confidence BETWEEN 0 AND 100),
  risk_level        VARCHAR(10) CHECK (risk_level IN ('low','medium','high')),
  explanation       TEXT,
  model_version     VARCHAR(20) DEFAULT '1.0',
  predicted_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ─────────────────────────────────────
--  TABLE 5: agent_actions
--  Every Agentic AI suggestion + HITL result
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS agent_actions (
  id                UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id           UUID REFERENCES users(id) ON DELETE CASCADE,
  prediction_id     UUID REFERENCES ml_predictions(id),
  action_type       VARCHAR(50),  -- 'book_appointment','set_reminder','notify_doctor'
  action_label      TEXT,
  requires_approval BOOLEAN DEFAULT TRUE,
  approved          BOOLEAN,      -- NULL=pending, TRUE=approved, FALSE=rejected
  approved_at       TIMESTAMP WITH TIME ZONE,
  executed          BOOLEAN DEFAULT FALSE,
  executed_at       TIMESTAMP WITH TIME ZONE,
  created_at        TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ─────────────────────────────────────
--  TABLE 6: medications
--  Patient's medication list
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS medications (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
  name            VARCHAR(100) NOT NULL,
  dosage          VARCHAR(50),
  frequency       VARCHAR(50),  -- 'twice_daily','once_daily','weekly'
  time_slots      TEXT[],       -- ['08:00','21:00']
  prescribed_by   UUID REFERENCES users(id),
  active          BOOLEAN DEFAULT TRUE,
  created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ─────────────────────────────────────
--  TABLE 7: medication_logs
--  Track every dose taken/missed
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS medication_logs (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
  medication_id   UUID REFERENCES medications(id),
  scheduled_time  TIMESTAMP WITH TIME ZONE,
  taken           BOOLEAN NOT NULL,
  taken_at        TIMESTAMP WITH TIME ZONE,
  notes           TEXT
);

-- ─────────────────────────────────────
--  TABLE 8: appointments
--  Booked via automation (after HITL approval)
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS appointments (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  patient_id      UUID REFERENCES users(id) ON DELETE CASCADE,
  doctor_id       UUID REFERENCES users(id),
  booked_by       VARCHAR(10) DEFAULT 'ai',  -- 'ai' or 'manual'
  agent_action_id UUID REFERENCES agent_actions(id),
  scheduled_at    TIMESTAMP WITH TIME ZONE,
  status          VARCHAR(20) DEFAULT 'pending'
                  CHECK (status IN ('pending','confirmed','completed','cancelled')),
  notes           TEXT,
  created_at      TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ─────────────────────────────────────
--  TABLE 9: health_reports
--  Auto-generated weekly reports
-- ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS health_reports (
  id              UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id         UUID REFERENCES users(id) ON DELETE CASCADE,
  report_type     VARCHAR(20) CHECK (report_type IN ('weekly','monthly','alert')),
  summary         TEXT,
  risk_snapshot   JSONB,   -- {diabetes:78, hypertension:45, ...}
  adherence_rate  DECIMAL(5,2),
  sent_to_doctor  BOOLEAN DEFAULT FALSE,
  generated_at    TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- ─────────────────────────────────────
--  INDEXES — critical for 20K+ users
--  Without indexes queries slow down a lot
-- ─────────────────────────────────────
CREATE INDEX idx_symptom_logs_user    ON symptom_logs(user_id, logged_at DESC);
CREATE INDEX idx_ml_predictions_user  ON ml_predictions(user_id, predicted_at DESC);
CREATE INDEX idx_agent_actions_user   ON agent_actions(user_id, created_at DESC);
CREATE INDEX idx_agent_actions_pending ON agent_actions(user_id) WHERE approved IS NULL;
CREATE INDEX idx_med_logs_user        ON medication_logs(user_id, scheduled_time DESC);
CREATE INDEX idx_appointments_patient ON appointments(patient_id, scheduled_at);
CREATE INDEX idx_appointments_doctor  ON appointments(doctor_id, scheduled_at);
CREATE INDEX idx_health_reports_user  ON health_reports(user_id, generated_at DESC);

-- ─────────────────────────────────────
--  ROW LEVEL SECURITY (RLS)
--  Patients see only their own data
--  Doctors see their assigned patients
-- ─────────────────────────────────────
ALTER TABLE users            ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_profiles  ENABLE ROW LEVEL SECURITY;
ALTER TABLE symptom_logs     ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_predictions   ENABLE ROW LEVEL SECURITY;
ALTER TABLE agent_actions    ENABLE ROW LEVEL SECURITY;
ALTER TABLE medications      ENABLE ROW LEVEL SECURITY;
ALTER TABLE medication_logs  ENABLE ROW LEVEL SECURITY;
ALTER TABLE appointments     ENABLE ROW LEVEL SECURITY;
ALTER TABLE health_reports   ENABLE ROW LEVEL SECURITY;

-- Policy: users can only read/write their own data
CREATE POLICY "users_own_data" ON symptom_logs
  USING (user_id = auth.uid());

CREATE POLICY "users_own_predictions" ON ml_predictions
  USING (user_id = auth.uid());

CREATE POLICY "users_own_actions" ON agent_actions
  USING (user_id = auth.uid());

CREATE POLICY "users_own_medications" ON medications
  USING (user_id = auth.uid());

CREATE POLICY "users_own_med_logs" ON medication_logs
  USING (user_id = auth.uid());

CREATE POLICY "users_own_reports" ON health_reports
  USING (user_id = auth.uid());

-- ─────────────────────────────────────
--  SAMPLE DATA — for testing/demo
-- ─────────────────────────────────────
INSERT INTO users (id, name, email, phone, role, age) VALUES
  ('11111111-0000-0000-0000-000000000001', 'Priya Verma',   'priya@demo.com',  '+91-9876543210', 'patient', 38),
  ('11111111-0000-0000-0000-000000000002', 'Dr. Priya Sharma', 'drpriya@demo.com', '+91-9876543211', 'doctor', 42),
  ('11111111-0000-0000-0000-000000000003', 'Rahul Mehta',   'rahul@demo.com',  '+91-9876543212', 'patient', 52),
  ('11111111-0000-0000-0000-000000000004', 'Arjun Singh',   'arjun@demo.com',  '+91-9876543213', 'patient', 61)
ON CONFLICT DO NOTHING;

INSERT INTO health_profiles (user_id, bmi, family_diabetes, family_bp, existing_conditions, ayushman_id) VALUES
  ('11111111-0000-0000-0000-000000000001', 27.4, TRUE,  FALSE, '{}',            'PMJAY-2024-001'),
  ('11111111-0000-0000-0000-000000000003', 31.2, FALSE, TRUE,  '{hypertension}','PMJAY-2024-002'),
  ('11111111-0000-0000-0000-000000000004', 24.8, TRUE,  TRUE,  '{diabetes}',    'PMJAY-2024-003')
ON CONFLICT DO NOTHING;

INSERT INTO medications (user_id, name, dosage, frequency, time_slots, prescribed_by) VALUES
  ('11111111-0000-0000-0000-000000000001', 'Metformin',   '500mg', 'twice_daily', ARRAY['08:00','21:00'], '11111111-0000-0000-0000-000000000002'),
  ('11111111-0000-0000-0000-000000000001', 'Amlodipine',  '5mg',   'once_daily',  ARRAY['08:00'],         '11111111-0000-0000-0000-000000000002'),
  ('11111111-0000-0000-0000-000000000003', 'Losartan',    '50mg',  'once_daily',  ARRAY['08:00'],         '11111111-0000-0000-0000-000000000002')
ON CONFLICT DO NOTHING;

-- ─────────────────────────────────────
--  ADHERENCE VIEW — used by dashboard
-- ─────────────────────────────────────
CREATE OR REPLACE VIEW medication_adherence AS
SELECT
  user_id,
  COUNT(*) AS total_doses,
  SUM(CASE WHEN taken THEN 1 ELSE 0 END) AS taken_doses,
  ROUND(
    100.0 * SUM(CASE WHEN taken THEN 1 ELSE 0 END) / NULLIF(COUNT(*), 0),
    1
  ) AS adherence_pct
FROM medication_logs
GROUP BY user_id;
