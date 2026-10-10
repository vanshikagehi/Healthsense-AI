"""
HealthSense AI — Synthetic Health Dataset Generator v3.0
=========================================================
Generates 4,800 labeled patient records for training ML model.

Classes (6 diseases):
  0 = Normal (healthy)
  1 = Diabetes risk
  2 = Hypertension risk
  3 = Mental Health risk
  4 = Thyroid risk
  5 = PCOS risk

Features (24 symptoms + 6 vitals + 4 risk factors = 34 total):
  Metabolic:      fatigue, excessive_thirst, blurred_vision, frequent_urination,
                  numbness, weight_loss, slow_healing, skin_darkening
  Cardiovascular: headache, chest_pain, shortness_breath, palpitations,
                  dizziness, ankle_swelling
  Mental:         anxiety, sleep_issues, low_mood, irritability
  Thyroid/PCOS:   weight_gain, cold_intolerance, hair_loss, nausea,
                  irregular_periods, acne
  Vitals:         glucose, blood_pressure, age, bmi, sleep_hrs, hba1c
  Risk factors:   family_diabetes, family_bp, smoking, sedentary
"""

import csv, random, os

random.seed(42)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "health_dataset.csv")

COLUMNS = [
    # 24 symptoms
    "fatigue","excessive_thirst","blurred_vision","frequent_urination",
    "numbness","weight_loss","slow_healing","skin_darkening",
    "headache","chest_pain","shortness_breath","palpitations",
    "dizziness","ankle_swelling",
    "anxiety","sleep_issues","low_mood","irritability",
    "weight_gain","cold_intolerance","hair_loss","nausea",
    "irregular_periods","acne",
    # 6 vitals
    "glucose","blood_pressure","age","bmi","sleep_hrs","hba1c",
    # 4 risk factors
    "family_diabetes","family_bp","smoking","sedentary",
    # label
    "label"
]

def r(p):    return 1 if random.random() < p else 0
def rf(a,b): return round(random.uniform(a, b), 1)
def ri(a,b): return random.randint(a, b)

rows = []

# ── CLASS 0: Normal / Healthy (800 samples) ──
for _ in range(800):
    rows.append([
        r(0.12), r(0.04), r(0.04), r(0.06),   # metabolic
        r(0.03), r(0.04), r(0.03), r(0.02),
        r(0.10), r(0.02), r(0.04), r(0.05),   # cardio
        r(0.08), r(0.02),
        r(0.08), r(0.12), r(0.06), r(0.05),   # mental
        r(0.08), r(0.04), r(0.05), r(0.06),   # thyroid/pcos
        r(0.06), r(0.05),
        rf(70,99), rf(90,119), ri(18,50), rf(18,24.9), rf(6.5,8.5), rf(4.8,5.6),  # vitals
        r(0.08), r(0.08), r(0.05), r(0.15),   # risk factors
        0
    ])

# ── CLASS 1: Diabetes Risk (800 samples) ──
for _ in range(800):
    age = ri(28,75)
    rows.append([
        r(0.82), r(0.92), r(0.78), r(0.88),
        r(0.72), r(0.68), r(0.65), r(0.30),
        r(0.18), r(0.05), r(0.08), r(0.10),
        r(0.12), r(0.08),
        r(0.12), r(0.28), r(0.10), r(0.08),
        r(0.35), r(0.05), r(0.12), r(0.22),
        r(0.08), r(0.05),
        rf(110,320), rf(110,155), age, rf(24,42), rf(5,6.5), rf(5.8,10.0),
        r(0.78), r(0.22), r(0.20), r(0.55),
        1
    ])

# ── CLASS 2: Hypertension Risk (800 samples) ──
for _ in range(800):
    age = ri(35,78)
    rows.append([
        r(0.40), r(0.08), r(0.22), r(0.12),
        r(0.10), r(0.08), r(0.05), r(0.04),
        r(0.82), r(0.48), r(0.58), r(0.40),
        r(0.52), r(0.35),
        r(0.22), r(0.38), r(0.15), r(0.20),
        r(0.20), r(0.05), r(0.10), r(0.28),
        r(0.05), r(0.04),
        rf(82,135), rf(140,210), age, rf(26,44), rf(5,7), rf(5.0,6.2),
        r(0.22), r(0.82), r(0.35), r(0.50),
        2
    ])

# ── CLASS 3: Mental Health Risk (800 samples) ──
for _ in range(800):
    age = ri(16,48)
    rows.append([
        r(0.72), r(0.06), r(0.08), r(0.06),
        r(0.08), r(0.12), r(0.05), r(0.03),
        r(0.38), r(0.12), r(0.22), r(0.28),
        r(0.20), r(0.08),
        r(0.92), r(0.90), r(0.88), r(0.85),
        r(0.25), r(0.10), r(0.30), r(0.42),
        r(0.12), r(0.08),
        rf(72,108), rf(95,138), age, rf(17,32), rf(3.5,5.5), rf(4.8,5.8),
        r(0.12), r(0.18), r(0.18), r(0.40),
        3
    ])

# ── CLASS 4: Thyroid Risk (800 samples) ──
for _ in range(800):
    age = ri(20,65)
    rows.append([
        r(0.78), r(0.05), r(0.08), r(0.06),
        r(0.12), r(0.35), r(0.05), r(0.04),
        r(0.30), r(0.08), r(0.18), r(0.38),
        r(0.22), r(0.15),
        r(0.45), r(0.55), r(0.50), r(0.52),
        r(0.72), r(0.80), r(0.75), r(0.35),
        r(0.12), r(0.10),
        rf(75,110), rf(90,135), age, rf(22,38), rf(5.5,7.5), rf(4.9,5.9),
        r(0.15), r(0.20), r(0.10), r(0.35),
        4
    ])

# ── CLASS 5: PCOS Risk (800 samples — female-specific) ──
for _ in range(800):
    age = ri(15,42)
    rows.append([
        r(0.70), r(0.06), r(0.05), r(0.08),
        r(0.08), r(0.10), r(0.05), r(0.65),
        r(0.25), r(0.05), r(0.12), r(0.28),
        r(0.20), r(0.08),
        r(0.55), r(0.50), r(0.45), r(0.48),
        r(0.80), r(0.15), r(0.72), r(0.55),
        r(0.88), r(0.82),
        rf(88,150), rf(100,140), age, rf(26,42), rf(5.5,7.0), rf(5.0,6.5),
        r(0.18), r(0.15), r(0.12), r(0.45),
        5
    ])

# ── SHUFFLE ──
random.shuffle(rows)

# ── WRITE CSV ──
with open(DATASET_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(COLUMNS)
    writer.writerows(rows)

print(f"✅ Dataset v3.0 generated: {DATASET_PATH}")
print(f"   Total rows : {len(rows)}")
print(f"   Features   : {len(COLUMNS) - 1}")
print(f"   Classes    : Normal(0), Diabetes(1), Hypertension(2), Mental(3), Thyroid(4), PCOS(5)")
