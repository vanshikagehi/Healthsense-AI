"""
HealthSense AI — Synthetic Health Dataset Generator
Generates 1200 labeled patient records for training ML model.

Classes:
  0 = Normal (healthy)
  1 = Diabetes risk
  2 = Hypertension risk
  3 = Mental health risk

Features (12 symptoms + 4 vitals + age + family_history = 18 total):
  fatigue, excessive_thirst, blurred_vision, frequent_urination,
  headache, chest_pain, shortness_breath, weight_loss,
  anxiety, sleep_issues, numbness, nausea,
  glucose, blood_pressure, age, bmi,
  family_diabetes, family_bp
"""

import csv
import random
import os

random.seed(42)  # reproducible results

DATASET_PATH = os.path.join(os.path.dirname(__file__), "health_dataset.csv")

# ── Column names ──
COLUMNS = [
    "fatigue","excessive_thirst","blurred_vision","frequent_urination",
    "headache","chest_pain","shortness_breath","weight_loss",
    "anxiety","sleep_issues","numbness","nausea",
    "glucose","blood_pressure","age","bmi",
    "family_diabetes","family_bp",
    "label"  # 0=Normal, 1=Diabetes, 2=Hypertension, 3=MentalHealth
]

def r(p):
    """Return 1 with probability p, else 0"""
    return 1 if random.random() < p else 0

def rf(lo, hi):
    """Random float in range"""
    return round(random.uniform(lo, hi), 1)

def ri(lo, hi):
    """Random int in range"""
    return random.randint(lo, hi)

rows = []

# ── CLASS 0: Normal / Healthy (300 samples) ──
for _ in range(300):
    rows.append([
        r(0.15), r(0.05), r(0.05), r(0.08),  # symptoms
        r(0.12), r(0.03), r(0.05), r(0.05),
        r(0.10), r(0.15), r(0.04), r(0.08),
        rf(70, 99),    # glucose (normal)
        rf(90, 119),   # BP (normal)
        ri(18, 50),    # age (younger, healthier)
        rf(18, 24.9),  # BMI (normal)
        r(0.10),       # family_diabetes
        r(0.10),       # family_bp
        0              # label: Normal
    ])

# ── CLASS 1: Diabetes Risk (300 samples) ──
for _ in range(300):
    rows.append([
        r(0.80), r(0.90), r(0.75), r(0.85),  # classic diabetes symptoms
        r(0.20), r(0.05), r(0.10), r(0.65),  # weight loss high
        r(0.15), r(0.30), r(0.70), r(0.20),  # numbness high
        rf(110, 300),  # glucose (elevated to diabetic)
        rf(115, 150),  # BP (slightly elevated)
        ri(30, 70),    # age
        rf(25, 40),    # BMI (overweight/obese)
        r(0.75),       # family_diabetes (high)
        r(0.25),       # family_bp
        1              # label: Diabetes
    ])

# ── CLASS 2: Hypertension Risk (300 samples) ──
for _ in range(300):
    rows.append([
        r(0.40), r(0.10), r(0.20), r(0.15),
        r(0.80), r(0.45), r(0.55), r(0.10),  # headache, chest, breath
        r(0.20), r(0.35), r(0.10), r(0.25),
        rf(85, 130),   # glucose (can be normal)
        rf(140, 200),  # BP (hypertensive range)
        ri(35, 75),    # age (older)
        rf(26, 42),    # BMI
        r(0.25),       # family_diabetes
        r(0.80),       # family_bp (high)
        2              # label: Hypertension
    ])

# ── CLASS 3: Mental Health Risk (300 samples) ──
for _ in range(300):
    rows.append([
        r(0.70), r(0.08), r(0.10), r(0.08),
        r(0.35), r(0.15), r(0.25), r(0.15),
        r(0.90), r(0.88), r(0.10), r(0.40),  # anxiety, sleep very high
        rf(75, 105),   # glucose (normal range)
        rf(100, 135),  # BP (mildly elevated from stress)
        ri(18, 45),    # age (younger)
        rf(18, 30),    # BMI
        r(0.15),       # family_diabetes
        r(0.20),       # family_bp
        3              # label: MentalHealth
    ])

# ── SHUFFLE to mix classes ──
random.shuffle(rows)

# ── WRITE CSV ──
with open(DATASET_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(COLUMNS)
    writer.writerows(rows)

print(f"✅ Dataset generated: {DATASET_PATH}")
print(f"   Total rows : {len(rows)}")
print(f"   Features   : {len(COLUMNS) - 1}")
print(f"   Classes    : Normal(0), Diabetes(1), Hypertension(2), MentalHealth(3)")
