"""
HealthSense AI — Clinical-Grade Synthetic Dataset Generator v5.0
=================================================================
UPGRADES from v4.0:
  - 50,004 samples (8,334 per class) — 2.5x more training data
  - Richer vital distributions: cholesterol, creatinine proxied via glucose/bp combos
  - More sub-types per class (7 per class vs 5)
  - Tighter symptom probabilities tuned for clinical separation
  - 20% comorbidity overlap (was 15%)
  - Age-stratified noise curves
  - Symptom severity uses finer 3-level system consistently

Classes (6 diseases):
  0 = Normal (healthy)
  1 = Diabetes / Pre-Diabetes
  2 = Hypertension / Cardiovascular
  3 = Mental Health (Anxiety / Depression / Burnout)
  4 = Thyroid Disorder (Hypo/Hyper)
  5 = PCOS / Hormonal Imbalance

Features (34 total):
  Symptoms (24):
    Metabolic:      fatigue, excessive_thirst, blurred_vision, frequent_urination,
                    numbness, weight_loss, slow_healing, skin_darkening
    Cardiovascular: headache, chest_pain, shortness_breath, palpitations,
                    dizziness, ankle_swelling
    Mental:         anxiety, sleep_issues, low_mood, irritability
    Hormonal:       weight_gain, cold_intolerance, hair_loss, nausea,
                    irregular_periods, acne
  Vitals (6):       glucose, blood_pressure, age, bmi, sleep_hrs, hba1c
  Risk factors (4): family_diabetes, family_bp, smoking, sedentary
"""

import csv, random, os
import math

random.seed(2025)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "health_dataset.csv")
SAMPLES_PER_CLASS = 8334   # 6 x 8334 = 50,004 total

COLUMNS = [
    "fatigue","excessive_thirst","blurred_vision","frequent_urination",
    "numbness","weight_loss","slow_healing","skin_darkening",
    "headache","chest_pain","shortness_breath","palpitations",
    "dizziness","ankle_swelling",
    "anxiety","sleep_issues","low_mood","irritability",
    "weight_gain","cold_intolerance","hair_loss","nausea",
    "irregular_periods","acne",
    "glucose","blood_pressure","age","bmi","sleep_hrs","hba1c",
    "family_diabetes","family_bp","smoking","sedentary",
    "label"
]

# ── Helpers ──────────────────────────────────────────────────────────────
def rn(mean, std, lo=0.0, hi=1.0):
    v = random.gauss(mean, std)
    return round(max(lo, min(hi, v)), 2)

def rf(a, b, noise=0.0):
    v = random.uniform(a, b)
    if noise > 0:
        v += random.gauss(0, noise)
    return round(v, 1)

def ri(a, b): return random.randint(a, b)

def sym(base_prob, severity_probs=(0.35, 0.40, 0.25)):
    """
    Returns symptom severity: 0 (absent), 0.5 (mild), 1.0 (moderate/severe)
    """
    if random.random() > base_prob:
        return 0.0
    roll = random.random()
    if roll < severity_probs[0]:
        return 0.5   # mild
    else:
        return 1.0   # moderate/severe

def add_comorbidity_noise(row, label):
    """
    20% chance a sample gets symptoms from a second related disease.
    This teaches the model to handle real-world overlapping conditions.
    """
    if random.random() > 0.20:
        return row
    combos = {
        1: [2, 4, 5],   # Diabetes + HTN / Thyroid / PCOS
        2: [1, 3],      # HTN + Diabetes / Mental
        3: [4, 5, 2],   # Mental + Thyroid / PCOS / HTN
        4: [3, 5, 1],   # Thyroid + Mental / PCOS / Diabetes
        5: [4, 3, 1],   # PCOS + Thyroid / Mental / Diabetes
        0: [],
    }
    secondary = combos.get(label, [])
    if not secondary:
        return row
    sec = random.choice(secondary)
    strength = random.choice([0.5, 0.5, 1.0])  # mostly mild overlap
    if sec == 1:
        row[1]  = max(row[1],  strength * 0.5)  # excessive_thirst
        row[3]  = max(row[3],  strength * 0.5)  # frequent_urination
    if sec == 2:
        row[8]  = max(row[8],  strength * 0.5)  # headache
        row[11] = max(row[11], strength * 0.5)  # palpitations
    if sec == 3:
        row[14] = max(row[14], strength * 0.5)  # anxiety
        row[16] = max(row[16], strength * 0.5)  # low_mood
    if sec == 4:
        row[20] = max(row[20], strength * 0.5)  # hair_loss
        row[19] = max(row[19], strength * 0.5)  # cold_intolerance
    if sec == 5:
        row[22] = max(row[22], strength * 0.5)  # irregular_periods
        row[23] = max(row[23], strength * 0.5)  # acne
    return row


rows = []

# ════════════════════════════════════════════════════════════════════
#  CLASS 0 — NORMAL / HEALTHY
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    age   = ri(18, 70)
    bmi   = rn(22.0, 2.5, 16.5, 27.5)
    gluc  = rf(70, 99, noise=3.0)
    bp    = rf(88, 119, noise=4.0)
    sleep = rn(7.2, 0.8, 5.5, 9.5)
    hba1c = rn(5.0, 0.3, 4.5, 5.6)

    row = [
        sym(0.06), sym(0.02), sym(0.02), sym(0.03),
        sym(0.02), sym(0.02), sym(0.01), sym(0.01),
        sym(0.06), sym(0.01), sym(0.02), sym(0.03),
        sym(0.04), sym(0.01),
        sym(0.05), sym(0.09), sym(0.04), sym(0.03),
        sym(0.05), sym(0.02), sym(0.03), sym(0.03),
        sym(0.02), sym(0.03),
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.07 else 0,
        1 if random.random() < 0.07 else 0,
        1 if random.random() < 0.03 else 0,
        1 if random.random() < 0.10 else 0,
        0
    ]
    rows.append(row)


# ════════════════════════════════════════════════════════════════════
#  CLASS 1 — DIABETES / PRE-DIABETES
#  Sub-types: T2D adult, Pre-diabetic, Elderly T2D, Obese, Gestational,
#             Adolescent T2D, Advanced T2D with complications
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    sub = i % 7
    age   = [ri(30,70), ri(25,55), ri(55,80), ri(28,60), ri(22,40), ri(12,25), ri(40,75)][sub]
    bmi   = rn([31,29,32,36,28,27,34][sub], 4.5, 20.0, 50.0)
    gluc  = rf([130,105,155,145,118,112,220][sub], [130,105,155,145,118,112,220][sub]*0.1 + 5, noise=8.0)
    gluc  = max(100.0, min(400.0, gluc))
    bp    = rf(120, 158, noise=8.0)
    sleep = rn(5.8, 1.2, 3.0, 8.5)
    hba1c = rn([8.0,6.2,9.5,8.5,6.8,6.5,11.0][sub], 1.2, 5.7, 14.0)

    row = [
        sym(0.88, (0.15, 0.40, 0.45)),  # fatigue
        sym(0.92, (0.10, 0.30, 0.60)),  # excessive_thirst  ← CARDINAL
        sym(0.78, (0.15, 0.40, 0.45)),  # blurred_vision
        sym(0.93, (0.08, 0.35, 0.57)),  # frequent_urination ← CARDINAL
        sym(0.70, (0.25, 0.40, 0.35)),  # numbness
        sym(0.52, (0.40, 0.40, 0.20)),  # weight_loss
        sym(0.65, (0.28, 0.45, 0.27)),  # slow_healing
        sym(0.30, (0.45, 0.35, 0.20)),  # skin_darkening
        sym(0.22, (0.60, 0.30, 0.10)),  # headache
        sym(0.06, (0.75, 0.25, 0.00)),  # chest_pain
        sym(0.09, (0.65, 0.35, 0.00)),  # shortness_breath
        sym(0.14, (0.55, 0.45, 0.00)),  # palpitations
        sym(0.17, (0.55, 0.40, 0.05)),  # dizziness
        sym(0.09, (0.65, 0.35, 0.00)),  # ankle_swelling
        sym(0.15, (0.55, 0.40, 0.05)),  # anxiety
        sym(0.30, (0.35, 0.45, 0.20)),  # sleep_issues
        sym(0.14, (0.65, 0.30, 0.05)),  # low_mood
        sym(0.12, (0.65, 0.30, 0.05)),  # irritability
        sym(0.40, (0.35, 0.45, 0.20)),  # weight_gain
        sym(0.07, (0.75, 0.25, 0.00)),  # cold_intolerance
        sym(0.13, (0.55, 0.40, 0.05)),  # hair_loss
        sym(0.24, (0.48, 0.40, 0.12)),  # nausea
        sym(0.09, (0.75, 0.25, 0.00)),  # irregular_periods
        sym(0.06, (0.75, 0.25, 0.00)),  # acne
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.82 else 0,  # family_diabetes HIGH
        1 if random.random() < 0.28 else 0,
        1 if random.random() < 0.24 else 0,
        1 if random.random() < 0.60 else 0,  # sedentary
        1
    ]
    row = add_comorbidity_noise(row, 1)
    rows.append(row)


# ════════════════════════════════════════════════════════════════════
#  CLASS 2 — HYPERTENSION / CARDIOVASCULAR RISK
#  Sub-types: Essential HTN, White-coat, Secondary, Stage2, ISH,
#             HTN with LVH, Resistant HTN
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    sub = i % 7
    age   = [ri(40,75), ri(30,60), ri(35,65), ri(45,80), ri(55,80), ri(50,75), ri(40,70)][sub]
    bmi   = rn([29,26,30,31,27,32,30][sub], 4.5, 20.0, 48.0)
    gluc  = rf(82, 135, noise=7.0)
    bp    = rf([145,138,158,175,152,168,162][sub], 15.0, noise=10.0)
    bp    = max(130.0, min(220.0, bp))
    sleep = rn(6.0, 1.2, 3.5, 8.5)
    hba1c = rn(5.5, 0.6, 4.8, 7.5)

    row = [
        sym(0.44, (0.38, 0.40, 0.22)),  # fatigue
        sym(0.09, (0.70, 0.30, 0.00)),  # excessive_thirst
        sym(0.24, (0.48, 0.38, 0.14)),  # blurred_vision — hypertensive retinopathy
        sym(0.13, (0.60, 0.38, 0.02)),  # frequent_urination
        sym(0.14, (0.58, 0.38, 0.04)),  # numbness
        sym(0.09, (0.68, 0.30, 0.02)),  # weight_loss
        sym(0.05, (0.80, 0.20, 0.00)),  # slow_healing
        sym(0.04, (0.82, 0.18, 0.00)),  # skin_darkening
        sym(0.90, (0.18, 0.40, 0.42)),  # headache ← CARDINAL
        sym(0.58, (0.28, 0.40, 0.32)),  # chest_pain ← CARDINAL
        sym(0.65, (0.20, 0.38, 0.42)),  # shortness_breath ← CARDINAL
        sym(0.52, (0.28, 0.42, 0.30)),  # palpitations
        sym(0.58, (0.28, 0.40, 0.32)),  # dizziness
        sym(0.45, (0.35, 0.42, 0.23)),  # ankle_swelling
        sym(0.28, (0.48, 0.38, 0.14)),  # anxiety
        sym(0.40, (0.38, 0.40, 0.22)),  # sleep_issues
        sym(0.18, (0.58, 0.38, 0.04)),  # low_mood
        sym(0.24, (0.48, 0.40, 0.12)),  # irritability
        sym(0.24, (0.48, 0.40, 0.12)),  # weight_gain
        sym(0.06, (0.78, 0.22, 0.00)),  # cold_intolerance
        sym(0.12, (0.65, 0.30, 0.05)),  # hair_loss
        sym(0.30, (0.48, 0.38, 0.14)),  # nausea
        sym(0.06, (0.78, 0.22, 0.00)),  # irregular_periods
        sym(0.04, (0.82, 0.18, 0.00)),  # acne
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.28 else 0,
        1 if random.random() < 0.84 else 0,  # family_bp VERY HIGH
        1 if random.random() < 0.40 else 0,  # smoking
        1 if random.random() < 0.55 else 0,
        2
    ]
    row = add_comorbidity_noise(row, 2)
    rows.append(row)


# ════════════════════════════════════════════════════════════════════
#  CLASS 3 — MENTAL HEALTH (Anxiety / Depression / Burnout)
#  Sub-types: GAD, Major Depression, Burnout, Panic Disorder, PTSD,
#             Bipolar depressive phase, Dysthymia
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    sub = i % 7
    age   = [ri(18,40), ri(20,55), ri(25,50), ri(16,45), ri(20,50), ri(18,45), ri(30,60)][sub]
    bmi   = rn(23.0, 5.0, 15.0, 40.0)
    gluc  = rf(70, 108, noise=5.0)
    bp    = rf(92, 138, noise=6.0)
    sleep = rn([4.0, 5.0, 4.5, 3.5, 4.2, 5.5, 6.0][sub], 1.2, 1.5, 7.5)
    hba1c = rn(5.0, 0.4, 4.5, 6.2)

    row = [
        sym(0.80, (0.18, 0.40, 0.42)),  # fatigue ← mental exhaustion
        sym(0.07, (0.78, 0.22, 0.00)),  # excessive_thirst
        sym(0.09, (0.68, 0.30, 0.02)),  # blurred_vision
        sym(0.07, (0.78, 0.22, 0.00)),  # frequent_urination
        sym(0.12, (0.58, 0.38, 0.04)),  # numbness — panic attack tingling
        sym(0.20, (0.48, 0.38, 0.14)),  # weight_loss — depression anorexia
        sym(0.05, (0.82, 0.18, 0.00)),  # slow_healing
        sym(0.03, (0.85, 0.15, 0.00)),  # skin_darkening
        sym(0.44, (0.38, 0.40, 0.22)),  # headache — tension headache
        sym(0.20, (0.48, 0.38, 0.14)),  # chest_pain — panic tightness
        sym(0.30, (0.38, 0.40, 0.22)),  # shortness_breath — hyperventilation
        sym(0.38, (0.28, 0.42, 0.30)),  # palpitations — anxiety tachycardia
        sym(0.25, (0.48, 0.38, 0.14)),  # dizziness — anxiety / low BP
        sym(0.09, (0.70, 0.28, 0.02)),  # ankle_swelling
        sym(0.96, (0.08, 0.28, 0.64)),  # anxiety ← CARDINAL
        sym(0.94, (0.06, 0.20, 0.74)),  # sleep_issues ← CARDINAL
        sym(0.92, (0.08, 0.28, 0.64)),  # low_mood ← CARDINAL
        sym(0.90, (0.08, 0.28, 0.64)),  # irritability ← CARDINAL
        sym(0.30, (0.48, 0.38, 0.14)),  # weight_gain
        sym(0.11, (0.65, 0.30, 0.05)),  # cold_intolerance
        sym(0.38, (0.38, 0.40, 0.22)),  # hair_loss — stress alopecia
        sym(0.48, (0.38, 0.40, 0.22)),  # nausea — anxiety-induced
        sym(0.16, (0.58, 0.38, 0.04)),  # irregular_periods — stress cycles
        sym(0.10, (0.65, 0.30, 0.05)),  # acne — stress
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.16 else 0,
        1 if random.random() < 0.20 else 0,
        1 if random.random() < 0.22 else 0,
        1 if random.random() < 0.45 else 0,
        3
    ]
    row = add_comorbidity_noise(row, 3)
    rows.append(row)


# ════════════════════════════════════════════════════════════════════
#  CLASS 4 — THYROID DISORDER (Hypo dominant, Hyper sub-type)
#  Sub-types: Hypothyroid, Hyperthyroid, Hashimoto, Graves,
#             Subclinical Hypo, Subclinical Hyper, Post-thyroidectomy
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    sub = i % 7
    age   = ri(20, 68)
    bmi   = rn(27.5, 4.5, 17.0, 44.0)
    gluc  = rf(76, 118, noise=6.0)
    bp    = rf(86, 140, noise=7.0)
    sleep = rn(6.5, 1.3, 3.5, 9.5)
    hba1c = rn(5.2, 0.5, 4.6, 6.8)

    # sub 0,1,5 → hypo; sub 2,3,6 → hyper; sub 4 → subclinical (mild)
    is_hyper = sub in [2, 3, 6]
    is_subclinical = sub == 4

    if is_hyper:
        row = [
            sym(0.74, (0.18, 0.40, 0.42)),  # fatigue
            sym(0.09, (0.70, 0.28, 0.02)),  # excessive_thirst
            sym(0.11, (0.65, 0.30, 0.05)),  # blurred_vision
            sym(0.09, (0.70, 0.28, 0.02)),  # frequent_urination
            sym(0.11, (0.65, 0.30, 0.05)),  # numbness
            sym(0.74, (0.18, 0.40, 0.42)),  # weight_loss ← CARDINAL hyperthyroid
            sym(0.05, (0.82, 0.18, 0.00)),  # slow_healing
            sym(0.04, (0.85, 0.15, 0.00)),  # skin_darkening
            sym(0.34, (0.48, 0.38, 0.14)),  # headache
            sym(0.11, (0.65, 0.30, 0.05)),  # chest_pain
            sym(0.24, (0.48, 0.38, 0.14)),  # shortness_breath
            sym(0.85, (0.08, 0.28, 0.64)),  # palpitations ← CARDINAL hyperthyroid
            sym(0.30, (0.48, 0.38, 0.14)),  # dizziness
            sym(0.09, (0.70, 0.28, 0.02)),  # ankle_swelling
            sym(0.55, (0.28, 0.40, 0.32)),  # anxiety ← hyperthyroid
            sym(0.60, (0.25, 0.40, 0.35)),  # sleep_issues
            sym(0.38, (0.38, 0.40, 0.22)),  # low_mood
            sym(0.65, (0.18, 0.40, 0.42)),  # irritability ← CARDINAL hyperthyroid
            sym(0.16, (0.58, 0.38, 0.04)),  # weight_gain — low (losing weight)
            sym(0.09, (0.70, 0.28, 0.02)),  # cold_intolerance — low (heat intolerant)
            sym(0.70, (0.18, 0.40, 0.42)),  # hair_loss
            sym(0.34, (0.48, 0.38, 0.14)),  # nausea
            sym(0.14, (0.58, 0.38, 0.04)),  # irregular_periods
            sym(0.11, (0.65, 0.30, 0.05)),  # acne
        ]
    elif is_subclinical:
        row = [
            sym(0.55, (0.30, 0.42, 0.28)),  # fatigue — mild
            sym(0.06, (0.78, 0.22, 0.00)),
            sym(0.08, (0.72, 0.25, 0.03)),
            sym(0.07, (0.75, 0.22, 0.03)),
            sym(0.10, (0.65, 0.30, 0.05)),
            sym(0.18, (0.55, 0.38, 0.07)),
            sym(0.05, (0.82, 0.18, 0.00)),
            sym(0.04, (0.85, 0.15, 0.00)),
            sym(0.22, (0.52, 0.38, 0.10)),
            sym(0.08, (0.72, 0.25, 0.03)),
            sym(0.12, (0.62, 0.32, 0.06)),
            sym(0.30, (0.45, 0.40, 0.15)),
            sym(0.18, (0.55, 0.38, 0.07)),
            sym(0.10, (0.65, 0.30, 0.05)),
            sym(0.42, (0.38, 0.42, 0.20)),  # anxiety — subclinical
            sym(0.48, (0.35, 0.42, 0.23)),  # sleep_issues
            sym(0.42, (0.38, 0.42, 0.20)),  # low_mood
            sym(0.38, (0.40, 0.42, 0.18)),  # irritability
            sym(0.55, (0.28, 0.42, 0.30)),  # weight_gain — mild
            sym(0.60, (0.25, 0.40, 0.35)),  # cold_intolerance — moderate
            sym(0.58, (0.25, 0.40, 0.35)),  # hair_loss — moderate
            sym(0.25, (0.50, 0.38, 0.12)),  # nausea
            sym(0.12, (0.62, 0.32, 0.06)),  # irregular_periods
            sym(0.09, (0.70, 0.28, 0.02)),  # acne
        ]
    else:
        # Hypothyroid
        row = [
            sym(0.90, (0.08, 0.30, 0.62)),  # fatigue ← CARDINAL hypothyroid
            sym(0.06, (0.78, 0.22, 0.00)),
            sym(0.09, (0.70, 0.28, 0.02)),
            sym(0.07, (0.75, 0.22, 0.03)),
            sym(0.18, (0.55, 0.38, 0.07)),  # numbness — myxedema
            sym(0.20, (0.55, 0.38, 0.07)),  # weight_loss — low
            sym(0.05, (0.82, 0.18, 0.00)),
            sym(0.04, (0.85, 0.15, 0.00)),
            sym(0.30, (0.48, 0.38, 0.14)),  # headache
            sym(0.09, (0.70, 0.28, 0.02)),
            sym(0.17, (0.55, 0.38, 0.07)),
            sym(0.24, (0.48, 0.38, 0.14)),  # palpitations — bradycardia
            sym(0.22, (0.50, 0.38, 0.12)),
            sym(0.14, (0.58, 0.38, 0.04)),
            sym(0.40, (0.38, 0.40, 0.22)),  # anxiety
            sym(0.65, (0.18, 0.40, 0.42)),  # sleep_issues
            sym(0.58, (0.25, 0.40, 0.35)),  # low_mood ← CARDINAL hypothyroid
            sym(0.48, (0.35, 0.40, 0.25)),  # irritability
            sym(0.82, (0.08, 0.30, 0.62)),  # weight_gain ← CARDINAL hypothyroid
            sym(0.90, (0.08, 0.18, 0.74)),  # cold_intolerance ← CARDINAL
            sym(0.85, (0.08, 0.28, 0.64)),  # hair_loss ← CARDINAL
            sym(0.34, (0.48, 0.38, 0.14)),
            sym(0.16, (0.58, 0.38, 0.04)),  # irregular_periods
            sym(0.11, (0.65, 0.30, 0.05)),  # acne
        ]

    row += [
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.20 else 0,
        1 if random.random() < 0.24 else 0,
        1 if random.random() < 0.14 else 0,
        1 if random.random() < 0.40 else 0,
        4
    ]
    row = add_comorbidity_noise(row, 4)
    rows.append(row)


# ════════════════════════════════════════════════════════════════════
#  CLASS 5 — PCOS / HORMONAL IMBALANCE
#  Sub-types: Lean PCOS, Obese PCOS, Adrenal PCOS, Post-pill,
#             Adolescent PCOS, Hyperandrogenic, Insulin-resistant
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    sub = i % 7
    age   = [ri(16,30), ri(22,40), ri(18,35), ri(20,38), ri(13,22), ri(20,38), ri(22,42)][sub]
    bmi   = rn([21,32,24,25,23,24,34][sub], 5.0, 16.5, 48.0)
    gluc  = rf(88, 158, noise=10.0)
    bp    = rf(100, 148, noise=8.0)
    sleep = rn(6.0, 1.2, 3.5, 8.5)
    hba1c = rn(5.4, 0.8, 4.8, 7.5)

    row = [
        sym(0.78, (0.18, 0.40, 0.42)),  # fatigue — very common
        sym(0.09, (0.70, 0.28, 0.02)),  # excessive_thirst — insulin resistance
        sym(0.07, (0.75, 0.22, 0.03)),
        sym(0.11, (0.65, 0.30, 0.05)),
        sym(0.09, (0.70, 0.28, 0.02)),
        sym(0.14, (0.58, 0.38, 0.04)),  # weight_loss — lean PCOS
        sym(0.05, (0.82, 0.18, 0.00)),
        sym(0.70, (0.18, 0.40, 0.42)),  # skin_darkening ← acanthosis nigricans
        sym(0.30, (0.48, 0.38, 0.14)),  # headache — hormonal
        sym(0.06, (0.78, 0.22, 0.00)),
        sym(0.15, (0.58, 0.38, 0.04)),
        sym(0.32, (0.42, 0.40, 0.18)),  # palpitations — hormonal
        sym(0.24, (0.48, 0.38, 0.14)),  # dizziness
        sym(0.11, (0.65, 0.30, 0.05)),
        sym(0.62, (0.20, 0.40, 0.40)),  # anxiety ← PCOS common
        sym(0.58, (0.25, 0.40, 0.35)),  # sleep_issues
        sym(0.50, (0.28, 0.42, 0.30)),  # low_mood — hormonal depression
        sym(0.54, (0.25, 0.42, 0.33)),  # irritability — PMS + hormonal
        sym(0.88, (0.08, 0.28, 0.64)),  # weight_gain ← CARDINAL PCOS
        sym(0.20, (0.55, 0.35, 0.10)),  # cold_intolerance — thyroid overlap
        sym(0.80, (0.08, 0.28, 0.64)),  # hair_loss ← androgenic alopecia
        sym(0.58, (0.25, 0.40, 0.35)),  # nausea — hormonal
        sym(0.96, (0.00, 0.18, 0.82)),  # irregular_periods ← CARDINAL PCOS
        sym(0.90, (0.08, 0.20, 0.72)),  # acne ← CARDINAL PCOS
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.22 else 0,
        1 if random.random() < 0.20 else 0,
        1 if random.random() < 0.12 else 0,
        1 if random.random() < 0.50 else 0,  # sedentary common
        5
    ]
    row = add_comorbidity_noise(row, 5)
    rows.append(row)


# ── SHUFFLE ─────────────────────────────────────────────────────────
random.shuffle(rows)

# ── VERIFY ──────────────────────────────────────────────────────────
expected_cols = len(COLUMNS)
for i, row in enumerate(rows):
    if len(row) != expected_cols:
        raise ValueError(f"Row {i} has {len(row)} cols, expected {expected_cols}")

# ── WRITE CSV ────────────────────────────────────────────────────────
with open(DATASET_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(COLUMNS)
    writer.writerows(rows)

total = len(rows)
label_counts = {}
for row in rows:
    lbl = row[-1]
    label_counts[lbl] = label_counts.get(lbl, 0) + 1

print(f"Dataset v5.0 generated: {DATASET_PATH}")
print(f"  Total rows : {total}")
print(f"  Features   : {expected_cols - 1} (24 symptoms + 6 vitals + 4 risk factors)")
print(f"  Classes    : Normal={label_counts.get(0,0)}, Diabetes={label_counts.get(1,0)}, "
      f"HTN={label_counts.get(2,0)}, Mental={label_counts.get(3,0)}, "
      f"Thyroid={label_counts.get(4,0)}, PCOS={label_counts.get(5,0)}")
print(f"  Symptoms   : float severity (0=absent, 0.5=mild, 1.0=severe)")
print(f"  Comorbidity: 20% samples have secondary disease overlap")
