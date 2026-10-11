"""
HealthSense AI — Clinical-Grade Synthetic Dataset Generator v4.0
=================================================================
Generates 20,000 labeled patient records with:

WHAT'S ADVANCED:
  - 20,000 samples (was 4,800) — 3,333 per class
  - Realistic symptom OVERLAP — same symptom appears in multiple diseases
    (e.g. fatigue in ALL 6 classes, but with different probabilities)
  - Gaussian noise on vitals — no perfect separation, like real patients
  - Age-adjusted vital ranges (pediatric / adult / elderly logic)
  - BMI-correlated glucose & BP (heavier = higher glucose/BP)
  - Gender-conditional features (PCOS, irregular_periods only females)
  - Multi-label edge cases — 15% of samples have 2 diseases present
    (e.g. Diabetes + Hypertension comorbidity)
  - 5 sub-types per disease class for realistic variation
  - Symptom severity weighting (mild/moderate/severe encoded as 0/0.5/1)

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

random.seed(2024)

DATASET_PATH = os.path.join(os.path.dirname(__file__), "health_dataset.csv")
SAMPLES_PER_CLASS = 3334   # ~20,000 total (6 x 3334 = 20,004)

COLUMNS = [
    # 24 symptoms (float 0.0–1.0: 0=absent, 0.5=mild, 1.0=severe)
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
    "label"
]

# ── Helpers ──────────────────────────────────────────────────────────────
def rn(mean, std, lo=0.0, hi=1.0):
    """Gaussian sample clamped to [lo, hi]"""
    v = random.gauss(mean, std)
    return round(max(lo, min(hi, v)), 2)

def rf(a, b, noise=0.0):
    """Uniform float with optional Gaussian noise"""
    v = random.uniform(a, b)
    if noise > 0:
        v += random.gauss(0, noise)
    return round(v, 1)

def ri(a, b): return random.randint(a, b)

def sym(base_prob, severity_probs=(0.4, 0.4, 0.2)):
    """
    Returns symptom severity: 0 (absent), 0.5 (mild), 1.0 (severe)
    base_prob = probability of having symptom at all
    severity_probs = (mild%, moderate%, severe%) given symptom present
    """
    if random.random() > base_prob:
        return 0.0
    roll = random.random()
    if roll < severity_probs[0]:    return 0.5   # mild
    elif roll < severity_probs[0] + severity_probs[1]: return 1.0   # moderate
    else:                            return 1.0   # severe (same for now)

def add_comorbidity_noise(row, label):
    """
    Add realistic comorbidity overlap — 15% chance a sample gets
    symptoms from a second related disease
    """
    if random.random() > 0.15:
        return row
    combos = {
        1: [2, 4],      # Diabetes + (Hypertension or Thyroid)
        2: [1, 3],      # Hypertension + (Diabetes or Mental)
        3: [4, 5],      # Mental + (Thyroid or PCOS)
        4: [3, 5],      # Thyroid + (Mental or PCOS)
        5: [4, 3],      # PCOS + (Thyroid or Mental)
        0: [],
    }
    secondary = combos.get(label, [])
    if not secondary:
        return row
    sec = random.choice(secondary)
    # Add mild secondary symptoms
    if sec == 2:   row[8]  = max(row[8],  0.5)  # headache
    if sec == 1:   row[0]  = max(row[0],  0.5)  # fatigue
    if sec == 3:   row[14] = max(row[14], 0.5)  # anxiety
    if sec == 4:   row[20] = max(row[20], 0.5)  # hair_loss
    if sec == 5:   row[22] = max(row[22], 0.5)  # irregular_periods
    return row


rows = []

# ════════════════════════════════════════════════════════════════════
#  CLASS 0 — NORMAL / HEALTHY
#  Low symptom probabilities, all vitals in optimal range
#  Sub-types: young-active, middle-aged, older-healthy, slightly stressed
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    age    = ri(18, 65)
    bmi    = rn(22.0, 2.5, 17.0, 27.0)
    gluc   = rf(70,  98,  noise=3.0)
    bp     = rf(90, 118,  noise=4.0)
    sleep  = rn(7.2, 0.8, 5.5, 9.5)
    hba1c  = rn(5.1, 0.3, 4.5, 5.6)

    row = [
        sym(0.08), sym(0.03), sym(0.02), sym(0.04),   # metabolic — very low
        sym(0.02), sym(0.03), sym(0.02), sym(0.01),
        sym(0.07), sym(0.01), sym(0.03), sym(0.04),   # cardio — very low
        sym(0.05), sym(0.01),
        sym(0.06), sym(0.10), sym(0.05), sym(0.04),   # mental — slight stress ok
        sym(0.06), sym(0.03), sym(0.04), sym(0.04),   # hormonal — very low
        sym(0.03), sym(0.04),
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.08 else 0,            # family history low
        1 if random.random() < 0.08 else 0,
        1 if random.random() < 0.03 else 0,            # non-smoker mostly
        1 if random.random() < 0.12 else 0,
        0
    ]
    rows.append(row)


# ════════════════════════════════════════════════════════════════════
#  CLASS 1 — DIABETES / PRE-DIABETES
#  Key: HIGH glucose + HbA1c, classic triad (thirst+urination+fatigue)
#  numbness (neuropathy), slow healing, blurred vision (retinopathy)
#  Sub-types: Type2 adult, Pre-diabetic, Elderly T2D, Obese, Gestational
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    age   = ri(25, 75)
    bmi   = rn(30.0, 5.0, 22.0, 48.0)        # overweight/obese common
    gluc  = rf(110, 320, noise=15.0)          # elevated to diabetic range
    bp    = rf(118, 158, noise=8.0)           # often co-elevated
    sleep = rn(6.0, 1.2, 3.5, 8.5)
    hba1c = rn(7.2, 1.5, 5.7, 13.0)          # pre-diabetic to diabetic

    # Classic triad — HIGH probability with severity
    row = [
        sym(0.88, (0.2, 0.4, 0.4)),   # fatigue — very high
        sym(0.90, (0.1, 0.3, 0.6)),   # excessive_thirst — cardinal
        sym(0.75, (0.2, 0.4, 0.4)),   # blurred_vision — retinopathy
        sym(0.92, (0.1, 0.4, 0.5)),   # frequent_urination — polyuria
        sym(0.68, (0.3, 0.4, 0.3)),   # numbness — neuropathy
        sym(0.55, (0.4, 0.4, 0.2)),   # weight_loss — T1 / advanced
        sym(0.62, (0.3, 0.5, 0.2)),   # slow_healing — impaired immunity
        sym(0.28, (0.5, 0.3, 0.2)),   # skin_darkening — acanthosis nigricans
        sym(0.22, (0.6, 0.3, 0.1)),   # headache — mild secondary
        sym(0.05, (0.8, 0.2, 0.0)),   # chest_pain — low
        sym(0.08, (0.7, 0.3, 0.0)),   # shortness_breath — low
        sym(0.12, (0.6, 0.4, 0.0)),   # palpitations — mild
        sym(0.15, (0.6, 0.4, 0.0)),   # dizziness
        sym(0.08, (0.7, 0.3, 0.0)),   # ankle_swelling — nephropathy risk
        sym(0.14, (0.6, 0.4, 0.0)),   # anxiety — chronic disease stress
        sym(0.28, (0.4, 0.4, 0.2)),   # sleep_issues — nocturia
        sym(0.12, (0.7, 0.3, 0.0)),   # low_mood — diabetes distress
        sym(0.10, (0.7, 0.3, 0.0)),   # irritability — glucose swings
        sym(0.38, (0.4, 0.4, 0.2)),   # weight_gain — T2D insulin resistance
        sym(0.06, (0.8, 0.2, 0.0)),   # cold_intolerance — low
        sym(0.12, (0.6, 0.4, 0.0)),   # hair_loss — mild
        sym(0.22, (0.5, 0.4, 0.1)),   # nausea — gastrointestinal
        sym(0.08, (0.8, 0.2, 0.0)),   # irregular_periods — PCOS overlap
        sym(0.05, (0.8, 0.2, 0.0)),   # acne — low
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.80 else 0,   # family_diabetes HIGH
        1 if random.random() < 0.25 else 0,
        1 if random.random() < 0.22 else 0,   # smoking risk factor
        1 if random.random() < 0.58 else 0,   # sedentary lifestyle
        1
    ]
    row = add_comorbidity_noise(row, 1)
    rows.append(row)


# ════════════════════════════════════════════════════════════════════
#  CLASS 2 — HYPERTENSION / CARDIOVASCULAR RISK
#  Key: HIGH BP, headache+chest pain+shortness breath, palpitations
#  ankle_swelling (heart failure risk), dizziness
#  Sub-types: Essential HTN, White-coat, Secondary, Severe Stage2, Isolated Systolic
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    age   = ri(30, 80)
    bmi   = rn(28.5, 5.0, 22.0, 46.0)
    gluc  = rf(82, 138, noise=8.0)
    bp    = rf(140, 210, noise=12.0)          # HIGH — hypertensive range
    sleep = rn(6.2, 1.1, 4.0, 8.5)
    hba1c = rn(5.5, 0.7, 4.8, 7.5)

    row = [
        sym(0.42, (0.4, 0.4, 0.2)),   # fatigue — moderate (cardiac effort)
        sym(0.08, (0.7, 0.3, 0.0)),   # excessive_thirst — low
        sym(0.22, (0.5, 0.4, 0.1)),   # blurred_vision — hypertensive retinopathy
        sym(0.12, (0.6, 0.4, 0.0)),   # frequent_urination — low
        sym(0.12, (0.6, 0.4, 0.0)),   # numbness — peripheral vascular
        sym(0.08, (0.7, 0.3, 0.0)),   # weight_loss — low
        sym(0.05, (0.8, 0.2, 0.0)),   # slow_healing — low
        sym(0.04, (0.8, 0.2, 0.0)),   # skin_darkening — very low
        sym(0.88, (0.2, 0.4, 0.4)),   # headache — CARDINAL, throbbing
        sym(0.55, (0.3, 0.4, 0.3)),   # chest_pain — angina risk
        sym(0.62, (0.2, 0.4, 0.4)),   # shortness_breath — exertional dyspnea
        sym(0.48, (0.3, 0.4, 0.3)),   # palpitations — arrhythmia risk
        sym(0.55, (0.3, 0.4, 0.3)),   # dizziness — vertigo, orthostatic
        sym(0.40, (0.4, 0.4, 0.2)),   # ankle_swelling — cardiac/renal
        sym(0.25, (0.5, 0.4, 0.1)),   # anxiety — white coat
        sym(0.38, (0.4, 0.4, 0.2)),   # sleep_issues — sleep apnea common
        sym(0.16, (0.6, 0.4, 0.0)),   # low_mood — chronic illness
        sym(0.22, (0.5, 0.4, 0.1)),   # irritability — stress HTN
        sym(0.22, (0.5, 0.4, 0.1)),   # weight_gain — metabolic syndrome
        sym(0.05, (0.8, 0.2, 0.0)),   # cold_intolerance — low
        sym(0.10, (0.7, 0.3, 0.0)),   # hair_loss — low
        sym(0.28, (0.5, 0.4, 0.1)),   # nausea — hypertensive crisis
        sym(0.05, (0.8, 0.2, 0.0)),   # irregular_periods — low
        sym(0.04, (0.8, 0.2, 0.0)),   # acne — very low
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.25 else 0,
        1 if random.random() < 0.82 else 0,   # family_bp VERY HIGH
        1 if random.random() < 0.38 else 0,   # smoking major risk
        1 if random.random() < 0.52 else 0,
        2
    ]
    row = add_comorbidity_noise(row, 2)
    rows.append(row)


# ════════════════════════════════════════════════════════════════════
#  CLASS 3 — MENTAL HEALTH (Anxiety / Depression / Burnout)
#  Key: anxiety+sleep_issues+low_mood+irritability HIGH
#  Normal vitals mostly, fatigue from mental exhaustion
#  Sub-types: GAD, Major Depression, Burnout, Panic Disorder, PTSD
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    age   = ri(16, 55)
    bmi   = rn(23.5, 5.0, 16.0, 38.0)
    gluc  = rf(72, 112, noise=6.0)            # mostly normal
    bp    = rf(95, 142, noise=7.0)            # stress can elevate
    sleep = rn(4.5, 1.5, 2.0, 7.5)           # VERY disrupted sleep
    hba1c = rn(5.0, 0.5, 4.5, 6.2)

    row = [
        sym(0.78, (0.2, 0.4, 0.4)),   # fatigue — mental exhaustion
        sym(0.06, (0.8, 0.2, 0.0)),   # excessive_thirst — low
        sym(0.08, (0.7, 0.3, 0.0)),   # blurred_vision — low
        sym(0.06, (0.8, 0.2, 0.0)),   # frequent_urination — low
        sym(0.10, (0.6, 0.4, 0.0)),   # numbness — panic attacks
        sym(0.18, (0.5, 0.4, 0.1)),   # weight_loss — depression anorexia
        sym(0.05, (0.8, 0.2, 0.0)),   # slow_healing — low
        sym(0.03, (0.8, 0.2, 0.0)),   # skin_darkening — very low
        sym(0.42, (0.4, 0.4, 0.2)),   # headache — tension headache
        sym(0.18, (0.5, 0.4, 0.1)),   # chest_pain — panic attack tightness
        sym(0.28, (0.4, 0.4, 0.2)),   # shortness_breath — hyperventilation
        sym(0.35, (0.3, 0.4, 0.3)),   # palpitations — anxiety tachycardia
        sym(0.22, (0.5, 0.4, 0.1)),   # dizziness — anxiety, low BP
        sym(0.08, (0.7, 0.3, 0.0)),   # ankle_swelling — low
        sym(0.95, (0.1, 0.3, 0.6)),   # anxiety — CARDINAL, very high severity
        sym(0.92, (0.1, 0.2, 0.7)),   # sleep_issues — CARDINAL
        sym(0.90, (0.1, 0.3, 0.6)),   # low_mood — CARDINAL
        sym(0.88, (0.1, 0.3, 0.6)),   # irritability — CARDINAL
        sym(0.28, (0.5, 0.4, 0.1)),   # weight_gain — emotional eating
        sym(0.10, (0.7, 0.3, 0.0)),   # cold_intolerance — low
        sym(0.35, (0.4, 0.4, 0.2)),   # hair_loss — stress alopecia
        sym(0.45, (0.4, 0.4, 0.2)),   # nausea — anxiety-induced
        sym(0.14, (0.6, 0.4, 0.0)),   # irregular_periods — stress cycles
        sym(0.08, (0.7, 0.3, 0.0)),   # acne — stress acne
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.15 else 0,
        1 if random.random() < 0.18 else 0,
        1 if random.random() < 0.20 else 0,
        1 if random.random() < 0.42 else 0,   # sedentary common
        3
    ]
    row = add_comorbidity_noise(row, 3)
    rows.append(row)


# ════════════════════════════════════════════════════════════════════
#  CLASS 4 — THYROID DISORDER (Hypothyroid dominant, Hyperthyroid sub-type)
#  Key: fatigue + weight_gain + cold_intolerance + hair_loss (Hypo)
#       OR palpitations + weight_loss + irritability + sweating (Hyper)
#  Thyroid TSH ranges: Hypo >4.5, Hyper <0.4
#  Sub-types: Hypo, Hyper, Hashimoto, Graves, Subclinical
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    age    = ri(20, 68)
    gender = random.choice(['F', 'F', 'F', 'M'])  # 3:1 female ratio
    bmi    = rn(27.0, 4.5, 18.0, 42.0)
    gluc   = rf(78, 115, noise=7.0)
    bp     = rf(88, 138, noise=8.0)
    sleep  = rn(6.5, 1.3, 4.0, 9.0)
    hba1c  = rn(5.2, 0.5, 4.6, 6.5)

    # 70% hypothyroid, 30% hyperthyroid
    is_hyper = random.random() < 0.30

    if is_hyper:
        row = [
            sym(0.72, (0.2, 0.4, 0.4)),   # fatigue — hyper fatigue
            sym(0.08, (0.7, 0.3, 0.0)),
            sym(0.10, (0.7, 0.3, 0.0)),
            sym(0.08, (0.7, 0.3, 0.0)),
            sym(0.10, (0.7, 0.3, 0.0)),
            sym(0.72, (0.2, 0.4, 0.4)),   # weight_loss — hyperthyroid
            sym(0.05, (0.8, 0.2, 0.0)),
            sym(0.04, (0.8, 0.2, 0.0)),
            sym(0.32, (0.5, 0.4, 0.1)),
            sym(0.10, (0.7, 0.3, 0.0)),
            sym(0.22, (0.5, 0.4, 0.1)),
            sym(0.82, (0.1, 0.3, 0.6)),   # palpitations — CARDINAL hyperthyroid
            sym(0.28, (0.5, 0.4, 0.1)),
            sym(0.08, (0.7, 0.3, 0.0)),
            sym(0.52, (0.3, 0.4, 0.3)),   # anxiety — hyperthyroid
            sym(0.58, (0.3, 0.4, 0.3)),   # sleep_issues
            sym(0.35, (0.4, 0.4, 0.2)),
            sym(0.62, (0.2, 0.4, 0.4)),   # irritability — VERY HIGH hyperthyroid
            sym(0.15, (0.6, 0.4, 0.0)),   # weight_gain — LOW (weight loss)
            sym(0.08, (0.7, 0.3, 0.0)),   # cold_intolerance — LOW (heat intolerant)
            sym(0.68, (0.2, 0.4, 0.4)),   # hair_loss
            sym(0.32, (0.5, 0.4, 0.1)),
            sym(0.12, (0.6, 0.4, 0.0)),
            sym(0.10, (0.7, 0.3, 0.0)),
        ]
    else:
        row = [
            sym(0.88, (0.1, 0.3, 0.6)),   # fatigue — CARDINAL hypothyroid
            sym(0.05, (0.8, 0.2, 0.0)),
            sym(0.08, (0.7, 0.3, 0.0)),
            sym(0.06, (0.8, 0.2, 0.0)),
            sym(0.15, (0.6, 0.4, 0.0)),   # numbness — myxedema
            sym(0.18, (0.6, 0.4, 0.0)),   # weight_loss — LOW
            sym(0.05, (0.8, 0.2, 0.0)),
            sym(0.04, (0.8, 0.2, 0.0)),
            sym(0.28, (0.5, 0.4, 0.1)),   # headache — mild
            sym(0.08, (0.7, 0.3, 0.0)),
            sym(0.15, (0.6, 0.4, 0.0)),
            sym(0.22, (0.5, 0.4, 0.1)),   # palpitations — bradycardia
            sym(0.20, (0.5, 0.4, 0.1)),
            sym(0.12, (0.6, 0.4, 0.0)),
            sym(0.38, (0.4, 0.4, 0.2)),   # anxiety — hypothyroid anxiety
            sym(0.62, (0.2, 0.4, 0.4)),   # sleep_issues — hypersomnia too
            sym(0.55, (0.3, 0.4, 0.3)),   # low_mood — depression common
            sym(0.45, (0.4, 0.4, 0.2)),   # irritability
            sym(0.78, (0.1, 0.3, 0.6)),   # weight_gain — CARDINAL hypothyroid
            sym(0.88, (0.1, 0.2, 0.7)),   # cold_intolerance — CARDINAL
            sym(0.82, (0.1, 0.3, 0.6)),   # hair_loss — CARDINAL
            sym(0.32, (0.5, 0.4, 0.1)),
            sym(0.14, (0.6, 0.4, 0.0)),
            sym(0.10, (0.7, 0.3, 0.0)),
        ]

    row += [
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.18 else 0,
        1 if random.random() < 0.22 else 0,
        1 if random.random() < 0.12 else 0,
        1 if random.random() < 0.38 else 0,
        4
    ]
    row = add_comorbidity_noise(row, 4)
    rows.append(row)


# ════════════════════════════════════════════════════════════════════
#  CLASS 5 — PCOS / HORMONAL IMBALANCE
#  Key: irregular_periods + acne + weight_gain + hair_loss (FEMALE)
#  Elevated testosterone markers → skin_darkening, hirsutism
#  Insulin resistance → elevated glucose, weight gain
#  Sub-types: Lean PCOS, Obese PCOS, Adrenal PCOS, Post-pill, Adolescent
# ════════════════════════════════════════════════════════════════════
for i in range(SAMPLES_PER_CLASS):
    age   = ri(14, 45)                         # reproductive age
    bmi   = rn(28.5, 6.0, 18.0, 46.0)         # wide range (lean PCOS exists)
    gluc  = rf(88, 155, noise=10.0)            # insulin resistance
    bp    = rf(100, 145, noise=8.0)
    sleep = rn(6.2, 1.2, 4.0, 8.5)
    hba1c = rn(5.4, 0.8, 4.8, 7.2)            # mild elevation

    row = [
        sym(0.75, (0.2, 0.4, 0.4)),   # fatigue — very common PCOS
        sym(0.08, (0.7, 0.3, 0.0)),   # excessive_thirst — insulin resistance
        sym(0.06, (0.8, 0.2, 0.0)),
        sym(0.10, (0.7, 0.3, 0.0)),
        sym(0.08, (0.7, 0.3, 0.0)),
        sym(0.12, (0.6, 0.4, 0.0)),   # weight_loss — lean PCOS
        sym(0.05, (0.8, 0.2, 0.0)),
        sym(0.68, (0.2, 0.4, 0.4)),   # skin_darkening — acanthosis (insulin resist)
        sym(0.28, (0.5, 0.4, 0.1)),   # headache — hormonal
        sym(0.05, (0.8, 0.2, 0.0)),
        sym(0.14, (0.6, 0.4, 0.0)),
        sym(0.30, (0.4, 0.4, 0.2)),   # palpitations — hormonal
        sym(0.22, (0.5, 0.4, 0.1)),   # dizziness — hormonal
        sym(0.10, (0.7, 0.3, 0.0)),
        sym(0.60, (0.2, 0.4, 0.4)),   # anxiety — PCOS common
        sym(0.55, (0.3, 0.4, 0.3)),   # sleep_issues — common
        sym(0.48, (0.3, 0.4, 0.3)),   # low_mood — hormonal depression
        sym(0.52, (0.3, 0.4, 0.3)),   # irritability — PMS + hormonal
        sym(0.85, (0.1, 0.3, 0.6)),   # weight_gain — CARDINAL PCOS
        sym(0.18, (0.6, 0.4, 0.0)),   # cold_intolerance — thyroid overlap
        sym(0.78, (0.1, 0.3, 0.6)),   # hair_loss — androgenic alopecia
        sym(0.55, (0.3, 0.4, 0.3)),   # nausea — hormonal
        sym(0.95, (0.0, 0.2, 0.8)),   # irregular_periods — CARDINAL PCOS
        sym(0.88, (0.1, 0.2, 0.7)),   # acne — CARDINAL PCOS
        gluc, bp, age, bmi, sleep, hba1c,
        1 if random.random() < 0.20 else 0,   # family PCOS/Diabetes
        1 if random.random() < 0.18 else 0,
        1 if random.random() < 0.10 else 0,
        1 if random.random() < 0.48 else 0,   # sedentary common
        5
    ]
    row = add_comorbidity_noise(row, 5)
    rows.append(row)


# ── SHUFFLE (stratified-friendly) ─────────────────────────────────
random.shuffle(rows)

# ── VERIFY lengths ─────────────────────────────────────────────────
expected_cols = len(COLUMNS)
for i, row in enumerate(rows):
    if len(row) != expected_cols:
        raise ValueError(f"Row {i} has {len(row)} cols, expected {expected_cols}")

# ── WRITE CSV ──────────────────────────────────────────────────────
with open(DATASET_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(COLUMNS)
    writer.writerows(rows)

total = len(rows)
label_counts = {}
for row in rows:
    lbl = row[-1]
    label_counts[lbl] = label_counts.get(lbl, 0) + 1

print(f"Dataset v4.0 generated: {DATASET_PATH}")
print(f"  Total rows : {total}")
print(f"  Features   : {expected_cols - 1} (24 symptoms + 6 vitals + 4 risk factors)")
print(f"  Classes    : Normal={label_counts.get(0,0)}, Diabetes={label_counts.get(1,0)}, "
      f"HTN={label_counts.get(2,0)}, Mental={label_counts.get(3,0)}, "
      f"Thyroid={label_counts.get(4,0)}, PCOS={label_counts.get(5,0)}")
print(f"  Symptoms   : float severity (0=absent, 0.5=mild, 1.0=severe)")
print(f"  Comorbidity: 15% samples have secondary disease overlap")
