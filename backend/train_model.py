"""
HealthSense AI — ML Model Trainer
Trains a Random Forest + Neural Network ensemble on health dataset.
Exports:
  - model.pkl         (scikit-learn, for backend API)
  - model_tfjs/       (TensorFlow.js, for browser frontend)
  - label_encoder.pkl
  - scaler.pkl

Run:  python backend/train_model.py
Needs: pip install scikit-learn tensorflow tensorflowjs pandas numpy joblib
"""

import os
import json
import numpy as np
import pandas as pd
import joblib

# ── sklearn ──
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline

DATASET_PATH = os.path.join(os.path.dirname(__file__), "health_dataset.csv")
MODEL_DIR    = os.path.join(os.path.dirname(__file__), "models")
TFJS_DIR     = os.path.join(os.path.dirname(__file__), "..", "js", "tfjs_model")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(TFJS_DIR, exist_ok=True)

CLASS_NAMES = ["Normal", "Diabetes", "Hypertension", "MentalHealth"]


# ══════════════════════════════════════
#   STEP 1 — LOAD & VERIFY DATASET
# ══════════════════════════════════════
print("\n" + "="*50)
print("  HealthSense AI — ML Model Training")
print("="*50)

# Auto-generate dataset if missing
if not os.path.exists(DATASET_PATH):
    print("⚙️  Dataset not found. Generating...")
    exec(open(os.path.join(os.path.dirname(__file__), "generate_dataset.py")).read())

df = pd.read_csv(DATASET_PATH)
print(f"\n📊 Dataset loaded: {len(df)} rows, {df.shape[1]-1} features")
print(f"   Class distribution:\n{df['label'].value_counts().rename(index={0:'Normal',1:'Diabetes',2:'Hypertension',3:'MentalHealth'})}")

# Features and labels
FEATURE_COLS = [c for c in df.columns if c != "label"]
X = df[FEATURE_COLS].values.astype(np.float32)
y = df["label"].values.astype(int)


# ══════════════════════════════════════
#   STEP 2 — PREPROCESS
# ══════════════════════════════════════
print("\n⚙️  Preprocessing...")

# Scale features (important for Neural Network)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
print("   ✓ StandardScaler fitted and saved")

# Train/test split — 80/20
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
print(f"   ✓ Train: {len(X_train)} | Test: {len(X_test)}")


# ══════════════════════════════════════
#   STEP 3 — TRAIN MODELS
# ══════════════════════════════════════

# ── Model A: Random Forest (fast, interpretable) ──
print("\n🌲 Training Random Forest...")
rf = RandomForestClassifier(
    n_estimators=200,       # 200 trees — more = better accuracy
    max_depth=12,           # prevent overfitting
    min_samples_leaf=2,
    class_weight="balanced",# handles class imbalance
    random_state=42,
    n_jobs=-1               # use all CPU cores
)
rf.fit(X_train, y_train)
rf_pred  = rf.predict(X_test)
rf_acc   = accuracy_score(y_test, rf_pred)
print(f"   ✓ Accuracy: {rf_acc*100:.1f}%")

# Feature importance
importances = rf.feature_importances_
feat_imp = sorted(zip(FEATURE_COLS, importances), key=lambda x: x[1], reverse=True)
print("   Top 5 features:")
for fname, imp in feat_imp[:5]:
    print(f"     {fname}: {imp:.3f}")

# ── Model B: Gradient Boosting (high accuracy) ──
print("\n⚡ Training Gradient Boosting...")
gb = GradientBoostingClassifier(
    n_estimators=150,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
gb.fit(X_train, y_train)
gb_pred = gb.predict(X_test)
gb_acc  = accuracy_score(y_test, gb_pred)
print(f"   ✓ Accuracy: {gb_acc*100:.1f}%")

# ── Model C: Neural Network MLP ──
print("\n🧠 Training Neural Network (MLP)...")
mlp = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3 hidden layers
    activation="relu",
    solver="adam",
    alpha=0.001,            # L2 regularization
    batch_size=32,
    learning_rate="adaptive",
    max_iter=500,
    random_state=42,
    early_stopping=True,    # stop if no improvement
    validation_fraction=0.1,
    verbose=False
)
mlp.fit(X_train, y_train)
mlp_pred = mlp.predict(X_test)
mlp_acc  = accuracy_score(y_test, mlp_pred)
print(f"   ✓ Accuracy: {mlp_acc*100:.1f}%")


# ══════════════════════════════════════
#   STEP 4 — PICK BEST MODEL
# ══════════════════════════════════════
results = {"RandomForest": (rf, rf_acc), "GradientBoosting": (gb, gb_acc), "MLP": (mlp, mlp_acc)}
best_name, (best_model, best_acc) = max(results.items(), key=lambda x: x[1][1])

print(f"\n🏆 Best Model: {best_name} — {best_acc*100:.1f}% accuracy")

# Cross-validation on best model
cv_scores = cross_val_score(best_model, X_scaled, y, cv=5, scoring="accuracy")
print(f"   Cross-validation (5-fold): {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

# Save best model
joblib.dump(best_model, os.path.join(MODEL_DIR, "model.pkl"))
print(f"   ✓ Saved to models/model.pkl")


# ══════════════════════════════════════
#   STEP 5 — DETAILED EVALUATION
# ══════════════════════════════════════
print("\n📋 Classification Report:")
print(classification_report(y_test, best_model.predict(X_test),
      target_names=CLASS_NAMES))

print("🔢 Confusion Matrix:")
cm = confusion_matrix(y_test, best_model.predict(X_test))
print(cm)


# ══════════════════════════════════════
#   STEP 6 — EXPORT MODEL METADATA
#   (Used by backend API + frontend)
# ══════════════════════════════════════
metadata = {
    "model_type": best_name,
    "accuracy": round(best_acc * 100, 1),
    "cv_mean": round(cv_scores.mean() * 100, 1),
    "cv_std":  round(cv_scores.std() * 100, 1),
    "features": FEATURE_COLS,
    "classes": CLASS_NAMES,
    "feature_importances": {k: round(float(v), 4) for k, v in feat_imp},
    "training_samples": len(X_train),
    "test_samples": len(X_test),
}
with open(os.path.join(MODEL_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)
print(f"\n✅ Metadata saved to models/model_metadata.json")


# ══════════════════════════════════════
#   STEP 7 — EXPORT FOR BROWSER (TF.js)
#   Requires: pip install tensorflow tensorflowjs
# ══════════════════════════════════════
print("\n🌐 Exporting for browser (TensorFlow.js)...")
try:
    import tensorflow as tf
    import tensorflowjs as tfjs

    # Build a Keras model matching the MLP structure
    keras_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(len(FEATURE_COLS),)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(4, activation="softmax")  # 4 classes
    ])

    keras_model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    keras_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=60,
        batch_size=32,
        verbose=1,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)
        ]
    )

    keras_acc = keras_model.evaluate(X_test, y_test, verbose=0)[1]
    print(f"   Keras model accuracy: {keras_acc*100:.1f}%")

    # Export to TF.js format (loads directly in browser)
    tfjs.converters.save_keras_model(keras_model, TFJS_DIR)
    print(f"   ✓ TF.js model saved to js/tfjs_model/")
    print(f"   ✓ Load in browser: tf.loadLayersModel('js/tfjs_model/model.json')")

    # Also save scaler params as JSON for browser use
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist(),
        "features": FEATURE_COLS
    }
    with open(os.path.join(TFJS_DIR, "scaler_params.json"), "w") as f:
        json.dump(scaler_params, f, indent=2)
    print("   ✓ Scaler params saved for browser normalization")

except ImportError:
    print("   ⚠️  TensorFlow not installed. Skipping TF.js export.")
    print("   To install: pip install tensorflow tensorflowjs")
    print("   TF.js export needed only for browser-based prediction.")

print("\n" + "="*50)
print("  TRAINING COMPLETE")
print(f"  Best model: {best_name}")
print(f"  Accuracy: {best_acc*100:.1f}%")
print("="*50 + "\n")
