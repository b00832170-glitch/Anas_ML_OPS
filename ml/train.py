"""
train.py - Entraînement du modèle de prédiction de risque crédit
Target : loan_status (0 = pas de défaut, 1 = défaut)
"""

import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    accuracy_score,
)

# ── Chemins ────────────────────────────────────────────────────────────────────
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "credit_risk_dataset.csv"
MODEL_DIR = BASE_DIR / "ml"

# ── 1. Chargement ──────────────────────────────────────────────────────────────
print("📂 Chargement du dataset...")
df = pd.read_csv(DATA_PATH)
print(f"   {df.shape[0]} lignes, {df.shape[1]} colonnes")

# ── 2. Nettoyage ───────────────────────────────────────────────────────────────
print("🧹 Nettoyage...")
df = df.dropna()

# Suppression des outliers évidents (âge > 100, emp_length > 60)
df = df[df["person_age"] <= 100]
df = df[df["person_emp_length"] <= 60]

# ── 3. Encodage des variables catégorielles ────────────────────────────────────
print("🔠 Encodage des variables catégorielles...")
categorical_cols = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]

encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    encoders[col] = le

# ── 4. Séparation features / target ────────────────────────────────────────────
TARGET = "loan_status"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

print(f"   Features : {FEATURES}")
print(f"   Répartition target : {y.value_counts().to_dict()}")

# ── 5. Split train / test ──────────────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ── 6. Entraînement ────────────────────────────────────────────────────────────
print("🤖 Entraînement du RandomForest...")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# ── 7. Évaluation ──────────────────────────────────────────────────────────────
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_proba)

print("\n📊 Résultats sur le jeu de test :")
print(f"   Accuracy : {accuracy:.4f}")
print(f"   ROC-AUC  : {roc_auc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["No default", "Default"]))

# ── 8. Sauvegarde ──────────────────────────────────────────────────────────────
print("💾 Sauvegarde du modèle et des encodeurs...")
joblib.dump(model,    MODEL_DIR / "model.pkl")
joblib.dump(encoders, MODEL_DIR / "encoders.pkl")

# Métadonnées lisibles par le serveur MCP
metadata = {
    "features":        FEATURES,
    "categorical_cols": categorical_cols,
    "target":          TARGET,
    "classes":         ["No default (0)", "Default (1)"],
    "accuracy":        round(accuracy, 4),
    "roc_auc":         round(roc_auc, 4),
    "n_train":         len(X_train),
    "n_test":          len(X_test),
}
with open(MODEL_DIR / "model_metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print("✅ Entraînement terminé !")
print(f"   → ml/model.pkl")
print(f"   → ml/encoders.pkl")
print(f"   → ml/model_metadata.json")
