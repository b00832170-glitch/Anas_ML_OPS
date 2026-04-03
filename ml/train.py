"""
train.py - Training a prediction model of credit default
Target : loan_status (0 = no default, 1 = default)
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

# Ways
BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "credit_risk_dataset.csv"
MODEL_DIR = BASE_DIR / "ml"

# Loading
print("Loading dataset")
df = pd.read_csv(DATA_PATH)
print(f"   {df.shape[0]} rows, {df.shape[1]} columns")

# Cleaning
print("Cleaning")
df = df.dropna()

# Deleting obvious outliers (âge > 100, emp_length > 60)
df = df[df["person_age"] <= 100]
df = df[df["person_emp_length"] <= 60]

# Encoding categorical variables
print("Encoding categorical variables")
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

#Separating the features from the target
TARGET = "loan_status"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

print(f"   Features : {FEATURES}")
print(f"   Target distribution : {y.value_counts().to_dict()}")

# Splitting train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Training the model
print("Training Random Forest")
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=12,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)

# Evaluation
y_pred  = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc  = roc_auc_score(y_test, y_proba)

print("\nResults :")
print(f"   Accuracy : {accuracy:.4f}")
print(f"   ROC-AUC  : {roc_auc:.4f}")
print("\n" + classification_report(y_test, y_pred, target_names=["No default", "Default"]))

# Saving
print("Saving model and encoders...")
joblib.dump(model,    MODEL_DIR / "model.pkl")
joblib.dump(encoders, MODEL_DIR / "encoders.pkl")

# Visible data for MCP Server
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

print("Training done")
print(f"   → ml/model.pkl")
print(f"   → ml/encoders.pkl")
print(f"   → ml/model_metadata.json")
