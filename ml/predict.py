"""
predict.py - Fonction de prédiction réutilisable par le serveur MCP et FastAPI
"""

import json
import joblib
import numpy as np
from pathlib import Path
from typing import Any

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "ml"

# Chargement une seule fois au démarrage du serveur
_model    = None
_encoders = None
_metadata = None


def _load_artifacts():
    global _model, _encoders, _metadata
    if _model is None:
        _model    = joblib.load(MODEL_DIR / "model.pkl")
        _encoders = joblib.load(MODEL_DIR / "encoders.pkl")
        with open(MODEL_DIR / "model_metadata.json") as f:
            _metadata = json.load(f)


def predict(input_data: dict[str, Any]) -> dict[str, Any]:
    """
    Prédit le risque de défaut de crédit.

    Paramètres attendus (tous requis) :
        person_age               (int)    : âge de l'emprunteur
        person_income            (float)  : revenu annuel
        person_home_ownership    (str)    : RENT | OWN | MORTGAGE | OTHER
        person_emp_length        (float)  : ancienneté emploi en années
        loan_intent              (str)    : PERSONAL | EDUCATION | MEDICAL |
                                           VENTURE | HOMEIMPROVEMENT | DEBTCONSOLIDATION
        loan_grade               (str)    : A | B | C | D | E | F | G
        loan_amnt                (float)  : montant du prêt
        loan_int_rate            (float)  : taux d'intérêt (%)
        loan_percent_income      (float)  : ratio prêt / revenu (ex: 0.35)
        cb_person_default_on_file (str)   : Y | N
        cb_person_cred_hist_length (int)  : longueur historique crédit (années)

    Retourne :
        dict avec prediction, probability, risk_level, explanation
    """
    _load_artifacts()

    features = _metadata["features"]
    cat_cols  = _metadata["categorical_cols"]

    # Encodage des variables catégorielles
    processed = dict(input_data)
    for col in cat_cols:
        if col not in processed:
            raise ValueError(f"Champ manquant : '{col}'")
        le = _encoders[col]
        val = processed[col]
        if val not in le.classes_:
            raise ValueError(
                f"Valeur inconnue '{val}' pour '{col}'. "
                f"Valeurs acceptées : {list(le.classes_)}"
            )
        processed[col] = int(le.transform([val])[0])

    # Construction du vecteur de features dans le bon ordre
    try:
        X = np.array([[processed[f] for f in features]])
    except KeyError as e:
        raise ValueError(f"Champ manquant dans les données : {e}")

    # Prédiction
    pred    = int(_model.predict(X)[0])
    prob    = float(_model.predict_proba(X)[0][1])  # proba de défaut

    # Niveau de risque lisible
    if prob < 0.25:
        risk_level = "Faible"
    elif prob < 0.55:
        risk_level = "Modéré"
    elif prob < 0.80:
        risk_level = "Élevé"
    else:
        risk_level = "Très élevé"

    return {
        "prediction":        pred,
        "label":             "Défaut probable" if pred == 1 else "Pas de défaut",
        "probability_default": round(prob, 4),
        "risk_level":        risk_level,
        "model_accuracy":    _metadata["accuracy"],
        "model_roc_auc":     _metadata["roc_auc"],
    }


def get_model_info() -> dict:
    """Retourne les métadonnées du modèle."""
    _load_artifacts()
    return _metadata
