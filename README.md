# Anas_ML_OPS — Credit Risk Prediction + MCP Server

Modèle de prédiction de risque de défaut de crédit, exposé via un serveur MCP pour être utilisé par un LLM (Claude).

## Architecture

```
Dataset CSV → Entraînement (scikit-learn) → model.pkl
                                                 ↓
                                          FastAPI /predict
                                                 ↓
                                         Serveur MCP (tool)
                                                 ↓
                                      LLM (Claude Desktop)
```

## Installation

```bash
pip install -r requirements.txt
```

## Utilisation

### 1. Entraîner le modèle
```bash
python ml/train.py
```

### 2. Lancer l'API REST (optionnel, pour tester)
```bash
uvicorn server.main:app --reload
# Doc disponible sur http://localhost:8000/docs
```

### 3. Lancer le serveur MCP
```bash
python server/mcp_server.py
```

### 4. Connecter à Claude Desktop

Dans `claude_desktop_config.json` :
```json
{
  "mcpServers": {
    "credit-risk": {
      "command": "python",
      "args": ["/chemin/vers/Anas_ML_OPS/server/mcp_server.py"]
    }
  }
}
```

## Dataset

- **Source** : `data/credit_risk_dataset.csv`
- **Target** : `loan_status` (0 = pas de défaut, 1 = défaut)
- **Modèle** : RandomForestClassifier

## Features

| Feature | Type | Description |
|---|---|---|
| person_age | int | Âge de l'emprunteur |
| person_income | float | Revenu annuel |
| person_home_ownership | str | RENT / OWN / MORTGAGE / OTHER |
| person_emp_length | float | Ancienneté emploi (années) |
| loan_intent | str | Objet du prêt |
| loan_grade | str | Grade du prêt (A-G) |
| loan_amnt | float | Montant du prêt |
| loan_int_rate | float | Taux d'intérêt (%) |
| loan_percent_income | float | Ratio prêt/revenu |
| cb_person_default_on_file | str | Défaut historique (Y/N) |
| cb_person_cred_hist_length | int | Longueur historique crédit |
