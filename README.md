# Anas_ML_OPS — Credit Risk Prediction + MCP Server

A machine learning model that predicts credit default risk, exposed via an MCP server to be used by a LLM (Claude).

## Architecture


Dataset CSV → Training (scikit-learn) → model.pkl
                                                 ↓
                                          FastAPI /predict
                                                 ↓
                                         MCP Server (tool)
                                                 ↓
                                      LLM (Claude Desktop)
```

## Installation
```bash
pip install -r requirements.txt
```

## Usage

### 1. Train the model
```bash
python ml/train.py
```

### 2. Start the REST API (for testing)
```bash
uvicorn server.main:app --reload
# Documentation available at http://localhost:8000/docs
```

### 3. Start the MCP server
```bash
python server/mcp_server.py
```

### 4. Connect to Claude Desktop

In `claude_desktop_config.json` :
```json
{
  "mcpServers": {
    "credit-risk": {
      "command": "python",
      "args": ["/path/to/Anas_ML_OPS/server/mcp_server.py"]
    }
  }
}
```

## Dataset

- **Source** : `data/credit_risk_dataset.csv`
- **Target** : `loan_status` (0 = no default, 1 = default)
- **Model** : RandomForestClassifier
- **Size** : 32,581 rows

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 92.96% |
| ROC-AUC | 92.46% |

## Features

| Feature | Type | Description |
|---|---|---|
| person_age | int | Borrower age |
| person_income | float | Annual income |
| person_home_ownership | str | RENT / OWN / MORTGAGE / OTHER |
| person_emp_length | float | Employment length (years) |
| loan_intent | str | Loan purpose |
| loan_grade | str | Loan grade (A-G) |
| loan_amnt | float | Loan amount |
| loan_int_rate | float | Interest rate (%) |
| loan_percent_income | float | Loan/income ratio |
| cb_person_default_on_file | str | Historical default (Y/N) |
| cb_person_cred_hist_length | int | Credit history length |