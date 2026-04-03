# Anas_ML_OPS — Credit Risk Prediction + LLM Assistant

A machine learning model that predicts credit default risk, connected to a Groq LLM (Llama 3.3) that answers questions in natural language.

## Architecture
```
Dataset CSV → Training (scikit-learn) → model.pkl
                                             ↓
                                      predict.py (ML tool)
                                             ↓
                                   Groq LLM (Llama 3.3-70b)
                                             ↓
                                  chat.py (natural language)
                                             ↓
                                          User
```

## Installation

**1. Clone the repository**
```bash
git clone https://github.com/b00832170-glitch/Anas_ML_OPS.git
cd Anas_ML_OPS
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model**
```bash
python ml/train.py
```

**4. Set up your Groq API key**

Create a free account at https://console.groq.com, generate an API key, then create a `.env` file:
```
GROQ_API_KEY=your_groq_api_key_here
```

**5. Start the AI Assistant**
```bash
python chat.py
```

## Example Questions
```
Is a 25-year-old renter with $50k income and a $10k personal loan grade B risky?
Analyze this borrower: 22 years old, $9k income, grade F loan of $35k, past default on file
What is the model accuracy?
```

## Model Performance

| Metric | Score |
|---|---|
| Accuracy | 92.96% |
| ROC-AUC | 92.46% |

## Dataset

- **Source** : data/credit_risk_dataset.csv
- **Target** : loan_status (0 = no default, 1 = default)
- **Model** : RandomForestClassifier
- **Size** : 32,581 rows

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

## Project Structure
```
Anas_ML_OPS/
├── data/
│   └── credit_risk_dataset.csv
├── ml/
│   ├── train.py
│   └── predict.py
├── server/
│   ├── main.py
│   └── mcp_server.py
├── chat.py
├── .env.example
├── requirements.txt
└── README.md
```
