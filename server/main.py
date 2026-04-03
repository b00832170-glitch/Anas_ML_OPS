"""
server/main.py - FastAPI REST API for the credit risk model
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ml.predict import predict, get_model_info

app = FastAPI(
    title="Credit Risk Prediction API",
    description="Predicts credit default risk using a RandomForest model.",
    version="1.0.0",
)


class LoanRequest(BaseModel):
    person_age:                  int   = Field(..., example=25, ge=18, le=100)
    person_income:               float = Field(..., example=50000)
    person_home_ownership:       str   = Field(..., example="RENT")
    person_emp_length:           float = Field(..., example=3.0)
    loan_intent:                 str   = Field(..., example="PERSONAL")
    loan_grade:                  str   = Field(..., example="B")
    loan_amnt:                   float = Field(..., example=10000)
    loan_int_rate:               float = Field(..., example=12.5)
    loan_percent_income:         float = Field(..., example=0.20)
    cb_person_default_on_file:   str   = Field(..., example="N")
    cb_person_cred_hist_length:  int   = Field(..., example=4)


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/model-info")
def model_info():
    try:
        return get_model_info()
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/predict")
def predict_endpoint(req: LoanRequest):
    try:
        result = predict(req.model_dump())
        return result
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run first: python ml/train.py",
        )