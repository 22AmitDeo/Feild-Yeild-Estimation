"""
api/main.py - FastAPI inference endpoint for yield prediction.

Run with:
    uvicorn api.main:app --reload
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.predict import predict

app = FastAPI(
    title="Field Yield Estimation API",
    description="Predict agricultural field yield from soil and weather features.",
    version="1.0.0",
)


class FieldInput(BaseModel):
    soil_ph: float = Field(..., ge=4.0, le=9.0, example=6.5)
    organic_matter: float = Field(..., ge=0.5, le=5.0, example=2.1)
    sand_pct: float = Field(..., ge=10.0, le=80.0, example=35.0)
    temperature: float = Field(..., ge=15.0, le=40.0, example=28.0)
    humidity: float = Field(..., ge=30.0, le=100.0, example=70.0)
    rainfall: float = Field(..., ge=50.0, le=250.0, example=120.0)
    ndvi: float = Field(..., ge=0.0, le=1.0, example=0.65)


class PredictionOutput(BaseModel):
    predicted_yield: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionOutput)
def predict_yield(data: FieldInput):
    try:
        result = predict(data.model_dump())
        return PredictionOutput(predicted_yield=round(result, 2))
    except FileNotFoundError:
        raise HTTPException(
            status_code=503,
            detail="Model not found. Run `python src/train_model.py` first.",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
