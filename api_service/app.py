"""Prediction API: a small FastAPI service that wraps the trained model.

Start it from the project root (the folder that contains manage.py):
    uvicorn api_service.app:app --port 8001

Interactive documentation: http://127.0.0.1:8001/docs
"""

from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from ml import config as cfg
from predictor import ml_service

app = FastAPI(
    title="Term Deposit Prediction API",
    description="Returns the probability that a bank client subscribes to a term deposit.",
    version="1.0",
)


class ClientFeatures(BaseModel):
    """The 11 inputs of the model."""

    age: float = Field(..., ge=0, le=120, examples=[35])
    job: str = Field(..., examples=["admin."])
    marital: str = Field(..., examples=["single"])
    education: str = Field(..., examples=["high.school"])
    housing: str = Field(..., examples=["yes"])
    loan: str = Field(..., examples=["no"])
    contact: str = Field(..., examples=["cellular"])
    month: str = Field(..., examples=["may"])
    campaign: int = Field(..., ge=0, examples=[2])
    previous: int = Field(..., ge=0, examples=[0])
    poutcome: str = Field(..., examples=["nonexistent"])


class PredictionResponse(BaseModel):
    prediction: Literal["yes", "no"]
    probability: float


@app.get("/health")
def health():
    """Tells the caller that the service is running and whether the model file exists."""
    return {"status": "ok", "model_loaded": cfg.MODEL_PATH.exists()}


@app.post("/predict", response_model=PredictionResponse)
def predict(features: ClientFeatures):
    """Predict whether a client will subscribe to a term deposit."""
    try:
        return ml_service.predict(features.model_dump())
    except ml_service.ModelNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ml_service.PredictionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
