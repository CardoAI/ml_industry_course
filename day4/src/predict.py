"""
Day 4 — MLOps: FastAPI prediction service.

Loads a trained MLflow model at startup and serves single-row predictions.
Designed to run inside a Docker container (see day4/Dockerfile).

Usage (local, model from registry):
    # Requires MLflow tracking server to be running
    MODEL_URI="models:/adult-income-classifier/Production" \\
    MLFLOW_TRACKING_URI="http://127.0.0.1:5000" \\
    uvicorn day4.src.predict:app --host 0.0.0.0 --port 8080

Usage (local, model from run artifact, no tracking server):
    MODEL_URI="runs:/<RUN_ID>/model" \\
    MLFLOW_TRACKING_URI="file://./mlruns" \\
    uvicorn day4.src.predict:app --host 0.0.0.0 --port 8080

Usage (Docker — model baked into image):
    docker run -p 8080:8080 adult-income-predictor:latest

Test it:
    curl http://localhost:8080/health
    curl -X POST http://localhost:8080/predict \\
         -H "Content-Type: application/json" \\
         -d '{"age": 35, "workclass": "Private", "education": "Bachelors",
              "education_num": 13, "marital_status": "Married-civ-spouse",
              "occupation": "Prof-specialty", "relationship": "Husband",
              "race": "White", "sex": "Male", "capital_gain": 0,
              "capital_loss": 0, "hours_per_week": 45,
              "native_country": "United-States"}'
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from typing import Optional

import mlflow
import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# ── Configuration from environment variables ──────────────────────────────────
MODEL_URI: str = os.getenv(
    "MODEL_URI", "models:/adult-income-classifier/Production"
)
MLFLOW_TRACKING_URI: str = os.getenv(
    "MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"
)

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

# Global model reference — loaded once at startup
_model = None


# ── Request / response schemas ────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    """
    Feature vector for a single Adult Income prediction.

    All fields are Optional so students can experiment with partial inputs
    (the preprocessing pipeline handles missing values via SimpleImputer).
    """
    # Original Adult Income features
    age: Optional[float] = None
    workclass: Optional[str] = None
    education: Optional[str] = None
    education_num: Optional[float] = None
    marital_status: Optional[str] = None
    occupation: Optional[str] = None
    relationship: Optional[str] = None
    race: Optional[str] = None
    sex: Optional[str] = None
    capital_gain: Optional[float] = None
    capital_loss: Optional[float] = None
    hours_per_week: Optional[float] = None
    native_country: Optional[str] = None
    # KYC / risk flags (from the augmented dataset)
    kyc_name_mismatch_flag: Optional[float] = None
    kyc_address_mismatch_flag: Optional[float] = None
    id_document_expired_flag: Optional[float] = None
    manual_review_required_flag: Optional[float] = None
    watchlist_screening_hit_flag: Optional[float] = None
    email_bounce_last_30d_flag: Optional[float] = None
    device_risk_high_flag: Optional[float] = None
    duplicate_application_flag: Optional[float] = None
    income_proof_unreadable_flag: Optional[float] = None


class PredictionResponse(BaseModel):
    prediction: int          # 0 = income ≤50K, 1 = income >50K
    probability: float       # P(income >50K)
    label: str               # human-readable label
    model_uri: str           # which model version served this prediction


# ── Application lifecycle ─────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once when the server starts; release on shutdown."""
    global _model
    try:
        _model = mlflow.sklearn.load_model(MODEL_URI)
    except Exception as exc:
        # Don't crash on startup — the /health endpoint will report not loaded
        import logging
        logging.getLogger("predict").warning("Could not load model: %s", exc)
    yield
    _model = None


app = FastAPI(
    title="Adult Income Classifier",
    description=(
        "Day 4 MLOps demo — serves predictions from a LightGBM model "
        "trained on the UCI Adult Income dataset."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", tags=["ops"])
async def health():
    """Liveness / readiness probe."""
    return {"status": "ok", "model_loaded": _model is not None, "model_uri": MODEL_URI}


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
async def predict(request: PredictionRequest):
    """
    Return a binary income prediction and its probability.

    The model handles missing values internally (SimpleImputer in the pipeline),
    so you can omit optional fields and the service will still respond.
    """
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    # Convert request to a single-row DataFrame matching the training schema
    row = pd.DataFrame([request.model_dump()])

    probability = float(_model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)
    label = ">50K" if prediction == 1 else "<=50K"

    return PredictionResponse(
        prediction=prediction,
        probability=round(probability, 4),
        label=label,
        model_uri=MODEL_URI,
    )
