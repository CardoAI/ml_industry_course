# ──────────────────────────────────────────────────────────────────────────────
# Day 4 — MLOps: Prediction Service Container
#
# Packages the FastAPI prediction service together with a trained model
# artifact baked into the image.
#
# Build (from repo root — after running `make train`):
#   make -f day4/Makefile docker-build
#
# Run:
#   docker run --rm -p 8080:8080 adult-income-predictor:latest
#
# Test:
#   curl http://localhost:8080/health
#
# ── Production note ────────────────────────────────────────────────────────────
# Baking the model into the image is the course simplification.
# In production you would:
#   1. Point MLFLOW_TRACKING_URI at a shared remote server (S3, Azure, GCS).
#   2. Pass MODEL_URI as an env var at runtime rather than at build time.
#   3. Pull the model artifact from the registry on container startup.
# ──────────────────────────────────────────────────────────────────────────────

FROM python:3.12-slim

WORKDIR /app

# LightGBM needs libgomp for OpenMP support
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install only what the prediction service needs
RUN pip install --no-cache-dir \
    mlflow \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pandas \
    scikit-learn \
    lightgbm

# Copy the prediction service
COPY day4/src/predict.py src/predict.py
COPY day4/src/__init__.py src/__init__.py

# Copy the pre-extracted model artifact (extracted by `make docker-build`)
COPY day4/outputs/model/ /app/model/

# Load directly from local path — no MLflow server needed
ENV MODEL_URI="/app/model"

EXPOSE 8080

CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8080"]
