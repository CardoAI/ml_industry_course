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

# Base image — slim variant keeps the image small (~150 MB vs ~1 GB for full)
# Pinned to 3.12 to match the training environment (reproducibility)
FROM python:3.12-slim

# All subsequent commands run relative to /app
WORKDIR /app

# LightGBM uses OpenMP for parallel tree building.
# libgomp1 provides the OpenMP runtime; without it, import lightgbm crashes.
# --no-install-recommends avoids pulling in unnecessary packages.
# rm -rf /var/lib/apt/lists/* shrinks the layer by removing the apt cache.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Install uv — the same package manager used for local dev and CI.
# Multi-stage COPY pulls the static binary from the official image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy dependency files FIRST (separate layer = Docker cache hit when only
# code changes, not deps — saves minutes on repeated builds).
COPY day4/pyproject.toml day4/uv.lock ./

# Install exact pinned versions from the lockfile.
#   --frozen       use uv.lock as-is, don't re-resolve (reproducible builds)
#   --no-dev       skip optional/dev deps (evidently, jupyter, matplotlib)
#   --no-install-project  don't install our package itself, just its deps
# This is the Docker equivalent of `make install`, but runtime-only.
RUN uv sync --frozen --no-dev --no-install-project

# Copy the FastAPI application code into the image.
# __init__.py is needed so Python treats src/ as a package.
COPY day4/src/predict.py src/predict.py
COPY day4/src/__init__.py src/__init__.py

# Copy the pre-extracted model artifact.
# `make docker-build` downloads this from MLflow before running `docker build`.
# The model is a directory containing MLmodel, model.pkl, and conda.yaml.
COPY day4/outputs/model/ /app/model/

# Tell predict.py to load the model from this local path.
# No MLflow server is needed at runtime — the model is already in the image.
ENV MODEL_URI="/app/model"

# The FastAPI service listens on port 8080
EXPOSE 8080

# Start the ASGI server. uvicorn serves the FastAPI app defined in src/predict.py.
# --host 0.0.0.0 makes it accessible from outside the container.
CMD ["uvicorn", "src.predict:app", "--host", "0.0.0.0", "--port", "8080"]
