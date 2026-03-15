"""
Day 4 — MLOps: Data drift and performance monitoring with NannyML.

This script:
1. Loads the Adult Income dataset and a trained MLflow model.
2. Builds reference (train) and analysis (test) sets with model scores.
3. Runs CBPE to estimate model performance without ground truth.
4. Runs univariate drift detection across all features.
5. Logs a summary and HTML reports to MLflow as artifacts.
6. Exits with code 1 if thresholds are breached (CI gate behaviour).

Usage:
    python day4/src/nannyml_check.py \\
        --model-uri "models:/adult-income-classifier/Production" \\
        --mlflow-uri http://127.0.0.1:5000 \\
        --output-dir day4/outputs \\
        --fail-on-drift

    # In CI (file-based MLflow, model from previous train step):
    python day4/src/nannyml_check.py \\
        --model-uri "runs://<RUN_ID>/model" \\
        --mlflow-uri "file://./mlruns" \\
        --output-dir /tmp/nannyml_outputs \\
        --auc-threshold 0.78 \\
        --fail-on-drift
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import logging
import mlflow
import mlflow.sklearn
import nannyml as nml
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# Re-use the same constants and helpers from train.py
from day4.src.train import (
    DATA_PATH,
    EXCLUDE_COLS,
    SEED,
    TARGET_BIN_COL,
    get_feature_columns,
    load_data,
    split_data,
)

logger = logging.getLogger(__name__)

np.random.seed(SEED)


# ── Data preparation ──────────────────────────────────────────────────────────

def load_data_and_model(
    data_path: str | Path,
    model_uri: str,
    mlflow_uri: str,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load the dataset and attach model scores to reference and analysis sets.

    Returns
    -------
    reference_df : pd.DataFrame
        Training split with ground truth (TARGET_BIN_COL) and y_pred_proba.
    analysis_df : pd.DataFrame
        Test split with y_pred_proba but *without* ground truth exposed to NannyML.
    feature_cols : list[str]
        Feature column names used by the model.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info("Loading model from URI: %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    df = load_data(data_path)
    train_df, _val_df, test_df = split_data(df)

    # Combine train + val as reference (more data → better CBPE calibration)
    reference_df = df[df["split"] == "train"].copy()
    analysis_df = test_df.copy()

    numeric_cols, categorical_cols = get_feature_columns(reference_df)
    feature_cols = numeric_cols + categorical_cols

    for part_df, name in [(reference_df, "reference"), (analysis_df, "analysis")]:
        part_df["y_pred_proba"] = model.predict_proba(part_df[feature_cols])[:, 1]
        part_df["y_pred"] = (part_df["y_pred_proba"] >= 0.5).astype(int)
        logger.info("%s set: %s rows", name, len(part_df))

    # Synthetic timestamp so NannyML can do chunk-based analysis
    reference_df["timestamp"] = pd.date_range(
        "2023-01-01", periods=len(reference_df), freq="h"
    )
    analysis_df["timestamp"] = pd.date_range(
        "2024-01-01", periods=len(analysis_df), freq="h"
    )

    return reference_df, analysis_df, feature_cols


# ── CBPE — performance estimation without ground truth ────────────────────────

def run_cbpe(
    reference_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    feature_cols: list[str],
    chunk_size: int = 400,
) -> tuple[nml.CBPE, float]:
    """
    Run Confidence-Based Performance Estimation (CBPE).

    CBPE learns the probability-to-performance mapping from the reference set
    (where ground truth is available), then applies it to the analysis set
    where ground truth may not yet exist.

    Returns the estimator and the mean estimated AUC on the analysis period.
    """
    logger.info("Running CBPE…")
    estimator = nml.CBPE(
        y_pred_proba="y_pred_proba",
        y_pred="y_pred",
        y_true=TARGET_BIN_COL,
        timestamp_column_name="timestamp",
        metrics=["roc_auc", "f1"],
        chunk_size=chunk_size,
        problem_type="binary_classification",
    )
    estimator.fit(reference_df)
    cbpe_results = estimator.estimate(analysis_df)

    results_df = cbpe_results.filter(period="analysis").to_df()
    estimated_auc = float(results_df["estimated_roc_auc"].mean())
    logger.info("Estimated AUC on analysis set: %.4f", estimated_auc)

    # Compare with actual AUC (we have ground truth in test set)
    actual_auc = roc_auc_score(analysis_df[TARGET_BIN_COL], analysis_df["y_pred_proba"])
    logger.info("Actual    AUC on analysis set: %.4f", actual_auc)
    logger.info("CBPE error: %.4f", abs(actual_auc - estimated_auc))

    return cbpe_results, estimated_auc


# ── Univariate drift detection ────────────────────────────────────────────────

def run_univariate_drift(
    reference_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    feature_cols: list[str],
    chunk_size: int = 400,
) -> tuple[nml.UnivariateDriftCalculator, list[str]]:
    """
    Run univariate drift detection.

    NannyML uses the KS test for continuous features and chi-squared for
    categorical ones.  Returns the calculator and a list of features flagged
    as drifted in at least one chunk.
    """
    logger.info("Running univariate drift detection…")
    calc = nml.UnivariateDriftCalculator(
        column_names=feature_cols,
        timestamp_column_name="timestamp",
        chunk_size=chunk_size,
    )
    calc.fit(reference_df)
    results = calc.calculate(analysis_df)

    results_df = results.filter(period="analysis").to_df()
    # Identify features where any chunk raised an alert
    alert_cols = [
        col for col in feature_cols
        if f"({col}, alert)" in results_df.columns
        and results_df[f"({col}, alert)"].any()
    ]
    logger.info(
        "Drifted features (%s / %s): %s",
        len(alert_cols), len(feature_cols), alert_cols or "none",
    )
    return results, alert_cols


# ── Report generation ─────────────────────────────────────────────────────────

def generate_reports(
    cbpe_results,
    univariate_results,
    output_dir: Path,
) -> dict[str, Path]:
    """Save HTML reports to output_dir and return a mapping of name → path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    cbpe_path = output_dir / "cbpe_report.html"
    cbpe_results.plot(kind="performance", metric="roc_auc").write_html(str(cbpe_path))
    paths["cbpe_report"] = cbpe_path
    logger.info("CBPE report saved: %s", cbpe_path)

    drift_path = output_dir / "drift_report.html"
    univariate_results.plot(kind="drift").write_html(str(drift_path))
    paths["drift_report"] = drift_path
    logger.info("Drift report saved: %s", drift_path)

    return paths


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(
    estimated_auc: float,
    drifted_features: list[str],
    report_paths: dict[str, Path],
    mlflow_uri: str,
    run_name: str = "nannyml-monitoring",
    experiment_name: str = "adult-income-monitoring",
) -> str:
    """Log summary metrics and HTML reports to MLflow. Returns the run ID."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metric("estimated_val_auc", estimated_auc)
        mlflow.log_metric("n_drifted_features", len(drifted_features))

        # Log HTML reports as artifacts
        for name, path in report_paths.items():
            mlflow.log_artifact(str(path), artifact_path="nannyml_reports")

        # Log drift summary as JSON
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(
                {"drifted_features": drifted_features, "n_drifted": len(drifted_features)},
                tmp, indent=2,
            )
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="nannyml_reports")

        run_id = run.info.run_id
        logger.info("NannyML run logged. Run ID: %s", run_id)

    return run_id


# ── CI gate ───────────────────────────────────────────────────────────────────

def check_thresholds(
    estimated_auc: float,
    drifted_features: list[str],
    auc_threshold: float = 0.80,
    max_drifted: int = 5,
    fail_on_drift: bool = True,
) -> bool:
    """
    Return True if all thresholds pass, False (or sys.exit(1)) if any fail.

    In CI, set fail_on_drift=True so the workflow step exits with code 1
    when thresholds are exceeded, which stops the pipeline.
    """
    passed = True

    if estimated_auc < auc_threshold:
        logger.warning(
            "⚠  Estimated AUC %.4f is below threshold %.4f", estimated_auc, auc_threshold
        )
        passed = False

    if len(drifted_features) > max_drifted:
        logger.warning(
            "⚠  %s features drifted (threshold: %s): %s",
            len(drifted_features), max_drifted, drifted_features,
        )
        passed = False

    if passed:
        logger.info("✓ All monitoring thresholds passed.")
    elif fail_on_drift:
        logger.error("Drift / performance check failed. Exiting with code 1.")
        sys.exit(1)

    return passed


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run NannyML drift and performance monitoring."
    )
    p.add_argument("--data-path", type=str, default=str(DATA_PATH))
    p.add_argument(
        "--model-uri", type=str,
        default="models:/adult-income-classifier/Production",
    )
    p.add_argument("--mlflow-uri", type=str, default="http://127.0.0.1:5000")
    p.add_argument("--experiment-name", type=str, default="adult-income-monitoring")
    p.add_argument("--output-dir", type=str, default="day4/outputs")
    p.add_argument("--auc-threshold", type=float, default=0.80)
    p.add_argument("--max-drifted-features", type=int, default=5)
    p.add_argument(
        "--fail-on-drift", action="store_true",
        help="Exit with code 1 if thresholds are exceeded (CI gate mode).",
    )
    p.add_argument("--chunk-size", type=int, default=400)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()

    reference_df, analysis_df, feature_cols = load_data_and_model(
        data_path=args.data_path,
        model_uri=args.model_uri,
        mlflow_uri=args.mlflow_uri,
    )

    cbpe_results, estimated_auc = run_cbpe(
        reference_df, analysis_df, feature_cols, chunk_size=args.chunk_size
    )
    univariate_results, drifted_features = run_univariate_drift(
        reference_df, analysis_df, feature_cols, chunk_size=args.chunk_size
    )

    output_dir = Path(args.output_dir)
    report_paths = generate_reports(cbpe_results, univariate_results, output_dir)

    log_to_mlflow(
        estimated_auc=estimated_auc,
        drifted_features=drifted_features,
        report_paths=report_paths,
        mlflow_uri=args.mlflow_uri,
        experiment_name=args.experiment_name,
    )

    check_thresholds(
        estimated_auc=estimated_auc,
        drifted_features=drifted_features,
        auc_threshold=args.auc_threshold,
        max_drifted=args.max_drifted_features,
        fail_on_drift=args.fail_on_drift,
    )
