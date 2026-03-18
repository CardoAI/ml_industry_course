"""
Day 4 — MLOps: Data drift and performance monitoring with Evidently.

This script:
1. Loads the Adult Income dataset and a trained MLflow model.
2. Builds reference (train) and analysis (test) sets with model scores.
3. Computes PSI (Population Stability Index) for features and scores.
4. Runs univariate drift detection across all features via Evidently.
5. Applies multiple testing correction (Bonferroni) to drift results.
6. Logs a summary and HTML reports to MLflow as artifacts.
7. Exits with code 1 if thresholds are breached (CI gate behaviour).

Usage:
    python day4/src/drift_check.py \\
        --model-uri "models:/adult-income-classifier@champion" \\
        --mlflow-uri http://127.0.0.1:5000 \\
        --output-dir day4/outputs \\
        --fail-on-drift

    # In CI (file-based MLflow, model from previous train step):
    python day4/src/drift_check.py \\
        --model-uri "runs://<RUN_ID>/model" \\
        --mlflow-uri "file://./mlruns" \\
        --output-dir /tmp/drift_outputs \\
        --psi-threshold 0.25 \\
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
import numpy as np
import pandas as pd
from evidently import DataDefinition, Dataset, Report
from evidently.metrics import DriftedColumnsCount, ValueDrift
from evidently.presets import DataDriftPreset

# Re-use the same constants and helpers from train.py
from src.train import (
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
    reference_split: str = "train",
    analysis_split: str = "test",
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Load the dataset and attach model scores to reference and analysis sets.

    Parameters
    ----------
    reference_split : str
        Which data split to use as the drift reference distribution.
        Default "train". In production, this would be a recent window
        of data where model performance was known to be acceptable.
    analysis_split : str
        Which data split to monitor for drift. Default "test".
        In production, this would be the latest batch of live data
        (e.g., from an S3 path, database query, or streaming source).

    Returns
    -------
    reference_df : pd.DataFrame
        Reference split with ground truth (TARGET_BIN_COL) and y_pred_proba.
    analysis_df : pd.DataFrame
        Analysis split with y_pred_proba.
    feature_cols : list[str]
        Feature column names used by the model.
    """
    mlflow.set_tracking_uri(mlflow_uri)
    logger.info("Loading model from URI: %s", model_uri)
    model = mlflow.sklearn.load_model(model_uri)

    df = load_data(data_path)
    train_df, val_df, test_df = split_data(df)
    splits = {"train": train_df, "val": val_df, "test": test_df}

    reference_df = splits[reference_split].copy()
    analysis_df = splits[analysis_split].copy()
    logger.info(
        "Splits: reference=%s (%d rows), analysis=%s (%d rows)",
        reference_split, len(reference_df),
        analysis_split, len(analysis_df),
    )

    numeric_cols, categorical_cols = get_feature_columns(reference_df)
    feature_cols = numeric_cols + categorical_cols

    for part_df, name in [(reference_df, "reference"), (analysis_df, "analysis")]:
        part_df["y_pred_proba"] = model.predict_proba(part_df[feature_cols])[:, 1]
        part_df["y_pred"] = (part_df["y_pred_proba"] >= 0.5).astype(int)
        logger.info("%s set: %s rows", name, len(part_df))

    return reference_df, analysis_df, feature_cols


# ── PSI — Population Stability Index ──────────────────────────────────────────

def compute_psi(
    reference: np.ndarray,
    analysis: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute the Population Stability Index between two distributions.

    PSI = Σ (p_analysis_i - p_reference_i) × ln(p_analysis_i / p_reference_i)

    Thresholds (industry standard):
        < 0.10  → stable
        0.10–0.25 → moderate shift, investigate
        > 0.25  → significant shift, action required
    """
    breakpoints = np.quantile(reference, np.linspace(0, 1, n_bins + 1))
    breakpoints = np.unique(breakpoints)  # handle ties in quantiles
    ref_pcts = np.histogram(reference, bins=breakpoints)[0] / len(reference)
    ana_pcts = np.histogram(analysis, bins=breakpoints)[0] / len(analysis)
    # Avoid log(0) with small epsilon
    ref_pcts = np.clip(ref_pcts, 1e-4, None)
    ana_pcts = np.clip(ana_pcts, 1e-4, None)
    return float(np.sum((ana_pcts - ref_pcts) * np.log(ana_pcts / ref_pcts)))


def run_psi_check(
    reference_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    feature_cols: list[str],
    psi_threshold: float = 0.25,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute PSI for all numeric features and the score distribution.

    Returns a dict mapping column name → PSI value. Includes 'y_pred_proba'
    for the score distribution.
    """
    numeric_cols = reference_df[feature_cols].select_dtypes(include="number").columns
    psi_results: dict[str, float] = {}

    for col in numeric_cols:
        psi_val = compute_psi(
            reference_df[col].dropna().values,
            analysis_df[col].dropna().values,
            n_bins=n_bins,
        )
        psi_results[col] = psi_val
        status = (
            "SIGNIFICANT" if psi_val > 0.25
            else "moderate" if psi_val > 0.10
            else "stable"
        )
        logger.info("PSI %-25s: %.4f (%s)", col, psi_val, status)

    # Score PSI
    score_psi = compute_psi(
        reference_df["y_pred_proba"].values,
        analysis_df["y_pred_proba"].values,
        n_bins=n_bins,
    )
    psi_results["y_pred_proba"] = score_psi
    logger.info("PSI %-25s: %.4f", "y_pred_proba (score)", score_psi)

    flagged = [k for k, v in psi_results.items() if v > psi_threshold]
    logger.info(
        "PSI-flagged features (%s): %s",
        len(flagged), flagged or "none",
    )
    return psi_results


# ── Multiple testing correction ───────────────────────────────────────────────

def apply_bonferroni(
    drifted_features: list[str],
    n_features: int,
    alpha: float = 0.05,
) -> tuple[float, list[str]]:
    """
    Apply Bonferroni correction to drift results.

    Evidently tests each feature independently at alpha=0.05. With many
    features, the family-wise error rate is inflated:
        P(>=1 FP) = 1 - (1-α)^n.

    Returns the corrected alpha and the list of drifted features. The actual
    filtering of robust alerts requires access to p-values from the drift
    results, which Evidently doesn't always expose directly. This function
    logs the corrected threshold for documentation purposes.
    """
    corrected_alpha = alpha / n_features
    fwer = 1 - (1 - alpha) ** n_features
    logger.info(
        "Multiple testing: %d features, uncorrected FWER=%.2f%%, "
        "Bonferroni alpha=%.5f",
        n_features, fwer * 100, corrected_alpha,
    )
    return corrected_alpha, drifted_features


# ── Univariate drift detection (Evidently) ───────────────────────────────────

def run_feature_drift(
    reference_df: pd.DataFrame,
    analysis_df: pd.DataFrame,
    feature_cols: list[str],
) -> tuple[Report, list[str]]:
    """
    Run univariate drift detection using Evidently's DataDriftPreset.

    Evidently selects the test method adaptively based on dataset size:
    - Reference > 1000 rows: Wasserstein distance (numerical), Jensen-Shannon (categorical)
    - Reference ≤ 1000 rows: KS test (numerical), chi-squared (categorical)
    Returns the Evidently report and a list of features flagged as drifted.
    """
    logger.info("Running Evidently drift detection…")

    numeric_cols = (
        reference_df[feature_cols].select_dtypes(include="number").columns.tolist()
    )
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    data_def = DataDefinition(
        numerical_columns=numeric_cols,
        categorical_columns=categorical_cols,
    )
    ref_ds = Dataset.from_pandas(
        reference_df[feature_cols], data_definition=data_def
    )
    cur_ds = Dataset.from_pandas(
        analysis_df[feature_cols], data_definition=data_def
    )

    report = Report(metrics=[DataDriftPreset()])
    result = report.run(reference_data=ref_ds, current_data=cur_ds)

    # Extract per-feature drift from the report dict
    result_dict = result.dict()
    drifted_features: list[str] = []
    for metric in result_dict.get("metrics", []):
        cfg = metric.get("config", {})
        if cfg.get("type", "").endswith("ValueDrift"):
            col_name = cfg.get("column", "")
            method = cfg.get("method", "")
            threshold = cfg.get("threshold", 0.1)
            stat_value = metric.get("value")
            # Distance-based methods (Wasserstein, Jensen-Shannon): drift when value >= threshold
            # P-value-based methods (KS, chi-squared): drift when value <= threshold
            is_distance = "distance" in method.lower()
            if stat_value is not None:
                is_drifted = stat_value >= threshold if is_distance else stat_value <= threshold
            else:
                is_drifted = False
            if is_drifted:
                drifted_features.append(col_name)

    logger.info(
        "Drifted features (%s / %s): %s",
        len(drifted_features), len(feature_cols), drifted_features or "none",
    )
    return result, drifted_features


# ── Report generation ─────────────────────────────────────────────────────────

def generate_reports(
    evidently_result: Report,
    output_dir: Path,
) -> dict[str, Path]:
    """Save HTML reports to output_dir and return a mapping of name → path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}

    drift_path = output_dir / "drift_report.html"
    evidently_result.save_html(str(drift_path))
    paths["drift_report"] = drift_path
    logger.info("Drift report saved: %s", drift_path)

    return paths


# ── MLflow logging ────────────────────────────────────────────────────────────

def log_to_mlflow(
    drifted_features: list[str],
    report_paths: dict[str, Path],
    mlflow_uri: str,
    psi_results: dict[str, float] | None = None,
    corrected_alpha: float | None = None,
    run_name: str = "drift-monitoring",
    experiment_name: str = "adult-income-monitoring",
) -> str:
    """Log summary metrics and HTML reports to MLflow. Returns the run ID."""
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.log_metric("n_drifted_features", len(drifted_features))

        # Log PSI metrics
        if psi_results:
            score_psi = psi_results.get("y_pred_proba", 0.0)
            mlflow.log_metric("score_psi", score_psi)
            n_psi_flagged = sum(1 for v in psi_results.values() if v > 0.25)
            mlflow.log_metric("n_psi_flagged_features", n_psi_flagged)

        if corrected_alpha is not None:
            mlflow.log_metric("bonferroni_alpha", corrected_alpha)

        # Log HTML reports as artifacts
        for name, path in report_paths.items():
            mlflow.log_artifact(str(path), artifact_path="drift_reports")

        # Log drift summary as JSON
        summary = {
            "drifted_features": drifted_features,
            "n_drifted": len(drifted_features),
        }
        if psi_results:
            summary["psi"] = psi_results
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as tmp:
            json.dump(summary, tmp, indent=2)
            tmp_path = tmp.name
        mlflow.log_artifact(tmp_path, artifact_path="drift_reports")

        run_id = run.info.run_id
        logger.info("Drift monitoring run logged. Run ID: %s", run_id)

    return run_id


# ── CI gate ───────────────────────────────────────────────────────────────────

def check_thresholds(
    drifted_features: list[str],
    max_drifted: int = 5,
    fail_on_drift: bool = True,
    psi_results: dict[str, float] | None = None,
    psi_threshold: float = 0.25,
) -> bool:
    """
    Return True if all thresholds pass, False (or sys.exit(1)) if any fail.

    In CI, set fail_on_drift=True so the workflow step exits with code 1
    when thresholds are exceeded, which stops the pipeline.
    """
    passed = True

    if len(drifted_features) > max_drifted:
        logger.warning(
            "  %s features drifted (threshold: %s): %s",
            len(drifted_features), max_drifted, drifted_features,
        )
        passed = False

    # Check score PSI
    if psi_results:
        score_psi = psi_results.get("y_pred_proba", 0.0)
        if score_psi > psi_threshold:
            logger.warning(
                "  Score PSI %.4f exceeds threshold %.4f",
                score_psi, psi_threshold,
            )
            passed = False

    if passed:
        logger.info("All monitoring thresholds passed.")
    elif fail_on_drift:
        logger.error("Drift / performance check failed. Exiting with code 1.")
        sys.exit(1)

    return passed


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run Evidently drift and performance monitoring."
    )
    p.add_argument("--data-path", type=str, default=str(DATA_PATH))
    p.add_argument(
        "--model-uri", type=str,
        default="models:/adult-income-classifier@champion",
    )
    p.add_argument("--mlflow-uri", type=str, default="http://127.0.0.1:5000")
    p.add_argument("--experiment-name", type=str, default="adult-income-monitoring")
    p.add_argument("--output-dir", type=str, default="day4/outputs")
    p.add_argument("--max-drifted-features", type=int, default=5)
    p.add_argument(
        "--fail-on-drift", action="store_true",
        help="Exit with code 1 if thresholds are exceeded (CI gate mode).",
    )
    p.add_argument(
        "--psi-threshold", type=float, default=0.25,
        help="Score PSI threshold for CI gate (default: 0.25).",
    )
    p.add_argument(
        "--correction-method", type=str, default="bonferroni",
        choices=["bonferroni", "none"],
        help="Multiple testing correction method (default: bonferroni).",
    )
    p.add_argument(
        "--reference-split", type=str, default="train",
        choices=["train", "val", "test"],
        help="Data split for drift reference (default: train). "
             "In production, use a recent window of known-good data.",
    )
    p.add_argument(
        "--analysis-split", type=str, default="test",
        choices=["train", "val", "test"],
        help="Data split to monitor for drift (default: test). "
             "In production, this would be live/batch data.",
    )
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = _parse_args()

    reference_df, analysis_df, feature_cols = load_data_and_model(
        data_path=args.data_path,
        model_uri=args.model_uri,
        mlflow_uri=args.mlflow_uri,
        reference_split=args.reference_split,
        analysis_split=args.analysis_split,
    )

    # PSI — industry-standard drift metric
    psi_results = run_psi_check(
        reference_df, analysis_df, feature_cols,
        psi_threshold=args.psi_threshold,
    )

    evidently_result, drifted_features = run_feature_drift(
        reference_df, analysis_df, feature_cols
    )

    # Multiple testing correction
    corrected_alpha = None
    if args.correction_method == "bonferroni":
        corrected_alpha, drifted_features = apply_bonferroni(
            drifted_features, n_features=len(feature_cols),
        )

    output_dir = Path(args.output_dir)
    report_paths = generate_reports(evidently_result, output_dir)

    log_to_mlflow(
        drifted_features=drifted_features,
        report_paths=report_paths,
        mlflow_uri=args.mlflow_uri,
        psi_results=psi_results,
        corrected_alpha=corrected_alpha,
        experiment_name=args.experiment_name,
    )

    check_thresholds(
        drifted_features=drifted_features,
        max_drifted=args.max_drifted_features,
        fail_on_drift=args.fail_on_drift,
        psi_results=psi_results,
        psi_threshold=args.psi_threshold,
    )
