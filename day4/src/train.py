"""
Day 4 — MLOps: Training script with MLflow experiment tracking.

This script trains a LightGBM classifier on the Adult Income dataset and logs
the run to MLflow: parameters, metrics, and the serialised model artifact.
It is designed to be run both from the command line (standalone) and imported
as a module by the Jupyter notebooks.

Usage (command line):
    # Start MLflow server first (separate terminal):
    #   mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///day4/mlflow.db --default-artifact-root ./day4/mlartifacts
    python day4/src/train.py --experiment-name adult-income-lgbm

Usage (from notebook):
    from day4.src.train import train_and_log
    run_id, metrics = train_and_log(params={"n_estimators": 300, ...})
"""

from __future__ import annotations

import argparse
from pathlib import Path

import logging
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.calibration import calibration_curve
from sklearn.metrics import accuracy_score, brier_score_loss, f1_score, roc_auc_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

logger = logging.getLogger(__name__)

# ── Constants — same dataset conventions as Days 1–2 ─────────────────────────
SEED = 42
np.random.seed(SEED)

DATA_PATH = Path("day1/generated/adult_income_issues.csv")
TARGET_COL = "class"          # raw label: "<=50K" or ">50K"
TARGET_BIN_COL = "target"     # 0 / 1 encoded
SPLIT_COL = "split"           # "train" or "test"

# Columns to exclude from features
ID_COLS = ["person_id"]
LEAKAGE_COLS = ["post_adjudication_risk_code"]
PROCESS_COLS = [
    "db_source_table", "db_etl_batch_id", "db_row_surrogate_key",
    "db_loaded_at_utc", "dataset_schema_version", "extract_country_code",
    "record_written_at", "dgp_regime",
    # Additional process / metadata columns
    "case_review_note", "constant_one", "db_row_surrogate_key",
]

EXCLUDE_COLS = set(
    ID_COLS + LEAKAGE_COLS + PROCESS_COLS + [TARGET_COL, TARGET_BIN_COL, SPLIT_COL]
)

MODEL_NAME = "adult-income-classifier"
OUTPUT_DIR = Path("day4/outputs")


# ── Data loading ──────────────────────────────────────────────────────────────

def load_data(data_path: str | Path = DATA_PATH) -> pd.DataFrame:
    """Load the Adult Income dataset and create a binary target column."""
    df = pd.read_csv(data_path)
    # Normalise whitespace in labels (the CSV sometimes has leading spaces)
    df[TARGET_COL] = df[TARGET_COL].str.strip()
    df[TARGET_BIN_COL] = (df[TARGET_COL] == ">50K").astype(int)
    logger.info("Loaded %s rows × %s cols from %s", len(df), df.shape[1], data_path)
    return df


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split into train / val / test sets.

    - Test  = rows where SPLIT_COL == "test"  (held out from the start)
    - Train / Val = rows where SPLIT_COL == "train", split 80/20 by person_id
      to avoid data leakage from the same person appearing in both sets.
    """
    test_df = df[df[SPLIT_COL] == "test"].copy()
    train_all = df[df[SPLIT_COL] == "train"].copy()

    # Entity-aware split: group by person_id so all rows for one person stay together
    person_ids = train_all["person_id"].unique()
    train_ids, val_ids = train_test_split(person_ids, test_size=0.2, random_state=SEED)
    train_df = train_all[train_all["person_id"].isin(train_ids)].copy()
    val_df = train_all[train_all["person_id"].isin(val_ids)].copy()

    logger.info(
        "Split sizes — train: %s | val: %s | test: %s",
        len(train_df), len(val_df), len(test_df),
    )
    return train_df, val_df, test_df


# ── Feature engineering ───────────────────────────────────────────────────────

def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Classify columns into numeric and categorical feature sets.
    Excludes all ID, leakage, process, target and split columns.
    """
    feature_cols = [c for c in df.columns if c not in EXCLUDE_COLS]
    numeric_cols = [
        c for c in feature_cols
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]
    logger.info(
        "Features: %s total (%s numeric, %s categorical)",
        len(feature_cols), len(numeric_cols), len(categorical_cols),
    )
    return numeric_cols, categorical_cols


# ── Model pipeline ────────────────────────────────────────────────────────────

def build_pipeline(
    numeric_cols: list[str],
    categorical_cols: list[str],
    n_estimators: int = 300,
    learning_rate: float = 0.1,
    max_depth: int = 5,
    num_leaves: int = 31,
) -> Pipeline:
    """
    Build a sklearn Pipeline with:
    - ColumnTransformer: median imputation + OneHotEncoder (tree-friendly, no scaling)
    - LightGBM classifier
    """
    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]).set_output(transform="pandas")
    clf = lgb.LGBMClassifier(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        num_leaves=num_leaves,
        random_state=SEED,
        verbose=-1,
    )
    return Pipeline([("preprocessor", preprocessor), ("clf", clf)])


# ── Training & MLflow logging ─────────────────────────────────────────────────

def train_and_log(
    params: dict | None = None,
    run_name: str = "lgbm-run",
    experiment_name: str = "adult-income-lgbm",
    mlflow_uri: str = "http://127.0.0.1:5000",
    data_path: str | Path = DATA_PATH,
    register_model: bool = False,
    model_name: str = MODEL_NAME,
) -> tuple[str, dict]:
    """
    Train the LightGBM pipeline and log everything to MLflow.

    Returns
    -------
    run_id : str
        The MLflow run ID.
    metrics : dict
        Validation metrics {val_auc, val_f1, val_accuracy}.
    """
    if params is None:
        params = {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 5}

    # Connect to MLflow
    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    # Prepare data
    df = load_data(data_path)
    train_df, val_df, test_df = split_data(df)
    numeric_cols, categorical_cols = get_feature_columns(train_df)

    X_train = train_df[numeric_cols + categorical_cols]
    y_train = train_df[TARGET_BIN_COL]
    X_val = val_df[numeric_cols + categorical_cols]
    y_val = val_df[TARGET_BIN_COL]

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        # Log hyperparameters
        mlflow.log_params(params)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("n_val", len(val_df))
        mlflow.log_param("n_features", len(numeric_cols) + len(categorical_cols))

        # Train
        pipe = build_pipeline(numeric_cols, categorical_cols, **params)
        pipe.fit(X_train, y_train)

        # Evaluate
        val_proba = pipe.predict_proba(X_val)[:, 1]
        val_pred = pipe.predict(X_val)
        metrics = {
            "val_auc": float(roc_auc_score(y_val, val_proba)),
            "val_f1": float(f1_score(y_val, val_pred)),
            "val_accuracy": float(accuracy_score(y_val, val_pred)),
            "val_brier": float(brier_score_loss(y_val, val_proba)),
        }
        mlflow.log_metrics(metrics)

        # Log feature importances
        _log_feature_importance(pipe, numeric_cols + categorical_cols)

        # Log calibration data
        _log_calibration_data(y_val, val_proba)

        # Log the serialised model
        mlflow.sklearn.log_model(pipe, artifact_path="model")

        # Optionally register in the Model Registry
        if register_model:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri=model_uri, name=model_name)
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(model_name, "champion", mv.version)
            logger.info(
                "Registered model '%s' v%s with @champion alias from run %s",
                model_name, mv.version, run_id,
            )

        logger.info("Run ID: %s", run_id)
        logger.info(
            "val_auc=%.4f  val_f1=%.4f  val_accuracy=%.4f",
            metrics["val_auc"], metrics["val_f1"], metrics["val_accuracy"],
        )

    return run_id, metrics


# ── Artifact helpers ──────────────────────────────────────────────────────────

def _log_feature_importance(pipe: Pipeline, feature_names: list[str]) -> None:
    """Extract LightGBM feature importances and log as a JSON artifact."""
    import json, tempfile
    clf = pipe.named_steps["clf"]
    preprocessor = pipe.named_steps["preprocessor"]
    # Get transformed feature names (OHE expands categoricals)
    try:
        transformed_names = preprocessor.get_feature_names_out()
    except Exception:
        transformed_names = [f"f{i}" for i in range(clf.n_features_)]
    importances = dict(zip(
        [str(n) for n in transformed_names],
        [float(v) for v in clf.feature_importances_],
    ))
    # Sort descending
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(importances, f, indent=2)
        mlflow.log_artifact(f.name, artifact_path="diagnostics")


def _log_calibration_data(y_true, y_proba, n_bins: int = 10) -> None:
    """Compute calibration curve and log as a JSON artifact."""
    import json, tempfile
    prob_true, prob_pred = calibration_curve(y_true, y_proba, n_bins=n_bins)
    cal_data = {
        "prob_true": [float(v) for v in prob_true],
        "prob_pred": [float(v) for v in prob_pred],
        "n_bins": n_bins,
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(cal_data, f, indent=2)
        mlflow.log_artifact(f.name, artifact_path="diagnostics")


# ── Cross-validated training ─────────────────────────────────────────────────

def train_and_log_cv(
    params: dict | None = None,
    n_folds: int = 3,
    run_name: str = "lgbm-cv-run",
    experiment_name: str = "adult-income-lgbm",
    mlflow_uri: str = "http://127.0.0.1:5000",
    data_path: str | Path = DATA_PATH,
    register_model: bool = False,
    model_name: str = MODEL_NAME,
) -> tuple[str, dict]:
    """
    Train with GroupKFold cross-validation and log mean+std metrics to MLflow.

    Uses person_id as the group key to ensure entity-aware splits across folds.
    After CV, retrains on the full training set and logs the final model.

    Returns
    -------
    run_id : str
        The MLflow run ID.
    metrics : dict
        Aggregated metrics {val_auc_mean, val_auc_std, val_f1_mean, val_f1_std, ...}.
    """
    if params is None:
        params = {"n_estimators": 300, "learning_rate": 0.1, "max_depth": 5}

    mlflow.set_tracking_uri(mlflow_uri)
    mlflow.set_experiment(experiment_name)

    df = load_data(data_path)
    train_df, val_df, test_df = split_data(df)
    numeric_cols, categorical_cols = get_feature_columns(train_df)
    feature_cols = numeric_cols + categorical_cols

    # Use all training data (train + val) for CV
    train_all = pd.concat([train_df, val_df], ignore_index=True)
    X_all = train_all[feature_cols]
    y_all = train_all[TARGET_BIN_COL]
    groups = train_all["person_id"]

    gkf = GroupKFold(n_splits=n_folds)
    fold_metrics: dict[str, list[float]] = {
        "auc": [], "f1": [], "accuracy": [], "brier": [],
    }

    for fold_idx, (train_idx, val_idx) in enumerate(gkf.split(X_all, y_all, groups)):
        X_tr, X_vl = X_all.iloc[train_idx], X_all.iloc[val_idx]
        y_tr, y_vl = y_all.iloc[train_idx], y_all.iloc[val_idx]

        pipe = build_pipeline(numeric_cols, categorical_cols, **params)
        pipe.fit(X_tr, y_tr)

        vl_proba = pipe.predict_proba(X_vl)[:, 1]
        vl_pred = pipe.predict(X_vl)

        fold_metrics["auc"].append(float(roc_auc_score(y_vl, vl_proba)))
        fold_metrics["f1"].append(float(f1_score(y_vl, vl_pred)))
        fold_metrics["accuracy"].append(float(accuracy_score(y_vl, vl_pred)))
        fold_metrics["brier"].append(float(brier_score_loss(y_vl, vl_proba)))

        logger.info(
            "Fold %d/%d: AUC=%.4f  F1=%.4f",
            fold_idx + 1, n_folds,
            fold_metrics["auc"][-1], fold_metrics["f1"][-1],
        )

    # Aggregate
    metrics = {}
    for key, values in fold_metrics.items():
        metrics[f"val_{key}_mean"] = float(np.mean(values))
        metrics[f"val_{key}_std"] = float(np.std(values))

    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id

        mlflow.log_params(params)
        mlflow.log_param("seed", SEED)
        mlflow.log_param("n_folds", n_folds)
        mlflow.log_param("n_total_train", len(train_all))
        mlflow.log_param("n_features", len(feature_cols))
        mlflow.log_metrics(metrics)

        # Retrain on full training set for the final model
        pipe_final = build_pipeline(numeric_cols, categorical_cols, **params)
        pipe_final.fit(X_all, y_all)

        _log_feature_importance(pipe_final, feature_cols)
        _log_calibration_data(
            y_all,
            pipe_final.predict_proba(X_all)[:, 1],
        )
        mlflow.sklearn.log_model(pipe_final, artifact_path="model")

        if register_model:
            model_uri = f"runs:/{run_id}/model"
            mv = mlflow.register_model(model_uri=model_uri, name=model_name)
            client = mlflow.tracking.MlflowClient()
            client.set_registered_model_alias(model_name, "champion", mv.version)
            logger.info(
                "Registered model '%s' v%s with @champion alias from run %s",
                model_name, mv.version, run_id,
            )

        logger.info("Run ID: %s", run_id)
        logger.info(
            "CV results — AUC=%.4f±%.4f  F1=%.4f±%.4f",
            metrics["val_auc_mean"], metrics["val_auc_std"],
            metrics["val_f1_mean"], metrics["val_f1_std"],
        )

    return run_id, metrics


# ── CLI entry point ───────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train LightGBM and log to MLflow.")
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--learning-rate", type=float, default=0.1)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--num-leaves", type=int, default=31)
    p.add_argument("--mlflow-uri", type=str, default="http://127.0.0.1:5000")
    p.add_argument("--experiment-name", type=str, default="adult-income-lgbm")
    p.add_argument("--run-name", type=str, default="cli-run")
    p.add_argument("--data-path", type=str, default=str(DATA_PATH))
    p.add_argument(
        "--register", type=lambda x: x.lower() == "true", default=False,
        help="Register the model in the MLflow Model Registry (true/false).",
    )
    p.add_argument("--model-name", type=str, default=MODEL_NAME)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    params = {
        "n_estimators": args.n_estimators,
        "learning_rate": args.learning_rate,
        "max_depth": args.max_depth,
        "num_leaves": args.num_leaves,
    }
    run_id, metrics = train_and_log(
        params=params,
        run_name=args.run_name,
        experiment_name=args.experiment_name,
        mlflow_uri=args.mlflow_uri,
        data_path=args.data_path,
        register_model=args.register,
        model_name=args.model_name,
    )

    # Save run ID so downstream Makefile targets can use it
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    run_id_file = OUTPUT_DIR / "last_run_id.txt"
    run_id_file.write_text(run_id)
    logger.info("Run ID saved to {}", run_id_file)
