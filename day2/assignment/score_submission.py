#!/usr/bin/env python3
"""
Days 1-2 assignment - submission scorer.

Scores a predictions file against a labels file with ROC-AUC, the same way
the in-course competition (day2/homework_credit_competition.ipynb) was scored.

Submission format (CSV):
    row_id,predicted_probability
    96,0.1234
    ...

- must contain exactly one row per row_id in the labels file
- predicted_probability must be a float in [0, 1]

Usage (instructor - private labels file):
    python day2/assignment/score_submission.py \
        --submission path/to/submission.csv \
        --labels day2/files/out/holdout_labels.csv

Usage (student self-check on your own validation split):
    python day2/assignment/score_submission.py \
        --submission my_val_predictions.csv \
        --labels my_val_labels.csv

The labels file must be a CSV with columns: row_id, target.
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd
from sklearn.metrics import roc_auc_score


def fail(msg: str) -> None:
    print(f"INVALID SUBMISSION: {msg}", file=sys.stderr)
    sys.exit(1)


def validate_and_score(submission_path: str, labels_path: str) -> float:
    sub = pd.read_csv(submission_path)
    labels = pd.read_csv(labels_path)

    expected_cols = {"row_id", "predicted_probability"}
    if not expected_cols.issubset(sub.columns):
        fail(f"submission must have columns {sorted(expected_cols)}, "
             f"got {sorted(sub.columns)}")
    if not {"row_id", "target"}.issubset(labels.columns):
        fail(f"labels file must have columns ['row_id', 'target'], "
             f"got {sorted(labels.columns)}")

    if sub["row_id"].duplicated().any():
        n = int(sub["row_id"].duplicated().sum())
        fail(f"{n} duplicate row_id values")

    missing = set(labels["row_id"]) - set(sub["row_id"])
    extra = set(sub["row_id"]) - set(labels["row_id"])
    if missing:
        fail(f"{len(missing)} row_ids missing from submission "
             f"(e.g. {sorted(missing)[:5]})")
    if extra:
        fail(f"{len(extra)} unexpected row_ids in submission "
             f"(e.g. {sorted(extra)[:5]})")

    preds = pd.to_numeric(sub["predicted_probability"], errors="coerce")
    if preds.isna().any():
        fail(f"{int(preds.isna().sum())} non-numeric or missing predictions")
    if not preds.between(0, 1).all():
        fail("predicted_probability values must be in [0, 1]")

    merged = labels.merge(sub, on="row_id", how="left")
    auc = roc_auc_score(merged["target"], merged["predicted_probability"])
    return float(auc)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score a submission with ROC-AUC.")
    parser.add_argument("--submission", required=True,
                        help="CSV with columns: row_id, predicted_probability")
    parser.add_argument("--labels", required=True,
                        help="CSV with columns: row_id, target")
    args = parser.parse_args()

    auc = validate_and_score(args.submission, args.labels)
    print(f"ROC-AUC: {auc:.4f}")


if __name__ == "__main__":
    main()
