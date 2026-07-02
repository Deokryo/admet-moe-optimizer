"""Metric helpers for regression and binary classification."""

from __future__ import annotations

import math

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, mean_absolute_error, mean_squared_error, r2_score, roc_auc_score


def regression_metrics(y_true: list[float] | np.ndarray, y_pred: list[float] | np.ndarray) -> dict[str, float]:
    """Calculate MAE, RMSE, and R2."""
    true = np.asarray(y_true, dtype=float)
    pred = np.asarray(y_pred, dtype=float)
    return {
        "mae": float(mean_absolute_error(true, pred)),
        "rmse": float(math.sqrt(mean_squared_error(true, pred))),
        "r2": float(r2_score(true, pred)),
    }


def classification_metrics(y_true: list[float] | np.ndarray, logits: list[float] | np.ndarray) -> dict[str, float | None]:
    """Calculate AUROC, AUPRC, F1, and accuracy from logits."""
    true = np.asarray(y_true, dtype=float).astype(int)
    logit_arr = np.asarray(logits, dtype=float)
    probabilities = 1.0 / (1.0 + np.exp(-np.clip(logit_arr, -40.0, 40.0)))
    labels = (probabilities >= 0.5).astype(int)

    metrics: dict[str, float | None] = {
        "f1": float(f1_score(true, labels, zero_division=0)),
        "accuracy": float(accuracy_score(true, labels)),
    }
    try:
        metrics["auroc"] = float(roc_auc_score(true, probabilities))
    except ValueError:
        metrics["auroc"] = None
    try:
        metrics["auprc"] = float(average_precision_score(true, probabilities))
    except ValueError:
        metrics["auprc"] = None
    return metrics

