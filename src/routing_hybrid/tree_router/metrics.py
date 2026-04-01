from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    mean_squared_error,
    roc_auc_score,
)


def compute_prediction_metrics(
    y_true: np.ndarray,
    pred_score: np.ndarray,
    task_type: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    if task_type == "utility_regression":
        out["rmse"] = float(np.sqrt(mean_squared_error(y_true, pred_score)))
        return out

    pred_label = (pred_score >= 0.5).astype(int)
    out["accuracy"] = float(accuracy_score(y_true, pred_label))
    out["balanced_accuracy"] = float(balanced_accuracy_score(y_true, pred_label))
    out["f1"] = float(f1_score(y_true, pred_label, zero_division=0))
    out["brier"] = float(brier_score_loss(y_true, np.clip(pred_score, 1e-6, 1 - 1e-6)))
    out["log_loss"] = float(log_loss(y_true, np.clip(pred_score, 1e-6, 1 - 1e-6), labels=[0, 1]))
    if len(set(y_true.tolist())) >= 2:
        out["roc_auc"] = float(roc_auc_score(y_true, pred_score))
        out["pr_auc"] = float(average_precision_score(y_true, pred_score))
    return out

