from __future__ import annotations

from itertools import product
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, mean_squared_error, roc_auc_score

from src.routing_hybrid.tree_router.models import make_tree_model, predict_score


def _score_target(y_true: np.ndarray, pred: np.ndarray, task_type: str) -> float:
    if task_type == "utility_regression":
        return -float(mean_squared_error(y_true, pred))
    # binary classification: prefer roc_auc then pr_auc then -brier
    if len(set(y_true.tolist())) < 2:
        return float(-brier_score_loss(y_true, np.clip(pred, 0.0, 1.0)))
    return float(roc_auc_score(y_true, pred) + average_precision_score(y_true, pred) - brier_score_loss(y_true, np.clip(pred, 0.0, 1.0)))


def hyperparameter_search(
    model_type: str,
    task_type: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
    grid: dict[str, list[Any]],
) -> tuple[Any, dict[str, Any], list[dict[str, Any]]]:
    keys = sorted(grid.keys())
    combos = [dict(zip(keys, vals, strict=True)) for vals in product(*[grid[k] for k in keys])] if keys else [{}]
    results: list[dict[str, Any]] = []
    best_model = None
    best_cfg: dict[str, Any] = {}
    best_score = -1e18
    for cfg in combos:
        model = make_tree_model(model_type=model_type, task_type=task_type, seed=seed, params=cfg)
        model.fit(X_train, y_train)
        pred = predict_score(model, X_val, task_type=task_type)
        score = _score_target(y_val, pred, task_type=task_type)
        row = {"model_type": model_type, "task_type": task_type, "score": score, **cfg}
        results.append(row)
        if score > best_score:
            best_score = score
            best_model = model
            best_cfg = cfg
    if best_model is None:
        raise RuntimeError("Hyperparameter search failed to produce a model.")
    return best_model, best_cfg, results

