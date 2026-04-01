from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    BaggingClassifier,
    BaggingRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def make_tree_model(
    model_type: str,
    task_type: str,
    seed: int,
    params: dict[str, Any] | None = None,
) -> Any:
    p = params or {}
    is_reg = task_type == "utility_regression"
    max_depth = p.get("max_depth", 6)
    min_samples_leaf = p.get("min_samples_leaf", 2)
    min_samples_split = p.get("min_samples_split", 2)
    if model_type == "decision_tree":
        if is_reg:
            return DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=seed,
            )
        return DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            class_weight=p.get("class_weight", "balanced"),
            random_state=seed,
        )
    if model_type == "bagging_tree":
        if is_reg:
            base = DecisionTreeRegressor(
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                random_state=seed,
            )
            return BaggingRegressor(
                estimator=base,
                n_estimators=int(p.get("n_estimators", 200)),
                random_state=seed,
            )
        base = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            class_weight=p.get("class_weight", "balanced"),
            random_state=seed,
        )
        return BaggingClassifier(
            estimator=base,
            n_estimators=int(p.get("n_estimators", 200)),
            random_state=seed,
        )
    if model_type == "random_forest":
        if is_reg:
            return RandomForestRegressor(
                n_estimators=int(p.get("n_estimators", 300)),
                max_depth=max_depth,
                min_samples_leaf=min_samples_leaf,
                min_samples_split=min_samples_split,
                max_features=p.get("max_features", "sqrt"),
                random_state=seed,
            )
        return RandomForestClassifier(
            n_estimators=int(p.get("n_estimators", 300)),
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            min_samples_split=min_samples_split,
            max_features=p.get("max_features", "sqrt"),
            class_weight=p.get("class_weight", "balanced"),
            random_state=seed,
        )
    if model_type == "gradient_boosting":
        if is_reg:
            return GradientBoostingRegressor(
                n_estimators=int(p.get("n_estimators", 200)),
                learning_rate=float(p.get("learning_rate", 0.05)),
                max_depth=max_depth,
                subsample=float(p.get("subsample", 1.0)),
                random_state=seed,
            )
        return GradientBoostingClassifier(
            n_estimators=int(p.get("n_estimators", 200)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            max_depth=max_depth,
            subsample=float(p.get("subsample", 1.0)),
            random_state=seed,
        )
    if model_type == "adaboost":
        if is_reg:
            base = DecisionTreeRegressor(max_depth=int(p.get("base_max_depth", 2)), random_state=seed)
            return AdaBoostRegressor(
                estimator=base,
                n_estimators=int(p.get("n_estimators", 200)),
                learning_rate=float(p.get("learning_rate", 0.05)),
                random_state=seed,
            )
        base = DecisionTreeClassifier(max_depth=int(p.get("base_max_depth", 1)), random_state=seed)
        return AdaBoostClassifier(
            estimator=base,
            n_estimators=int(p.get("n_estimators", 200)),
            learning_rate=float(p.get("learning_rate", 0.05)),
            random_state=seed,
        )
    raise ValueError(f"Unknown model_type '{model_type}'")


def predict_score(model: Any, X: np.ndarray, task_type: str) -> np.ndarray:
    if task_type == "utility_regression":
        return model.predict(X).astype(float)
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1].astype(float)
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def feature_importance(model: Any, feature_names: list[str]) -> dict[str, float]:
    if hasattr(model, "feature_importances_"):
        arr = model.feature_importances_
        return {name: float(v) for name, v in zip(feature_names, arr, strict=False)}
    if hasattr(model, "coef_"):
        arr = np.abs(model.coef_[0])
        return {name: float(v) for name, v in zip(feature_names, arr, strict=False)}
    return {name: 0.0 for name in feature_names}

