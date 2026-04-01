from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.tree import DecisionTreeClassifier


class TreeHybridModel:
    name = "tree"

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        self.model = DecisionTreeClassifier(
            max_depth=kwargs.get("max_depth", 6),
            min_samples_leaf=kwargs.get("min_samples_leaf", 2),
            class_weight=kwargs.get("class_weight", "balanced"),
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        imp = self.model.feature_importances_
        return {name: float(val) for name, val in zip(feature_names, imp, strict=True)}

