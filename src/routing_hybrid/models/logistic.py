from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression


class LogisticHybridModel:
    name = "logistic"

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        self.model = LogisticRegression(
            max_iter=int(kwargs.get("max_iter", 2000)),
            class_weight=kwargs.get("class_weight", "balanced"),
            random_state=seed,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        p = self.predict_proba(X)
        return p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        coef = np.abs(self.model.coef_[0])
        return {name: float(val) for name, val in zip(feature_names, coef, strict=True)}

