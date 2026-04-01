from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoostingHybridModel:
    name = "gradient_boosting"

    def __init__(self, seed: int = 42, **kwargs: Any) -> None:
        self.model = GradientBoostingClassifier(
            n_estimators=int(kwargs.get("n_estimators", 200)),
            learning_rate=float(kwargs.get("learning_rate", 0.05)),
            max_depth=int(kwargs.get("max_depth", 3)),
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

