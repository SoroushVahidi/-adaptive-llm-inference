from __future__ import annotations

import numpy as np


class DummyHybridModel:
    name = "dummy"

    def __init__(self, **kwargs: object) -> None:
        self.p_: float = 0.5

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.p_ = float(np.mean(y)) if len(y) else 0.5

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.full(shape=(len(X),), fill_value=self.p_, dtype=float)

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        return {name: 0.0 for name in feature_names}

