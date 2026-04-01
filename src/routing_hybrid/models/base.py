from __future__ import annotations

from typing import Protocol

import numpy as np


class HybridModel(Protocol):
    name: str

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    def predict_score(self, X: np.ndarray) -> np.ndarray:
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        ...

    def feature_importance(self, feature_names: list[str]) -> dict[str, float]:
        ...

