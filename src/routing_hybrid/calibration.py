from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class ProbabilityCalibrator:
    def __init__(self, method: str = "none") -> None:
        self.method = method
        self.model: object | None = None

    def fit(self, scores: np.ndarray, y: np.ndarray) -> "ProbabilityCalibrator":
        x = scores.reshape(-1, 1)
        if self.method == "none":
            self.model = None
        elif self.method == "sigmoid":
            lr = LogisticRegression(max_iter=2000)
            lr.fit(x, y)
            self.model = lr
        elif self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(scores, y)
            self.model = iso
        else:
            raise ValueError(f"Unknown calibration method '{self.method}'")
        return self

    def transform(self, scores: np.ndarray) -> np.ndarray:
        if self.model is None:
            return np.clip(scores, 0.0, 1.0)
        if self.method == "sigmoid":
            return self.model.predict_proba(scores.reshape(-1, 1))[:, 1]  # type: ignore[union-attr]
        if self.method == "isotonic":
            return self.model.predict(scores)  # type: ignore[union-attr]
        return np.clip(scores, 0.0, 1.0)

