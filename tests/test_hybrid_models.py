from __future__ import annotations

import numpy as np

from src.routing_hybrid.models.registry import make_model


def test_model_fit_predict_toy() -> None:
    X = np.array([[0.0], [1.0], [0.2], [0.9]], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)
    model = make_model("logistic", seed=1, model_params={"max_iter": 500})
    model.fit(X, y)
    p = model.predict_proba(X)
    assert len(p) == len(X)
    assert float(np.min(p)) >= 0.0
    assert float(np.max(p)) <= 1.0


def test_gradient_boosting_registration() -> None:
    X = np.array([[0.0], [1.0], [0.2], [0.9]], dtype=float)
    y = np.array([0, 1, 0, 1], dtype=int)
    model = make_model("gradient_boosting", seed=1, model_params={"n_estimators": 20})
    model.fit(X, y)
    p = model.predict_proba(X)
    assert len(p) == len(X)
