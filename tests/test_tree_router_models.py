from __future__ import annotations

import numpy as np

from src.routing_hybrid.tree_router.models import make_tree_model, predict_score


def test_tree_family_models_fit_predict_binary() -> None:
    X = np.array([[0.0], [1.0], [0.1], [0.9], [0.2], [0.8]], dtype=float)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    for m in ["decision_tree", "bagging_tree", "random_forest", "gradient_boosting", "adaboost"]:
        model = make_tree_model(m, task_type="success_binary", seed=1, params={"n_estimators": 20})
        model.fit(X, y)
        p = predict_score(model, X, task_type="success_binary")
        assert len(p) == len(X)
