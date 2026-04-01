from __future__ import annotations

import numpy as np

from src.routing_hybrid.tree_router.tuning import hyperparameter_search


def test_hyperparameter_search_runs() -> None:
    X = np.array([[0.0], [1.0], [0.2], [0.8], [0.1], [0.9]], dtype=float)
    y = np.array([0, 1, 0, 1, 0, 1], dtype=int)
    model, cfg, rows = hyperparameter_search(
        model_type="decision_tree",
        task_type="success_binary",
        X_train=X,
        y_train=y,
        X_val=X,
        y_val=y,
        seed=1,
        grid={"max_depth": [2, 3], "min_samples_leaf": [1], "min_samples_split": [2]},
    )
    assert model is not None
    assert isinstance(cfg, dict)
    assert len(rows) == 2
