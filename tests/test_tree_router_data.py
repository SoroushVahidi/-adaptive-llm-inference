from __future__ import annotations

from src.routing_hybrid.tree_router.data import build_feature_matrix, filter_rows, transform_feature_matrix


def test_per_regime_and_pooled_handling() -> None:
    rows = [
        {"prompt_id": "p1", "regime": "r1", "action_name": "a", "feat_x": "1.0"},
        {"prompt_id": "p2", "regime": "r2", "action_name": "b", "feat_x": "2.0"},
    ]
    r1 = filter_rows(rows, regimes=["r1"])
    assert len(r1) == 1
    X, _, meta = build_feature_matrix(rows, include_regime_indicator=True, include_action_indicator=True)
    Xt = transform_feature_matrix(rows, meta)
    assert X.shape == Xt.shape
