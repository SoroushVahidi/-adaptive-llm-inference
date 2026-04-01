from __future__ import annotations

import pandas as pd

from scripts.build_routing_ml_dataset_expanded import _check_complete_action_outcomes, _resolve_sources
from scripts.build_routing_ml_dataset import build_dataset


def test_expanded_dataset_invariants_and_complete_action_outcomes() -> None:
    regime_files, blockers = _resolve_sources()
    assert not blockers
    result = build_dataset(regime_files)
    df = result.dataset

    assert len(df) >= 400
    assert not df.duplicated(subset=["regime", "prompt_id"]).any()
    assert df.groupby(["regime", "prompt_id"])["split"].nunique().max() == 1

    actions = result.report["actions_included"]
    assert set(df["best_action_label"]).issubset(set(actions))
    assert _check_complete_action_outcomes(df, actions) == []

    # Ensure all four canonical regimes are present.
    assert set(df["regime"]) == {
        "gsm8k_random_100",
        "hard_gsm8k_100",
        "hard_gsm8k_b2",
        "math500_100",
    }

    # Check split file-like projection is one row per prompt/regime.
    splits = df[["prompt_id", "regime", "split"]].copy()
    assert len(splits) == len(df)
