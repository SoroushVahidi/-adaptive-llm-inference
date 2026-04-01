from __future__ import annotations

from scripts.build_routing_ml_dataset import REGIME_FILES, build_dataset


def test_build_routing_ml_dataset_basic_invariants() -> None:
    result = build_dataset(REGIME_FILES)
    df = result.dataset

    assert len(df) == 400
    assert {"train", "validation", "test"}.issubset(set(df["split"]))

    actions = set(result.report["actions_included"])
    assert set(df["best_action_label"]).issubset(actions)

    dup = df.duplicated(subset=["regime", "prompt_id"]).any()
    assert not dup

    split_leak = df.groupby(["regime", "prompt_id"])["split"].nunique().max()
    assert split_leak == 1
