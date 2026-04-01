from __future__ import annotations

from src.datasets.strategyqa import load_strategyqa_records
from src.utils.answer_extraction import extract_boolean_answer, normalize_boolean_answer


def test_strategyqa_loader_local_sample_schema() -> None:
    rows = load_strategyqa_records(allow_external=False, max_samples=5)
    assert len(rows) == 5
    r0 = rows[0]
    assert r0["dataset"] == "strategyqa"
    assert r0["answer_format"] == "boolean"
    assert r0["answer"] in {"true", "false"}
    assert isinstance(r0["metadata"], dict)


def test_strategyqa_subset_is_deterministic_with_seed() -> None:
    a = load_strategyqa_records(allow_external=False, max_samples=8, seed=3)
    b = load_strategyqa_records(allow_external=False, max_samples=8, seed=3)
    assert [x["question_id"] for x in a] == [x["question_id"] for x in b]


def test_boolean_normalization_helpers() -> None:
    assert normalize_boolean_answer("Yes") == "true"
    assert normalize_boolean_answer("0") == "false"
    assert extract_boolean_answer("Final answer: TRUE.") == "true"
