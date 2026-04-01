from __future__ import annotations

import json

from src.datasets.mmlu_pro import load_mmlu_pro_records
from src.utils.answer_extraction import extract_mc_answer


def test_mmlu_pro_loader_local_sample_schema() -> None:
    rows = load_mmlu_pro_records(allow_external=False, max_samples=5)
    assert len(rows) == 5
    r0 = rows[0]
    assert r0["dataset"] == "mmlu_pro"
    assert r0["answer_format"] == "multiple_choice"
    assert isinstance(r0["options"], list)
    assert isinstance(r0["metadata"], dict)


def test_mmlu_pro_subset_is_deterministic_with_seed() -> None:
    a = load_mmlu_pro_records(allow_external=False, max_samples=8, seed=7)
    b = load_mmlu_pro_records(allow_external=False, max_samples=8, seed=7)
    assert [x["question_id"] for x in a] == [x["question_id"] for x in b]


def test_mmlu_pro_prefers_normalized_over_sample(tmp_path) -> None:
    normalized = tmp_path / "normalized.jsonl"
    sample = tmp_path / "sample.jsonl"
    normalized.write_text(
        json.dumps(
            {
                "dataset": "mmlu_pro",
                "question_id": "from_normalized",
                "question": "q",
                "options": ["a", "b"],
                "answer": "A",
                "answer_format": "multiple_choice",
                "category": "cat",
                "source_split": "test",
                "metadata": {},
            }
        )
        + "\n"
    )
    sample.write_text(
        json.dumps(
            {
                "dataset": "mmlu_pro",
                "question_id": "from_sample",
                "question": "q",
                "options": ["a", "b"],
                "answer": "B",
                "answer_format": "multiple_choice",
                "category": "cat",
                "source_split": "test",
                "metadata": {},
            }
        )
        + "\n"
    )
    rows = load_mmlu_pro_records(
        normalized_path=normalized,
        sample_path=sample,
        allow_external=False,
    )
    assert rows[0]["question_id"] == "from_normalized"


def test_mmlu_pro_mc_extraction_supports_extended_letters() -> None:
    assert extract_mc_answer("Final answer: (I)") == "I"
