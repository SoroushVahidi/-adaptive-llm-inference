from __future__ import annotations

import json

from src.datasets.bbh import load_bbh_records
from src.utils.answer_extraction import normalize_text_answer


def test_bbh_loader_local_sample_schema() -> None:
    rows = load_bbh_records(allow_external=False, max_samples=5)
    assert len(rows) == 5
    r0 = rows[0]
    assert r0["dataset"] == "bbh"
    assert r0["answer_format"] == "text"
    assert isinstance(r0["metadata"], dict)
    assert "task" in r0["metadata"]


def test_bbh_subset_is_deterministic_with_seed() -> None:
    a = load_bbh_records(allow_external=False, max_samples=8, seed=21)
    b = load_bbh_records(allow_external=False, max_samples=8, seed=21)
    assert [x["question_id"] for x in a] == [x["question_id"] for x in b]


def test_text_normalization_helper() -> None:
    assert normalize_text_answer("  The Answer.  ") == "the answer"


def test_bbh_falls_back_to_sample_when_normalized_missing(tmp_path) -> None:
    sample = tmp_path / "bbh_sample.jsonl"
    sample.write_text(
        json.dumps(
            {
                "dataset": "bbh",
                "question_id": "bbh_s_0",
                "question": "q",
                "options": None,
                "answer": "x",
                "answer_format": "text",
                "category": "task",
                "task": "task",
                "source_split": "test",
                "metadata": {"task": "task"},
            }
        )
        + "\n"
    )
    rows = load_bbh_records(
        normalized_path=tmp_path / "missing.jsonl",
        sample_path=sample,
        allow_external=False,
    )
    assert rows[0]["question_id"] == "bbh_s_0"
