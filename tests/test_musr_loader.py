from __future__ import annotations

from src.datasets.musr import load_musr_records


def test_musr_loader_local_sample_schema() -> None:
    rows = load_musr_records(allow_external=False, max_samples=5)
    assert len(rows) == 5
    r0 = rows[0]
    assert r0["dataset"] == "musr"
    assert r0["answer_format"] == "multiple_choice"
    assert r0["category"] in {"murder_mysteries", "object_placements", "team_allocation"}
    assert isinstance(r0["options"], list)
    assert "subtask" in r0["metadata"]


def test_musr_subset_is_deterministic_with_seed() -> None:
    a = load_musr_records(allow_external=False, max_samples=8, seed=13)
    b = load_musr_records(allow_external=False, max_samples=8, seed=13)
    assert [x["question_id"] for x in a] == [x["question_id"] for x in b]
