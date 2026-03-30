from __future__ import annotations

import json

from src.data.hard_gsm8k_selection import select_hard_gsm8k_queries, write_hard_selection_artifacts


def test_select_hard_gsm8k_returns_sorted_subset(tmp_path) -> None:
    gsm = tmp_path / "gsm.jsonl"
    records = []
    for i in range(5):
        records.append(
            {
                "question_id": f"q{i}",
                "question": "x " * (i + 1) + f" and {i + 1} apples cost {i + 2} dollars total?",
                "gold_answer": str(i),
            }
        )
    gsm.write_text("\n".join(json.dumps(r) for r in records) + "\n")

    queries, rows, summary = select_hard_gsm8k_queries(
        gsm8k_data_file=gsm,
        subset_size=3,
        pool_size=None,
    )
    assert len(queries) == 3
    assert len(rows) == 3
    assert rows[0]["hardness_score"] >= rows[-1]["hardness_score"]
    assert summary["pool_size"] == 5

    paths = write_hard_selection_artifacts(rows, summary, tmp_path / "out")
    assert paths["csv"].exists()
    assert paths["json"].exists()
