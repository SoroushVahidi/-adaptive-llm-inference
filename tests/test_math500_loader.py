from __future__ import annotations

import json

from src.datasets.math500 import load_math500


def test_load_math500_from_local_file_normalizes_shape_and_answers(tmp_path) -> None:
    data_file = tmp_path / "math500.json"
    data_file.write_text(
        json.dumps(
            [
                {
                    "problem": "What is 1/2 + 1/2?",
                    "answer": r"\left( 1 \right)",
                    "unique_id": "test/algebra/1.json",
                },
                {
                    "question": "Solve for x.",
                    "gold_answer": r"\boxed{\frac{3}{4}}",
                    "question_id": "q-2",
                },
            ]
        )
    )

    queries = load_math500(data_file=data_file, max_samples=2)

    assert [query.id for query in queries] == ["test/algebra/1.json", "q-2"]
    assert queries[0].question == "What is 1/2 + 1/2?"
    assert queries[0].answer == "1"
    assert queries[1].answer == r"\frac{3}{4}"


def test_load_math500_respects_max_samples(tmp_path) -> None:
    data_file = tmp_path / "math500.json"
    data_file.write_text(
        json.dumps(
            [
                {"problem": "q1", "answer": "1"},
                {"problem": "q2", "answer": "2"},
            ]
        )
    )

    queries = load_math500(data_file=data_file, max_samples=1)

    assert len(queries) == 1
    assert queries[0].id == "math500_test_0"
