from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

import pyarrow as pa
import pyarrow.parquet as pq

from src.datasets.gsm8k import load_gsm8k
from src.datasets.math500 import load_math500
from src.datasets.validate_uploaded_datasets import (
    find_uploaded_zip_files,
    normalize_valid_archive,
    run_uploaded_dataset_validation,
    validate_uploaded_archive,
)


def _write_zip(path: Path, member_name: str, payload: str) -> None:
    with ZipFile(path, "w") as zf:
        zf.writestr(member_name, payload)


def _write_parquet_zip(path: Path, member_name: str, rows: list[dict]) -> None:
    table = pa.Table.from_pylist(rows)
    parquet_path = path.with_suffix(".parquet")
    pq.write_table(table, parquet_path)
    with ZipFile(path, "w") as zf:
        zf.write(parquet_path, arcname=member_name)


def test_find_uploaded_zip_files_detects_zip(tmp_path) -> None:
    _write_zip(tmp_path / "gsm8k_upload.zip", "gsm8k_test.json", "[]")
    found = find_uploaded_zip_files(tmp_path)
    assert len(found) == 1
    assert found[0].name == "gsm8k_upload.zip"


def test_validate_gsm8k_archive_and_normalize(tmp_path) -> None:
    rows = [
        {"question": "Tom has 2 apples and buys 3 more. How many?", "answer": "5"},
        {"question": "Sara had 10 and gave away 4. How many left?", "answer": "6"},
    ]
    zip_path = tmp_path / "gsm8k_upload.zip"
    _write_zip(zip_path, "train.json", json.dumps(rows))

    val = validate_uploaded_archive(zip_path)
    assert val.valid_gsm8k is True
    assert val.valid_math500 is False

    out = normalize_valid_archive(zip_path, val, tmp_path / "gsm8k_uploaded_normalized.jsonl")
    assert out["num_rows"] == 2
    lines = (tmp_path / "gsm8k_uploaded_normalized.jsonl").read_text().strip().splitlines()
    parsed = [json.loads(line) for line in lines]
    assert set(parsed[0].keys()) == {"question_id", "question", "gold_answer", "answer_mode"}
    assert parsed[0]["answer_mode"] == "numeric"


def test_validate_math500_archive_and_normalize(tmp_path) -> None:
    rows = [
        {
            "problem": "Compute x if 2x=10.",
            "solution": "x=\\boxed{5}",
            "final_answer": "5",
            "question_id": "m1",
        }
    ]
    zip_path = tmp_path / "math500_upload.zip"
    _write_zip(zip_path, "test.jsonl", "\n".join(json.dumps(r) for r in rows))

    val = validate_uploaded_archive(zip_path)
    assert val.valid_math500 is True
    assert val.valid_gsm8k is False

    out = normalize_valid_archive(zip_path, val, tmp_path / "math500_uploaded_normalized.jsonl")
    assert out["num_rows"] == 1
    one = json.loads((tmp_path / "math500_uploaded_normalized.jsonl").read_text().strip())
    assert one["answer_mode"] == "math"


def test_validate_gsm8k_parquet_archive(tmp_path) -> None:
    rows = [{"question": "A has 1, gets 1.", "answer": "2"}]
    zip_path = tmp_path / "gsm8k_archive.zip"
    _write_parquet_zip(zip_path, "gsm8k/main/test-00000-of-00001.parquet", rows)

    val = validate_uploaded_archive(zip_path)
    assert val.valid_gsm8k is True
    assert val.inferred_dataset == "gsm8k"


def test_validate_math500_uppercase_csv_archive(tmp_path) -> None:
    csv_payload = "Question,Answer\n\"Solve x+1=2\",\"x=\\boxed{1}\"\n"
    zip_path = tmp_path / "archive_math.zip"
    _write_zip(zip_path, "math_500_test.csv", csv_payload)

    val = validate_uploaded_archive(zip_path)
    assert val.valid_math500 is True
    assert val.inferred_dataset == "math500"


def test_loader_compatibility_with_normalized_jsonl(tmp_path) -> None:
    gsm_path = tmp_path / "gsm.jsonl"
    gsm_path.write_text(
        json.dumps(
            {
                "question_id": "g-1",
                "question": "Two plus two?",
                "gold_answer": "4",
                "answer_mode": "numeric",
            }
        )
        + "\n"
    )
    math_path = tmp_path / "math.jsonl"
    math_path.write_text(
        json.dumps(
            {
                "question_id": "m-1",
                "question": "What is 1/2+1/2?",
                "gold_answer": r"\boxed{1}",
                "answer_mode": "math",
            }
        )
        + "\n"
    )
    gsm_queries = load_gsm8k(data_file=gsm_path)
    math_queries = load_math500(data_file=math_path)

    assert gsm_queries[0].id == "g-1"
    assert gsm_queries[0].answer == "4"
    assert math_queries[0].id == "m-1"
    assert math_queries[0].answer == "1"


def test_invalid_archive_graceful_failure(tmp_path) -> None:
    zip_path = tmp_path / "invalid_upload.zip"
    _write_zip(zip_path, "notes.txt", "hello")

    val = validate_uploaded_archive(zip_path)
    assert val.uncertain is True
    assert val.valid_gsm8k is False
    assert val.valid_math500 is False


def test_run_validation_blocked_when_no_zip(tmp_path) -> None:
    out = run_uploaded_dataset_validation(repo_root=tmp_path, output_dir=tmp_path / "outputs")
    assert out["summary"]["status"] == "blocked"
    assert out["summary"]["num_zip_files_found"] == 0
