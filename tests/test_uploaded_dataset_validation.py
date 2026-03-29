from __future__ import annotations

import json
from pathlib import Path
from zipfile import ZipFile

from src.datasets.validate_uploaded_datasets import (
    find_uploaded_zip_files,
    normalize_valid_archive,
    run_uploaded_dataset_validation,
    validate_uploaded_archive,
)


def _write_zip(path: Path, member_name: str, payload: str) -> None:
    with ZipFile(path, "w") as zf:
        zf.writestr(member_name, payload)


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
