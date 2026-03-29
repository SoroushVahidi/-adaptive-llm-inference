"""Validation and normalization utilities for uploaded GSM8K/MATH500 zip files."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from zipfile import ZipFile


@dataclass(frozen=True)
class ArchiveValidation:
    archive_path: str
    inferred_dataset: str
    valid_gsm8k: bool
    valid_math500: bool
    uncertain: bool
    reason: str
    record_count_scanned: int
    schema_fields: list[str]
    splits_present: list[str]
    sample_records: list[dict]


def find_uploaded_zip_files(repo_root: str | Path) -> list[Path]:
    """Find zip files under the repository root."""
    root = Path(repo_root)
    return sorted(root.rglob("*.zip"))


def _guess_split(member_name: str) -> str | None:
    lowered = member_name.lower()
    for split in ("train", "test", "validation", "dev"):
        if split in lowered:
            return split
    return None


def _load_member_records(zf: ZipFile, member_name: str, limit: int = 2000) -> list[dict]:
    lowered = member_name.lower()
    raw = zf.read(member_name).decode("utf-8", errors="ignore")

    if lowered.endswith(".jsonl"):
        rows: list[dict] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            obj = json.loads(line)
            if isinstance(obj, dict):
                rows.append(obj)
            if len(rows) >= limit:
                break
        return rows

    if lowered.endswith(".json"):
        obj = json.loads(raw)
        if isinstance(obj, list):
            return [r for r in obj if isinstance(r, dict)][:limit]
        if isinstance(obj, dict):
            for key in ("data", "records", "examples", "items"):
                maybe = obj.get(key)
                if isinstance(maybe, list):
                    return [r for r in maybe if isinstance(r, dict)][:limit]
            return [obj]
        return []

    if lowered.endswith(".csv"):
        reader = csv.DictReader(StringIO(raw))
        rows = []
        for idx, row in enumerate(reader):
            rows.append(dict(row))
            if idx + 1 >= limit:
                break
        return rows

    return []


def _extract_records_from_zip(zip_path: Path, per_file_limit: int = 300) -> tuple[list[dict], list[str]]:
    records: list[dict] = []
    splits: set[str] = set()
    with ZipFile(zip_path) as zf:
        for member in zf.namelist():
            if member.endswith("/"):
                continue
            split = _guess_split(member)
            if split:
                splits.add(split)
            if not member.lower().endswith((".json", ".jsonl", ".csv")):
                continue
            parsed = _load_member_records(zf, member, limit=per_file_limit)
            records.extend(parsed)
            if len(records) >= per_file_limit:
                break
    return records[:per_file_limit], sorted(splits)


def _infer_dataset(records: list[dict], archive_name: str) -> tuple[str, str, bool, bool, bool]:
    if not records:
        return (
            "unknown",
            "No parseable JSON/JSONL/CSV records found in archive.",
            False,
            False,
            True,
        )

    fields = {k for rec in records for k in rec.keys()}
    lowered_name = archive_name.lower()

    has_question = any(k in fields for k in ("question", "problem"))
    has_answer = any(k in fields for k in ("answer", "gold_answer", "solution", "final_answer"))

    sample_text = " ".join(
        str(rec.get("question") or rec.get("problem") or "")[:400]
        + " "
        + str(rec.get("answer") or rec.get("solution") or "")[:200]
        for rec in records[:40]
    ).lower()

    gsm_signals = sum(
        [
            int("gsm" in lowered_name),
            int("grade school" in sample_text),
            int("####" in sample_text),
            int(has_question and has_answer and "boxed" not in sample_text),
        ]
    )
    math_signals = sum(
        [
            int("math500" in lowered_name or "math-500" in lowered_name),
            int("boxed" in sample_text),
            int("\\frac" in sample_text or "\\" in sample_text),
            int("problem" in fields and has_answer),
        ]
    )

    if math_signals >= 2 and math_signals > gsm_signals:
        inferred = "math500"
        reason = "Archive contents look MATH500-like (symbolic/problem-solution patterns)."
        return inferred, reason, False, True, False

    if gsm_signals >= 2 and gsm_signals >= math_signals:
        inferred = "gsm8k"
        reason = "Archive contents look GSM8K-like (question/answer numeric word problems)."
        return inferred, reason, True, False, False

    uncertain = not (has_question and has_answer)
    reason = "Could not confidently classify archive as GSM8K or MATH500."
    return "unknown", reason, False, False, uncertain


def validate_uploaded_archive(zip_path: str | Path) -> ArchiveValidation:
    """Validate one uploaded zip archive as GSM8K or MATH500-like."""
    path = Path(zip_path)
    records, splits = _extract_records_from_zip(path)
    inferred, reason, valid_gsm8k, valid_math500, uncertain = _infer_dataset(records, path.name)
    fields = sorted({k for rec in records for k in rec.keys()})

    return ArchiveValidation(
        archive_path=str(path),
        inferred_dataset=inferred,
        valid_gsm8k=valid_gsm8k,
        valid_math500=valid_math500,
        uncertain=uncertain,
        reason=reason,
        record_count_scanned=len(records),
        schema_fields=fields,
        splits_present=splits,
        sample_records=records[:3],
    )


def _normalize_row(row: dict, dataset: str, idx: int, prefix: str) -> dict | None:
    if dataset == "gsm8k":
        question = str(row.get("question") or row.get("problem") or "").strip()
        answer = str(row.get("gold_answer") or row.get("answer") or row.get("solution") or "").strip()
        if not question or not answer:
            return None
        qid = str(row.get("question_id") or row.get("id") or f"{prefix}_{idx}")
        return {
            "question_id": qid,
            "question": question,
            "gold_answer": answer,
            "answer_mode": "numeric",
        }

    if dataset == "math500":
        question = str(row.get("problem") or row.get("question") or "").strip()
        answer = str(
            row.get("gold_answer")
            or row.get("final_answer")
            or row.get("answer")
            or row.get("solution")
            or ""
        ).strip()
        if not question or not answer:
            return None
        qid = str(row.get("question_id") or row.get("unique_id") or row.get("id") or f"{prefix}_{idx}")
        return {
            "question_id": qid,
            "question": question,
            "gold_answer": answer,
            "answer_mode": "math",
        }

    return None


def normalize_valid_archive(
    zip_path: str | Path,
    validation: ArchiveValidation,
    output_path: str | Path,
) -> dict[str, int | str]:
    """Normalize a validated archive into canonical JSONL rows."""
    if validation.inferred_dataset not in {"gsm8k", "math500"}:
        raise ValueError("Cannot normalize archive with unknown dataset type")

    records, _ = _extract_records_from_zip(Path(zip_path), per_file_limit=100000)
    normalized: list[dict] = []
    for idx, row in enumerate(records):
        mapped = _normalize_row(row, validation.inferred_dataset, idx, validation.inferred_dataset)
        if mapped is not None:
            normalized.append(mapped)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for row in normalized:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    return {
        "output_path": str(out),
        "num_rows": len(normalized),
        "dataset": validation.inferred_dataset,
    }


def run_uploaded_dataset_validation(
    repo_root: str | Path = ".",
    output_dir: str | Path = "outputs/dataset_validation",
    data_dir: str | Path = "data",
) -> dict:
    """Find uploaded zip files, validate them, and normalize valid archives."""
    zip_files = find_uploaded_zip_files(repo_root)
    output_base = Path(output_dir)
    output_base.mkdir(parents=True, exist_ok=True)

    validations: list[ArchiveValidation] = [validate_uploaded_archive(p) for p in zip_files]

    normalized_outputs: list[dict] = []
    for item in validations:
        if item.valid_gsm8k:
            normalized_outputs.append(
                normalize_valid_archive(
                    item.archive_path,
                    item,
                    Path(data_dir) / "gsm8k_uploaded_normalized.jsonl",
                )
            )
        elif item.valid_math500:
            normalized_outputs.append(
                normalize_valid_archive(
                    item.archive_path,
                    item,
                    Path(data_dir) / "math500_uploaded_normalized.jsonl",
                )
            )

    summary = {
        "zip_files_found": [str(p) for p in zip_files],
        "num_zip_files_found": len(zip_files),
        "validations": [
            {
                "archive_path": v.archive_path,
                "inferred_dataset": v.inferred_dataset,
                "valid_gsm8k": v.valid_gsm8k,
                "valid_math500": v.valid_math500,
                "uncertain": v.uncertain,
                "reason": v.reason,
                "record_count_scanned": v.record_count_scanned,
                "schema_fields": v.schema_fields,
                "splits_present": v.splits_present,
            }
            for v in validations
        ],
        "normalized_outputs": normalized_outputs,
        "status": "ok" if zip_files else "blocked",
        "block_reason": "No ZIP files were found in repository." if not zip_files else "",
    }

    summary_path = output_base / "validation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    gsm_preview = next((v.sample_records for v in validations if v.valid_gsm8k), [])
    math_preview = next((v.sample_records for v in validations if v.valid_math500), [])
    (output_base / "gsm8k_sample_preview.json").write_text(json.dumps(gsm_preview, indent=2))
    (output_base / "math500_sample_preview.json").write_text(json.dumps(math_preview, indent=2))

    return {
        "summary_path": str(summary_path),
        "gsm_preview_path": str(output_base / "gsm8k_sample_preview.json"),
        "math_preview_path": str(output_base / "math500_sample_preview.json"),
        "summary": summary,
    }
