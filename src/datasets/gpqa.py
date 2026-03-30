"""GPQA Diamond (multiple-choice) loader.

Primary integration path: public HuggingFace mirror ``hendrydong/gpqa_diamond_mc``
(split ``test``), which includes (A)–(D) blocks and ``\\boxed{X}`` solutions.

The official ``idavidrein/gpqa`` GitHub ``dataset.zip`` is **password-protected**
(encrypted ZIP); this loader documents that and avoids it by default.

Normalized JSONL format (written under ``data/gpqa_diamond_normalized.jsonl``):

.. code-block:: json

   {"question": "...", "choices": ["...", "..."], "answer": "D"}
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from datasets import load_dataset
from src.datasets.gsm8k import Query

DEFAULT_HF_MC_SOURCE = "hendrydong/gpqa_diamond_mc"
DEFAULT_JSONL_REL = Path("data/gpqa_diamond_normalized.jsonl")

_OPTION_LINE_RE = re.compile(
    r"^\s*\(\s*([A-Da-d])\s*\)\s*(.+?)\s*$",
    re.MULTILINE,
)


def _parse_choices_from_problem(problem: str) -> tuple[str, tuple[str, ...]]:
    """Split GPQA-style problem into stem and ordered choices A–D."""
    text = problem.strip()
    matches = list(_OPTION_LINE_RE.finditer(text))
    if len(matches) >= 4:
        start = matches[0].start()
        stem = text[:start].strip()
        by_letter: dict[str, str] = {}
        for m in matches:
            letter = m.group(1).upper()
            body = m.group(2).strip()
            by_letter[letter] = body
        ordered = tuple(by_letter.get(L, "") for L in "ABCD")
        return stem, ordered
    return text, tuple()


def _record_from_hf_row(ex: dict, index: int) -> dict[str, object]:
    problem = str(ex.get("problem") or "")
    solution = str(ex.get("solution") or "")
    stem, choices = _parse_choices_from_problem(problem)
    from src.utils.mcq_answer import gold_letter_from_solution

    letter = gold_letter_from_solution(solution)
    if not letter and solution:
        letter = gold_letter_from_solution("\\boxed{" + solution.strip() + "}")
    qid = f"gpqa_diamond_{index}"
    return {
        "id": qid,
        "question": stem or problem,
        "choices": list(choices) if choices else [],
        "answer": letter,
        "domain": ex.get("domain", ""),
        "raw_problem": problem,
        "raw_solution": solution,
    }


def write_gpqa_normalized_jsonl(records: list[dict[str, object]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            out = {
                "question": rec["question"],
                "choices": rec["choices"],
                "answer": rec["answer"],
            }
            if rec.get("id"):
                out["id"] = rec["id"]
            fh.write(json.dumps(out, ensure_ascii=False) + "\n")


def load_gpqa_diamond(
    max_samples: int | None = None,
    cache_dir: str = "data",
    hf_source: str = DEFAULT_HF_MC_SOURCE,
    split: str = "test",
    jsonl_path: str | Path | None = DEFAULT_JSONL_REL,
    data_file: str | Path | None = None,
) -> list[Query]:
    """Load GPQA Diamond MC as ``Query`` with ``choices`` and letter ``answer``."""
    if data_file is not None:
        p = Path(data_file)
        rows: list[dict] = []
        if str(data_file).endswith(".jsonl"):
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rows.append(json.loads(line))
        else:
            payload = json.loads(p.read_text(encoding="utf-8"))
            rows = payload if isinstance(payload, list) else [payload]
        out: list[Query] = []
        for idx, rec in enumerate(rows):
            if max_samples is not None and idx >= max_samples:
                break
            ch = rec.get("choices") or []
            if isinstance(ch, str):
                ch = json.loads(ch)
            qid = str(rec.get("id") or rec.get("question_id") or f"gpqa_{idx}")
            out.append(
                Query(
                    id=qid,
                    question=str(rec["question"]),
                    answer=str(rec["answer"]).strip().upper()[:1],
                    choices=tuple(str(c) for c in ch) if ch else tuple(),
                )
            )
        return out

    ds = load_dataset(hf_source, split=split, cache_dir=cache_dir)
    all_norm: list[dict[str, object]] = []
    queries: list[Query] = []
    for idx, ex in enumerate(ds):
        rec = _record_from_hf_row(dict(ex), idx)
        all_norm.append(rec)
        if max_samples is not None and idx >= max_samples:
            continue
        ch_tuple = tuple(str(c) for c in rec["choices"]) if rec["choices"] else tuple()
        queries.append(
            Query(
                id=str(rec["id"]),
                question=str(rec["question"]),
                answer=str(rec["answer"]),
                choices=ch_tuple if ch_tuple else None,
            )
        )

    if jsonl_path is not None:
        write_gpqa_normalized_jsonl(all_norm, jsonl_path)

    return queries
