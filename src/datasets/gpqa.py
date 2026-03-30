"""GPQA Diamond (multiple-choice) loader.

**Preferred source:** Hugging Face ``Idavidrein/gpqa``, config ``gpqa_diamond``,
split ``train`` (gated; set ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN``).

**Official-only normalization:** The Hub row includes ``Question`` plus four answer
strings: ``Correct Answer``, ``Incorrect Answer 1`` … ``Incorrect Answer 3``. Inspecting
the released schema shows the correct option is always the first column; we set::

    choices = (correct, incorrect_1, incorrect_2, incorrect_3)
    answer = 0

So ``answer`` is **always 0** on the official path. This is **not** the same as
randomized A/B/C/D presentation order on a test form—the public release does not
encode which letter was (A) on the original exam. For shuffled-letter evaluation,
shuffle ``choices`` and remap ``answer`` in your experiment code with a fixed RNG seed.

**Fallback:** ``hendrydong/gpqa_diamond_mc`` (split ``test``) only if the official
dataset cannot be loaded. Mirror rows use ``(A)``…``(D)`` blocks and ``\\boxed{}``;
parsed ``choices`` are in **letter A–D order** and ``answer`` is 0–3 accordingly.

**Ordering risk:** If the Hub maintainers ever reorder columns or change semantics,
normalization must be re-validated. Optional ``verify_official_mirror_dataset_pair()``
(below) cross-checks official vs mirror when **both** are loadable—use in tests or
release audits, not in the default load path.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from datasets import load_dataset

logger = logging.getLogger(__name__)

OFFICIAL_DATASET = "Idavidrein/gpqa"
OFFICIAL_CONFIG = "gpqa_diamond"
OFFICIAL_SPLIT = "train"
EXPECTED_ROW_COUNT = 198

FALLBACK_DATASET = "hendrydong/gpqa_diamond_mc"
FALLBACK_SPLIT = "test"

DEFAULT_NORMALIZED_PATH = Path("data/gpqa_diamond_normalized.jsonl")

_RE_OPT_START = re.compile(r"(?:^|\n)\s*\(([A-Da-d])\)\s*", re.MULTILINE)

_RE_BOXED = re.compile(r"\\boxed\{([A-Da-d])\}")

_RE_FINAL_INSTR = re.compile(
    r"\nPlease write your final answer in the form of.*$",
    re.DOTALL | re.IGNORECASE,
)


def _norm_choice_text(s: str) -> str:
    return " ".join(s.strip().split())


@dataclass(frozen=True)
class GPQAMCRow:
    """One GPQA Diamond item in repo-normalized form."""

    id: str
    question: str
    choices: tuple[str, str, str, str]
    answer: int

    def to_json_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["choices"] = list(self.choices)
        return d


def _strip_final_instruction(stem: str) -> str:
    return _RE_FINAL_INSTR.sub("", stem).rstrip()


def _official_answer_columns(record: dict[str, Any]) -> tuple[str, str, str, str]:
    c0 = str(record.get("Correct Answer", "")).strip()
    c1 = str(record.get("Incorrect Answer 1", "")).strip()
    c2 = str(record.get("Incorrect Answer 2", "")).strip()
    c3 = str(record.get("Incorrect Answer 3", "")).strip()
    if not c0 or not c1 or not c2 or not c3:
        raise ValueError("official row has empty Correct or Incorrect Answer field")
    return c0, c1, c2, c3


def _normalize_official_row(record: dict[str, Any], index: int) -> GPQAMCRow:
    qid = str(record.get("Record ID") or f"gpqa_diamond_{index}")
    question = str(record.get("Question", "")).strip()
    if not question:
        raise ValueError(f"{qid}: empty Question")

    c0, c1, c2, c3 = _official_answer_columns(record)
    # Schema contract from Idavidrein/gpqa gpqa_diamond: first field is the gold string.
    ca_field = str(record.get("Correct Answer", "")).strip()
    if _norm_choice_text(c0) != _norm_choice_text(ca_field):
        raise ValueError(f"{qid}: Correct Answer column does not match choices[0]")

    return GPQAMCRow(
        id=qid,
        question=question,
        choices=(c0, c1, c2, c3),
        answer=0,
    )


def _parse_mirror_problem(problem: str) -> tuple[str, list[str]]:
    """Stem + four option texts in A–D order from ``hendrydong/gpqa_diamond_mc``."""
    body = _strip_final_instruction(problem)
    starts = list(_RE_OPT_START.finditer(body))
    if len(starts) < 4:
        raise ValueError("mirror problem missing (A)–(D) options")

    best_start_idx: Optional[int] = None
    for i in range(len(starts) - 3):
        letters = [starts[i + j].group(1).upper() for j in range(4)]
        if letters == ["A", "B", "C", "D"]:
            best_start_idx = i

    if best_start_idx is None:
        raise ValueError("mirror problem has no A–D option block in order")

    block = starts[best_start_idx : best_start_idx + 4]
    opts: list[str] = []
    for j, m in enumerate(block):
        text_begin = m.end()
        text_end = block[j + 1].start() if j + 1 < len(block) else len(body)
        opts.append(body[text_begin:text_end].strip())

    stem = body[: block[0].start()].rstrip()
    return stem, opts


def _boxed_index(solution: str) -> int:
    m = _RE_BOXED.search(solution)
    if not m:
        raise ValueError("mirror solution missing \\\\boxed{A|B|C|D}")
    return ord(m.group(1).upper()) - ord("A")


def _normalize_fallback_only(record: dict[str, Any], index: int) -> GPQAMCRow:
    problem = str(record.get("problem", "")).strip()
    sol = str(record.get("solution", ""))
    dom = str(record.get("domain", "")).strip()
    qid = f"gpqa_diamond_mc_{index}" if not dom else f"gpqa_diamond_mc_{index}_{dom}"

    stem, opts = _parse_mirror_problem(problem)
    answer = _boxed_index(sol)
    return GPQAMCRow(
        id=qid,
        question=stem,
        choices=(opts[0], opts[1], opts[2], opts[3]),
        answer=answer,
    )


def _try_load_official(cache_dir: str) -> Optional[Any]:
    try:
        ds = load_dataset(
            OFFICIAL_DATASET,
            OFFICIAL_CONFIG,
            split=OFFICIAL_SPLIT,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        logger.warning(
            "Official GPQA load failed (%s); using fallback %s",
            exc,
            FALLBACK_DATASET,
        )
        return None

    n = len(ds)
    if n != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Official GPQA Diamond expected {EXPECTED_ROW_COUNT} rows, got {n}. "
            "Refuse to normalize: dataset may have changed."
        )
    return ds


def _try_load_fallback(cache_dir: str) -> Any:
    ds = load_dataset(
        FALLBACK_DATASET,
        split=FALLBACK_SPLIT,
        cache_dir=cache_dir,
    )
    n = len(ds)
    if n != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Fallback GPQA expected {EXPECTED_ROW_COUNT} rows, got {n}. "
            "Refuse to normalize: dataset may have changed."
        )
    return ds


def verify_official_mirror_dataset_pair(official: Any, mirror: Any) -> None:
    """Cross-check official vs mirror splits (same length, paired content, boxed vs gold).

    **Does not** load Hub—pass ``datasets`` split objects. Use in tests when both
    sources are available. Fails loudly if maintainers reorder rows or change schema.
    """
    if len(official) != len(mirror):
        raise ValueError(
            f"Alignment failure: official len {len(official)} != mirror len {len(mirror)}"
        )
    if len(official) != EXPECTED_ROW_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_ROW_COUNT} rows, got {len(official)} — update EXPECTED_ROW_COUNT?"
        )

    for i in range(len(official)):
        off = dict(official[i])
        mir = dict(mirror[i])
        qid = str(off.get("Record ID", f"row_{i}"))

        stem_mir, opts_mir = _parse_mirror_problem(str(mir.get("problem", "")))
        off_q = str(off.get("Question", "")).strip()
        ns = _norm_choice_text(stem_mir)
        nq = _norm_choice_text(off_q)
        prefix_n = min(100, len(ns), len(nq))
        if prefix_n < 20:
            raise ValueError(f"{qid} row {i}: stem/question too short for alignment check")
        if ns[:prefix_n] != nq[:prefix_n]:
            raise ValueError(
                f"{qid} row {i}: official Question and mirror stem prefix mismatch — "
                "row ordering or content may have diverged"
            )

        c0, _, _, _ = _official_answer_columns(off)
        gold_norm = _norm_choice_text(c0)
        mir_norms = [_norm_choice_text(x) for x in opts_mir]
        if gold_norm not in mir_norms:
            raise ValueError(
                f"{qid} row {i}: official correct text not found among mirror A–D options"
            )
        gold_slot = mir_norms.index(gold_norm)
        boxed = _boxed_index(str(mir.get("solution", "")))
        if gold_slot != boxed:
            raise ValueError(
                f"{qid} row {i}: mirror boxed letter index {boxed} != "
                f"slot of official correct {gold_slot}"
            )


def iter_gpqa_diamond_mc(
    *,
    cache_dir: str = "data",
    max_samples: Optional[int] = None,
    prefer_official: bool = True,
) -> Iterator[GPQAMCRow]:
    """Yield normalized GPQA Diamond rows (official path does not load the mirror)."""
    if prefer_official:
        official = _try_load_official(cache_dir)
        if official is not None:
            for i, off in enumerate(official):
                if max_samples is not None and i >= max_samples:
                    break
                yield _normalize_official_row(dict(off), i)
            return

    try:
        mirror = _try_load_fallback(cache_dir)
    except Exception as exc:
        raise RuntimeError(
            f"Official GPQA unavailable and fallback {FALLBACK_DATASET} failed: {exc}"
        ) from exc

    for i, record in enumerate(mirror):
        if max_samples is not None and i >= max_samples:
            break
        yield _normalize_fallback_only(dict(record), i)


def load_gpqa_diamond_mc(
    *,
    cache_dir: str = "data",
    max_samples: Optional[int] = None,
    prefer_official: bool = True,
) -> list[GPQAMCRow]:
    return list(
        iter_gpqa_diamond_mc(
            cache_dir=cache_dir,
            max_samples=max_samples,
            prefer_official=prefer_official,
        )
    )


def write_normalized_gpqa_jsonl(
    path: str | Path = DEFAULT_NORMALIZED_PATH,
    *,
    cache_dir: str = "data",
    prefer_official: bool = True,
) -> Path:
    """Write ``data/gpqa_diamond_normalized.jsonl`` (198 rows when full)."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = load_gpqa_diamond_mc(cache_dir=cache_dir, prefer_official=prefer_official)
    with out.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row.to_json_dict(), ensure_ascii=False) + "\n")
    return out


def load_gpqa_from_jsonl(
    path: str | Path = DEFAULT_NORMALIZED_PATH,
    max_samples: Optional[int] = None,
) -> list[GPQAMCRow]:
    """Load normalized JSONL (offline)."""
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(p)
    out: list[GPQAMCRow] = []
    with p.open(encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            if max_samples is not None and i >= max_samples:
                break
            if not line.strip():
                continue
            d = json.loads(line)
            ch = d["choices"]
            out.append(
                GPQAMCRow(
                    id=str(d["id"]),
                    question=str(d["question"]),
                    choices=(ch[0], ch[1], ch[2], ch[3]),
                    answer=int(d["answer"]),
                )
            )
    return out
