"""GPQA Diamond (multiple-choice) loader.

**Preferred source:** Hugging Face ``Idavidrein/gpqa`` with config ``gpqa_diamond``
and split ``train``. Access is gated; use ``HF_TOKEN`` / ``HUGGINGFACE_HUB_TOKEN``.

**Why a second Hub read:** The official ``Question`` field often omits the ``(A)``…``(D)``
block (only the stem appears) while ``Correct Answer`` may be a short digit or letter
that is *not* the full option text. The public mirror ``hendrydong/gpqa_diamond_mc``
has the same 198 rows in the same order with explicit ``(A)``…``(D)`` lines and
``\\boxed{letter}`` gold. When the official dataset loads, we **also** load the mirror
(only metadata + four option strings + boxed letter) to fix option order and gold index.
If the official load fails entirely, we fall back to mirror-only normalization.

**Fallback-only:** ``hendrydong/gpqa_diamond_mc`` (split ``test``) if ``Idavidrein/gpqa``
cannot be loaded.

Normalized records:
  - ``question``: stem (mirror layout, instruction line stripped)
  - ``choices``: four strings in A–D order
  - ``answer``: 0–3 index into ``choices``
"""

from __future__ import annotations

import itertools
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

FALLBACK_DATASET = "hendrydong/gpqa_diamond_mc"
FALLBACK_SPLIT = "test"

DEFAULT_NORMALIZED_PATH = Path("data/gpqa_diamond_normalized.jsonl")

# Option line starts at beginning of string or after newline (avoids "(A)" inside sentences).
_RE_OPT_START = re.compile(r"(?:^|\n)\s*\(([A-Da-d])\)\s*", re.MULTILINE)

# Single-line ``(A) rest`` pairs inside official ``Question`` (rare; no newline before letter).
_RE_PAREN_MC = re.compile(r"\(([A-Da-d])\)\s*([^\n]*)", re.MULTILINE)

_RE_LINE_LOWER_MC = re.compile(
    r"^\s*([a-dA-D])\)\s*(.+?)\s*$",
    re.MULTILINE,
)

_RE_BOXED = re.compile(r"\\boxed\{([A-Da-d])\}")

_RE_FINAL_INSTR = re.compile(
    r"\nPlease write your final answer in the form of.*$",
    re.DOTALL | re.IGNORECASE,
)


def _norm_choice_text(s: str) -> str:
    return " ".join(s.strip().split())


@dataclass(frozen=True)
class GPQAMCRow:
    """One GPQA Diamond multiple-choice item in repo-normalized form."""

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


def _parse_mirror_problem(problem: str) -> tuple[str, list[str]]:
    """Stem + four option texts in A–D order from the public mirror ``problem`` field.

    The stem sometimes contains literal ``(A)`` / ``(B)`` inside a sentence; we take the
    **last** run of four line-initial markers ``(A)``…``(D)`` in order. Option bodies may
    span multiple lines up to the next line-initial ``(A)``–``(D)`` marker.
    """
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
        if j + 1 < len(block):
            text_end = block[j + 1].start()
        else:
            text_end = len(body)
        opts.append(body[text_begin:text_end].strip())

    stem = body[: block[0].start()].rstrip()
    return stem, opts


def _boxed_index(solution: str) -> int:
    m = _RE_BOXED.search(solution)
    if not m:
        raise ValueError("mirror solution missing \\\\boxed{A|B|C|D}")
    return ord(m.group(1).upper()) - ord("A")


def _official_columns(record: dict[str, Any]) -> list[str]:
    return [
        str(record.get("Correct Answer", "")).strip(),
        str(record.get("Incorrect Answer 1", "")).strip(),
        str(record.get("Incorrect Answer 2", "")).strip(),
        str(record.get("Incorrect Answer 3", "")).strip(),
    ]


def _permutation_aligning_choices(
    choices_mir: list[str], off_cols: list[str]
) -> Optional[list[int]]:
    """Return p such that norm(choices_mir[j]) == norm(off_cols[p[j]]) for all j.

    ``p[j]`` is the index into ``off_cols`` (0=correct, 1..3 incorrect) for mirror
    slot ``j``. If multiple permutations fit (duplicate texts), return the first that
    also makes slot 0 correspond to the official correct column when unique.
    """
    nm = [_norm_choice_text(c) for c in choices_mir]
    no = [_norm_choice_text(c) for c in off_cols]
    valid: list[list[int]] = []
    for perm in itertools.permutations(range(4), 4):
        if all(nm[j] == no[perm[j]] for j in range(4)):
            valid.append(list(perm))
    if not valid:
        return None
    if len(valid) == 1:
        return valid[0]
    # Prefer permutations where mirror slot matching boxed letter will be checked later;
    # if still ambiguous, prefer perm where perm[0]==0 when texts allow (rare).
    return valid[0]


def _gold_from_official_only(
    q_full: str, c_ans: str, i1: str, i2: str, i3: str
) -> tuple[str, list[str], int] | None:
    """When mirror is unavailable: parse inline options or column order."""
    for parser in (_extract_paren_options, _extract_lowercase_line_options):
        parsed = parser(q_full)
        if parsed is None:
            continue
        stem, choices_list = parsed
        try:
            gold = _gold_index_from_correct_field(c_ans, choices_list)
        except ValueError:
            continue
        return stem.strip(), choices_list, gold
    choices_list = [c_ans.strip(), i1.strip(), i2.strip(), i3.strip()]
    try:
        gold = _gold_index_from_correct_field(c_ans, choices_list)
    except ValueError:
        return None
    return q_full.strip(), choices_list, gold


def _gold_index_from_correct_field(correct_raw: str, choices: list[str]) -> int:
    cr = correct_raw.strip()
    if not cr:
        raise ValueError("empty Correct Answer")
    if len(cr) == 1 and cr.lower() in "abcd":
        return ord(cr.lower()) - ord("a")

    target = _norm_choice_text(cr)
    matches = [i for i, c in enumerate(choices) if _norm_choice_text(c) == target]
    if len(matches) >= 1:
        return matches[0]
    for i, c in enumerate(choices):
        if target in _norm_choice_text(c) or _norm_choice_text(c) in target:
            return i
    raise ValueError(f"Could not match Correct Answer {correct_raw!r} to choices")


def _extract_paren_options(question: str) -> tuple[str, list[str]] | None:
    matches = list(_RE_PAREN_MC.finditer(question))
    if len(matches) < 4:
        return None
    letters = [m.group(1).upper() for m in matches[:4]]
    if letters != ["A", "B", "C", "D"]:
        return None
    opts = [m.group(2).strip() for m in matches[:4]]
    stem = question[: matches[0].start()].rstrip()
    return stem, opts


def _extract_lowercase_line_options(question: str) -> tuple[str, list[str]] | None:
    matches = list(_RE_LINE_LOWER_MC.finditer(question))
    if len(matches) < 4:
        return None
    letters = [m.group(1).upper() for m in matches[:4]]
    if letters != ["A", "B", "C", "D"]:
        return None
    opts = [m.group(2).strip() for m in matches[:4]]
    stem = question[: matches[0].start()].rstrip()
    return stem, opts


def _normalize_pair_official_mirror(
    off: dict[str, Any], mir: dict[str, Any], index: int
) -> GPQAMCRow:
    qid = str(off.get("Record ID") or f"gpqa_diamond_{index}")
    stem_mir, choices_mir = _parse_mirror_problem(str(mir.get("problem", "")))
    boxed_idx = _boxed_index(str(mir.get("solution", "")))
    off_cols = _official_columns(off)

    perm = _permutation_aligning_choices(choices_mir, off_cols)
    if perm is None:
        raise ValueError(f"{qid}: could not align mirror options to official columns")

    # Gold index in mirror A–D order: official correct is off_cols[0] at mirror slot j
    gold_slots = [j for j in range(4) if perm[j] == 0]
    if len(gold_slots) != 1:
        raise ValueError(f"{qid}: ambiguous which mirror slot is official correct")
    gold = gold_slots[0]
    if gold != boxed_idx:
        raise ValueError(
            f"{qid}: mirror boxed index {boxed_idx} != alignment gold {gold}"
        )

    return GPQAMCRow(
        id=qid,
        question=stem_mir,
        choices=(choices_mir[0], choices_mir[1], choices_mir[2], choices_mir[3]),
        answer=gold,
    )


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
        return load_dataset(
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


def _try_load_mirror(cache_dir: str) -> Optional[Any]:
    try:
        return load_dataset(
            FALLBACK_DATASET,
            split=FALLBACK_SPLIT,
            cache_dir=cache_dir,
        )
    except Exception as exc:
        logger.warning("Mirror GPQA load failed (%s)", exc)
        return None


def iter_gpqa_diamond_mc(
    *,
    cache_dir: str = "data",
    max_samples: Optional[int] = None,
    prefer_official: bool = True,
) -> Iterator[GPQAMCRow]:
    """Yield normalized GPQA Diamond rows."""
    official = _try_load_official(cache_dir) if prefer_official else None
    mirror = _try_load_mirror(cache_dir)

    if official is not None:
        if mirror is None or len(mirror) != len(official):
            logger.warning(
                "Official GPQA loaded but mirror missing or length mismatch; "
                "attempting official-only normalization (may fail on some rows)."
            )
        for i, off in enumerate(official):
            if max_samples is not None and i >= max_samples:
                break
            rec_off = dict(off)
            if mirror is not None and i < len(mirror):
                try:
                    yield _normalize_pair_official_mirror(rec_off, dict(mirror[i]), i)
                    continue
                except Exception as exc:
                    logger.warning(
                        "Row %s: official+mirror normalization failed (%s); trying official-only.",
                        i,
                        exc,
                    )
            only = _gold_from_official_only(
                str(rec_off.get("Question", "")),
                str(rec_off.get("Correct Answer", "")),
                str(rec_off.get("Incorrect Answer 1", "")),
                str(rec_off.get("Incorrect Answer 2", "")),
                str(rec_off.get("Incorrect Answer 3", "")),
            )
            if only is None:
                raise ValueError(f"Row {i}: could not normalize without mirror") from None
            stem, ch, gold = only
            qid = str(rec_off.get("Record ID") or f"gpqa_diamond_{i}")
            yield GPQAMCRow(
                id=qid,
                question=stem,
                choices=(ch[0], ch[1], ch[2], ch[3]),
                answer=gold,
            )
        return

    if mirror is None:
        raise RuntimeError(
            f"Neither {OFFICIAL_DATASET} nor {FALLBACK_DATASET} could be loaded."
        )

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
