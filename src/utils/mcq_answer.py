"""Multiple-choice answer normalization and extraction from model text."""

from __future__ import annotations

import logging
import re

_LOG = logging.getLogger(__name__)

_BOXED_LETTER_RE = re.compile(
    r"\\boxed\s*\{\s*\\?(?:mathrm|text|textbf)?\s*([A-Da-d])\s*\}",
    re.IGNORECASE,
)
_FINAL_ANSWER_LETTER_RE = re.compile(
    r"(?:final answer|answer)\s*[:.]?\s*([A-Da-d])\b",
    re.IGNORECASE,
)
_STANDALONE_LETTER_PAREN_RE = re.compile(r"\(\s*([A-Da-d])\s*\)")
_LINE_START_LETTER_RE = re.compile(r"^\s*([A-Da-d])\s*[.)]?\s*$", re.MULTILINE)


def normalize_mcq_letter(raw: str | None) -> str:
    """Return upper-case A–D or empty string."""
    if raw is None:
        return ""
    s = str(raw).strip().upper()
    if len(s) == 1 and s in "ABCD":
        return s
    return ""


def extract_mcq_letter(text: str, *, log_failures: bool = False) -> str:
    """Best-effort letter from model output (boxed, final answer, parens, last line)."""
    if not text or not str(text).strip():
        if log_failures:
            _LOG.warning("mcq_parse: empty model output")
        return ""

    stripped = text.strip()
    m = _BOXED_LETTER_RE.search(stripped)
    if m:
        return normalize_mcq_letter(m.group(1))

    m = _FINAL_ANSWER_LETTER_RE.search(stripped)
    if m:
        return normalize_mcq_letter(m.group(1))

    for match in _STANDALONE_LETTER_PAREN_RE.finditer(stripped):
        letter = normalize_mcq_letter(match.group(1))
        if letter:
            return letter

    lines = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    for ln in reversed(lines[-5:]):
        m = _LINE_START_LETTER_RE.match(ln)
        if m:
            return normalize_mcq_letter(m.group(1))
        if len(ln) == 1:
            letter = normalize_mcq_letter(ln)
            if letter:
                return letter

    tail = stripped[-80:]
    for ch in reversed(tail):
        letter = normalize_mcq_letter(ch)
        if letter:
            return letter

    if log_failures:
        _LOG.warning("mcq_parse: could not extract A–D from output (len=%d)", len(stripped))
    return ""


def gold_letter_from_solution(solution: str) -> str:
    """Parse gold letter from \\boxed{X} in official-style solution strings."""
    return extract_mcq_letter(solution, log_failures=False)
