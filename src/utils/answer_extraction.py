"""Utilities for extracting numeric answers from model output.

The goal is to prefer a model's *final answer* rather than arbitrary
intermediate numbers from chain-of-thought text. The extractor is still
rule-based and lightweight, but it now looks for explicit final-answer cues
before falling back to the last plausible number near the end of the response.
"""

from __future__ import annotations

import re
from decimal import Decimal, InvalidOperation

NUMBER_PATTERN = r"-?[$]?[\d,]+(?:\.\d+)?"
NUMBER_RE = re.compile(NUMBER_PATTERN)
BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")
FINAL_ANSWER_CUE_RE = re.compile(
    r"(?:final answer|answer\s*:|so the answer is|therefore|thus|hence|result\s*:)",
    re.IGNORECASE,
)


def _normalize_number(raw: str) -> str:
    cleaned = raw.strip()
    cleaned = cleaned.replace("$", "").replace(",", "")
    cleaned = cleaned.rstrip(".")
    try:
        number = Decimal(cleaned)
    except InvalidOperation:
        return cleaned

    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _extract_last_number(text: str) -> str:
    matches = NUMBER_RE.findall(text)
    if not matches:
        return ""
    return _normalize_number(matches[-1])


def _extract_from_final_cues(text: str) -> str:
    last_match = None
    for match in FINAL_ANSWER_CUE_RE.finditer(text):
        last_match = match
    if last_match is None:
        return ""

    trailing = text[last_match.end() :]
    line = trailing.splitlines()[0] if trailing else ""
    answer = _extract_last_number(line)
    if answer:
        return answer

    final_window = trailing[:200]
    return _extract_last_number(final_window)


def _extract_from_final_lines(text: str) -> str:
    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not nonempty_lines:
        return ""
    tail = "\n".join(nonempty_lines[-3:])
    return _extract_last_number(tail)


def extract_numeric_answer(text: str) -> str:
    """Extract a final numeric answer from model output.

    Strategy:
    1. Prefer ``\\boxed{...}`` answers when present.
    2. Prefer explicit final-answer markers such as ``Answer:``, ``Final answer``,
       ``Therefore``, or ``Thus``.
    3. Otherwise inspect the final non-empty lines.
    4. Fall back to the last number anywhere in the text.
    5. If no number exists, return an empty string so evaluation can treat it as
       an extraction failure.
    """
    stripped = text.strip()
    if not stripped:
        return ""

    boxed_matches = BOXED_RE.findall(stripped)
    if boxed_matches:
        boxed_answer = _extract_last_number(boxed_matches[-1])
        if boxed_answer:
            return boxed_answer

    for extractor in (_extract_from_final_cues, _extract_from_final_lines, _extract_last_number):
        answer = extractor(stripped)
        if answer:
            return answer

    return ""
