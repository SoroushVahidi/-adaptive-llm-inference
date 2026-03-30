"""Utilities for extracting numeric and lightweight symbolic math answers.

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
FINAL_ANSWER_CUE_RE = re.compile(
    r"(?:final answer|answer\s*:|so the answer is|therefore|thus|hence|result\s*:)",
    re.IGNORECASE,
)
INLINE_MATH_RE = re.compile(r"\$(.+?)\$", re.DOTALL)
EMBEDDED_NUMBER_RE = re.compile(r"(?<![A-Za-z\\])-?\d+(?:,\d{3})*(?:\.\d+)?")


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


def _extract_last_boxed_content(text: str) -> str:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return ""

    idx = start + len(marker)
    depth = 1
    chars: list[str] = []
    while idx < len(text):
        char = text[idx]
        if char == "{":
            depth += 1
            chars.append(char)
        elif char == "}":
            depth -= 1
            if depth == 0:
                return "".join(chars).strip()
            chars.append(char)
        else:
            chars.append(char)
        idx += 1
    return ""


def _normalize_embedded_numbers(text: str) -> str:
    return EMBEDDED_NUMBER_RE.sub(lambda match: _normalize_number(match.group(0)), text)


def normalize_math_answer(text: str) -> str:
    """Normalize a lightweight math answer for exact-match comparison.

    This is intentionally simple: it removes common LaTeX wrappers and spacing
    artifacts without trying to symbolically simplify arbitrary expressions.
    """
    candidate = text.strip()
    if not candidate:
        return ""

    boxed = _extract_last_boxed_content(candidate)
    if boxed:
        candidate = boxed

    while candidate.startswith("$") and candidate.endswith("$") and len(candidate) >= 2:
        candidate = candidate[1:-1].strip()

    candidate = candidate.strip().rstrip(".;,")
    candidate = candidate.lstrip(":= ").strip()
    candidate = candidate.replace("\\left", "").replace("\\right", "")
    candidate = candidate.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac")
    candidate = (
        candidate.replace("\\,", "")
        .replace("\\!", "")
        .replace("\\;", "")
        .replace("\\:", "")
        .replace("\n", "")
    )
    candidate = "".join(candidate.split())
    candidate = _normalize_embedded_numbers(candidate)

    if candidate.startswith("(") and candidate.endswith(")") and "," not in candidate[1:-1]:
        candidate = candidate[1:-1].strip()

    if NUMBER_RE.fullmatch(candidate.replace("$", "")):
        return _normalize_number(candidate)
    return candidate


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


_MC_FINAL_RE = re.compile(
    r"(?:final answer|answer)\s*:\s*\(?\s*([ABCDabcd])\s*\)?",
    re.IGNORECASE,
)
_MC_PAREN_RE = re.compile(r"\(\s*([ABCDabcd])\s*\)")
_MC_STANDALONE_LETTER_RE = re.compile(r"(?<![A-Za-z])([ABCDabcd])(?![A-Za-z])")


def extract_mc_answer(text: str) -> str:
    """Extract a multiple-choice letter (A–D) from model output.

    Prefer explicit ``Final answer: (B)`` / ``Answer: C`` patterns, then the last
    parenthesized letter, then a standalone letter token in the final lines.
    Returns upper-case letter or empty string if none found.
    """
    stripped = text.strip()
    if not stripped:
        return ""

    matches = list(_MC_FINAL_RE.finditer(stripped))
    if matches:
        return matches[-1].group(1).upper()

    last_paren = None
    for m in _MC_PAREN_RE.finditer(stripped):
        last_paren = m
    if last_paren:
        return last_paren.group(1).upper()

    nonempty = [ln.strip() for ln in stripped.splitlines() if ln.strip()]
    tail = "\n".join(nonempty[-5:]) if nonempty else stripped
    letters = _MC_STANDALONE_LETTER_RE.findall(tail)
    if letters:
        return letters[-1].upper()

    return ""


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

    boxed_content = _extract_last_boxed_content(stripped)
    if boxed_content:
        boxed_answer = _extract_last_number(boxed_content)
        if boxed_answer:
            return boxed_answer

    for extractor in (_extract_from_final_cues, _extract_from_final_lines, _extract_last_number):
        answer = extractor(stripped)
        if answer:
            return answer

    return ""


def extract_math_answer(text: str) -> str:
    """Extract a final answer from reasoning text for MATH-style evaluation."""
    stripped = text.strip()
    if not stripped:
        return ""

    boxed_content = _extract_last_boxed_content(stripped)
    if boxed_content:
        return normalize_math_answer(boxed_content)

    last_match = None
    for match in FINAL_ANSWER_CUE_RE.finditer(stripped):
        last_match = match
    if last_match is not None:
        trailing = stripped[last_match.end() :].strip()
        boxed_trailing = _extract_last_boxed_content(trailing)
        if boxed_trailing:
            return normalize_math_answer(boxed_trailing)

        inline_math = INLINE_MATH_RE.findall(trailing[:200])
        if inline_math:
            return normalize_math_answer(inline_math[-1])

        first_line = trailing.splitlines()[0].strip() if trailing else ""
        if first_line:
            return normalize_math_answer(first_line)

    tail_lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    if tail_lines:
        tail = "\n".join(tail_lines[-3:])
        boxed_tail = _extract_last_boxed_content(tail)
        if boxed_tail:
            return normalize_math_answer(boxed_tail)

        inline_math = INLINE_MATH_RE.findall(tail)
        if inline_math:
            return normalize_math_answer(inline_math[-1])

        return normalize_math_answer(tail_lines[-1])

    return normalize_math_answer(stripped)
