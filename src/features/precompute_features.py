"""Lightweight, offline feature extraction for adaptive strategy selection.

This module computes cheap feature vectors z(x) from a query (and optionally
from a first-pass model output) without any model calls.  The features are
purely string- and regex-based and are intended to be used *before* choosing
an inference strategy so that, in future work, a routing model can be trained
on top of them.

See docs/PRECOMPUTATION_FEATURES.md for design rationale and feature
descriptions.  See docs/ROUTING_DATASET.md for how these features are combined
with oracle labels into a routing dataset.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Shared compiled patterns (module-level for efficiency)
# ---------------------------------------------------------------------------

_NUMERIC_PATTERN = re.compile(r"-?\d+(?:[,_]\d{3})*(?:\.\d+)?")
_EQUATION_PATTERN = re.compile(r"\d[\s]*[+\-*/=][\s]*\d")
_FRACTION_PATTERN = re.compile(r"\d+\s*/\s*\d+")
_PERCENT_PATTERN = re.compile(r"\d+\s*%")
_CURRENCY_PATTERN = re.compile(r"[$€£¥₹]")
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")

_MULTI_STEP_KEYWORDS = re.compile(
    r"\b(?:total|remaining|after|left|difference|each|every|altogether|twice|"
    r"half|percent|ratio|average|consecutive)\b",
    re.IGNORECASE,
)

_FINAL_ANSWER_CUES = re.compile(
    r"\b(?:final answer|the answer is|therefore|thus|so the|answer:|= \d)",
    re.IGNORECASE,
)

_UNCERTAINTY_PHRASES = re.compile(
    r"\b(?:not sure|uncertain|unclear|i don'?t know|cannot determine|"
    r"it depends|may be|might be|possibly|probably)\b",
    re.IGNORECASE,
)


def _extract_numbers(text: str) -> list[float]:
    """Return all numbers found in *text* as floats (commas stripped)."""
    raw = _NUMERIC_PATTERN.findall(text)
    values: list[float] = []
    for tok in raw:
        try:
            values.append(float(tok.replace(",", "").replace("_", "")))
        except ValueError:
            pass
    return values


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_query_features(question_text: str) -> dict:
    """Compute a cheap feature vector from a query string.

    Parameters
    ----------
    question_text:
        The raw question string (e.g. a GSM8K math word problem).

    Returns
    -------
    dict
        A flat dictionary of scalar features.  All values are either int,
        float, or bool so that they can be fed directly into a downstream
        classifier or logged for analysis.

    Features
    --------
    question_length_chars : int
        Raw character count.
    question_length_tokens_approx : int
        Approximate token count via whitespace splitting.
    num_numeric_mentions : int
        Number of numeric tokens (integers and decimals, with optional commas).
    num_sentences_approx : int
        Approximate sentence count (split on ``.``, ``!``, ``?``).
    has_multi_step_cue : bool
        True when any multi-step keyword is present.
    has_equation_like_pattern : bool
        True when the text contains an inline arithmetic expression.
    has_percent_symbol : bool
        True when a ``%`` following a digit is present.
    has_fraction_pattern : bool
        True when a ``a/b`` fraction pattern is present.
    has_currency_symbol : bool
        True when ``$``, ``€``, ``£``, ``¥``, or ``₹`` appears.
    max_numeric_value_approx : float
        Maximum numeric value found, or 0.0 if none.
    min_numeric_value_approx : float
        Minimum numeric value found, or 0.0 if none.
    numeric_range_approx : float
        ``max - min`` of all numeric values, or 0.0 if fewer than two numbers.
    repeated_number_flag : bool
        True when the same numeric token appears more than once.
    """
    numbers = _extract_numbers(question_text)
    raw_numeric_tokens = _NUMERIC_PATTERN.findall(question_text)

    max_val = max(numbers) if numbers else 0.0
    min_val = min(numbers) if numbers else 0.0
    numeric_range = (max_val - min_val) if len(numbers) >= 2 else 0.0

    repeated = len(raw_numeric_tokens) != len(set(raw_numeric_tokens))

    sentences = [s for s in _SENTENCE_SPLIT_PATTERN.split(question_text) if s.strip()]

    return {
        "question_length_chars": len(question_text),
        "question_length_tokens_approx": len(question_text.split()),
        "num_numeric_mentions": len(numbers),
        "num_sentences_approx": len(sentences),
        "has_multi_step_cue": bool(_MULTI_STEP_KEYWORDS.search(question_text)),
        "has_equation_like_pattern": bool(_EQUATION_PATTERN.search(question_text)),
        "has_percent_symbol": bool(_PERCENT_PATTERN.search(question_text)),
        "has_fraction_pattern": bool(_FRACTION_PATTERN.search(question_text)),
        "has_currency_symbol": bool(_CURRENCY_PATTERN.search(question_text)),
        "max_numeric_value_approx": max_val,
        "min_numeric_value_approx": min_val,
        "numeric_range_approx": numeric_range,
        "repeated_number_flag": repeated,
    }


def extract_first_pass_features(
    question_text: str,
    output_text: str,
    parsed_answer: str | None = None,
) -> dict:
    """Compute cheap features from a first-pass model output.

    This helper is entirely offline and makes no model calls.  It is designed
    to be called after a single cheap inference pass so that a router can
    decide whether to escalate to a more expensive strategy.

    Parameters
    ----------
    question_text:
        The original query string (used to count numeric mentions for
        comparison).
    output_text:
        The raw text produced by the model on the first pass.
    parsed_answer:
        Optional pre-parsed answer string (e.g. the result of
        ``extract_numeric_answer``).  When supplied, it is used to determine
        ``first_pass_parse_success`` more accurately.

    Returns
    -------
    dict
        A flat dictionary of scalar features.

    Features
    --------
    first_pass_parse_success : bool
        True when ``parsed_answer`` is a non-empty string, or (if
        ``parsed_answer`` is not supplied) when at least one number can be
        extracted from the output.
    first_pass_output_length : int
        Character length of the output text.
    first_pass_has_final_answer_cue : bool
        True when the output contains a recognisable final-answer phrase.
    first_pass_has_uncertainty_phrase : bool
        True when the output contains hedging / uncertainty language.
    first_pass_num_numeric_mentions : int
        Number of numeric tokens in the output.
    first_pass_empty_or_malformed_flag : bool
        True when the output is empty, whitespace-only, or very short
        (< 3 characters after stripping).
    """
    stripped = output_text.strip()
    is_empty_or_malformed = len(stripped) < 3

    output_numbers = _extract_numbers(output_text)

    if parsed_answer is not None:
        parse_success = bool(parsed_answer.strip())
    else:
        parse_success = len(output_numbers) > 0

    return {
        "first_pass_parse_success": parse_success,
        "first_pass_output_length": len(output_text),
        "first_pass_has_final_answer_cue": bool(_FINAL_ANSWER_CUES.search(output_text)),
        "first_pass_has_uncertainty_phrase": bool(_UNCERTAINTY_PHRASES.search(output_text)),
        "first_pass_num_numeric_mentions": len(output_numbers),
        "first_pass_empty_or_malformed_flag": is_empty_or_malformed,
    }
