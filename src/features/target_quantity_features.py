"""Target-quantity and wording-trap feature extraction for math word problems.

This module provides ``extract_target_quantity_features``, a lightweight,
regex-only feature extractor that captures:

  A. Target-type cues — what *kind* of quantity the question asks for
     (remainder, total, difference, rate, money, time).

  B. Wording-trap signals — surface-form verbs and structures that
     commonly cause a direct-greedy pass to return the wrong quantity.

  C. Answer-risk signals — heuristics for whether the question has
     multiple plausible numeric endpoints (intermediate quantity risk) or
     might lead the model to echo a number already in the prompt.

All features are boolean.  The extractor is entirely offline (no model
calls) and has no external dependencies beyond the Python standard library.

See docs/TARGET_QUANTITY_FEATURES.md for full design rationale and examples.
"""

from __future__ import annotations

import re

# ---------------------------------------------------------------------------
# Compiled patterns — module-level for efficiency
# ---------------------------------------------------------------------------

# A. Target-type cues
_REMAINING_PATTERN = re.compile(
    r"\b(?:remaining|left\s*over|left|have\s+left|are\s+left|is\s+left)\b",
    re.IGNORECASE,
)

_TOTAL_PATTERN = re.compile(
    r"\b(?:total|altogether|in\s+all|combined|grand\s+total)\b",
    re.IGNORECASE,
)

_DIFFERENCE_PATTERN = re.compile(
    r"\b(?:difference|how\s+much\s+more|how\s+many\s+more|"
    r"more\s+than|less\s+than|fewer\s+than|how\s+much\s+less)\b",
    re.IGNORECASE,
)

_RATE_OR_UNIT_PATTERN = re.compile(
    r"\b(?:per|each|every|apiece|a\s+piece|per\s+day|per\s+week|"
    r"per\s+hour|per\s+month|per\s+year|rate|per\s+minute)\b",
    re.IGNORECASE,
)

_MONEY_PATTERN = re.compile(
    r"(?:[$€£¥₹]|\b(?:dollar|dollars|cent|cents|euro|euros|pound|pounds|"
    r"rupee|rupees|yen)\b)",
    re.IGNORECASE,
)

_TIME_PATTERN = re.compile(
    r"\b(?:minute|minutes|hour|hours|day|days|week|weeks|month|months|"
    r"year|years|second|seconds)\b",
    re.IGNORECASE,
)

# B. Wording-trap signals
_SUBTRACTION_TRAP_PATTERN = re.compile(
    r"\b(?:spent|spend|lost|lose|loses|gave\s+away|give\s+away|"
    r"sold|sell|sells|used|uses|use\s+up|used\s+up|ate|eaten|"
    r"consumed|donated|gave|given)\b",
    re.IGNORECASE,
)

_ADDITION_TRAP_PATTERN = re.compile(
    r"\b(?:also|then|together|as\s+well|additionally|on\s+top\s+of|"
    r"plus|added|add|combined\s+with)\b",
    re.IGNORECASE,
)

# Multi-operation hint: look for ≥ 2 arithmetic-related verbs in the text.
# We count distinct action verbs that imply a step; ≥ 2 = multi-operation.
_OPERATION_VERB_PATTERN = re.compile(
    r"\b(?:bought|buy|buys|sold|sell|sells|earned|earn|earns|"
    r"spent|spend|spends|received|receive|receives|"
    r"gave|give|gives|took|take|takes|found|find|finds|"
    r"added|add|adds|subtracted|subtract|subtracts|"
    r"multiplied|multiply|divided|divide|"
    r"collected|collect|collects|paid|pay|pays|"
    r"saved|save|saves|won|win|wins|lost|lose|loses|"
    r"ate|eat|eats|eaten|used|uses|donated|donate)\b",
    re.IGNORECASE,
)

# Numeric token pattern (reused from precompute_features)
_NUMERIC_PATTERN = re.compile(r"-?\d+(?:[,_]\d{3})*(?:\.\d+)?")

# Sentence splitter
_SENTENCE_SPLIT_PATTERN = re.compile(r"[.!?]+")

# The final sentence of a question is often short and contains the ask.
# "Short final question" heuristic: last sentence has ≤ 12 tokens.
_FINAL_SENTENCE_MAX_TOKENS = 12

# Thresholds for answer-risk heuristics
_MIN_NUMBERS_FOR_ECHO_RISK = 4
_MIN_SENTENCES_FOR_INTERMEDIATE_RISK = 3
_MIN_NUMBERS_FOR_INTERMEDIATE_RISK = 3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_target_quantity_features(question_text: str) -> dict[str, bool]:
    """Compute target-quantity and wording-trap features from a question string.

    This extractor is designed to complement ``extract_query_features`` from
    ``src.features.precompute_features``.  It does not duplicate any existing
    feature and can be merged with the base feature dict via ``{**base,
    **tq}``.

    Parameters
    ----------
    question_text:
        The raw question string (e.g. a GSM8K math word problem).

    Returns
    -------
    dict
        Flat dictionary of boolean features.  All values are ``bool``.

    Features (A — target-type cues)
    --------------------------------
    asks_remaining_or_left : bool
        True when the question uses "remaining", "left over", or "left"
        indicating the answer is a remainder after subtraction.
    asks_total : bool
        True when the question uses "total", "altogether", or "in all".
    asks_difference : bool
        True when the question asks for a difference ("more than", "less
        than", "difference").
    asks_rate_or_unit : bool
        True when the question uses "per", "each", "every", or similar,
        indicating a rate or per-unit quantity.
    asks_money : bool
        True when the question mentions money (currency symbols or word
        forms like "dollars", "cents").
    asks_time : bool
        True when the question mentions a time unit (minutes, hours, days,
        weeks, months, years, seconds).

    Features (B — wording-trap signals)
    -------------------------------------
    has_subtraction_trap_verb : bool
        True when verbs like "spent", "lost", "gave away", "sold", or "used"
        appear, suggesting a hidden two-step subtraction structure.
    has_addition_trap_structure : bool
        True when words like "also", "then", "together", or "additionally"
        appear, suggesting chained additive steps.
    has_multi_operation_hint : bool
        True when two or more distinct operation verbs are found in the
        question, indicating a multi-step chain.

    Features (C — answer-risk signals)
    ------------------------------------
    likely_intermediate_quantity_ask : bool
        Heuristic: True when the question has multiple sentences AND multiple
        numbers AND no explicit total/remaining cue — a configuration that
        often leads the model to stop at an intermediate result.
    potential_answer_echo_risk : bool
        Heuristic: True when the question contains many numeric tokens AND
        the final sentence is very short — a configuration where the model
        may copy a given value rather than compute the answer.
    """
    # Precompute shared values
    numbers = _NUMERIC_PATTERN.findall(question_text)
    n_numbers = len(numbers)

    sentences = [s for s in _SENTENCE_SPLIT_PATTERN.split(question_text) if s.strip()]
    n_sentences = len(sentences)

    # A. Target-type cues
    asks_remaining_or_left = bool(_REMAINING_PATTERN.search(question_text))
    asks_total = bool(_TOTAL_PATTERN.search(question_text))
    asks_difference = bool(_DIFFERENCE_PATTERN.search(question_text))
    asks_rate_or_unit = bool(_RATE_OR_UNIT_PATTERN.search(question_text))
    asks_money = bool(_MONEY_PATTERN.search(question_text))
    asks_time = bool(_TIME_PATTERN.search(question_text))

    # B. Wording-trap signals
    has_subtraction_trap_verb = bool(_SUBTRACTION_TRAP_PATTERN.search(question_text))
    has_addition_trap_structure = bool(_ADDITION_TRAP_PATTERN.search(question_text))
    operation_verbs = _OPERATION_VERB_PATTERN.findall(question_text)
    unique_verbs = {v.lower() for v in operation_verbs}
    has_multi_operation_hint = len(unique_verbs) >= 2

    # C. Answer-risk signals

    # likely_intermediate_quantity_ask:
    # Multiple sentences + multiple numbers + no "total" or "remaining" cue.
    # This configuration has multiple plausible numeric endpoints along the
    # reasoning chain and no surface cue anchoring which one is final.
    no_anchor = not asks_total and not asks_remaining_or_left
    likely_intermediate_quantity_ask = (
        n_sentences >= _MIN_SENTENCES_FOR_INTERMEDIATE_RISK
        and n_numbers >= _MIN_NUMBERS_FOR_INTERMEDIATE_RISK
        and no_anchor
    )

    # potential_answer_echo_risk:
    # Many numeric tokens + short final question sentence.
    # The model has many candidates to echo from the prompt.
    short_final_question = False
    if sentences:
        last = sentences[-1]
        if len(last.split()) <= _FINAL_SENTENCE_MAX_TOKENS:
            short_final_question = True
    potential_answer_echo_risk = (
        n_numbers >= _MIN_NUMBERS_FOR_ECHO_RISK and short_final_question
    )

    return {
        # A — target-type cues
        "asks_remaining_or_left": asks_remaining_or_left,
        "asks_total": asks_total,
        "asks_difference": asks_difference,
        "asks_rate_or_unit": asks_rate_or_unit,
        "asks_money": asks_money,
        "asks_time": asks_time,
        # B — wording-trap signals
        "has_subtraction_trap_verb": has_subtraction_trap_verb,
        "has_addition_trap_structure": has_addition_trap_structure,
        "has_multi_operation_hint": has_multi_operation_hint,
        # C — answer-risk signals
        "likely_intermediate_quantity_ask": likely_intermediate_quantity_ask,
        "potential_answer_echo_risk": potential_answer_echo_risk,
    }
