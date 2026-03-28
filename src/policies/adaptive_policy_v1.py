"""Adaptive policy v1: simple rule-based strategy router.

This first version of ``pi(x, z)`` is intentionally small and interpretable.
It uses:

- lightweight question-side features ``z(x)`` to choose direct vs reasoning
- lightweight first-pass output features to decide whether to keep the
  reasoning answer or escalate to an existing fallback strategy

The current repository branch does not contain a standalone
``precompute_features.py`` file, so v1 computes the same style of cheap
features on the fly while reusing existing output-stability heuristics from the
selective-escalation code path.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from src.methods.selective_escalation import parse_numeric_details

NUMBER_RE = re.compile(r"-?[\d,]+(?:\.\d+)?")
UNCERTAINTY_RE = re.compile(
    (
        r"\b(?:maybe|perhaps|probably|possibly|i think|i'm not sure|"
        r"not sure|guess|approximately|about)\b"
    ),
    re.IGNORECASE,
)
MULTI_STEP_CUES = (
    "after",
    "before",
    "then",
    "left",
    "remain",
    "remaining",
    "total",
    "altogether",
    "each",
    "every",
    "per",
    "share",
    "split",
    "twice",
    "times",
    "more than",
    "less than",
)


@dataclass(frozen=True)
class AdaptivePolicyV1Config:
    """Small set of interpretable thresholds for the rule-based router."""

    simple_max_numeric_mentions: int = 2
    simple_max_word_count: int = 35
    high_complexity_numeric_mentions: int = 4
    high_complexity_word_count: int = 60
    unstable_max_output_numbers: int = 4
    allow_reasoning_best_of_3: bool = True
    allow_strong_direct: bool = False


def _raw_question_features(
    question_text: str,
) -> dict[str, Any]:
    lowered = question_text.lower()
    numeric_mentions = NUMBER_RE.findall(question_text)
    word_count = len(question_text.split())
    has_multi_step_cue = any(cue in lowered for cue in MULTI_STEP_CUES)
    return {
        "question_length_words": word_count,
        "question_length_chars": len(question_text),
        "num_numeric_mentions": len(numeric_mentions),
        "has_multi_step_cue": has_multi_step_cue,
        "multi_step_cue": has_multi_step_cue,
    }


def extract_question_features(
    question_text: str,
    config: AdaptivePolicyV1Config | None = None,
) -> dict[str, Any]:
    """Compute cheap question-side features ``z(x)`` for routing."""
    resolved = config or AdaptivePolicyV1Config()
    features = _raw_question_features(question_text)
    features.update(enrich_question_features(question_text, features, resolved))
    features["simple_question"] = bool(features["is_simple"])
    return features


def enrich_question_features(
    question_text: str,
    features: dict[str, Any] | None,
    config: AdaptivePolicyV1Config,
) -> dict[str, Any]:
    """Merge supplied features with derived convenience booleans."""
    base = _raw_question_features(question_text)
    if features is not None:
        base.update(features)

    num_numeric_mentions = int(base.get("num_numeric_mentions", 0))
    word_count = int(base.get("question_length_words", len(question_text.split())))
    has_multi_step_cue = bool(base.get("has_multi_step_cue", False))

    is_simple = (
        num_numeric_mentions <= config.simple_max_numeric_mentions
        and not has_multi_step_cue
        and word_count <= config.simple_max_word_count
    )
    is_high_complexity = (
        num_numeric_mentions >= config.high_complexity_numeric_mentions
        or (has_multi_step_cue and word_count >= config.high_complexity_word_count)
    )
    base["is_simple"] = is_simple
    base["is_high_complexity"] = is_high_complexity
    return base


def extract_first_pass_features(
    first_pass_output: str,
    config: AdaptivePolicyV1Config,
) -> dict[str, Any]:
    """Extract cheap output-stability features from one reasoning pass."""
    details = parse_numeric_details(first_pass_output)
    uncertainty = bool(UNCERTAINTY_RE.search(first_pass_output))
    num_output_numbers = len(NUMBER_RE.findall(first_pass_output))
    too_many_numbers = num_output_numbers >= config.unstable_max_output_numbers
    unstable = bool(
        details["parse_failure"]
        or details["malformed_output"]
        or uncertainty
        or too_many_numbers
    )
    return {
        "parsed_answer": details["parsed_answer"],
        "parse_failure": bool(details["parse_failure"]),
        "malformed_output": bool(details["malformed_output"]),
        "low_confidence_format": bool(details["low_confidence_format"]),
        "contains_uncertainty_phrase": uncertainty,
        "num_output_numbers": num_output_numbers,
        "too_many_numbers": too_many_numbers,
        "unstable_output": unstable,
        "looks_unstable": unstable,
    }


def extract_output_stability_features(
    first_pass_output: str,
    config: AdaptivePolicyV1Config | None = None,
) -> dict[str, Any]:
    """Convenience wrapper with default thresholds for tests/callers."""
    return extract_first_pass_features(
        first_pass_output,
        config or AdaptivePolicyV1Config(),
    )


def choose_strategy(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV1Config | None = None,
    policy_config: AdaptivePolicyV1Config | None = None,
    enable_strong_direct: bool | None = None,
) -> str:
    """Choose a strategy using question features and optional first-pass output.

    Routing rules:
    1. Simple questions -> ``direct_greedy``
    2. Otherwise -> ``reasoning_greedy``
    3. After reasoning: unstable output -> ``direct_plus_revise``
    4. Rare fallback: high-complexity + many-number reasoning output ->
       ``reasoning_best_of_3``
    5. Optional strong-model escape hatch for severe failure on high-complexity
       questions -> ``strong_direct``
    """
    resolved = policy_config or config or AdaptivePolicyV1Config()
    if enable_strong_direct is not None:
        resolved = AdaptivePolicyV1Config(
            simple_max_numeric_mentions=resolved.simple_max_numeric_mentions,
            simple_max_word_count=resolved.simple_max_word_count,
            high_complexity_numeric_mentions=resolved.high_complexity_numeric_mentions,
            high_complexity_word_count=resolved.high_complexity_word_count,
            unstable_max_output_numbers=resolved.unstable_max_output_numbers,
            allow_reasoning_best_of_3=resolved.allow_reasoning_best_of_3,
            allow_strong_direct=bool(enable_strong_direct),
        )
    enriched = enrich_question_features(question_text, features, resolved)

    if first_pass_output is None:
        return "direct_greedy" if enriched["is_simple"] else "reasoning_greedy"

    output_features = extract_first_pass_features(first_pass_output, resolved)
    if not output_features["unstable_output"]:
        return "reasoning_greedy"

    if (
        resolved.allow_strong_direct
        and enriched["is_high_complexity"]
        and (
            output_features["parse_failure"]
            or output_features["malformed_output"]
            or output_features["contains_uncertainty_phrase"]
        )
    ):
        return "strong_direct"

    if (
        resolved.allow_reasoning_best_of_3
        and enriched["is_high_complexity"]
        and output_features["too_many_numbers"]
    ):
        return "reasoning_best_of_3"

    return "direct_plus_revise"


def explain_policy_decision(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV1Config | None = None,
) -> dict[str, Any]:
    """Return the policy decision plus the derived feature state for logging."""
    resolved = config or AdaptivePolicyV1Config()
    enriched = enrich_question_features(question_text, features, resolved)
    output_features = (
        None
        if first_pass_output is None
        else extract_first_pass_features(first_pass_output, resolved)
    )
    chosen = choose_strategy(
        question_text=question_text,
        features=enriched,
        first_pass_output=first_pass_output,
        config=resolved,
    )
    return {
        "chosen_strategy": chosen,
        "question_features": enriched,
        "first_pass_features": output_features,
        "config": {
            "simple_max_numeric_mentions": resolved.simple_max_numeric_mentions,
            "simple_max_word_count": resolved.simple_max_word_count,
            "high_complexity_numeric_mentions": resolved.high_complexity_numeric_mentions,
            "high_complexity_word_count": resolved.high_complexity_word_count,
            "unstable_max_output_numbers": resolved.unstable_max_output_numbers,
            "allow_reasoning_best_of_3": resolved.allow_reasoning_best_of_3,
            "allow_strong_direct": resolved.allow_strong_direct,
        },
    }


def explain_policy_decision_json(
    question_text: str,
    features: dict[str, Any],
    first_pass_output: str | None = None,
    config: AdaptivePolicyV1Config | None = None,
) -> str:
    """Convenience JSON renderer for CSV/logging."""
    return json.dumps(
        explain_policy_decision(
            question_text=question_text,
            features=features,
            first_pass_output=first_pass_output,
            config=config,
        ),
        sort_keys=True,
    )
