"""Selective compute escalation prototype for real LLM reasoning tasks.

This is a first proposed-method prototype, not a final production system. It is
motivated by current findings that:
1) real-data headroom appears limited,
2) extra compute helps only a minority of queries, and
3) naive always-more-compute is inefficient.

The method is intentionally conservative:
- every query gets a first-pass answer,
- only queries with strong cheap escalation signals are reconsidered, and
- extra samples are allocated only to the top-scoring queries under a budget.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
import re
from typing import Any

from src.models.openai_llm import OpenAILLMModel
from src.utils.answer_extraction import extract_numeric_answer

NUMERIC_PATTERN = re.compile(r"-?[\d,]+\.?\d*")
NUMERIC_ONLY_PATTERN = re.compile(r"^\s*-?[\d,]+(?:\.\d*)?\s*$")


@dataclass
class SelectiveEscalationConfig:
    """Rule-based configuration for selective escalation."""

    total_budget: int
    escalation_target_k: int = 3
    use_second_sample_for_disagreement: bool = True
    weight_parse_failure: float = 1.0
    weight_disagreement_2sample: float = 1.0
    weight_malformed_output: float = 0.5
    weight_low_confidence_format: float = 0.5
    malformed_length_threshold: int = 2
    min_score_to_escalate: float = 1.5
    enable_post_ranking_escalation: bool = True


def _normalize_numeric(value: str) -> str:
    cleaned = value.strip().replace(",", "")
    if not cleaned:
        return ""
    try:
        normalized = Decimal(cleaned)
    except InvalidOperation:
        return cleaned
    rendered = format(normalized.normalize(), "f")
    if "." in rendered:
        rendered = rendered.rstrip("0").rstrip(".")
    return rendered or "0"


def parse_numeric_details(text: str, malformed_length_threshold: int = 2) -> dict[str, Any]:
    """Extract lightweight formatting and parseability signals from one output."""
    extracted = extract_numeric_answer(text)
    normalized = _normalize_numeric(extracted)
    stripped = text.strip()
    has_numeric_token = bool(NUMERIC_PATTERN.search(stripped))
    numeric_only = bool(NUMERIC_ONLY_PATTERN.fullmatch(stripped))

    parse_failure = normalized == "" or not has_numeric_token
    malformed_output = stripped == "" or (
        len(stripped) <= malformed_length_threshold and parse_failure
    )
    lowered = stripped.lower()
    has_final_answer_marker = "answer:" in lowered or "final answer" in lowered
    low_confidence_format = (
        (not parse_failure)
        and (not has_final_answer_marker)
        and (not numeric_only)
    )

    return {
        "raw_text": text,
        "raw_extracted_answer": extracted,
        "parsed_answer": normalized,
        "parse_failure": parse_failure,
        "malformed_output": malformed_output,
        "low_confidence_format": low_confidence_format,
    }


def compute_escalation_signals(
    first_output: str,
    second_output: str | None,
    malformed_length_threshold: int = 2,
) -> dict[str, Any]:
    """Compute cheap observable signals from one or two low-cost samples."""
    first = parse_numeric_details(
        first_output,
        malformed_length_threshold=malformed_length_threshold,
    )
    second = (
        None
        if second_output is None
        else parse_numeric_details(
            second_output,
            malformed_length_threshold=malformed_length_threshold,
        )
    )

    disagreement = False
    if second is not None:
        disagreement = first["parsed_answer"] != second["parsed_answer"]

    return {
        "first_output": first,
        "second_output": second,
        "parse_failure": bool(first["parse_failure"]),
        "malformed_output": bool(first["malformed_output"]),
        "low_confidence_format": bool(first["low_confidence_format"]),
        "disagreement_2sample": bool(disagreement),
    }


def score_escalation(signals: dict[str, Any], config: SelectiveEscalationConfig) -> float:
    """Turn binary diagnostic signals into a deterministic escalation score."""
    score = 0.0
    if signals["parse_failure"]:
        score += config.weight_parse_failure
    if signals["disagreement_2sample"]:
        score += config.weight_disagreement_2sample
    if signals["malformed_output"]:
        score += config.weight_malformed_output
    if signals["low_confidence_format"]:
        score += config.weight_low_confidence_format
    return float(score)


def _majority_vote(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], values.index(item[0]), item[0]))
    return ranked[0][0]


def run_selective_escalation(
    model: OpenAILLMModel,
    questions: list[str],
    total_budget: int,
    config: SelectiveEscalationConfig | None = None,
) -> dict[str, Any]:
    """Run conservative selective escalation under a global sample budget.

    Returns per-query diagnostics but does not require labels; evaluation is
    handled separately so the method stays reusable.

    Accounting semantics:
    - ``used_second_sample_for_gating`` means the query consumed a second cheap
      sample only for uncertainty checking / gate-only majority vote.
    - ``escalated_beyond_gating_stage`` means the query received additional
      post-ranking compute beyond the cheap gating stage.
    - ``used_more_than_1_sample`` covers both of the above.
    """
    resolved = (
        config
        if config is not None
        else SelectiveEscalationConfig(total_budget=int(total_budget))
    )
    if resolved.total_budget != int(total_budget):
        resolved = SelectiveEscalationConfig(
            total_budget=int(total_budget),
            escalation_target_k=resolved.escalation_target_k,
            use_second_sample_for_disagreement=resolved.use_second_sample_for_disagreement,
            weight_parse_failure=resolved.weight_parse_failure,
            weight_disagreement_2sample=resolved.weight_disagreement_2sample,
            weight_malformed_output=resolved.weight_malformed_output,
            weight_low_confidence_format=resolved.weight_low_confidence_format,
            malformed_length_threshold=resolved.malformed_length_threshold,
            min_score_to_escalate=resolved.min_score_to_escalate,
            enable_post_ranking_escalation=resolved.enable_post_ranking_escalation,
        )

    n_queries = len(questions)
    if n_queries == 0:
        raise ValueError("questions must be non-empty")
    if resolved.total_budget < n_queries:
        raise ValueError(
            f"Budget {resolved.total_budget} is infeasible for {n_queries} queries: "
            "at least one sample per query is required."
        )
    if resolved.escalation_target_k < 1:
        raise ValueError("escalation_target_k must be >= 1")

    diagnostics: list[dict[str, Any]] = []
    total_samples_used = 0

    # Stage 1: one cheap first pass for every query.
    for question in questions:
        first_output = model.generate(question)
        total_samples_used += 1

        second_output = None
        second_parsed = None
        first_details = parse_numeric_details(
            first_output,
            malformed_length_threshold=resolved.malformed_length_threshold,
        )
        initial_uncertainty_signals = (
            bool(first_details["parse_failure"])
            or bool(first_details["malformed_output"])
            or bool(first_details["low_confidence_format"])
        )
        used_second_sample_for_gating = False
        if (
            resolved.use_second_sample_for_disagreement
            and initial_uncertainty_signals
            and total_samples_used < resolved.total_budget
        ):
            second_output = model.generate(question)
            total_samples_used += 1
            used_second_sample_for_gating = True
            second_details = parse_numeric_details(
                second_output,
                malformed_length_threshold=resolved.malformed_length_threshold,
            )
            second_parsed = second_details["parsed_answer"]

        signals = compute_escalation_signals(
            first_output=first_output,
            second_output=second_output,
            malformed_length_threshold=resolved.malformed_length_threshold,
        )
        first_parsed = first_details["parsed_answer"]
        score = score_escalation(signals, resolved)

        candidate_answers = [first_parsed]
        if second_parsed is not None:
            candidate_answers.append(second_parsed)

        diagnostics.append(
            {
                "question": question,
                "first_pass_answer": first_parsed,
                "first_pass_raw_extracted_answer": first_details["raw_extracted_answer"],
                "first_pass_raw_output": first_output,
                "normalization_changed_first_pass_answer": (
                    first_details["raw_extracted_answer"] != first_parsed
                ),
                "used_second_sample_for_gating": used_second_sample_for_gating,
                "second_sample_answer": second_parsed,
                "second_sample_raw_output": second_output,
                "escalation_score": score,
                "signals": {
                    key: bool(signals[key])
                    for key in (
                        "parse_failure",
                        "disagreement_2sample",
                        "malformed_output",
                        "low_confidence_format",
                    )
                },
                "gating_stage_answers": list(candidate_answers),
                "candidate_answers": candidate_answers,
                "samples_used": len(candidate_answers),
                "used_more_than_1_sample": len(candidate_answers) > 1,
                "escalated_beyond_gating_stage": False,
            }
        )

    # Stage 3: spend remaining budget only on the strongest flagged queries.
    remaining_budget = resolved.total_budget - total_samples_used
    ranked_indices = sorted(
        range(n_queries),
        key=lambda idx: (
            -float(diagnostics[idx]["escalation_score"]),
            idx,
        ),
    )

    for idx in ranked_indices:
        if not resolved.enable_post_ranking_escalation:
            break
        if remaining_budget <= 0:
            break
        if diagnostics[idx]["escalation_score"] < resolved.min_score_to_escalate:
            break

        already_have = len(diagnostics[idx]["candidate_answers"])
        target_total = resolved.escalation_target_k
        needed = max(0, target_total - already_have)
        if needed == 0 or needed > remaining_budget:
            continue

        extra_outputs = model.generate_n(diagnostics[idx]["question"], needed)
        extra_answers = [
            parse_numeric_details(
                output,
                malformed_length_threshold=resolved.malformed_length_threshold,
            )["parsed_answer"]
            for output in extra_outputs
        ]
        diagnostics[idx]["candidate_answers"].extend(extra_answers)
        diagnostics[idx]["samples_used"] += needed
        diagnostics[idx]["used_more_than_1_sample"] = (
            diagnostics[idx]["samples_used"] > 1
        )
        diagnostics[idx]["escalated_beyond_gating_stage"] = True
        remaining_budget -= needed

    for item in diagnostics:
        item["final_answer"] = _majority_vote(item["candidate_answers"])
        item["signals_fired"] = [
            key for key, fired in item["signals"].items() if bool(fired)
        ]

    return {
        "diagnostics": diagnostics,
        "total_samples_used": sum(int(item["samples_used"]) for item in diagnostics),
        "queries_with_more_than_1_sample": sum(
            int(bool(item["used_more_than_1_sample"])) for item in diagnostics
        ),
        "queries_escalated_beyond_gating_stage": sum(
            int(bool(item["escalated_beyond_gating_stage"])) for item in diagnostics
        ),
    }
