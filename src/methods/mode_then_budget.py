"""Mode-then-budget hybrid prototype for adaptive real-data reasoning.

This prototype is motivated by current findings that:
1) direct greedy prompting is strong and cheap,
2) reasoning prompting alone is weak on average,
3) reasoning plus extra compute can help some queries, and
4) adaptive methods therefore need to choose both inference mode and compute.

The method is intentionally simple and conservative:
- every query gets one cheap direct answer,
- uncertain queries may receive one cheap reasoning probe,
- only top-scoring candidates switch fully into reasoning mode, and
- switched queries receive a small reasoning budget up to a target ``k``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.utils.answer_extraction import extract_numeric_answer


@dataclass
class ModeThenBudgetConfig:
    """Configuration for the simple rule-based mode-then-budget hybrid."""

    total_budget: int
    reasoning_target_k: int = 3
    use_reasoning_probe: bool = True
    weight_parse_failure: float = 2.0
    weight_malformed_output: float = 1.0
    weight_low_confidence_format: float = 1.0
    weight_direct_reasoning_disagreement: float = 1.5
    malformed_length_threshold: int = 2
    min_switch_score: float = 1.5


def _normalize_numeric(value: str) -> str:
    return extract_numeric_answer(value)


def compute_direct_mode_signals(
    direct_output: str,
    direct_answer: str,
    reasoning_probe_output: str | None = None,
    reasoning_probe_answer: str | None = None,
    malformed_length_threshold: int = 2,
) -> dict[str, Any]:
    """Compute cheap routing signals from direct output and optional reasoning probe."""
    stripped = direct_output.strip()
    parse_failure = direct_answer == ""
    malformed_output = stripped == "" or (
        len(stripped) <= malformed_length_threshold and parse_failure
    )
    lowered = stripped.lower()
    has_final_answer_marker = "answer:" in lowered or "final answer" in lowered
    numeric_only = stripped.replace(",", "").replace("$", "").replace(".", "", 1).isdigit()
    low_confidence_format = (
        (not parse_failure)
        and (not has_final_answer_marker)
        and (not numeric_only)
    )

    direct_reasoning_disagreement = False
    if reasoning_probe_answer is not None:
        direct_reasoning_disagreement = direct_answer != reasoning_probe_answer

    return {
        "parse_failure": bool(parse_failure),
        "malformed_output": bool(malformed_output),
        "low_confidence_format": bool(low_confidence_format),
        "direct_reasoning_disagreement": bool(direct_reasoning_disagreement),
        "reasoning_probe_output": reasoning_probe_output,
        "reasoning_probe_answer": reasoning_probe_answer,
    }


def score_mode_switch(
    signals: dict[str, Any],
    config: ModeThenBudgetConfig,
) -> float:
    """Score whether a query should switch from direct to reasoning mode."""
    score = 0.0
    if signals["parse_failure"]:
        score += config.weight_parse_failure
    if signals["malformed_output"]:
        score += config.weight_malformed_output
    if signals["low_confidence_format"]:
        score += config.weight_low_confidence_format
    if signals["direct_reasoning_disagreement"]:
        score += config.weight_direct_reasoning_disagreement
    return float(score)


def direct_mode_signals(
    first_output: str,
    auxiliary_output: str | None = None,
    malformed_length_threshold: int = 2,
) -> dict[str, Any]:
    """Compatibility wrapper used in tests.

    The direct answer is extracted from the first output, and the optional
    auxiliary answer is treated as a cheap disagreement probe.
    """
    direct_answer = _normalize_numeric(first_output)
    auxiliary_answer = None if auxiliary_output is None else _normalize_numeric(auxiliary_output)
    signals = compute_direct_mode_signals(
        direct_output=first_output,
        direct_answer=direct_answer,
        reasoning_probe_output=auxiliary_output,
        reasoning_probe_answer=auxiliary_answer,
        malformed_length_threshold=malformed_length_threshold,
    )
    return {
        "parse_failure": signals["parse_failure"],
        "malformed_output": signals["malformed_output"],
        "low_confidence_format": signals["low_confidence_format"],
        "disagreement_probe": signals["direct_reasoning_disagreement"],
    }


def decide_routing(
    query_records: list[dict[str, Any]],
    total_budget: int,
    first_pass_cost: int,
    reasoning_target_k: int,
    switch_threshold: float,
) -> dict[str, bool]:
    """Compatibility helper for deterministic routing decisions in tests."""
    if first_pass_cost <= 0:
        raise ValueError("first_pass_cost must be positive")
    remaining_budget = int(total_budget) - len(query_records) * int(first_pass_cost)
    extra_cost_per_switched_query = max(0, int(reasoning_target_k) - int(first_pass_cost))
    ranked = sorted(
        query_records,
        key=lambda row: (-float(row["switch_score"]), str(row["question_id"])),
    )
    decisions = {str(row["question_id"]): False for row in query_records}
    for row in ranked:
        if float(row["switch_score"]) < float(switch_threshold):
            break
        if extra_cost_per_switched_query > remaining_budget:
            break
        decisions[str(row["question_id"])] = True
        remaining_budget -= extra_cost_per_switched_query
    return decisions


def _majority_vote(values: list[str]) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: (-item[1], values.index(item[0]), item[0]))
    return ranked[0][0]


def run_mode_then_budget(
    direct_model: Any,
    reasoning_model: Any | None = None,
    questions: list[str] | None = None,
    total_budget: int | None = None,
    config: ModeThenBudgetConfig | None = None,
    model: Any | None = None,
) -> dict[str, Any]:
    """Run the mode-then-budget hybrid under a global sample budget.

    Output rows are unlabeled method diagnostics; evaluation against gold answers
    is handled separately to keep the method reusable.
    """
    if model is not None:
        direct_model = model
        reasoning_model = model
    if reasoning_model is None:
        raise ValueError("reasoning_model must be provided")
    if questions is None or not questions:
        raise ValueError("questions must be non-empty")
    resolved = (
        config
        if config is not None
        else ModeThenBudgetConfig(total_budget=int(total_budget) if total_budget is not None else 0)
    )
    if total_budget is not None and resolved.total_budget != int(total_budget):
        resolved = ModeThenBudgetConfig(
            total_budget=int(total_budget),
            reasoning_target_k=resolved.reasoning_target_k,
            use_reasoning_probe=resolved.use_reasoning_probe,
            weight_parse_failure=resolved.weight_parse_failure,
            weight_malformed_output=resolved.weight_malformed_output,
            weight_low_confidence_format=resolved.weight_low_confidence_format,
            weight_direct_reasoning_disagreement=resolved.weight_direct_reasoning_disagreement,
            malformed_length_threshold=resolved.malformed_length_threshold,
            min_switch_score=resolved.min_switch_score,
        )
    if resolved.total_budget < len(questions):
        raise ValueError(
            f"Budget {resolved.total_budget} is infeasible for {len(questions)} queries: "
            "at least one direct sample per query is required."
        )

    rows: list[dict[str, Any]] = []
    total_samples_used = 0

    # Stage 1: one cheap direct pass for every query.
    for question in questions:
        direct_output = direct_model.generate(question)
        total_samples_used += 1
        direct_answer = _normalize_numeric(direct_output)

        reasoning_probe_output = None
        reasoning_probe_answer = None
        used_reasoning_probe = False
        if resolved.use_reasoning_probe and total_samples_used < resolved.total_budget:
            signals_for_probe = compute_direct_mode_signals(
                direct_output=direct_output,
                direct_answer=direct_answer,
                malformed_length_threshold=resolved.malformed_length_threshold,
            )
            if (
                signals_for_probe["parse_failure"]
                or signals_for_probe["malformed_output"]
                or signals_for_probe["low_confidence_format"]
            ):
                reasoning_probe_output = reasoning_model.generate(question)
                reasoning_probe_answer = _normalize_numeric(reasoning_probe_output)
                used_reasoning_probe = True
                total_samples_used += 1

        signals = compute_direct_mode_signals(
                direct_output=direct_output,
                direct_answer=direct_answer,
                reasoning_probe_output=reasoning_probe_output,
                reasoning_probe_answer=reasoning_probe_answer,
                malformed_length_threshold=resolved.malformed_length_threshold,
        )
        switch_score = score_mode_switch(signals, resolved)

        reasoning_answers = (
            []
            if reasoning_probe_answer is None
            else [reasoning_probe_answer]
        )

        rows.append(
            {
                "question": question,
                "direct_output": direct_output,
                "direct_answer": direct_answer,
                "switch_score": switch_score,
                "switch_signals": {
                    key: bool(signals[key])
                    for key in (
                        "parse_failure",
                        "malformed_output",
                        "low_confidence_format",
                        "direct_reasoning_disagreement",
                    )
                },
                "used_reasoning_probe": used_reasoning_probe,
                "reasoning_probe_output": reasoning_probe_output,
                "reasoning_probe_answer": reasoning_probe_answer,
                "reasoning_answers": reasoning_answers,
                "switched_to_reasoning": False,
                "final_mode": "direct",
                "final_answer": direct_answer,
                "samples_used": 1 + len(reasoning_answers),
            }
        )

    # Stage 3: use the remaining budget only on top-scoring reasoning candidates.
    remaining_budget = resolved.total_budget - total_samples_used
    ranked = sorted(
        range(len(rows)),
        key=lambda idx: (-float(rows[idx]["switch_score"]), idx),
    )
    for idx in ranked:
        if remaining_budget <= 0:
            break
        if rows[idx]["switch_score"] < resolved.min_switch_score:
            break

        already_have = len(rows[idx]["reasoning_answers"])
        needed = max(0, resolved.reasoning_target_k - already_have)
        if needed == 0 or needed > remaining_budget:
            continue

        extra_outputs = reasoning_model.generate_n(rows[idx]["question"], needed)
        extra_answers = [_normalize_numeric(output) for output in extra_outputs]
        rows[idx]["reasoning_answers"].extend(extra_answers)
        rows[idx]["switched_to_reasoning"] = True
        rows[idx]["final_mode"] = "reasoning"
        rows[idx]["final_answer"] = _majority_vote(rows[idx]["reasoning_answers"])
        rows[idx]["samples_used"] = 1 + len(rows[idx]["reasoning_answers"])
        remaining_budget -= needed

    for row in rows:
        row["signals_fired"] = [
            key for key, fired in row["switch_signals"].items() if bool(fired)
        ]

    return {
        "rows": rows,
        "diagnostics": rows,
        "total_samples_used": sum(int(row["samples_used"]) for row in rows),
        "queries_switched_to_reasoning": sum(
            int(bool(row["switched_to_reasoning"])) for row in rows
        ),
    }
