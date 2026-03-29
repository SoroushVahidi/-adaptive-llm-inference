"""Expanded strategy evaluation: six inference strategies compared on GSM8K.

Builds on top of strategy_expansion_eval and adds:
  - New prompt families:  critique, hint-guided reasoning
  - New stage structures: direct_plus_critique_plus_final,
                          first_pass_then_hint_guided_reason

Full strategy list:
  1. direct_greedy                   (inherited from strategy_expansion_eval)
  2. structured_sampling_3           (inherited)
  3. direct_plus_verify              (inherited)
  4. direct_plus_revise              (inherited)
  5. direct_plus_critique_plus_final (new – 3-stage: direct → critique → final)
  6. first_pass_then_hint_guided_reason (new – 2-stage: direct → hint-guided re-solve)

The optional ``strong_direct`` strategy is included when a ``strong_model``
config key is present and points to a different (stronger) model name.  It
uses the same one-shot direct approach but with greedy temperature.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

# Re-export the inherited strategies so callers only need this module.
from src.evaluation.strategy_expansion_eval import (  # noqa: F401
    run_direct_greedy,
    run_direct_plus_revise,
    run_direct_plus_verify,
    run_reasoning_greedy,
    run_reasoning_then_revise,
    run_structured_sampling_3,
)
from src.utils.answer_extraction import extract_numeric_answer

# ---------------------------------------------------------------------------
# Typing shim
# ---------------------------------------------------------------------------

class _ModelProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...
    def generate_n(self, prompt: str, n: int) -> list[str]: ...


# ---------------------------------------------------------------------------
# Prompt templates — new families
# ---------------------------------------------------------------------------

# --- Critique prompt ---
# Stage 1: direct answer (re-uses _DIRECT pattern from strategy_expansion_eval)
# Stage 2: critique prompt asks for a detailed fault-finding pass
_CRITIQUE_PROMPT = (
    "Question: {question}\n\n"
    "A student gave this answer: {first_answer}\n\n"
    "Critique that answer: identify any reasoning errors or calculation mistakes. "
    "Then state the single correct numeric answer on its own line as "
    "'Corrected answer: <number>'."
)

# Stage 3: final answer prompt uses the critique to produce a definitive answer
_FINAL_AFTER_CRITIQUE_PROMPT = (
    "Question: {question}\n\n"
    "Initial answer: {first_answer}\n"
    "Critique: {critique}\n\n"
    "Based on the critique above, state the final numeric answer on its own line as "
    "'Final answer: <number>'."
)

# --- Hint-guided reasoning prompt ---
# Stage 1: direct answer (greedy)
# Stage 2: re-solve with a hint derived from the first answer
_HINT_GUIDED_PROMPT = (
    "Question: {question}\n\n"
    "Hint: a preliminary estimate gives {hint}. Use this as a sanity check "
    "while solving the problem step by step. "
    "End with 'Final answer: <number>'."
)


# ---------------------------------------------------------------------------
# Numeric normalization (mirror of strategy_expansion_eval to avoid coupling)
# ---------------------------------------------------------------------------

def _normalize(value: str) -> str:
    candidate = value.strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        number = Decimal(candidate)
    except InvalidOperation:
        return candidate
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _majority_vote(answers: list[str]) -> str:
    if not answers:
        return ""
    normalized = [_normalize(a) for a in answers]
    counter: Counter[str] = Counter(normalized)
    return counter.most_common(1)[0][0]


# ---------------------------------------------------------------------------
# New strategy runners
# ---------------------------------------------------------------------------

def run_direct_plus_critique_plus_final(
    model: _ModelProtocol,
    question: str,
) -> dict[str, Any]:
    """Strategy E-NEW: 3-stage pipeline — direct → critique → final answer.

    Stage 1: Greedy direct answer.
    Stage 2: Critique the first answer for errors.
    Stage 3: Generate the final corrected answer guided by the critique.
    """
    # Stage 1 – direct answer
    stage1_raw = model.generate(question)
    first_answer = _normalize(extract_numeric_answer(stage1_raw))

    # Stage 2 – critique
    critique_prompt = _CRITIQUE_PROMPT.format(
        question=question,
        first_answer=first_answer or stage1_raw[:200],
    )
    critique_raw = model.generate(critique_prompt)
    critique_text = critique_raw.strip()

    # Stage 3 – final answer guided by critique
    final_prompt = _FINAL_AFTER_CRITIQUE_PROMPT.format(
        question=question,
        first_answer=first_answer or stage1_raw[:200],
        critique=critique_text[:500],
    )
    final_raw = model.generate(final_prompt)
    final_answer = _normalize(extract_numeric_answer(final_raw))

    # If stage 3 extraction fails, try extracting from critique directly
    if not final_answer:
        final_answer = _normalize(extract_numeric_answer(critique_raw))
    # Last resort: keep first answer
    if not final_answer:
        final_answer = first_answer

    return {
        "raw_outputs": [stage1_raw, critique_raw, final_raw],
        "predicted_answer": final_answer,
        "samples_used": 3,
        "first_answer": first_answer,
        "critique_text": critique_text,
        "revised_answer": final_answer,
    }


def run_first_pass_then_hint_guided_reason(
    model: _ModelProtocol,
    question: str,
) -> dict[str, Any]:
    """Strategy F-NEW: 2-stage pipeline — direct → hint-guided re-solve.

    Stage 1: Greedy direct answer (the "hint").
    Stage 2: Re-solve the problem using the first answer as a ballpark hint
             and apply step-by-step reasoning.
    """
    # Stage 1 – quick direct pass to get a hint value
    stage1_raw = model.generate(question)
    hint_answer = _normalize(extract_numeric_answer(stage1_raw))

    # Stage 2 – hint-guided reasoning
    hint_prompt = _HINT_GUIDED_PROMPT.format(
        question=question,
        hint=hint_answer if hint_answer else "unknown",
    )
    stage2_raw = model.generate(hint_prompt)
    final_answer = _normalize(extract_numeric_answer(stage2_raw))

    if not final_answer:
        final_answer = hint_answer  # fall back to stage-1 answer

    return {
        "raw_outputs": [stage1_raw, stage2_raw],
        "predicted_answer": final_answer,
        "samples_used": 2,
        "first_answer": hint_answer,
        "revised_answer": final_answer,
    }


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

_EXPANDED_STRATEGY_RUNNERS = {
    "direct_greedy": run_direct_greedy,
    "reasoning_greedy": run_reasoning_greedy,
    "reasoning_then_revise": run_reasoning_then_revise,
    "structured_sampling_3": run_structured_sampling_3,
    "direct_plus_verify": run_direct_plus_verify,
    "direct_plus_revise": run_direct_plus_revise,
    "direct_plus_critique_plus_final": run_direct_plus_critique_plus_final,
    "first_pass_then_hint_guided_reason": run_first_pass_then_hint_guided_reason,
}

ALL_EXPANDED_STRATEGIES = list(_EXPANDED_STRATEGY_RUNNERS.keys())


def run_expanded_strategy(
    strategy: str,
    model: _ModelProtocol,
    question: str,
) -> dict[str, Any]:
    """Dispatch to the correct strategy runner."""
    if strategy not in _EXPANDED_STRATEGY_RUNNERS:
        raise ValueError(
            f"Unknown strategy '{strategy}'. Valid: {ALL_EXPANDED_STRATEGIES}"
        )
    return _EXPANDED_STRATEGY_RUNNERS[strategy](model, question)


# ---------------------------------------------------------------------------
# Evaluation loop
# ---------------------------------------------------------------------------

def run_expanded_strategy_eval(
    model: _ModelProtocol,
    queries: list[Any],
    strategies: list[str] | None = None,
) -> dict[str, Any]:
    """Run all (or a subset of) expanded strategies over the provided queries.

    Args:
        model: Any object with ``generate(prompt)`` and ``generate_n(prompt, n)``.
        queries: List of objects with ``.id``, ``.question``, ``.answer`` fields.
        strategies: Which strategies to run. Defaults to ALL_EXPANDED_STRATEGIES.

    Returns:
        Dict with per-query rows and per-strategy summaries.
    """
    if strategies is None:
        strategies = ALL_EXPANDED_STRATEGIES

    per_query_rows: list[dict[str, Any]] = []

    for query in queries:
        for strategy in strategies:
            result = run_expanded_strategy(strategy, model, query.question)
            predicted = result["predicted_answer"]
            gold = _normalize(query.answer)
            correct = predicted == gold
            row: dict[str, Any] = {
                "question_id": query.id,
                "strategy": strategy,
                "predicted_answer": predicted,
                "gold_answer": gold,
                "correct": correct,
                "samples_used": result["samples_used"],
            }
            # Carry through optional multi-stage fields
            for key in ("first_answer", "revised_answer", "critique_text"):
                if key in result:
                    row[key] = result[key]
            per_query_rows.append(row)

    # Per-strategy summaries
    strategy_summaries: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        rows = [r for r in per_query_rows if r["strategy"] == strategy]
        n = len(rows)
        correct = sum(1 for r in rows if r["correct"])
        total_samples = sum(int(r["samples_used"]) for r in rows)
        strategy_summaries[strategy] = {
            "strategy": strategy,
            "accuracy": correct / n if n > 0 else 0.0,
            "correct": correct,
            "total_queries": n,
            "total_samples_used": total_samples,
            "avg_samples_per_query": total_samples / n if n > 0 else 0.0,
        }

    # Pairwise comparisons (baseline = direct_greedy)
    def _pairwise(base_strategy: str, target_strategy: str) -> dict[str, Any]:
        base_map = {
            r["question_id"]: r
            for r in per_query_rows
            if r["strategy"] == base_strategy
        }
        target_map = {
            r["question_id"]: r
            for r in per_query_rows
            if r["strategy"] == target_strategy
        }
        improved = worsened = 0
        for qid, trow in target_map.items():
            brow = base_map.get(qid)
            if brow is None:
                continue
            if trow["correct"] and not brow["correct"]:
                improved += 1
            elif not trow["correct"] and brow["correct"]:
                worsened += 1
        return {
            "baseline": base_strategy,
            "target": target_strategy,
            "queries_improved": improved,
            "queries_worsened": worsened,
            "net_gain": improved - worsened,
        }

    pairwise: list[dict[str, Any]] = []
    for base in ["direct_greedy", "structured_sampling_3"]:
        if base not in strategies:
            continue
        for target in strategies:
            if target == base:
                continue
            pairwise.append(_pairwise(base, target))

    return {
        "per_query_rows": per_query_rows,
        "strategy_summaries": strategy_summaries,
        "pairwise_comparisons": pairwise,
        "total_queries": len(queries),
        "strategies_run": strategies,
    }


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def write_expanded_strategy_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Save summary.json, summary.csv and per_query_results.csv."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    summary_payload = {
        "total_queries": result["total_queries"],
        "strategies_run": result["strategies_run"],
        "strategy_summaries": result["strategy_summaries"],
        "pairwise_comparisons": result["pairwise_comparisons"],
    }
    summary_json = base / "summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2))

    summaries = list(result["strategy_summaries"].values())
    summary_csv = base / "summary.csv"
    if summaries:
        fieldnames = list(summaries[0].keys())
        with summary_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)

    per_query_rows = result["per_query_rows"]
    per_query_csv = base / "per_query_results.csv"
    if per_query_rows:
        all_fields: list[str] = []
        for row in per_query_rows:
            for key in row:
                if key not in all_fields:
                    all_fields.append(key)
        with per_query_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            for row in per_query_rows:
                writer.writerow({k: row.get(k, "") for k in all_fields})

    return {
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "per_query_csv": str(per_query_csv),
    }


def format_expanded_strategy_summary(
    result: dict[str, Any],
    paths: dict[str, str],
) -> str:
    """Format a human-readable summary of the expanded strategy comparison."""
    lines: list[str] = [
        "--- Expanded Strategy Smoke-Test Evaluation ---",
        f"total queries: {result['total_queries']}",
        "",
        f"{'Strategy':<38} {'Accuracy':>8} {'Correct':>8} {'Samples':>8} {'Avg S/Q':>8}",
        "-" * 74,
    ]
    for summary in result["strategy_summaries"].values():
        lines.append(
            f"{summary['strategy']:<38} "
            f"{summary['accuracy']:>8.4f} "
            f"{summary['correct']:>8} "
            f"{summary['total_samples_used']:>8} "
            f"{summary['avg_samples_per_query']:>8.2f}"
        )

    lines += ["", "Pairwise improvements (vs baseline):"]
    for pw in result["pairwise_comparisons"]:
        lines.append(
            f"  {pw['target']:<38} vs {pw['baseline']:<30} "
            f"improved={pw['queries_improved']} worsened={pw['queries_worsened']} "
            f"net={pw['net_gain']:+d}"
        )

    lines += [
        "",
        f"summary_json:  {paths['summary_json']}",
        f"summary_csv:   {paths['summary_csv']}",
        f"per_query_csv: {paths['per_query_csv']}",
    ]
    return "\n".join(lines)
