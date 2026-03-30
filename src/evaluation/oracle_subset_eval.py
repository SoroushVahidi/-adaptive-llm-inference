"""Oracle-subset evaluation: run all core implemented strategies on a shared
small query subset and compute oracle-style summaries.

This module provides:
  - ``run_oracle_subset_eval``  – main evaluation loop (model-agnostic)
  - ``compute_oracle_summaries``  – per-query and global oracle metrics
  - ``compute_pairwise_win_matrix``  – NxN strategy comparison table
  - ``write_oracle_outputs``  – write all output artefacts to disk

Implemented strategies evaluated (no placeholders):
  direct_greedy, reasoning_best_of_3, structured_sampling_3,
  direct_plus_verify, direct_plus_revise,
  direct_plus_critique_plus_final, first_pass_then_hint_guided_reason

``strong_direct`` is included only when the caller supplies a separate
``strong_model`` argument.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable, Protocol

# Import individual runners from the two existing eval modules so this module
# does not duplicate any logic.
from src.evaluation.expanded_strategy_eval import (
    run_direct_plus_critique_plus_final,
    run_first_pass_then_hint_guided_reason,
)
from src.evaluation.strategy_diagnostic import classify_access_error
from src.evaluation.strategy_expansion_eval import (
    run_direct_greedy,
    run_direct_plus_revise,
    run_direct_plus_verify,
    run_reasoning_best_of_3,
    run_reasoning_then_revise,
    run_self_consistency_3,
    run_structured_sampling_3,
)
from src.models.openai_llm import OpenAILLMModel
from src.utils.answer_extraction import extract_mc_answer, extract_numeric_answer

# ---------------------------------------------------------------------------
# Typing shim
# ---------------------------------------------------------------------------


class _ModelProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...
    def generate_n(self, prompt: str, n: int) -> list[str]: ...


REASONING_GREEDY_PROMPT = (
    "Solve this step by step and end with 'Final answer: <number>'.\n\n{question}"
)


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    runner: Callable[[Any, str], dict[str, Any]]
    model_slot: str = "current"

# Cost proxy: number of model invocations (samples) each strategy makes.
# This is a simple, explicit proxy for computational cost — a strategy that
# calls the model k times has cost_proxy = k.  Strategies with multiple stages
# count each stage call separately (e.g. direct_plus_verify = 1 direct + 1
# verify call = 2).
STRATEGY_COST_PROXY: dict[str, int] = {
    "direct_greedy": 1,           # 1 model call
    "reasoning_greedy": 1,        # 1 reasoning-style sample
    "reasoning_best_of_3": 3,     # 3 parallel samples
    "structured_sampling_3": 3,   # 3 sequential prompts
    "direct_plus_verify": 2,      # direct + verify
    "direct_plus_revise": 2,      # direct + revise
    "reasoning_then_revise": 2,   # reasoning + review/revise pass
    "self_consistency_3": 3,      # 3 reasoning samples + vote
    "direct_plus_critique_plus_final": 3,  # direct + critique + final
    "first_pass_then_hint_guided_reason": 2,  # first pass + hint-guided
    "strong_direct": 1,           # 1 call on the strong model
}

# Registry of runners for each strategy.
_ORACLE_RUNNERS: dict[str, Any] = {
    "direct_greedy": run_direct_greedy,
    "reasoning_greedy": None,  # filled after helper definition
    "reasoning_best_of_3": run_reasoning_best_of_3,
    "structured_sampling_3": run_structured_sampling_3,
    "direct_plus_verify": run_direct_plus_verify,
    "direct_plus_revise": run_direct_plus_revise,
    "reasoning_then_revise": run_reasoning_then_revise,
    "self_consistency_3": run_self_consistency_3,
    "direct_plus_critique_plus_final": run_direct_plus_critique_plus_final,
    "first_pass_then_hint_guided_reason": run_first_pass_then_hint_guided_reason,
    # strong_direct uses the same runner as direct_greedy but with a different
    # model object that the caller supplies separately.
    "strong_direct": run_direct_greedy,
}

CORE_ORACLE_STRATEGIES: list[str] = [
    "direct_greedy",
    "reasoning_best_of_3",
    "structured_sampling_3",
    "direct_plus_verify",
    "direct_plus_revise",
    "reasoning_then_revise",
    "direct_plus_critique_plus_final",
    "first_pass_then_hint_guided_reason",
]

# Multi-action supervised routing (narrow action set for learned controllers).
MULTI_ACTION_ORACLE_STRATEGIES: list[str] = [
    "reasoning_greedy",
    "direct_plus_revise",
    "reasoning_then_revise",
    "self_consistency_3",
]


# ---------------------------------------------------------------------------
# Numeric normalisation (shared, consistent with existing eval modules)
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


def run_reasoning_greedy(
    model: Any,
    question: str,
    answer_mode: str = "numeric",
) -> dict[str, Any]:
    """One reasoning-style sample with the current model."""
    if answer_mode == "multiple_choice":
        prompt = (
            "Solve this step by step. The question lists choices (A) through (D). "
            "End with 'Final answer: (X)' where X is exactly one letter A, B, C, or D.\n\n"
            f"{question}"
        )
        raw = model.generate(prompt)
        letter = extract_mc_answer(raw)
        answer = letter.upper() if letter else ""
    else:
        raw = model.generate(REASONING_GREEDY_PROMPT.format(question=question))
        answer = _normalize(extract_numeric_answer(raw))
    return {
        "raw_outputs": [raw],
        "predicted_answer": answer,
        "samples_used": 1,
    }


_ORACLE_RUNNERS["reasoning_greedy"] = run_reasoning_greedy

STRATEGY_DEFINITIONS = {
    "direct_greedy": StrategyDefinition("direct_greedy", run_direct_greedy),
    "reasoning_greedy": StrategyDefinition("reasoning_greedy", run_reasoning_greedy),
    "reasoning_best_of_3": StrategyDefinition(
        "reasoning_best_of_3",
        run_reasoning_best_of_3,
    ),
    "structured_sampling_3": StrategyDefinition(
        "structured_sampling_3",
        run_structured_sampling_3,
    ),
    "direct_plus_verify": StrategyDefinition("direct_plus_verify", run_direct_plus_verify),
    "direct_plus_revise": StrategyDefinition("direct_plus_revise", run_direct_plus_revise),
    "reasoning_then_revise": StrategyDefinition(
        "reasoning_then_revise",
        run_reasoning_then_revise,
    ),
    "self_consistency_3": StrategyDefinition(
        "self_consistency_3",
        run_self_consistency_3,
    ),
    "direct_plus_critique_plus_final": StrategyDefinition(
        "direct_plus_critique_plus_final",
        run_direct_plus_critique_plus_final,
    ),
    "first_pass_then_hint_guided_reason": StrategyDefinition(
        "first_pass_then_hint_guided_reason",
        run_first_pass_then_hint_guided_reason,
    ),
    "strong_direct": StrategyDefinition(
        "strong_direct",
        run_direct_greedy,
        model_slot="strong",
    ),
}


def _build_model(
    model_name: str,
    config: dict[str, Any],
    max_tokens_override: int | None = None,
) -> OpenAILLMModel:
    openai_cfg = config.get("openai", {})
    model_cfg = config.get("model", {})
    max_tokens = (
        int(max_tokens_override)
        if max_tokens_override is not None
        else int(
            openai_cfg.get(
                "max_tokens",
                model_cfg.get("max_tokens", 256),
            )
        )
    )
    return OpenAILLMModel(
        model_name=model_name,
        base_url=openai_cfg.get("base_url", model_cfg.get("base_url")),
        greedy_temperature=float(
            openai_cfg.get("greedy_temperature", model_cfg.get("greedy_temperature", 0.0))
        ),
        sample_temperature=float(
            openai_cfg.get("sample_temperature", model_cfg.get("sample_temperature", 0.7))
        ),
        max_tokens=max_tokens,
        timeout_seconds=float(
            openai_cfg.get("timeout_seconds", model_cfg.get("timeout_seconds", 60.0))
        ),
    )


def _probe_model_access(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    try:
        model = _build_model(
            model_name=model_name,
            config=config,
            max_tokens_override=8,
        )
        model.generate("What is 1 + 1?")
    except ValueError as exc:
        return {
            "model": model_name,
            "status": "failed",
            "error_type": classify_access_error(str(exc)),
            "error": str(exc),
        }
    except RuntimeError as exc:
        return {
            "model": model_name,
            "status": "failed",
            "error_type": classify_access_error(str(exc)),
            "error": str(exc),
        }

    return {
        "model": model_name,
        "status": "accessible",
        "error_type": None,
        "error": None,
    }


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def run_oracle_subset_eval(
    model: _ModelProtocol,
    queries: list[Any],
    strategies: list[str] | None = None,
    strong_model: _ModelProtocol | None = None,
    answer_mode: str = "numeric",
    gold_normalizer: Any | None = None,
) -> dict[str, Any]:
    """Run all oracle strategies on *queries* and return raw per-query rows.

    Args:
        model: Primary model (cheap_model slot).
        queries: Sequence of objects with ``.id``, ``.question``, ``.answer``.
        strategies: Which strategies to run.  Defaults to
            ``CORE_ORACLE_STRATEGIES``.  Pass ``["strong_direct", ...]`` only
            if *strong_model* is not None.
        strong_model: Optional stronger model used for ``strong_direct`` only.
        answer_mode: ``numeric`` (default) or ``multiple_choice`` (A–D grading).
        gold_normalizer: Optional ``callable(str) -> str`` applied to gold before
            comparison. Defaults to ``_normalize`` for numeric mode and identity
            for multiple_choice (caller should pass upper-case letters).

    Returns:
        Dict with ``per_query_rows``, ``strategies_run``, ``query_ids``.
    """
    if strategies is None:
        strategies = list(CORE_ORACLE_STRATEGIES)
    if "strong_direct" in strategies and strong_model is None:
        strategies = [s for s in strategies if s != "strong_direct"]

    unknown = [s for s in strategies if s not in _ORACLE_RUNNERS]
    if unknown:
        raise ValueError(f"Unknown oracle strategies: {unknown}")

    if answer_mode not in ("numeric", "multiple_choice"):
        raise ValueError(f"answer_mode must be 'numeric' or 'multiple_choice', got {answer_mode!r}")

    def _default_gold_norm(ans: str) -> str:
        if answer_mode == "multiple_choice":
            return (ans or "").strip().upper()[:1]
        return _normalize(ans)

    gold_norm_fn = gold_normalizer if gold_normalizer is not None else _default_gold_norm

    per_query_rows: list[dict[str, Any]] = []
    query_ids: list[str] = []

    for query in queries:
        if query.id not in query_ids:
            query_ids.append(query.id)

        gold = gold_norm_fn(query.answer)

        for strategy in strategies:
            runner = _ORACLE_RUNNERS[strategy]
            effective_model = strong_model if strategy == "strong_direct" else model
            if strategy == "reasoning_then_revise":
                result = runner(
                    effective_model,
                    query.question,
                    answer_mode=answer_mode,
                )
            elif strategy in (
                "reasoning_greedy",
                "direct_plus_revise",
                "self_consistency_3",
            ):
                result = runner(effective_model, query.question, answer_mode=answer_mode)
            else:
                result = runner(effective_model, query.question)

            predicted = result["predicted_answer"]
            correct = predicted == gold

            row: dict[str, Any] = {
                "question_id": query.id,
                "strategy": strategy,
                "predicted_answer": predicted,
                "gold_answer": gold,
                "correct": int(correct),
                "samples_used": result["samples_used"],
                "cost_proxy": result["samples_used"],
            }
            # Carry optional multi-stage / diagnostic fields when present.
            for key in (
                "first_answer",
                "revised_answer",
                "critique_text",
                "self_consistency_ambiguous",
                "self_consistency_tied_values",
            ):
                if key in result:
                    row[key] = result[key]
            per_query_rows.append(row)

    return {
        "per_query_rows": per_query_rows,
        "strategies_run": strategies,
        "query_ids": query_ids,
    }


# ---------------------------------------------------------------------------
# Oracle summary computation
# ---------------------------------------------------------------------------

def compute_oracle_summaries(
    per_query_rows: list[dict[str, Any]],
    strategies: list[str],
) -> dict[str, Any]:
    """Compute per-query oracle assignments and global oracle metrics.

    Per-query oracle fields:
      - ``best_accuracy_strategies``: list of strategies that are correct
      - ``cheapest_correct_strategy``: cheapest strategy that is correct
      - ``direct_greedy_correct``: whether direct_greedy was correct
      - ``direct_already_optimal``: True when direct_greedy is correct AND is
        the cheapest correct strategy

    Global summaries:
      - per-strategy accuracy
      - oracle_accuracy (fraction of queries where ≥1 strategy correct)
      - oracle_minus_direct_gap
      - uniquely_best_count per strategy
      - among_best_count per strategy
      - cheapest_correct_count per strategy
      - fixes_direct_greedy_count per strategy
    """
    # Index rows by (question_id, strategy)
    rows_by_qid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in per_query_rows:
        rows_by_qid[row["question_id"]][row["strategy"]] = row

    query_ids = list(rows_by_qid.keys())

    # Per-query oracle records
    per_query_oracle: list[dict[str, Any]] = []
    for qid in query_ids:
        strategy_rows = rows_by_qid[qid]
        correct_strategies = [
            s for s in strategies if strategy_rows.get(s, {}).get("correct", 0)
        ]
        direct_correct = bool(strategy_rows.get("direct_greedy", {}).get("correct", 0))

        # Cheapest correct strategy (ties broken by first in canonical order)
        cheapest_correct: str | None = None
        if correct_strategies:
            cheapest_correct = min(
                correct_strategies,
                key=lambda s: STRATEGY_COST_PROXY.get(s, 99),
            )

        direct_already_optimal = (
            direct_correct
            and cheapest_correct == "direct_greedy"
        )

        per_query_oracle.append({
            "question_id": qid,
            "any_correct": len(correct_strategies) > 0,
            "best_accuracy_strategies": correct_strategies,
            "cheapest_correct_strategy": cheapest_correct or "",
            "direct_greedy_correct": int(direct_correct),
            "direct_already_optimal": int(direct_already_optimal),
        })

    n_queries = len(query_ids)

    # Global: per-strategy accuracy
    strategy_accuracy: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        correct_count = sum(
            1
            for qid in query_ids
            if rows_by_qid[qid].get(strategy, {}).get("correct", 0)
        )
        strategy_accuracy[strategy] = {
            "strategy": strategy,
            "correct": correct_count,
            "total_queries": n_queries,
            "accuracy": correct_count / n_queries if n_queries > 0 else 0.0,
        }

    oracle_correct = sum(1 for r in per_query_oracle if r["any_correct"])
    oracle_accuracy = oracle_correct / n_queries if n_queries > 0 else 0.0
    direct_accuracy = (
        strategy_accuracy.get("direct_greedy", {}).get("accuracy", 0.0)
    )
    oracle_minus_direct_gap = oracle_accuracy - direct_accuracy

    # Per-strategy contribution counts
    uniquely_best: dict[str, int] = defaultdict(int)
    among_best: dict[str, int] = defaultdict(int)
    cheapest_correct_counts: dict[str, int] = defaultdict(int)
    fixes_direct_greedy: dict[str, int] = defaultdict(int)

    for rec in per_query_oracle:
        best_list = rec["best_accuracy_strategies"]
        if len(best_list) == 1:
            uniquely_best[best_list[0]] += 1
        for s in best_list:
            among_best[s] += 1
        cc = rec["cheapest_correct_strategy"]
        if cc:
            cheapest_correct_counts[cc] += 1
        direct_was_wrong = not bool(rec["direct_greedy_correct"])
        for s in best_list:
            if s != "direct_greedy" and direct_was_wrong:
                fixes_direct_greedy[s] += 1

    direct_already_optimal_fraction = (
        sum(1 for r in per_query_oracle if r["direct_already_optimal"]) / n_queries
        if n_queries > 0 else 0.0
    )

    return {
        "per_query_oracle": per_query_oracle,
        "strategy_accuracy": strategy_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "oracle_correct": oracle_correct,
        "direct_accuracy": direct_accuracy,
        "oracle_minus_direct_gap": oracle_minus_direct_gap,
        "direct_already_optimal_fraction": direct_already_optimal_fraction,
        "uniquely_best": dict(uniquely_best),
        "among_best": dict(among_best),
        "cheapest_correct_counts": dict(cheapest_correct_counts),
        "fixes_direct_greedy": dict(fixes_direct_greedy),
        "total_queries": n_queries,
    }


# ---------------------------------------------------------------------------
# Pairwise win matrix
# ---------------------------------------------------------------------------

def compute_pairwise_win_matrix(
    per_query_rows: list[dict[str, Any]],
    strategies: list[str],
) -> dict[str, Any]:
    """Compute an NxN pairwise win/loss table.

    ``matrix[A][B]`` = number of queries where A is correct and B is wrong.
    """
    rows_by_qid: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in per_query_rows:
        rows_by_qid[row["question_id"]][row["strategy"]] = row

    query_ids = list(rows_by_qid.keys())

    matrix: dict[str, dict[str, int]] = {s: {t: 0 for t in strategies} for s in strategies}

    for qid in query_ids:
        for s in strategies:
            for t in strategies:
                if s == t:
                    continue
                s_correct = bool(rows_by_qid[qid].get(s, {}).get("correct", 0))
                t_correct = bool(rows_by_qid[qid].get(t, {}).get("correct", 0))
                if s_correct and not t_correct:
                    matrix[s][t] += 1

    return {"matrix": matrix, "strategies": strategies}


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------

def write_oracle_outputs(
    eval_result: dict[str, Any],
    oracle_summaries: dict[str, Any],
    pairwise: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write all oracle evaluation artefacts to *output_dir*.

    Files written:
      - ``per_query_matrix.csv``
      - ``summary.json``
      - ``summary.csv``
      - ``oracle_assignments.csv``
      - ``pairwise_win_matrix.csv``

    Returns a dict mapping artefact name → absolute path.
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    # per_query_matrix.csv
    per_query_rows = eval_result["per_query_rows"]
    matrix_csv = base / "per_query_matrix.csv"
    if per_query_rows:
        all_fields: list[str] = []
        for row in per_query_rows:
            for key in row:
                if key not in all_fields:
                    all_fields.append(key)
        with matrix_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=all_fields, extrasaction="ignore")
            writer.writeheader()
            for row in per_query_rows:
                writer.writerow({k: row.get(k, "") for k in all_fields})
    paths["per_query_matrix_csv"] = str(matrix_csv)

    # summary.json
    summary_payload = {
        "total_queries": oracle_summaries["total_queries"],
        "strategies_run": eval_result["strategies_run"],
        "query_ids": eval_result["query_ids"],
        "strategy_accuracy": oracle_summaries["strategy_accuracy"],
        "oracle_accuracy": oracle_summaries["oracle_accuracy"],
        "oracle_correct": oracle_summaries["oracle_correct"],
        "direct_accuracy": oracle_summaries["direct_accuracy"],
        "oracle_minus_direct_gap": oracle_summaries["oracle_minus_direct_gap"],
        "direct_already_optimal_fraction": oracle_summaries["direct_already_optimal_fraction"],
        "uniquely_best": oracle_summaries["uniquely_best"],
        "among_best": oracle_summaries["among_best"],
        "cheapest_correct_counts": oracle_summaries["cheapest_correct_counts"],
        "fixes_direct_greedy": oracle_summaries["fixes_direct_greedy"],
    }
    summary_json = base / "summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2))
    paths["summary_json"] = str(summary_json)

    # summary.csv – one row per strategy
    summaries = list(oracle_summaries["strategy_accuracy"].values())
    summary_csv = base / "summary.csv"
    if summaries:
        fieldnames = list(summaries[0].keys())
        with summary_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summaries)
    paths["summary_csv"] = str(summary_csv)

    # oracle_assignments.csv
    oracle_csv = base / "oracle_assignments.csv"
    per_q_oracle = oracle_summaries["per_query_oracle"]
    if per_q_oracle:
        fieldnames_oracle = [
            "question_id",
            "any_correct",
            "cheapest_correct_strategy",
            "direct_greedy_correct",
            "direct_already_optimal",
            "best_accuracy_strategies",
        ]
        with oracle_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames_oracle, extrasaction="ignore")
            writer.writeheader()
            for row in per_q_oracle:
                row_out = dict(row)
                row_out["best_accuracy_strategies"] = "|".join(
                    row_out.get("best_accuracy_strategies", [])
                )
                writer.writerow({k: row_out.get(k, "") for k in fieldnames_oracle})
    paths["oracle_assignments_csv"] = str(oracle_csv)

    # pairwise_win_matrix.csv
    pairwise_csv = base / "pairwise_win_matrix.csv"
    matrix = pairwise["matrix"]
    strats = pairwise["strategies"]
    with pairwise_csv.open("w", newline="") as fh:
        writer_pw = csv.writer(fh)
        writer_pw.writerow(["strategy_row_wins_vs_col"] + strats)
        for s in strats:
            writer_pw.writerow([s] + [matrix[s][t] for t in strats])
    paths["pairwise_win_matrix_csv"] = str(pairwise_csv)

    return paths


# ---------------------------------------------------------------------------
# Human-readable summary formatter
# ---------------------------------------------------------------------------

def format_oracle_summary(
    oracle_summaries: dict[str, Any],
    paths: dict[str, str],
) -> str:
    """Return a human-readable multi-line summary."""
    lines: list[str] = [
        "─── Oracle Subset Evaluation ───",
        f"Queries:          {oracle_summaries['total_queries']}",
        f"Oracle accuracy:  {oracle_summaries['oracle_accuracy']:.4f}",
        f"Direct accuracy:  {oracle_summaries['direct_accuracy']:.4f}",
        f"Oracle-direct gap:{oracle_summaries['oracle_minus_direct_gap']:+.4f}",
        f"Direct already optimal: "
        f"{oracle_summaries['direct_already_optimal_fraction']:.1%} of queries",
        "",
        f"{'Strategy':<40} {'Acc':>6} {'Corr':>6} {'Fixes↑':>7} {'CheapCorr':>10}",
        "─" * 73,
    ]
    for st, acc in oracle_summaries["strategy_accuracy"].items():
        fixes = oracle_summaries["fixes_direct_greedy"].get(st, 0)
        cc = oracle_summaries["cheapest_correct_counts"].get(st, 0)
        lines.append(
            f"{st:<40} {acc['accuracy']:>6.4f} {acc['correct']:>6} "
            f"{fixes:>7} {cc:>10}"
        )
    lines += [
        "",
        "Outputs:",
    ]
    for name, path in paths.items():
        lines.append(f"  {name}: {path}")
    return "\n".join(lines)
