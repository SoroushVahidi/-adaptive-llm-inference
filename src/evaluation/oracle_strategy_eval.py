"""Oracle strategy evaluation over a literature-informed strategy set.

This module evaluates every strategy per query, then computes:
- oracle (max-accuracy) assignments,
- cost-aware oracle assignments,
- global oracle-vs-direct gaps,
- per-strategy contribution statistics,
- optional pairwise wins.
"""

from __future__ import annotations

import csv
import json
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Protocol

from src.evaluation.expanded_strategy_eval import (
    run_direct_plus_critique_plus_final,
    run_first_pass_then_hint_guided_reason,
)
from src.evaluation.strategy_expansion_eval import (
    run_direct_greedy,
    run_direct_plus_revise,
    run_direct_plus_verify,
    run_reasoning_best_of_3,
    run_structured_sampling_3,
)


class _ModelProtocol(Protocol):
    def generate(self, prompt: str) -> str: ...
    def generate_n(self, prompt: str, n: int) -> list[str]: ...


BASE_REQUIRED_STRATEGIES = [
    "direct_greedy",
    "structured_sampling_3",
    "reasoning_best_of_3",
    "direct_plus_verify",
    "direct_plus_revise",
    "direct_plus_critique_plus_final",
    "first_pass_then_hint_guided_reason",
]
OPTIONAL_STRATEGIES = ["strong_direct"]

_STRATEGY_RUNNERS = {
    "direct_greedy": run_direct_greedy,
    "structured_sampling_3": run_structured_sampling_3,
    "reasoning_best_of_3": run_reasoning_best_of_3,
    "direct_plus_verify": run_direct_plus_verify,
    "direct_plus_revise": run_direct_plus_revise,
    "direct_plus_critique_plus_final": run_direct_plus_critique_plus_final,
    "first_pass_then_hint_guided_reason": run_first_pass_then_hint_guided_reason,
}


def _normalize(value: str) -> str:
    candidate = str(value).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        number = Decimal(candidate)
    except InvalidOperation:
        return candidate
    normalized = format(number.normalize(), "f")
    if "." in normalized:
        normalized = normalized.rstrip("0").rstrip(".")
    return normalized or "0"


def _to_int_bool(value: bool) -> int:
    return 1 if value else 0


def _argmax_by_score(
    rows: list[dict[str, Any]],
    *,
    score_key: str,
) -> dict[str, Any]:
    return sorted(
        rows,
        key=lambda r: (-float(r[score_key]), float(r["cost"]), str(r["strategy"])),
    )[0]


def run_oracle_strategy_eval(
    model: _ModelProtocol,
    queries: list[Any],
    *,
    strategies: list[str] | None = None,
    strong_model: _ModelProtocol | None = None,
    lambda_penalty: float = 0.0,
) -> dict[str, Any]:
    """Run full per-query strategy matrix and compute oracle statistics."""
    if strategies is None:
        strategies = list(BASE_REQUIRED_STRATEGIES)
        if strong_model is not None:
            strategies.append("strong_direct")

    unknown = [s for s in strategies if s not in _STRATEGY_RUNNERS and s != "strong_direct"]
    if unknown:
        raise ValueError(f"Unknown strategy names: {unknown}")
    if "strong_direct" in strategies and strong_model is None:
        raise ValueError("'strong_direct' requested but strong_model is not provided.")

    per_query_matrix: list[dict[str, Any]] = []

    for query in queries:
        gold = _normalize(query.answer)
        for strategy in strategies:
            runner_model = strong_model if strategy == "strong_direct" else model
            if runner_model is None:
                raise ValueError("Missing model for strategy execution.")
            if strategy == "strong_direct":
                runner = run_direct_greedy
            else:
                runner = _STRATEGY_RUNNERS[strategy]
            result = runner(runner_model, query.question)
            predicted = _normalize(result.get("predicted_answer", ""))
            correct = predicted == gold
            samples_used = int(result.get("samples_used", 1))
            per_query_matrix.append(
                {
                    "question_id": query.id,
                    "question": query.question,
                    "strategy": strategy,
                    "predicted_answer": predicted,
                    "gold_answer": gold,
                    "correct": _to_int_bool(correct),
                    "samples_used": samples_used,
                    "cost": samples_used,
                }
            )

    oracle_assignments: list[dict[str, Any]] = []
    qids = sorted({str(row["question_id"]) for row in per_query_matrix})

    for qid in qids:
        qrows = [row for row in per_query_matrix if str(row["question_id"]) == qid]
        best_correct = max(int(r["correct"]) for r in qrows)

        best_accuracy_rows = [r for r in qrows if int(r["correct"]) == best_correct]
        best_accuracy_row = sorted(
            best_accuracy_rows,
            key=lambda r: (float(r["cost"]), str(r["strategy"])),
        )[0]

        direct_row = next(r for r in qrows if r["strategy"] == "direct_greedy")
        correct_rows = [r for r in qrows if int(r["correct"]) == 1]
        cheapest_correct = (
            sorted(correct_rows, key=lambda r: (float(r["cost"]), str(r["strategy"])))[0]
            if correct_rows
            else None
        )

        scored_rows: list[dict[str, Any]] = []
        for row in qrows:
            score = float(row["correct"]) - (lambda_penalty * float(row["cost"]))
            scored_rows.append({**row, "score": score})
        best_cost_aware = _argmax_by_score(scored_rows, score_key="score")

        oracle_assignments.append(
            {
                "question_id": qid,
                "best_accuracy_strategy": best_accuracy_row["strategy"],
                "oracle_correct": best_correct,
                "cheapest_correct_strategy": (
                    cheapest_correct["strategy"] if cheapest_correct is not None else ""
                ),
                "direct_correct": int(direct_row["correct"]),
                "direct_is_optimal": int(direct_row["strategy"] == best_accuracy_row["strategy"]),
                "cost_aware_best_strategy": best_cost_aware["strategy"],
                "cost_aware_score": float(best_cost_aware["score"]),
            }
        )

    n_queries = len(qids)
    direct_rows = [r for r in per_query_matrix if r["strategy"] == "direct_greedy"]
    direct_accuracy = (
        sum(int(r["correct"]) for r in direct_rows) / n_queries if n_queries else 0.0
    )
    oracle_accuracy = (
        sum(int(row["oracle_correct"]) for row in oracle_assignments) / n_queries
        if n_queries
        else 0.0
    )

    def _fraction_where(strategy_name: str) -> float:
        count = 0
        for qid in qids:
            qrows = [r for r in per_query_matrix if str(r["question_id"]) == qid]
            direct = next(r for r in qrows if r["strategy"] == "direct_greedy")
            target = [r for r in qrows if r["strategy"] == strategy_name]
            if target and int(direct["correct"]) == 0 and int(target[0]["correct"]) == 1:
                count += 1
        return count / n_queries if n_queries else 0.0

    def _fraction_multisample_help() -> float:
        count = 0
        multi = {"reasoning_best_of_3", "structured_sampling_3"}
        for qid in qids:
            qrows = [r for r in per_query_matrix if str(r["question_id"]) == qid]
            direct = next(r for r in qrows if r["strategy"] == "direct_greedy")
            if int(direct["correct"]) == 1:
                continue
            if any(int(r["correct"]) == 1 for r in qrows if r["strategy"] in multi):
                count += 1
        return count / n_queries if n_queries else 0.0

    strategy_contribution: dict[str, dict[str, Any]] = {}
    for strategy in strategies:
        rows = [r for r in per_query_matrix if r["strategy"] == strategy]
        accuracy = sum(int(r["correct"]) for r in rows) / n_queries if n_queries else 0.0

        unique_best = 0
        fixes_direct_failures = 0
        for qid in qids:
            qrows = [r for r in per_query_matrix if str(r["question_id"]) == qid]
            best_correct = max(int(r["correct"]) for r in qrows)
            best = [r for r in qrows if int(r["correct"]) == best_correct]
            if best_correct == 1 and len(best) == 1 and best[0]["strategy"] == strategy:
                unique_best += 1
            direct = next(r for r in qrows if r["strategy"] == "direct_greedy")
            target = next(r for r in qrows if r["strategy"] == strategy)
            if int(direct["correct"]) == 0 and int(target["correct"]) == 1:
                fixes_direct_failures += 1

        strategy_contribution[strategy] = {
            "strategy": strategy,
            "accuracy": accuracy,
            "unique_best_queries": unique_best,
            "fixes_direct_failures": fixes_direct_failures,
            "marginal_gain_over_direct": accuracy - direct_accuracy,
        }

    pairwise_wins: list[dict[str, Any]] = []
    for a in strategies:
        for b in strategies:
            if a == b:
                continue
            a_rows = {str(r["question_id"]): r for r in per_query_matrix if r["strategy"] == a}
            b_rows = {str(r["question_id"]): r for r in per_query_matrix if r["strategy"] == b}
            a_beats_b = 0
            ties = 0
            for qid in qids:
                a_c = int(a_rows[qid]["correct"])
                b_c = int(b_rows[qid]["correct"])
                if a_c > b_c:
                    a_beats_b += 1
                elif a_c == b_c:
                    ties += 1
            pairwise_wins.append(
                {
                    "strategy_a": a,
                    "strategy_b": b,
                    "a_beats_b": a_beats_b,
                    "ties": ties,
                    "b_beats_a": n_queries - a_beats_b - ties,
                }
            )

    summary = {
        "total_queries": n_queries,
        "strategies_run": strategies,
        "lambda_penalty": lambda_penalty,
        "direct_accuracy": direct_accuracy,
        "oracle_accuracy": oracle_accuracy,
        "oracle_direct_gap": oracle_accuracy - direct_accuracy,
        "fraction_direct_optimal": (
            sum(int(row["direct_is_optimal"]) for row in oracle_assignments) / n_queries
            if n_queries
            else 0.0
        ),
        "fraction_critique_helps": _fraction_where("direct_plus_critique_plus_final"),
        "fraction_hint_guided_helps": _fraction_where("first_pass_then_hint_guided_reason"),
        "fraction_strong_model_helps": (
            _fraction_where("strong_direct") if "strong_direct" in strategies else 0.0
        ),
        "fraction_multi_sample_helps": _fraction_multisample_help(),
        "strategy_contribution": strategy_contribution,
    }

    return {
        "summary": summary,
        "per_query_matrix": per_query_matrix,
        "oracle_assignments": oracle_assignments,
        "pairwise_wins": pairwise_wins,
    }


def write_oracle_strategy_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write summary + detailed CSV artifacts for oracle strategy evaluation."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    summary_json = base / "summary.json"
    summary_payload = {**result["summary"], "pairwise_wins": result["pairwise_wins"]}
    summary_json.write_text(json.dumps(summary_payload, indent=2))

    summary_csv = base / "summary.csv"
    contribution_rows = list(result["summary"]["strategy_contribution"].values())
    if contribution_rows:
        fields = list(contribution_rows[0].keys())
        with summary_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(contribution_rows)

    per_query_csv = base / "per_query_matrix.csv"
    per_query_rows = result["per_query_matrix"]
    if per_query_rows:
        fields = list(per_query_rows[0].keys())
        with per_query_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(per_query_rows)

    oracle_csv = base / "oracle_assignments.csv"
    oracle_rows = result["oracle_assignments"]
    if oracle_rows:
        fields = list(oracle_rows[0].keys())
        with oracle_csv.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(oracle_rows)

    return {
        "summary_json": str(summary_json),
        "summary_csv": str(summary_csv),
        "per_query_csv": str(per_query_csv),
        "oracle_assignments_csv": str(oracle_csv),
    }
