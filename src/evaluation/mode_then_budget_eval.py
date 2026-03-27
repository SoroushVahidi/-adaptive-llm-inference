"""Evaluation helpers for the mode-then-budget hybrid prototype.

This method family is motivated by the current real-data findings:
1) direct greedy is strong and cheap,
2) reasoning-only can be weaker,
3) reasoning plus extra compute can help on a subset of queries, and
4) the adaptive decision therefore includes both inference *mode* and compute.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.baselines.base import BaselineResult
from src.baselines.best_of_n import BestOfNBaseline
from src.baselines.greedy import GreedyBaseline
from src.datasets.gsm8k import Query, load_gsm8k
from src.evaluation.metrics import compute_accuracy
from src.methods.mode_then_budget import ModeThenBudgetConfig, run_mode_then_budget
from src.models.openai_llm import OpenAILLMModel

DIRECT_PROMPT = "Answer the following question. Give only the final numeric answer."
REASONING_PROMPT = (
    "Solve the following math word problem carefully. "
    "Think step by step, then give the final numeric answer clearly at the end."
)


def _write_json(payload: dict[str, Any], output_path: str | Path) -> Path:
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(json.dumps(payload, indent=2))
    return resolved


def _write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> Path:
    if not rows:
        raise ValueError("rows must be non-empty")
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    with resolved.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return resolved


def _build_model(config: dict[str, Any], prompt_prefix: str) -> OpenAILLMModel:
    model_cfg = config.get("model", {})
    return OpenAILLMModel(
        model_name=str(model_cfg["name"]),
        base_url=model_cfg.get("base_url"),
        prompt_prefix=prompt_prefix,
        greedy_temperature=float(model_cfg.get("greedy_temperature", 0.0)),
        sample_temperature=float(model_cfg.get("sample_temperature", 0.7)),
        max_tokens=int(model_cfg.get("max_tokens", 256)),
        timeout_seconds=float(model_cfg.get("timeout_seconds", 60.0)),
    )


def _run_greedy_mode(
    queries: list[Query],
    config: dict[str, Any],
    prompt_prefix: str,
) -> tuple[dict[str, Any], list[BaselineResult]]:
    model = _build_model(config, prompt_prefix)
    baseline = GreedyBaseline(model)
    results: list[BaselineResult] = []
    for query in queries:
        results.append(
            baseline.solve(
                query_id=query.id,
                question=query.question,
                ground_truth=query.answer,
                n_samples=1,
            )
        )
    summary = compute_accuracy(results)
    return (
        {
            "method": "direct_greedy" if prompt_prefix == DIRECT_PROMPT else "reasoning_greedy",
            "accuracy": float(summary["accuracy"]),
            "total_samples_used": int(summary["total_samples"]),
            "average_samples_per_query": float(summary["avg_samples_per_query"]),
        },
        results,
    )


def _run_reasoning_best_of_3(
    queries: list[Query],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[BaselineResult]]:
    model = _build_model(config, REASONING_PROMPT)
    baseline = BestOfNBaseline(model)
    results: list[BaselineResult] = []
    for query in queries:
        results.append(
            baseline.solve(
                query_id=query.id,
                question=query.question,
                ground_truth=query.answer,
                n_samples=3,
            )
        )
    summary = compute_accuracy(results)
    return (
        {
            "method": "reasoning_best_of_3",
            "accuracy": float(summary["accuracy"]),
            "total_samples_used": int(summary["total_samples"]),
            "average_samples_per_query": float(summary["avg_samples_per_query"]),
        },
        results,
    )


def _majority_vote(values: list[str]) -> str:
    counts = Counter(values)
    ranked = sorted(counts.items(), key=lambda item: (-item[1], values.index(item[0]), item[0]))
    return ranked[0][0]


def _run_selective_v1_if_enabled(
    queries: list[Query],
    config: dict[str, Any],
) -> dict[str, Any] | None:
    if not bool(config.get("include_selective_escalation_v1", False)):
        return None

    from src.evaluation.selective_escalation_eval import run_selective_escalation_eval

    selective_method_cfg = config.get("selective_method")
    if selective_method_cfg is None:
        mode_cfg = config.get("mode_then_budget", {})
        selective_method_cfg = {
            "total_budget": int(mode_cfg["total_budget"]),
            "extra_samples_per_escalated_query": int(
                mode_cfg.get("extra_reasoning_samples", 2)
            ),
            "use_second_sample_for_disagreement": bool(
                mode_cfg.get("use_reasoning_probe", True)
            ),
            "parse_failure_weight": float(mode_cfg.get("parse_failure_weight", 2.0)),
            "disagreement_weight": float(
                mode_cfg.get("weight_direct_reasoning_disagreement", 1.5)
            ),
            "malformed_output_weight": float(mode_cfg.get("malformed_output_weight", 1.0)),
            "missing_numeric_weight": float(
                mode_cfg.get("weight_low_confidence_format", 1.0)
            ),
            "min_score_to_escalate": float(mode_cfg.get("min_switch_score", 1.5)),
        }

    selective_result = run_selective_escalation_eval(
        {
            "dataset": config["dataset"],
            "model": config["model"],
            "selective_method": selective_method_cfg,
        }
    )
    summary = next(
        item
        for item in selective_result["method_summaries"]
        if item["method"] == "selective_escalation_v1"
    )
    return {
        "summary": summary,
        "per_query_rows": selective_result["per_query_rows"],
    }


def run_mode_then_budget_eval(config: dict[str, Any]) -> dict[str, Any]:
    """Run the mode-then-budget comparison on a small GSM8K subset."""
    dataset_cfg = config.get("dataset", {})
    queries = load_gsm8k(
        split=str(dataset_cfg.get("split", "test")),
        max_samples=dataset_cfg.get("max_samples"),
    )

    direct_model = _build_model(config, DIRECT_PROMPT)
    reasoning_model = _build_model(config, REASONING_PROMPT)
    method_cfg = config.get("mode_then_budget", {})
    mode_then_budget = run_mode_then_budget(
        direct_model=direct_model,
        reasoning_model=reasoning_model,
        questions=[query.question for query in queries],
        total_budget=int(method_cfg["total_budget"]),
        config=ModeThenBudgetConfig(
            total_budget=int(method_cfg["total_budget"]),
            reasoning_target_k=1 + int(method_cfg.get("extra_reasoning_samples", 2)),
            min_switch_score=float(method_cfg.get("min_switch_score", 1.5)),
            use_reasoning_probe=bool(method_cfg.get("use_reasoning_probe", True)),
            weight_parse_failure=float(method_cfg.get("parse_failure_weight", 2.0)),
            weight_malformed_output=float(method_cfg.get("malformed_output_weight", 1.0)),
            weight_low_confidence_format=float(method_cfg.get("low_confidence_weight", 1.0)),
            weight_direct_reasoning_disagreement=float(
                method_cfg.get("mode_disagreement_weight", 1.5)
            ),
            malformed_length_threshold=int(method_cfg.get("malformed_length_threshold", 2)),
        ),
    )

    direct_summary, direct_results = _run_greedy_mode(
        queries=queries,
        config=config,
        prompt_prefix=DIRECT_PROMPT,
    )
    reasoning_greedy_summary, reasoning_greedy_results = _run_greedy_mode(
        queries=queries,
        config=config,
        prompt_prefix=REASONING_PROMPT,
    )
    reasoning_best_of_3_summary, reasoning_best_results = _run_reasoning_best_of_3(
        queries=queries,
        config=config,
    )

    direct_map = {result.query_id: result for result in direct_results}
    reasoning_greedy_map = {
        result.query_id: result for result in reasoning_greedy_results
    }
    reasoning_best_map = {result.query_id: result for result in reasoning_best_results}
    per_query_rows: list[dict[str, Any]] = []

    for query, item in zip(queries, mode_then_budget["diagnostics"]):
        direct_result = direct_map[query.id]
        reasoning_greedy_result = reasoning_greedy_map[query.id]
        reasoning_best_result = reasoning_best_map[query.id]
        direct_answer = direct_result.final_answer
        reasoning_probe = reasoning_greedy_result.final_answer
        reasoning_best = reasoning_best_result.final_answer
        final_answer = item["final_answer"]

        direct_is_correct = bool(direct_result.correct)
        reasoning_probe_correct = bool(reasoning_greedy_result.correct)
        reasoning_best_correct_flag = bool(reasoning_best_result.correct)
        final_correct = final_answer == query.answer

        per_query_rows.append(
            {
                "question_id": query.id,
                "gold_answer": query.answer,
                "direct_answer": direct_answer,
                "direct_correct": direct_is_correct,
                "switched_to_reasoning": bool(item["switched_to_reasoning"]),
                "signals_for_switching": ",".join(item["signals_fired"]),
                "final_mode": item["final_mode"],
                "final_answer": final_answer,
                "final_correct": final_correct,
                "samples_used": int(item["samples_used"]),
                "reasoning_probe_answer": reasoning_probe,
                "reasoning_probe_correct": reasoning_probe_correct,
                "reasoning_best_of_3_answer": reasoning_best,
                "reasoning_best_of_3_correct": reasoning_best_correct_flag,
            }
        )

    n_queries = len(queries)
    switched = int(mode_then_budget["queries_switched_to_reasoning"])
    switched_rows = [
        row for row in mode_then_budget["diagnostics"] if bool(row["switched_to_reasoning"])
    ]
    avg_reasoning_samples = (
        0.0
        if not switched_rows
        else sum(int(row["reasoning_samples_used"]) for row in switched_rows) / len(switched_rows)
    )
    hybrid_summary = {
        "method": "mode_then_budget_v2",
        "accuracy": (
            0.0
            if n_queries == 0
            else sum(int(row["final_correct"]) for row in per_query_rows) / n_queries
        ),
        "total_samples_used": int(mode_then_budget["total_samples_used"]),
        "average_samples_per_query": (
            0.0 if n_queries == 0 else mode_then_budget["total_samples_used"] / n_queries
        ),
        "queries_staying_in_direct_mode": n_queries - switched,
        "queries_switched_to_reasoning": switched,
        "fraction_switched_to_reasoning": 0.0 if n_queries == 0 else switched / n_queries,
        "average_reasoning_samples_among_switched": avg_reasoning_samples,
    }

    method_summaries = [
        direct_summary,
        reasoning_greedy_summary,
        reasoning_best_of_3_summary,
        hybrid_summary,
    ]

    selective_result = _run_selective_v1_if_enabled(queries, config)
    if selective_result is not None:
        method_summaries.append(selective_result["summary"])

    return {
        "provider": "openai",
        "model_name": direct_model.model_name,
        "total_queries": n_queries,
        "method_summaries": method_summaries,
        "per_query_rows": per_query_rows,
    }


def write_mode_then_budget_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    base_dir = Path(output_dir)
    return {
        "summary_json": str(
            _write_json(result, base_dir / "mode_then_budget_results.json")
        ),
        "per_query_csv": str(
            _write_csv(result["per_query_rows"], base_dir / "per_query_diagnostic.csv")
        ),
    }


def format_mode_then_budget_summary(result: dict[str, Any], paths: dict[str, str]) -> str:
    lines = [
        "--- Mode-Then-Budget Summary ---",
        f"provider:                  {result['provider']}",
        f"model:                     {result['model_name']}",
        f"queries:                   {result['total_queries']}",
        "method results:",
    ]
    for summary in result["method_summaries"]:
        base = (
            "  "
            f"{summary['method']}: accuracy={summary['accuracy']:.4f}, "
            f"total_samples={summary['total_samples_used']}, "
            f"avg_samples/query={summary['average_samples_per_query']:.2f}"
        )
        if summary["method"] == "mode_then_budget_v2":
            base += (
                ", switched_to_reasoning="
                f"{summary['queries_switched_to_reasoning']} "
                f"({summary['fraction_switched_to_reasoning']:.4f})"
            )
        lines.append(base)

    lines.extend(
        [
            "",
            f"summary_json:              {paths['summary_json']}",
            f"per_query_csv:             {paths['per_query_csv']}",
        ]
    )
    return "\n".join(lines)
