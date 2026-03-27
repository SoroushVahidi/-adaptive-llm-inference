"""Evaluation helpers for the selective compute escalation prototype.

This prototype is motivated by current findings:
1) real-data headroom appears limited,
2) extra compute helps only a minority of queries,
3) naive always-more-compute is inefficient,
4) selective escalation is therefore a promising conservative direction.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.baselines.base import BaselineResult
from src.baselines.best_of_n import BestOfNBaseline
from src.baselines.greedy import GreedyBaseline
from src.datasets.gsm8k import Query, load_gsm8k
from src.evaluation.metrics import compute_accuracy
from src.methods.selective_escalation import (
    SelectiveEscalationConfig,
    run_selective_escalation,
)
from src.models.openai_llm import OpenAILLMModel

DIRECT_PROMPT = "Answer the following question. Give only the final numeric answer."


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


def _build_model(config: dict[str, Any]) -> OpenAILLMModel:
    model_cfg = config.get("model", {})
    return OpenAILLMModel(
        model_name=str(model_cfg["name"]),
        base_url=model_cfg.get("base_url"),
        prompt_prefix=str(model_cfg.get("prompt_prefix", DIRECT_PROMPT)),
        greedy_temperature=float(model_cfg.get("greedy_temperature", 0.0)),
        sample_temperature=float(model_cfg.get("sample_temperature", 0.7)),
        max_tokens=int(model_cfg.get("max_tokens", 128)),
        timeout_seconds=float(model_cfg.get("timeout_seconds", 60.0)),
    )


def _format_prompt(question: str) -> str:
    return question


def _run_greedy(
    queries: list[Query],
    model: OpenAILLMModel,
) -> tuple[dict[str, Any], list[BaselineResult]]:
    baseline = GreedyBaseline(model)
    results: list[BaselineResult] = []
    for query in queries:
        results.append(
            baseline.solve(
                query_id=query.id,
                question=_format_prompt(query.question),
                ground_truth=query.answer,
                n_samples=1,
            )
        )
    summary = compute_accuracy(results)
    return (
        {
            "method": "greedy",
            "accuracy": float(summary["accuracy"]),
            "total_samples_used": int(summary["total_samples"]),
            "average_samples_per_query": float(summary["avg_samples_per_query"]),
            "total_queries": int(summary["total_queries"]),
            "queries_escalated": 0,
            "fraction_escalated": 0.0,
        },
        results,
    )


def _run_always_best_of_3(
    queries: list[Query],
    model: OpenAILLMModel,
) -> tuple[dict[str, Any], list[BaselineResult]]:
    baseline = BestOfNBaseline(model)
    results: list[BaselineResult] = []
    for query in queries:
        results.append(
            baseline.solve(
                query_id=query.id,
                question=_format_prompt(query.question),
                ground_truth=query.answer,
                n_samples=3,
            )
        )
    summary = compute_accuracy(results)
    return (
        {
            "method": "always_best_of_3",
            "accuracy": float(summary["accuracy"]),
            "total_samples_used": int(summary["total_samples"]),
            "average_samples_per_query": float(summary["avg_samples_per_query"]),
            "total_queries": int(summary["total_queries"]),
            "queries_escalated": len(queries),
            "fraction_escalated": 1.0,
        },
        results,
    )


def _run_selective(
    queries: list[Query],
    model: OpenAILLMModel,
    config: SelectiveEscalationConfig,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    run = run_selective_escalation(
        model=model,
        questions=[_format_prompt(query.question) for query in queries],
        total_budget=config.total_budget,
        config=config,
    )
    diagnostics = run["diagnostics"]
    total_queries = len(diagnostics)
    correct = 0
    per_query_rows: list[dict[str, Any]] = []
    for query, item in zip(queries, diagnostics):
        final_correct = item["final_answer"] == query.answer
        first_correct = item["first_pass_answer"] == query.answer
        signals_fired = [
            signal_name
            for signal_name, fired in item["signals"].items()
            if bool(fired)
        ]
        if final_correct:
            correct += 1
        per_query_rows.append(
            {
                "question_id": query.id,
                "gold_answer": query.answer,
                "first_pass_answer": item["first_pass_answer"],
                "first_pass_correct": first_correct,
                "escalation_score": float(item["escalation_score"]),
                "escalated": bool(item["escalated"]),
                "final_answer": item["final_answer"],
                "final_correct": final_correct,
                "samples_used": int(item["samples_used"]),
                "signals_fired": signals_fired,
            }
        )

    total_samples = sum(int(row["samples_used"]) for row in per_query_rows)
    escalated = sum(1 for row in per_query_rows if bool(row["escalated"]))
    return (
        {
            "method": "selective_escalation_v1",
            "accuracy": 0.0 if total_queries == 0 else correct / total_queries,
            "total_samples_used": total_samples,
            "average_samples_per_query": (
                0.0 if total_queries == 0 else total_samples / total_queries
            ),
            "total_queries": total_queries,
            "queries_escalated": escalated,
            "fraction_escalated": 0.0 if total_queries == 0 else escalated / total_queries,
        },
        per_query_rows,
    )


def _result_map(results: list[BaselineResult]) -> dict[str, BaselineResult]:
    return {result.query_id: result for result in results}


def build_pairwise_comparisons(
    greedy_results: list[BaselineResult],
    best_of_3_results: list[BaselineResult],
    selective_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    greedy_map = _result_map(greedy_results)
    best_map = _result_map(best_of_3_results)

    improved_vs_greedy = 0
    worsened_vs_greedy = 0
    improved_vs_best = 0
    worsened_vs_best = 0
    for row in selective_rows:
        query_id = str(row["question_id"])
        selective_correct = bool(row["final_correct"])
        greedy_correct = bool(greedy_map[query_id].correct)
        best_correct = bool(best_map[query_id].correct)

        if selective_correct and not greedy_correct:
            improved_vs_greedy += 1
        if greedy_correct and not selective_correct:
            worsened_vs_greedy += 1
        if selective_correct and not best_correct:
            improved_vs_best += 1
        if best_correct and not selective_correct:
            worsened_vs_best += 1

    return {
        "selective_vs_greedy": {
            "queries_improved": improved_vs_greedy,
            "queries_worsened": worsened_vs_greedy,
        },
        "selective_vs_always_best_of_3": {
            "queries_improved": improved_vs_best,
            "queries_worsened": worsened_vs_best,
        },
    }


def run_selective_escalation_eval(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    queries = load_gsm8k(
        split=str(dataset_cfg.get("split", "test")),
        max_samples=dataset_cfg.get("max_samples"),
    )

    model = _build_model(config)
    greedy_summary, greedy_results = _run_greedy(queries=queries, model=model)
    best_summary, best_results = _run_always_best_of_3(queries=queries, model=model)
    method_cfg = config.get("selective_method", {})
    selective_summary, selective_rows = _run_selective(
        queries=queries,
        model=model,
        config=SelectiveEscalationConfig(
            total_budget=int(method_cfg["total_budget"]),
            escalation_target_k=1 + int(method_cfg.get("extra_samples_per_escalated_query", 2)),
            use_second_sample_for_disagreement=bool(
                method_cfg.get("use_second_sample_for_disagreement", True)
            ),
            weight_parse_failure=float(method_cfg.get("parse_failure_weight", 2.0)),
            weight_disagreement_2sample=float(method_cfg.get("disagreement_weight", 1.5)),
            weight_malformed_output=float(method_cfg.get("malformed_output_weight", 1.0)),
            weight_low_confidence_format=float(method_cfg.get("missing_numeric_weight", 1.0)),
            malformed_length_threshold=int(method_cfg.get("malformed_length_threshold", 2)),
        ),
    )

    per_query_rows: list[dict[str, Any]] = []
    greedy_map = _result_map(greedy_results)
    best_map = _result_map(best_results)
    for row in selective_rows:
        query_id = str(row["question_id"])
        greedy_result = greedy_map[query_id]
        best_result = best_map[query_id]
        per_query_rows.append(
            {
                "question_id": query_id,
                "gold_answer": row["gold_answer"],
                "greedy_final_answer": greedy_result.final_answer,
                "greedy_correct": bool(greedy_result.correct),
                "always_best_of_3_final_answer": best_result.final_answer,
                "always_best_of_3_correct": bool(best_result.correct),
                "first_pass_answer": row["first_pass_answer"],
                "first_pass_correct": bool(row["first_pass_correct"]),
                "escalation_score": float(row["escalation_score"]),
                "escalated": bool(row["escalated"]),
                "final_answer": row["final_answer"],
                "final_correct": bool(row["final_correct"]),
                "samples_used": int(row["samples_used"]),
                "signals_fired": ",".join(str(signal) for signal in row["signals_fired"]),
            }
        )

    pairwise = build_pairwise_comparisons(
        greedy_results=greedy_results,
        best_of_3_results=best_results,
        selective_rows=selective_rows,
    )

    return {
        "provider": "openai",
        "model_name": model.model_name,
        "total_queries": len(queries),
        "method_summaries": [
            greedy_summary,
            best_summary,
            selective_summary,
        ],
        "pairwise_comparisons": pairwise,
        "per_query_rows": per_query_rows,
    }


def write_selective_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    base_dir = Path(output_dir)
    return {
        "summary_json": str(
            _write_json(result, base_dir / "selective_escalation_results.json")
        ),
        "per_query_csv": str(
            _write_csv(result["per_query_rows"], base_dir / "per_query_diagnostic.csv")
        ),
    }


def format_selective_summary(result: dict[str, Any], paths: dict[str, str]) -> str:
    lines = [
        "--- Selective Escalation Summary ---",
        f"provider:                  {result['provider']}",
        f"model:                     {result['model_name']}",
        f"queries:                   {result['total_queries']}",
        "method results:",
    ]
    for summary in result["method_summaries"]:
        lines.append(
            "  "
            f"{summary['method']}: accuracy={summary['accuracy']:.4f}, "
            f"total_samples={summary['total_samples_used']}, "
            f"avg_samples/query={summary['average_samples_per_query']:.2f}, "
            f"queries_escalated={summary['queries_escalated']}, "
            f"fraction_escalated={summary['fraction_escalated']:.4f}"
        )

    pairwise = result["pairwise_comparisons"]
    lines.extend(
        [
            "",
            "pairwise comparisons:",
            "  selective vs greedy: "
            f"improved={pairwise['selective_vs_greedy']['queries_improved']}, "
            f"worsened={pairwise['selective_vs_greedy']['queries_worsened']}",
            "  selective vs always_best_of_3: "
            f"improved={pairwise['selective_vs_always_best_of_3']['queries_improved']}, "
            f"worsened={pairwise['selective_vs_always_best_of_3']['queries_worsened']}",
            "",
            f"summary_json:              {paths['summary_json']}",
            f"per_query_csv:             {paths['per_query_csv']}",
        ]
    )
    return "\n".join(lines)


def run_selective_escalation_experiment(config: dict[str, Any]) -> dict[str, Any]:
    """Compatibility wrapper used by the runner."""
    return run_selective_escalation_eval(config)


def write_selective_escalation_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Compatibility wrapper used by the runner."""
    return write_selective_outputs(result, output_dir)


def format_selective_escalation_summary(
    result: dict[str, Any],
    paths: dict[str, str],
) -> str:
    """Compatibility wrapper used by the runner."""
    return format_selective_summary(result, paths)
