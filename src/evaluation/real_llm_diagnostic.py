"""Focused real-LLM diagnostics for prompt style and extra-compute usefulness.

This module is intentionally lightweight. Its purpose is to answer whether
extra compute on real GSM8K queries helps at all, whether prompting matters,
and whether any gains are concentrated on a subset of queries.
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
from src.baselines.self_consistency import SelfConsistencyBaseline
from src.datasets.gsm8k import Query, load_gsm8k
from src.evaluation.metrics import compute_accuracy
from src.models.openai_llm import OpenAILLMModel
from src.utils.answer_extraction import extract_numeric_answer

DIRECT_PROMPT = "Answer the following question. Give only the final numeric answer."
REASONING_PROMPT = (
    "Solve the following math word problem carefully. "
    "Think step by step, then give the final numeric answer clearly at the end."
)


class NumericSelfConsistencyBaseline(SelfConsistencyBaseline):
    """Numeric majority vote with deterministic tie-breaking for diagnostics."""

    def solve(
        self, query_id: str, question: str, ground_truth: str, n_samples: int = 3
    ) -> BaselineResult:
        raw_answers = self.model.generate_n(question, n_samples)
        extracted = [extract_numeric_answer(answer) for answer in raw_answers]
        counts = Counter(extracted)
        ranked = sorted(
            counts.items(),
            key=lambda item: (-item[1], extracted.index(item[0]), item[0]),
        )
        majority = ranked[0][0]
        return BaselineResult(
            query_id=query_id,
            question=question,
            candidates=raw_answers,
            final_answer=majority,
            ground_truth=ground_truth,
            correct=(majority == ground_truth),
            samples_used=n_samples,
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


def _method_specs(config: dict[str, Any]) -> list[dict[str, Any]]:
    experiment_cfg = config.get("experiment", {})
    methods = experiment_cfg.get("methods")
    if methods is not None:
        return [dict(method) for method in methods]
    return [
        {"name": "greedy_direct", "baseline": "greedy", "n_samples": 1, "prompt_style": "direct"},
        {
            "name": "greedy_reasoning",
            "baseline": "greedy",
            "n_samples": 1,
            "prompt_style": "reasoning",
        },
        {
            "name": "best_of_3_reasoning",
            "baseline": "best_of_n",
            "n_samples": 3,
            "prompt_style": "reasoning",
        },
        {
            "name": "self_consistency_3_reasoning",
            "baseline": "self_consistency",
            "n_samples": 3,
            "prompt_style": "reasoning",
        },
    ]


def _prompt_text(prompt_style: str) -> str:
    if prompt_style == "direct":
        return DIRECT_PROMPT
    if prompt_style == "reasoning":
        return REASONING_PROMPT
    raise ValueError(f"Unsupported prompt_style '{prompt_style}'")


def _build_prompt(question: str, prompt_style: str) -> str:
    return question


def _build_model(config: dict[str, Any], prompt_style: str) -> OpenAILLMModel:
    model_cfg = config.get("model", {})
    return OpenAILLMModel(
        model_name=str(model_cfg["name"]),
        base_url=model_cfg.get("base_url"),
        prompt_prefix=_prompt_text(prompt_style),
        greedy_temperature=float(model_cfg.get("greedy_temperature", 0.0)),
        sample_temperature=float(model_cfg.get("sample_temperature", 0.7)),
        max_tokens=int(model_cfg.get("max_tokens", 256)),
        timeout_seconds=float(model_cfg.get("timeout_seconds", 60.0)),
    )


def _build_baseline(name: str, model: OpenAILLMModel):
    if name == "greedy":
        return GreedyBaseline(model)
    if name == "best_of_n":
        return BestOfNBaseline(model)
    if name == "self_consistency":
        return NumericSelfConsistencyBaseline(model)
    raise ValueError(f"Unsupported baseline '{name}'")


def _run_method(
    queries: list[Query],
    method_spec: dict[str, Any],
    config: dict[str, Any],
) -> tuple[dict[str, Any], list[BaselineResult]]:
    prompt_style = str(method_spec["prompt_style"])
    baseline_name = str(method_spec["baseline"])
    n_samples = int(method_spec["n_samples"])
    method_name = str(method_spec["name"])

    model = _build_model(config=config, prompt_style=prompt_style)
    baseline = _build_baseline(name=baseline_name, model=model)
    results: list[BaselineResult] = []

    for query in queries:
        result = baseline.solve(
            query_id=query.id,
            question=_build_prompt(query.question, prompt_style=prompt_style),
            ground_truth=query.answer,
            n_samples=n_samples,
        )
        results.append(result)

    summary = compute_accuracy(results)
    return (
        {
            "method": method_name,
            "baseline": baseline_name,
            "prompt_style": prompt_style,
            "n_samples": n_samples,
            "accuracy": float(summary["accuracy"]),
            "total_samples_used": int(summary["total_samples"]),
            "average_samples_per_query": float(summary["avg_samples_per_query"]),
            "total_queries": int(summary["total_queries"]),
        },
        results,
    )


def build_per_query_rows(
    queries: list[Query],
    method_results: dict[str, list[BaselineResult]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    method_names = list(method_results)
    for idx, query in enumerate(queries):
        row: dict[str, Any] = {
            "question_id": query.id,
            "gold_answer": query.answer,
        }
        for method_name in method_names:
            result = method_results[method_name][idx]
            row[f"{method_name}_correct"] = bool(result.correct)
            row[f"{method_name}_final_answer"] = result.final_answer
        rows.append(row)
    return rows


def _count_condition(rows: list[dict[str, Any]], lhs: str, rhs: str) -> int:
    return sum(1 for row in rows if bool(row[lhs]) and not bool(row[rhs]))


def build_overlap_summary(per_query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    greedy_direct = "greedy_direct_correct"
    greedy_reasoning = "greedy_reasoning_correct"
    best_of_3 = "best_of_3_reasoning_correct"
    self_consistency = "self_consistency_3_reasoning_correct"

    improved_by_best_of_3 = sum(
        1
        for row in per_query_rows
        if not bool(row[greedy_reasoning]) and bool(row[best_of_3])
    )
    worsened_by_best_of_3 = sum(
        1
        for row in per_query_rows
        if bool(row[greedy_reasoning]) and not bool(row[best_of_3])
    )
    improved_by_self_consistency = sum(
        1
        for row in per_query_rows
        if not bool(row[greedy_reasoning]) and bool(row[self_consistency])
    )
    worsened_by_self_consistency = sum(
        1
        for row in per_query_rows
        if bool(row[greedy_reasoning]) and not bool(row[self_consistency])
    )

    return {
        "greedy_direct_only_vs_best_of_3": _count_condition(
            per_query_rows, greedy_direct, best_of_3
        ),
        "greedy_direct_only_vs_self_consistency": _count_condition(
            per_query_rows, greedy_direct, self_consistency
        ),
        "self_consistency_only_vs_greedy_direct": _count_condition(
            per_query_rows, self_consistency, greedy_direct
        ),
        "self_consistency_only_vs_greedy_reasoning": _count_condition(
            per_query_rows, self_consistency, greedy_reasoning
        ),
        "queries_changed_correctness_best_of_3_vs_greedy_reasoning": (
            improved_by_best_of_3 + worsened_by_best_of_3
        ),
        "queries_improved_best_of_3_vs_greedy_reasoning": improved_by_best_of_3,
        "queries_worsened_best_of_3_vs_greedy_reasoning": worsened_by_best_of_3,
        "queries_changed_correctness_self_consistency_vs_greedy_reasoning": (
            improved_by_self_consistency + worsened_by_self_consistency
        ),
        "queries_improved_self_consistency_vs_greedy_reasoning": (
            improved_by_self_consistency
        ),
        "queries_worsened_self_consistency_vs_greedy_reasoning": (
            worsened_by_self_consistency
        ),
    }


def summarize_method_results(
    method_summaries: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Return method summaries in a stable display order."""
    order = {
        "greedy_direct": 0,
        "greedy_reasoning": 1,
        "best_of_3_reasoning": 2,
        "self_consistency_3_reasoning": 3,
    }
    return sorted(
        method_summaries,
        key=lambda item: order.get(str(item["method"]), 999),
    )


def build_pairwise_overlap_summary(per_query_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility wrapper used by tests and the runner."""
    return build_overlap_summary(per_query_rows)


def run_real_llm_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    """Run the focused real-GSM8K prompt/extra-compute diagnostic."""
    dataset_cfg = config.get("dataset", {})
    queries = load_gsm8k(
        split=str(dataset_cfg.get("split", "test")),
        max_samples=dataset_cfg.get("max_samples"),
    )

    method_summaries: list[dict[str, Any]] = []
    method_results: dict[str, list[BaselineResult]] = {}
    for method_spec in _method_specs(config):
        summary, results = _run_method(
            queries=queries,
            method_spec=method_spec,
            config=config,
        )
        method_summaries.append(summary)
        method_results[str(method_spec["name"])] = results

    per_query_rows = build_per_query_rows(queries=queries, method_results=method_results)
    overlap_summary = build_overlap_summary(per_query_rows)

    return {
        "total_queries": len(queries),
        "method_summaries": summarize_method_results(method_summaries),
        "per_query_rows": per_query_rows,
        "overlap_summary": overlap_summary,
    }


def write_diagnostic_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write summary JSON and per-query CSV/JSON outputs."""
    base_dir = Path(output_dir)
    paths = {
        "summary_json": str(
            _write_json(result, base_dir / "real_llm_diagnostic_results.json")
        ),
        "per_query_csv": str(
            _write_csv(result["per_query_rows"], base_dir / "per_query_diagnostic.csv")
        ),
        "per_query_json": str(
            _write_json(
                {"per_query": result["per_query_rows"]},
                base_dir / "per_query_diagnostic.json",
            )
        ),
    }
    return paths


def format_diagnostic_summary(result: dict[str, Any]) -> str:
    """Render a concise terminal summary for the diagnostic run."""
    lines = [
        "--- Real LLM Diagnostic Summary ---",
        f"queries:                   {result['total_queries']}",
        "method accuracies:",
    ]
    for summary in result["method_summaries"]:
        lines.append(
            "  "
            f"{summary['method']}: accuracy={summary['accuracy']:.4f}, "
            f"total_samples={summary['total_samples_used']}, "
            f"avg_samples/query={summary['average_samples_per_query']:.2f}"
        )

    overlap = result["overlap_summary"]
    lines.extend(
        [
            "",
            "overlap diagnostics:",
            "  queries improved by extra compute (best_of_3 vs greedy_reasoning): "
            f"{overlap['queries_improved_best_of_3_vs_greedy_reasoning']}",
            "  queries worsened by extra compute (best_of_3 vs greedy_reasoning): "
            f"{overlap['queries_worsened_best_of_3_vs_greedy_reasoning']}",
            "  queries improved by self_consistency (vs greedy_reasoning): "
            f"{overlap['queries_improved_self_consistency_vs_greedy_reasoning']}",
            "  queries worsened by self_consistency (vs greedy_reasoning): "
            f"{overlap['queries_worsened_self_consistency_vs_greedy_reasoning']}",
        ]
    )
    return "\n".join(lines)
