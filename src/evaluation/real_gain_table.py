"""Empirical real-data gain-table experiment for adaptive allocation headroom.

This module measures per-query success estimates at small compute levels on
real GSM8K queries using a real OpenAI model. Its purpose is to estimate:
1) how much headroom adaptive allocation may have on real data,
2) whether extra compute helps many queries or only a minority, and
3) whether selective allocation looks promising enough to justify a fuller
   adaptive method.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.allocators.mckp_allocator import MCKPAllocator
from src.datasets.gsm8k import Query, load_gsm8k
from src.models.openai_llm import OpenAILLMModel
from src.utils.answer_extraction import extract_numeric_answer

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


def _prompt_text(prompt_style: str) -> str:
    if prompt_style == "direct":
        return DIRECT_PROMPT
    if prompt_style == "reasoning":
        return REASONING_PROMPT
    raise ValueError(f"Unsupported prompt_style '{prompt_style}'")


def _majority_vote(parsed_answers: list[str]) -> str:
    counts = Counter(parsed_answers)
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], parsed_answers.index(item[0]), item[0]),
    )
    return ranked[0][0]


def _build_model(config: dict[str, Any]) -> OpenAILLMModel:
    model_cfg = config.get("model", {})
    return OpenAILLMModel(
        model_name=str(model_cfg["name"]),
        base_url=model_cfg.get("base_url"),
        prompt_prefix=_prompt_text(str(config.get("prompt_style", "direct"))),
        greedy_temperature=float(model_cfg.get("greedy_temperature", 0.0)),
        sample_temperature=float(model_cfg.get("sample_temperature", 0.7)),
        max_tokens=int(model_cfg.get("max_tokens", 256)),
        timeout_seconds=float(model_cfg.get("timeout_seconds", 60.0)),
    )


def _sample_levels(config: dict[str, Any]) -> list[int]:
    levels = [int(level) for level in config.get("sample_levels", [1, 2, 3])]
    if not levels:
        raise ValueError("sample_levels must be non-empty")
    if levels != [1, 2, 3]:
        raise ValueError(
            "This lightweight experiment currently supports "
            "sample_levels [1, 2, 3] only."
        )
    return levels


def _estimate_query_success(
    model: OpenAILLMModel,
    query: Query,
    n_trials: int,
    sample_levels: list[int],
) -> dict[str, Any]:
    successes = {level: 0 for level in sample_levels}
    prompt = query.question

    for _ in range(n_trials):
        raw_answers = model.generate_n(prompt, max(sample_levels))
        parsed_answers = [extract_numeric_answer(answer) for answer in raw_answers]
        for level in sample_levels:
            voted_answer = _majority_vote(parsed_answers[:level])
            successes[level] += int(voted_answer == query.answer)

    success_k1 = successes[1] / n_trials
    success_k2 = successes[2] / n_trials
    success_k3 = successes[3] / n_trials
    gain_1_to_2 = success_k2 - success_k1
    gain_2_to_3 = success_k3 - success_k2

    return {
        "question_id": query.id,
        "gold_answer": query.answer,
        "empirical_success_k1": success_k1,
        "empirical_success_k2": success_k2,
        "empirical_success_k3": success_k3,
        "marginal_gain_1_to_2": gain_1_to_2,
        "marginal_gain_2_to_3": gain_2_to_3,
        "positive_gain": bool(success_k3 > success_k1),
        "diminishing_returns": bool(gain_1_to_2 >= gain_2_to_3),
        "no_gain": bool(success_k1 == success_k2 == success_k3),
    }


def build_gain_table_rows(raw_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert nested empirical success records into flat gain-table rows."""
    flattened: list[dict[str, Any]] = []
    for row in raw_rows:
        success = row["empirical_success"]
        success_k1 = float(success[1])
        success_k2 = float(success[2])
        success_k3 = float(success[3])
        gain_1_to_2 = success_k2 - success_k1
        gain_2_to_3 = success_k3 - success_k2
        flattened.append(
            {
                "question_id": row["question_id"],
                "gold_answer": row["gold_answer"],
                "empirical_success_k1": success_k1,
                "empirical_success_k2": success_k2,
                "empirical_success_k3": success_k3,
                "marginal_gain_1_to_2": gain_1_to_2,
                "marginal_gain_2_to_3": gain_2_to_3,
                "positive_gain": bool(success_k3 > success_k1),
                "diminishing_returns": bool(gain_1_to_2 >= gain_2_to_3),
                "no_gain": bool(success_k1 == success_k2 == success_k3),
            }
        )
    return flattened


def _uniform_levels(n_queries: int, budget: int) -> list[int]:
    """Allocate nearly uniform sample counts in {1,2,3} under a total budget."""
    if budget < n_queries or budget > 3 * n_queries:
        raise ValueError(
            f"budget must lie in [{n_queries}, {3 * n_queries}] for {n_queries} queries"
        )

    base = budget // n_queries
    remainder = budget % n_queries
    levels = [base] * n_queries

    if base < 1 or base > 3:
        raise ValueError("uniform base level must fall in [1, 3]")
    if base == 3 and remainder > 0:
        raise ValueError("budget exceeds maximum feasible uniform assignment")

    for idx in range(remainder):
        if levels[idx] >= 3:
            raise ValueError("uniform allocation exceeded supported sample level 3")
        levels[idx] += 1
    return levels


def build_budget_comparison(
    gain_rows: list[dict[str, Any]],
    budgets: list[int],
) -> list[dict[str, Any]]:
    if not gain_rows:
        raise ValueError("gain_rows must be non-empty")
    if not budgets:
        raise ValueError("budgets must be non-empty")

    n_queries = len(gain_rows)
    min_budget = n_queries
    max_budget = 3 * n_queries
    if any(budget < min_budget or budget > max_budget for budget in budgets):
        raise ValueError(
            f"budgets must lie in [{min_budget}, {max_budget}] "
            f"for {n_queries} queries and k in [1, 3]"
        )

    utility_table = [
        [
            float(row["empirical_success_k1"]),
            float(row["empirical_success_k2"]),
            float(row["empirical_success_k3"]),
        ]
        for row in gain_rows
    ]
    costs = [1, 2, 3]

    oracle_allocator = MCKPAllocator()
    comparisons: list[dict[str, Any]] = []

    for budget in budgets:
        uniform_counts = _uniform_levels(n_queries, budget)
        uniform_expected = float(
            sum(utility_table[idx][count - 1] for idx, count in enumerate(uniform_counts))
        )

        oracle_result = oracle_allocator.allocate(
            profits=utility_table,
            costs=costs,
            budget=budget,
        )
        oracle_expected = float(oracle_result["total_profit"])
        absolute_gap = oracle_expected - uniform_expected
        relative_gap = None
        if abs(uniform_expected) > 1e-12:
            relative_gap = absolute_gap / uniform_expected

        comparisons.append(
            {
                "budget": int(budget),
                "uniform_expected_solved": uniform_expected,
                "oracle_expected_solved": oracle_expected,
                "absolute_gap_oracle_minus_uniform": absolute_gap,
                "relative_gap_vs_uniform": relative_gap,
            }
        )

    return comparisons


def summarize_gain_table(
    gain_rows: list[dict[str, Any]],
    budget_comparison: list[dict[str, Any]],
) -> dict[str, Any]:
    if not gain_rows:
        raise ValueError("gain_rows must be non-empty")
    if not budget_comparison:
        raise ValueError("budget_comparison must be non-empty")

    n_queries = len(gain_rows)
    positive_gain_count = sum(
        int(
            bool(
                row.get(
                    "positive_gain",
                    row["empirical_success_k3"] > row["empirical_success_k1"],
                )
            )
        )
        for row in gain_rows
    )
    diminishing_returns_count = sum(
        int(
            bool(
                row.get(
                    "diminishing_returns",
                    row["marginal_gain_1_to_2"] >= row["marginal_gain_2_to_3"],
                )
            )
        )
        for row in gain_rows
    )
    no_gain_count = sum(
        int(
            bool(
                row.get(
                    "no_gain",
                    row["empirical_success_k1"]
                    == row["empirical_success_k2"]
                    == row["empirical_success_k3"],
                )
            )
        )
        for row in gain_rows
    )
    strongest_budget = max(
        budget_comparison,
        key=lambda row: float(row["absolute_gap_oracle_minus_uniform"]),
    )

    return {
        "total_queries": n_queries,
        "fraction_positive_gain_from_extra_compute": positive_gain_count / n_queries,
        "fraction_diminishing_returns": diminishing_returns_count / n_queries,
        "fraction_no_gain": no_gain_count / n_queries,
        "budget_where_oracle_gap_peaks": int(strongest_budget["budget"]),
        "max_oracle_uniform_gap": float(strongest_budget["absolute_gap_oracle_minus_uniform"]),
    }


def run_real_gain_table(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    queries = load_gsm8k(
        split=str(dataset_cfg.get("split", "test")),
        max_samples=dataset_cfg.get("max_samples"),
    )
    n_trials = int(config.get("trials_per_query", config.get("n_trials", 4)))
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")

    sample_levels = _sample_levels(config)
    model = _build_model(config)
    gain_rows = [
        _estimate_query_success(
            model=model,
            query=query,
            n_trials=n_trials,
            sample_levels=sample_levels,
        )
        for query in queries
    ]
    budget_comparison = build_budget_comparison(
        gain_rows=gain_rows,
        budgets=[int(budget) for budget in config.get("budgets", [])],
    )
    summary = summarize_gain_table(gain_rows=gain_rows, budget_comparison=budget_comparison)

    total_requests = len(queries) * n_trials
    total_model_samples = len(queries) * n_trials * max(sample_levels)

    return {
        "provider": "openai",
        "model_name": model.model_name,
        "prompt_style": str(config.get("prompt_style", "direct")),
        "n_trials": n_trials,
        "sample_levels": sample_levels,
        "total_queries": len(queries),
        "total_api_requests": total_requests,
        "total_model_samples": total_model_samples,
        "per_query_gain_table": gain_rows,
        "budget_comparison": budget_comparison,
        "summary": summary,
    }


def write_gain_table_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    base_dir = Path(output_dir)
    return {
        "per_query_csv": str(
            _write_csv(result["per_query_gain_table"], base_dir / "per_query_gain_table.csv")
        ),
        "per_query_json": str(
            _write_json(
                {"per_query_gain_table": result["per_query_gain_table"]},
                base_dir / "per_query_gain_table.json",
            )
        ),
        "budget_csv": str(
            _write_csv(result["budget_comparison"], base_dir / "budget_comparison.csv")
        ),
        "summary_json": str(
            _write_json(
                {
                    "provider": result["provider"],
                    "model_name": result["model_name"],
                    "prompt_style": result["prompt_style"],
                    "n_trials": result["n_trials"],
                    "sample_levels": result["sample_levels"],
                    "total_queries": result["total_queries"],
                    "total_api_requests": result["total_api_requests"],
                    "total_model_samples": result["total_model_samples"],
                    "summary": result["summary"],
                },
                base_dir / "summary.json",
            )
        ),
    }


def format_gain_table_summary(result: dict[str, Any], paths: dict[str, str]) -> str:
    lines = [
        "--- Real Gain Table Summary ---",
        f"provider:                  {result['provider']}",
        f"model:                     {result['model_name']}",
        f"queries:                   {result['total_queries']}",
        f"prompt_style:              {result['prompt_style']}",
        f"n_trials:                  {result['n_trials']}",
        f"total_api_requests:        {result['total_api_requests']}",
        f"total_model_samples:       {result['total_model_samples']}",
        "",
        "headroom summary:",
        "  fraction positive gain:   "
        f"{result['summary']['fraction_positive_gain_from_extra_compute']:.4f}",
        "  fraction diminishing returns: "
        f"{result['summary']['fraction_diminishing_returns']:.4f}",
        "  fraction no gain:         "
        f"{result['summary']['fraction_no_gain']:.4f}",
        "  best oracle-vs-uniform budget: "
        f"{result['summary']['budget_where_oracle_gap_peaks']}",
        "  max oracle-vs-uniform gap: "
        f"{result['summary']['max_oracle_uniform_gap']:.4f}",
        "",
        f"per_query_csv:             {paths['per_query_csv']}",
        f"budget_csv:                {paths['budget_csv']}",
        f"summary_json:              {paths['summary_json']}",
    ]
    return "\n".join(lines)
