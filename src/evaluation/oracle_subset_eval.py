"""Oracle subset evaluation over the current real-LLM strategy set.

This module intentionally reuses the repository's existing strategy
implementations instead of introducing new methods.  Its purpose is to answer:

1. Whether one-pass reasoning is the main useful intervention.
2. Whether extra sampling is mostly wasteful on the current subset.
3. Whether multi-stage corrective strategies recover failures that simple
   direct or reasoning baselines miss.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Callable

from src.datasets.gsm8k import Query, load_gsm8k
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
    run_structured_sampling_3,
)
from src.models.openai_llm import OpenAILLMModel
from src.utils.answer_extraction import extract_numeric_answer

REASONING_GREEDY_PROMPT = (
    "Solve this step by step and end with 'Final answer: <number>'.\n\n{question}"
)


@dataclass(frozen=True)
class StrategyDefinition:
    name: str
    runner: Callable[[Any, str], dict[str, Any]]
    model_slot: str = "current"


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


def run_reasoning_greedy(model: Any, question: str) -> dict[str, Any]:
    """One reasoning-style sample with the current model."""
    raw = model.generate(REASONING_GREEDY_PROMPT.format(question=question))
    answer = _normalize(extract_numeric_answer(raw))
    return {
        "raw_outputs": [raw],
        "predicted_answer": answer,
        "samples_used": 1,
    }


STRATEGY_DEFINITIONS = {
    "direct_greedy": StrategyDefinition("direct_greedy", run_direct_greedy),
    "reasoning_greedy": StrategyDefinition("reasoning_greedy", run_reasoning_greedy),
    "reasoning_best_of_3": StrategyDefinition(
        "reasoning_best_of_3", run_reasoning_best_of_3
    ),
    "structured_sampling_3": StrategyDefinition(
        "structured_sampling_3", run_structured_sampling_3
    ),
    "direct_plus_verify": StrategyDefinition("direct_plus_verify", run_direct_plus_verify),
    "direct_plus_revise": StrategyDefinition("direct_plus_revise", run_direct_plus_revise),
    "direct_plus_critique_plus_final": StrategyDefinition(
        "direct_plus_critique_plus_final",
        run_direct_plus_critique_plus_final,
    ),
    "first_pass_then_hint_guided_reason": StrategyDefinition(
        "first_pass_then_hint_guided_reason",
        run_first_pass_then_hint_guided_reason,
    ),
    "strong_direct": StrategyDefinition("strong_direct", run_direct_greedy, model_slot="strong"),
}


def _build_model(
    model_name: str,
    config: dict[str, Any],
    max_tokens_override: int | None = None,
) -> OpenAILLMModel:
    openai_cfg = config.get("openai", {})
    max_tokens = (
        int(max_tokens_override)
        if max_tokens_override is not None
        else int(openai_cfg.get("max_tokens", 256))
    )
    return OpenAILLMModel(
        model_name=model_name,
        base_url=openai_cfg.get("base_url"),
        greedy_temperature=float(openai_cfg.get("greedy_temperature", 0.0)),
        sample_temperature=float(openai_cfg.get("sample_temperature", 0.7)),
        max_tokens=max_tokens,
        timeout_seconds=float(openai_cfg.get("timeout_seconds", 60.0)),
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


def _load_queries(config: dict[str, Any]) -> tuple[list[Query], dict[str, Any]]:
    dataset_cfg = config.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name", "gsm8k"))
    if dataset_name != "gsm8k":
        raise RuntimeError(
            "Current oracle subset runner only supports gsm8k. The existing multi-stage "
            "strategy implementations are numeric-answer-only, so extending the full oracle "
            "strategy set to MATH500 symbolic answers would require broader changes."
        )

    split = str(dataset_cfg.get("split", "test"))
    data_file = dataset_cfg.get("data_file")
    requested_max_samples = int(dataset_cfg.get("max_samples", 20))
    queries = load_gsm8k(
        split=split,
        max_samples=requested_max_samples,
        cache_dir=str(dataset_cfg.get("cache_dir", "data")),
        data_file=data_file,
    )
    return queries, {
        "name": "gsm8k",
        "source": data_file or "openai/gsm8k",
        "split": split,
        "requested_max_samples": requested_max_samples,
        "num_queries": len(queries),
        "question_ids": [query.id for query in queries],
    }


def _choose_cheapest_correct(
    rows: list[dict[str, Any]],
    strategy_order: list[str],
) -> dict[str, Any] | None:
    correct_rows = [row for row in rows if bool(row["correct"])]
    if not correct_rows:
        return None
    order_lookup = {strategy: idx for idx, strategy in enumerate(strategy_order)}
    return min(
        correct_rows,
        key=lambda row: (
            int(row["samples_used"]),
            order_lookup.get(str(row["strategy"]), 999),
        ),
    )


def build_oracle_subset_artifacts(
    per_query_rows: list[dict[str, Any]],
    strategies: list[str],
    total_queries: int,
) -> dict[str, Any]:
    """Aggregate long-form per-query strategy rows into oracle-style outputs."""
    by_query: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_query_rows:
        by_query[str(row["question_id"])].append(row)

    direct_name = "direct_greedy"
    reasoning_name = "reasoning_greedy"
    order_lookup = {strategy: idx for idx, strategy in enumerate(strategies)}

    per_query_matrix_rows: list[dict[str, Any]] = []
    oracle_assignments: list[dict[str, Any]] = []
    cheapest_correct_counts = {strategy: 0 for strategy in strategies}
    unique_wins = {strategy: 0 for strategy in strategies}
    fixes_direct_failures = {strategy: 0 for strategy in strategies}
    fixes_reasoning_failures = {strategy: 0 for strategy in strategies}
    no_strategy_succeeded_count = 0

    for question_id, rows in by_query.items():
        sorted_rows = sorted(
            rows,
            key=lambda row: order_lookup.get(str(row["strategy"]), 999),
        )
        row_by_strategy = {str(row["strategy"]): row for row in sorted_rows}
        gold_answer = str(sorted_rows[0]["gold_answer"])
        correct_strategies = [
            str(row["strategy"]) for row in sorted_rows if bool(row["correct"])
        ]
        cheapest_correct = _choose_cheapest_correct(sorted_rows, strategies)
        cheapest_correct_strategy = (
            "" if cheapest_correct is None else str(cheapest_correct["strategy"])
        )
        if cheapest_correct_strategy:
            cheapest_correct_counts[cheapest_correct_strategy] += 1
        if not correct_strategies:
            no_strategy_succeeded_count += 1
        if len(correct_strategies) == 1:
            unique_wins[correct_strategies[0]] += 1

        direct_failed = (
            direct_name in row_by_strategy and not bool(row_by_strategy[direct_name]["correct"])
        )
        reasoning_failed = (
            reasoning_name in row_by_strategy
            and not bool(row_by_strategy[reasoning_name]["correct"])
        )
        for strategy in correct_strategies:
            if direct_failed:
                fixes_direct_failures[strategy] += 1
            if reasoning_failed:
                fixes_reasoning_failures[strategy] += 1

        matrix_row: dict[str, Any] = {
            "question_id": question_id,
            "gold_answer": gold_answer,
            "num_correct_strategies": len(correct_strategies),
            "correct_strategies": json.dumps(correct_strategies),
            "oracle_strategy": cheapest_correct_strategy,
            "oracle_correct": bool(correct_strategies),
            "oracle_cost": (
                "" if cheapest_correct is None else int(cheapest_correct["samples_used"])
            ),
        }
        for strategy in strategies:
            row = row_by_strategy.get(strategy)
            if row is None:
                matrix_row[f"{strategy}_predicted_answer"] = ""
                matrix_row[f"{strategy}_correct"] = False
                matrix_row[f"{strategy}_samples_used"] = ""
            else:
                matrix_row[f"{strategy}_predicted_answer"] = row["predicted_answer"]
                matrix_row[f"{strategy}_correct"] = bool(row["correct"])
                matrix_row[f"{strategy}_samples_used"] = int(row["samples_used"])
        per_query_matrix_rows.append(matrix_row)

        oracle_assignments.append(
            {
                "question_id": question_id,
                "gold_answer": gold_answer,
                "oracle_strategy": cheapest_correct_strategy,
                "oracle_correct": bool(correct_strategies),
                "oracle_cost": (
                    "" if cheapest_correct is None else int(cheapest_correct["samples_used"])
                ),
                "correct_strategies": json.dumps(correct_strategies),
            }
        )

    summary_rows: list[dict[str, Any]] = []
    for strategy in strategies:
        rows = [row for row in per_query_rows if str(row["strategy"]) == strategy]
        correct_count = sum(1 for row in rows if bool(row["correct"]))
        total_cost = sum(int(row["samples_used"]) for row in rows)
        summary_rows.append(
            {
                "strategy": strategy,
                "accuracy": 0.0 if total_queries == 0 else correct_count / total_queries,
                "correct": correct_count,
                "total_queries": total_queries,
                "wins": correct_count,
                "unique_wins": unique_wins[strategy],
                "fixes_direct_failures": fixes_direct_failures[strategy],
                "fixes_reasoning_failures": (
                    fixes_reasoning_failures[strategy]
                    if reasoning_name in strategies
                    else ""
                ),
                "cheapest_correct_count": cheapest_correct_counts[strategy],
                "cheapest_correct_fraction": (
                    0.0 if total_queries == 0 else cheapest_correct_counts[strategy] / total_queries
                ),
                "avg_cost": 0.0 if total_queries == 0 else total_cost / total_queries,
            }
        )

    summary_by_strategy = {row["strategy"]: row for row in summary_rows}
    oracle_correct_count = sum(1 for row in oracle_assignments if bool(row["oracle_correct"]))
    oracle_costs = [
        int(row["oracle_cost"])
        for row in oracle_assignments
        if str(row["oracle_cost"]).strip()
    ]
    direct_accuracy = summary_by_strategy.get(direct_name, {}).get("accuracy")
    reasoning_accuracy = summary_by_strategy.get(reasoning_name, {}).get("accuracy")
    strong_direct_accuracy = summary_by_strategy.get("strong_direct", {}).get("accuracy")
    oracle_accuracy = 0.0 if total_queries == 0 else oracle_correct_count / total_queries

    global_metrics = {
        "direct_accuracy": direct_accuracy,
        "reasoning_greedy_accuracy": reasoning_accuracy if reasoning_name in strategies else None,
        "strong_direct_accuracy": (
            strong_direct_accuracy if "strong_direct" in summary_by_strategy else None
        ),
        "oracle_accuracy": oracle_accuracy,
        "oracle_minus_direct_gap": (
            None if direct_accuracy is None else round(oracle_accuracy - float(direct_accuracy), 4)
        ),
        "oracle_minus_reasoning_greedy_gap": (
            None
            if reasoning_accuracy is None
            else round(oracle_accuracy - float(reasoning_accuracy), 4)
        ),
        "fraction_direct_greedy_already_optimal": (
            0.0
            if total_queries == 0
            else cheapest_correct_counts.get(direct_name, 0) / total_queries
        ),
        "fraction_reasoning_greedy_already_optimal": (
            None
            if reasoning_name not in strategies or total_queries == 0
            else cheapest_correct_counts.get(reasoning_name, 0) / total_queries
        ),
        "no_strategy_succeeded_count": no_strategy_succeeded_count,
        "no_strategy_succeeded_fraction": (
            0.0 if total_queries == 0 else no_strategy_succeeded_count / total_queries
        ),
        "average_oracle_cost_on_success": (
            0.0 if not oracle_costs else sum(oracle_costs) / len(oracle_costs)
        ),
    }

    pairwise_win_matrix_rows: list[dict[str, Any]] = []
    for strategy_a in strategies:
        row = {"strategy": strategy_a}
        rows_a = {
            str(item["question_id"]): item
            for item in per_query_rows
            if str(item["strategy"]) == strategy_a
        }
        for strategy_b in strategies:
            rows_b = {
                str(item["question_id"]): item
                for item in per_query_rows
                if str(item["strategy"]) == strategy_b
            }
            wins = 0
            for question_id, row_a in rows_a.items():
                row_b = rows_b.get(question_id)
                if row_b is None:
                    continue
                if bool(row_a["correct"]) and not bool(row_b["correct"]):
                    wins += 1
            row[strategy_b] = wins
        pairwise_win_matrix_rows.append(row)

    return {
        "summary_rows": summary_rows,
        "per_query_matrix_rows": per_query_matrix_rows,
        "oracle_assignments": oracle_assignments,
        "pairwise_win_matrix_rows": pairwise_win_matrix_rows,
        "global_metrics": global_metrics,
    }


def run_oracle_subset_eval(config: dict[str, Any]) -> dict[str, Any]:
    queries, dataset_metadata = _load_queries(config)
    models_cfg = config.get("models", {})
    current_model_name = str(models_cfg["current"])
    strong_model_name = models_cfg.get("strong")
    strong_model_label = None if strong_model_name is None else str(strong_model_name)

    requested_strategies = list(
        config.get(
            "strategies",
            [
                "direct_greedy",
                "reasoning_greedy",
                "reasoning_best_of_3",
                "structured_sampling_3",
                "direct_plus_verify",
                "direct_plus_revise",
                "direct_plus_critique_plus_final",
                "first_pass_then_hint_guided_reason",
            ],
        )
    )
    unknown = [
        strategy
        for strategy in requested_strategies
        if strategy not in STRATEGY_DEFINITIONS
    ]
    if unknown:
        raise ValueError(f"Unknown strategies requested: {unknown}")

    access_checks: list[dict[str, Any]] = []
    current_probe = _probe_model_access(current_model_name, config)
    access_checks.append(current_probe)
    if current_probe["status"] != "accessible":
        raise RuntimeError(
            f"Current model '{current_model_name}' is not accessible: {current_probe['error']}"
        )

    strong_status: dict[str, Any] | None = None
    strong_model: OpenAILLMModel | None = None
    active_strategies = list(requested_strategies)
    if "strong_direct" in active_strategies:
        if strong_model_label is None:
            active_strategies = [s for s in active_strategies if s != "strong_direct"]
            strong_status = {
                "model": None,
                "status": "not_configured",
                "error_type": "config",
                "error": "No strong model configured for strong_direct",
            }
        else:
            strong_probe = _probe_model_access(strong_model_label, config)
            access_checks.append(strong_probe)
            if strong_probe["status"] == "accessible":
                strong_model = _build_model(strong_model_label, config)
                strong_status = {
                    "model": strong_model_label,
                    "status": "tested",
                    "error_type": None,
                    "error": None,
                }
            else:
                active_strategies = [s for s in active_strategies if s != "strong_direct"]
                strong_status = strong_probe

    current_model = _build_model(current_model_name, config)

    per_query_rows: list[dict[str, Any]] = []
    for query in queries:
        for strategy in active_strategies:
            definition = STRATEGY_DEFINITIONS[strategy]
            model = current_model if definition.model_slot == "current" else strong_model
            if model is None:
                raise RuntimeError(f"Strategy '{strategy}' requires an accessible strong model")
            try:
                result = definition.runner(model, query.question)
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Strategy '{strategy}' failed on query '{query.id}': {exc}"
                ) from exc
            predicted_answer = _normalize(str(result["predicted_answer"]))
            gold_answer = _normalize(query.answer)
            per_query_rows.append(
                {
                    "question_id": query.id,
                    "strategy": strategy,
                    "predicted_answer": predicted_answer,
                    "gold_answer": gold_answer,
                    "correct": predicted_answer == gold_answer,
                    "samples_used": int(result["samples_used"]),
                    "model": current_model_name
                    if definition.model_slot == "current"
                    else strong_model_label,
                }
            )

    artifacts = build_oracle_subset_artifacts(
        per_query_rows=per_query_rows,
        strategies=active_strategies,
        total_queries=len(queries),
    )

    return {
        "run_status": "COMPLETED",
        "dataset": dataset_metadata,
        "models": {
            "current": current_model_name,
            "strong": strong_model_label,
            "strong_model_status": strong_status,
        },
        "access_checks": access_checks,
        "strategies_run": active_strategies,
        "per_query_rows": per_query_rows,
        "summary_rows": artifacts["summary_rows"],
        "per_query_matrix_rows": artifacts["per_query_matrix_rows"],
        "oracle_assignments": artifacts["oracle_assignments"],
        "pairwise_win_matrix_rows": artifacts["pairwise_win_matrix_rows"],
        "global_metrics": artifacts["global_metrics"],
    }


def _write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> str:
    resolved = Path(output_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        resolved.write_text("")
        return str(resolved)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with resolved.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return str(resolved)


def write_oracle_subset_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    summary_payload = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "per_query_rows",
            "per_query_matrix_rows",
            "oracle_assignments",
            "pairwise_win_matrix_rows",
        }
    }
    summary_json = base / "summary.json"
    summary_json.write_text(json.dumps(summary_payload, indent=2))

    summary_rows = list(result["summary_rows"])
    summary_rows.append(
        {
            "strategy": "__oracle__",
            "accuracy": result["global_metrics"]["oracle_accuracy"],
            "correct": int(
                round(
                    float(result["global_metrics"]["oracle_accuracy"])
                    * int(result["dataset"]["num_queries"])
                )
            ),
            "total_queries": result["dataset"]["num_queries"],
            "wins": "",
            "unique_wins": "",
            "fixes_direct_failures": "",
            "fixes_reasoning_failures": "",
            "cheapest_correct_count": "",
            "cheapest_correct_fraction": "",
            "avg_cost": result["global_metrics"]["average_oracle_cost_on_success"],
        }
    )

    return {
        "summary_json": str(summary_json),
        "summary_csv": _write_csv(summary_rows, base / "summary.csv"),
        "per_query_matrix_csv": _write_csv(
            result["per_query_matrix_rows"],
            base / "per_query_matrix.csv",
        ),
        "oracle_assignments_csv": _write_csv(
            result["oracle_assignments"],
            base / "oracle_assignments.csv",
        ),
        "pairwise_win_matrix_csv": _write_csv(
            result["pairwise_win_matrix_rows"],
            base / "pairwise_win_matrix.csv",
        ),
    }


def format_oracle_subset_summary(
    result: dict[str, Any],
    paths: dict[str, str] | None = None,
) -> str:
    lines = [
        "--- Oracle Subset Evaluation ---",
        f"dataset: {result['dataset']['name']}",
        f"queries: {result['dataset']['num_queries']}",
        f"strategies: {', '.join(result['strategies_run'])}",
        "",
        "per-strategy accuracy:",
    ]
    for row in result["summary_rows"]:
        lines.append(
            "  "
            f"{row['strategy']} | accuracy={float(row['accuracy']):.4f} | "
            f"avg_cost={float(row['avg_cost']):.2f} | "
            f"cheapest_correct={row['cheapest_correct_count']}"
        )
    lines.extend(
        [
            "",
            (
                "oracle_accuracy: "
                f"{float(result['global_metrics']['oracle_accuracy']):.4f}"
            ),
            (
                "oracle_minus_direct_gap: "
                f"{result['global_metrics']['oracle_minus_direct_gap']}"
            ),
            (
                "oracle_minus_reasoning_greedy_gap: "
                f"{result['global_metrics']['oracle_minus_reasoning_greedy_gap']}"
            ),
        ]
    )
    if paths is not None:
        lines.extend(
            [
                "",
                f"summary_json: {paths['summary_json']}",
                f"summary_csv: {paths['summary_csv']}",
                f"per_query_matrix_csv: {paths['per_query_matrix_csv']}",
                f"oracle_assignments_csv: {paths['oracle_assignments_csv']}",
                f"pairwise_win_matrix_csv: {paths['pairwise_win_matrix_csv']}",
            ]
        )
    return "\n".join(lines)
