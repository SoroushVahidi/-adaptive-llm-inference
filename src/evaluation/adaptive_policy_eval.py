"""Evaluation helpers for the first rule-based adaptive policy baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.datasets.gsm8k import Query, load_gsm8k
from src.evaluation.oracle_subset_eval import (
    STRATEGY_DEFINITIONS,
    _build_model,
    _normalize,
    _probe_model_access,
)
from src.policies.adaptive_policy_v1 import (
    AdaptivePolicyV1Config,
    choose_strategy,
    extract_first_pass_features,
    extract_question_features,
)


def _write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> str:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return str(path)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
    return str(path)


def _load_queries(config: dict[str, Any]) -> tuple[list[Query], dict[str, Any]]:
    dataset_cfg = config.get("dataset", {})
    queries = load_gsm8k(
        split=str(dataset_cfg.get("split", "test")),
        max_samples=int(dataset_cfg.get("max_samples", 20)),
        cache_dir=str(dataset_cfg.get("cache_dir", "data")),
        data_file=dataset_cfg.get("data_file"),
    )
    return queries, {
        "name": "gsm8k",
        "source": dataset_cfg.get("data_file") or "openai/gsm8k",
        "split": str(dataset_cfg.get("split", "test")),
        "requested_max_samples": int(dataset_cfg.get("max_samples", 20)),
        "num_queries": len(queries),
        "question_ids": [query.id for query in queries],
    }


def _policy_config(config: dict[str, Any]) -> AdaptivePolicyV1Config:
    policy_cfg = config.get("policy", {})
    return AdaptivePolicyV1Config(
        simple_max_numeric_mentions=int(
            policy_cfg.get("simple_numeric_mentions_threshold", 2)
        ),
        simple_max_word_count=int(policy_cfg.get("simple_word_count_threshold", 28)),
        high_complexity_word_count=int(
            policy_cfg.get("high_complexity_word_count_threshold", 60)
        ),
        unstable_max_output_numbers=int(
            policy_cfg.get("unstable_many_numbers_threshold", 4)
        ),
        allow_reasoning_best_of_3=bool(
            policy_cfg.get("use_reasoning_best_of_3_fallback", False)
        ),
        allow_strong_direct=bool(policy_cfg.get("use_strong_direct_fallback", False)),
    )


def _run_strategy_rows(
    strategy_name: str,
    queries: list[Query],
    current_model: Any,
    strong_model: Any | None,
) -> list[dict[str, Any]]:
    definition = STRATEGY_DEFINITIONS[strategy_name]
    model = current_model if definition.model_slot == "current" else strong_model
    if model is None:
        raise RuntimeError(f"Strategy '{strategy_name}' requires an accessible strong model")

    rows: list[dict[str, Any]] = []
    for query in queries:
        result = definition.runner(model, query.question)
        predicted_answer = _normalize(str(result["predicted_answer"]))
        gold_answer = _normalize(query.answer)
        rows.append(
            {
                "question_id": query.id,
                "strategy": strategy_name,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "correct": predicted_answer == gold_answer,
                "samples_used": int(result["samples_used"]),
            }
        )
    return rows


def _summarize_rows(strategy_name: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_queries = len(rows)
    correct = sum(1 for row in rows if bool(row["correct"]))
    total_cost = sum(int(row["samples_used"]) for row in rows)
    return {
        "strategy": strategy_name,
        "accuracy": 0.0 if total_queries == 0 else correct / total_queries,
        "correct": correct,
        "total_queries": total_queries,
        "avg_cost": 0.0 if total_queries == 0 else total_cost / total_queries,
        "total_cost": total_cost,
    }


def run_adaptive_policy_eval(config: dict[str, Any]) -> dict[str, Any]:
    queries, dataset_metadata = _load_queries(config)
    models_cfg = config.get("models", {})
    current_model_name = str(models_cfg["current"])
    strong_model_name = models_cfg.get("strong")
    strong_model_label = None if strong_model_name is None else str(strong_model_name)

    access_checks: list[dict[str, Any]] = []
    current_probe = _probe_model_access(current_model_name, config)
    access_checks.append(current_probe)
    if current_probe["status"] != "accessible":
        raise RuntimeError(
            f"Current model '{current_model_name}' is not accessible: {current_probe['error']}"
        )

    policy_config = _policy_config(config)
    current_model = _build_model(current_model_name, config)

    strong_status: dict[str, Any] | None = None
    strong_model = None
    if policy_config.allow_strong_direct:
        if strong_model_label is None:
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
                strong_status = strong_probe

    baseline_names = ["direct_greedy", "reasoning_greedy", "direct_plus_revise"]
    baseline_rows = {
        name: _run_strategy_rows(name, queries, current_model, strong_model)
        for name in baseline_names
    }

    oracle_metrics = None
    oracle_summary_path = Path("outputs/oracle_subset_eval/summary.json")
    if oracle_summary_path.exists():
        oracle_metrics = json.loads(oracle_summary_path.read_text()).get("global_metrics")

    per_query_results: list[dict[str, Any]] = []
    revise_trigger_count = 0
    fallback_trigger_count = 0

    for query in queries:
        question_features = extract_question_features(query.question)
        initial_strategy = choose_strategy(
            question_text=query.question,
            features=question_features,
            first_pass_output=None,
            config=policy_config,
        )

        reasoning_first_output = None
        first_pass_features = None
        final_strategy = initial_strategy
        route_reason = (
            "simple_question"
            if initial_strategy == "direct_greedy"
            else "default_reasoning"
        )

        if initial_strategy != "direct_greedy":
            reasoning_result = STRATEGY_DEFINITIONS["reasoning_greedy"].runner(
                current_model, query.question
            )
            reasoning_first_output = str(reasoning_result["raw_outputs"][0])
            first_pass_features = extract_first_pass_features(
                reasoning_first_output,
                policy_config,
            )
            final_strategy = choose_strategy(
                question_text=query.question,
                features=question_features,
                first_pass_output=reasoning_first_output,
                config=policy_config,
            )
            if final_strategy == "reasoning_greedy":
                route_reason = "reasoning_greedy_stable"
            elif final_strategy == "direct_plus_revise":
                revise_trigger_count += 1
                route_reason = "unstable_reasoning_output"
            else:
                fallback_trigger_count += 1
                route_reason = "rare_fallback"

        definition = STRATEGY_DEFINITIONS[final_strategy]
        final_model = current_model if definition.model_slot == "current" else strong_model
        if final_model is None:
            raise RuntimeError(
                f"Adaptive policy selected '{final_strategy}' without an accessible strong model"
            )
        final_result = definition.runner(final_model, query.question)
        predicted_answer = _normalize(str(final_result["predicted_answer"]))
        gold_answer = _normalize(query.answer)

        baseline_lookup = {
            name: next(row for row in rows if row["question_id"] == query.id)
            for name, rows in baseline_rows.items()
        }
        per_query_results.append(
            {
                "question_id": query.id,
                "question": query.question,
                "chosen_strategy": final_strategy,
                "route_reason": route_reason,
                "predicted_answer": predicted_answer,
                "gold_answer": gold_answer,
                "correct": predicted_answer == gold_answer,
                "samples_used": int(final_result["samples_used"]),
                "num_numeric_mentions": question_features["num_numeric_mentions"],
                "question_length_words": question_features["question_length_words"],
                "question_length_chars": question_features["question_length_chars"],
                "has_multi_step_cue": question_features["has_multi_step_cue"],
                "is_simple": question_features["is_simple"],
                "is_high_complexity": question_features["is_high_complexity"],
                "reasoning_first_output": reasoning_first_output,
                "reasoning_parse_failure": (
                    None
                    if first_pass_features is None
                    else first_pass_features["parse_failure"]
                ),
                "reasoning_malformed_output": (
                    None
                    if first_pass_features is None
                    else first_pass_features["malformed_output"]
                ),
                "reasoning_uncertainty_phrase": (
                    None
                    if first_pass_features is None
                    else first_pass_features["contains_uncertainty_phrase"]
                ),
                "reasoning_too_many_numbers": (
                    None
                    if first_pass_features is None
                    else first_pass_features["too_many_numbers"]
                ),
                "reasoning_unstable_output": (
                    None
                    if first_pass_features is None
                    else first_pass_features["unstable_output"]
                ),
                "direct_greedy_correct": baseline_lookup["direct_greedy"]["correct"],
                "reasoning_greedy_correct": baseline_lookup["reasoning_greedy"]["correct"],
                "direct_plus_revise_correct": baseline_lookup["direct_plus_revise"]["correct"],
            }
        )

    adaptive_rows = [
        {
            "question_id": row["question_id"],
            "strategy": "adaptive_policy_v1",
            "predicted_answer": row["predicted_answer"],
            "gold_answer": row["gold_answer"],
            "correct": row["correct"],
            "samples_used": row["samples_used"],
        }
        for row in per_query_results
    ]

    summary_rows = [
        _summarize_rows("adaptive_policy_v1", adaptive_rows),
        *[
            _summarize_rows(name, rows)
            for name, rows in baseline_rows.items()
        ],
    ]
    summary_by_name = {row["strategy"]: row for row in summary_rows}

    adaptive = summary_by_name["adaptive_policy_v1"]
    direct = summary_by_name["direct_greedy"]
    reasoning = summary_by_name["reasoning_greedy"]

    comparisons = {
        "adaptive_minus_direct_accuracy": round(
            float(adaptive["accuracy"]) - float(direct["accuracy"]),
            4,
        ),
        "adaptive_minus_reasoning_greedy_accuracy": round(
            float(adaptive["accuracy"]) - float(reasoning["accuracy"]),
            4,
        ),
        "adaptive_minus_direct_avg_cost": round(
            float(adaptive["avg_cost"]) - float(direct["avg_cost"]),
            4,
        ),
        "adaptive_minus_reasoning_greedy_avg_cost": round(
            float(adaptive["avg_cost"]) - float(reasoning["avg_cost"]),
            4,
        ),
        "adaptive_minus_oracle_accuracy": (
            None
            if oracle_metrics is None
            else round(
                float(adaptive["accuracy"]) - float(oracle_metrics["oracle_accuracy"]),
                4,
            )
        ),
    }

    route_counts: dict[str, int] = {}
    for row in per_query_results:
        route_counts[row["chosen_strategy"]] = route_counts.get(row["chosen_strategy"], 0) + 1

    return {
        "run_status": "COMPLETED",
        "dataset": dataset_metadata,
        "models": {
            "current": current_model_name,
            "strong": strong_model_label,
            "strong_model_status": strong_status,
        },
        "access_checks": access_checks,
        "policy_config": {
            "simple_max_numeric_mentions": policy_config.simple_max_numeric_mentions,
            "simple_max_word_count": policy_config.simple_max_word_count,
            "high_complexity_numeric_mentions": policy_config.high_complexity_numeric_mentions,
            "high_complexity_word_count": policy_config.high_complexity_word_count,
            "unstable_max_output_numbers": policy_config.unstable_max_output_numbers,
            "allow_reasoning_best_of_3": policy_config.allow_reasoning_best_of_3,
            "allow_strong_direct": policy_config.allow_strong_direct,
        },
        "summary_rows": summary_rows,
        "comparisons": comparisons,
        "route_counts": route_counts,
        "revise_trigger_count": revise_trigger_count,
        "rare_fallback_trigger_count": fallback_trigger_count,
        "oracle_metrics": oracle_metrics,
        "per_query_results": per_query_results,
    }


def write_adaptive_policy_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    summary_json = base / "summary.json"
    summary_json.write_text(
        json.dumps(
            {key: value for key, value in result.items() if key != "per_query_results"},
            indent=2,
        )
    )
    return {
        "summary_json": str(summary_json),
        "summary_csv": _write_csv(result["summary_rows"], base / "summary.csv"),
        "per_query_csv": _write_csv(result["per_query_results"], base / "per_query_results.csv"),
    }


def format_adaptive_policy_summary(
    result: dict[str, Any],
    paths: dict[str, str] | None = None,
) -> str:
    lines = [
        "--- Adaptive Policy V1 ---",
        f"dataset: {result['dataset']['name']}",
        f"queries: {result['dataset']['num_queries']}",
        "summary:",
    ]
    for row in result["summary_rows"]:
        lines.append(
            "  "
            f"{row['strategy']} | accuracy={float(row['accuracy']):.4f} | "
            f"avg_cost={float(row['avg_cost']):.2f}"
        )
    lines.extend(
        [
            "",
            (
                "adaptive_minus_direct_accuracy: "
                f"{result['comparisons']['adaptive_minus_direct_accuracy']}"
            ),
            "adaptive_minus_reasoning_greedy_accuracy: "
            f"{result['comparisons']['adaptive_minus_reasoning_greedy_accuracy']}",
            f"revise_trigger_count: {result['revise_trigger_count']}",
        ]
    )
    if paths is not None:
        lines.extend(
            [
                "",
                f"summary_json: {paths['summary_json']}",
                f"summary_csv: {paths['summary_csv']}",
                f"per_query_csv: {paths['per_query_csv']}",
            ]
        )
    return "\n".join(lines)
