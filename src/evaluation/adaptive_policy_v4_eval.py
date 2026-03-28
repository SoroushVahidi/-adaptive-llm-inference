"""Evaluation helpers for the constraint-aware adaptive policy v4 baseline."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.datasets.gsm8k import Query, load_gsm8k
from src.evaluation.oracle_subset_eval import (
    STRATEGY_DEFINITIONS,
    _build_model,
    _normalize,
    _probe_model_access,
)
from src.policies.adaptive_policy_v3 import (
    AdaptivePolicyV3Config,
)
from src.policies.adaptive_policy_v3 import (
    choose_strategy as choose_strategy_v3,
)
from src.policies.adaptive_policy_v3 import (
    extract_question_features as extract_question_features_v3,
)
from src.policies.adaptive_policy_v4 import (
    AdaptivePolicyV4Config,
    choose_strategy,
    explain_policy_decision,
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


def _json_safe(value: Any) -> Any:
    """Recursively convert small non-JSON-native values into JSON-safe ones."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if hasattr(value, "as_tuple"):  # Decimal
        return str(value)
    return value


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


def _config_v3(config: dict[str, Any]) -> AdaptivePolicyV3Config:
    policy_cfg = config.get("policy_v3_baseline", {})
    return AdaptivePolicyV3Config(
        simple_max_numeric_mentions=int(
            policy_cfg.get("simple_max_numeric_mentions", 2)
        ),
        simple_max_word_count=int(policy_cfg.get("simple_max_word_count", 28)),
        high_complexity_numeric_mentions=int(
            policy_cfg.get("high_complexity_numeric_mentions", 4)
        ),
        high_complexity_word_count=int(
            policy_cfg.get("high_complexity_word_count", 60)
        ),
        many_numbers_threshold=int(policy_cfg.get("many_numbers_threshold", 6)),
        weight_final_answer_missing_or_unclear=int(
            policy_cfg.get("weight_final_answer_missing_or_unclear", 2)
        ),
        weight_parse_failure=int(policy_cfg.get("weight_parse_failure", 2)),
        weight_malformed_output=int(policy_cfg.get("weight_malformed_output", 2)),
        weight_uncertainty_phrase_present=int(
            policy_cfg.get("weight_uncertainty_phrase_present", 1)
        ),
        weight_too_many_intermediate_numbers_without_clear_final=int(
            policy_cfg.get("weight_too_many_intermediate_numbers_without_clear_final", 1)
        ),
        weight_contradiction_like_phrase_present=int(
            policy_cfg.get("weight_contradiction_like_phrase_present", 1)
        ),
        weight_target_mismatch_suspected=int(
            policy_cfg.get("weight_target_mismatch_suspected", 1)
        ),
        weight_unit_mismatch_suspected=int(
            policy_cfg.get("weight_unit_mismatch_suspected", 1)
        ),
        weight_impossible_value_suspected=int(
            policy_cfg.get("weight_impossible_value_suspected", 1)
        ),
        revise_threshold=int(policy_cfg.get("revise_threshold", 2)),
        allow_reasoning_best_of_3=bool(
            policy_cfg.get("allow_reasoning_best_of_3", False)
        ),
        allow_strong_direct=bool(
            policy_cfg.get("allow_strong_direct", False)
        ),
    )


def _config_v4(config: dict[str, Any]) -> AdaptivePolicyV4Config:
    policy_cfg = config.get("policy", {})
    return AdaptivePolicyV4Config(
        simple_max_numeric_mentions=int(
            policy_cfg.get("simple_max_numeric_mentions", 2)
        ),
        simple_max_word_count=int(policy_cfg.get("simple_max_word_count", 28)),
        high_complexity_numeric_mentions=int(
            policy_cfg.get("high_complexity_numeric_mentions", 4)
        ),
        high_complexity_word_count=int(
            policy_cfg.get("high_complexity_word_count", 60)
        ),
        allow_reasoning_best_of_3=bool(
            policy_cfg.get("allow_reasoning_best_of_3", False)
        ),
        allow_strong_direct=bool(policy_cfg.get("allow_strong_direct", False)),
        revise_threshold=int(policy_cfg.get("revise_threshold", 2)),
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


def _run_policy(
    policy_name: str,
    chooser: Any,
    policy_config: Any,
    queries: list[Query],
    current_model: Any,
    strong_model: Any | None,
    question_feature_fn: Any,
    explanation_fn: Any | None = None,
) -> tuple[list[dict[str, Any]], int, int]:
    rows: list[dict[str, Any]] = []
    revise_trigger_count = 0
    fallback_trigger_count = 0

    for query in queries:
        question_features = question_feature_fn(query.question)
        initial_strategy = chooser(
            question_text=query.question,
            features=question_features,
            first_pass_output=None,
            config=policy_config,
        )
        final_strategy = initial_strategy
        reasoning_first_output = None
        policy_explanation = None
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
            final_strategy = chooser(
                question_text=query.question,
                features=question_features,
                first_pass_output=reasoning_first_output,
                config=policy_config,
            )
            if explanation_fn is not None:
                policy_explanation = explanation_fn(
                    question_text=query.question,
                    features=question_features,
                    first_pass_output=reasoning_first_output,
                    config=policy_config,
                )
            if final_strategy == "reasoning_greedy":
                route_reason = "reasoning_greedy_stable"
            elif final_strategy == "direct_plus_revise":
                revise_trigger_count += 1
                route_reason = "constraint_signal_triggered"
            else:
                fallback_trigger_count += 1
                route_reason = "rare_fallback"

        definition = STRATEGY_DEFINITIONS[final_strategy]
        final_model = current_model if definition.model_slot == "current" else strong_model
        if final_model is None:
            raise RuntimeError(
                f"Policy '{policy_name}' selected '{final_strategy}' "
                "without an accessible strong model"
            )
        final_result = definition.runner(final_model, query.question)
        predicted_answer = _normalize(str(final_result["predicted_answer"]))
        gold_answer = _normalize(query.answer)

        row = {
            "question_id": query.id,
            "question": query.question,
            "policy": policy_name,
            "chosen_strategy": final_strategy,
            "route_reason": route_reason,
            "predicted_answer": predicted_answer,
            "gold_answer": gold_answer,
            "correct": predicted_answer == gold_answer,
            "samples_used": int(final_result["samples_used"]),
        }
        if policy_explanation is not None:
            row["policy_explanation"] = json.dumps(
                _json_safe(policy_explanation),
                sort_keys=True,
            )
            question_side = policy_explanation["question_features"]
            signal_side = policy_explanation.get("constraint_violation_state")
            row["num_numeric_mentions"] = question_side["num_numeric_mentions"]
            row["question_length_words"] = question_side["question_length_words"]
            row["has_multi_step_cue"] = question_side["has_multi_step_cue"]
            row["is_simple"] = question_side["is_simple"]
            row["reasoning_first_output"] = reasoning_first_output
            if signal_side is not None:
                row["signal_count"] = signal_side["constraint_signal_count"]
                row["triggered_signals"] = ",".join(
                    signal_side["triggered_constraint_signals"]
                )
                for signal_name in signal_side["triggered_constraint_signals"]:
                    row[signal_name] = True
        rows.append(row)
    return rows, revise_trigger_count, fallback_trigger_count


def _signal_firing_summary(
    policy_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    signal_totals: Counter[str] = Counter()
    signal_on_corrected_queries: Counter[str] = Counter()

    for row in policy_rows:
        signals = [item for item in str(row.get("triggered_signals", "")).split(",") if item]
        for signal in signals:
            signal_totals[signal] += 1
            if bool(row["correct"]) and str(row["chosen_strategy"]) == "direct_plus_revise":
                signal_on_corrected_queries[signal] += 1

    rows: list[dict[str, Any]] = []
    for signal, count in sorted(signal_totals.items(), key=lambda item: (-item[1], item[0])):
        rows.append(
            {
                "signal": signal,
                "fired_count": count,
                "fired_fraction": 0.0 if not policy_rows else count / len(policy_rows),
                "fires_on_corrected_queries": signal_on_corrected_queries.get(signal, 0),
            }
        )
    return rows


def run_adaptive_policy_v4_eval(config: dict[str, Any]) -> dict[str, Any]:
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

    current_model = _build_model(current_model_name, config)
    strong_status: dict[str, Any] | None = None
    strong_model = None
    if strong_model_label is not None:
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
    baseline_summaries = {
        name: _summarize_rows(name, rows)
        for name, rows in baseline_rows.items()
    }

    oracle_metrics = None
    oracle_summary_path = Path("outputs/oracle_subset_eval/summary.json")
    if oracle_summary_path.exists():
        oracle_metrics = json.loads(oracle_summary_path.read_text()).get("global_metrics")

    v3_rows, v3_revise_count, _ = _run_policy(
        policy_name="adaptive_policy_v3",
        chooser=choose_strategy_v3,
        policy_config=_config_v3(config),
        queries=queries,
        current_model=current_model,
        strong_model=strong_model,
        question_feature_fn=extract_question_features_v3,
    )
    v4_rows, v4_revise_count, _ = _run_policy(
        policy_name="adaptive_policy_v4",
        chooser=choose_strategy,
        policy_config=_config_v4(config),
        queries=queries,
        current_model=current_model,
        strong_model=strong_model,
        question_feature_fn=extract_question_features,
        explanation_fn=explain_policy_decision,
    )

    v3_summary = _summarize_rows("adaptive_policy_v3", v3_rows)
    v4_summary = _summarize_rows("adaptive_policy_v4", v4_rows)
    signal_summary_rows = _signal_firing_summary(v4_rows)

    comparisons = {
        "v4_minus_v3_accuracy": round(
            float(v4_summary["accuracy"]) - float(v3_summary["accuracy"]),
            4,
        ),
        "v4_minus_v3_avg_cost": round(
            float(v4_summary["avg_cost"]) - float(v3_summary["avg_cost"]),
            4,
        ),
        "v4_minus_reasoning_greedy_accuracy": round(
            float(v4_summary["accuracy"])
            - float(baseline_summaries["reasoning_greedy"]["accuracy"]),
            4,
        ),
        "v4_minus_reasoning_greedy_avg_cost": round(
            float(v4_summary["avg_cost"])
            - float(baseline_summaries["reasoning_greedy"]["avg_cost"]),
            4,
        ),
        "v4_minus_direct_plus_revise_accuracy": round(
            float(v4_summary["accuracy"])
            - float(baseline_summaries["direct_plus_revise"]["accuracy"]),
            4,
        ),
        "v4_minus_direct_plus_revise_avg_cost": round(
            float(v4_summary["avg_cost"])
            - float(baseline_summaries["direct_plus_revise"]["avg_cost"]),
            4,
        ),
        "v4_minus_oracle_accuracy": (
            None
            if oracle_metrics is None
            else round(
                float(v4_summary["accuracy"]) - float(oracle_metrics["oracle_accuracy"]),
                4,
            )
        ),
    }

    route_counts_v4: dict[str, int] = defaultdict(int)
    for row in v4_rows:
        route_counts_v4[row["chosen_strategy"]] += 1

    return {
        "run_status": "COMPLETED",
        "dataset": dataset_metadata,
        "models": {
            "current": current_model_name,
            "strong": strong_model_label,
            "strong_model_status": strong_status,
        },
        "access_checks": access_checks,
        "policy_v4_config": {
            "simple_max_numeric_mentions": _config_v4(config).simple_max_numeric_mentions,
            "simple_max_word_count": _config_v4(config).simple_max_word_count,
            "high_complexity_numeric_mentions": _config_v4(config).high_complexity_numeric_mentions,
            "high_complexity_word_count": _config_v4(config).high_complexity_word_count,
            "revise_threshold": _config_v4(config).revise_threshold,
        },
        "summary_rows": [
            v4_summary,
            v3_summary,
            baseline_summaries["direct_greedy"],
            baseline_summaries["reasoning_greedy"],
            baseline_summaries["direct_plus_revise"],
        ],
        "comparisons": comparisons,
        "route_counts_v4": dict(route_counts_v4),
        "revise_trigger_count_v3": v3_revise_count,
        "revise_trigger_fraction_v3": 0.0 if not v3_rows else v3_revise_count / len(v3_rows),
        "revise_trigger_count_v4": v4_revise_count,
        "revise_trigger_fraction_v4": 0.0 if not v4_rows else v4_revise_count / len(v4_rows),
        "oracle_metrics": oracle_metrics,
        "signal_firing_summary": signal_summary_rows,
        "per_query_results": v4_rows,
    }


def write_adaptive_policy_v4_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    summary_json = base / "summary.json"
    summary_json.write_text(
        json.dumps(
            {
                key: value
                for key, value in result.items()
                if key not in {"per_query_results", "signal_firing_summary"}
            },
            indent=2,
        )
    )
    return {
        "summary_json": str(summary_json),
        "summary_csv": _write_csv(result["summary_rows"], base / "summary.csv"),
        "per_query_csv": _write_csv(result["per_query_results"], base / "per_query_results.csv"),
        "signal_summary_csv": _write_csv(
            result["signal_firing_summary"],
            base / "signal_firing_summary.csv",
        ),
    }


def format_adaptive_policy_v4_summary(
    result: dict[str, Any],
    paths: dict[str, str] | None = None,
) -> str:
    lines = [
        "--- Adaptive Policy V4 ---",
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
            f"revise_trigger_count_v4: {result['revise_trigger_count_v4']}",
            (
                "revise_trigger_fraction_v4: "
                f"{result['revise_trigger_fraction_v4']:.4f}"
            ),
            f"v4_minus_v3_accuracy: {result['comparisons']['v4_minus_v3_accuracy']}",
            (
                "v4_minus_reasoning_greedy_accuracy: "
                f"{result['comparisons']['v4_minus_reasoning_greedy_accuracy']}"
            ),
        ]
    )
    if paths is not None:
        lines.extend(
            [
                "",
                f"summary_json: {paths['summary_json']}",
                f"summary_csv: {paths['summary_csv']}",
                f"per_query_csv: {paths['per_query_csv']}",
                f"signal_summary_csv: {paths['signal_summary_csv']}",
            ]
        )
    return "\n".join(lines)
