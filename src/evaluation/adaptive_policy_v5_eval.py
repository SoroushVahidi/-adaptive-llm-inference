"""Evaluation helpers for adaptive policy v5 (role-coverage aware)."""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from src.datasets.gsm8k import Query, load_gsm8k
from src.evaluation.oracle_subset_eval import STRATEGY_DEFINITIONS, _build_model, _normalize, _probe_model_access
from src.policies.adaptive_policy_v3 import AdaptivePolicyV3Config
from src.policies.adaptive_policy_v3 import choose_strategy as choose_strategy_v3
from src.policies.adaptive_policy_v3 import extract_question_features as extract_question_features_v3
from src.policies.adaptive_policy_v4 import AdaptivePolicyV4Config
from src.policies.adaptive_policy_v4 import choose_strategy as choose_strategy_v4
from src.policies.adaptive_policy_v4 import extract_question_features_v4
from src.policies.adaptive_policy_v5 import (
    AdaptivePolicyV5Config,
    choose_strategy as choose_strategy_v5,
    explain_policy_decision,
    extract_question_features_v5,
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


def _config_v3(config: dict[str, Any]) -> AdaptivePolicyV3Config:
    return AdaptivePolicyV3Config()


def _config_v4(config: dict[str, Any]) -> AdaptivePolicyV4Config:
    return AdaptivePolicyV4Config()


def _config_v5(config: dict[str, Any]) -> AdaptivePolicyV5Config:
    p = config.get("policy", {})
    return AdaptivePolicyV5Config(
        simple_max_numeric_mentions=int(p.get("simple_max_numeric_mentions", 2)),
        simple_max_word_count=int(p.get("simple_max_word_count", 28)),
        high_complexity_numeric_mentions=int(p.get("high_complexity_numeric_mentions", 4)),
        high_complexity_word_count=int(p.get("high_complexity_word_count", 60)),
        missing_required_number_threshold=int(p.get("missing_required_number_threshold", 1)),
        role_coverage_score_threshold=float(p.get("role_coverage_score_threshold", 0.7)),
        revise_threshold=int(p.get("revise_threshold", 3)),
        allow_reasoning_best_of_3=bool(p.get("allow_reasoning_best_of_3", False)),
        allow_strong_direct=bool(p.get("allow_strong_direct", False)),
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
) -> tuple[list[dict[str, Any]], int]:
    rows: list[dict[str, Any]] = []
    revise_trigger_count = 0

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

        if initial_strategy != "direct_greedy":
            reasoning_result = STRATEGY_DEFINITIONS["reasoning_greedy"].runner(current_model, query.question)
            reasoning_first_output = str(reasoning_result["raw_outputs"][0])
            final_strategy = chooser(
                question_text=query.question,
                features=question_features,
                first_pass_output=reasoning_first_output,
                config=policy_config,
            )
            if final_strategy == "direct_plus_revise":
                revise_trigger_count += 1
            if explanation_fn is not None:
                policy_explanation = explanation_fn(
                    question_text=query.question,
                    features=question_features,
                    first_pass_output=reasoning_first_output,
                    config=policy_config,
                )

        definition = STRATEGY_DEFINITIONS[final_strategy]
        final_model = current_model if definition.model_slot == "current" else strong_model
        if final_model is None:
            raise RuntimeError(f"Policy '{policy_name}' selected '{final_strategy}' without strong model")

        final_result = definition.runner(final_model, query.question)
        predicted_answer = _normalize(str(final_result["predicted_answer"]))
        gold_answer = _normalize(query.answer)

        row = {
            "question_id": query.id,
            "question": query.question,
            "policy": policy_name,
            "chosen_strategy": final_strategy,
            "predicted_answer": predicted_answer,
            "gold_answer": gold_answer,
            "correct": predicted_answer == gold_answer,
            "samples_used": int(final_result["samples_used"]),
            "reasoning_first_output": reasoning_first_output or "",
        }
        if policy_explanation is not None:
            row["policy_explanation"] = json.dumps(policy_explanation, sort_keys=True, default=str)
            role_state = policy_explanation.get("role_coverage_state") or {}
            role_feats = role_state.get("role_coverage_features") or {}
            row["missing_required_number_count"] = role_feats.get("missing_required_number_count", "")
            row["possible_intermediate_stop_suspected"] = role_feats.get(
                "possible_intermediate_stop_suspected", ""
            )
            row["role_coverage_score"] = role_feats.get("role_coverage_score", "")
            row["triggered_signals"] = ",".join(role_feats.get("role_coverage_triggered_signals", []))
        rows.append(row)

    return rows, revise_trigger_count


def _signal_firing_summary(policy_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: Counter[str] = Counter()
    for row in policy_rows:
        for s in [x for x in str(row.get("triggered_signals", "")).split(",") if x]:
            counts[s] += 1
    out: list[dict[str, Any]] = []
    for signal, c in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        out.append(
            {
                "signal": signal,
                "fired_count": c,
                "fired_fraction": 0.0 if not policy_rows else c / len(policy_rows),
            }
        )
    return out


def run_adaptive_policy_v5_eval(config: dict[str, Any]) -> dict[str, Any]:
    queries, dataset_metadata = _load_queries(config)

    current_model_name = str(config.get("models", {}).get("current", "gpt-4o-mini"))
    strong_model_name = config.get("models", {}).get("strong")
    strong_model_label = None if strong_model_name is None else str(strong_model_name)

    access_checks: list[dict[str, Any]] = []
    current_probe = _probe_model_access(current_model_name, config)
    access_checks.append(current_probe)
    if current_probe["status"] != "accessible":
        return {
            "run_status": "BLOCKED",
            "blocker_type": "model_access",
            "blocker_detail": current_probe["error"],
            "dataset": dataset_metadata,
            "models": {"current": current_model_name, "strong": strong_model_label},
            "access_checks": access_checks,
            "summary_rows": [],
            "comparisons": {},
            "signal_firing_summary": [],
            "per_query_results": [],
        }

    current_model = _build_model(current_model_name, config)
    strong_model = None
    if strong_model_label is not None:
        strong_probe = _probe_model_access(strong_model_label, config)
        access_checks.append(strong_probe)
        if strong_probe["status"] == "accessible":
            strong_model = _build_model(strong_model_label, config)

    baseline_names = ["direct_greedy", "reasoning_greedy", "direct_plus_revise"]
    baseline_rows = {n: _run_strategy_rows(n, queries, current_model, strong_model) for n in baseline_names}
    baseline_summaries = {n: _summarize_rows(n, rows) for n, rows in baseline_rows.items()}

    v3_rows, v3_revise = _run_policy(
        "adaptive_policy_v3", choose_strategy_v3, _config_v3(config), queries,
        current_model, strong_model, extract_question_features_v3,
    )
    v4_rows, v4_revise = _run_policy(
        "adaptive_policy_v4", choose_strategy_v4, _config_v4(config), queries,
        current_model, strong_model, extract_question_features_v4,
    )
    v5_rows, v5_revise = _run_policy(
        "adaptive_policy_v5", choose_strategy_v5, _config_v5(config), queries,
        current_model, strong_model, extract_question_features_v5, explanation_fn=explain_policy_decision,
    )

    v3_summary = _summarize_rows("adaptive_policy_v3", v3_rows)
    v4_summary = _summarize_rows("adaptive_policy_v4", v4_rows)
    v5_summary = _summarize_rows("adaptive_policy_v5", v5_rows)

    oracle_metrics = None
    p = Path("outputs/oracle_subset_eval/summary.json")
    if p.exists():
        oracle_metrics = json.loads(p.read_text()).get("global_metrics")

    comparisons = {
        "v5_minus_reasoning_greedy_accuracy": round(v5_summary["accuracy"] - baseline_summaries["reasoning_greedy"]["accuracy"], 4),
        "v5_minus_reasoning_greedy_avg_cost": round(v5_summary["avg_cost"] - baseline_summaries["reasoning_greedy"]["avg_cost"], 4),
        "v5_minus_direct_plus_revise_accuracy": round(v5_summary["accuracy"] - baseline_summaries["direct_plus_revise"]["accuracy"], 4),
        "v5_minus_direct_plus_revise_avg_cost": round(v5_summary["avg_cost"] - baseline_summaries["direct_plus_revise"]["avg_cost"], 4),
        "v5_minus_v4_accuracy": round(v5_summary["accuracy"] - v4_summary["accuracy"], 4),
        "v5_minus_v4_avg_cost": round(v5_summary["avg_cost"] - v4_summary["avg_cost"], 4),
        "v5_minus_oracle_accuracy": None if oracle_metrics is None else round(v5_summary["accuracy"] - float(oracle_metrics["oracle_accuracy"]), 4),
    }

    return {
        "run_status": "COMPLETED",
        "evidence_status": "measured_now",
        "dataset": dataset_metadata,
        "models": {"current": current_model_name, "strong": strong_model_label},
        "access_checks": access_checks,
        "summary_rows": [
            v5_summary,
            v4_summary,
            v3_summary,
            baseline_summaries["reasoning_greedy"],
            baseline_summaries["direct_plus_revise"],
            baseline_summaries["direct_greedy"],
        ],
        "comparisons": comparisons,
        "revise_trigger_count_v5": v5_revise,
        "revise_trigger_fraction_v5": 0.0 if not v5_rows else v5_revise / len(v5_rows),
        "revise_trigger_count_v4": v4_revise,
        "revise_trigger_fraction_v4": 0.0 if not v4_rows else v4_revise / len(v4_rows),
        "revise_trigger_count_v3": v3_revise,
        "revise_trigger_fraction_v3": 0.0 if not v3_rows else v3_revise / len(v3_rows),
        "oracle_metrics": oracle_metrics,
        "signal_firing_summary": _signal_firing_summary(v5_rows),
        "per_query_results": v5_rows,
    }


def write_adaptive_policy_v5_outputs(result: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    summary_path = base / "summary.json"
    summary_path.write_text(json.dumps({k: v for k, v in result.items() if k not in {"per_query_results", "signal_firing_summary"}}, indent=2))

    return {
        "summary_json": str(summary_path),
        "summary_csv": _write_csv(result.get("summary_rows", []), base / "summary.csv"),
        "per_query_csv": _write_csv(result.get("per_query_results", []), base / "per_query_results.csv"),
        "signal_summary_csv": _write_csv(result.get("signal_firing_summary", []), base / "signal_firing_summary.csv"),
    }


def format_adaptive_policy_v5_summary(result: dict[str, Any], paths: dict[str, str] | None = None) -> str:
    if result.get("run_status") == "BLOCKED":
        lines = [
            "--- Adaptive Policy V5 ---",
            "run_status: BLOCKED",
            f"blocker_type: {result.get('blocker_type')}",
            f"blocker_detail: {result.get('blocker_detail')}",
        ]
        if paths:
            lines.append(f"summary_json: {paths['summary_json']}")
        return "\n".join(lines)

    lines = [
        "--- Adaptive Policy V5 ---",
        f"dataset: {result['dataset']['name']}",
        f"queries: {result['dataset']['num_queries']}",
        f"revise_trigger_count_v5: {result['revise_trigger_count_v5']}",
        f"revise_trigger_fraction_v5: {result['revise_trigger_fraction_v5']:.4f}",
    ]
    for row in result["summary_rows"]:
        lines.append(f"{row['strategy']} | accuracy={row['accuracy']:.4f} | avg_cost={row['avg_cost']:.2f}")
    if paths:
        lines.extend(["", f"summary_json: {paths['summary_json']}", f"summary_csv: {paths['summary_csv']}", f"per_query_csv: {paths['per_query_csv']}", f"signal_summary_csv: {paths['signal_summary_csv']}"])
    return "\n".join(lines)
