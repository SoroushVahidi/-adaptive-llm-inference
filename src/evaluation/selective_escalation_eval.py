"""Evaluation helpers for the selective compute escalation prototype.

This cleanup pass is specifically meant to identify where current gains come
from:
1) normalization / parsing cleanup,
2) cheap two-sample gating,
3) or true post-ranking escalation beyond the gating stage.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.datasets.gsm8k import Query, load_gsm8k
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


def _summarize_method(
    method: str,
    per_query_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    total_queries = len(per_query_rows)
    total_samples = sum(int(row["samples_used"]) for row in per_query_rows)
    correct = sum(1 for row in per_query_rows if bool(row["final_correct"]))
    queries_with_more_than_1 = sum(
        1 for row in per_query_rows if int(row["samples_used"]) > 1
    )
    queries_escalated_beyond_gating = sum(
        1 for row in per_query_rows if bool(row["escalated_beyond_gating"])
    )
    return {
        "method": method,
        "accuracy": 0.0 if total_queries == 0 else correct / total_queries,
        "total_samples_used": total_samples,
        "average_samples_per_query": (
            0.0 if total_queries == 0 else total_samples / total_queries
        ),
        "total_queries": total_queries,
        "queries_with_more_than_1_sample": queries_with_more_than_1,
        "fraction_with_more_than_1_sample": (
            0.0 if total_queries == 0 else queries_with_more_than_1 / total_queries
        ),
        "queries_escalated_beyond_gating_stage": queries_escalated_beyond_gating,
        "fraction_escalated_beyond_gating_stage": (
            0.0
            if total_queries == 0
            else queries_escalated_beyond_gating / total_queries
        ),
    }


def _normalize_only_rows(
    queries: list[Query],
    selective_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query, row in zip(queries, selective_rows):
        rows.append(
            {
                "question_id": query.id,
                "gold_answer": query.answer,
                "first_pass_raw_output": row["first_pass_raw_output"],
                "first_pass_answer": row["first_pass_answer"],
                "final_answer": row["first_pass_answer"],
                "final_correct": bool(row["first_pass_correct"]),
                "samples_used": 1,
                "used_second_sample_for_gating": False,
                "escalated_beyond_gating": False,
                "normalization_changed_correctness": bool(
                    row["normalization_changed_correctness"]
                ),
                "signals_fired": [],
            }
        )
    return rows


def _two_sample_gate_only_rows(
    queries: list[Query],
    selective_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query, row in zip(queries, selective_rows):
        final_answer = row["gating_answer"]
        rows.append(
            {
                "question_id": query.id,
                "gold_answer": query.answer,
                "first_pass_raw_output": row["first_pass_raw_output"],
                "first_pass_answer": row["first_pass_answer"],
                "final_answer": final_answer,
                "final_correct": bool(final_answer == query.answer),
                "samples_used": int(row["gating_samples_used"]),
                "used_second_sample_for_gating": bool(row["used_second_sample_for_gating"]),
                "escalated_beyond_gating": False,
                "normalization_changed_correctness": bool(
                    row["normalization_changed_correctness"]
                ),
                "signals_fired": list(row["signals_fired"]),
            }
        )
    return rows


def _always_best_of_3_rows(
    queries: list[Query],
    selective_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query, row in zip(queries, selective_rows):
        final_answer = row["best_of_3_answer"]
        rows.append(
            {
                "question_id": query.id,
                "gold_answer": query.answer,
                "first_pass_raw_output": row["first_pass_raw_output"],
                "first_pass_answer": row["first_pass_answer"],
                "final_answer": final_answer,
                "final_correct": bool(final_answer == query.answer),
                "samples_used": 3,
                "used_second_sample_for_gating": True,
                "escalated_beyond_gating": True,
                "normalization_changed_correctness": bool(
                    row["normalization_changed_correctness"]
                ),
                "signals_fired": list(row["signals_fired"]),
            }
        )
    return rows


def _run_selective(
    queries: list[Query],
    model: OpenAILLMModel,
    config: SelectiveEscalationConfig,
) -> dict[str, list[dict[str, Any]]]:
    run = run_selective_escalation(
        model=model,
        questions=[_format_prompt(query.question) for query in queries],
        total_budget=config.total_budget,
        config=config,
    )
    diagnostics = run["diagnostics"]
    selective_rows: list[dict[str, Any]] = []
    for query, item in zip(queries, diagnostics):
        final_correct = item["final_answer"] == query.answer
        first_correct = item["first_pass_answer"] == query.answer
        signals_fired = [
            signal_name
            for signal_name, fired in item["signals"].items()
            if bool(fired)
        ]
        selective_rows.append(
            {
                "question_id": query.id,
                "gold_answer": query.answer,
                "first_pass_raw_output": item["first_pass_raw_output"],
                "first_pass_answer": item["first_pass_answer"],
                "first_pass_correct": first_correct,
                "escalation_score": float(item["escalation_score"]),
                "used_second_sample_for_gating": bool(item["used_second_sample_for_gating"]),
                "escalated_beyond_gating": bool(item["escalated_beyond_gating"]),
                "final_answer": item["final_answer"],
                "final_correct": final_correct,
                "samples_used": int(item["samples_used"]),
                "signals_fired": signals_fired,
                "normalization_changed_correctness": bool(
                    item["normalization_changed_correctness"]
                ),
                "gating_answer": item["gating_answer"],
                "gating_samples_used": int(item["gating_samples_used"]),
                "best_of_3_answer": item["best_of_3_answer"],
            }
        )
    return {
        "greedy": _normalize_only_rows(queries, selective_rows),
        "normalization_only": _normalize_only_rows(queries, selective_rows),
        "two_sample_gate_only": _two_sample_gate_only_rows(queries, selective_rows),
        "always_best_of_3": _always_best_of_3_rows(queries, selective_rows),
        "selective_escalation_v1": selective_rows,
    }


def _per_query_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(row["question_id"]): row for row in rows}


def _pairwise_with_attribution(
    baseline_rows: list[dict[str, Any]],
    target_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    baseline_map = _per_query_map(baseline_rows)
    target_map = _per_query_map(target_rows)
    improved = 0
    worsened = 0
    attribution = {
        "normalization_only": 0,
        "gating_stage_only": 0,
        "true_post_ranking_escalation": 0,
        "unattributed": 0,
    }
    for query_id, row in target_map.items():
        target_correct = bool(row["final_correct"])
        baseline_correct = bool(baseline_map[query_id]["final_correct"])
        if target_correct == baseline_correct:
            continue
        if target_correct:
            improved += 1
        else:
            worsened += 1

        if bool(row.get("normalization_changed_correctness")) and int(row["samples_used"]) == 1:
            attribution["normalization_only"] += 1
        elif bool(row.get("used_second_sample_for_gating")) and not bool(
            row.get("escalated_beyond_gating")
        ):
            attribution["gating_stage_only"] += 1
        elif bool(row.get("escalated_beyond_gating")):
            attribution["true_post_ranking_escalation"] += 1
        else:
            attribution["unattributed"] += 1
    return {
        "queries_improved": improved,
        "queries_worsened": worsened,
        "attribution": attribution,
    }


def run_selective_escalation_eval(config: dict[str, Any]) -> dict[str, Any]:
    dataset_cfg = config.get("dataset", {})
    queries = load_gsm8k(
        split=str(dataset_cfg.get("split", "test")),
        max_samples=dataset_cfg.get("max_samples"),
    )

    model = _build_model(config)
    method_cfg = config.get("selective_method", {})
    method_rows = _run_selective(
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
    method_summaries = [
        _summarize_method(name, rows) for name, rows in method_rows.items()
    ]
    order = {
        "greedy": 0,
        "normalization_only": 1,
        "two_sample_gate_only": 2,
        "always_best_of_3": 3,
        "selective_escalation_v1": 4,
    }
    method_summaries = sorted(
        method_summaries,
        key=lambda item: order.get(str(item["method"]), 999),
    )

    per_query_rows: list[dict[str, Any]] = []
    normalization_map = _per_query_map(method_rows["normalization_only"])
    gating_map = _per_query_map(method_rows["two_sample_gate_only"])
    best_map = _per_query_map(method_rows["always_best_of_3"])
    selective_map = _per_query_map(method_rows["selective_escalation_v1"])
    for query in queries:
        query_id = query.id
        normalization_row = normalization_map[query_id]
        gating_row = gating_map[query_id]
        best_row = best_map[query_id]
        selective_row = selective_map[query_id]
        per_query_rows.append(
            {
                "question_id": query_id,
                "gold_answer": query.answer,
                "normalization_only_answer": normalization_row["final_answer"],
                "normalization_only_correct": bool(normalization_row["final_correct"]),
                "greedy_answer": normalization_row["first_pass_raw_output"],
                "greedy_correct": bool(normalization_row["first_pass_answer"] == query.answer),
                "two_sample_gate_only_answer": gating_row["final_answer"],
                "two_sample_gate_only_correct": bool(gating_row["final_correct"]),
                "always_best_of_3_answer": best_row["final_answer"],
                "always_best_of_3_correct": bool(best_row["final_correct"]),
                "first_pass_answer": selective_row["first_pass_answer"],
                "first_pass_correct": bool(selective_row["first_pass_correct"]),
                "escalation_score": float(selective_row["escalation_score"]),
                "used_second_sample_for_gating": bool(
                    selective_row["used_second_sample_for_gating"]
                ),
                "escalated_beyond_gating": bool(
                    selective_row["escalated_beyond_gating"]
                ),
                "final_answer": selective_row["final_answer"],
                "final_correct": bool(selective_row["final_correct"]),
                "samples_used": int(selective_row["samples_used"]),
                "normalization_changed_correctness": bool(
                    selective_row["normalization_changed_correctness"]
                ),
                "signals_fired": ",".join(str(signal) for signal in selective_row["signals_fired"]),
            }
        )

    pairwise = {
        "normalization_only_vs_greedy": _pairwise_with_attribution(
            method_rows["greedy"],
            method_rows["normalization_only"],
        ),
        "two_sample_gate_only_vs_normalization_only": _pairwise_with_attribution(
            method_rows["normalization_only"],
            method_rows["two_sample_gate_only"],
        ),
        "selective_vs_greedy": _pairwise_with_attribution(
            method_rows["greedy"],
            method_rows["selective_escalation_v1"],
        ),
        "selective_vs_always_best_of_3": _pairwise_with_attribution(
            method_rows["always_best_of_3"],
            method_rows["selective_escalation_v1"],
        ),
    }

    attribution_summary = {
        "normalization_only": pairwise["normalization_only_vs_greedy"]["attribution"],
        "two_sample_gate_only": pairwise["two_sample_gate_only_vs_normalization_only"][
            "attribution"
        ],
        "selective_vs_greedy": pairwise["selective_vs_greedy"]["attribution"],
        "selective_vs_always_best_of_3": pairwise["selective_vs_always_best_of_3"][
            "attribution"
        ],
    }

    return {
        "provider": "openai",
        "model_name": model.model_name,
        "total_queries": len(queries),
        "method_summaries": method_summaries,
        "pairwise_comparisons": pairwise,
        "attribution_summary": attribution_summary,
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
            "  normalization_only vs greedy: "
            f"improved={pairwise['normalization_only_vs_greedy']['queries_improved']}, "
            f"worsened={pairwise['normalization_only_vs_greedy']['queries_worsened']}",
            "  two_sample_gate_only vs normalization_only: "
            f"improved={pairwise['two_sample_gate_only_vs_normalization_only']['queries_improved']}, "
            f"worsened={pairwise['two_sample_gate_only_vs_normalization_only']['queries_worsened']}",
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
