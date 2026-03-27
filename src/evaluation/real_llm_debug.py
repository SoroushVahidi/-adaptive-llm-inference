"""Debug helpers for suspiciously identical real-LM GSM8K results.

This module is intentionally tiny and diagnostic-only. It is meant to answer:
1) whether repeated samples from the OpenAI backend are actually diverse,
2) whether distinct raw outputs collapse to the same parsed numeric answer, and
3) whether the current selective-escalation thresholds ever create true
   post-gating escalation candidates on a tiny real subset.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.datasets.gsm8k import Query, load_gsm8k
from src.methods.selective_escalation import (
    SelectiveEscalationConfig,
    compute_escalation_signals,
    parse_numeric_details,
    score_escalation,
)
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
        max_tokens=int(model_cfg.get("max_tokens", 128)),
        timeout_seconds=float(model_cfg.get("timeout_seconds", 60.0)),
    )


def _question_prompt(question: str) -> str:
    return question


def _sample_prompt_style(
    model: OpenAILLMModel,
    queries: list[Query],
    prompt_style: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for query in queries:
        raw_samples = model.generate_n(_question_prompt(query.question), 3)
        parsed = [parse_numeric_details(sample) for sample in raw_samples]
        parsed_answers = [item["parsed_answer"] for item in parsed]
        rows.append(
            {
                "question_id": query.id,
                "prompt_style": prompt_style,
                "raw_samples": raw_samples,
                "parsed_answers": parsed_answers,
                "raw_outputs_identical": len(set(raw_samples)) == 1,
                "parsed_answers_identical": len(set(parsed_answers)) == 1,
                "extraction_failures": [bool(item["parse_failure"]) for item in parsed],
            }
        )
    return rows


def _selective_debug(
    direct_rows: list[dict[str, Any]],
    method_cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    strict_config = SelectiveEscalationConfig(
        total_budget=int(method_cfg.get("total_budget", 0)),
        escalation_target_k=1 + int(method_cfg.get("extra_samples_per_escalated_query", 2)),
        use_second_sample_for_disagreement=bool(
            method_cfg.get("use_second_sample_for_disagreement", True)
        ),
        weight_parse_failure=float(method_cfg.get("parse_failure_weight", 2.0)),
        weight_disagreement_2sample=float(method_cfg.get("disagreement_weight", 1.5)),
        weight_malformed_output=float(method_cfg.get("malformed_output_weight", 1.0)),
        weight_low_confidence_format=float(method_cfg.get("missing_numeric_weight", 1.0)),
        malformed_length_threshold=int(method_cfg.get("malformed_length_threshold", 2)),
        min_score_to_escalate=float(method_cfg.get("min_score_to_escalate", 1.5)),
        enable_post_ranking_escalation=bool(
            method_cfg.get("enable_post_ranking_escalation", True)
        ),
    )
    loose_config = SelectiveEscalationConfig(
        total_budget=strict_config.total_budget,
        escalation_target_k=strict_config.escalation_target_k,
        use_second_sample_for_disagreement=strict_config.use_second_sample_for_disagreement,
        weight_parse_failure=strict_config.weight_parse_failure,
        weight_disagreement_2sample=strict_config.weight_disagreement_2sample,
        weight_malformed_output=strict_config.weight_malformed_output,
        weight_low_confidence_format=strict_config.weight_low_confidence_format,
        malformed_length_threshold=strict_config.malformed_length_threshold,
        min_score_to_escalate=0.5,
        enable_post_ranking_escalation=True,
    )

    rows: list[dict[str, Any]] = []
    for row in direct_rows:
        first_output = row["raw_samples"][0]
        second_output = row["raw_samples"][1]
        signals = compute_escalation_signals(first_output=first_output, second_output=second_output)
        strict_score = score_escalation(signals, strict_config)
        rows.append(
            {
                "question_id": row["question_id"],
                "first_output": first_output,
                "second_output": second_output,
                "first_parsed_answer": row["parsed_answers"][0],
                "second_parsed_answer": row["parsed_answers"][1],
                "parse_failure": bool(signals["parse_failure"]),
                "disagreement_2sample": bool(signals["disagreement_2sample"]),
                "malformed_output": bool(signals["malformed_output"]),
                "low_confidence_format": bool(signals["low_confidence_format"]),
                "strict_escalation_score": strict_score,
                "strict_threshold_would_escalate": (
                    strict_score >= strict_config.min_score_to_escalate
                ),
                "looser_threshold_would_escalate": (
                    strict_score >= loose_config.min_score_to_escalate
                ),
            }
        )
    return rows


def run_real_llm_debug(config: dict[str, Any]) -> dict[str, Any]:
    """Run a tiny sampling/parsing/escalation debug pass on real GSM8K queries."""
    dataset_cfg = config.get("dataset", {})
    queries = load_gsm8k(
        split=str(dataset_cfg.get("split", "test")),
        max_samples=dataset_cfg.get("max_samples", 5),
    )
    if len(queries) > 5:
        raise ValueError("This debug harness is limited to at most 5 queries.")

    direct_model = _build_model(config, DIRECT_PROMPT)
    reasoning_model = _build_model(config, REASONING_PROMPT)

    direct_rows = _sample_prompt_style(direct_model, queries, "direct")
    reasoning_rows = _sample_prompt_style(reasoning_model, queries, "reasoning")
    selective_rows = _selective_debug(
        direct_rows=direct_rows,
        method_cfg=config.get("selective_method", {}),
    )

    all_rows = direct_rows + reasoning_rows
    total_query_prompt_pairs = len(all_rows)
    raw_identical_count = sum(int(bool(row["raw_outputs_identical"])) for row in all_rows)
    parsed_identical_count = sum(int(bool(row["parsed_answers_identical"])) for row in all_rows)
    parsed_collapse_count = sum(
        1
        for row in all_rows
        if not bool(row["raw_outputs_identical"]) and bool(row["parsed_answers_identical"])
    )
    strict_escalation_candidates = sum(
        int(bool(row["strict_threshold_would_escalate"])) for row in selective_rows
    )
    loose_escalation_candidates = sum(
        int(bool(row["looser_threshold_would_escalate"])) for row in selective_rows
    )

    summary = {
        "model_name": direct_model.model_name,
        "greedy_temperature": direct_model.greedy_temperature,
        "sample_temperature": direct_model.sample_temperature,
        "multiple_samples_requested_in_single_call": True,
        "n_samples_per_debug_call": 3,
        "total_query_prompt_pairs": total_query_prompt_pairs,
        "raw_outputs_identical_count": raw_identical_count,
        "parsed_answers_identical_count": parsed_identical_count,
        "parsed_collapse_count": parsed_collapse_count,
        "strict_escalation_candidate_count": strict_escalation_candidates,
        "looser_threshold_candidate_count": loose_escalation_candidates,
    }

    return {
        "raw_samples": all_rows,
        "parsed_samples": [
            {
                "question_id": row["question_id"],
                "prompt_style": row["prompt_style"],
                "parsed_answers": row["parsed_answers"],
                "extraction_failures": row["extraction_failures"],
                "raw_outputs_identical": row["raw_outputs_identical"],
                "parsed_answers_identical": row["parsed_answers_identical"],
            }
            for row in all_rows
        ],
        "selective_debug": selective_rows,
        "summary": summary,
    }


def write_real_llm_debug_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    base_dir = Path(output_dir)
    raw_path = _write_json({"raw_samples": result["raw_samples"]}, base_dir / "raw_samples.json")
    parsed_path = _write_json(
        {"parsed_samples": result["parsed_samples"]},
        base_dir / "parsed_samples.json",
    )
    summary_path = _write_json(
        {
            "summary": result["summary"],
            "selective_debug": result["selective_debug"],
        },
        base_dir / "sampling_debug_summary.json",
    )
    csv_path = _write_csv(result["selective_debug"], base_dir / "selective_debug.csv")
    return {
        "raw_samples_json": str(raw_path),
        "parsed_samples_json": str(parsed_path),
        "summary_json": str(summary_path),
        "selective_debug_csv": str(csv_path),
    }


def format_real_llm_debug_summary(result: dict[str, Any], paths: dict[str, str]) -> str:
    summary = result["summary"]
    lines = [
        "--- Real LLM Sampling Debug Summary ---",
        f"model:                     {summary['model_name']}",
        f"greedy_temperature:        {summary['greedy_temperature']}",
        f"sample_temperature:        {summary['sample_temperature']}",
        f"n_samples/debug_call:      {summary['n_samples_per_debug_call']}",
        f"query_prompt_pairs:        {summary['total_query_prompt_pairs']}",
        f"raw_identical_count:       {summary['raw_outputs_identical_count']}",
        f"parsed_identical_count:    {summary['parsed_answers_identical_count']}",
        f"parsed_collapse_count:     {summary['parsed_collapse_count']}",
        f"strict_escalation_candidates: {summary['strict_escalation_candidate_count']}",
        f"looser_escalation_candidates: {summary['looser_threshold_candidate_count']}",
        f"raw_samples_json:          {paths['raw_samples_json']}",
        f"parsed_samples_json:       {paths['parsed_samples_json']}",
        f"summary_json:              {paths['summary_json']}",
    ]
    return "\n".join(lines)
