"""Build a real query-level GSM8K routing dataset from live strategy outputs."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

from src.datasets.gsm8k import load_gsm8k
from src.evaluation.oracle_subset_eval import run_reasoning_greedy
from src.evaluation.strategy_expansion_eval import run_direct_plus_revise
from src.features.calibration_features import extract_calibration_features
from src.features.constraint_violation_features import extract_constraint_violation_features
from src.features.number_role_features import (
    compute_calibrated_role_decision,
    compute_role_coverage_features,
)
from src.features.precompute_features import extract_query_features
from src.features.selective_prediction_features import extract_selective_prediction_features
from src.features.self_verification_features import extract_self_verification_features
from src.features.step_verification_features import extract_step_verification_features
from src.features.target_quantity_features import extract_target_quantity_features
from src.features.unified_error_signal import compute_unified_error_signal
from src.models.openai_llm import OpenAILLMModel
from src.models.revise_helpful_classifier import detect_sklearn_support


@dataclass(frozen=True)
class BuildConfig:
    gsm8k_data_file: str | Path = "data/gsm8k_uploaded_normalized.jsonl"
    subset_size: int = 20
    output_dir: str | Path = "outputs/real_routing_dataset"
    output_dataset_csv: str | Path = "data/real_gsm8k_routing_dataset.csv"
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 256
    timeout: int = 60


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


def _compute_features(question: str, reasoning_text: str, predicted_answer: str) -> dict[str, Any]:
    parsed_answer = predicted_answer.strip()
    query_feats = extract_query_features(question)
    target_feats = extract_target_quantity_features(question)
    constraint_feats = extract_constraint_violation_features(
        question_text=question,
        reasoning_output=reasoning_text,
        predicted_answer=parsed_answer,
    )
    role_feats = compute_role_coverage_features(
        question_text=question,
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    calibrated = compute_calibrated_role_decision(
        question_text=question,
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    self_verify = extract_self_verification_features(
        question_text=question,
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    selective = extract_selective_prediction_features(
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    calibration = extract_calibration_features(
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    step = extract_step_verification_features(
        question_text=question,
        reasoning_text=reasoning_text,
    )
    unified = compute_unified_error_signal(
        question_text=question,
        reasoning_text=reasoning_text,
        parsed_answer=parsed_answer,
    )
    calibration_bin = calibration["calibration_bin"]
    calibrated_decision = calibrated["calibrated_decision"]
    return {
        **query_feats,
        **target_feats,
        **constraint_feats,
        "role_coverage_score": role_feats["role_coverage_score"],
        "missing_required_number_count": role_feats["missing_required_number_count"],
        "missing_strong_required_number_count": role_feats["missing_strong_required_number_count"],
        "possible_intermediate_stop_suspected": role_feats["possible_intermediate_stop_suspected"],
        "required_subtractive_number_missing": role_feats["required_subtractive_number_missing"],
        "required_rate_number_missing": role_feats["required_rate_number_missing"],
        "required_capacity_number_missing": role_feats["required_capacity_number_missing"],
        "role_warning_score": calibrated["role_warning_score"],
        "role_strong_error_score": calibrated["role_strong_error_score"],
        "calibrated_no_escalation": int(calibrated_decision == "no_escalation"),
        "calibrated_maybe_escalate": int(calibrated_decision == "maybe_escalate"),
        "calibrated_strong_escalation_candidate": int(
            calibrated_decision == "strong_escalation_candidate"
        ),
        **self_verify,
        **selective,
        "predicted_answer_format_confidence": calibration["predicted_answer_format_confidence"],
        "calibration_bin_high": int(calibration_bin == "high_confidence"),
        "calibration_bin_medium": int(calibration_bin == "medium_confidence"),
        "calibration_bin_low": int(calibration_bin == "low_confidence"),
        **step,
        "unified_error_score": unified["unified_error_score"],
        "unified_confidence_score": unified["unified_confidence_score"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _environment_status(config: BuildConfig) -> dict[str, Any]:
    dataset_path = Path(config.gsm8k_data_file)
    api_key_present = bool(os.getenv("OPENAI_API_KEY"))
    sklearn_support = detect_sklearn_support()
    return {
        "openai_api_key_present": api_key_present,
        "model_access_working": False,
        "sklearn_available": sklearn_support.available,
        "sklearn_reason": sklearn_support.reason,
        "gsm8k_file_readable": dataset_path.exists() and dataset_path.is_file(),
    }


def build_real_routing_dataset(config: BuildConfig) -> dict[str, Any]:
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "gsm8k_subset_run_summary.json"
    per_query_csv = out_dir / "gsm8k_per_query_outputs.csv"

    env = _environment_status(config)
    blockers: list[str] = []
    if not env["gsm8k_file_readable"]:
        blockers.append(f"Missing GSM8K normalized file: {config.gsm8k_data_file}")
    if not env["openai_api_key_present"]:
        blockers.append("OPENAI_API_KEY is not set; real inference blocked.")

    if blockers:
        summary = {
            "run_status": "BLOCKED",
            "evidence_status": "blocked",
            "subset_size_requested": config.subset_size,
            "subset_size_executed": 0,
            "blockers": blockers,
            "environment": env,
            "output_dataset_csv": str(config.output_dataset_csv),
            "per_query_csv": str(per_query_csv),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        return {
            "summary": summary,
            "summary_path": str(summary_path),
            "per_query_csv": str(per_query_csv),
            "dataset_csv": str(config.output_dataset_csv),
        }

    model = OpenAILLMModel(
        model_name=config.model_name,
        max_tokens=config.max_tokens,
        timeout=config.timeout,
    )

    try:
        _ = model.generate("Return just the number 4.")
        env["model_access_working"] = True
    except Exception as exc:  # pragma: no cover - network-dependent
        blockers.append(f"Model access check failed: {exc}")

    if blockers:
        summary = {
            "run_status": "BLOCKED",
            "evidence_status": "blocked",
            "subset_size_requested": config.subset_size,
            "subset_size_executed": 0,
            "blockers": blockers,
            "environment": env,
            "output_dataset_csv": str(config.output_dataset_csv),
            "per_query_csv": str(per_query_csv),
        }
        summary_path.write_text(json.dumps(summary, indent=2))
        return {
            "summary": summary,
            "summary_path": str(summary_path),
            "per_query_csv": str(per_query_csv),
            "dataset_csv": str(config.output_dataset_csv),
        }

    queries = load_gsm8k(data_file=config.gsm8k_data_file, max_samples=config.subset_size)
    rows: list[dict[str, Any]] = []

    for query in queries:
        reasoning_result = run_reasoning_greedy(model, query.question)
        revise_result = run_direct_plus_revise(model, query.question)

        gold = _normalize(query.answer)
        reasoning_answer = _normalize(str(reasoning_result["predicted_answer"]))
        revise_answer = _normalize(str(revise_result["predicted_answer"]))
        reasoning_correct = int(reasoning_answer == gold)
        revise_correct = int(revise_answer == gold)
        revise_helpful = int(reasoning_correct == 0 and revise_correct == 1)

        reasoning_text = "\n".join(str(x) for x in reasoning_result.get("raw_outputs", []))
        if not reasoning_text.strip():
            reasoning_text = f"Final answer: {reasoning_answer}"

        rows.append(
            {
                "question_id": query.id,
                "question": query.question,
                "gold_answer": gold,
                "reasoning_answer": reasoning_answer,
                "revise_answer": revise_answer,
                "reasoning_correct": reasoning_correct,
                "revise_correct": revise_correct,
                "revise_helpful": revise_helpful,
                "reasoning_cost": 1,
                "revise_cost": 2,
                **_compute_features(query.question, reasoning_text, reasoning_answer),
            }
        )

    fieldnames = list(rows[0].keys()) if rows else []
    if rows:
        _write_csv(per_query_csv, rows, fieldnames)
        _write_csv(Path(config.output_dataset_csv), rows, fieldnames)

    positives = sum(int(r["revise_helpful"]) for r in rows)
    summary = {
        "run_status": "OK",
        "evidence_status": "measured_now",
        "subset_size_requested": config.subset_size,
        "subset_size_executed": len(rows),
        "revise_helpful_positives": positives,
        "revise_helpful_rate": (positives / len(rows)) if rows else 0.0,
        "strongest_cheap_baseline": "reasoning_greedy",
        "strongest_corrective_baseline": "direct_plus_revise",
        "environment": env,
        "output_dataset_csv": str(config.output_dataset_csv),
        "per_query_csv": str(per_query_csv),
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "per_query_csv": str(per_query_csv),
        "dataset_csv": str(config.output_dataset_csv),
    }
