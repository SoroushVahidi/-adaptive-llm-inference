"""Standalone consistency-check benchmark for math word-problem answers.

Supports side-by-side comparison of:
- old_checker (legacy consistency signals)
- raw_role_checker (legacy + raw role-coverage signals)
- calibrated_role_checker (legacy + calibrated role decision)
- unified_error_checker (legacy + unified error signal)
"""

from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.features.number_role_features import (
    compute_calibrated_role_decision,
    compute_role_coverage_features,
)
from src.features.unified_error_signal import compute_unified_error_signal

NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")
RATIO_RE = re.compile(r"ratio\s+of\s+(\d+)\s*:\s*(\d+)", re.IGNORECASE)
AT_MOST_RE = re.compile(r"at most\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
AT_LEAST_RE = re.compile(r"at least\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
NO_MORE_THAN_RE = re.compile(r"no more than\s+(\d+(?:\.\d+)?)", re.IGNORECASE)
TOTAL_OF_RE = re.compile(r"total of\s+(\d+(?:\.\d+)?)", re.IGNORECASE)

FAILURE_TYPES = {
    "wrong_target_quantity",
    "intermediate_as_final",
    "rate_vs_total",
    "total_vs_remaining",
    "floor_ceiling",
    "hidden_constraint",
    "ratio_constraint",
}

VARIANTS = (
    "old_checker",
    "raw_role_checker",
    "calibrated_role_checker",
    "unified_error_checker",
)


@dataclass(frozen=True)
class CandidateRecord:
    question_id: str
    question: str
    gold_answer: str
    answer: str
    is_correct: bool
    failure_type: str | None


def _extract_first_number(text: str) -> float | None:
    m = NUM_RE.search(str(text))
    if not m:
        return None
    try:
        return float(m.group(0))
    except ValueError:
        return None


def _extract_all_numbers(text: str) -> list[float]:
    out: list[float] = []
    for tok in NUM_RE.findall(str(text)):
        try:
            out.append(float(tok))
        except ValueError:
            continue
    return out


def evaluate_candidate(question: str, candidate_answer: str) -> dict[str, Any]:
    """Evaluate legacy lightweight consistency signals for one pair."""
    q = question.lower()
    signals: list[str] = []

    value = _extract_first_number(candidate_answer)
    nums = _extract_all_numbers(q)

    numeric_target = any(
        cue in q
        for cue in (
            "how many",
            "how much",
            "what is",
            "what are",
            "how long",
            "minimum number",
        )
    )
    if numeric_target and value is None:
        signals.append("numeric_type_mismatch")

    if value is None:
        return {"flagged": bool(signals), "triggered_signals": signals, "parsed_value": None}

    if ("how many" in q or "minimum number" in q or "number of" in q) and value != int(value):
        signals.append("non_integer_count")

    if value < 0 and any(
        c in q for c in ("how many", "students", "books", "boxes", "apples", "tickets")
    ):
        signals.append("negative_impossible")

    asks_remaining = any(c in q for c in ("left", "remaining", "remain"))
    asks_total = any(c in q for c in ("total", "altogether", "in all"))
    if asks_remaining and nums and value >= max(nums):
        signals.append("remaining_conflict")
    if asks_total and nums and value < min(nums):
        signals.append("total_conflict")

    if any(c in q for c in ("per ", "each", "every")) and any(
        c in q for c in ("total", "altogether", "in all")
    ):
        if value in nums:
            signals.append("rate_vs_total_conflict")

    at_most_match = AT_MOST_RE.search(q)
    if at_most_match and any(c in q for c in ("minimum number", "at least", "needed", "required")):
        capacity = float(at_most_match.group(1))
        demand = max(nums) if nums else None
        if demand is not None and capacity > 0:
            required = int((demand + capacity - 1) // capacity)
            if value < required:
                signals.append("floor_ceiling_conflict")

    at_least_match = AT_LEAST_RE.search(q)
    if at_least_match and value < float(at_least_match.group(1)):
        signals.append("below_at_least_constraint")

    no_more_match = NO_MORE_THAN_RE.search(q)
    if no_more_match and value > float(no_more_match.group(1)):
        signals.append("exceeds_no_more_than_constraint")

    ratio_match = RATIO_RE.search(q)
    total_match = TOTAL_OF_RE.search(q)
    if ratio_match and total_match:
        a = float(ratio_match.group(1))
        b = float(ratio_match.group(2))
        total = float(total_match.group(1))
        ratio_unit = a + b
        if ratio_unit > 0 and total % ratio_unit == 0:
            unit = total / ratio_unit
            valid_values = {a * unit, b * unit, total}
            if value not in valid_values:
                signals.append("ratio_constraint_conflict")

    if value in nums and any(
        c in q for c in ("then", "after", "remaining", "left", "twice", "half")
    ):
        signals.append("intermediate_echo_risk")

    return {
        "flagged": bool(signals),
        "triggered_signals": sorted(set(signals)),
        "parsed_value": value,
    }


def evaluate_candidate_with_role_features(question: str, candidate_answer: str) -> dict[str, Any]:
    legacy = evaluate_candidate(question, candidate_answer)
    parsed = None if legacy["parsed_value"] is None else str(legacy["parsed_value"])
    role_feats = compute_role_coverage_features(
        question_text=question,
        reasoning_text=f"Final answer: {candidate_answer}",
        parsed_answer=parsed,
    )
    role_signals = list(role_feats["role_coverage_triggered_signals"])
    all_signals = sorted(set(list(legacy["triggered_signals"]) + role_signals))
    return {
        "flagged": bool(all_signals),
        "triggered_signals": all_signals,
        "parsed_value": legacy["parsed_value"],
        "legacy": legacy,
        "role_features": role_feats,
    }


def evaluate_candidate_with_calibrated_role_features(
    question: str,
    candidate_answer: str,
) -> dict[str, Any]:
    legacy = evaluate_candidate(question, candidate_answer)
    parsed = None if legacy["parsed_value"] is None else str(legacy["parsed_value"])
    calibrated = compute_calibrated_role_decision(
        question_text=question,
        reasoning_text=f"Final answer: {candidate_answer}",
        parsed_answer=parsed,
    )
    role_signals = []
    if calibrated["escalation_recommended"]:
        role_signals.append(f"calibrated_{calibrated['calibrated_decision']}")
    all_signals = sorted(set(list(legacy["triggered_signals"]) + role_signals))
    return {
        "flagged": bool(legacy["flagged"] or calibrated["escalation_recommended"]),
        "triggered_signals": all_signals,
        "parsed_value": legacy["parsed_value"],
        "legacy": legacy,
        "calibrated": calibrated,
    }


def evaluate_candidate_with_unified_error(question: str, candidate_answer: str) -> dict[str, Any]:
    legacy = evaluate_candidate(question, candidate_answer)
    parsed = None if legacy["parsed_value"] is None else str(legacy["parsed_value"])
    unified = compute_unified_error_signal(
        question_text=question,
        reasoning_text=f"Final answer: {candidate_answer}",
        parsed_answer=parsed,
    )

    error = float(unified["unified_error_score"])
    confidence = float(unified["unified_confidence_score"])
    escalate = error >= 0.34 or (error >= 0.25 and confidence < 0.45)

    signals = list(legacy["triggered_signals"])
    if escalate:
        signals.append("unified_error_escalate")

    return {
        "flagged": bool(legacy["flagged"] or escalate),
        "triggered_signals": sorted(set(signals)),
        "parsed_value": legacy["parsed_value"],
        "legacy": legacy,
        "unified": unified,
    }


def load_benchmark(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text())


def flatten_candidates(benchmark_rows: list[dict[str, Any]]) -> list[CandidateRecord]:
    flat: list[CandidateRecord] = []
    for row in benchmark_rows:
        qid = str(row["question_id"])
        question = str(row["question"])
        gold = str(row["gold_answer"])
        for cand in row["candidate_answers"]:
            failure_type = cand.get("failure_type")
            if failure_type is not None and failure_type not in FAILURE_TYPES:
                raise ValueError(f"Unknown failure_type '{failure_type}' for {qid}")
            flat.append(
                CandidateRecord(
                    question_id=qid,
                    question=question,
                    gold_answer=gold,
                    answer=str(cand["answer"]),
                    is_correct=bool(cand["is_correct"]),
                    failure_type=str(failure_type) if failure_type else None,
                )
            )
    return flat


def _compute_metrics(
    flat: list[CandidateRecord],
    flagged_lookup: dict[int, bool],
) -> tuple[float, float, list[dict[str, Any]]]:
    wrong_total = 0
    wrong_flagged = 0
    correct_total = 0
    correct_flagged = 0
    by_type_total: dict[str, int] = defaultdict(int)
    by_type_flagged: dict[str, int] = defaultdict(int)

    for idx, rec in enumerate(flat):
        flagged = bool(flagged_lookup[idx])
        if rec.is_correct:
            correct_total += 1
            if flagged:
                correct_flagged += 1
        else:
            wrong_total += 1
            if flagged:
                wrong_flagged += 1
            if rec.failure_type:
                by_type_total[rec.failure_type] += 1
                if flagged:
                    by_type_flagged[rec.failure_type] += 1

    recall = (wrong_flagged / wrong_total) if wrong_total else 0.0
    fpr = (correct_flagged / correct_total) if correct_total else 0.0

    rows: list[dict[str, Any]] = []
    for ft in sorted(FAILURE_TYPES):
        total = by_type_total.get(ft, 0)
        flagged = by_type_flagged.get(ft, 0)
        rows.append(
            {
                "failure_type": ft,
                "num_wrong_candidates": total,
                "num_flagged": flagged,
                "recall": 0.0 if total == 0 else flagged / total,
            }
        )

    return recall, fpr, rows


def evaluate_benchmark(benchmark_rows: list[dict[str, Any]]) -> dict[str, Any]:
    flat = flatten_candidates(benchmark_rows)

    results: list[dict[str, Any]] = []
    lookups: dict[str, dict[int, bool]] = {name: {} for name in VARIANTS}

    for idx, rec in enumerate(flat):
        old_out = evaluate_candidate(rec.question, rec.answer)
        raw_out = evaluate_candidate_with_role_features(rec.question, rec.answer)
        calibrated_out = evaluate_candidate_with_calibrated_role_features(rec.question, rec.answer)
        unified_out = evaluate_candidate_with_unified_error(rec.question, rec.answer)

        lookups["old_checker"][idx] = bool(old_out["flagged"])
        lookups["raw_role_checker"][idx] = bool(raw_out["flagged"])
        lookups["calibrated_role_checker"][idx] = bool(calibrated_out["flagged"])
        lookups["unified_error_checker"][idx] = bool(unified_out["flagged"])

        calibrated = calibrated_out["calibrated"]
        role_feats = raw_out["role_features"]
        unified = unified_out["unified"]

        results.append(
            {
                "question_id": rec.question_id,
                "question": rec.question,
                "gold_answer": rec.gold_answer,
                "candidate_answer": rec.answer,
                "is_correct": int(rec.is_correct),
                "failure_type": rec.failure_type or "",
                "old_flagged": int(bool(old_out["flagged"])),
                "raw_role_flagged": int(bool(raw_out["flagged"])),
                "calibrated_role_flagged": int(bool(calibrated_out["flagged"])),
                "unified_error_flagged": int(bool(unified_out["flagged"])),
                "raw_role_triggered_signals": "|".join(raw_out["triggered_signals"]),
                "calibrated_role_triggered_signals": "|".join(calibrated_out["triggered_signals"]),
                "unified_triggered_signals": "|".join(unified_out["triggered_signals"]),
                "role_coverage_score": role_feats["role_coverage_score"],
                "missing_required_number_count": role_feats["missing_required_number_count"],
                "calibrated_decision": calibrated["calibrated_decision"],
                "unified_error_score": unified["unified_error_score"],
                "unified_confidence_score": unified["unified_confidence_score"],
            }
        )

    variant_metrics: dict[str, dict[str, Any]] = {}
    per_variant_by_type: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in VARIANTS:
        recall, fpr, by_type_rows = _compute_metrics(flat, lookups[variant])
        variant_metrics[variant] = {
            "wrong_recall": recall,
            "false_positive_rate_on_correct": fpr,
            "recall_minus_fpr": recall - fpr,
            "balanced_utility": (recall + (1 - fpr)) / 2,
            "num_flagged": int(sum(1 for v in lookups[variant].values() if v)),
        }
        per_variant_by_type[variant] = {row["failure_type"]: row for row in by_type_rows}

    by_type_rows: list[dict[str, Any]] = []
    for ft in sorted(FAILURE_TYPES):
        row: dict[str, Any] = {
            "failure_type": ft,
            "num_wrong_candidates": per_variant_by_type["old_checker"][ft]["num_wrong_candidates"],
        }
        for variant in VARIANTS:
            row[f"{variant}_recall"] = per_variant_by_type[variant][ft]["recall"]
        by_type_rows.append(row)

    return {
        "num_questions": len(benchmark_rows),
        "num_candidates": len(flat),
        "num_wrong_candidates": sum(1 for r in flat if not r.is_correct),
        "num_correct_candidates": sum(1 for r in flat if r.is_correct),
        "variant_metrics": variant_metrics,
        "recall_by_failure_type": by_type_rows,
        "evidence_status": "measured_now",
        "claim_status": "exploratory_only",
        "per_candidate_results": results,
    }


def save_outputs(evaluation: dict[str, Any], output_dir: str | Path) -> dict[str, str]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "summary.json"
    summary = {k: v for k, v in evaluation.items() if k != "per_candidate_results"}
    summary_path.write_text(json.dumps(summary, indent=2))

    unified_summary_path = out / "unified_summary.json"
    unified_summary = {
        "unified_error_checker": evaluation["variant_metrics"]["unified_error_checker"],
        "evidence_status": evaluation.get("evidence_status", "measured_now"),
        "claim_status": evaluation.get("claim_status", "exploratory_only"),
    }
    unified_summary_path.write_text(json.dumps(unified_summary, indent=2))

    per_candidate_path = out / "per_candidate_results.csv"
    candidate_rows = evaluation["per_candidate_results"]
    with per_candidate_path.open("w", newline="") as fh:
        if candidate_rows:
            writer = csv.DictWriter(fh, fieldnames=list(candidate_rows[0].keys()))
            writer.writeheader()
            writer.writerows(candidate_rows)
        else:
            writer = csv.writer(fh)
            writer.writerow(["question_id", "candidate_answer", "is_correct"])

    failure_type_path = out / "failure_type_summary.csv"
    with failure_type_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "failure_type",
                "num_wrong_candidates",
                "old_checker_recall",
                "raw_role_checker_recall",
                "calibrated_role_checker_recall",
                "unified_error_checker_recall",
            ],
        )
        writer.writeheader()
        writer.writerows(evaluation["recall_by_failure_type"])

    return {
        "summary_json": str(summary_path),
        "unified_summary_json": str(unified_summary_path),
        "per_candidate_results_csv": str(per_candidate_path),
        "failure_type_summary_csv": str(failure_type_path),
    }
