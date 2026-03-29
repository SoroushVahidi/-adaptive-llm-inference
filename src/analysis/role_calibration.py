"""Role-signal calibration analysis helpers."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

from src.analysis.consistency_benchmark import evaluate_benchmark, load_benchmark


def _pattern_bucket(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ("remaining", "left", "remain")):
        return "remaining_target"
    if any(k in q for k in ("minimum number", "at most", "capacity", "holds")):
        return "capacity_ceiling"
    if any(k in q for k in ("per", "each", "every", "rate")):
        return "rate_total"
    if any(k in q for k in ("ratio", "twice", "half", "double")):
        return "ratio_transform"
    return "other"


def analyze_false_positives(evaluation: dict[str, Any]) -> dict[str, Any]:
    rows = evaluation["per_candidate_results"]
    fps = [r for r in rows if r["is_correct"] == 1 and r["raw_role_flagged"] == 1]

    signal_counter: Counter[str] = Counter()
    pattern_counter: Counter[str] = Counter()
    reason_counter: Counter[str] = Counter()
    analysis_rows: list[dict[str, Any]] = []

    for row in fps:
        raw_signals = [s for s in row["raw_role_triggered_signals"].split("|") if s]
        for sig in raw_signals:
            signal_counter[sig] += 1

        pattern = _pattern_bucket(row["question"])
        pattern_counter[pattern] += 1

        # Simple root-cause labels for audit readability.
        causes: list[str] = []
        if "missing_required_number" in raw_signals:
            causes.append("implicit_usage_missed")
        if "possible_intermediate_stop_suspected" in raw_signals:
            causes.append("target_type_over_inference")
        if "required_rate_number_missing" in raw_signals:
            causes.append("operator_cue_over_interpretation")
        if not causes:
            causes.append("multi_step_over_penalized")
        for cause in causes:
            reason_counter[cause] += 1

        analysis_rows.append(
            {
                "question_id": row["question_id"],
                "question_pattern": pattern,
                "candidate_answer": row["candidate_answer"],
                "raw_role_triggered_signals": row["raw_role_triggered_signals"],
                "calibrated_decision": row["calibrated_decision"],
                "likely_cause": "|".join(causes),
            }
        )

    return {
        "num_false_positives_raw_role": len(fps),
        "false_positives_by_signal": dict(signal_counter),
        "false_positives_by_question_pattern": dict(pattern_counter),
        "false_positive_likely_causes": dict(reason_counter),
        "false_positive_rows": analysis_rows,
    }


def build_signal_tradeoff_summary(evaluation: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for variant, metrics in evaluation["variant_metrics"].items():
        rows.append(
            {
                "variant": variant,
                "wrong_recall": metrics["wrong_recall"],
                "false_positive_rate_on_correct": metrics["false_positive_rate_on_correct"],
                "recall_minus_fpr": metrics["recall_minus_fpr"],
                "balanced_utility": metrics["balanced_utility"],
                "num_flagged": metrics["num_flagged"],
            }
        )
    return rows


def run_role_calibration(benchmark_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    benchmark = load_benchmark(benchmark_path)
    evaluation = evaluate_benchmark(benchmark)
    fp = analyze_false_positives(evaluation)
    tradeoff_rows = build_signal_tradeoff_summary(evaluation)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    calibration_summary = {
        "num_questions": evaluation["num_questions"],
        "num_candidates": evaluation["num_candidates"],
        "variant_metrics": evaluation["variant_metrics"],
        "false_positive_summary": {
            "num_false_positives_raw_role": fp["num_false_positives_raw_role"],
            "false_positives_by_signal": fp["false_positives_by_signal"],
            "false_positives_by_question_pattern": fp["false_positives_by_question_pattern"],
            "false_positive_likely_causes": fp["false_positive_likely_causes"],
        },
        "evidence_status": "measured_now",
        "real_routing_status": "blocked",
        "claim_status": "exploratory_only",
    }

    summary_path = out / "calibration_summary.json"
    summary_path.write_text(json.dumps(calibration_summary, indent=2))

    fp_path = out / "false_positive_analysis.csv"
    with fp_path.open("w", newline="") as fh:
        rows = fp["false_positive_rows"]
        if rows:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        else:
            writer = csv.writer(fh)
            writer.writerow(
                [
                    "question_id",
                    "question_pattern",
                    "candidate_answer",
                    "raw_role_triggered_signals",
                    "calibrated_decision",
                    "likely_cause",
                ]
            )

    tradeoff_path = out / "signal_tradeoff_summary.csv"
    with tradeoff_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "variant",
                "wrong_recall",
                "false_positive_rate_on_correct",
                "recall_minus_fpr",
                "balanced_utility",
                "num_flagged",
            ],
        )
        writer.writeheader()
        writer.writerows(tradeoff_rows)

    return {
        "calibration_summary_json": str(summary_path),
        "false_positive_analysis_csv": str(fp_path),
        "signal_tradeoff_summary_csv": str(tradeoff_path),
        "evaluation": evaluation,
        "false_positive_analysis": fp,
        "tradeoff_rows": tradeoff_rows,
    }
