"""Offline evaluation: adaptive policy v7 vs v5/v6 on fixtures + real probe snapshot."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.evaluation.adaptive_policy_v6_eval import (
    _RECALL_FIXTURES,
    FALSE_POSITIVE_FIXTURES,
)
from src.policies.adaptive_policy_v5 import (
    AdaptivePolicyV5Config,
    extract_question_features_v5,
)
from src.policies.adaptive_policy_v5 import (
    choose_strategy as choose_strategy_v5,
)
from src.policies.adaptive_policy_v6 import (
    AdaptivePolicyV6Config,
    compute_v6_scores,
    extract_question_features_v6,
)
from src.policies.adaptive_policy_v6 import (
    choose_strategy as choose_strategy_v6,
)
from src.policies.adaptive_policy_v7 import (
    AdaptivePolicyV7Config,
    compute_v7_scores,
)
from src.policies.adaptive_policy_v7 import (
    choose_strategy as choose_strategy_v7,
)

_SNAPSHOT = (
    Path(__file__).resolve().parent.parent
    / "datasets"
    / "bundled"
    / "real_v6_false_negative_probe_snapshot.jsonl"
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
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({f: row.get(f, "") for f in fieldnames})
    return str(path)


def _choose(
    version: str,
    question: str,
    first_pass: str,
    v5: AdaptivePolicyV5Config,
    v6: AdaptivePolicyV6Config,
    v7: AdaptivePolicyV7Config,
) -> str:
    if version == "v5":
        f = extract_question_features_v5(question, v5)
        return choose_strategy_v5(question, f, first_pass, v5)
    if version == "v6":
        f = extract_question_features_v6(question, v6)
        return choose_strategy_v6(question, f, first_pass, v6)
    f = extract_question_features_v6(question, v7)
    return choose_strategy_v7(question, f, first_pass, v7)


def run_adaptive_policy_v7_offline_eval(
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    out = Path(output_dir or "outputs/adaptive_policy_v7")
    if not out.is_absolute():
        out = Path(__file__).resolve().parent.parent.parent / out

    v5 = AdaptivePolicyV5Config()
    v6 = AdaptivePolicyV6Config()
    v7 = AdaptivePolicyV7Config()

    fp_rows: list[dict[str, Any]] = []
    for row in FALSE_POSITIVE_FIXTURES:
        q, t = row["question"], row["first_pass_output"]
        fp_rows.append(
            {
                "question_id": row["question_id"],
                "chosen_v5": _choose("v5", q, t, v5, v6, v7),
                "chosen_v6": _choose("v6", q, t, v5, v6, v7),
                "chosen_v7": _choose("v7", q, t, v5, v6, v7),
                "v7_revise": _choose("v7", q, t, v5, v6, v7) == "direct_plus_revise",
            }
        )

    recall_rows: list[dict[str, Any]] = []
    for row in _RECALL_FIXTURES:
        q, t = row["question"], row["first_pass_output"]
        recall_rows.append(
            {
                "question_id": row["question_id"],
                "chosen_v5": _choose("v5", q, t, v5, v6, v7),
                "chosen_v6": _choose("v6", q, t, v5, v6, v7),
                "chosen_v7": _choose("v7", q, t, v5, v6, v7),
            }
        )

    fn_rows: list[dict[str, Any]] = []
    if _SNAPSHOT.exists():
        for line in _SNAPSHOT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            o = json.loads(line)
            q, t = o["question"], o["raw_model_output"]
            s6 = compute_v6_scores(q, t, v6)
            s7 = compute_v7_scores(q, t, v7)
            fn_rows.append(
                {
                    "question_id": o["question_id"],
                    "gold_answer": o.get("gold_answer", ""),
                    "probe_correct": o.get("correct", ""),
                    "v6_revise": s6["revise_recommended"],
                    "v7_revise": s7["revise_recommended"],
                    "v6_answer_error": s6["answer_error_score"],
                    "v7_answer_error": s7["answer_error_score"],
                    "v7_extra_error": s7["v7_extra_answer_error"],
                    "v7_revise_reason": s7["revise_reason"],
                    "v7_signals_json": json.dumps(s7["v7_signals"], sort_keys=True),
                }
            )

    fp_v7_rev = sum(1 for r in fp_rows if r["v7_revise"])
    fp_v6_rev = sum(1 for r in fp_rows if r["chosen_v6"] == "direct_plus_revise")

    fn_wrong = [r for r in fn_rows if r.get("probe_correct") is False]
    fn_wrong_v6_miss = sum(1 for r in fn_wrong if not r["v6_revise"])
    fn_wrong_v7_miss = sum(1 for r in fn_wrong if not r["v7_revise"])

    summary = {
        "run_status": "COMPLETED",
        "evidence_status": "measured_now",
        "false_positive_fixtures": len(fp_rows),
        "false_positive_v6_revise_count": fp_v6_rev,
        "false_positive_v7_revise_count": fp_v7_rev,
        "recall_fixtures": len(recall_rows),
        "snapshot_probe_rows": len(fn_rows),
        "snapshot_path": str(_SNAPSHOT),
        "snapshot_wrong_count": len(fn_wrong),
        "snapshot_wrong_v6_no_revise": fn_wrong_v6_miss,
        "snapshot_wrong_v7_no_revise": fn_wrong_v7_miss,
    }

    _write_csv(fp_rows, out / "false_positive_recheck.csv")
    _write_csv(fn_rows, out / "false_negative_probe.csv")

    fp5 = sum(1 for r in fp_rows if r["chosen_v5"] == "direct_plus_revise")
    rv7 = sum(1 for r in recall_rows if r["chosen_v7"] == "direct_plus_revise")
    sig_rows = [
        {"metric": "fp_v5_revise", "value": fp5},
        {"metric": "fp_v6_revise", "value": fp_v6_rev},
        {"metric": "fp_v7_revise", "value": fp_v7_rev},
        {"metric": "recall_v7_revise", "value": rv7},
    ]
    _write_csv(sig_rows, out / "signal_summary.csv")

    for r in fp_rows:
        r["suite"] = "false_positive"
    for r in recall_rows:
        r["suite"] = "recall"
    _write_csv(fp_rows + recall_rows, out / "per_case_results.csv")

    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {
        **summary,
        "false_positive_rows": fp_rows,
        "recall_rows": recall_rows,
        "false_negative_probe_rows": fn_rows,
    }

