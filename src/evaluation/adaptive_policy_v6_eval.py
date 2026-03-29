"""Offline evaluation for adaptive policy v6 vs v4/v5 (no LLM calls)."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.policies.adaptive_policy_v4 import (
    AdaptivePolicyV4Config,
    extract_question_features_v4,
)
from src.policies.adaptive_policy_v4 import (
    choose_strategy as choose_strategy_v4,
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


# Exact traces from docs/FALSE_POSITIVE_ANALYSIS.md (offline false-positive study).
FALSE_POSITIVE_FIXTURES: list[dict[str, Any]] = [
    {
        "question_id": "gsm8k_test_8",
        "category": "false_positive_doc",
        "question": (
            "Alexis is applying for a new job and bought a new set of business clothes "
            "to wear to the interview. She went to a department store with a budget of $200 "
            "and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, "
            "$11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost "
            "the receipt for them. She has $16 left from her budget. "
            "How much did Alexis pay for the shoes?"
        ),
        "gold_answer": "41",
        "first_pass_output": "Itemized spending leaves 41 for shoes.\nFinal answer: 41",
    },
    {
        "question_id": "gsm8k_test_11",
        "category": "false_positive_doc",
        "question": (
            "Tobias is buying a new pair of shoes that costs $95. He has been saving up his "
            "allowance for several weeks. He gets a $5 allowance per week. He has already "
            "spent $15 out of his savings. If he has been saving for 3 weeks, how much more "
            "money does he need to buy the shoes?"
        ),
        "gold_answer": "65",
        "first_pass_output": "Computed savings and need.\nFinal answer: 65",
    },
    {
        "question_id": "gsm8k_test_18",
        "category": "false_positive_doc",
        "question": (
            "At the beginning of the day there were 74 apples in a basket. During the day, "
            "17 more apples were added to the basket and 31 apples were removed. "
            "How many apples are in the basket at the end of the day?"
        ),
        "gold_answer": "60",
        "first_pass_output": "Worked it out.\nFinal answer: 60",
    },
    {
        "question_id": "gsm8k_test_0",
        "category": "false_positive_doc",
        "question": (
            "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
            "clips in May. How many clips did Natalia sell altogether in April and May?"
        ),
        "gold_answer": "72",
        "first_pass_output": "Worked it out.\nFinal answer: 72",
    },
    {
        "question_id": "gsm8k_test_13",
        "category": "false_positive_doc",
        "question": (
            "Jasmine had 3 paperclips on Monday, then she had 6 on Tuesday, and her number of "
            "paperclips proceeded to double on each subsequent day. On what day of the week "
            "did she first have more than 100 paperclips?"
        ),
        "gold_answer": "Sunday",
        "first_pass_output": "Doubling each day passes 100 on Sunday.\nFinal answer: Sunday",
    },
]


# Revise-worthy proxies: wrong first pass that should still trigger revise under a good policy.
_RECALL_FIXTURES: list[dict[str, Any]] = [
    {
        "question_id": "synthetic_bus_seats",
        "category": "recall_proxy",
        "question": "A bus has 40 seats. 26 are occupied. How many seats are left?",
        "gold_answer": "14",
        "first_pass_output": "Final answer: 26",
    },
    {
        "question_id": "synthetic_apples_remaining",
        "category": "recall_proxy",
        "question": (
            "There are 20 apples in a crate. After selling 8 apples, how many apples remain "
            "in the crate?"
        ),
        "gold_answer": "12",
        "first_pass_output": "Final answer: 8",
    },
]


def _policy_choice_after_reasoning(
    policy: str,
    question: str,
    first_pass: str,
    v4_cfg: AdaptivePolicyV4Config,
    v5_cfg: AdaptivePolicyV5Config,
    v6_cfg: AdaptivePolicyV6Config,
) -> str:
    if policy == "v4":
        feats = extract_question_features_v4(question, v4_cfg)
        return choose_strategy_v4(question, feats, first_pass, v4_cfg)
    if policy == "v5":
        feats = extract_question_features_v5(question, v5_cfg)
        return choose_strategy_v5(question, feats, first_pass, v5_cfg)
    feats = extract_question_features_v6(question, v6_cfg)
    return choose_strategy_v6(question, feats, first_pass, v6_cfg)


def run_offline_adaptive_policy_v6_eval(
    v4_config: AdaptivePolicyV4Config | None = None,
    v5_config: AdaptivePolicyV5Config | None = None,
    v6_config: AdaptivePolicyV6Config | None = None,
    include_recall_fixtures: bool = True,
) -> dict[str, Any]:
    """Compare v4/v5/v6 routing on documented false positives + optional recall proxies."""
    v4_cfg = v4_config or AdaptivePolicyV4Config()
    v5_cfg = v5_config or AdaptivePolicyV5Config()
    v6_cfg = v6_config or AdaptivePolicyV6Config()

    fixtures = list(FALSE_POSITIVE_FIXTURES)
    if include_recall_fixtures:
        fixtures.extend(_RECALL_FIXTURES)

    per_case: list[dict[str, Any]] = []
    fp_v5_revise = 0
    fp_v6_revise = 0
    fp_total = 0
    recall_v6_revise = 0
    recall_total = 0

    for row in fixtures:
        q = row["question"]
        fp = row["first_pass_output"]
        cat = row["category"]
        c4 = _policy_choice_after_reasoning("v4", q, fp, v4_cfg, v5_cfg, v6_cfg)
        c5 = _policy_choice_after_reasoning("v5", q, fp, v4_cfg, v5_cfg, v6_cfg)
        c6 = _policy_choice_after_reasoning("v6", q, fp, v4_cfg, v5_cfg, v6_cfg)
        v6s = compute_v6_scores(q, fp, v6_cfg)

        is_fp = cat == "false_positive_doc"
        if is_fp:
            fp_total += 1
            fp_v5_revise += int(c5 == "direct_plus_revise")
            fp_v6_revise += int(c6 == "direct_plus_revise")
        else:
            recall_total += 1
            recall_v6_revise += int(c6 == "direct_plus_revise")

        per_case.append(
            {
                "question_id": row["question_id"],
                "category": cat,
                "gold_answer": row["gold_answer"],
                "chosen_v4": c4,
                "chosen_v5": c5,
                "chosen_v6": c6,
                "v5_revise": c5 == "direct_plus_revise",
                "v6_revise": c6 == "direct_plus_revise",
                "v6_explanation_warning_score": v6s["explanation_warning_score"],
                "v6_answer_error_score": v6s["answer_error_score"],
                "v6_final_answer_confident": v6s["final_answer_confident"],
                "v6_revise_recommended": v6s["revise_recommended"],
                "v6_revise_reason": v6s["revise_reason"],
                "v6_contributing_explanation": ",".join(v6s["contributing_explanation_signals"]),
                "v6_contributing_answer_error": ",".join(v6s["contributing_answer_error_signals"]),
            }
        )

    signal_rows: list[dict[str, Any]] = []
    for name, count in sorted(
        {
            "false_positive_v5_revise": fp_v5_revise,
            "false_positive_v6_revise": fp_v6_revise,
            "false_positive_cases": fp_total,
            "recall_proxy_v6_revise": recall_v6_revise,
            "recall_proxy_cases": recall_total,
        }.items(),
        key=lambda x: x[0],
    ):
        signal_rows.append({"metric": name, "value": count})

    return {
        "run_status": "COMPLETED",
        "evidence_status": "measured_now",
        "description": (
            "Offline routing: FALSE_POSITIVE_ANALYSIS fixtures + recall proxies"
        ),
        "false_positive_v5_revise_count": fp_v5_revise,
        "false_positive_v6_revise_count": fp_v6_revise,
        "false_positive_cases": fp_total,
        "recall_proxy_v6_revise_count": recall_v6_revise,
        "recall_proxy_cases": recall_total,
        "per_case_results": per_case,
        "signal_summary": signal_rows,
    }


def write_adaptive_policy_v6_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    summary_path = base / "summary.json"
    payload = {k: v for k, v in result.items() if k not in {"per_case_results", "signal_summary"}}
    summary_path.write_text(json.dumps(payload, indent=2))
    per_csv = _write_csv(result.get("per_case_results", []), base / "per_case_results.csv")
    sig_csv = _write_csv(result.get("signal_summary", []), base / "signal_summary.csv")
    return {
        "summary_json": str(summary_path),
        "per_case_csv": per_csv,
        "signal_summary_csv": sig_csv,
    }


def format_offline_summary(result: dict[str, Any], paths: dict[str, str] | None = None) -> str:
    fp_n = result["false_positive_cases"]
    v5_fp = result["false_positive_v5_revise_count"]
    v6_fp = result["false_positive_v6_revise_count"]
    rec_r = result["recall_proxy_v6_revise_count"]
    rec_n = result["recall_proxy_cases"]
    lines = [
        "--- Adaptive Policy V6 (offline benchmark) ---",
        f"run_status: {result.get('run_status')}",
        f"false_positive_cases: {fp_n}",
        f"v5 revise on false positives: {v5_fp}/{fp_n}",
        f"v6 revise on false positives: {v6_fp}/{fp_n}",
        f"recall_proxy v6 revise: {rec_r}/{rec_n}",
    ]
    if paths:
        lines.extend(
            [
                "",
                f"summary_json: {paths['summary_json']}",
                f"per_case_csv: {paths['per_case_csv']}",
                f"signal_summary_csv: {paths['signal_summary_csv']}",
            ]
        )
    return "\n".join(lines)
