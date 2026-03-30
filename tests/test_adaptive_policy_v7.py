from __future__ import annotations

import json
from pathlib import Path

from src.policies.adaptive_policy_v6 import compute_v6_scores
from src.policies.adaptive_policy_v7 import (
    AdaptivePolicyV7Config,
    choose_strategy,
    compute_v7_scores,
    extract_question_features_v6,
)


def test_v7_preserves_false_positive_concise_correct() -> None:
    question = (
        "Natalia sold clips to 48 of her friends in April, and then she sold half as many "
        "clips in May. How many clips did Natalia sell altogether in April and May?"
    )
    first_pass = "Worked it out.\nFinal answer: 72"
    cfg = AdaptivePolicyV7Config()
    feats = extract_question_features_v6(question, cfg)
    assert choose_strategy(question, feats, first_pass, cfg) == "reasoning_greedy"
    s = compute_v7_scores(question, first_pass, cfg)
    assert s["revise_recommended"] is False


def test_v7_revise_on_tobias_real_probe_text() -> None:
    cfg = AdaptivePolicyV7Config()
    snap = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "datasets"
        / "bundled"
        / "real_v6_false_negative_probe_snapshot.jsonl"
    )
    line = [ln for ln in snap.read_text().splitlines() if '"gsm8k_test_11"' in ln][0]
    o = json.loads(line)
    question, trace = o["question"], o["raw_model_output"]
    s = compute_v7_scores(question, trace, cfg)
    assert s["v7_signals"]["need_more_answer_equals_list_price"] is True
    assert s["answer_error_score"] >= cfg.answer_error_revise_threshold
    assert s["revise_recommended"] is True


def test_v7_revise_on_jasmine_real_probe_text() -> None:
    cfg = AdaptivePolicyV7Config()
    snap = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "datasets"
        / "bundled"
        / "real_v6_false_negative_probe_snapshot.jsonl"
    )
    line = [ln for ln in snap.read_text().splitlines() if "gsm8k_test_13" in ln][0]
    o = json.loads(line)
    s = compute_v7_scores(o["question"], o["raw_model_output"], cfg)
    assert s["v7_signals"]["weekday_question_numeric_final"] is True
    assert s["revise_recommended"] is True


def test_v6_still_no_revise_tobias_probe() -> None:
    cfg = AdaptivePolicyV7Config()
    question = (
        "Tobias is buying a new pair of shoes that costs $95. He has been saving up his "
        "allowance for several weeks. He gets a $5 allowance per week. He has already "
        "spent $15 out of his savings. If he has been saving for 3 weeks, how much more "
        "money does he need to buy the shoes?"
    )
    trace = "Final answer: 95"
    s6 = compute_v6_scores(question, trace, cfg)
    assert s6["revise_recommended"] is False
