"""Feature gap analysis for revise-helps cases.

This module identifies the "revise_helps", "reasoning_enough", and
"revise_not_enough" query groups from existing oracle outputs and adaptive
policy outputs, then compares their cheap feature distributions to expose
what current signals fail to capture.

Inputs (all optional — missing files are skipped gracefully):
    outputs/oracle_subset_eval/per_query_matrix.csv
    outputs/oracle_subset_eval/oracle_assignments.csv
    outputs/revise_case_analysis/case_table.csv
    outputs/revise_case_analysis/category_summary.csv
    outputs/adaptive_policy_v3/per_query_results.csv
    outputs/adaptive_policy_v4/per_query_results.csv

Outputs written to outputs/feature_gap_analysis/:
    group_feature_summary.csv
    missed_revise_cases.csv
    pattern_notes.json

See docs/FEATURE_GAP_ANALYSIS.md for full design rationale.
"""

from __future__ import annotations

import csv
import json
import re
import statistics
from pathlib import Path
from typing import Any

from src.features.precompute_features import extract_query_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Groups produced by _assign_group()
GROUP_REVISE_HELPS = "revise_helps"
GROUP_REASONING_ENOUGH = "reasoning_enough"
GROUP_REVISE_NOT_ENOUGH = "revise_not_enough"
GROUP_UNKNOWN = "unknown"

# Cheap wording-trap signals grounded in GSM8K problem structure.
# Each pattern targets a class of surface cue known to correlate with
# incorrect target quantity identification.
_REMAINING_PATTERN = re.compile(
    r"\b(?:remaining|left over|left|have left|are left)\b", re.IGNORECASE
)
_SUBTRACTION_TRAP_PATTERN = re.compile(
    r"\b(?:subtract|take away|spend|spent|gave away|gives away|lost|loses|"
    r"used up|uses up|sold|sells)\b",
    re.IGNORECASE,
)
_TOTAL_EARNED_PATTERN = re.compile(
    r"\b(?:total|altogether|in all|combined|earned|makes|made|receives|received)\b",
    re.IGNORECASE,
)
_UNIT_MISMATCH_PATTERN = re.compile(
    r"\b(?:per|each|every|apiece|a piece|per day|per week|per hour|per month)\b",
    re.IGNORECASE,
)
_INTERMEDIATE_QUANTITY_PATTERN = re.compile(
    r"\b(?:how many (?:does|did|do|will)|how much (?:does|did|do|will)|"
    r"what is the (?:total|number|amount|cost|price|value))\b",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def _read_csv(path: str | Path) -> list[dict[str, str]]:
    """Read a CSV into a list of dicts; return [] if file not found."""
    p = Path(path)
    if not p.exists():
        return []
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))


def load_oracle_assignments(path: str | Path) -> list[dict[str, str]]:
    """Load oracle_assignments.csv.

    Expected columns (subset used):
        question_id, cheapest_correct_strategy, direct_greedy_correct,
        direct_already_optimal
    Missing file returns an empty list.
    """
    return _read_csv(path)


def load_per_query_matrix(path: str | Path) -> list[dict[str, str]]:
    """Load per_query_matrix.csv.

    Expected columns (subset used):
        question_id, strategy, correct, question_text (optional)
    Missing file returns an empty list.
    """
    return _read_csv(path)


def load_case_table(path: str | Path) -> list[dict[str, str]]:
    """Load revise_case_analysis/case_table.csv if present."""
    return _read_csv(path)


def load_policy_results(path: str | Path) -> list[dict[str, str]]:
    """Load adaptive policy per_query_results.csv if present."""
    return _read_csv(path)


# ---------------------------------------------------------------------------
# Group assignment
# ---------------------------------------------------------------------------


def assign_group(
    qid: str,
    direct_greedy_correct: int,
    revise_correct: int | None,
    oracle_any_correct: int,
) -> str:
    """Assign a query to one of the three analysis groups.

    Groups (mutually exclusive, exhaustive over answerable queries):
        revise_helps      — direct_greedy wrong, direct_plus_revise correct
        reasoning_enough  — direct_greedy already correct (no extra compute needed)
        revise_not_enough — no strategy was correct, or revise was tried and
                            still wrong (oracle failed or revise failed)

    Parameters
    ----------
    qid:
        Query identifier (used only for error messages).
    direct_greedy_correct:
        1 if direct_greedy answered correctly, else 0.
    revise_correct:
        1/0 if direct_plus_revise result is known, else None.
    oracle_any_correct:
        1 if at least one strategy was correct, else 0.
    """
    if direct_greedy_correct:
        return GROUP_REASONING_ENOUGH
    if revise_correct == 1:
        return GROUP_REVISE_HELPS
    if oracle_any_correct == 0:
        return GROUP_REVISE_NOT_ENOUGH
    if revise_correct == 0:
        return GROUP_REVISE_NOT_ENOUGH
    # revise_correct is None and some other strategy was correct
    # → reasoning helped, but not specifically revise
    return GROUP_REASONING_ENOUGH


def build_group_map(
    oracle_assignments: list[dict[str, str]],
    per_query_matrix: list[dict[str, str]],
) -> dict[str, str]:
    """Return {question_id: group_label} from oracle outputs.

    Combines oracle_assignments (for direct_greedy_correct) with
    per_query_matrix (for strategy-level correct flags).
    """
    # Build per-query strategy→correct map from matrix
    strategy_correct: dict[str, dict[str, int]] = {}
    for row in per_query_matrix:
        qid = row.get("question_id", "")
        strat = row.get("strategy", "")
        correct_raw = row.get("correct", "0")
        try:
            correct = int(float(correct_raw))
        except (ValueError, TypeError):
            correct = 0
        if qid not in strategy_correct:
            strategy_correct[qid] = {}
        strategy_correct[qid][strat] = correct

    group_map: dict[str, str] = {}

    for row in oracle_assignments:
        qid = row.get("question_id", "")
        if not qid:
            continue
        direct_correct_raw = row.get("direct_greedy_correct", "0")
        try:
            direct_correct = int(float(direct_correct_raw))
        except (ValueError, TypeError):
            direct_correct = 0

        # Revise correct from matrix if available, else from cheapest_correct
        revise_correct: int | None = None
        if qid in strategy_correct:
            sc = strategy_correct[qid]
            if "direct_plus_revise" in sc:
                revise_correct = sc["direct_plus_revise"]

        # oracle_any_correct: at least one strategy correct
        cheapest_strat = row.get("cheapest_correct_strategy", "")
        oracle_any_correct = 1 if cheapest_strat else 0
        if qid in strategy_correct:
            if any(v == 1 for v in strategy_correct[qid].values()):
                oracle_any_correct = 1

        group_map[qid] = assign_group(
            qid, direct_correct, revise_correct, oracle_any_correct
        )

    return group_map


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _extract_wording_trap_features(question_text: str) -> dict[str, int]:
    """Extract lightweight qualitative wording-trap signals from question text."""
    return {
        "has_remaining_left_cue": int(bool(_REMAINING_PATTERN.search(question_text))),
        "has_subtraction_trap_verb": int(bool(_SUBTRACTION_TRAP_PATTERN.search(question_text))),
        "has_total_earned_cue": int(bool(_TOTAL_EARNED_PATTERN.search(question_text))),
        "has_unit_per_cue": int(bool(_UNIT_MISMATCH_PATTERN.search(question_text))),
        "has_intermediate_quantity_ask": int(
            bool(_INTERMEDIATE_QUANTITY_PATTERN.search(question_text))
        ),
    }


def get_question_text_map(per_query_matrix: list[dict[str, str]]) -> dict[str, str]:
    """Build {question_id: question_text} from per_query_matrix rows."""
    qmap: dict[str, str] = {}
    for row in per_query_matrix:
        qid = row.get("question_id", "")
        text = row.get("question_text", row.get("question", ""))
        if qid and text and qid not in qmap:
            qmap[qid] = text
    return qmap


def build_per_query_features(
    qid: str,
    question_text: str,
) -> dict[str, Any]:
    """Combine cheap query features with wording-trap features."""
    base = extract_query_features(question_text)
    wording = _extract_wording_trap_features(question_text)
    return {"question_id": qid, "question_text": question_text, **base, **wording}


# ---------------------------------------------------------------------------
# Policy-triggered revise detection
# ---------------------------------------------------------------------------


def build_policy_revise_set(
    policy_rows: list[dict[str, str]],
) -> set[str]:
    """Return the set of question_ids for which a policy used direct_plus_revise."""
    _REVISE_STRATEGIES = {"direct_plus_revise"}
    triggered: set[str] = set()
    for row in policy_rows:
        strat = row.get("strategy", row.get("chosen_strategy", "")).lower()
        if strat in _REVISE_STRATEGIES or "revise" in strat:
            qid = row.get("question_id", "")
            if qid:
                triggered.add(qid)
    return triggered


# ---------------------------------------------------------------------------
# Feature summary computation
# ---------------------------------------------------------------------------


def _mean_or_none(values: list[float]) -> float | None:
    if not values:
        return None
    return statistics.mean(values)


def _rate_or_none(values: list[int]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


NUMERIC_FEATURES = [
    "question_length_chars",
    "question_length_tokens_approx",
    "num_numeric_mentions",
    "num_sentences_approx",
    "max_numeric_value_approx",
    "numeric_range_approx",
]

BOOL_FEATURES = [
    "has_multi_step_cue",
    "has_currency_symbol",
    "has_percent_symbol",
    "has_fraction_pattern",
    "has_equation_like_pattern",
    "repeated_number_flag",
    "has_remaining_left_cue",
    "has_subtraction_trap_verb",
    "has_total_earned_cue",
    "has_unit_per_cue",
    "has_intermediate_quantity_ask",
]


def compute_group_feature_summary(
    group_features: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Aggregate feature statistics per group.

    Parameters
    ----------
    group_features:
        {group_label: [per_query_feature_dict, ...]}

    Returns
    -------
    list of dicts, one per group, with mean/rate for each feature.
    """
    rows: list[dict[str, Any]] = []
    for group_label, feat_list in sorted(group_features.items()):
        row: dict[str, Any] = {"group": group_label, "n": len(feat_list)}
        for feat in NUMERIC_FEATURES:
            vals = [float(f[feat]) for f in feat_list if feat in f]
            row[f"{feat}_mean"] = round(_mean_or_none(vals) or 0.0, 3)
        for feat in BOOL_FEATURES:
            vals = [int(f.get(feat, 0)) for f in feat_list]
            row[f"{feat}_rate"] = round(_rate_or_none(vals) or 0.0, 3)
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Missed revise case analysis
# ---------------------------------------------------------------------------


def find_missed_revise_cases(
    group_map: dict[str, str],
    per_query_features: dict[str, dict[str, Any]],
    v3_revise_set: set[str],
    v4_revise_set: set[str],
) -> list[dict[str, Any]]:
    """Find queries where revise would help but neither v3 nor v4 triggered revise.

    Returns a list of dicts, one per missed query, with the query's features
    and an annotation of which cheap signals were absent.
    """
    missed: list[dict[str, Any]] = []
    for qid, group in group_map.items():
        if group != GROUP_REVISE_HELPS:
            continue
        v3_triggered = qid in v3_revise_set
        v4_triggered = qid in v4_revise_set
        if v3_triggered or v4_triggered:
            continue  # at least one policy caught it
        feats = per_query_features.get(qid, {})
        absent_signals = _find_absent_signals(feats)
        row: dict[str, Any] = {
            "question_id": qid,
            "group": GROUP_REVISE_HELPS,
            "v3_triggered_revise": int(v3_triggered),
            "v4_triggered_revise": int(v4_triggered),
            "question_text": feats.get("question_text", ""),
            "absent_cheap_signals": "; ".join(absent_signals) if absent_signals else "none",
        }
        row.update({k: feats.get(k, "") for k in BOOL_FEATURES + NUMERIC_FEATURES})
        missed.append(row)
    return missed


def _find_absent_signals(feats: dict[str, Any]) -> list[str]:
    """Return names of cheap signals that are False/0 in a query's feature dict."""
    absent = []
    for feat in BOOL_FEATURES:
        if not feats.get(feat, 0):
            absent.append(feat)
    return absent


# ---------------------------------------------------------------------------
# Pattern extraction
# ---------------------------------------------------------------------------


def extract_qualitative_patterns(
    missed_cases: list[dict[str, Any]],
    revise_helps_features: list[dict[str, Any]],
    reasoning_enough_features: list[dict[str, Any]],
) -> dict[str, Any]:
    """Produce a structured JSON of qualitative patterns grounded in the data.

    Patterns are based on comparing feature rates between the revise_helps
    and reasoning_enough groups and on inspecting wording-trap signals in
    the missed cases.  No patterns are invented; all claims are quantified
    with observed counts.
    """
    n_missed = len(missed_cases)
    n_revise_helps = len(revise_helps_features)
    n_reasoning = len(reasoning_enough_features)
    no_revise_data = n_revise_helps == 0

    def _rate(feats: list[dict], key: str) -> float:
        if not feats:
            return 0.0
        return sum(int(f.get(key, 0)) for f in feats) / len(feats)

    # Compute rate differences for wording-trap features
    wording_gaps: list[dict[str, Any]] = []
    for feat in [
        "has_remaining_left_cue",
        "has_subtraction_trap_verb",
        "has_total_earned_cue",
        "has_unit_per_cue",
        "has_intermediate_quantity_ask",
        "has_multi_step_cue",
    ]:
        rh_rate = _rate(revise_helps_features, feat)
        re_rate = _rate(reasoning_enough_features, feat)
        wording_gaps.append({
            "feature": feat,
            "revise_helps_rate": round(rh_rate, 3),
            "reasoning_enough_rate": round(re_rate, 3),
            "gap": round(rh_rate - re_rate, 3),
        })

    # Sort by absolute gap descending — most discriminative first
    wording_gaps.sort(key=lambda x: abs(x["gap"]), reverse=True)

    # Count how often each signal was absent in missed cases
    absent_counts: dict[str, int] = {}
    for case in missed_cases:
        absent_str = case.get("absent_cheap_signals", "")
        for sig in absent_str.split("; "):
            sig = sig.strip()
            if sig and sig != "none":
                absent_counts[sig] = absent_counts.get(sig, 0) + 1

    # Top absent signals (most commonly absent in missed cases)
    top_absent = sorted(absent_counts.items(), key=lambda x: x[1], reverse=True)[:5]

    patterns: dict[str, Any] = {
        "data_summary": {
            "n_revise_helps": n_revise_helps,
            "n_reasoning_enough": n_reasoning,
            "n_missed_revise_cases": n_missed,
        },
        "wording_trap_feature_gaps": wording_gaps,
        "most_absent_signals_in_missed_cases": [
            {"signal": sig, "absent_count": cnt} for sig, cnt in top_absent
        ],
        "qualitative_patterns": _build_qualitative_patterns(
            wording_gaps, missed_cases, no_revise_data
        ),
        "current_feature_failures": _summarize_feature_failures(
            wording_gaps, n_missed, n_revise_helps
        ),
        "candidate_next_signals": _candidate_next_signals(),
        "suggested_direction": _suggested_direction(n_missed, n_revise_helps, n_reasoning),
    }
    return patterns


def _build_qualitative_patterns(
    wording_gaps: list[dict[str, Any]],
    missed_cases: list[dict[str, Any]],
    no_revise_data: bool,
) -> list[dict[str, str]]:
    """Extract up to 5 grounded qualitative patterns from the data."""
    patterns: list[dict[str, str]] = []

    # Pattern 1: remaining/left cue
    remaining_rate = next(
        (g["revise_helps_rate"] for g in wording_gaps if g["feature"] == "has_remaining_left_cue"),
        0.0,
    )
    remaining_gap = next(
        (g["gap"] for g in wording_gaps if g["feature"] == "has_remaining_left_cue"),
        0.0,
    )
    if remaining_rate > 0 or no_revise_data:
        patterns.append({
            "pattern": "remaining/left wording trap",
            "description": (
                "Questions using 'remaining', 'left', or 'left over' often ask for a "
                "final quantity after subtraction. Direct greedy tends to report an "
                "intermediate total rather than the remainder."
            ),
            "grounding": (
                f"has_remaining_left_cue rate in revise_helps={remaining_rate:.3f}; "
                f"gap vs reasoning_enough={remaining_gap:+.3f}"
            ),
        })

    # Pattern 2: subtraction trap
    sub_rate = next(
        (
            g["revise_helps_rate"]
            for g in wording_gaps
            if g["feature"] == "has_subtraction_trap_verb"
        ),
        0.0,
    )
    sub_gap = next(
        (g["gap"] for g in wording_gaps if g["feature"] == "has_subtraction_trap_verb"),
        0.0,
    )
    if sub_rate > 0 or no_revise_data:
        patterns.append({
            "pattern": "hidden subtraction/addition structure",
            "description": (
                "Verbs like 'spent', 'gave away', 'sold', 'lost' signal a two-step "
                "computation (compute intermediate, then subtract). Direct greedy "
                "sometimes returns the pre-subtraction amount."
            ),
            "grounding": (
                f"has_subtraction_trap_verb rate in revise_helps={sub_rate:.3f}; "
                f"gap vs reasoning_enough={sub_gap:+.3f}"
            ),
        })

    # Pattern 3: unit/per mismatch
    unit_rate = next(
        (g["revise_helps_rate"] for g in wording_gaps if g["feature"] == "has_unit_per_cue"),
        0.0,
    )
    unit_gap = next(
        (g["gap"] for g in wording_gaps if g["feature"] == "has_unit_per_cue"),
        0.0,
    )
    if unit_rate > 0 or no_revise_data:
        patterns.append({
            "pattern": "unit or target quantity mismatch",
            "description": (
                "Words like 'per', 'each', 'every' introduce a per-unit rate that must "
                "be multiplied. Direct greedy may return the rate rather than the total, "
                "or vice versa."
            ),
            "grounding": (
                f"has_unit_per_cue rate in revise_helps={unit_rate:.3f}; "
                f"gap vs reasoning_enough={unit_gap:+.3f}"
            ),
        })

    # Pattern 4: intermediate quantity ask
    iq_rate = next(
        (
            g["revise_helps_rate"]
            for g in wording_gaps
            if g["feature"] == "has_intermediate_quantity_ask"
        ),
        0.0,
    )
    iq_gap = next(
        (g["gap"] for g in wording_gaps if g["feature"] == "has_intermediate_quantity_ask"),
        0.0,
    )
    patterns.append({
        "pattern": "answering an intermediate quantity",
        "description": (
            "Questions with 'how many does/did' or 'what is the total' can have "
            "multiple plausible numeric answers along the reasoning chain. "
            "The direct pass may stop at an intermediate result."
        ),
        "grounding": (
            f"has_intermediate_quantity_ask rate in revise_helps={iq_rate:.3f}; "
            f"gap vs reasoning_enough={iq_gap:+.3f}"
        ),
    })

    # Pattern 5: multi-step complexity
    ms_rate = next(
        (g["revise_helps_rate"] for g in wording_gaps if g["feature"] == "has_multi_step_cue"),
        0.0,
    )
    ms_gap = next(
        (g["gap"] for g in wording_gaps if g["feature"] == "has_multi_step_cue"),
        0.0,
    )
    patterns.append({
        "pattern": "multi-step keyword present but not captured by existing cue",
        "description": (
            "Even when 'total', 'remaining', 'after', etc. are present, the existing "
            "has_multi_step_cue boolean does not distinguish between queries where the "
            "multi-step structure causes a wrong answer and those where direct is fine."
        ),
        "grounding": (
            f"has_multi_step_cue rate in revise_helps={ms_rate:.3f}; "
            f"gap vs reasoning_enough={ms_gap:+.3f}"
        ),
    })

    return patterns[:5]


def _summarize_feature_failures(
    wording_gaps: list[dict[str, Any]],
    n_missed: int,
    n_revise_helps: int,
) -> list[str]:
    """Return a bullet-list summary of what current features fail to capture."""
    coverage = n_missed / max(n_revise_helps, 1)
    failures = [
        (
            f"Coverage gap: {n_missed}/{n_revise_helps} revise_helps queries "
            f"({coverage:.1%}) were not triggered by v3/v4 — current signals "
            "are insufficient to identify these cases."
        ),
        (
            "has_multi_step_cue is a single boolean that collapses all multi-step "
            "patterns into one signal; it cannot distinguish subtraction traps "
            "from additive accumulations, which require different strategies."
        ),
        (
            "No signal captures the *target quantity* of the question — whether the "
            "question asks for a final remainder, a total, or a per-unit rate is "
            "invisible to the current feature set."
        ),
        (
            "Wording-trap features (remaining/left, subtraction verbs, per-unit cues) "
            "are absent from the current cheap feature vector and would directly "
            "address the most common revise-help patterns."
        ),
        (
            "Current features are purely syntactic (character counts, keyword flags). "
            "They carry no semantic signal about whether the question has a "
            "single-step or multi-step solution path."
        ),
    ]
    # Mention the most discriminative feature gap found
    if wording_gaps:
        top = wording_gaps[0]
        failures.append(
            f"The most discriminative missing feature is '{top['feature']}' "
            f"(gap={top['gap']:+.3f} between revise_helps and reasoning_enough)."
        )
    return failures


def _candidate_next_signals() -> list[dict[str, str]]:
    """Return a list of 5 concrete candidate next-step signal ideas."""
    return [
        {
            "signal": "has_remainder_ask",
            "description": (
                "Boolean: True when the question surface-form asks for what "
                "'remains', 'is left', or 'is still needed'. Captures the most "
                "common form of subtraction-final-step problems."
            ),
            "implementation": "Regex over 'remaining|left over|left|have left' in question.",
        },
        {
            "signal": "subtraction_verb_count",
            "description": (
                "Integer count of spend/gave/lost/sold verbs. A higher count "
                "correlates with hidden multi-step subtraction chains that the "
                "direct pass often short-circuits."
            ),
            "implementation": "Regex count of subtraction-class verbs in question.",
        },
        {
            "signal": "per_unit_rate_flag",
            "description": (
                "Boolean: True when 'per', 'each', or 'every' appears alongside "
                "a numeric token. Catches rate×quantity multiplication steps that "
                "the direct pass may skip."
            ),
            "implementation": "Regex: look for numeric + per/each/every within 5 tokens.",
        },
        {
            "signal": "first_pass_answer_in_question_flag",
            "description": (
                "Boolean: True when the parsed first-pass answer appears verbatim "
                "as a number in the question text. This is a strong signal that the "
                "model echoed a given value instead of computing the target."
            ),
            "implementation": "Check if extracted_answer ∈ numeric_tokens(question_text).",
        },
        {
            "signal": "sentence_count_vs_numeric_token_ratio",
            "description": (
                "Float: num_sentences / max(num_numeric_mentions, 1). High ratio "
                "means many sentences with few numbers — often a narrative problem "
                "requiring careful tracking of which quantity is final."
            ),
            "implementation": "Derived from existing question_length and num_numeric fields.",
        },
    ]


def _suggested_direction(
    n_missed: int, n_revise_helps: int, n_reasoning: int
) -> str:
    """Suggest the most useful next method direction based on group sizes."""
    if n_revise_helps == 0:
        return (
            "No revise_helps cases found in current data. "
            "Run oracle evaluation with real outputs before drawing conclusions. "
            "Once data is available, revisit this analysis."
        )
    coverage = 1.0 - n_missed / max(n_revise_helps, 1)
    if n_missed == 0:
        direction = "learned routing"
        rationale = (
            "All revise_helps cases were already caught by v3/v4. The bottleneck "
            "is likely precision (false positives). A lightweight learned router "
            "trained on the routing dataset (routing_dataset.csv) would improve "
            "precision without needing new hand-crafted signals."
        )
    elif coverage < 0.5:
        direction = "stronger hand-crafted signals"
        rationale = (
            f"Only {coverage:.0%} of revise_helps cases were caught. "
            "The current cheap feature set misses most revise-help patterns. "
            "Adding the 5 candidate signals above (remainder ask, subtraction verb "
            "count, per-unit flag, answer-echo flag, sentence/numeric ratio) should "
            "close the majority of the gap before a learned model is warranted."
        )
    else:
        direction = "stronger verifier"
        rationale = (
            f"{coverage:.0%} of revise_helps cases were caught, but some were missed. "
            "A stronger verifier that checks whether the predicted answer matches "
            "one of the numeric tokens in the question (the answer-echo heuristic) "
            "could catch additional cases without retraining the routing policy."
        )
    return f"Recommended direction: {direction}. Rationale: {rationale}"


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def write_group_feature_summary(
    summary_rows: list[dict[str, Any]],
    output_dir: str | Path,
) -> str:
    """Write group_feature_summary.csv; return the written path."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    out = base / "group_feature_summary.csv"
    if not summary_rows:
        out.write_text("group,n\n")
        return str(out)
    fieldnames = list(summary_rows[0].keys())
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)
    return str(out)


def write_missed_revise_cases(
    missed_cases: list[dict[str, Any]],
    output_dir: str | Path,
) -> str:
    """Write missed_revise_cases.csv; return the written path."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    out = base / "missed_revise_cases.csv"
    if not missed_cases:
        out.write_text("question_id,group,v3_triggered_revise,v4_triggered_revise\n")
        return str(out)
    fieldnames = list(missed_cases[0].keys())
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(missed_cases)
    return str(out)


def write_pattern_notes(
    patterns: dict[str, Any],
    output_dir: str | Path,
) -> str:
    """Write pattern_notes.json; return the written path."""
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    out = base / "pattern_notes.json"
    out.write_text(json.dumps(patterns, indent=2))
    return str(out)


# ---------------------------------------------------------------------------
# Top-level entry point
# ---------------------------------------------------------------------------


def run_feature_gap_analysis(
    oracle_assignments_path: str | Path,
    per_query_matrix_path: str | Path,
    case_table_path: str | Path,
    category_summary_path: str | Path,
    v3_results_path: str | Path,
    v4_results_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Run the full feature gap analysis pipeline.

    All input paths are optional — missing files are handled gracefully.
    Outputs are written to *output_dir*.

    Returns
    -------
    dict with keys:
        group_sizes, n_missed_revise_cases, output_paths,
        pattern_notes, summary_rows, missed_cases
    """
    # ---- Load inputs -------------------------------------------------------
    oracle_assignments = load_oracle_assignments(oracle_assignments_path)
    per_query_matrix = load_per_query_matrix(per_query_matrix_path)
    case_table = load_case_table(case_table_path)
    v3_rows = load_policy_results(v3_results_path)
    v4_rows = load_policy_results(v4_results_path)

    # ---- Merge case_table into oracle_assignments if needed ----------------
    # case_table may contain category labels (revise_helps, etc.) from a
    # prior manual analysis pass — use them to augment oracle_assignments.
    case_category_map: dict[str, str] = {}
    for row in case_table:
        qid = row.get("question_id", "")
        cat = row.get("category", row.get("group", ""))
        if qid and cat:
            case_category_map[qid] = cat

    # ---- Build group map ---------------------------------------------------
    group_map = build_group_map(oracle_assignments, per_query_matrix)

    # Override with explicit case_table categories where available
    for qid, cat in case_category_map.items():
        if cat in {GROUP_REVISE_HELPS, GROUP_REASONING_ENOUGH, GROUP_REVISE_NOT_ENOUGH}:
            group_map[qid] = cat

    # ---- Build question text map -------------------------------------------
    qtext_map = get_question_text_map(per_query_matrix)

    # Also pull question texts from oracle_assignments if available
    for row in oracle_assignments:
        qid = row.get("question_id", "")
        text = row.get("question_text", row.get("question", ""))
        if qid and text and qid not in qtext_map:
            qtext_map[qid] = text

    # ---- Compute per-query features ----------------------------------------
    per_query_features: dict[str, dict[str, Any]] = {}
    all_qids = set(group_map.keys()) | set(qtext_map.keys())
    for qid in all_qids:
        text = qtext_map.get(qid, "")
        per_query_features[qid] = build_per_query_features(qid, text)

    # ---- Build policy revise sets ------------------------------------------
    v3_revise_set = build_policy_revise_set(v3_rows)
    v4_revise_set = build_policy_revise_set(v4_rows)

    # ---- Group feature lists -----------------------------------------------
    group_features: dict[str, list[dict[str, Any]]] = {
        GROUP_REVISE_HELPS: [],
        GROUP_REASONING_ENOUGH: [],
        GROUP_REVISE_NOT_ENOUGH: [],
    }
    for qid, group in group_map.items():
        feats = per_query_features.get(qid, build_per_query_features(qid, ""))
        if group in group_features:
            group_features[group].append(feats)

    # ---- Compute summaries -------------------------------------------------
    summary_rows = compute_group_feature_summary(group_features)

    # ---- Find missed cases -------------------------------------------------
    missed_cases = find_missed_revise_cases(
        group_map, per_query_features, v3_revise_set, v4_revise_set
    )

    # ---- Extract qualitative patterns -------------------------------------
    pattern_notes = extract_qualitative_patterns(
        missed_cases,
        group_features[GROUP_REVISE_HELPS],
        group_features[GROUP_REASONING_ENOUGH],
    )

    # ---- Write outputs -----------------------------------------------------
    out_dir = Path(output_dir)
    summary_path = write_group_feature_summary(summary_rows, out_dir)
    missed_path = write_missed_revise_cases(missed_cases, out_dir)
    patterns_path = write_pattern_notes(pattern_notes, out_dir)

    group_sizes = {g: len(fl) for g, fl in group_features.items()}

    return {
        "group_sizes": group_sizes,
        "n_missed_revise_cases": len(missed_cases),
        "output_paths": {
            "group_feature_summary_csv": summary_path,
            "missed_revise_cases_csv": missed_path,
            "pattern_notes_json": patterns_path,
        },
        "pattern_notes": pattern_notes,
        "summary_rows": summary_rows,
        "missed_cases": missed_cases,
    }
