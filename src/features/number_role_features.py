"""Per-number role extraction and role-coverage features for math word problems.

Lightweight, interpretable heuristics only (regex/string/rule-based).
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

DIGIT_RE = re.compile(r"-?\d+(?:\.\d+)?")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
TOKEN_RE = re.compile(r"\b\w+\b")

NUMBER_WORDS: dict[str, float] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
}
MULT_WORDS: dict[str, float] = {
    "half": 0.5,
    "double": 2,
    "twice": 2,
    "triple": 3,
    "thrice": 3,
    "dozen": 12,
}

UNIT_HINT_RE = re.compile(
    r"\b(?:dollars?|cents?|hours?|minutes?|days?|weeks?|months?|years?|"
    r"students?|people|friends?|apples?|books?|boxes?|trips?|buses?|"
    r"tickets?|pages?|liters?|kg|miles?|km|items?|seats?|cookies?|"
    r"candies?|cars?|balls?)\b",
    re.IGNORECASE,
)
VERB_RE = re.compile(
    r"\b(?:add|added|gain|gained|receive|received|buy|bought|earn|earned|"
    r"spend|spent|lose|lost|gave|give|sold|sell|used|use|ate|consume|"
    r"divide|divided|multiply|multiplied|split|share|left|remaining|remain|"
    r"need|needed|required|minimum|maximum|compare|difference|more|less)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class _RoleCfg:
    add_words: tuple[str, ...] = (
        "add",
        "added",
        "gain",
        "gained",
        "receive",
        "received",
        "buy",
        "bought",
        "more",
    )
    sub_words: tuple[str, ...] = ("spent", "lost", "gave", "sold", "used", "ate", "removed", "away")
    mul_words: tuple[str, ...] = ("each", "per", "every", "times", "double", "twice", "triple")
    div_words: tuple[str, ...] = ("average", "split", "share", "divided", "per")
    ratio_words: tuple[str, ...] = ("ratio", "half", "twice", "double", "triple")
    compare_words: tuple[str, ...] = ("difference", "more than", "less than", "fewer than")
    capacity_words: tuple[str, ...] = (
        "at most",
        "minimum number",
        "capacity",
        "holds",
        "fit",
        "required",
    )


ROLE_CFG = _RoleCfg()


def _safe_float(text: str) -> float | None:
    try:
        return float(text)
    except ValueError:
        return None


def _sentences(text: str) -> list[str]:
    parts = [p.strip() for p in SENTENCE_SPLIT_RE.split(text.strip()) if p.strip()]
    return parts if parts else [text.strip()]


def _local_context(sentence: str, match_start: int, match_end: int, window: int = 28) -> str:
    lo = max(0, match_start - window)
    hi = min(len(sentence), match_end + window)
    return sentence[lo:hi]


def _find_unit_hint(context: str) -> str | None:
    m = UNIT_HINT_RE.search(context)
    return None if not m else m.group(0).lower()


def _find_nearby_verbs(context: str) -> list[str]:
    return [m.group(0).lower() for m in VERB_RE.finditer(context)]


def _token_span(sentence: str, start: int, end: int) -> tuple[int, int] | None:
    spans: list[tuple[int, int]] = []
    for m in TOKEN_RE.finditer(sentence):
        spans.append((m.start(), m.end()))
    first = None
    last = None
    for i, (s, e) in enumerate(spans):
        if s <= start < e and first is None:
            first = i
        if s < end <= e:
            last = i
            break
    if first is None or last is None:
        return None
    return (first, last)


def extract_problem_numbers(question_text: str) -> list[dict[str, Any]]:
    """Extract number-like quantities from a question with lightweight metadata."""
    records: list[dict[str, Any]] = []
    for s_idx, sent in enumerate(_sentences(question_text)):
        for m in DIGIT_RE.finditer(sent):
            text = m.group(0)
            value = _safe_float(text)
            context = _local_context(sent, m.start(), m.end())
            records.append(
                {
                    "surface_text": text,
                    "normalized_value": value,
                    "source_type": "numeric_literal",
                    "sentence_index": s_idx,
                    "local_context": context,
                    "token_span": _token_span(sent, m.start(), m.end()),
                    "unit_hint": _find_unit_hint(context),
                    "nearby_verbs": _find_nearby_verbs(context),
                }
            )

        for m in TOKEN_RE.finditer(sent):
            token = m.group(0).lower()
            if token in NUMBER_WORDS:
                context = _local_context(sent, m.start(), m.end())
                records.append(
                    {
                        "surface_text": token,
                        "normalized_value": float(NUMBER_WORDS[token]),
                        "source_type": "number_word",
                        "sentence_index": s_idx,
                        "local_context": context,
                        "token_span": _token_span(sent, m.start(), m.end()),
                        "unit_hint": _find_unit_hint(context),
                        "nearby_verbs": _find_nearby_verbs(context),
                    }
                )
            elif token in MULT_WORDS:
                context = _local_context(sent, m.start(), m.end())
                source = "fraction_word" if token == "half" else "multiplicative_word"
                records.append(
                    {
                        "surface_text": token,
                        "normalized_value": float(MULT_WORDS[token]),
                        "source_type": source,
                        "sentence_index": s_idx,
                        "local_context": context,
                        "token_span": _token_span(sent, m.start(), m.end()),
                        "unit_hint": _find_unit_hint(context),
                        "nearby_verbs": _find_nearby_verbs(context),
                    }
                )

    return records


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(w in lowered for w in words)


def _global_target_relation(question_text: str) -> str:
    q = question_text.lower()
    if any(w in q for w in ("left", "remaining", "remain")):
        return "contributes_to_remaining"
    if any(w in q for w in ("total", "altogether", "in all")):
        return "contributes_to_total"
    if any(w in q for w in ("per", "each", "every", "hour", "minute")):
        return "contributes_to_rate"
    if any(w in q for w in ("minimum number", "at most", "required", "capacity", "holds")):
        return "contributes_to_capacity"
    return "unknown"


def assign_number_roles(question_text: str) -> list[dict[str, Any]]:
    """Assign expected semantic roles to extracted numbers."""
    records = extract_problem_numbers(question_text)
    target_relation_global = _global_target_relation(question_text)

    assigned: list[dict[str, Any]] = []
    for rec in records:
        ctx = rec["local_context"].lower()

        role = "unknown"
        confidence = "low"
        target_relation = target_relation_global
        strong_required = False

        if _contains_any(ctx, ROLE_CFG.capacity_words):
            role = "capacity_ceiling"
            confidence = "high"
            target_relation = "contributes_to_capacity"
            strong_required = True
        elif _contains_any(ctx, ROLE_CFG.sub_words):
            role = "subtract"
            confidence = "high"
            target_relation = (
                "contributes_to_remaining"
                if target_relation_global == "contributes_to_remaining"
                else target_relation_global
            )
            strong_required = True
        elif _contains_any(ctx, ROLE_CFG.add_words):
            role = "add"
            confidence = "medium"
        elif _contains_any(ctx, ROLE_CFG.ratio_words):
            role = "ratio"
            confidence = "medium"
            strong_required = True
        elif _contains_any(ctx, ROLE_CFG.mul_words):
            role = "multiply"
            confidence = "medium"
        elif _contains_any(ctx, ROLE_CFG.div_words):
            role = "divide"
            confidence = "low"
        elif _contains_any(ctx, ROLE_CFG.compare_words):
            role = "compare_difference"
            confidence = "medium"

        required = role not in {"irrelevant", "unknown"}
        if rec["source_type"] in {"fraction_word", "multiplicative_word"}:
            required = True
            strong_required = True

        # keep remaining questions broad, but avoid forcing all digits as strict requirements
        if (
            target_relation_global == "contributes_to_remaining"
            and rec["source_type"] == "numeric_literal"
        ):
            if role == "unknown":
                role = "subtract"
                confidence = "low"
            required = True

        if role == "unknown" and rec["unit_hint"] is None and rec["source_type"] == "number_word":
            required = False

        if not required:
            target_relation = "contributes_to_intermediate" if role != "unknown" else "unknown"

        assigned.append(
            {
                **rec,
                "expected_role": role,
                "required_for_final_answer": bool(required),
                "strongly_required_for_final_answer": bool(strong_required or confidence == "high"),
                "role_confidence": confidence,
                "target_relation": target_relation,
            }
        )

    return assigned


def _normalize_num_for_match(value: float | None) -> set[str]:
    if value is None:
        return set()
    if float(value).is_integer():
        iv = int(value)
        return {str(iv), f"{iv}.0"}
    return {str(value)}


def _has_finalization_cue(reasoning_lower: str) -> bool:
    return any(
        c in reasoning_lower
        for c in ("final answer", "therefore", "answer:", "so the answer")
    )


def _answer_looks_count_like(question_lower: str, parsed_answer: str | None) -> bool:
    if not parsed_answer:
        return False
    if not re.fullmatch(r"-?\d+(?:\.\d+)?", parsed_answer.strip()):
        return False
    value = float(parsed_answer)
    asks_count = any(c in question_lower for c in ("how many", "number of", "minimum number"))
    if asks_count and value == int(value) and value >= 0:
        return True
    return False


def compute_role_coverage_features(
    question_text: str,
    reasoning_text: str,
    parsed_answer: str | None = None,
) -> dict[str, Any]:
    """Compute role-coverage features from question numbers vs reasoning text."""
    assigned = assign_number_roles(question_text)
    required = [r for r in assigned if bool(r["required_for_final_answer"])]
    strongly_required = [
        r for r in required if bool(r.get("strongly_required_for_final_answer", False))
    ]

    reasoning_lower = reasoning_text.lower()
    reasoning_numbers = {m.group(0) for m in DIGIT_RE.finditer(reasoning_text)}

    detected_required = 0
    detected_strong_required = 0
    missing_by_role: dict[str, int] = defaultdict(int)

    for rec in required:
        variants = _normalize_num_for_match(rec["normalized_value"])
        variants.add(str(rec["surface_text"]).lower())
        found = any(v.lower() in reasoning_lower or v in reasoning_numbers for v in variants if v)
        if found:
            detected_required += 1
            if rec.get("strongly_required_for_final_answer", False):
                detected_strong_required += 1
        else:
            missing_by_role[str(rec["expected_role"])] += 1

    num_required = len(required)
    num_strong_required = len(strongly_required)
    missing_required = max(0, num_required - detected_required)
    missing_strong_required = max(0, num_strong_required - detected_strong_required)
    missing_fraction = 0.0 if num_required == 0 else missing_required / num_required

    required_sub_missing = missing_by_role.get("subtract", 0) > 0
    required_add_missing = missing_by_role.get("add", 0) > 0
    required_rate_missing = (
        missing_by_role.get("multiply", 0) > 0
        or missing_by_role.get("divide", 0) > 0
        or missing_by_role.get("ratio", 0) > 0
    )
    required_capacity_missing = missing_by_role.get("capacity_ceiling", 0) > 0

    question_lower = question_text.lower()
    has_final_cue = _has_finalization_cue(reasoning_lower)
    parsed_present = bool(parsed_answer and parsed_answer.strip())
    target_remaining = any(c in question_lower for c in ("left", "remaining", "remain"))
    coherent_answer_type = _answer_looks_count_like(question_lower, parsed_answer)

    possible_intermediate_stop = bool(
        missing_strong_required > 0
        and parsed_present
        and (has_final_cue or len(reasoning_numbers) > 0)
        and (target_remaining or required_sub_missing)
    )

    all_required_covered = (num_required == 0) or (missing_required == 0)
    coverage = 1.0 if num_required == 0 else detected_required / num_required
    score = max(0.0, min(1.0, coverage - (0.20 if possible_intermediate_stop else 0.0)))

    signals: list[str] = []
    if missing_required > 0:
        signals.append("missing_required_number")
    if required_sub_missing:
        signals.append("required_subtractive_number_missing")
    if required_add_missing:
        signals.append("required_additive_number_missing")
    if required_rate_missing:
        signals.append("required_rate_number_missing")
    if required_capacity_missing:
        signals.append("required_capacity_number_missing")
    if possible_intermediate_stop:
        signals.append("possible_intermediate_stop_suspected")

    return {
        "num_extracted_numbers": len(assigned),
        "num_required_numbers": num_required,
        "num_strong_required_numbers": num_strong_required,
        "num_required_numbers_detected_in_reasoning": detected_required,
        "num_strong_required_numbers_detected_in_reasoning": detected_strong_required,
        "missing_required_number_count": missing_required,
        "missing_strong_required_number_count": missing_strong_required,
        "missing_required_number_fraction": missing_fraction,
        "required_subtractive_number_missing": required_sub_missing,
        "required_additive_number_missing": required_add_missing,
        "required_rate_number_missing": required_rate_missing,
        "required_capacity_number_missing": required_capacity_missing,
        "possible_intermediate_stop_suspected": possible_intermediate_stop,
        "all_required_numbers_covered": all_required_covered,
        "role_coverage_score": score,
        "role_coverage_triggered_signals": signals,
        "coherent_answer_type": coherent_answer_type,
        "has_finalization_cue": has_final_cue,
        "assigned_number_roles": assigned,
    }


def compute_calibrated_role_decision(
    question_text: str,
    reasoning_text: str,
    parsed_answer: str | None = None,
) -> dict[str, Any]:
    """Calibrate raw role signals into weak/moderate/strong escalation guidance."""
    feats = compute_role_coverage_features(
        question_text,
        reasoning_text,
        parsed_answer=parsed_answer,
    )

    warning_score = 0
    strong_error_score = 0
    strength_labels: dict[str, str] = {}

    if feats["missing_strong_required_number_count"] > 0:
        strong_error_score += 2
        warning_score += 1
        strength_labels["missing_strong_required_number"] = "strong_signal"
    elif feats["missing_required_number_count"] > 0:
        warning_score += 1
        strength_labels["missing_required_number"] = "weak_signal"

    if feats["required_subtractive_number_missing"]:
        strong_error_score += 2
        warning_score += 1
        strength_labels["required_subtractive_number_missing"] = "strong_signal"

    if feats["required_capacity_number_missing"]:
        strong_error_score += 2
        warning_score += 1
        strength_labels["required_capacity_number_missing"] = "strong_signal"

    if feats["required_rate_number_missing"] and feats["missing_strong_required_number_count"] > 0:
        warning_score += 1
        strength_labels["required_rate_number_missing"] = "medium_signal"

    if feats["possible_intermediate_stop_suspected"]:
        strong_error_score += 1
        warning_score += 1
        strength_labels["possible_intermediate_stop_suspected"] = "strong_signal"

    # downweight for coherent, finalized answers with high coverage
    if (
        feats["coherent_answer_type"]
        and feats["has_finalization_cue"]
        and feats["role_coverage_score"] >= 0.7
    ):
        warning_score = max(0, warning_score - 1)
        if strong_error_score == 0:
            strength_labels["coherent_finalized_answer"] = "weakening_factor"

    decision = "no_escalation"
    escalation_recommended = False
    if strong_error_score >= 4:
        decision = "strong_escalation_candidate"
        escalation_recommended = True
    elif strong_error_score >= 3 and warning_score >= 2:
        decision = "maybe_escalate"
        escalation_recommended = True

    return {
        "role_warning_score": warning_score,
        "role_strong_error_score": strong_error_score,
        "signal_strength_labels": strength_labels,
        "calibrated_decision": decision,
        "escalation_recommended": escalation_recommended,
        "role_features": feats,
    }
