"""Build real routing rows: reasoning_greedy + direct_plus_revise + engineered features.

Supports GSM8K, MATH500, or a caller-provided query list (e.g. hard subset).
Checkpoints after each query (JSONL append + CSV rewrite).
"""

from __future__ import annotations

import csv
import json
import os
import traceback
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Literal

from src.datasets.gsm8k import Query, load_gsm8k
from src.datasets.math500 import load_math500
from src.evaluation.strategy_expansion_eval import (
    run_direct_plus_revise,
    run_reasoning_then_revise_review_only,
)
from src.features.calibration_features import extract_calibration_features
from src.features.constraint_violation_features import extract_constraint_violation_features
from src.features.number_role_features import compute_calibrated_role_decision
from src.features.precompute_features import extract_first_pass_features, extract_query_features
from src.features.self_verification_features import extract_self_verification_features
from src.features.step_verification_features import extract_step_verification_features
from src.features.target_quantity_features import extract_target_quantity_features
from src.features.unified_error_signal import compute_unified_error_signal
from src.models.openai_llm import OpenAILLMModel
from src.policies.adaptive_policy_v6 import compute_v6_scores
from src.policies.adaptive_policy_v7 import compute_v7_scores
from src.utils.answer_extraction import (
    extract_math_answer,
    extract_numeric_answer,
    normalize_math_answer,
)


def _norm_numeric(value: str) -> str:
    candidate = str(value).strip().replace(",", "").replace("$", "").rstrip(".")
    try:
        number = Decimal(candidate)
        normalized = format(number.normalize(), "f")
        if "." in normalized:
            normalized = normalized.rstrip("0").rstrip(".")
        return normalized or "0"
    except InvalidOperation:
        return candidate.casefold()


def _answers_match_numeric(gold: str, predicted: str) -> bool:
    if not predicted or not str(predicted).strip():
        return False
    g, p = _norm_numeric(gold), _norm_numeric(predicted)
    try:
        Decimal(g)
        Decimal(p)
        return g == p
    except InvalidOperation:
        return g == p


def _answers_match_math(gold: str, predicted: str) -> bool:
    if not predicted or not str(predicted).strip():
        return False
    g = normalize_math_answer(gold)
    p = normalize_math_answer(predicted)
    return bool(g) and g == p


def _predicted_from_reasoning_raw(raw: str, mode: Literal["numeric", "math"]) -> str:
    if mode == "math":
        parsed = extract_math_answer(raw).strip()
        return normalize_math_answer(parsed) if parsed else ""
    return _norm_numeric(extract_numeric_answer(raw))


def _predicted_from_revise_outputs(raw_outputs: list[Any], mode: Literal["numeric", "math"]) -> str:
    last = str(raw_outputs[-1]) if raw_outputs else ""
    if mode == "math":
        parsed = extract_math_answer(last).strip()
        return normalize_math_answer(parsed) if parsed else ""
    return _norm_numeric(extract_numeric_answer(last))


def _flatten_features(prefix: str, d: dict[str, Any], out: dict[str, float]) -> None:
    for k, v in d.items():
        key = f"{prefix}_{k}" if prefix else k
        if isinstance(v, bool):
            out[key] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[key] = float(v)
        elif v is None:
            out[key] = 0.0
        elif isinstance(v, dict):
            _flatten_features(key, v, out)


def _numeric_feature_row(
    question: str,
    reasoning_raw: str,
    parsed_reasoning: str,
) -> dict[str, float]:
    out: dict[str, float] = {}
    _flatten_features("q", extract_query_features(question), out)
    _flatten_features("tq", extract_target_quantity_features(question), out)
    cons = extract_constraint_violation_features(
        question, reasoning_raw, predicted_answer=parsed_reasoning or None
    )
    for k, v in cons.items():
        if k.endswith("_suspected") and isinstance(v, bool):
            out[f"cons_{k}"] = 1.0 if v else 0.0
    role = compute_calibrated_role_decision(question, reasoning_raw, parsed_answer=parsed_reasoning)
    out["role_warning_score"] = float(role["role_warning_score"])
    out["role_strong_error_score"] = float(role["role_strong_error_score"])
    out["calibrated_strong_escalation_candidate"] = (
        1.0 if role["calibrated_decision"] == "strong_escalation_candidate" else 0.0
    )
    self_v = extract_self_verification_features(
        question, reasoning_raw, parsed_answer=parsed_reasoning
    )
    for k, v in self_v.items():
        if isinstance(v, (int, float)):
            out[f"self_{k}"] = float(v)
        elif isinstance(v, bool):
            out[f"self_{k}"] = 1.0 if v else 0.0
    cal = extract_calibration_features(reasoning_raw, parsed_answer=parsed_reasoning)
    for k, v in cal.items():
        if isinstance(v, (int, float)):
            out[f"cal_{k}"] = float(v)
    step = extract_step_verification_features(question, reasoning_raw)
    for k, v in step.items():
        if isinstance(v, (int, float)):
            out[f"step_{k}"] = float(v)
    fp = extract_first_pass_features(question, reasoning_raw, parsed_answer=parsed_reasoning)
    for k, v in fp.items():
        if isinstance(v, bool):
            out[f"fp_{k}"] = 1.0 if v else 0.0
        elif isinstance(v, (int, float)):
            out[f"fp_{k}"] = float(v)
    uni = compute_unified_error_signal(question, reasoning_raw, parsed_answer=parsed_reasoning)
    out["unified_error_score"] = float(uni["unified_error_score"])
    out["unified_confidence_score"] = float(uni["unified_confidence_score"])
    v6 = compute_v6_scores(question, reasoning_raw)
    out["v6_answer_error_score"] = float(v6["answer_error_score"])
    out["v6_explanation_warning_score"] = float(v6["explanation_warning_score"])
    out["v6_final_answer_confident"] = 1.0 if v6["final_answer_confident"] else 0.0
    out["v6_revise_recommended"] = 1.0 if v6["revise_recommended"] else 0.0
    v7 = compute_v7_scores(question, reasoning_raw)
    out["v7_answer_error_score"] = float(v7["answer_error_score"])
    out["v7_extra_answer_error"] = float(v7["v7_extra_answer_error"])
    out["v7_final_answer_confident"] = 1.0 if v7["final_answer_confident"] else 0.0
    out["v7_revise_recommended"] = 1.0 if v7["revise_recommended"] else 0.0
    return out


@dataclass
class BuildConfig:
    """Configuration for building a real routing dataset."""

    subset_size: int
    output_dir: str | Path
    output_dataset_csv: str | Path
    gsm8k_data_file: str | Path | None = None
    dataset: Literal["gsm8k", "math500", "custom"] = "gsm8k"
    math500_data_file: str | Path | None = None
    queries_override: list[Query] | None = None
    answer_match_mode: Literal["numeric", "math"] | None = None
    model_name: str = "gpt-4o-mini"
    max_tokens: int = 512
    timeout: int = 90
    checkpoint_every: int = 1
    bundled_fallback: str | Path | None = None
    summary_filename: str = "gsm8k_subset_run_summary.json"
    per_query_csv_filename: str = "gsm8k_per_query_outputs.csv"
    regime_label: str = ""
    include_reasoning_then_revise: bool = False

    def effective_match_mode(self) -> Literal["numeric", "math"]:
        if self.answer_match_mode is not None:
            return self.answer_match_mode
        return "math" if self.dataset == "math500" else "numeric"


def build_real_routing_dataset(config: BuildConfig) -> dict[str, Any]:
    cfg = config

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_csv = Path(cfg.output_dataset_csv)
    dataset_csv.parent.mkdir(parents=True, exist_ok=True)

    summary_path = out_dir / cfg.summary_filename
    per_query_csv = out_dir / cfg.per_query_csv_filename
    raw_jsonl = out_dir / "raw_responses.jsonl"
    provider_meta = out_dir / "provider_metadata.json"
    checkpoint_path = out_dir / "checkpoint.json"

    match_mode = cfg.effective_match_mode()
    answers_match = _answers_match_math if match_mode == "math" else _answers_match_numeric

    blockers: list[str] = []
    if not os.getenv("OPENAI_API_KEY", "").strip():
        blockers.append("OPENAI_API_KEY missing")

    meta = {
        "provider": "openai",
        "model_name": cfg.model_name,
        "max_tokens": cfg.max_tokens,
        "timeout_seconds": cfg.timeout,
        "subset_size_requested": cfg.subset_size,
        "dataset": cfg.dataset,
        "answer_match_mode": match_mode,
        "regime_label": cfg.regime_label or "",
        "gsm8k_data_file": str(cfg.gsm8k_data_file) if cfg.gsm8k_data_file else None,
        "math500_data_file": str(cfg.math500_data_file) if cfg.math500_data_file else None,
    }
    provider_meta.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    if blockers:
        summary = {
            "run_status": "BLOCKED",
            "evidence_status": "blocked",
            "blockers": blockers,
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "summary": summary,
            "summary_path": str(summary_path),
            "per_query_csv": str(per_query_csv),
            "dataset_csv": str(dataset_csv),
        }

    reasoning_prefix = (
        "Solve this step by step. Put your final answer in \\boxed{...} "
        "or end with 'Final answer: ...'.\n\n"
        if match_mode == "math"
        else "Solve this step by step and end with 'Final answer: <number>'.\n\n"
    )
    revise_model_prefix = (
        "Answer the following math question. Give only the final answer; "
        "use \\boxed{...} when appropriate.\n\n"
        if match_mode == "math"
        else "Answer the following math question. Give only the final numeric answer."
    )

    queries: list[Query] = []
    data_source = "unknown"

    try:
        if cfg.queries_override is not None:
            queries = list(cfg.queries_override)[: cfg.subset_size]
            data_source = "queries_override"
        elif cfg.dataset == "math500":
            if cfg.math500_data_file and Path(cfg.math500_data_file).exists():
                queries = load_math500(
                    max_samples=cfg.subset_size,
                    data_file=cfg.math500_data_file,
                )
                data_source = "local_math500_jsonl"
            else:
                queries = load_math500(max_samples=cfg.subset_size, cache_dir="data")
                data_source = "huggingface_math500"
        else:
            bundled = Path(
                cfg.bundled_fallback
                or (
                    Path(__file__).resolve().parent.parent
                    / "datasets"
                    / "bundled"
                    / "gsm8k_test_sample.json"
                )
            )
            if cfg.gsm8k_data_file and Path(cfg.gsm8k_data_file).exists():
                queries = load_gsm8k(
                    split="test",
                    max_samples=cfg.subset_size,
                    data_file=cfg.gsm8k_data_file,
                )
                data_source = "local_json_file"
            elif bundled.exists():
                bundled_queries = load_gsm8k(
                    split="test",
                    max_samples=cfg.subset_size,
                    data_file=str(bundled),
                )
                if len(bundled_queries) >= cfg.subset_size:
                    queries = bundled_queries[: cfg.subset_size]
                    data_source = "bundled_gsm8k_test_sample"
                else:
                    queries = load_gsm8k(
                        split="test",
                        max_samples=cfg.subset_size,
                        cache_dir="data",
                    )
                    data_source = "huggingface_openai_gsm8k_test"
            else:
                queries = load_gsm8k(
                    split="test",
                    max_samples=cfg.subset_size,
                    cache_dir="data",
                )
                data_source = "huggingface_openai_gsm8k_test"
    except Exception as exc:
        summary = {
            "run_status": "BLOCKED",
            "evidence_status": "blocked",
            "blockers": [f"load_queries failed: {exc}"],
        }
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return {
            "summary": summary,
            "summary_path": str(summary_path),
            "per_query_csv": str(per_query_csv),
            "dataset_csv": str(dataset_csv),
        }

    n = min(len(queries), cfg.subset_size)
    queries = queries[:n]

    model_r = OpenAILLMModel(
        model_name=cfg.model_name,
        greedy_temperature=0.0,
        sample_temperature=0.0,
        max_tokens=cfg.max_tokens,
        timeout_seconds=float(cfg.timeout),
        prompt_prefix=reasoning_prefix,
    )
    model_rev = OpenAILLMModel(
        model_name=cfg.model_name,
        greedy_temperature=0.0,
        sample_temperature=0.0,
        max_tokens=cfg.max_tokens,
        timeout_seconds=float(cfg.timeout),
        prompt_prefix=revise_model_prefix,
    )
    rtr_prefix = (
        "You verify step-by-step math reasoning. "
        "Check the reasoning and final answer carefully. If incorrect, fix it. "
        "If correct, return the same answer."
    )
    model_rtr = (
        OpenAILLMModel(
            model_name=cfg.model_name,
            greedy_temperature=0.0,
            sample_temperature=0.0,
            max_tokens=cfg.max_tokens,
            timeout_seconds=float(cfg.timeout),
            prompt_prefix=rtr_prefix,
        )
        if cfg.include_reasoning_then_revise
        else None
    )

    start_idx = 0
    completed: list[dict[str, Any]] = []
    if checkpoint_path.exists():
        try:
            ck = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            start_idx = int(ck.get("next_index", 0))
            if raw_jsonl.exists() and start_idx > 0:
                for line in raw_jsonl.read_text(encoding="utf-8").splitlines():
                    if line.strip():
                        completed.append(json.loads(line))
        except (json.JSONDecodeError, ValueError, OSError):
            start_idx = 0
            completed = []

    _CSV_SKIP_FULL = frozenset({"revise_raw_json", "reasoning_then_revise_raw_json", "error_trace"})
    _REASONING_CSV_MAX = 12000

    def _rewrite_outputs() -> None:
        if not completed:
            return
        fieldnames = [k for k in completed[0] if k not in _CSV_SKIP_FULL]
        with per_query_csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
            w.writeheader()
            for row in completed:
                out_row = {k: row.get(k, "") for k in fieldnames}
                rr = str(out_row.get("reasoning_raw", ""))
                if len(rr) > _REASONING_CSV_MAX:
                    out_row["reasoning_raw"] = rr[:_REASONING_CSV_MAX] + "\n...[truncated]"
                w.writerow(out_row)

        ds_rows: list[dict[str, Any]] = []
        for row in completed:
            if row.get("status") != "ok":
                continue
            base = {
                k: row[k]
                for k in row
                if not k.startswith("feat_") and k not in _CSV_SKIP_FULL
            }
            rr = str(base.get("reasoning_raw", ""))
            if len(rr) > _REASONING_CSV_MAX:
                base["reasoning_raw"] = rr[:_REASONING_CSV_MAX] + "\n...[truncated]"
            for k, v in row.items():
                if k.startswith("feat_"):
                    base[k[5:]] = v
            ds_rows.append(base)
        if ds_rows:
            df_names = list(ds_rows[0].keys())
            with dataset_csv.open("w", newline="", encoding="utf-8") as fh:
                w = csv.DictWriter(fh, fieldnames=df_names, extrasaction="ignore")
                w.writeheader()
                w.writerows(ds_rows)

    for i in range(start_idx, len(queries)):
        q = queries[i]
        row: dict[str, Any] = {
            "question_id": q.id,
            "question": q.question,
            "gold_answer": q.answer,
            "status": "pending",
        }
        try:
            reasoning_raw = model_r.generate(q.question)
            reasoning_ans = _predicted_from_reasoning_raw(reasoning_raw, match_mode)
            parsed_r = extract_math_answer(reasoning_raw).strip() or reasoning_ans

            dr = run_direct_plus_revise(model_rev, q.question)
            revise_raw: list[Any] = list(dr["raw_outputs"])
            revise_ans = _predicted_from_revise_outputs(revise_raw, match_mode)

            r_ok = answers_match(q.answer, reasoning_ans)
            v_ok = answers_match(q.answer, revise_ans)
            helpful = 1 if (not r_ok and v_ok) else 0

            rtr_ans = ""
            rtr_correct = 0
            rtr_helpful = 0
            rtr_raw_json = ""
            if model_rtr is not None:
                rtr = run_reasoning_then_revise_review_only(
                    model_rtr,
                    q.question,
                    reasoning_raw,
                    mode=match_mode,
                )
                rtr_ans = str(rtr.get("predicted_answer", ""))
                rtr_raw_json = json.dumps(rtr["raw_outputs"], ensure_ascii=False)
                rtr_correct = int(answers_match(q.answer, rtr_ans))
                rtr_helpful = int((not r_ok) and bool(rtr_correct))

            feat = _numeric_feature_row(q.question, reasoning_raw, parsed_r)
            for fk, fv in feat.items():
                row[f"feat_{fk}"] = fv

            upd: dict[str, Any] = {
                "status": "ok",
                "reasoning_raw": reasoning_raw,
                "revise_raw_json": json.dumps(revise_raw, ensure_ascii=False),
                "reasoning_answer": reasoning_ans,
                "revise_answer": revise_ans,
                "reasoning_correct": int(r_ok),
                "revise_correct": int(v_ok),
                "revise_helpful": helpful,
                "reasoning_cost": 1,
                "revise_cost": 2,
                "reasoning_raw_chars": len(reasoning_raw),
                "revise_num_calls": len(revise_raw),
            }
            if model_rtr is not None:
                upd["reasoning_then_revise_answer"] = rtr_ans
                upd["reasoning_then_revise_correct"] = rtr_correct
                upd["reasoning_then_revise_helpful"] = rtr_helpful
                upd["reasoning_then_revise_raw_json"] = rtr_raw_json
                upd["reasoning_then_revise_cost"] = 2
            row.update(upd)
        except Exception as exc:
            row["status"] = "error"
            row["error_message"] = str(exc)
            row["error_trace"] = traceback.format_exc()[-2000:]

        completed.append(row)
        with raw_jsonl.open("a", encoding="utf-8") as jf:
            jf.write(json.dumps(row, ensure_ascii=False) + "\n")

        _rewrite_outputs()
        checkpoint_path.write_text(
            json.dumps({"next_index": i + 1, "total": len(queries)}, indent=2),
            encoding="utf-8",
        )

    ok_rows = [r for r in completed if r.get("status") == "ok"]
    err_rows = [r for r in completed if r.get("status") == "error"]
    helpful_count = sum(int(r.get("revise_helpful", 0)) for r in ok_rows)
    rtr_acc = None
    rtr_helpful_rate = None
    if cfg.include_reasoning_then_revise and ok_rows:
        rtr_acc = sum(int(r.get("reasoning_then_revise_correct", 0)) for r in ok_rows) / len(
            ok_rows
        )
        rtr_h = sum(int(r.get("reasoning_then_revise_helpful", 0)) for r in ok_rows)
        rtr_helpful_rate = rtr_h / len(ok_rows)

    summary = {
        "run_status": "COMPLETED" if len(ok_rows) == len(queries) else "PARTIAL",
        "evidence_status": "measured_now",
        "provider": "openai",
        "model_name": cfg.model_name,
        "dataset": cfg.dataset,
        "answer_match_mode": match_mode,
        "data_source": data_source,
        "regime_label": cfg.regime_label or "",
        "num_queries_requested": len(queries),
        "num_queries_ok": len(ok_rows),
        "num_queries_error": len(err_rows),
        "revise_helpful_count": helpful_count,
        "revise_helpful_rate": helpful_count / max(1, len(ok_rows)),
        "reasoning_accuracy": sum(int(r["reasoning_correct"]) for r in ok_rows)
        / max(1, len(ok_rows)),
        "revise_accuracy": sum(int(r["revise_correct"]) for r in ok_rows)
        / max(1, len(ok_rows)),
        "reasoning_then_revise_accuracy": rtr_acc,
        "reasoning_then_revise_helpful_rate": rtr_helpful_rate,
        "include_reasoning_then_revise": cfg.include_reasoning_then_revise,
        "paths": {
            "summary_json": str(summary_path),
            "per_query_csv": str(per_query_csv),
            "dataset_csv": str(dataset_csv),
            "raw_jsonl": str(raw_jsonl),
            "provider_metadata": str(provider_meta),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    return {
        "summary": summary,
        "summary_path": str(summary_path),
        "per_query_csv": str(per_query_csv),
        "dataset_csv": str(dataset_csv),
    }
