#!/usr/bin/env python3
"""End-to-end recent baselines + oracle + routing sweep; writes outputs/recent_baselines/.

Requires OPENAI_API_KEY and network for OpenAI + HuggingFace dataset downloads.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import traceback
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from src.datasets.aime2024 import load_aime2024  # noqa: E402
from src.datasets.gsm8k import load_gsm8k  # noqa: E402
from src.datasets.hard_gsm8k import load_hard_gsm8k  # noqa: E402
from src.datasets.math500 import load_math500  # noqa: E402
from src.evaluation.recent_baselines_eval import (  # noqa: E402
    STATIC_LADDER_BASELINES,
    compute_oracle_summaries,
    evaluate_routing_baselines,
    evaluate_static_ladder,
    hard_gsm8k_validation_summary,
)
from src.models.openai_llm import OpenAILLMModel  # noqa: E402


def _ensure_api_key() -> str | None:
    if not os.getenv("OPENAI_API_KEY"):
        return None
    return "ok"


def _build_model(model_name: str, max_tokens: int) -> OpenAILLMModel:
    return OpenAILLMModel(
        model_name=model_name,
        max_tokens=max_tokens,
        timeout_seconds=120.0,
    )


def _reasoning_first_pass_map(static_payload: dict) -> dict[str, dict[str, str]]:
    rows = static_payload["per_baseline_results"].get("reasoning_greedy", [])
    out: dict[str, dict[str, str]] = {}
    for r in rows:
        raw = (r.get("candidates") or [""])[0] if r.get("candidates") else ""
        out[r["query_id"]] = {
            "first_raw": raw,
            "first_pred": str(r.get("final_answer", "")),
        }
    return out


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def _dataset_blocker(name: str, source: str, exc: BaseException) -> dict:
    return {
        "dataset": name,
        "source_attempted": source,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
        "fix": _fix_hint(exc),
    }


def _fix_hint(exc: BaseException) -> str:
    msg = str(exc).lower()
    if "openai" in msg or "401" in msg or "api key" in msg:
        return "Set OPENAI_API_KEY and ensure billing/model access for the chosen model."
    if "gated" in msg or "authenticated" in msg:
        return "Run huggingface-cli login (HF_TOKEN) for gated datasets."
    if "couldn't connect" in msg or "connection" in msg or "network" in msg:
        return "Check internet connectivity and retry."
    return "See error_message; fix upstream (dataset id, split, or local data_file)."


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="outputs/recent_baselines")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--gsm8k-slice", type=int, default=100)
    parser.add_argument("--hard-k", type=int, default=100)
    parser.add_argument("--math500-max", type=int, default=100)
    parser.add_argument(
        "--aime-max",
        type=int,
        default=None,
        help="Cap AIME queries (default: all rows in math-ai/aime24).",
    )
    parser.add_argument(
        "--with-routing",
        action="store_true",
        default=False,
        help=(
            "Run threshold routing sweeps (many extra API calls). "
            "Default: write stub routing JSON only."
        ),
    )
    parser.add_argument(
        "--routing-max-queries",
        type=int,
        default=40,
        help="Cap queries for routing sweeps (full static ladder still uses all queries).",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    blockers: list[dict] = []
    rollup_rows: list[dict[str, str]] = []
    dataset_summaries: list[dict[str, str]] = []

    if _ensure_api_key() is None:
        blocker = {
            "dataset": "_global",
            "source_attempted": "OpenAI API",
            "error_type": "MissingEnv",
            "error_message": "OPENAI_API_KEY is not set in the environment",
            "fix": "Export OPENAI_API_KEY before running this script.",
        }
        blockers.append(blocker)
        _write_json(
            out_dir / "_blockers.json",
            {"blockers": blockers, "note": "Experiment did not run."},
        )
        print("BLOCKED: OPENAI_API_KEY missing", file=sys.stderr)
        return 2

    try:
        model = _build_model(args.model, args.max_tokens)
    except Exception as exc:
        blockers.append(
            {
                "dataset": "_global",
                "source_attempted": "OpenAILLMModel",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "fix": _fix_hint(exc),
            }
        )
        _write_json(out_dir / "_blockers.json", {"blockers": blockers})
        print(f"BLOCKED: model init: {exc}", file=sys.stderr)
        return 2

    # --- GPQA probe (expected blocked without auth) ---
    try:
        from datasets import load_dataset

        load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train", cache_dir="data")
    except Exception as exc:
        blockers.append(
            {
                "dataset": "gpqa_diamond",
                "source_attempted": "HuggingFace Idavidrein/gpqa config gpqa_diamond",
                "error_type": type(exc).__name__,
                "error_message": str(exc),
                "fix": (
                    "Accept the dataset agreement on the Hub and run "
                    "huggingface-cli login with a token that has access."
                ),
            }
        )

    def run_one_dataset(
        key: str,
        queries: list,
        mode: str,
        static_path: Path,
        oracle_path: Path,
        routing_path: Path | None,
    ) -> None:
        nonlocal rollup_rows, dataset_summaries
        routing = None
        print(f"\n=== Dataset {key} ({len(queries)} queries, mode={mode}) ===")
        static = evaluate_static_ladder(model, queries, mode)  # type: ignore[arg-type]
        static_out = {
            "dataset": key,
            "answer_mode": mode,
            "baselines": STATIC_LADDER_BASELINES,
            "summaries": static["summaries"],
            "per_baseline_results": static["per_baseline_results"],
        }
        _write_json(static_path, static_out)

        oracle = compute_oracle_summaries(static["per_baseline_results"])
        oracle_out = {"dataset": key, **oracle}
        _write_json(oracle_path, oracle_out)

        if routing_path and args.with_routing:
            pq = _reasoning_first_pass_map(static_out)
            n_route = min(len(queries), max(1, args.routing_max_queries))
            q_sub = queries[:n_route]
            routing = evaluate_routing_baselines(model, q_sub, mode, pq)  # type: ignore[arg-type]
            routing["routing_subset_size"] = n_route
            routing["total_dataset_size"] = len(queries)
            _write_json(routing_path, {"dataset": key, **routing})
        elif routing_path:
            _write_json(
                routing_path,
                {
                    "dataset": key,
                    "skipped": True,
                    "reason": (
                        "Pass --with-routing to run threshold routing sweeps "
                        "(extra API calls)."
                    ),
                },
            )

        # Rollup CSV rows
        for bname, summ in static["summaries"].items():
            notes = []
            if summ.get("self_consistency_ambiguity_rate") is not None:
                notes.append(f"sc_ambiguity={summ['self_consistency_ambiguity_rate']:.4f}")
            rollup_rows.append(
                {
                    "dataset": key,
                    "baseline_name": bname,
                    "accuracy": f"{summ['accuracy']:.6f}",
                    "avg_cost": f"{summ['avg_cost_proxy']:.6f}",
                    "extra_compute_rate": f"{summ.get('extra_compute_or_revise_rate', 0.0):.6f}",
                    "revise_helpful_rate_if_defined": ""
                    if summ.get("revise_helpful_rate") is None
                    else f"{summ['revise_helpful_rate']:.6f}",
                    "notes": "; ".join(notes),
                }
            )

        rg = static["summaries"]["reasoning_greedy"]["accuracy"]
        best_static = max(
            static["summaries"].items(),
            key=lambda x: (x[1]["accuracy"], -x[1]["avg_cost_proxy"]),
        )
        best_route = None
        if routing_path and routing is not None:
            rp = routing
            for label in (
                "baseline_A_confidence_difficulty_threshold",
                "baseline_B_output_aware",
                "baseline_C_best_route_inspired_ladder",
            ):
                pt = rp.get(label, {}).get("best_accuracy_point")
                if pt:
                    cand = (label, pt.get("accuracy", 0), pt.get("avg_cost_proxy", 0))
                    if best_route is None or cand[1] > best_route[1]:
                        best_route = cand
        ds_note = ""
        oracle_acc = oracle["multi_action_oracle"]["oracle_accuracy"]
        oracle_cost = oracle["multi_action_oracle"]["oracle_avg_cost_proxy"]
        dataset_summaries.append(
            {
                "dataset": key,
                "one_shot_accuracy": f"{rg:.6f}",
                "best_static_baseline": best_static[0],
                "best_static_accuracy": f"{best_static[1]['accuracy']:.6f}",
                "best_static_cost": f"{best_static[1]['avg_cost_proxy']:.6f}",
                "oracle_accuracy": f"{oracle_acc:.6f}",
                "oracle_cost": f"{oracle_cost:.6f}",
                "best_routing_baseline": "" if best_route is None else best_route[0],
                "best_routing_accuracy": "" if best_route is None else f"{best_route[1]:.6f}",
                "best_routing_cost": "" if best_route is None else f"{best_route[2]:.6f}",
                "blocker_notes": "routing_skipped_default" if not args.with_routing else ds_note,
            }
        )

    # --- GSM8K 100 ---
    try:
        gsm_easy = load_gsm8k(split="test", max_samples=args.gsm8k_slice)
        run_one_dataset(
            "gsm8k100",
            gsm_easy,
            "numeric",
            out_dir / "gsm8k100_static_compute_summary.json",
            out_dir / "gsm8k100_oracle_summary.json",
            out_dir / "gsm8k100_routing_baseline_summary.json",
        )
    except Exception as exc:
        blockers.append(_dataset_blocker("gsm8k100", "openai/gsm8k", exc))
        traceback.print_exc()

    # --- Hard GSM8K ---
    try:
        hard_q = load_hard_gsm8k(split="test", k=args.hard_k, data_file=None)
        run_one_dataset(
            "hard_gsm8k",
            hard_q,
            "numeric",
            out_dir / "hard_gsm8k_static_compute_summary.json",
            out_dir / "hard_gsm8k_oracle_summary.json",
            out_dir / "hard_gsm8k_routing_baseline_summary.json",
        )
        easy_compare = load_gsm8k(split="test", max_samples=args.gsm8k_slice)
        easy_static_path = out_dir / "gsm8k100_static_compute_summary.json"
        hard_static_path = out_dir / "hard_gsm8k_static_compute_summary.json"
        easy_acc = hard_acc = 0.0
        easy_payload: dict | None = None
        hard_payload: dict | None = None
        if easy_static_path.exists():
            easy_payload = json.loads(easy_static_path.read_text())
            easy_acc = easy_payload["summaries"]["reasoning_greedy"]["accuracy"]
        if hard_static_path.exists():
            hard_payload = json.loads(hard_static_path.read_text())
            hard_acc = hard_payload["summaries"]["reasoning_greedy"]["accuracy"]
        revise_easy = revise_hard = None
        if easy_payload is not None:
            revise_easy = easy_payload["summaries"]["direct_plus_revise"].get(
                "revise_helpful_rate"
            )
        if hard_payload is not None:
            revise_hard = hard_payload["summaries"]["direct_plus_revise"].get(
                "revise_helpful_rate"
            )
        val = hard_gsm8k_validation_summary(
            easy_compare,
            hard_q,
            easy_acc,
            hard_acc,
        )
        val["revise_helpful_rate_easy_slice"] = revise_easy
        val["revise_helpful_rate_hard_slice"] = revise_hard
        _write_json(out_dir / "hard_gsm8k_validation_summary.json", val)
    except Exception as exc:
        blockers.append(_dataset_blocker("hard_gsm8k", "derived from openai/gsm8k", exc))
        traceback.print_exc()

    # --- MATH500 ---
    try:
        math_q = load_math500(max_samples=args.math500_max)
        run_one_dataset(
            "math500",
            math_q,
            "math",
            out_dir / "math500_static_compute_summary.json",
            out_dir / "math500_oracle_summary.json",
            out_dir / "math500_routing_baseline_summary.json",
        )
    except Exception as exc:
        blockers.append(_dataset_blocker("math500", "HuggingFaceH4/MATH-500", exc))
        traceback.print_exc()

    # --- AIME 2024 (math-ai/aime24) ---
    try:
        aime_q = load_aime2024(max_samples=args.aime_max)
        run_one_dataset(
            "aime2024",
            aime_q,
            "math",
            out_dir / "aime2024_static_compute_summary.json",
            out_dir / "aime2024_oracle_summary.json",
            out_dir / "aime2024_routing_baseline_summary.json",
        )
    except Exception as exc:
        blockers.append(_dataset_blocker("aime2024", "math-ai/aime24", exc))
        traceback.print_exc()

    if any(b.get("dataset") == "gpqa_diamond" for b in blockers):
        note = next(
            (b["error_message"] for b in blockers if b.get("dataset") == "gpqa_diamond"),
            "blocked",
        )
        dataset_summaries.append(
            {
                "dataset": "gpqa_diamond",
                "one_shot_accuracy": "",
                "best_static_baseline": "",
                "best_static_accuracy": "",
                "best_static_cost": "",
                "oracle_accuracy": "",
                "oracle_cost": "",
                "best_routing_baseline": "",
                "best_routing_accuracy": "",
                "best_routing_cost": "",
                "blocker_notes": note[:500],
            }
        )

    # --- Unified CSVs ---
    out_dir.mkdir(parents=True, exist_ok=True)
    cross_path = out_dir / "final_cross_dataset_baseline_summary.csv"
    if rollup_rows:
        fields = list(rollup_rows[0].keys())
        with cross_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(rollup_rows)

    rollup_path = out_dir / "final_dataset_rollup.csv"
    if dataset_summaries:
        fields = list(dataset_summaries[0].keys())
        with rollup_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fields)
            w.writeheader()
            w.writerows(dataset_summaries)

    _write_json(
        out_dir / "_blockers.json",
        {"blockers": blockers, "completed": True},
    )

    print("\nDone. Outputs under", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
