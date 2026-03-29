#!/usr/bin/env python3
"""Run strong baseline suite: compute ladder, routers, summaries.

Example:
  python3 scripts/run_strong_baselines.py --model dummy --max-samples 30
  python3 scripts/run_strong_baselines.py --config configs/strong_baselines.yaml
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.gsm8k import load_gsm8k
from src.datasets.hard_gsm8k import load_hard_gsm8k
from src.datasets.math500 import load_math500
from src.evaluation.strong_baselines_eval import (
    append_confidence_router_csv,
    build_summary_rows,
    evaluate_best_route_style,
    evaluate_compute_ladder,
    evaluate_confidence_router_curve,
    evaluate_output_router,
    write_best_route_style_json,
    write_compute_ladder_json,
    write_dataset_rollup_csv,
    write_final_summary_csv,
    write_output_router_json,
)
from src.models.dummy import DummyModel
from src.utils.config import load_config


def _load_model(model_type: str, model_cfg: dict) -> object:
    if model_type == "dummy":
        return DummyModel(
            correct_prob=float(model_cfg.get("correct_prob", 0.35)),
            seed=model_cfg.get("seed"),
        )
    if model_type == "openai":
        key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not key:
            raise RuntimeError(
                "BLOCKER: OPENAI_API_KEY missing. Set the environment variable and retry."
            )
        from src.models.openai_llm import OpenAILLM

        return OpenAILLM(
            model_name=model_cfg.get("model", "gpt-4o-mini"),
            temperature=float(model_cfg.get("temperature", 0.7)),
            max_tokens=int(model_cfg.get("max_tokens", 1024)),
        )
    raise ValueError(f"Unknown model type: {model_type}")


def _load_dataset(
    name: str,
    max_samples: int | None,
    cache_dir: str,
    gsm8k_data_file: str | None,
    math500_data_file: str | None,
) -> tuple[str, list, bool]:
    """Returns (dataset_key, queries, use_math_extraction)."""
    if name in ("gsm8k", "gsm8k_test"):
        q = load_gsm8k(
            split="test",
            max_samples=max_samples,
            cache_dir=cache_dir,
            data_file=gsm8k_data_file,
        )
        return "gsm8k", q, False
    if name in ("hard_gsm8k", "gsm8k_hard"):
        q = load_hard_gsm8k(
            split="test",
            max_samples=max_samples if max_samples is not None else 500,
            cache_dir=cache_dir,
            data_file=gsm8k_data_file,
        )
        return "hard_gsm8k", q, False
    if name in ("math500", "math_500"):
        q = load_math500(
            split="test",
            max_samples=max_samples,
            cache_dir=cache_dir,
            data_file=math500_data_file,
        )
        return "math500", q, True
    raise ValueError(f"Unknown dataset: {name}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Strong baselines evaluation suite")
    parser.add_argument("--config", default=None, help="Optional YAML/JSON config")
    parser.add_argument(
        "--model",
        choices=["dummy", "openai"],
        default="dummy",
        help="Model backend (default dummy for CI/smoke)",
    )
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=["gsm8k", "hard_gsm8k", "math500"],
        help="Dataset keys to run",
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/baselines",
        help="Output directory (default outputs/baselines)",
    )
    args = parser.parse_args()

    cfg: dict = {}
    if args.config:
        cfg = load_config(args.config)

    model_type = cfg.get("model", {}).get("type", args.model)
    model_cfg = cfg.get("model", {})
    max_samples = cfg.get("max_samples", args.max_samples)
    datasets = cfg.get("datasets", args.datasets)
    out_dir = Path(cfg.get("out_dir", args.out_dir))
    cache_dir = cfg.get("cache_dir", "data")
    gsm8k_data_file = cfg.get("gsm8k_data_file")
    math500_data_file = cfg.get("math500_data_file")

    try:
        model = _load_model(model_type, model_cfg)
    except Exception as e:
        print("EXACT ERROR:", repr(e))
        traceback.print_exc()
        return 1

    out_dir.mkdir(parents=True, exist_ok=True)
    all_summary_rows: list[dict] = []
    run_log: dict = {"model_type": model_type, "datasets": {}, "blockers": []}

    for ds_name in datasets:
        curve_dpr: list[dict] = []
        curve_sc3: list[dict] = []
        output_payloads: list[dict] = []
        br: dict = {}

        try:
            ds_key, queries, use_math = _load_dataset(
                ds_name, max_samples, cache_dir, gsm8k_data_file, math500_data_file
            )
        except Exception as e:
            print(f"EXACT ERROR loading dataset {ds_name}:", repr(e))
            traceback.print_exc()
            err_s = str(e).lower()
            if "connection" in err_s or "network" in err_s or "timeout" in err_s:
                run_log["blockers"].append("internet / network / timeout")
            elif "401" in err_s or "403" in err_s or "gated" in err_s:
                run_log["blockers"].append("HF dataset access failure")
            else:
                run_log["blockers"].append(f"dataset load ({ds_name}): {e}")
            run_log["datasets"][ds_name] = {"status": "failed_load", "error": str(e)}
            continue

        if not queries:
            msg = f"Dataset {ds_key} returned 0 queries."
            print("EXACT ERROR:", msg)
            run_log["blockers"].append(msg)
            run_log["datasets"][ds_key] = {"status": "empty"}
            continue

        print(f"\n=== {ds_key} ({len(queries)} queries, math_extract={use_math}) ===")

        def _set_gt(m: object, q) -> None:
            if isinstance(m, DummyModel):
                m.set_ground_truth(q.answer)

        try:
            for q in queries:
                _set_gt(model, q)
            ladder = evaluate_compute_ladder(
                model, queries, dataset_key=ds_key, use_math_extraction=use_math
            )
            write_compute_ladder_json(ladder, out_dir / f"{ds_key}_compute_ladder.json")
        except Exception as e:
            print("EXACT ERROR in compute_ladder:", repr(e))
            traceback.print_exc()
            run_log["blockers"].append(f"{ds_key} ladder: {e}")
            run_log["datasets"][ds_key] = {"status": "failed_ladder", "error": str(e)}
            continue

        conf_path = out_dir / f"{ds_key}_confidence_router.csv"
        if conf_path.exists():
            conf_path.unlink()
        try:
            for q in queries:
                _set_gt(model, q)
            curve_dpr = evaluate_confidence_router_curve(
                model,
                queries,
                dataset_key=ds_key,
                use_math_extraction=use_math,
                strong_action="direct_plus_revise",
            )
            for q in queries:
                _set_gt(model, q)
            curve_sc3 = evaluate_confidence_router_curve(
                model,
                queries,
                dataset_key=ds_key,
                use_math_extraction=use_math,
                strong_action="self_consistency_3",
            )
            append_confidence_router_csv(curve_dpr + curve_sc3, conf_path)
        except Exception as e:
            print("EXACT ERROR in confidence_router:", repr(e))
            traceback.print_exc()
            run_log["blockers"].append(f"{ds_key} confidence_router: {e}")

        try:
            for action in ("reasoning_then_revise", "self_consistency_3"):
                for q in queries:
                    _set_gt(model, q)
                pl = evaluate_output_router(
                    model,
                    queries,
                    dataset_key=ds_key,
                    use_math_extraction=use_math,
                    escalate_action=action,
                )
                output_payloads.append(pl)
            write_output_router_json(
                {"dataset": ds_key, "variants": output_payloads},
                out_dir / f"{ds_key}_output_router.json",
            )
        except Exception as e:
            print("EXACT ERROR in output_router:", repr(e))
            traceback.print_exc()
            run_log["blockers"].append(f"{ds_key} output_router: {e}")

        try:
            for q in queries:
                _set_gt(model, q)
            br = evaluate_best_route_style(
                model, queries, dataset_key=ds_key, use_math_extraction=use_math
            )
            write_best_route_style_json(br, out_dir / f"{ds_key}_best_route_style.json")
        except Exception as e:
            print("EXACT ERROR in best_route_style:", repr(e))
            traceback.print_exc()
            run_log["blockers"].append(f"{ds_key} best_route_style: {e}")
            br = {
                "accuracy": 0.0,
                "avg_cost_proxy": 0.0,
                "note": f"failed: {e}",
            }

        run_log["datasets"][ds_key] = {
            "status": "ok",
            "n_queries": len(queries),
            "ladder_methods": list(ladder.get("methods", {}).keys()),
        }

        rows = build_summary_rows(
            ds_key,
            ladder,
            output_payloads,
            br,
            curve_dpr + curve_sc3,
        )
        all_summary_rows.extend(rows)

    write_final_summary_csv(all_summary_rows, out_dir / "final_baseline_summary.csv")
    write_dataset_rollup_csv(all_summary_rows, out_dir / "dataset_rollup.csv")
    (out_dir / "run_log.json").write_text(json.dumps(run_log, indent=2))

    print(f"\nWrote summaries under {out_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
