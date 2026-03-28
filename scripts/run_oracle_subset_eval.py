#!/usr/bin/env python3
"""Run the oracle-subset evaluation on a shared small GSM8K query set.

Usage:
    python3 scripts/run_oracle_subset_eval.py \
        --config configs/oracle_subset_eval_gsm8k.yaml

Requires OPENAI_API_KEY to be set in the environment.

If API access is unavailable, the script stops immediately and writes
docs/EXPERIMENT_LOG_ORACLE_SUBSET.md and outputs/oracle_subset_eval/summary.json
with a BLOCKED status — it does NOT invent numeric results.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.datasets.gsm8k import load_gsm8k
from src.evaluation.oracle_subset_eval import (
    CORE_ORACLE_STRATEGIES,
    compute_oracle_summaries,
    compute_pairwise_win_matrix,
    format_oracle_summary,
    run_oracle_subset_eval,
    write_oracle_outputs,
)
from src.models.openai_llm import OpenAILLMModel
from src.utils.config import load_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_DOCS_LOG_PATH = Path("docs/EXPERIMENT_LOG_ORACLE_SUBSET.md")
_DOCS_RESULTS_PATH = Path("docs/RESULTS_ORACLE_SUBSET.md")


def _now_utc() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_blocker_summary(output_dir: str, blocker: str, detail: str) -> str:
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_status": "BLOCKED",
        "blocker_type": blocker,
        "blocker_detail": detail,
        "strategy_accuracy": {},
        "oracle_accuracy": None,
        "direct_accuracy": None,
        "oracle_minus_direct_gap": None,
        "total_queries": 0,
        "strategies_run": [],
    }
    path = base / "summary.json"
    path.write_text(json.dumps(payload, indent=2))
    return str(path)


def _write_blocked_experiment_log(
    blocker_type: str,
    blocker_detail: str,
    config_path: str,
    output_dir: str,
    timestamp: str,
    strategies: list[str],
) -> None:
    _DOCS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DOCS_LOG_PATH.write_text(
        f"""# Experiment Log — Oracle Subset Evaluation

## Status: BLOCKED

**Date/time (UTC):** {timestamp}

## Command

```bash
python3 scripts/run_oracle_subset_eval.py --config {config_path}
```

## Configuration

- **Config file:** `{config_path}`
- **Dataset:** GSM8K (bundled test sample, max 15 queries)
- **Strategies intended:** {", ".join(f"`{s}`" for s in strategies)}
- **Output directory:** `{output_dir}`

## Blocker

| Field | Value |
|-------|-------|
| Blocker type | `{blocker_type}` |
| Detail | {blocker_detail} |

## Why the run could not complete

The live OpenAI API key (`OPENAI_API_KEY`) is not available in this
execution environment.  Without it the `OpenAILLMModel` initialiser raises a
`ValueError` before any queries are processed.

The script stops here and does **not** invent any numeric results.
All outputs are marked `"run_status": "BLOCKED"`.

## Where outputs were written

- `{output_dir}/summary.json` — BLOCKED sentinel
"""
    )


def _write_results_blocked(timestamp: str) -> None:
    _DOCS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DOCS_RESULTS_PATH.write_text(
        f"""# Results — Oracle Subset Evaluation

## Status: BLOCKED (no live API results)

**Date/time (UTC):** {timestamp}

This file will be populated with real numbers once `OPENAI_API_KEY` is
available and the script completes successfully.

### Planned metrics

| Metric | Value |
|--------|-------|
| Direct accuracy | — (BLOCKED) |
| Oracle accuracy | — (BLOCKED) |
| Oracle-direct gap | — (BLOCKED) |
| Direct already optimal | — (BLOCKED) |
| Strategies with most fixes | — (BLOCKED) |

### Interpretation

_Not yet available._

### What this means for the paper

Once results are available, this section will describe which strategy families
contribute most oracle gain over direct_greedy and whether cheap single-call
strategies leave significant headroom on the table.
"""
    )


def _write_results_doc(
    oracle_summaries: dict,
    eval_result: dict,
    config_path: str,
    output_dir: str,
    timestamp: str,
) -> None:
    """Write RESULTS_ORACLE_SUBSET.md with actual numbers."""
    strategies = eval_result["strategies_run"]
    query_ids = eval_result["query_ids"]
    n = oracle_summaries["total_queries"]
    direct_acc = oracle_summaries["direct_accuracy"]
    oracle_acc = oracle_summaries["oracle_accuracy"]
    gap = oracle_summaries["oracle_minus_direct_gap"]
    opt_frac = oracle_summaries["direct_already_optimal_fraction"]

    # Accuracy rows
    acc_rows = ""
    for st, acc in oracle_summaries["strategy_accuracy"].items():
        acc_rows += f"| `{st}` | {acc['accuracy']:.4f} | {acc['correct']}/{n} |\n"

    # Top contributors (strategies sorted by "fixes direct" count)
    fixes = oracle_summaries["fixes_direct_greedy"]
    top_fixes = sorted(fixes.items(), key=lambda x: x[1], reverse=True)
    fixes_rows = ""
    for st, cnt in top_fixes:
        fixes_rows += f"| `{st}` | {cnt} |\n"

    cc = oracle_summaries["cheapest_correct_counts"]
    cc_rows = ""
    for st in strategies:
        cc_rows += f"| `{st}` | {cc.get(st, 0)} |\n"

    _DOCS_RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DOCS_RESULTS_PATH.write_text(
        f"""# Results — Oracle Subset Evaluation

**Date/time (UTC):** {timestamp}
**Config:** `{config_path}`
**Dataset:** GSM8K bundled test sample, {n} queries
**Query IDs:** {", ".join(query_ids)}

## Strategy Accuracy

| Strategy | Accuracy | Correct/Total |
|----------|----------|---------------|
{acc_rows}
## Oracle Metrics

| Metric | Value |
|--------|-------|
| Oracle accuracy (≥1 strategy correct) | {oracle_acc:.4f} |
| Direct (`direct_greedy`) accuracy | {direct_acc:.4f} |
| **Oracle-direct gap** | **{gap:+.4f}** |
| Queries where direct was already optimal | {opt_frac:.1%} |

## Strategy Contributions

### Fixes over `direct_greedy` (queries where direct_greedy was wrong and this strategy was correct)

| Strategy | Queries fixed |
|----------|---------------|
{fixes_rows}
### Cheapest correct strategy count

| Strategy | Times cheapest correct |
|----------|------------------------|
{cc_rows}
## Interpretation

- Oracle accuracy of {oracle_acc:.4f} vs direct accuracy of {direct_acc:.4f} gives
  a gap of {gap:+.4f}, showing {'significant' if gap > 0.1 else 'modest'} headroom
  for adaptive strategy selection.
- {opt_frac:.1%} of queries are already solved by `direct_greedy` at minimum cost,
  meaning these queries do not benefit from more expensive strategies.
- Strategies that most frequently fix `direct_greedy` failures are the best
  candidates for a selective-escalation or adaptive-routing policy.

## What this means for the paper

The oracle gap of {gap:+.4f} over {n} queries quantifies the upper bound on accuracy
improvement that any oracle adaptive policy could achieve.  The cheapest-correct
distribution shows that low-cost strategies often suffice, while the "fixes" column
identifies which multi-stage or multi-sample strategies add the most value.
These numbers motivate the adaptive compute allocation approach proposed in the paper.
"""
    )


def _write_experiment_log(
    eval_result: dict,
    oracle_summaries: dict,
    paths: dict,
    config_path: str,
    output_dir: str,
    timestamp: str,
    model_name: str,
) -> None:
    strategies = eval_result["strategies_run"]
    query_ids = eval_result["query_ids"]
    n = oracle_summaries["total_queries"]

    _DOCS_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DOCS_LOG_PATH.write_text(
        f"""# Experiment Log — Oracle Subset Evaluation

## Status: COMPLETED

**Date/time (UTC):** {timestamp}

## Command

```bash
python3 scripts/run_oracle_subset_eval.py --config {config_path}
```

## Configuration

- **Config file:** `{config_path}`
- **Dataset:** GSM8K bundled test sample (split: test, max_samples: {n})
- **Model:** `{model_name}`
- **Output directory:** `{output_dir}`

## Strategies evaluated

{chr(10).join(f"- `{s}`" for s in strategies)}

## Query subset ({n} queries)

{", ".join(f"`{qid}`" for qid in query_ids)}

## Output files

| File | Path |
|------|------|
{chr(10).join(f"| `{k}` | `{v}` |" for k, v in paths.items())}

## Runtime notes

Run completed successfully.  See `docs/RESULTS_ORACLE_SUBSET.md` for metrics.
"""
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run oracle-subset evaluation on GSM8K"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML config (e.g. configs/oracle_subset_eval_gsm8k.yaml)",
    )
    args = parser.parse_args()

    timestamp = _now_utc()
    config = load_config(args.config)
    output_dir = config.get("output_dir", "outputs/oracle_subset_eval")

    strategies: list[str] = list(config.get("strategies", CORE_ORACLE_STRATEGIES))

    # --- Dataset ---
    dataset_cfg = config.get("dataset", {})
    data_file = dataset_cfg.get("data_file")
    print(
        f"Loading GSM8K ({dataset_cfg.get('split', 'test')} split, "
        f"max {dataset_cfg.get('max_samples', 15)} queries)..."
    )
    try:
        queries = load_gsm8k(
            split=str(dataset_cfg.get("split", "test")),
            max_samples=int(dataset_cfg.get("max_samples", 15)),
            data_file=data_file,
        )
    except Exception as exc:  # noqa: BLE001
        msg = str(exc)
        print(f"BLOCKER: Failed to load GSM8K dataset.\n  Error: {msg}", file=sys.stderr)
        _write_blocker_summary(output_dir, "dataset_access", msg)
        _write_blocked_experiment_log(
            "dataset_access", msg, args.config, output_dir, timestamp, strategies
        )
        _write_results_blocked(timestamp)
        sys.exit(1)
    print(f"Loaded {len(queries)} queries.")

    # --- Primary model ---
    model_cfg = config.get("model", {})
    model_name = str(model_cfg.get("name", "gpt-4o-mini"))
    print(f"Initialising primary model '{model_name}'...")
    try:
        model = OpenAILLMModel(
            model_name=model_name,
            base_url=model_cfg.get("base_url"),
            greedy_temperature=float(model_cfg.get("greedy_temperature", 0.0)),
            sample_temperature=float(model_cfg.get("sample_temperature", 0.7)),
            max_tokens=int(model_cfg.get("max_tokens", 512)),
            timeout_seconds=float(model_cfg.get("timeout_seconds", 60.0)),
        )
    except ValueError as exc:
        msg = str(exc)
        print(f"BLOCKER: Cannot initialise primary model.\n  Error: {msg}", file=sys.stderr)
        _write_blocker_summary(output_dir, "openai_api_key", msg)
        _write_blocked_experiment_log(
            "openai_api_key", msg, args.config, output_dir, timestamp, strategies
        )
        _write_results_blocked(timestamp)
        sys.exit(1)

    # --- Optional strong model ---
    strong_model = None
    strong_model_cfg = config.get("strong_model")
    if strong_model_cfg:
        strong_name = str(strong_model_cfg.get("name", "gpt-4o"))
        print(f"Initialising strong model '{strong_name}' for strong_direct strategy...")
        try:
            strong_model = OpenAILLMModel(
                model_name=strong_name,
                base_url=strong_model_cfg.get("base_url"),
                greedy_temperature=float(strong_model_cfg.get("greedy_temperature", 0.0)),
                max_tokens=int(strong_model_cfg.get("max_tokens", 512)),
                timeout_seconds=float(strong_model_cfg.get("timeout_seconds", 90.0)),
            )
            if "strong_direct" not in strategies:
                strategies.append("strong_direct")
        except ValueError:
            print(
                "  WARNING: strong model initialisation failed; "
                "strong_direct strategy will be skipped.",
                file=sys.stderr,
            )

    print(f"Strategies: {strategies}\n")

    # --- Run evaluation ---
    print("Running evaluation...")
    try:
        eval_result = run_oracle_subset_eval(
            model=model,
            queries=queries,
            strategies=strategies,
            strong_model=strong_model,
        )
    except RuntimeError as exc:
        msg = str(exc)
        print(f"BLOCKER: Evaluation failed.\n  Error: {msg}", file=sys.stderr)
        _write_blocker_summary(output_dir, "openai_api_error", msg)
        _write_blocked_experiment_log(
            "openai_api_error", msg, args.config, output_dir, timestamp, strategies
        )
        _write_results_blocked(timestamp)
        sys.exit(1)

    # --- Oracle summaries ---
    oracle_summaries = compute_oracle_summaries(
        eval_result["per_query_rows"],
        eval_result["strategies_run"],
    )
    pairwise = compute_pairwise_win_matrix(
        eval_result["per_query_rows"],
        eval_result["strategies_run"],
    )

    # --- Save outputs ---
    paths = write_oracle_outputs(eval_result, oracle_summaries, pairwise, output_dir)

    # --- Write docs ---
    _write_experiment_log(
        eval_result, oracle_summaries, paths, args.config, output_dir, timestamp, model_name
    )
    _write_results_doc(
        oracle_summaries, eval_result, args.config, output_dir, timestamp
    )

    # --- Print summary ---
    print(format_oracle_summary(oracle_summaries, paths))
    print(f"\nExperiment log: {_DOCS_LOG_PATH}")
    print(f"Results doc:    {_DOCS_RESULTS_PATH}")


if __name__ == "__main__":
    main()
