#!/usr/bin/env python3
"""Learned routing eval on paired-outcomes CSVs (``revise_helpful`` label).

Calls ``src.evaluation.real_routing_model_eval.run_real_routing_model_eval``:
CV metrics for tree ensembles, ``routing_simulation.csv``, and
``outputs/real_routing_model/``-style artifacts by default.

**Oracle router baselines** (decision tree / logistic regression on
``best_accuracy_strategy`` from ``routing_dataset.csv``) are a different
pipeline — use::

    python3 scripts/run_router_baseline_eval.py

Prerequisites
-------------
- Paired-outcomes CSV (e.g. from ``run_build_real_routing_dataset.py
  --paired-outcomes --output-dataset-csv data/real_gsm8k_routing_dataset.csv``).
- scikit-learn for model fitting.

Usage
-----
    python3 scripts/run_real_routing_model_eval.py

    python3 scripts/run_real_routing_model_eval.py \\
        --dataset-csv data/real_math500_routing_dataset.csv \\
        --output-dir outputs/real_math500_routing_model

Outputs (default)
-----------------
``outputs/real_routing_model/``
    ├── summary.json
    ├── model_metrics.csv
    ├── per_query_predictions.csv
    ├── routing_simulation.csv
    └── feature_importance.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.real_routing_model_eval import run_real_routing_model_eval  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Evaluate revise_helpful routing models (tree ensembles + simulation)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--dataset-csv",
        default="data/real_gsm8k_routing_dataset.csv",
        metavar="PATH",
        help="Paired-outcomes routing CSV (default: data/real_gsm8k_routing_dataset.csv)",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/real_routing_model",
        metavar="DIR",
        help="Output directory (default: outputs/real_routing_model)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    result = run_real_routing_model_eval(
        dataset_csv=args.dataset_csv,
        output_dir=args.output_dir,
    )
    print(json.dumps(result["summary"], indent=2))
    status = result["summary"].get("run_status", "")
    return 0 if status == "OK" else 1


if __name__ == "__main__":
    raise SystemExit(main())
