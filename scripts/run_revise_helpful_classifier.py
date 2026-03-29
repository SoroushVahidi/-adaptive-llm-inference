"""Run the offline learned revise-helpful classifier evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.evaluation.revise_helpful_classifier_eval import (  # noqa: E402
    run_revise_helpful_classifier_eval,
)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--benchmark-path",
        default="data/consistency_benchmark.json",
        help="Path to consistency benchmark JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/revise_helpful_classifier",
        help="Directory for classifier artifacts",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    outputs = run_revise_helpful_classifier_eval(
        benchmark_path=args.benchmark_path,
        output_dir=args.output_dir,
    )

    summary = outputs["summary"]
    print("=== revise_helpful_classifier ===")
    print(json.dumps(summary, indent=2))
    print(f"summary_json={outputs['summary_json']}")
    print(f"model_metrics_csv={outputs['model_metrics_csv']}")
    print(f"per_query_predictions_csv={outputs['per_query_predictions_csv']}")
    print(f"routing_simulation_csv={outputs['routing_simulation_csv']}")
    print(f"feature_importance_csv={outputs['feature_importance_csv']}")


if __name__ == "__main__":
    main()
