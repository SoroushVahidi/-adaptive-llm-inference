#!/usr/bin/env python3
"""Run the standalone consistency-check benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.analysis.consistency_benchmark import (  # noqa: E402
    evaluate_benchmark,
    load_benchmark,
    save_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run consistency-check benchmark")
    parser.add_argument(
        "--benchmark",
        default="data/consistency_benchmark.json",
        help="Path to benchmark JSON",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/consistency_benchmark",
        help="Directory for output artifacts",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    benchmark = load_benchmark(args.benchmark)
    evaluation = evaluate_benchmark(benchmark)
    paths = save_outputs(evaluation, args.output_dir)

    printable = {
        "num_questions": evaluation["num_questions"],
        "num_candidates": evaluation["num_candidates"],
        "num_wrong_candidates": evaluation["num_wrong_candidates"],
        "num_correct_candidates": evaluation["num_correct_candidates"],
        "variant_metrics": evaluation["variant_metrics"],
        "output_paths": paths,
    }
    print(json.dumps(printable, indent=2))


if __name__ == "__main__":
    main()
