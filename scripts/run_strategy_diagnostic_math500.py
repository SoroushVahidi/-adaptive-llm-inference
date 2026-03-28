#!/usr/bin/env python3
"""Run the small MATH500 strategy-comparison diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.strategy_diagnostic import (  # noqa: E402
    format_strategy_diagnostic_summary,
    run_strategy_diagnostic,
    write_strategy_diagnostic_outputs,
)
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the MATH500 strategy comparison diagnostic with real OpenAI models"
    )
    parser.add_argument(
        "--config",
        default="configs/strategy_diagnostic_math500.yaml",
        help="Path to YAML/JSON config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_strategy_diagnostic(config)
    paths = write_strategy_diagnostic_outputs(
        result=result,
        output_dir=config.get("output_dir", "outputs/strategy_diagnostic_math500"),
    )
    print(format_strategy_diagnostic_summary(result, paths))


if __name__ == "__main__":
    main()
