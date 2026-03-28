#!/usr/bin/env python3
"""Run oracle subset evaluation on GSM8K or MATH500 with real OpenAI models."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.oracle_subset_eval import (  # noqa: E402
    format_oracle_subset_summary,
    run_oracle_subset_eval,
    write_oracle_subset_outputs,
)
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run oracle subset evaluation with real OpenAI models"
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to YAML/JSON config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_oracle_subset_eval(config)
    paths = write_oracle_subset_outputs(
        result=result,
        output_dir=config.get("output_dir", "outputs/oracle_subset_eval"),
    )
    print(format_oracle_subset_summary(result, paths))


if __name__ == "__main__":
    main()
