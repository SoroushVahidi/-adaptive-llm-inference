#!/usr/bin/env python3
"""Run the mode-then-budget v2 GSM8K experiment."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.mode_then_budget_eval import (
    format_mode_then_budget_summary,
    run_mode_then_budget_eval,
    write_mode_then_budget_outputs,
)
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the mode-then-budget v2 real OpenAI GSM8K experiment"
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_mode_then_budget_eval(config)
    paths = write_mode_then_budget_outputs(
        result=result,
        output_dir=config.get("output_dir", "outputs/mode_then_budget"),
    )
    print(format_mode_then_budget_summary(result, paths))


if __name__ == "__main__":
    main()
