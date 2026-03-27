#!/usr/bin/env python3
"""Run a small real-data empirical gain-table experiment on GSM8K."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.real_gain_table import (
    format_gain_table_summary,
    run_real_gain_table,
    write_gain_table_outputs,
)
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a real OpenAI GSM8K empirical gain-table experiment"
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_real_gain_table(config)
    paths = write_gain_table_outputs(
        result=result,
        output_dir=config.get("output_dir", "outputs/real_gain_table"),
    )
    print(format_gain_table_summary(result, paths))


if __name__ == "__main__":
    main()
