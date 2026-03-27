#!/usr/bin/env python3
"""Run a small GSM8K diagnostic experiment with a real OpenAI model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.real_llm_diagnostic import (
    format_diagnostic_summary,
    run_real_llm_diagnostic,
    write_diagnostic_outputs,
)
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a focused real OpenAI GSM8K diagnostic experiment"
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_real_llm_diagnostic(config)
    paths = write_diagnostic_outputs(
        result=result,
        output_dir=config.get("output_dir", "outputs/real_llm_diagnostic"),
    )
    print(format_diagnostic_summary(result, paths))


if __name__ == "__main__":
    main()
