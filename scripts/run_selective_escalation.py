#!/usr/bin/env python3
"""Run a small real-data selective compute escalation experiment on GSM8K."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.selective_escalation_eval import (
    format_selective_summary,
    run_selective_escalation_eval,
    write_selective_outputs,
)
from src.utils.config import load_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the selective escalation v1 GSM8K experiment"
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_selective_escalation_eval(config)
    paths = write_selective_outputs(
        result=result,
        output_dir=config.get("output_dir", "outputs/selective_escalation"),
    )
    print(format_selective_summary(result, paths))


if __name__ == "__main__":
    main()
