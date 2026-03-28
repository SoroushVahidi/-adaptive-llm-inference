#!/usr/bin/env python3
"""Run the adaptive policy v1 GSM8K evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.adaptive_policy_eval import (  # noqa: E402
    format_adaptive_policy_summary,
    run_adaptive_policy_eval,
    write_adaptive_policy_outputs,
)
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the adaptive policy v1 GSM8K evaluation"
    )
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config file")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_adaptive_policy_eval(config)
    paths = write_adaptive_policy_outputs(
        result=result,
        output_dir=config.get("output_dir", "outputs/adaptive_policy_v1"),
    )
    print(format_adaptive_policy_summary(result, paths))


if __name__ == "__main__":
    main()
