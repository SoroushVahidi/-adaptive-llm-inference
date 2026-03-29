#!/usr/bin/env python3
"""Run adaptive policy v5 evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.adaptive_policy_v5_eval import (  # noqa: E402
    format_adaptive_policy_v5_summary,
    run_adaptive_policy_v5_eval,
    write_adaptive_policy_v5_outputs,
)
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive policy v5 GSM8K evaluation")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON config")
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_adaptive_policy_v5_eval(config)
    paths = write_adaptive_policy_v5_outputs(
        result=result,
        output_dir=config.get("output_dir", "outputs/adaptive_policy_v5"),
    )
    print(format_adaptive_policy_v5_summary(result, paths))


if __name__ == "__main__":
    main()
