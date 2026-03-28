#!/usr/bin/env python3
"""Run the small GSM8K model-strength and structured-sampling diagnostic."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.model_sampling_diagnostic import (  # noqa: E402
    format_model_sampling_diagnostic_summary,
    run_model_sampling_diagnostic,
    write_model_sampling_diagnostic_outputs,
)
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the GSM8K model/sampling diagnostic with real OpenAI models"
    )
    parser.add_argument(
        "--config",
        default="configs/model_sampling_diagnostic_gsm8k.yaml",
        help="Path to YAML/JSON config file",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    result = run_model_sampling_diagnostic(config)
    paths = write_model_sampling_diagnostic_outputs(
        result=result,
        output_dir=config.get("output_dir", "outputs/model_sampling_diagnostic"),
    )
    print(format_model_sampling_diagnostic_summary(result, paths))


if __name__ == "__main__":
    main()
