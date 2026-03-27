#!/usr/bin/env python3
"""Debug real OpenAI sampling diversity on a tiny GSM8K subset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.real_llm_debug import (
    format_real_llm_debug_summary,
    run_real_llm_debug,
    write_real_llm_debug_outputs,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Debug real OpenAI GSM8K sampling diversity and parsing collapse"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=5,
        help="Maximum GSM8K queries to inspect (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/debug_real_llm",
        help="Directory for debug artifacts",
    )
    parser.add_argument(
        "--model",
        default="gpt-4o-mini",
        help="OpenAI model name",
    )
    args = parser.parse_args()

    result = run_real_llm_debug(
        {
            "dataset": {"name": "gsm8k", "split": "test", "max_samples": args.max_samples},
            "model": {"name": args.model},
            "selective_method": {
                "total_budget": args.max_samples * 3,
                "extra_samples_per_escalated_query": 2,
                "use_second_sample_for_disagreement": True,
                "parse_failure_weight": 2.0,
                "disagreement_weight": 1.5,
                "malformed_output_weight": 1.0,
                "missing_numeric_weight": 1.0,
                "min_score_to_escalate": 1.5,
            },
        }
    )
    paths = write_real_llm_debug_outputs(result, args.output_dir)
    print(format_real_llm_debug_summary(result, paths))


if __name__ == "__main__":
    main()
