#!/usr/bin/env python3
"""Summarize simulated sweep outputs into tables and optional plots."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.analysis_summary import (
    format_terminal_summary,
    summarize_simulated_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize simulated sweep comparison outputs"
    )
    parser.add_argument(
        "--input-dir",
        default="outputs/simulated_sweep",
        help="Directory containing simulated sweep comparison CSV files",
    )
    parser.add_argument(
        "--small-gap-threshold",
        type=float,
        default=0.5,
        help="Absolute utility gap threshold used to flag when the MCKP advantage becomes small",
    )
    args = parser.parse_args()

    summary = summarize_simulated_results(
        input_dir=args.input_dir,
        small_gap_threshold=args.small_gap_threshold,
    )
    print(format_terminal_summary(summary))


if __name__ == "__main__":
    main()
