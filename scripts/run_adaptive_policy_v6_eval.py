#!/usr/bin/env python3
"""Run offline adaptive policy v6 benchmark (no LLM; compares v4/v5/v6 routing)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluation.adaptive_policy_v6_eval import (  # noqa: E402
    format_offline_summary,
    run_offline_adaptive_policy_v6_eval,
    write_adaptive_policy_v6_outputs,
)
from src.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline adaptive policy v6 vs v4/v5 benchmark")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional YAML/JSON config (uses output_dir key if present)",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for summary.json and CSV outputs (overrides config)",
    )
    parser.add_argument(
        "--no-recall-proxies",
        action="store_true",
        help="Only run the 5 false-positive doc fixtures (skip synthetic recall proxies)",
    )
    args = parser.parse_args()

    default_out = "outputs/adaptive_policy_v6"
    if args.config:
        cfg = load_config(args.config)
        default_out = str(cfg.get("output_dir", default_out))
    output_dir = args.output_dir or default_out

    result = run_offline_adaptive_policy_v6_eval(
        include_recall_fixtures=not args.no_recall_proxies,
    )
    paths = write_adaptive_policy_v6_outputs(result, output_dir)
    print(format_offline_summary(result, paths))


if __name__ == "__main__":
    main()
