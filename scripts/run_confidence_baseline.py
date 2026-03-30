#!/usr/bin/env python3
"""Run confidence-threshold routing baseline across all manuscript regimes.

Sweeps ``unified_confidence_score`` thresholds (0.00–1.00) on all four main
manuscript routing datasets (GSM8K, Hard-GSM8K×2, MATH500) and optionally
AIME-2024, producing:

- ``confidence_threshold_sweep.csv`` — full threshold sweep per regime
- ``confidence_threshold_summary.csv`` — operating-point result per regime
- ``confidence_threshold_summary.json`` — same as JSON

No API calls required — all routing datasets are committed.

Usage::

    python scripts/run_confidence_baseline.py
    python scripts/run_confidence_baseline.py --output-dir outputs/baselines/confidence_threshold
    python scripts/run_confidence_baseline.py --include-aime --target-cost 1.3
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.baselines.confidence_threshold_router import (  # noqa: E402
    REGIME_FILES,
    sweep_and_summarise,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--output-dir",
        default="outputs/baselines/confidence_threshold",
        help="Output directory (default: %(default)s)",
    )
    p.add_argument(
        "--target-cost",
        type=float,
        default=1.2,
        help="Target avg-cost for operating-point selection (default: %(default)s)",
    )
    p.add_argument(
        "--include-aime",
        action="store_true",
        default=False,
        help="Also run the confidence sweep on the AIME-2024 routing dataset",
    )
    args = p.parse_args()

    # Build absolute regime file paths (the module stores repo-relative paths)
    regime_files: dict[str, str] = {
        k: str(REPO_ROOT / v) for k, v in REGIME_FILES.items()
    }

    if args.include_aime:
        aime_path = REPO_ROOT / "data/real_aime2024_routing_dataset.csv"
        if aime_path.exists():
            regime_files["aime2024"] = str(aime_path)
        else:
            print(
                f"WARNING: AIME-2024 CSV not found at {aime_path}; skipping.",
                file=sys.stderr,
            )

    results = sweep_and_summarise(
        regime_files=regime_files,
        output_dir=args.output_dir,
        target_cost=args.target_cost,
    )

    summary_rows = [r.to_summary_dict() for r in results]
    print(json.dumps(summary_rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
