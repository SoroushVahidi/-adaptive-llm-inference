#!/usr/bin/env python3
"""Run the complete small experiment pass (AIME + confidence baseline).

Orchestrates:
1. AIME-2024 policy evaluation (offline, committed data).
2. Confidence-threshold routing baseline on all four main manuscript regimes.
3. Combined comparison table export.

Outputs are written to outputs/small_pass/ and outputs/paper_tables_small_pass/.
No API calls required.

Usage::

    python scripts/run_small_pass.py
    python scripts/run_small_pass.py --output-dir outputs/small_pass
    python scripts/run_small_pass.py --tables-dir outputs/paper_tables_small_pass
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.evaluation.small_pass_combined_eval import run_small_pass  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir", default="outputs/small_pass")
    p.add_argument("--tables-dir", default="outputs/paper_tables_small_pass")
    p.add_argument("--conf-target-cost", type=float, default=1.2)
    args = p.parse_args()

    result = run_small_pass(
        output_dir=args.output_dir,
        tables_dir=args.tables_dir,
        conf_target_cost=args.conf_target_cost,
    )
    print(json.dumps(result["run_status"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
