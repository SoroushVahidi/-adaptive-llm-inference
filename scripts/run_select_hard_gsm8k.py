#!/usr/bin/env python3
"""Select top-N hardest GSM8K test questions by question-side proxies."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.hard_gsm8k_selection import (  # noqa: E402
    select_hard_gsm8k_queries,
    write_hard_selection_artifacts,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--subset-size", type=int, default=100)
    p.add_argument(
        "--pool-size",
        type=int,
        default=None,
        help="Max GSM8K test rows to score (default: full test split)",
    )
    p.add_argument("--gsm8k-data-file", default="")
    p.add_argument("--output-dir", default="outputs/hard_regime_selection")
    args = p.parse_args()
    data_file = args.gsm8k_data_file or None
    if data_file and not Path(data_file).exists():
        data_file = None

    _queries, rows, summary = select_hard_gsm8k_queries(
        pool_size=args.pool_size,
        subset_size=args.subset_size,
        gsm8k_data_file=data_file,
    )
    paths = write_hard_selection_artifacts(rows, summary, args.output_dir)
    print(json.dumps(summary, indent=2))
    print(f"wrote {paths['csv']}")
    print(f"wrote {paths['json']}")


if __name__ == "__main__":
    main()
