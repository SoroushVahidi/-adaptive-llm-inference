#!/usr/bin/env python3
"""Build per-(prompt, action) candidate rows for hybrid routing framework."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.routing_hybrid.dataset_builder import (  # noqa: E402
    _read_csv,
    build_candidate_rows,
    write_candidate_artifacts,
)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input-csv", default="data/routing_ml_dataset.csv")
    p.add_argument("--output-dir", default="outputs/hybrid_routing_dataset")
    p.add_argument("--utility-lambdas", default="0.5,1.0,1.5")
    args = p.parse_args()

    rows = _read_csv(Path(args.input_csv))
    lambdas = [float(x.strip()) for x in args.utility_lambdas.split(",") if x.strip()]
    candidate_rows = build_candidate_rows(rows, utility_lambdas=lambdas)
    paths = write_candidate_artifacts(candidate_rows, Path(args.output_dir))
    print(json.dumps({"run_status": "OK", **paths}, indent=2))


if __name__ == "__main__":
    main()

