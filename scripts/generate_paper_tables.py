#!/usr/bin/env python3
"""Export manuscript-ready CSV tables from existing experiment artifacts.

Does not run experiments or invent numbers. Missing inputs are reported as
blockers; exports that succeed still write under outputs/paper_tables/.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.paper_artifacts.exports_tables import print_blockers, run_all_table_exports  # noqa: E402
from src.paper_artifacts.paths import ArtifactPaths  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--repo-root",
        type=Path,
        default=_REPO,
        help="Repository root (default: parent of scripts/)",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Override output directory (default: <repo>/outputs/paper_tables)",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Subset of exporters: simulated_sweep, baselines, cross_regime, "
        "final_cross_regime, oracle_routing, oracle_subset, real_policy_comparison, "
        "next_stage_budget_curves",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if any exporter is blocked (default: exit 0 after partial export).",
    )
    args = p.parse_args()
    art = ArtifactPaths.from_root(args.repo_root)
    out = args.out_dir
    if out is not None:
        out = out.resolve()
    only_set = set(args.only) if args.only else None
    written, blockers = run_all_table_exports(art, out, only=only_set)
    print_blockers(blockers)
    print("Wrote:", len(written), "paths under", (out or art.paper_tables))
    if args.strict and blockers:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
