#!/usr/bin/env python3
"""Export manuscript-ready figures from existing CSV/JSON artifacts.

Requires matplotlib for PNG generation. Does not fabricate experiment results.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.paper_artifacts.exports_figures import print_blockers, run_all_figure_exports  # noqa: E402
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
        help="Override output directory (default: <repo>/outputs/paper_figures)",
    )
    p.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Subset: simulated_sweep, next_stage_budget, next_stage_cascade, real_policy",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit 2 if any figure group is blocked.",
    )
    args = p.parse_args()
    art = ArtifactPaths.from_root(args.repo_root)
    out = args.out_dir.resolve() if args.out_dir else None
    only_set = set(args.only) if args.only else None
    written, blockers = run_all_figure_exports(art, out, only=only_set)
    print_blockers(blockers)
    print("Wrote:", len(written), "paths under", (out or art.paper_figures))
    if args.strict and blockers:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
