"""Validate uploaded GSM8K/MATH500 ZIP files and normalize valid ones."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.datasets.validate_uploaded_datasets import run_uploaded_dataset_validation  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--output-dir", default="outputs/dataset_validation")
    parser.add_argument("--data-dir", default="data")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    out = run_uploaded_dataset_validation(
        repo_root=args.repo_root,
        output_dir=args.output_dir,
        data_dir=args.data_dir,
    )
    print(json.dumps(out["summary"], indent=2))
    print(f"summary_json={out['summary_path']}")
    print(f"gsm_preview_json={out['gsm_preview_path']}")
    print(f"math_preview_json={out['math_preview_path']}")


if __name__ == "__main__":
    main()
