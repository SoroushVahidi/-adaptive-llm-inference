#!/usr/bin/env python3
"""Offline smoke test for newly added public reasoning datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.bbh import load_bbh_records  # noqa: E402
from src.datasets.mmlu_pro import load_mmlu_pro_records  # noqa: E402
from src.datasets.musr import load_musr_records  # noqa: E402
from src.datasets.strategyqa import load_strategyqa_records  # noqa: E402


def main() -> None:
    payload = {
        "mmlu_pro": len(load_mmlu_pro_records(allow_external=False, max_samples=10, seed=1)),
        "musr": len(load_musr_records(allow_external=False, max_samples=10, seed=1)),
        "strategyqa": len(load_strategyqa_records(allow_external=False, max_samples=10, seed=1)),
        "bbh": len(load_bbh_records(allow_external=False, max_samples=10, seed=1)),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
