from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPT = REPO_ROOT / "scripts" / "generate_final_manuscript_artifacts.py"
TABLE = REPO_ROOT / "outputs" / "paper_tables_final" / "policy_comparison_main.csv"


def _ensure_token_budget_outputs() -> None:
    required = [
        REPO_ROOT / "outputs/token_budget_router/gsm8k_random_100/policy_comparison.csv",
        REPO_ROOT / "outputs/token_budget_router/budget_curves/gsm8k_random_100_token_budget_curve.csv",
    ]
    if all(p.exists() for p in required):
        return
    subprocess.run(
        [
            "python",
            "-m",
            "routing.token_budget_router.tune",
            "--config",
            "config/token_budget_router_default.yaml",
        ],
        cwd=REPO_ROOT,
        check=True,
    )
    subprocess.run(
        [
            "python",
            "-m",
            "routing.token_budget_router.eval",
            "--config",
            "config/token_budget_router_default.yaml",
        ],
        cwd=REPO_ROOT,
        check=True,
    )


def test_final_manuscript_export_includes_token_budget_router() -> None:
    pytest.importorskip("matplotlib")
    _ensure_token_budget_outputs()
    subprocess.run(["python", str(SCRIPT)], cwd=REPO_ROOT, check=True)

    assert TABLE.exists(), f"Expected table missing: {TABLE}"
    with TABLE.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    token_rows = [r for r in rows if r.get("policy") == "token_budget_router"]
    assert len(token_rows) == 4, f"Expected 4 token-budget rows, got {len(token_rows)}"
    regimes = {r["regime"] for r in token_rows}
    assert regimes == {
        "gsm8k_random_100",
        "hard_gsm8k_100",
        "hard_gsm8k_b2",
        "math500_100",
    }
