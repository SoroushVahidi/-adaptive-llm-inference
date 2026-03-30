"""Bootstrap uncertainty analysis for the main manuscript comparisons.

Computes **paired bootstrap confidence intervals** for key pairwise accuracy
comparisons across the four main manuscript regimes:

1. ``adaptive_best_policy`` vs ``always_reasoning``
   — the primary claim of the paper.
2. ``oracle`` vs ``always_reasoning``
   — upper-bound gain.
3. ``adaptive_best_policy`` vs ``oracle``
   — policy–oracle gap.

Methodology
-----------
For each comparison we draw ``n_bootstrap`` samples (with replacement) of size
``n`` (= 100 for each regime) from the per-query correct/incorrect indicator
vectors.  We compute the accuracy difference for each bootstrap replicate, then
read off the 2.5th and 97.5th percentiles as the 95 % confidence interval.

All results are grounded in the committed routing datasets — no API calls, no
fabricated numbers.

Public API
----------
- ``REGIME_FILES``  — mapping from regime id to enriched CSV path.
- ``bootstrap_ci(a, b, n_bootstrap, seed)`` → ``BootstrapResult``
- ``run_uncertainty_analysis(regime_files, output_dir, n_bootstrap)``
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Regime registry (same as confidence_threshold_router for consistency)
# ---------------------------------------------------------------------------

REGIME_FILES: dict[str, str] = {
    "gsm8k_random_100": "data/real_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_100": "data/real_hard_gsm8k_routing_dataset_enriched.csv",
    "hard_gsm8k_b2": "data/real_hard_gsm8k_b2_routing_dataset_enriched.csv",
    "math500_100": "data/real_math500_routing_dataset_enriched.csv",
}

N_BOOTSTRAP_DEFAULT = 10_000
ALPHA = 0.05  # 95 % CI

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class BootstrapResult:
    """Result of a paired bootstrap comparison between two indicator vectors."""

    comparison: str  # e.g. "adaptive_best_policy_vs_always_reasoning"
    regime: str
    observed_delta: float  # policy_acc - baseline_acc  (point estimate)
    ci_lower: float  # 2.5th percentile of bootstrap deltas
    ci_upper: float  # 97.5th percentile of bootstrap deltas
    n_bootstrap: int
    n: int
    significant_at_95pct: bool  # CI excludes zero


@dataclass
class RegimeUncertaintyResult:
    """All pairwise comparisons for a single regime."""

    regime: str
    n: int
    comparisons: list[BootstrapResult]


# ---------------------------------------------------------------------------
# Best-policy selection from routing dataset columns
# ---------------------------------------------------------------------------

# Policy routing columns (v6 and v7 recommendations are pre-computed)
_POLICY_RECOMMEND_COLS = ("v6_revise_recommended", "v7_revise_recommended")


def _get_best_policy_correct(df: pd.DataFrame) -> np.ndarray:
    """Return a per-query correctness array for the best adaptive policy.

    We evaluate v6 and v7 using their pre-computed ``revise_recommended`` flags,
    pick whichever policy achieves higher mean accuracy on this regime, and
    return its per-query indicator vector.
    """
    best_acc = -1.0
    best_correct: np.ndarray | None = None

    for col in _POLICY_RECOMMEND_COLS:
        if col not in df.columns:
            continue
        revise_mask = df[col].astype(bool).values
        correct = revise_mask * df["revise_correct"].values + (~revise_mask) * df["reasoning_correct"].values
        acc = float(correct.mean())
        if acc > best_acc:
            best_acc = acc
            best_correct = correct.astype(float)

    if best_correct is None:
        # Fallback: always reasoning
        best_correct = df["reasoning_correct"].values.astype(float)
    return best_correct


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


def bootstrap_ci(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute a paired bootstrap 95 % CI for ``mean(a) - mean(b)``.

    Returns
    -------
    (observed_delta, ci_lower, ci_upper)
    """
    rng = np.random.default_rng(seed)
    n = len(a)
    assert len(b) == n, "Indicator vectors must be the same length."

    observed = float(np.mean(a) - np.mean(b))

    delta_dist = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        delta_dist[i] = np.mean(a[idx]) - np.mean(b[idx])

    ci_lower = float(np.percentile(delta_dist, 100 * ALPHA / 2))
    ci_upper = float(np.percentile(delta_dist, 100 * (1 - ALPHA / 2)))
    return observed, ci_lower, ci_upper


# ---------------------------------------------------------------------------
# Per-regime analysis
# ---------------------------------------------------------------------------


def analyse_regime(
    regime: str,
    df: pd.DataFrame,
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = 42,
) -> RegimeUncertaintyResult:
    """Run all pairwise bootstrap comparisons for one regime."""
    n = len(df)
    always_reasoning = df["reasoning_correct"].values.astype(float)
    always_revise = df["revise_correct"].values.astype(float)
    oracle = np.where(
        df["revise_helpful"].values.astype(bool),
        df["revise_correct"].values.astype(float),
        df["reasoning_correct"].values.astype(float),
    )
    best_policy = _get_best_policy_correct(df)

    comparisons_spec = [
        ("adaptive_best_policy_vs_always_reasoning", best_policy, always_reasoning),
        ("oracle_vs_always_reasoning", oracle, always_reasoning),
        ("adaptive_best_policy_vs_oracle", best_policy, oracle),
        ("always_revise_vs_always_reasoning", always_revise, always_reasoning),
    ]

    results: list[BootstrapResult] = []
    for label, vec_a, vec_b in comparisons_spec:
        obs, lo, hi = bootstrap_ci(vec_a, vec_b, n_bootstrap=n_bootstrap, seed=seed)
        results.append(
            BootstrapResult(
                comparison=label,
                regime=regime,
                observed_delta=round(obs, 4),
                ci_lower=round(lo, 4),
                ci_upper=round(hi, 4),
                n_bootstrap=n_bootstrap,
                n=n,
                significant_at_95pct=bool(lo > 0 or hi < 0),
            )
        )

    return RegimeUncertaintyResult(regime=regime, n=n, comparisons=results)


# ---------------------------------------------------------------------------
# Full analysis runner
# ---------------------------------------------------------------------------


def run_uncertainty_analysis(
    regime_files: dict[str, str] | None = None,
    output_dir: str | Path = "outputs/manuscript_support",
    n_bootstrap: int = N_BOOTSTRAP_DEFAULT,
    seed: int = 42,
) -> list[RegimeUncertaintyResult]:
    """Run bootstrap uncertainty analysis for all four regimes.

    Writes
    ------
    ``<output_dir>/uncertainty_analysis.json``
        Full results (all regimes, all comparisons, all CI values).
    ``<output_dir>/uncertainty_analysis_summary.csv``
        Flat CSV for easy copy/paste into the manuscript.
    """
    files = regime_files if regime_files is not None else REGIME_FILES
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_results: list[RegimeUncertaintyResult] = []
    for regime, csv_path in files.items():
        df = pd.read_csv(csv_path)
        result = analyse_regime(regime, df, n_bootstrap=n_bootstrap, seed=seed)
        all_results.append(result)

    # Write JSON
    json_path = out_dir / "uncertainty_analysis.json"
    json_path.write_text(
        json.dumps(
            [
                {
                    "regime": r.regime,
                    "n": r.n,
                    "comparisons": [asdict(c) for c in r.comparisons],
                }
                for r in all_results
            ],
            indent=2,
        )
    )

    # Write CSV
    import csv

    csv_path = out_dir / "uncertainty_analysis_summary.csv"
    fieldnames = [
        "regime",
        "comparison",
        "observed_delta",
        "ci_lower",
        "ci_upper",
        "significant_at_95pct",
        "n",
        "n_bootstrap",
    ]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            for c in r.comparisons:
                writer.writerow(
                    {
                        "regime": c.regime,
                        "comparison": c.comparison,
                        "observed_delta": c.observed_delta,
                        "ci_lower": c.ci_lower,
                        "ci_upper": c.ci_upper,
                        "significant_at_95pct": c.significant_at_95pct,
                        "n": c.n,
                        "n_bootstrap": c.n_bootstrap,
                    }
                )

    return all_results
