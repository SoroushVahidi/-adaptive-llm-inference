"""Evaluation entry point for the learned router baseline.

Loads a trained router (saved by ``train.py``) and evaluates it on every
manuscript regime.  Writes per-regime metrics in the same format used by
v5/v6/v7 so that existing table/figure generation scripts automatically
include the learned router results.

Usage
-----
    python -m src.routing.learned_router.eval \\
        --config configs/learned_router_default.yaml \\
        [--model-dir outputs/learned_router/model]

Output files (per regime, under ``<output_dir>/eval/<regime>/``)
----------------------------------------------------------------
``metrics.json``
    Accuracy, avg_cost, revise_rate, oracle_gap, n.
``budget_curve.csv``
    Per-threshold cost–accuracy curve (for plotting alongside v6/v7 and oracle).
``per_query_decisions.csv``
    Per-query escalation decision, ground-truth label, and probabilities.

Aggregated outputs (under ``<output_dir>/eval/``)
-------------------------------------------------
``learned_router_eval_summary.csv``
    One row per regime with operating-point metrics.
``learned_router_eval_summary.json``
    Same data as JSON.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config (mirrors train.py)
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "model_type": "logistic_regression",
    "output_dir": "outputs/learned_router",
    "threshold": None,  # None → load from train_summary.json
    "threshold_sweep": {"start": 0.0, "stop": 1.0, "step": 0.05},
    "cheap_cost": 1.0,
    "expensive_cost": 2.0,
}


def _load_config(config_path: str | Path | None) -> dict:
    cfg = dict(_DEFAULT_CONFIG)
    if config_path is not None:
        with open(config_path) as fh:
            user = yaml.safe_load(fh) or {}
        cfg.update(user)
    return cfg


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class RegimeEvalResult:
    """Per-regime evaluation result for the learned router."""

    regime: str
    model_type: str
    threshold: float
    accuracy: float
    avg_cost: float
    revise_rate: float
    oracle_accuracy: float
    oracle_gap: float
    n: int
    note: str = ""

    def to_summary_dict(self) -> dict:
        return {
            "regime": self.regime,
            "policy": f"learned_router_{self.model_type}",
            "threshold": self.threshold,
            "accuracy": self.accuracy,
            "avg_cost": self.avg_cost,
            "revise_rate": self.revise_rate,
            "oracle_accuracy": self.oracle_accuracy,
            "oracle_gap": self.oracle_gap,
            "n": self.n,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Oracle accuracy helper
# ---------------------------------------------------------------------------


def _oracle_accuracy(df: pd.DataFrame) -> float:
    """Compute oracle routing accuracy (best of RG and DPR per query)."""
    best = np.maximum(
        df["reasoning_correct"].fillna(0).values.astype(int),
        df["revise_correct"].fillna(0).values.astype(int),
    )
    return float(best.mean())


# ---------------------------------------------------------------------------
# Per-regime evaluation
# ---------------------------------------------------------------------------


def evaluate_regime(
    regime: str,
    df: pd.DataFrame,
    router: object,
    threshold: float,
    cheap_cost: float = 1.0,
    expensive_cost: float = 2.0,
    thresholds_for_sweep: list[float] | None = None,
) -> tuple[RegimeEvalResult, list[dict], list[dict]]:
    """Evaluate a trained router on one regime's DataFrame.

    Returns
    -------
    result : RegimeEvalResult
        Operating-point metrics.
    budget_curve : list[dict]
        Per-threshold cost–accuracy rows.
    per_query : list[dict]
        Per-query decisions (threshold applied at *threshold*).
    """
    from src.routing.learned_router.features import build_feature_matrix

    X, y_true, feat_names = build_feature_matrix(df)
    proba = router.predict_proba(X)

    # Operating point
    preds = (proba >= threshold).astype(int)
    revise_rate = float(preds.mean())
    avg_cost = cheap_cost * (1 - revise_rate) + expensive_cost * revise_rate

    # Routing accuracy: use revise_correct when escalated, reasoning_correct otherwise
    reasoning_corr = df["reasoning_correct"].fillna(0).values.astype(int)
    revise_corr = df["revise_correct"].fillna(0).values.astype(int)
    correct = preds * revise_corr + (1 - preds) * reasoning_corr
    accuracy = float(correct.mean())

    oracle_acc = _oracle_accuracy(df)
    oracle_gap = round(oracle_acc - accuracy, 6)

    result = RegimeEvalResult(
        regime=regime,
        model_type=getattr(router, "name", "router"),
        threshold=threshold,
        accuracy=round(accuracy, 6),
        avg_cost=round(avg_cost, 6),
        revise_rate=round(revise_rate, 6),
        oracle_accuracy=round(oracle_acc, 6),
        oracle_gap=oracle_gap,
        n=len(df),
    )

    # Budget curve (threshold sweep)
    sweep_taus = thresholds_for_sweep or [round(t * 0.05, 2) for t in range(21)]
    budget_curve: list[dict] = []
    for tau in sweep_taus:
        p = (proba >= tau).astype(int)
        rr = float(p.mean())
        ac = cheap_cost * (1 - rr) + expensive_cost * rr
        corr_t = p * revise_corr + (1 - p) * reasoning_corr
        acc_t = float(corr_t.mean())
        budget_curve.append(
            {
                "regime": regime,
                "threshold": round(tau, 4),
                "accuracy": round(acc_t, 6),
                "avg_cost": round(ac, 6),
                "revise_rate": round(rr, 6),
                "n": len(df),
            }
        )

    # Per-query decisions
    qids = df.get("question_id", pd.Series(range(len(df)))).values
    per_query: list[dict] = []
    for i in range(len(df)):
        per_query.append(
            {
                "question_id": qids[i],
                "p_escalate": round(float(proba[i]), 6),
                "decision": "DPR" if preds[i] == 1 else "RG",
                "reasoning_correct": int(reasoning_corr[i]),
                "revise_correct": int(revise_corr[i]),
                "correct": int(correct[i]),
            }
        )

    return result, budget_curve, per_query


# ---------------------------------------------------------------------------
# Main evaluation function (importable for testing)
# ---------------------------------------------------------------------------


def evaluate(config: dict) -> list[RegimeEvalResult]:
    """Evaluate the trained router on all manuscript regimes.

    Parameters
    ----------
    config:
        Configuration dictionary.

    Returns
    -------
    list[RegimeEvalResult]
        One result per regime.
    """
    from src.routing.learned_router.features import REGIME_FILES, load_regime_df
    from src.routing.learned_router.models import LogisticRegressionRouter, MLPRouter

    out_dir = Path(config["output_dir"])
    model_dir = out_dir / "model"
    eval_dir = out_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Load trained model
    model_type = config.get("model_type", "logistic_regression")
    _LOG.info("Loading model from %s", model_dir)
    if model_type == "logistic_regression":
        router = LogisticRegressionRouter.load(model_dir)
    elif model_type == "mlp":
        router = MLPRouter.load(model_dir)
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    # Determine operating threshold
    threshold = config.get("threshold")
    if threshold is None:
        # Try to read from train_summary.json
        summary_path = out_dir / "train_summary.json"
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            threshold = float(summary.get("selected_threshold", 0.5))
            _LOG.info("Using threshold τ=%.2f from train_summary.json", threshold)
        else:
            threshold = 0.5
            _LOG.info("No train_summary.json found; using default τ=0.5")

    cheap_cost = float(config.get("cheap_cost", 1.0))
    expensive_cost = float(config.get("expensive_cost", 2.0))

    sweep_cfg = config.get("threshold_sweep", {})
    start = float(sweep_cfg.get("start", 0.0))
    stop = float(sweep_cfg.get("stop", 1.0))
    step = float(sweep_cfg.get("step", 0.05))
    sweep_taus = [round(float(v), 4) for v in np.linspace(start, stop, int(round((stop - start) / step)) + 1)]

    all_results: list[RegimeEvalResult] = []
    all_budget_rows: list[dict] = []

    for regime, csv_path in REGIME_FILES.items():
        _LOG.info("Evaluating regime: %s", regime)
        try:
            df = load_regime_df(csv_path)
        except FileNotFoundError:
            _LOG.warning("Regime file not found, skipping: %s", csv_path)
            continue

        result, budget_curve, per_query = evaluate_regime(
            regime=regime,
            df=df,
            router=router,
            threshold=float(threshold),
            cheap_cost=cheap_cost,
            expensive_cost=expensive_cost,
            thresholds_for_sweep=sweep_taus,
        )
        all_results.append(result)
        all_budget_rows.extend(budget_curve)

        # Write per-regime outputs
        regime_dir = eval_dir / regime
        regime_dir.mkdir(parents=True, exist_ok=True)

        # metrics.json
        (regime_dir / "metrics.json").write_text(json.dumps(asdict(result), indent=2))

        # budget_curve.csv
        if budget_curve:
            _write_csv(budget_curve, regime_dir / "budget_curve.csv")

        # per_query_decisions.csv
        if per_query:
            _write_csv(per_query, regime_dir / "per_query_decisions.csv")

        _LOG.info(
            "  %s: acc=%.3f avg_cost=%.3f revise_rate=%.3f oracle_gap=%.3f",
            regime, result.accuracy, result.avg_cost, result.revise_rate, result.oracle_gap,
        )

    # Write aggregated outputs
    summary_rows = [r.to_summary_dict() for r in all_results]
    if summary_rows:
        _write_csv(summary_rows, eval_dir / "learned_router_eval_summary.csv")
    (eval_dir / "learned_router_eval_summary.json").write_text(
        json.dumps([asdict(r) for r in all_results], indent=2)
    )

    if all_budget_rows:
        _write_csv(all_budget_rows, eval_dir / "learned_router_budget_curves.csv")

    _LOG.info("Evaluation complete. Results written to %s", eval_dir)
    return all_results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the trained learned router on all manuscript regimes."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Override: directory containing the saved model.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override: output directory.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Override: decision threshold τ.",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.threshold is not None:
        cfg["threshold"] = args.threshold

    results = evaluate(cfg)

    print(
        f"\n{'Regime':<25} {'Accuracy':>10} {'AvgCost':>10} "
        f"{'RevRate':>10} {'OracleGap':>12}"
    )
    print("-" * 72)
    for r in results:
        print(
            f"{r.regime:<25} {r.accuracy:>10.3f} {r.avg_cost:>10.3f} "
            f"{r.revise_rate:>10.3f} {r.oracle_gap:>12.3f}"
        )


if __name__ == "__main__":
    main()
