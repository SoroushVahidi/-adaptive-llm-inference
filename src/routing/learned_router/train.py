"""Training entry point for the learned router baseline.

Trains a logistic regression or MLP router on all manuscript regimes except
the nominated test regime, then saves the model, scaler, and a validation
summary.

Usage
-----
    python -m src.routing.learned_router.train \\
        --config configs/learned_router_default.yaml \\
        [--test-regime hard_gsm8k_100]

Config keys (YAML)
------------------
See ``configs/learned_router_default.yaml`` for a fully annotated example.

Key output files
----------------
``<output_dir>/model/``
    Saved model weights and scaler.
``<output_dir>/train_summary.json``
    Validation accuracy, selected threshold τ, and training metadata.
``<output_dir>/threshold_sweep.csv``
    Per-threshold accuracy / avg_cost / revise_rate on the validation set.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default config
# ---------------------------------------------------------------------------

_DEFAULT_CONFIG = {
    "model_type": "logistic_regression",
    "test_regime": None,
    "val_fraction": 0.2,
    "random_seed": 42,
    "output_dir": "outputs/learned_router",
    # Logistic regression hyperparameters
    "logreg": {"C": 1.0, "max_iter": 1000},
    # MLP hyperparameters
    "mlp": {
        "hidden_sizes": [64, 32],
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "epochs": 50,
        "batch_size": 32,
        "cost_weight": 0.0,
        "dropout": 0.1,
    },
    # Threshold sweep for validation operating point selection
    "threshold_sweep": {"start": 0.05, "stop": 0.95, "step": 0.05},
    # Feature configuration
    "feature_cols": None,  # None → use all FEATURE_COLS
    # Routing cost model
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
# Threshold sweep helpers
# ---------------------------------------------------------------------------


def _sweep_thresholds(
    router: object,
    X: np.ndarray,
    y_true: np.ndarray,
    df_val: "pd.DataFrame | None" = None,
    thresholds: list[float] | None = None,
    cheap_cost: float = 1.0,
    expensive_cost: float = 2.0,
) -> list[dict]:
    """Sweep decision thresholds and compute accuracy / avg_cost / revise_rate."""
    if thresholds is None:
        thresholds = [round(t * 0.05, 2) for t in range(1, 20)]

    proba = router.predict_proba(X)
    rows = []
    for tau in thresholds:
        preds = (proba >= tau).astype(int)
        revise_rate = float(preds.mean())
        accuracy = float(np.mean(preds == y_true))
        avg_cost = cheap_cost * (1 - revise_rate) + expensive_cost * revise_rate
        rows.append(
            {
                "threshold": round(tau, 4),
                "accuracy": round(accuracy, 6),
                "avg_cost": round(avg_cost, 6),
                "revise_rate": round(revise_rate, 6),
                "n": len(y_true),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Main training function (importable for testing)
# ---------------------------------------------------------------------------


def train(config: dict) -> dict:
    """Train the learned router and save outputs.  Returns the training summary.

    Parameters
    ----------
    config:
        Configuration dictionary (see ``_DEFAULT_CONFIG`` for keys).

    Returns
    -------
    dict
        Training summary including validation accuracy and selected threshold.
    """
    from src.routing.learned_router.features import (
        FEATURE_COLS,
        REGIME_FILES,
        build_training_dataset,
        load_regime_df,
        build_feature_matrix,
    )
    from src.routing.learned_router.models import make_router

    out_dir = Path(config["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_cols = config.get("feature_cols") or FEATURE_COLS
    test_regime = config.get("test_regime")

    _LOG.info("Loading training data (test_regime=%s)", test_regime)
    X_train, y_train, X_val, y_val, X_test, y_test, feat_names = build_training_dataset(
        regime_files=REGIME_FILES,
        test_regime=test_regime,
        val_fraction=float(config.get("val_fraction", 0.2)),
        random_seed=int(config.get("random_seed", 42)),
        feature_cols=feature_cols,
    )

    _LOG.info(
        "Dataset sizes — train=%d pos=%d, val=%d pos=%d, test=%d pos=%d",
        len(X_train), int(y_train.sum()),
        len(X_val), int(y_val.sum()),
        len(X_test), int(y_test.sum()),
    )

    # Build router
    model_type = config.get("model_type", "logistic_regression")
    if model_type == "logistic_regression":
        kwargs = config.get("logreg", {})
    elif model_type == "mlp":
        kwargs = config.get("mlp", {})
    else:
        raise ValueError(f"Unknown model_type: {model_type!r}")

    router = make_router(model_type, **kwargs)
    router._feature_names = feat_names

    _LOG.info("Fitting %s router …", model_type)
    router.fit(X_train, y_train, X_val if len(X_val) else None, y_val if len(y_val) else None)

    # Threshold sweep on validation set
    sweep_cfg = config.get("threshold_sweep", {})
    start = float(sweep_cfg.get("start", 0.05))
    stop = float(sweep_cfg.get("stop", 0.95))
    step = float(sweep_cfg.get("step", 0.05))
    thresholds = [round(start + i * step, 4) for i in range(int((stop - start) / step) + 1)]

    cheap_cost = float(config.get("cheap_cost", 1.0))
    expensive_cost = float(config.get("expensive_cost", 2.0))

    sweep_rows: list[dict] = []
    best_tau = 0.5
    if len(X_val) > 0:
        sweep_rows = _sweep_thresholds(
            router, X_val, y_val,
            thresholds=thresholds,
            cheap_cost=cheap_cost,
            expensive_cost=expensive_cost,
        )
        # Select threshold with best validation accuracy
        best_tau = max(sweep_rows, key=lambda r: r["accuracy"])["threshold"]
        _LOG.info("Best validation threshold τ=%.2f", best_tau)

        # Write threshold sweep CSV
        sweep_path = out_dir / "threshold_sweep.csv"
        if sweep_rows:
            with sweep_path.open("w", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(sweep_rows[0].keys()))
                writer.writeheader()
                writer.writerows(sweep_rows)
            _LOG.info("Threshold sweep written to %s", sweep_path)
    else:
        best_tau = router.best_threshold(X_train, y_train) if len(X_train) > 0 else 0.5

    # Save model
    model_dir = out_dir / "model"
    router.save(model_dir)
    _LOG.info("Model saved to %s", model_dir)

    # Compute validation metrics at best threshold
    val_metrics: dict = {}
    if len(X_val) > 0:
        val_preds = router.predict(X_val, threshold=best_tau)
        val_acc = float(np.mean(val_preds == y_val))
        val_revise_rate = float(val_preds.mean())
        val_avg_cost = cheap_cost * (1 - val_revise_rate) + expensive_cost * val_revise_rate
        val_metrics = {
            "val_accuracy": round(val_acc, 6),
            "val_avg_cost": round(val_avg_cost, 6),
            "val_revise_rate": round(val_revise_rate, 6),
            "val_n": len(X_val),
        }

    # Build summary
    summary = {
        "model_type": model_type,
        "test_regime": test_regime,
        "selected_threshold": best_tau,
        "feature_names": feat_names,
        "n_features": len(feat_names),
        "train_n": len(X_train),
        "train_n_pos": int(y_train.sum()),
        "val_n": len(X_val),
        "val_n_pos": int(y_val.sum()),
        "test_n": len(X_test),
        "test_n_pos": int(y_test.sum()),
        **val_metrics,
        "config": {k: v for k, v in config.items() if k != "feature_cols"},
    }

    summary_path = out_dir / "train_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _LOG.info("Training summary written to %s", summary_path)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Train the learned router (logistic regression or MLP) "
        "on manuscript routing artifacts."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to YAML config file (default: uses built-in defaults).",
    )
    parser.add_argument(
        "--test-regime",
        default=None,
        help="Regime to hold out as test set (overrides config).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory (overrides config).",
    )
    parser.add_argument(
        "--model-type",
        default=None,
        choices=["logistic_regression", "mlp"],
        help="Model type (overrides config).",
    )
    args = parser.parse_args(argv)

    cfg = _load_config(args.config)
    if args.test_regime is not None:
        cfg["test_regime"] = args.test_regime
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    if args.model_type is not None:
        cfg["model_type"] = args.model_type

    _LOG.info("Config: %s", json.dumps({k: v for k, v in cfg.items() if k != "feature_cols"}, indent=2))

    summary = train(cfg)

    _LOG.info(
        "Done. val_accuracy=%.3f, threshold=%.2f",
        summary.get("val_accuracy", float("nan")),
        summary["selected_threshold"],
    )


if __name__ == "__main__":
    main()
