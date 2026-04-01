"""LearnedRouterPolicy: plug-in policy for the manuscript evaluation pipeline.

Wraps a trained :class:`~src.routing.learned_router.models.RouterBase` model
so that it can be used in the same policy registry as v5/v6/v7.

Each query is evaluated offline (no LLM calls).  Given pre-computed features
from the enriched routing CSV, the router outputs ``p_escalate`` and the
policy applies a decision threshold τ to choose between:

- ``reasoning_greedy`` (RG, cheap route) when ``p_escalate < τ``
- ``direct_plus_revise`` (DPR, expensive revise route) when ``p_escalate >= τ``

Usage
-----
::

    from src.policies.learned_router_policy import LearnedRouterPolicy

    policy = LearnedRouterPolicy.from_saved("outputs/learned_router/model")
    decisions = policy.decide_batch(df)   # returns pd.Series of str
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.routing.learned_router.features import FEATURE_COLS, build_feature_matrix


# ---------------------------------------------------------------------------
# Config dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LearnedRouterPolicyConfig:
    """Configuration for the learned router policy.

    Parameters
    ----------
    model_type:
        ``"logistic_regression"`` or ``"mlp"``.
    threshold:
        Decision threshold τ.  Escalate to DPR if ``p_escalate >= threshold``.
    model_dir:
        Path to the saved model directory (output of ``train.py``).
    feature_cols:
        Feature columns to use.  Defaults to ``FEATURE_COLS``.
    cheap_action:
        Name of the cheap routing action (default: ``"reasoning_greedy"``).
    expensive_action:
        Name of the expensive routing action (default: ``"direct_plus_revise"``).
    """

    model_type: str = "logistic_regression"
    threshold: float = 0.5
    model_dir: str = "outputs/learned_router/model"
    feature_cols: tuple[str, ...] = field(default_factory=lambda: tuple(FEATURE_COLS))
    cheap_action: str = "reasoning_greedy"
    expensive_action: str = "direct_plus_revise"


# ---------------------------------------------------------------------------
# Policy class
# ---------------------------------------------------------------------------


class LearnedRouterPolicy:
    """Offline learned routing policy for the manuscript evaluation pipeline.

    The policy is loaded from a pre-trained model directory produced by
    :mod:`src.routing.learned_router.train`.

    Parameters
    ----------
    router:
        A fitted :class:`~src.routing.learned_router.models.RouterBase` instance.
    config:
        Policy configuration.
    """

    name = "learned_router"

    def __init__(
        self,
        router: Any,
        config: LearnedRouterPolicyConfig | None = None,
    ) -> None:
        self._router = router
        self._config = config or LearnedRouterPolicyConfig()

    # ------------------------------------------------------------------
    # Factory methods
    # ------------------------------------------------------------------

    @classmethod
    def from_saved(
        cls,
        model_dir: str | Path,
        model_type: str = "logistic_regression",
        threshold: float = 0.5,
        **kwargs: Any,
    ) -> "LearnedRouterPolicy":
        """Load a trained router from *model_dir*.

        Parameters
        ----------
        model_dir:
            Directory produced by :func:`src.routing.learned_router.train.train`.
        model_type:
            ``"logistic_regression"`` or ``"mlp"``.
        threshold:
            Decision threshold τ.  Can also be loaded from ``train_summary.json``
            via :meth:`from_train_output`.
        """
        from src.routing.learned_router.models import (
            LogisticRegressionRouter,
            MLPRouter,
        )

        path = Path(model_dir)
        if model_type == "logistic_regression":
            router = LogisticRegressionRouter.load(path)
        elif model_type == "mlp":
            router = MLPRouter.load(path)
        else:
            raise ValueError(f"Unknown model_type: {model_type!r}")

        config = LearnedRouterPolicyConfig(
            model_type=model_type,
            threshold=threshold,
            model_dir=str(path),
            **kwargs,
        )
        return cls(router, config)

    @classmethod
    def from_train_output(
        cls,
        output_dir: str | Path,
        model_type: str | None = None,
    ) -> "LearnedRouterPolicy":
        """Load a trained router from a training output directory.

        Reads ``train_summary.json`` to retrieve the model type and selected
        threshold τ, then loads the model from ``model/``.

        Parameters
        ----------
        output_dir:
            Directory written by :func:`src.routing.learned_router.train.train`
            (contains ``train_summary.json`` and ``model/``).
        model_type:
            Override the model type (default: read from ``train_summary.json``).
        """
        out = Path(output_dir)
        summary_path = out / "train_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"train_summary.json not found in {out}")

        summary = __import__("json").loads(summary_path.read_text())
        mtype = model_type or summary["model_type"]
        tau = float(summary.get("selected_threshold", 0.5))

        return cls.from_saved(out / "model", model_type=mtype, threshold=tau)

    # ------------------------------------------------------------------
    # Decision methods
    # ------------------------------------------------------------------

    def decide(self, features: np.ndarray) -> str:
        """Route a single query.

        Parameters
        ----------
        features:
            1-D feature vector of shape (d,).

        Returns
        -------
        str
            ``"direct_plus_revise"`` or ``"reasoning_greedy"``.
        """
        proba = self._router.predict_proba(features.reshape(1, -1))[0]
        if proba >= self._config.threshold:
            return self._config.expensive_action
        return self._config.cheap_action

    def decide_batch(self, df: pd.DataFrame) -> pd.Series:
        """Route a batch of queries from a routing DataFrame.

        Parameters
        ----------
        df:
            DataFrame containing the feature columns (from an enriched routing
            CSV produced by the pipeline).  Missing columns are zero-filled.

        Returns
        -------
        pd.Series of str
            Per-query action (``"reasoning_greedy"`` or ``"direct_plus_revise"``).
        """
        X, _, _ = build_feature_matrix(df, feature_cols=list(self._config.feature_cols))
        proba = self._router.predict_proba(X)
        decisions = np.where(
            proba >= self._config.threshold,
            self._config.expensive_action,
            self._config.cheap_action,
        )
        return pd.Series(decisions, index=df.index)

    def predict_proba_batch(self, df: pd.DataFrame) -> np.ndarray:
        """Return escalation probabilities for a batch of queries.

        Parameters
        ----------
        df:
            DataFrame with feature columns.

        Returns
        -------
        np.ndarray of shape (n,)
            Escalation probability ``p_escalate`` per query.
        """
        X, _, _ = build_feature_matrix(df, feature_cols=list(self._config.feature_cols))
        return self._router.predict_proba(X)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def threshold(self) -> float:
        """Current decision threshold τ."""
        return self._config.threshold

    def with_threshold(self, threshold: float) -> "LearnedRouterPolicy":
        """Return a new policy with a different threshold (for sweeping τ)."""
        import dataclasses

        new_config = dataclasses.replace(self._config, threshold=threshold)
        return LearnedRouterPolicy(self._router, new_config)
