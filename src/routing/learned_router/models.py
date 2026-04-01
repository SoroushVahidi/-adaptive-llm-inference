"""Router model definitions for the learned routing baseline.

Implements two router architectures in the style of FrugalGPT / RouteLLM:

1. **LogisticRegressionRouter** — a sklearn logistic regression model that
   outputs a calibrated ``p_escalate = P(DPR > RG | features)`` score.
   Fast, interpretable, and always available (sklearn is a dev dependency).

2. **MLPRouter** — a 1–2 layer PyTorch MLP that outputs ``p_escalate``.
   Supports cost-aware training where false positives (unnecessary escalations)
   incur an extra penalty.  Gracefully skipped when PyTorch is not installed.

Both models share a common :class:`RouterBase` interface:
- ``fit(X_train, y_train, X_val, y_val)``
- ``predict_proba(X)`` → float array of escalation probabilities
- ``predict(X, threshold)`` → binary escalation decisions
- ``save(path)`` / ``load(path)`` — for persistence

Cost model
----------
The cost-aware MLP loss adds a weighted penalty for false positives:

    loss = BCE(y_hat, y) + cost_weight * FP_rate

where ``cost_weight`` controls the budget–accuracy tradeoff.  Setting
``cost_weight=0`` recovers standard binary cross-entropy.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np

_LOG = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Availability probes
# ---------------------------------------------------------------------------

_SKLEARN_AVAILABLE: bool | None = None
_TORCH_AVAILABLE: bool | None = None


def _has_sklearn() -> bool:
    global _SKLEARN_AVAILABLE  # noqa: PLW0603
    if _SKLEARN_AVAILABLE is None:
        try:
            import sklearn  # noqa: F401
            _SKLEARN_AVAILABLE = True
        except ImportError:
            _SKLEARN_AVAILABLE = False
    return _SKLEARN_AVAILABLE


def _has_torch() -> bool:
    global _TORCH_AVAILABLE  # noqa: PLW0603
    if _TORCH_AVAILABLE is None:
        try:
            import torch  # noqa: F401
            _TORCH_AVAILABLE = True
        except ImportError:
            _TORCH_AVAILABLE = False
    return _TORCH_AVAILABLE


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class RouterBase(ABC):
    """Abstract interface for learned routing models."""

    name: str = "router"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "RouterBase":
        """Train the router on *X_train* / *y_train*.

        Parameters
        ----------
        X_train:
            Feature matrix of shape (n, d).
        y_train:
            Binary label vector of shape (n,).  1 = escalate to DPR.
        X_val, y_val:
            Optional validation set for early stopping / threshold selection.
        """

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return escalation probabilities of shape (n,)."""

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary escalation decisions.

        Parameters
        ----------
        X:
            Feature matrix of shape (n, d).
        threshold:
            Escalate if ``p_escalate >= threshold``.

        Returns
        -------
        np.ndarray of shape (n,), dtype int, values in {0, 1}.
        """
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist model to *path*."""

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "RouterBase":
        """Load model from *path*."""

    def best_threshold(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
        thresholds: list[float] | None = None,
        metric: str = "accuracy",
    ) -> float:
        """Select the decision threshold that maximises *metric* on the validation set.

        Parameters
        ----------
        thresholds:
            Grid of thresholds to sweep.  Defaults to 0.05 … 0.95 in steps of 0.05.
        metric:
            ``"accuracy"`` (default) or ``"f1"``.

        Returns
        -------
        float
            Best threshold value.
        """
        if thresholds is None:
            thresholds = [round(t * 0.05, 2) for t in range(1, 20)]

        proba = self.predict_proba(X_val)
        best_tau, best_score = 0.5, -1.0
        for tau in thresholds:
            preds = (proba >= tau).astype(int)
            if metric == "f1":
                tp = int(np.sum((preds == 1) & (y_val == 1)))
                fp = int(np.sum((preds == 1) & (y_val == 0)))
                fn = int(np.sum((preds == 0) & (y_val == 1)))
                score = tp / (tp + 0.5 * (fp + fn)) if (tp + fp + fn) > 0 else 0.0
            else:
                score = float(np.mean(preds == y_val))
            if score > best_score:
                best_score = score
                best_tau = tau
        return best_tau


# ---------------------------------------------------------------------------
# Logistic Regression Router (sklearn)
# ---------------------------------------------------------------------------


class LogisticRegressionRouter(RouterBase):
    """Logistic regression router using scikit-learn.

    Trained with ``class_weight='balanced'`` to handle class imbalance
    (the escalate class is typically only 5–20 % of queries).

    Parameters
    ----------
    C:
        Inverse regularisation strength.
    max_iter:
        Maximum number of solver iterations.
    random_state:
        Random seed for reproducibility.
    """

    name = "logistic_regression"

    def __init__(self, C: float = 1.0, max_iter: int = 1000, random_state: int = 42) -> None:
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        self._clf: Any = None
        self._scaler: Any = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "LogisticRegressionRouter":
        if not _has_sklearn():
            raise ImportError("scikit-learn is required for LogisticRegressionRouter")
        if len(X_train) == 0:
            raise ValueError("Cannot fit on empty training set")

        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train)

        self._clf = LogisticRegression(
            C=self.C,
            class_weight="balanced",
            max_iter=self.max_iter,
            random_state=self.random_state,
        )
        self._clf.fit(X_scaled, y_train)

        if X_val is not None and y_val is not None and len(X_val) > 0:
            val_acc = float(np.mean(self._clf.predict(self._scaler.transform(X_val)) == y_val))
            _LOG.info("LogReg val accuracy: %.3f", val_acc)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._clf is None:
            raise RuntimeError("Model not fitted; call fit() first")
        X_scaled = self._scaler.transform(X)
        return self._clf.predict_proba(X_scaled)[:, 1].astype(np.float32)

    def feature_importances(self, feature_names: list[str] | None = None) -> dict[str, float]:
        """Return absolute logistic regression coefficients as importance scores."""
        if self._clf is None:
            return {}
        coef = np.abs(self._clf.coef_[0])
        names = feature_names or self._feature_names or [f"f{i}" for i in range(len(coef))]
        return {n: float(c) for n, c in zip(names, coef)}

    def save(self, path: str | Path) -> None:
        """Persist model to a directory.  Creates ``model.pkl`` and ``meta.json``."""
        import pickle

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        with (out / "model.pkl").open("wb") as fh:
            pickle.dump({"clf": self._clf, "scaler": self._scaler}, fh)
        (out / "meta.json").write_text(
            json.dumps(
                {
                    "model_type": self.name,
                    "C": self.C,
                    "max_iter": self.max_iter,
                    "random_state": self.random_state,
                    "feature_names": self._feature_names,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "LogisticRegressionRouter":
        import pickle

        out = Path(path)
        meta = json.loads((out / "meta.json").read_text())
        instance = cls(C=meta["C"], max_iter=meta["max_iter"], random_state=meta["random_state"])
        with (out / "model.pkl").open("rb") as fh:
            state = pickle.load(fh)  # noqa: S301
        instance._clf = state["clf"]
        instance._scaler = state["scaler"]
        instance._feature_names = meta.get("feature_names", [])
        return instance


# ---------------------------------------------------------------------------
# MLP Router (PyTorch)
# ---------------------------------------------------------------------------


class _MLP:
    """Minimal PyTorch MLP implementation with lazy import."""

    def __init__(self, in_dim: int, hidden_sizes: list[int], dropout: float = 0.1) -> None:
        import torch
        import torch.nn as nn

        layers: list[Any] = []
        prev = in_dim
        for h in hidden_sizes:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
        self.in_dim = in_dim  # stored for serialisation
        self._torch = torch
        self._nn = nn

    def __call__(self, x: Any) -> Any:
        return self.net(x)

    def parameters(self) -> Any:
        return self.net.parameters()

    def train(self) -> None:
        self.net.train()

    def eval(self) -> None:
        self.net.eval()

    def state_dict(self) -> Any:
        return self.net.state_dict()

    def load_state_dict(self, state: Any) -> None:
        self.net.load_state_dict(state)


class MLPRouter(RouterBase):
    """Shallow MLP router with optional cost-aware training (PyTorch).

    Parameters
    ----------
    hidden_sizes:
        List of hidden layer sizes.  E.g. ``[64, 32]`` for a two-layer MLP.
    lr:
        Adam learning rate.
    weight_decay:
        L2 regularisation coefficient.
    epochs:
        Number of training epochs.
    batch_size:
        Mini-batch size.
    cost_weight:
        Extra penalty for false positives (unnecessary escalations).
        ``0`` = standard BCE; ``> 0`` = cost-aware BCE.
    dropout:
        Dropout probability applied to hidden layers.
    random_state:
        Random seed.
    """

    name = "mlp"

    def __init__(
        self,
        hidden_sizes: list[int] | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 32,
        cost_weight: float = 0.0,
        dropout: float = 0.1,
        random_state: int = 42,
    ) -> None:
        self.hidden_sizes = hidden_sizes or [64, 32]
        self.lr = lr
        self.weight_decay = weight_decay
        self.epochs = epochs
        self.batch_size = batch_size
        self.cost_weight = cost_weight
        self.dropout = dropout
        self.random_state = random_state
        self._net: Any = None
        self._scaler: Any = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "MLPRouter":
        if not _has_torch():
            raise ImportError(
                "PyTorch is required for MLPRouter. Install with: pip install torch"
            )
        if not _has_sklearn():
            raise ImportError("scikit-learn is required for feature scaling in MLPRouter")
        if len(X_train) == 0:
            raise ValueError("Cannot fit on empty training set")

        import torch
        from sklearn.preprocessing import StandardScaler

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X_train).astype(np.float32)

        in_dim = X_scaled.shape[1]
        self._net = _MLP(in_dim, self.hidden_sizes, self.dropout)

        optimizer = torch.optim.Adam(
            self._net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

        y_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

        # Class imbalance: positive weight = n_neg / n_pos
        n_pos = int(y_tensor.sum())
        n_neg = len(y_tensor) - n_pos
        pos_weight = torch.tensor([n_neg / max(n_pos, 1)], dtype=torch.float32)
        bce = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        self._net.train()
        n = len(X_tensor)
        for epoch in range(self.epochs):
            # Shuffle
            perm = torch.randperm(n)
            X_tensor = X_tensor[perm]
            y_tensor = y_tensor[perm]

            for start in range(0, n, self.batch_size):
                X_b = X_tensor[start : start + self.batch_size]
                y_b = y_tensor[start : start + self.batch_size]

                logits = self._net(X_b).squeeze(1)
                loss = bce(logits, y_b)

                # Cost-aware penalty: penalise false positives additionally
                if self.cost_weight > 0:
                    proba = torch.sigmoid(logits)
                    fp_rate = (proba * (1 - y_b)).mean()
                    loss = loss + self.cost_weight * fp_rate

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if X_val is not None and y_val is not None and len(X_val) > 0 and (epoch + 1) % 10 == 0:
                self._net.eval()
                with torch.no_grad():
                    X_v = torch.tensor(
                        self._scaler.transform(X_val).astype(np.float32),
                        dtype=torch.float32,
                    )
                    logits_v = self._net(X_v).squeeze(1)
                    preds_v = (torch.sigmoid(logits_v) >= 0.5).float()
                    val_acc = float((preds_v == torch.tensor(y_val, dtype=torch.float32)).float().mean())
                _LOG.info("MLP epoch %d/%d — val acc %.3f", epoch + 1, self.epochs, val_acc)
                self._net.train()

        self._net.eval()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._net is None:
            raise RuntimeError("Model not fitted; call fit() first")
        import torch

        X_scaled = self._scaler.transform(X).astype(np.float32)
        X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
        with torch.no_grad():
            logits = self._net(X_tensor).squeeze(1)
            proba = torch.sigmoid(logits).numpy()
        return proba.astype(np.float32)

    def save(self, path: str | Path) -> None:
        import torch

        out = Path(path)
        out.mkdir(parents=True, exist_ok=True)
        torch.save(self._net.state_dict(), out / "weights.pt")

        import pickle
        with (out / "scaler.pkl").open("wb") as fh:
            pickle.dump(self._scaler, fh)

        (out / "meta.json").write_text(
            json.dumps(
                {
                    "model_type": self.name,
                    "hidden_sizes": self.hidden_sizes,
                    "lr": self.lr,
                    "weight_decay": self.weight_decay,
                    "epochs": self.epochs,
                    "batch_size": self.batch_size,
                    "cost_weight": self.cost_weight,
                    "dropout": self.dropout,
                    "random_state": self.random_state,
                    "in_dim": self._net.net[0].in_features if self._net else 0,
                    "feature_names": self._feature_names,
                },
                indent=2,
            )
        )

    @classmethod
    def load(cls, path: str | Path) -> "MLPRouter":
        import pickle
        import torch

        out = Path(path)
        meta = json.loads((out / "meta.json").read_text())
        instance = cls(
            hidden_sizes=meta["hidden_sizes"],
            lr=meta["lr"],
            weight_decay=meta["weight_decay"],
            epochs=meta["epochs"],
            batch_size=meta["batch_size"],
            cost_weight=meta["cost_weight"],
            dropout=meta["dropout"],
            random_state=meta["random_state"],
        )
        with (out / "scaler.pkl").open("rb") as fh:
            instance._scaler = pickle.load(fh)  # noqa: S301

        in_dim = meta["in_dim"]
        instance._net = _MLP(in_dim, meta["hidden_sizes"], meta["dropout"])
        instance._net.load_state_dict(torch.load(out / "weights.pt", weights_only=True))
        instance._net.eval()
        instance._feature_names = meta.get("feature_names", [])
        return instance


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def make_router(model_type: str, **kwargs: Any) -> RouterBase:
    """Instantiate a router by name.

    Parameters
    ----------
    model_type:
        ``"logistic_regression"`` or ``"mlp"``.
    **kwargs:
        Forwarded to the router constructor.

    Returns
    -------
    RouterBase
    """
    if model_type == "logistic_regression":
        return LogisticRegressionRouter(**kwargs)
    if model_type == "mlp":
        return MLPRouter(**kwargs)
    raise ValueError(
        f"Unknown model_type {model_type!r}. "
        "Choose 'logistic_regression' or 'mlp'."
    )
