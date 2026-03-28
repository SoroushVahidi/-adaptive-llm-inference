"""Lightweight router baseline for strategy prediction.

Given the routing dataset (from ``src.datasets.routing_dataset``), this module
trains and evaluates two simple prediction baselines:

**Binary task** — predict ``direct_already_optimal`` (0 or 1).
**Multiclass task** — predict ``best_accuracy_strategy`` (one of the oracle
    strategy names).

Three models are tried in order of sophistication:

1. **MajorityBaseline** — always predicts the most common class.  Zero external
   dependencies; always available.
2. **DecisionTreeRouter** — a shallow decision tree (max depth 3).  Uses
   ``scikit-learn`` when available; falls back gracefully when absent.
3. **LogisticRegressionRouter** — binary logistic regression.  Uses
   ``scikit-learn``; skipped on the multiclass task and when unavailable.

All classes share the same interface: ``fit(X, y)`` and ``predict(X)``.

Public API
----------
- ``QUERY_FEATURE_COLS`` — ordered list of the 13 cheap query features.
- ``load_routing_csv(path)``  → ``(rows, oracle_available)``
- ``prepare_features(rows)``  → ``(X_list, feature_names)``
- ``fit_and_evaluate(X, y, task, feature_names)``  → ``EvalResult``
- ``save_router_outputs(binary_result, multi_result, output_dir)``
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: The 13 cheap query-only feature columns (matches precompute_features.py).
QUERY_FEATURE_COLS: list[str] = [
    "question_length_chars",
    "question_length_tokens_approx",
    "num_numeric_mentions",
    "num_sentences_approx",
    "has_multi_step_cue",
    "has_equation_like_pattern",
    "has_percent_symbol",
    "has_fraction_pattern",
    "has_currency_symbol",
    "max_numeric_value_approx",
    "min_numeric_value_approx",
    "numeric_range_approx",
    "repeated_number_flag",
]

DEFAULT_ROUTING_CSV = Path("outputs/routing_dataset/routing_dataset.csv")
DEFAULT_OUTPUT_DIR = Path("outputs/router_baseline")

# ---------------------------------------------------------------------------
# sklearn availability probe (lazy import)
# ---------------------------------------------------------------------------

_SKLEARN_AVAILABLE: bool | None = None  # None = not yet checked


def _check_sklearn() -> bool:
    global _SKLEARN_AVAILABLE  # noqa: PLW0603
    if _SKLEARN_AVAILABLE is None:
        try:
            import sklearn  # noqa: F401

            _SKLEARN_AVAILABLE = True
        except ImportError:
            _SKLEARN_AVAILABLE = False
    return _SKLEARN_AVAILABLE


# ---------------------------------------------------------------------------
# Majority-class baseline (zero dependencies)
# ---------------------------------------------------------------------------


class MajorityBaseline:
    """Always predicts the most common class seen during training."""

    def __init__(self) -> None:
        self._majority: Any = None

    def fit(self, _X: list[list[float]], y: list[Any]) -> "MajorityBaseline":
        if not y:
            raise ValueError("Cannot fit on empty label list.")
        counts = Counter(y)
        self._majority = counts.most_common(1)[0][0]
        return self

    def predict(self, X: list[list[float]]) -> list[Any]:
        if self._majority is None:
            raise RuntimeError("Call fit() before predict().")
        return [self._majority] * len(X)

    @property
    def name(self) -> str:
        return "majority_baseline"


# ---------------------------------------------------------------------------
# Minimal decision tree (no sklearn dependency)
# ---------------------------------------------------------------------------


@dataclass
class _DTNode:
    """A single node in a binary decision tree."""

    feature_idx: int = -1
    threshold: float = 0.0
    left: "_DTNode | None" = None
    right: "_DTNode | None" = None
    label: Any = None  # leaf label (None for internal nodes)


def _gini(labels: list[Any]) -> float:
    n = len(labels)
    if n == 0:
        return 0.0
    counts = Counter(labels)
    return 1.0 - sum((c / n) ** 2 for c in counts.values())


def _majority_label(labels: list[Any]) -> Any:
    return Counter(labels).most_common(1)[0][0]


def _best_split(
    X: list[list[float]], y: list[Any]
) -> tuple[int, float, float]:
    """Return (feature_idx, threshold, gain) for the best Gini split."""
    n = len(y)
    base_gini = _gini(y)
    best_gain = -1.0
    best_feat = 0
    best_thr = 0.0

    n_features = len(X[0]) if X else 0
    for fi in range(n_features):
        values = sorted(set(row[fi] for row in X))
        for i in range(len(values) - 1):
            thr = (values[i] + values[i + 1]) / 2.0
            left_y = [y[j] for j in range(n) if X[j][fi] <= thr]
            right_y = [y[j] for j in range(n) if X[j][fi] > thr]
            if not left_y or not right_y:
                continue
            gain = base_gini - (
                len(left_y) / n * _gini(left_y) + len(right_y) / n * _gini(right_y)
            )
            if gain > best_gain:
                best_gain = gain
                best_feat = fi
                best_thr = thr

    return best_feat, best_thr, best_gain


def _build_tree(
    X: list[list[float]], y: list[Any], depth: int, max_depth: int
) -> _DTNode:
    node = _DTNode()
    if not y or depth >= max_depth or len(set(y)) == 1:
        node.label = _majority_label(y) if y else None
        return node

    fi, thr, gain = _best_split(X, y)
    if gain <= 0:
        node.label = _majority_label(y)
        return node

    n = len(y)
    left_mask = [X[j][fi] <= thr for j in range(n)]
    right_mask = [not m for m in left_mask]
    X_left = [X[j] for j in range(n) if left_mask[j]]
    y_left = [y[j] for j in range(n) if left_mask[j]]
    X_right = [X[j] for j in range(n) if right_mask[j]]
    y_right = [y[j] for j in range(n) if right_mask[j]]

    if not X_left or not X_right:
        node.label = _majority_label(y)
        return node

    node.feature_idx = fi
    node.threshold = thr
    node.left = _build_tree(X_left, y_left, depth + 1, max_depth)
    node.right = _build_tree(X_right, y_right, depth + 1, max_depth)
    return node


def _predict_one(node: _DTNode, x: list[float]) -> Any:
    if node.label is not None:
        return node.label
    if x[node.feature_idx] <= node.threshold:
        return _predict_one(node.left, x)  # type: ignore[arg-type]
    return _predict_one(node.right, x)  # type: ignore[arg-type]


class MinimalDecisionTree:
    """Pure-Python CART decision tree — no sklearn required."""

    def __init__(self, max_depth: int = 3) -> None:
        self.max_depth = max_depth
        self._root: _DTNode | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X: list[list[float]],
        y: list[Any],
        feature_names: list[str] | None = None,
    ) -> "MinimalDecisionTree":
        if not X or not y:
            raise ValueError("Cannot fit on empty data.")
        self._root = _build_tree(X, y, 0, self.max_depth)
        self._feature_names = feature_names or [f"f{i}" for i in range(len(X[0]))]
        return self

    def predict(self, X: list[list[float]]) -> list[Any]:
        if self._root is None:
            raise RuntimeError("Call fit() before predict().")
        return [_predict_one(self._root, x) for x in X]

    def feature_importances(self) -> dict[str, float]:
        """Return a rough impurity-decrease importance per feature."""
        counts: dict[int, float] = {}
        _collect_importances(self._root, counts)
        total = sum(counts.values()) or 1.0
        return {
            self._feature_names[i]: round(v / total, 4)
            for i, v in sorted(counts.items())
        }

    @property
    def name(self) -> str:
        return f"decision_tree_depth{self.max_depth}"


def _collect_importances(node: _DTNode | None, acc: dict[int, float]) -> None:
    if node is None or node.label is not None:
        return
    acc[node.feature_idx] = acc.get(node.feature_idx, 0.0) + 1.0
    _collect_importances(node.left, acc)
    _collect_importances(node.right, acc)


# ---------------------------------------------------------------------------
# sklearn wrappers (used when sklearn is available)
# ---------------------------------------------------------------------------


class SklearnDecisionTree:
    """Thin wrapper around sklearn DecisionTreeClassifier."""

    def __init__(self, max_depth: int = 3) -> None:
        self.max_depth = max_depth
        self._clf: Any = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X: list[list[float]],
        y: list[Any],
        feature_names: list[str] | None = None,
    ) -> "SklearnDecisionTree":
        from sklearn.tree import DecisionTreeClassifier  # type: ignore[import]

        self._clf = DecisionTreeClassifier(max_depth=self.max_depth, random_state=42)
        self._clf.fit(X, y)
        self._feature_names = feature_names or [f"f{i}" for i in range(len(X[0]))]
        return self

    def predict(self, X: list[list[float]]) -> list[Any]:
        return list(self._clf.predict(X))

    def feature_importances(self) -> dict[str, float]:
        return {
            name: round(float(imp), 4)
            for name, imp in zip(self._feature_names, self._clf.feature_importances_)
        }

    @property
    def name(self) -> str:
        return f"sklearn_decision_tree_depth{self.max_depth}"


class SklearnLogisticRegression:
    """Thin wrapper around sklearn LogisticRegression for binary tasks."""

    def __init__(self) -> None:
        self._clf: Any = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X: list[list[float]],
        y: list[Any],
        feature_names: list[str] | None = None,
    ) -> "SklearnLogisticRegression":
        from sklearn.linear_model import LogisticRegression  # type: ignore[import]
        from sklearn.preprocessing import StandardScaler  # type: ignore[import]

        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)
        self._clf = LogisticRegression(
            max_iter=200, random_state=42, solver="lbfgs"
        )
        self._clf.fit(X_scaled, y)
        self._feature_names = feature_names or [f"f{i}" for i in range(len(X[0]))]
        return self

    def predict(self, X: list[list[float]]) -> list[Any]:
        X_scaled = self._scaler.transform(X)
        return list(self._clf.predict(X_scaled))

    @property
    def name(self) -> str:
        return "sklearn_logistic_regression"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_routing_csv(path: str | Path) -> tuple[list[dict[str, Any]], bool]:
    """Load the routing dataset CSV.

    Parameters
    ----------
    path:
        Path to ``routing_dataset.csv``.

    Returns
    -------
    (rows, oracle_available)
        ``rows`` is a list of dicts; ``oracle_available`` is True when at
        least one row has ``oracle_label_available`` == ``True`` / ``"True"``.

    Raises
    ------
    FileNotFoundError
        When the CSV does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Routing dataset not found: {p}")

    rows: list[dict[str, Any]] = []
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)

    oracle_available = any(
        str(r.get("oracle_label_available", "")).lower() in ("true", "1")
        for r in rows
    )
    return rows, oracle_available


# ---------------------------------------------------------------------------
# Feature preparation
# ---------------------------------------------------------------------------


def prepare_features(
    rows: list[dict[str, Any]],
    feature_cols: list[str] | None = None,
) -> tuple[list[list[float]], list[str]]:
    """Convert routing dataset rows to a numeric feature matrix.

    Parameters
    ----------
    rows:
        List of routing dataset row dicts.
    feature_cols:
        Which columns to use.  Defaults to ``QUERY_FEATURE_COLS``.

    Returns
    -------
    (X, feature_names)
        ``X`` is a list of float lists; ``feature_names`` is the column list.
    """
    if feature_cols is None:
        feature_cols = QUERY_FEATURE_COLS

    X: list[list[float]] = []
    for row in rows:
        vec: list[float] = []
        for col in feature_cols:
            raw = row.get(col, 0)
            try:
                # bool-like strings
                if str(raw).lower() == "true":
                    vec.append(1.0)
                elif str(raw).lower() == "false":
                    vec.append(0.0)
                else:
                    vec.append(float(raw))
            except (ValueError, TypeError):
                vec.append(0.0)
        X.append(vec)

    return X, list(feature_cols)


# ---------------------------------------------------------------------------
# Train / test split (simple holdout)
# ---------------------------------------------------------------------------


def _split(
    X: list[list[float]],
    y: list[Any],
    test_fraction: float = 0.2,
) -> tuple[list[list[float]], list[Any], list[list[float]], list[Any]]:
    """Deterministic 80/20 split (first 80% train, last 20% test)."""
    n = len(X)
    split_idx = max(1, n - math.floor(n * test_fraction))
    # If dataset is very small, use leave-one-out style: train on all, test on all
    if n <= 5:
        return X, y, X, y
    return X[:split_idx], y[:split_idx], X[split_idx:], y[split_idx:]


# ---------------------------------------------------------------------------
# Evaluation result container
# ---------------------------------------------------------------------------


@dataclass
class EvalResult:
    """Results from fitting and evaluating one router model on one task."""

    task: str  # "binary" or "multiclass"
    model_name: str
    n_train: int
    n_test: int
    accuracy: float
    class_distribution: dict[str, int] = field(default_factory=dict)
    feature_importances: dict[str, float] = field(default_factory=dict)
    predictions: list[dict[str, Any]] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task": self.task,
            "model_name": self.model_name,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "accuracy": self.accuracy,
            "class_distribution": self.class_distribution,
            "feature_importances": self.feature_importances,
            "note": self.note,
        }


# ---------------------------------------------------------------------------
# Core fit-and-evaluate
# ---------------------------------------------------------------------------


def fit_and_evaluate(
    X: list[list[float]],
    y: list[Any],
    task: str,
    feature_names: list[str],
    question_ids: list[str] | None = None,
) -> list[EvalResult]:
    """Fit all applicable models and return evaluation results.

    Parameters
    ----------
    X:
        Feature matrix.
    y:
        Target labels.
    task:
        ``"binary"`` or ``"multiclass"``.
    feature_names:
        Column names matching X columns (for importance reporting).
    question_ids:
        Optional question IDs for the prediction CSV.

    Returns
    -------
    list[EvalResult]
        One result per fitted model.
    """
    if not X or not y:
        return []

    X_tr, y_tr, X_te, y_te = _split(X, y)
    ids_te: list[str] = []
    if question_ids:
        n = len(X)
        split_idx = max(1, n - math.floor(n * 0.2))
        if n <= 5:
            ids_te = list(question_ids)
        else:
            ids_te = list(question_ids[split_idx:])

    results: list[EvalResult] = []

    # Build list of (model_instance, feature_names_arg)
    models_to_try: list[Any] = [MajorityBaseline()]

    if _check_sklearn():
        models_to_try.append(SklearnDecisionTree(max_depth=3))
        if task == "binary":
            models_to_try.append(SklearnLogisticRegression())
    else:
        models_to_try.append(MinimalDecisionTree(max_depth=3))

    class_dist = {str(k): v for k, v in Counter(y).items()}

    for model in models_to_try:
        try:
            if isinstance(model, MajorityBaseline):
                model.fit(X_tr, y_tr)
                preds = model.predict(X_te)
                importances: dict[str, float] = {}
            else:
                model.fit(X_tr, y_tr, feature_names=feature_names)
                preds = model.predict(X_te)
                importances = (
                    model.feature_importances()
                    if hasattr(model, "feature_importances")
                    else {}
                )

            correct = sum(int(str(p) == str(t)) for p, t in zip(preds, y_te))
            acc = correct / len(y_te) if y_te else 0.0

            pred_rows = [
                {
                    "question_id": ids_te[i] if i < len(ids_te) else f"idx_{i}",
                    "true_label": str(y_te[i]),
                    "predicted_label": str(preds[i]),
                    "correct": int(str(preds[i]) == str(y_te[i])),
                }
                for i in range(len(y_te))
            ]

            results.append(
                EvalResult(
                    task=task,
                    model_name=model.name,
                    n_train=len(X_tr),
                    n_test=len(X_te),
                    accuracy=round(acc, 4),
                    class_distribution=class_dist,
                    feature_importances=importances,
                    predictions=pred_rows,
                )
            )
        except Exception as exc:  # noqa: BLE001
            results.append(
                EvalResult(
                    task=task,
                    model_name=getattr(model, "name", repr(model)),
                    n_train=len(X_tr),
                    n_test=len(X_te),
                    accuracy=0.0,
                    note=f"Failed: {exc}",
                )
            )

    return results


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def save_router_outputs(
    binary_results: list[EvalResult],
    multi_results: list[EvalResult],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write router baseline outputs to *output_dir*.

    Files written:
      - ``summary.json``
      - ``binary_predictions.csv``
      - ``multiclass_predictions.csv``

    Returns
    -------
    dict[str, str]
        Mapping of artefact names to absolute paths.
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    # summary.json
    summary: dict[str, Any] = {
        "binary_task": [r.to_dict() for r in binary_results],
        "multiclass_task": [r.to_dict() for r in multi_results],
        "sklearn_available": _check_sklearn(),
    }
    summary_path = base / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    paths["summary"] = str(summary_path.resolve())

    # binary_predictions.csv
    bin_csv = base / "binary_predictions.csv"
    _write_predictions_csv(
        bin_csv, binary_results, ["question_id", "true_label", "predicted_label", "correct"]
    )
    paths["binary_predictions"] = str(bin_csv.resolve())

    # multiclass_predictions.csv
    multi_csv = base / "multiclass_predictions.csv"
    _write_predictions_csv(
        multi_csv, multi_results, ["question_id", "true_label", "predicted_label", "correct"]
    )
    paths["multiclass_predictions"] = str(multi_csv.resolve())

    return paths


def _write_predictions_csv(
    path: Path,
    results: list[EvalResult],
    fieldnames: list[str],
) -> None:
    # Use predictions from the best model (last in list, most sophisticated)
    preds: list[dict[str, Any]] = []
    if results:
        # prefer the last result (most sophisticated model) that has predictions
        for r in reversed(results):
            if r.predictions:
                preds = r.predictions
                break

    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(preds)
