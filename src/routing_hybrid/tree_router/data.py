from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import numpy as np


def read_candidate_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def filter_rows(
    rows: list[dict[str, str]],
    regimes: list[str] | None = None,
    splits: list[str] | None = None,
) -> list[dict[str, str]]:
    out = rows
    if regimes:
        keep = set(regimes)
        out = [r for r in out if r.get("regime", "") in keep]
    if splits:
        skeep = set(splits)
        out = [r for r in out if r.get("split", "") in skeep]
    return out


def _to_float(v: str | float | int, default: float = 0.0) -> float:
    try:
        return float(v)
    except Exception:
        return default


def build_targets(
    rows: list[dict[str, str]],
    target_type: str,
    lambda_cost: float,
) -> np.ndarray:
    if target_type == "success_binary":
        return np.array([int(_to_float(r.get("correctness_label", 0))) for r in rows], dtype=int)
    if target_type == "gain_binary":
        return np.array(
            [1 if _to_float(r.get("gain_vs_baseline", 0.0)) > 0.0 else 0 for r in rows],
            dtype=int,
        )
    if target_type == "utility_regression":
        return np.array(
            [
                _to_float(r.get("correctness_label", 0.0)) - lambda_cost * _to_float(r.get("action_cost", 0.0))
                for r in rows
            ],
            dtype=float,
        )
    raise ValueError(f"Unknown target_type '{target_type}'")


def build_feature_matrix(
    rows: list[dict[str, str]],
    include_regime_indicator: bool = True,
    include_action_indicator: bool = True,
    include_feature_prefixes: tuple[str, ...] = ("feat_", "hfeat_"),
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    if not rows:
        return np.zeros((0, 0)), [], {"medians": {}, "regimes": [], "actions": []}
    # numeric base features
    raw_feature_cols = sorted(
        {
            k
            for r in rows
            for k in r.keys()
            if any(k.startswith(pfx) for pfx in include_feature_prefixes)
        }
    )

    mat: list[list[float]] = []
    for r in rows:
        mat.append([_to_float(r.get(c, 0.0)) for c in raw_feature_cols])
    X = np.array(mat, dtype=float)
    medians = np.nanmedian(X, axis=0) if X.size else np.array([])
    if X.size:
        inds = np.where(~np.isfinite(X))
        if len(inds[0]) > 0:
            X[inds] = medians[inds[1]]

    feature_names = list(raw_feature_cols)

    # one-hot regime and action
    regimes = sorted({r.get("regime", "") for r in rows}) if include_regime_indicator else []
    actions = sorted({r.get("action_name", "") for r in rows}) if include_action_indicator else []
    extra_cols: list[np.ndarray] = []
    extra_names: list[str] = []
    if regimes:
        reg_map = {v: i for i, v in enumerate(regimes)}
        reg_hot = np.zeros((len(rows), len(regimes)), dtype=float)
        for i, r in enumerate(rows):
            reg_hot[i, reg_map[r.get("regime", "")]] = 1.0
        extra_cols.append(reg_hot)
        extra_names.extend([f"cat_regime__{r}" for r in regimes])
    if actions:
        act_map = {v: i for i, v in enumerate(actions)}
        act_hot = np.zeros((len(rows), len(actions)), dtype=float)
        for i, r in enumerate(rows):
            act_hot[i, act_map[r.get("action_name", "")]] = 1.0
        extra_cols.append(act_hot)
        extra_names.extend([f"cat_action__{a}" for a in actions])
    if extra_cols:
        X = np.concatenate([X] + extra_cols, axis=1) if X.size else np.concatenate(extra_cols, axis=1)
        feature_names.extend(extra_names)

    metadata = {
        "medians": {name: float(m) for name, m in zip(raw_feature_cols, medians, strict=False)},
        "regimes": regimes,
        "actions": actions,
    }
    return X, feature_names, metadata


def transform_feature_matrix(
    rows: list[dict[str, str]],
    feature_metadata: dict[str, Any],
    include_feature_prefixes: tuple[str, ...] = ("feat_", "hfeat_"),
) -> np.ndarray:
    raw_feature_cols = sorted(
        {
            k
            for r in rows
            for k in r.keys()
            if any(k.startswith(pfx) for pfx in include_feature_prefixes)
        }
    )
    # Keep only columns seen in training medians.
    train_cols = sorted(feature_metadata.get("medians", {}).keys())
    cols = [c for c in train_cols if c in raw_feature_cols]
    mat = np.array([[_to_float(r.get(c, feature_metadata["medians"].get(c, 0.0))) for c in cols] for r in rows], dtype=float)
    if mat.size:
        med = np.array([feature_metadata["medians"].get(c, 0.0) for c in cols], dtype=float)
        inds = np.where(~np.isfinite(mat))
        if len(inds[0]) > 0:
            mat[inds] = med[inds[1]]
    # one-hot categories from training vocab
    regimes = feature_metadata.get("regimes", [])
    actions = feature_metadata.get("actions", [])
    extras: list[np.ndarray] = []
    if regimes:
        reg_map = {v: i for i, v in enumerate(regimes)}
        reg_hot = np.zeros((len(rows), len(regimes)), dtype=float)
        for i, r in enumerate(rows):
            idx = reg_map.get(r.get("regime", ""))
            if idx is not None:
                reg_hot[i, idx] = 1.0
        extras.append(reg_hot)
    if actions:
        act_map = {v: i for i, v in enumerate(actions)}
        act_hot = np.zeros((len(rows), len(actions)), dtype=float)
        for i, r in enumerate(rows):
            idx = act_map.get(r.get("action_name", ""))
            if idx is not None:
                act_hot[i, idx] = 1.0
        extras.append(act_hot)
    if extras:
        if mat.size:
            return np.concatenate([mat] + extras, axis=1)
        return np.concatenate(extras, axis=1)
    return mat

