"""Routing dataset assembler.

Combines per-query text, lightweight precomputation features z(x), and
optional oracle-subset evaluation labels into a single flat CSV ready for
offline ML or rule-based strategy-routing experiments.

Two operating modes:

``full`` (default)
    Oracle label columns are populated from
    ``outputs/oracle_subset_eval/oracle_assignments.csv`` (and optionally
    ``per_query_matrix.csv`` for ``num_strategies_correct``).  Requires the
    oracle eval to have been run beforehand.

``schema_only``
    Oracle labels are left blank; every row gets ``oracle_label_available=False``.
    Useful when oracle outputs have not yet been produced, or as a dry run to
    inspect the feature schema.

Public API
----------
- ``load_oracle_files(oracle_dir)``   → ``OracleData``
- ``assemble_routing_dataset(queries, oracle_data=None, first_pass_rows=None)``
- ``save_routing_dataset(rows, output_dir)``
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from src.features.precompute_features import extract_query_features

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Default paths relative to repository root.
DEFAULT_ORACLE_DIR = Path("outputs/oracle_subset_eval")
DEFAULT_OUTPUT_DIR = Path("outputs/routing_dataset")

# Columns that constitute oracle labels in the output schema.
ORACLE_LABEL_COLUMNS: list[str] = [
    "best_accuracy_strategy",
    "cheapest_correct_strategy",
    "direct_already_optimal",
    "oracle_any_correct",
    "num_strategies_correct",
]

# Columns derived from first-pass features (optional).
FIRST_PASS_COLUMNS: list[str] = [
    "first_pass_parse_success",
    "first_pass_output_length",
    "first_pass_has_final_answer_cue",
    "first_pass_has_uncertainty_phrase",
    "first_pass_num_numeric_mentions",
    "first_pass_empty_or_malformed_flag",
]


# ---------------------------------------------------------------------------
# Oracle data container
# ---------------------------------------------------------------------------


@dataclass
class OracleData:
    """Holds per-query oracle label data loaded from CSV files.

    Attributes
    ----------
    assignments : dict[str, dict[str, Any]]
        Mapping from ``question_id`` to a dict with oracle assignment fields
        loaded from ``oracle_assignments.csv``.
    strategy_correct_counts : dict[str, int]
        Mapping from ``question_id`` to the number of strategies that were
        correct on that query, derived from ``per_query_matrix.csv``.
    source_files : list[str]
        Paths of the CSV files that were successfully loaded.
    missing_files : list[str]
        Paths that were expected but absent.
    """

    assignments: dict[str, dict[str, Any]] = field(default_factory=dict)
    strategy_correct_counts: dict[str, int] = field(default_factory=dict)
    source_files: list[str] = field(default_factory=list)
    missing_files: list[str] = field(default_factory=list)

    @property
    def available(self) -> bool:
        """True when at least one oracle file was loaded."""
        return len(self.assignments) > 0


# ---------------------------------------------------------------------------
# Oracle file loader
# ---------------------------------------------------------------------------


def load_oracle_files(oracle_dir: str | Path) -> OracleData:
    """Load oracle assignment CSVs from *oracle_dir*.

    Missing files are recorded in ``OracleData.missing_files`` rather than
    raising an exception, so callers can always check ``.available``.

    Parameters
    ----------
    oracle_dir:
        Directory that may contain ``oracle_assignments.csv`` and
        ``per_query_matrix.csv``.

    Returns
    -------
    OracleData
        Populated with whatever files were found.
    """
    base = Path(oracle_dir)
    data = OracleData()

    # --- oracle_assignments.csv ---
    assignments_path = base / "oracle_assignments.csv"
    if assignments_path.exists():
        with assignments_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                qid = row.get("question_id", "").strip()
                if not qid:
                    continue
                data.assignments[qid] = {
                    "any_correct": row.get("any_correct", ""),
                    "cheapest_correct_strategy": row.get("cheapest_correct_strategy", ""),
                    "direct_already_optimal": row.get("direct_already_optimal", ""),
                    "best_accuracy_strategies": row.get("best_accuracy_strategies", ""),
                }
        data.source_files.append(str(assignments_path))
    else:
        data.missing_files.append(str(assignments_path))

    # --- per_query_matrix.csv ---
    matrix_path = base / "per_query_matrix.csv"
    if matrix_path.exists():
        counts: dict[str, int] = {}
        with matrix_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                qid = row.get("question_id", "").strip()
                if not qid:
                    continue
                try:
                    correct = int(row.get("correct", 0))
                except (ValueError, TypeError):
                    correct = 0
                counts[qid] = counts.get(qid, 0) + correct
        data.strategy_correct_counts = counts
        data.source_files.append(str(matrix_path))
    else:
        data.missing_files.append(str(matrix_path))

    return data


# ---------------------------------------------------------------------------
# Row assembler
# ---------------------------------------------------------------------------


@dataclass
class _QueryInput:
    """Minimal query representation accepted by the assembler."""

    question_id: str
    question_text: str


def assemble_routing_dataset(
    queries: list[Any],
    oracle_data: OracleData | None = None,
    first_pass_rows: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Assemble one flat routing-dataset row per query.

    Parameters
    ----------
    queries:
        Any sequence of objects with ``.id`` / ``.question`` attributes
        (compatible with the ``Query`` dataclass in ``src.datasets.gsm8k``),
        **or** plain dicts with ``"question_id"`` / ``"question_text"`` keys,
        **or** ``_QueryInput`` instances.
    oracle_data:
        Pre-loaded oracle data.  If ``None`` or not available, the row is
        assembled in ``schema_only`` mode.
    first_pass_rows:
        Optional mapping from ``question_id`` to a first-pass feature dict
        (as returned by ``extract_first_pass_features``).

    Returns
    -------
    list[dict[str, Any]]
        One dict per query.  All feature columns are present; oracle columns
        are present with empty string values when unavailable.
    """
    if first_pass_rows is None:
        first_pass_rows = {}

    use_oracle = oracle_data is not None and oracle_data.available

    rows: list[dict[str, Any]] = []
    for q in queries:
        # --- Normalise query object ---
        if isinstance(q, dict):
            qid = q.get("question_id") or q.get("id", "")
            qtext = q.get("question_text") or q.get("question", "")
        else:
            # Works with gsm8k.Query and _QueryInput alike
            qid = getattr(q, "id", getattr(q, "question_id", ""))
            qtext = getattr(q, "question", getattr(q, "question_text", ""))

        # --- Query features ---
        qfeats = extract_query_features(qtext)

        row: dict[str, Any] = {
            "question_id": qid,
            "question_text": qtext,
            **qfeats,
        }

        # --- First-pass features (optional) ---
        if qid in first_pass_rows:
            row.update(first_pass_rows[qid])
        # ensure columns are present even when absent
        for col in FIRST_PASS_COLUMNS:
            if col not in row:
                row[col] = ""

        # --- Oracle labels ---
        oracle_label_available = False
        if use_oracle and qid in oracle_data.assignments:
            oracle_label_available = True
            asgn = oracle_data.assignments[qid]

            # best_accuracy_strategy: take first token of the pipe-delimited list
            best_str = asgn.get("best_accuracy_strategies", "")
            row["best_accuracy_strategy"] = (
                best_str.split("|")[0].strip() if best_str else ""
            )
            row["cheapest_correct_strategy"] = asgn.get("cheapest_correct_strategy", "")

            # direct_already_optimal: normalise to int (0/1)
            try:
                row["direct_already_optimal"] = int(asgn.get("direct_already_optimal", 0))
            except (ValueError, TypeError):
                row["direct_already_optimal"] = 0

            # oracle_any_correct
            any_c = asgn.get("any_correct", "")
            try:
                row["oracle_any_correct"] = int(any_c)
            except (ValueError, TypeError):
                row["oracle_any_correct"] = any_c

            row["num_strategies_correct"] = oracle_data.strategy_correct_counts.get(qid, "")
        else:
            for col in ORACLE_LABEL_COLUMNS:
                row[col] = ""

        row["oracle_label_available"] = oracle_label_available

        rows.append(row)

    return rows


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def save_routing_dataset(
    rows: list[dict[str, Any]],
    output_dir: str | Path,
    oracle_data: OracleData | None = None,
) -> dict[str, str]:
    """Write the routing dataset to CSV and a summary JSON.

    Parameters
    ----------
    rows:
        Assembled rows from ``assemble_routing_dataset``.
    output_dir:
        Directory in which to write outputs (created if absent).
    oracle_data:
        The ``OracleData`` used during assembly; recorded in the summary.

    Returns
    -------
    dict[str, str]
        Mapping of ``"csv_path"`` and ``"summary_path"`` to absolute paths.
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    csv_path = base / "routing_dataset.csv"
    summary_path = base / "routing_dataset_summary.json"

    if not rows:
        csv_path.write_text("")
        _write_summary(summary_path, rows=[], oracle_data=oracle_data)
        return {
            "csv_path": str(csv_path.resolve()),
            "summary_path": str(summary_path.resolve()),
        }

    # Determine column order: id/text → query features → fp features → oracle
    sample = rows[0]
    all_cols = list(sample.keys())

    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=all_cols, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in all_cols})

    _write_summary(summary_path, rows=rows, oracle_data=oracle_data)

    return {
        "csv_path": str(csv_path.resolve()),
        "summary_path": str(summary_path.resolve()),
    }


def _write_summary(
    path: Path,
    rows: list[dict[str, Any]],
    oracle_data: OracleData | None,
) -> None:
    """Write the summary JSON to *path*."""
    oracle_available = oracle_data is not None and oracle_data.available
    n_queries = len(rows)

    # Count feature columns vs label columns
    if rows:
        all_cols = list(rows[0].keys())
        feature_cols = [
            c for c in all_cols
            if c not in ("question_id", "question_text", "oracle_label_available")
            and c not in ORACLE_LABEL_COLUMNS
        ]
        label_cols = [c for c in all_cols if c in ORACLE_LABEL_COLUMNS]
    else:
        feature_cols = []
        label_cols = []

    summary: dict[str, Any] = {
        "num_queries": n_queries,
        "oracle_labels_available": oracle_available,
        "num_feature_columns": len(feature_cols),
        "num_label_columns": len(label_cols),
        "feature_columns": feature_cols,
        "label_columns": label_cols,
        "source_files": (oracle_data.source_files if oracle_data else []),
        "missing_optional_inputs": (oracle_data.missing_files if oracle_data else []),
    }

    path.write_text(json.dumps(summary, indent=2))
