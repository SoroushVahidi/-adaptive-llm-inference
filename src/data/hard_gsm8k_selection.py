"""Select a harder-than-average GSM8K test slice using question-side proxies."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path
from typing import Any

from src.datasets.gsm8k import Query, load_gsm8k
from src.features.precompute_features import extract_query_features
from src.features.target_quantity_features import extract_target_quantity_features


def _bool_sum(d: dict[str, Any]) -> int:
    return sum(1 for v in d.values() if v is True)


def _z(x: float, mu: float, sigma: float) -> float:
    if sigma <= 1e-9:
        return 0.0
    return (x - mu) / sigma


def select_hard_gsm8k_queries(
    *,
    pool_size: int | None = None,
    subset_size: int = 100,
    gsm8k_data_file: str | Path | None = None,
    cache_dir: str = "data",
    seed: int = 42,
) -> tuple[list[Query], list[dict[str, Any]], dict[str, Any]]:
    """Score GSM8K test items; return top ``subset_size`` by composite hardness.

    Hardness (higher = harder) blends z-scored length and numeric density with
    cheap structural cues from ``extract_query_features`` and a count of True
    flags in ``extract_target_quantity_features``.
    """
    if gsm8k_data_file and Path(gsm8k_data_file).exists():
        queries = load_gsm8k(split="test", max_samples=pool_size, data_file=gsm8k_data_file)
        source = "local_json_file"
    else:
        queries = load_gsm8k(split="test", max_samples=pool_size, cache_dir=cache_dir)
        source = "huggingface_openai_gsm8k_test"

    rows_raw: list[dict[str, Any]] = []
    for q in queries:
        qf = extract_query_features(q.question)
        tq = extract_target_quantity_features(q.question)
        rows_raw.append({"query": q, "qf": qf, "tq": tq})

    if not rows_raw:
        return [], [], {"error": "empty_pool", "data_source": source}

    chars = [r["qf"]["question_length_chars"] for r in rows_raw]
    nums = [r["qf"]["num_numeric_mentions"] for r in rows_raw]
    mu_c, sig_c = statistics.mean(chars), statistics.pstdev(chars)
    mu_n, sig_n = statistics.mean(nums), statistics.pstdev(nums)

    scored: list[dict[str, Any]] = []
    for r in rows_raw:
        qf, tq = r["qf"], r["tq"]
        z_len = _z(float(qf["question_length_chars"]), mu_c, sig_c)
        z_num = _z(float(qf["num_numeric_mentions"]), mu_n, sig_n)
        z_multi = 1.0 if qf["has_multi_step_cue"] else 0.0
        z_eq = 1.0 if qf["has_equation_like_pattern"] else 0.0
        z_pct = 1.0 if qf["has_percent_symbol"] else 0.0
        z_frac = 1.0 if qf["has_fraction_pattern"] else 0.0
        z_tq = float(_bool_sum(tq))
        hardness = (
            z_len
            + z_num
            + 0.5 * z_multi
            + 0.5 * z_eq
            + 0.3 * z_pct
            + 0.3 * z_frac
            + 0.15 * z_tq
        )
        scored.append(
            {
                "question_id": r["query"].id,
                "question": r["query"].question,
                "gold_answer": r["query"].answer,
                "hardness_score": hardness,
                "z_length": z_len,
                "z_numeric_mentions": z_num,
                "has_multi_step_cue": int(bool(qf["has_multi_step_cue"])),
                "has_equation_like_pattern": int(bool(qf["has_equation_like_pattern"])),
                "target_quantity_true_count": int(_bool_sum(tq)),
                "question_length_chars": qf["question_length_chars"],
                "num_numeric_mentions": qf["num_numeric_mentions"],
            }
        )

    scored.sort(key=lambda x: x["hardness_score"], reverse=True)
    k = min(subset_size, len(scored))
    top = scored[:k]
    id_order = [s["question_id"] for s in top]
    q_by_id = {q.id: q for q in queries}
    selected_ordered = [q_by_id[i] for i in id_order if i in q_by_id]

    summary = {
        "selection_method": "z_score_blend_question_side",
        "formula": (
            "hardness = z_len + z_num + 0.5*multi_step + 0.5*equation + "
            "0.3*percent + 0.3*fraction + 0.15*target_quantity_true_count"
        ),
        "pool_size": len(queries),
        "subset_size": k,
        "data_source": source,
        "seed": seed,
        "population_mean_chars": mu_c,
        "population_std_chars": sig_c,
        "population_mean_numeric_mentions": mu_n,
        "population_std_numeric_mentions": sig_n,
    }
    return selected_ordered, top, summary


def write_hard_selection_artifacts(
    selection_rows: list[dict[str, Any]],
    summary: dict[str, Any],
    out_dir: str | Path,
) -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "hard_gsm8k_selection.csv"
    json_path = out / "selection_summary.json"

    fieldnames = [
        "question_id",
        "question",
        "gold_answer",
        "hardness_score",
        "z_length",
        "z_numeric_mentions",
        "has_multi_step_cue",
        "has_equation_like_pattern",
        "target_quantity_true_count",
        "question_length_chars",
        "num_numeric_mentions",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in selection_rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})

    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return {"csv": csv_path, "json": json_path}
