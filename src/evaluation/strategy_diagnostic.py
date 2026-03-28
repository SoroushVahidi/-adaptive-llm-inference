"""Generic real-LLM strategy diagnostic for small reasoning benchmarks.

Scientific intent:
1. Check whether a harder benchmark creates more room for adaptive inference
   strategy selection than the current GSM8K diagnostic.
2. Check whether extra reasoning compute helps more once the questions are
   harder and the model is stronger.
3. Check whether simple structured sampling is more useful than naive repeated
   reasoning samples at the same small sample budget.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from src.datasets.gsm8k import Query, load_gsm8k
from src.datasets.math500 import DEFAULT_DATASET_SOURCE, load_math500
from src.models.openai_llm import OpenAILLMModel
from src.utils.answer_extraction import extract_math_answer, extract_numeric_answer

PROMPT_DIRECT = "Answer the following math problem. Give only the final answer."
PROMPT_REASONING = (
    "Solve the following math problem carefully step by step. "
    "At the end, state the final answer clearly."
)
PROMPT_STRUCTURED_CONCISE = (
    "Attempt 1 of 3. Solve the following math problem with a concise direct answer. "
    "Give only the final answer."
)
PROMPT_STRUCTURED_REASONING = (
    "Attempt 2 of 3. Solve the following math problem carefully step by step. "
    "Conclude with: Final answer: <answer>."
)
PROMPT_STRUCTURED_DOUBLE_CHECK = (
    "Attempt 3 of 3. Solve the following math problem, then explicitly double-check "
    "the result before giving the verified final answer."
)

SCIENTIFIC_INTENT = (
    "Diagnostic run for whether harder reasoning data creates more room for "
    "adaptive inference strategy selection, whether the weak compute-allocation "
    "signal on GSM8K is partly a dataset-regime issue, and whether structured "
    "sampling beats naive repeated reasoning at the same small budget."
)
MAX_QUERY_LIMIT = 30


@dataclass(frozen=True)
class StrategySpec:
    """One inference strategy in the diagnostic."""

    name: str
    prompt_labels_and_prefixes: tuple[tuple[str, str], ...]
    samples_per_prompt: int

    @property
    def total_samples(self) -> int:
        return len(self.prompt_labels_and_prefixes) * self.samples_per_prompt


STRATEGY_SPECS: tuple[StrategySpec, ...] = (
    StrategySpec(
        name="direct_greedy",
        prompt_labels_and_prefixes=(("direct", PROMPT_DIRECT),),
        samples_per_prompt=1,
    ),
    StrategySpec(
        name="reasoning_greedy",
        prompt_labels_and_prefixes=(("reasoning", PROMPT_REASONING),),
        samples_per_prompt=1,
    ),
    StrategySpec(
        name="reasoning_best_of_3",
        prompt_labels_and_prefixes=(("reasoning", PROMPT_REASONING),),
        samples_per_prompt=3,
    ),
    StrategySpec(
        name="structured_sampling_3",
        prompt_labels_and_prefixes=(
            ("concise_direct", PROMPT_STRUCTURED_CONCISE),
            ("step_by_step", PROMPT_STRUCTURED_REASONING),
            ("solve_then_double_check", PROMPT_STRUCTURED_DOUBLE_CHECK),
        ),
        samples_per_prompt=1,
    ),
)

STRATEGY_ORDER = {spec.name: index for index, spec in enumerate(STRATEGY_SPECS)}


def _build_model(
    model_name: str,
    config: dict[str, Any],
    prompt_prefix: str = PROMPT_DIRECT,
    max_tokens_override: int | None = None,
) -> OpenAILLMModel:
    openai_cfg = config.get("openai", {})
    max_tokens = (
        int(max_tokens_override)
        if max_tokens_override is not None
        else int(openai_cfg.get("max_tokens", 256))
    )
    return OpenAILLMModel(
        model_name=model_name,
        base_url=openai_cfg.get("base_url"),
        greedy_temperature=float(openai_cfg.get("greedy_temperature", 0.0)),
        sample_temperature=float(openai_cfg.get("sample_temperature", 0.7)),
        max_tokens=max_tokens,
        timeout_seconds=float(openai_cfg.get("timeout_seconds", 60.0)),
        prompt_prefix=prompt_prefix,
    )


def _majority_vote(parsed_answers: list[str]) -> str:
    """Majority vote over normalized parsed answers with deterministic ties."""
    if not parsed_answers:
        return ""
    non_empty = [answer for answer in parsed_answers if answer]
    if not non_empty:
        return ""

    counts = Counter(non_empty)
    ranked = sorted(
        counts.items(),
        key=lambda item: (-item[1], parsed_answers.index(item[0]), item[0]),
    )
    return ranked[0][0]


def classify_access_error(error_message: str) -> str:
    """Classify a model-access/runtime failure into a compact diagnostic label."""
    lowered = error_message.lower()
    if "missing openai_api_key" in lowered or "invalid api key" in lowered:
        return "config"
    if "insufficient_quota" in lowered or "quota" in lowered:
        return "quota"
    if "rate limit" in lowered or "rate_limit" in lowered:
        return "rate_limit"
    if (
        "model_not_found" in lowered
        or "does not exist" in lowered
        or "not have access" in lowered
        or "access to the model" in lowered
        or "permission" in lowered
    ):
        return "model_access"
    if "openai api request failed" in lowered or "timed out" in lowered:
        return "network"
    return "unknown"


def _probe_model_access(model_name: str, config: dict[str, Any]) -> dict[str, Any]:
    """Issue one tiny request to verify that a model is actually callable."""
    try:
        model = _build_model(
            model_name=model_name,
            config=config,
            prompt_prefix="Reply only with the number 2.",
            max_tokens_override=8,
        )
        model.generate("What is 1 + 1?")
    except ValueError as exc:
        return {
            "model": model_name,
            "status": "failed",
            "error_type": classify_access_error(str(exc)),
            "error": str(exc),
        }
    except RuntimeError as exc:
        return {
            "model": model_name,
            "status": "failed",
            "error_type": classify_access_error(str(exc)),
            "error": str(exc),
        }

    return {
        "model": model_name,
        "status": "accessible",
        "error_type": None,
        "error": None,
    }


def _get_answer_extractor(answer_mode: str) -> Callable[[str], str]:
    if answer_mode == "numeric":
        return extract_numeric_answer
    if answer_mode == "math":
        return extract_math_answer
    raise ValueError(f"Unsupported answer_mode '{answer_mode}'")


def _load_queries(config: dict[str, Any]) -> tuple[list[Query], dict[str, Any]]:
    dataset_cfg = config.get("dataset", {})
    dataset_name = str(dataset_cfg.get("name", "gsm8k"))
    split = str(dataset_cfg.get("split", "test"))
    requested_max_samples = int(dataset_cfg.get("max_samples", 15))
    max_samples = min(requested_max_samples, MAX_QUERY_LIMIT)
    data_file = dataset_cfg.get("data_file")

    if dataset_name == "gsm8k":
        queries = load_gsm8k(
            split=split,
            max_samples=max_samples,
            cache_dir=str(dataset_cfg.get("cache_dir", "data")),
            data_file=data_file,
        )
        metadata = {
            "name": "gsm8k",
            "source": dataset_cfg.get("source", "openai/gsm8k"),
            "split": split,
            "requested_max_samples": requested_max_samples,
            "num_queries": len(queries),
            "question_ids": [query.id for query in queries],
            "answer_mode": dataset_cfg.get("answer_mode", "numeric"),
        }
        return queries, metadata

    if dataset_name == "math500":
        dataset_source = str(dataset_cfg.get("source", DEFAULT_DATASET_SOURCE))
        queries = load_math500(
            split=split,
            max_samples=max_samples,
            cache_dir=str(dataset_cfg.get("cache_dir", "data")),
            data_file=data_file,
            dataset_source=dataset_source,
        )
        metadata = {
            "name": "math500",
            "source": dataset_source,
            "split": split,
            "requested_max_samples": requested_max_samples,
            "num_queries": len(queries),
            "question_ids": [query.id for query in queries],
            "answer_mode": dataset_cfg.get("answer_mode", "math"),
        }
        return queries, metadata

    raise ValueError(f"Unsupported dataset '{dataset_name}'")


def _run_strategy_for_query(
    model: OpenAILLMModel,
    query: Query,
    strategy: StrategySpec,
    answer_extractor: Callable[[str], str],
) -> dict[str, Any]:
    raw_candidates: list[str] = []
    parsed_candidates: list[str] = []
    attempt_labels: list[str] = []

    for label, prompt_prefix in strategy.prompt_labels_and_prefixes:
        attempt_labels.extend([label] * strategy.samples_per_prompt)
        strategy_model = model.with_prompt_prefix(prompt_prefix)
        if strategy.samples_per_prompt == 1:
            raw_responses = [strategy_model.generate(query.question)]
        else:
            raw_responses = strategy_model.generate_n(
                query.question,
                strategy.samples_per_prompt,
            )
        raw_candidates.extend(raw_responses)
        parsed_candidates.extend(answer_extractor(response) for response in raw_responses)

    final_answer = _majority_vote(parsed_candidates)
    return {
        "question_id": query.id,
        "question": query.question,
        "predicted_answer": final_answer,
        "gold_answer": query.answer,
        "correct": final_answer == query.answer,
        "strategy": strategy.name,
        "samples_used": strategy.total_samples,
        "aggregation_method": "normalized_majority_vote",
        "attempt_labels": json.dumps(attempt_labels),
        "parsed_candidate_answers": json.dumps(parsed_candidates),
        "raw_candidate_answers": json.dumps(raw_candidates),
    }


def _run_model(
    model_name: str,
    queries: list[Query],
    config: dict[str, Any],
    answer_extractor: Callable[[str], str],
) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
    try:
        base_model = _build_model(
            model_name=model_name,
            config=config,
        )
    except ValueError as exc:
        return [], {
            "model": model_name,
            "status": "failed",
            "error_type": classify_access_error(str(exc)),
            "error": str(exc),
        }

    rows: list[dict[str, Any]] = []
    for strategy in STRATEGY_SPECS:
        for query in queries:
            try:
                row = _run_strategy_for_query(
                    model=base_model,
                    query=query,
                    strategy=strategy,
                    answer_extractor=answer_extractor,
                )
            except RuntimeError as exc:
                return [], {
                    "model": model_name,
                    "status": "failed",
                    "error_type": classify_access_error(str(exc)),
                    "error": str(exc),
                }
            row["model"] = model_name
            rows.append(row)
    return rows, None


def _sort_summary_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        summary_rows,
        key=lambda item: (
            str(item["model"]),
            STRATEGY_ORDER.get(str(item["strategy"]), 999),
        ),
    )


def summarize_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate per-query rows into model x strategy metrics."""
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["model"]), str(row["strategy"]))].append(row)

    summaries: list[dict[str, Any]] = []
    for (model_name, strategy_name), group in sorted(
        grouped.items(),
        key=lambda item: (item[0][0], STRATEGY_ORDER.get(item[0][1], 999)),
    ):
        total_queries = len(group)
        total_correct = sum(1 for row in group if bool(row["correct"]))
        total_samples = sum(int(row["samples_used"]) for row in group)
        summaries.append(
            {
                "model": model_name,
                "strategy": strategy_name,
                "accuracy": 0.0 if total_queries == 0 else total_correct / total_queries,
                "correct": total_correct,
                "total_queries": total_queries,
                "total_samples": total_samples,
                "avg_samples_per_query": (
                    0.0 if total_queries == 0 else total_samples / total_queries
                ),
            }
        )
    return _sort_summary_rows(summaries)


def build_comparison_summary(
    summary_rows: list[dict[str, Any]],
    current_model: str,
    stronger_model: str | None,
) -> dict[str, Any]:
    """Build comparison deltas requested by the diagnostic task."""
    lookup = {
        (str(row["model"]), str(row["strategy"])): row
        for row in summary_rows
    }

    per_model_improvements: list[dict[str, Any]] = []
    tested_models = sorted({str(row["model"]) for row in summary_rows})
    for model_name in tested_models:
        reasoning_greedy = lookup.get((model_name, "reasoning_greedy"))
        reasoning_best_of_3 = lookup.get((model_name, "reasoning_best_of_3"))
        structured_sampling_3 = lookup.get((model_name, "structured_sampling_3"))
        if not reasoning_greedy or not reasoning_best_of_3 or not structured_sampling_3:
            continue

        reasoning_delta = round(
            float(reasoning_best_of_3["accuracy"]) - float(reasoning_greedy["accuracy"]),
            4,
        )
        structured_delta = round(
            float(structured_sampling_3["accuracy"]) - float(reasoning_best_of_3["accuracy"]),
            4,
        )
        per_model_improvements.append(
            {
                "model": model_name,
                "reasoning_best_of_3_minus_reasoning_greedy": reasoning_delta,
                "structured_sampling_3_minus_reasoning_best_of_3": structured_delta,
                "structured_sampling_3_beats_reasoning_best_of_3": structured_delta > 0,
            }
        )

    stronger_model_effect: dict[str, Any] = {
        "current_model": current_model,
        "stronger_model": stronger_model,
        "available": False,
    }
    if stronger_model and stronger_model in tested_models and current_model in tested_models:
        current_reasoning_delta = float(
            lookup[(current_model, "reasoning_best_of_3")]["accuracy"]
        ) - float(lookup[(current_model, "reasoning_greedy")]["accuracy"])
        stronger_reasoning_delta = float(
            lookup[(stronger_model, "reasoning_best_of_3")]["accuracy"]
        ) - float(lookup[(stronger_model, "reasoning_greedy")]["accuracy"])
        current_structured_delta = float(
            lookup[(current_model, "structured_sampling_3")]["accuracy"]
        ) - float(lookup[(current_model, "reasoning_best_of_3")]["accuracy"])
        stronger_structured_delta = float(
            lookup[(stronger_model, "structured_sampling_3")]["accuracy"]
        ) - float(lookup[(stronger_model, "reasoning_best_of_3")]["accuracy"])
        stronger_model_effect = {
            "current_model": current_model,
            "stronger_model": stronger_model,
            "available": True,
            "reasoning_best_of_3_minus_reasoning_greedy": {
                current_model: round(current_reasoning_delta, 4),
                stronger_model: round(stronger_reasoning_delta, 4),
                "difference": round(stronger_reasoning_delta - current_reasoning_delta, 4),
            },
            "structured_sampling_3_minus_reasoning_best_of_3": {
                current_model: round(current_structured_delta, 4),
                stronger_model: round(stronger_structured_delta, 4),
                "difference": round(
                    stronger_structured_delta - current_structured_delta,
                    4,
                ),
            },
        }

    return {
        "per_model_improvements": per_model_improvements,
        "stronger_model_effect": stronger_model_effect,
    }


def run_strategy_diagnostic(config: dict[str, Any]) -> dict[str, Any]:
    """Run the configured small strategy-comparison diagnostic."""
    try:
        queries, dataset_metadata = _load_queries(config)
    except Exception as exc:  # pragma: no cover - exercised by runtime dataset loads
        dataset_cfg = config.get("dataset", {})
        dataset_name = str(dataset_cfg.get("name", "gsm8k"))
        dataset_source = str(
            dataset_cfg.get(
                "source",
                DEFAULT_DATASET_SOURCE if dataset_name == "math500" else "openai/gsm8k",
            )
        )
        raise RuntimeError(
            f"Dataset load failed for '{dataset_name}' from '{dataset_source}': {exc}"
        ) from exc

    answer_extractor = _get_answer_extractor(str(dataset_metadata["answer_mode"]))
    models_cfg = config.get("models", {})
    current_model = str(models_cfg["current"])
    stronger_model = models_cfg.get("stronger")
    stronger_model_name = None if stronger_model is None else str(stronger_model)

    access_checks: list[dict[str, Any]] = []
    current_probe = _probe_model_access(current_model, config)
    access_checks.append(current_probe)
    if current_probe["status"] != "accessible":
        raise RuntimeError(
            f"Current model '{current_model}' is not accessible: "
            f"{current_probe['error']}"
        )

    all_rows: list[dict[str, Any]] = []
    current_rows, current_error = _run_model(
        current_model,
        queries,
        config,
        answer_extractor=answer_extractor,
    )
    if current_error is not None:
        raise RuntimeError(
            f"Current model '{current_model}' failed during execution: "
            f"{current_error['error']}"
        )
    all_rows.extend(current_rows)

    stronger_status: dict[str, Any] | None = None
    if stronger_model_name:
        stronger_probe = _probe_model_access(stronger_model_name, config)
        access_checks.append(stronger_probe)
        if stronger_probe["status"] == "accessible":
            stronger_rows, stronger_error = _run_model(
                stronger_model_name,
                queries,
                config,
                answer_extractor=answer_extractor,
            )
            if stronger_error is None:
                all_rows.extend(stronger_rows)
                stronger_status = {
                    "model": stronger_model_name,
                    "status": "tested",
                    "error_type": None,
                    "error": None,
                }
            else:
                stronger_status = stronger_error
        else:
            stronger_status = stronger_probe

    summary_rows = summarize_rows(all_rows)
    comparisons = build_comparison_summary(
        summary_rows=summary_rows,
        current_model=current_model,
        stronger_model=stronger_model_name,
    )

    return {
        "scientific_intent": SCIENTIFIC_INTENT,
        "dataset": dataset_metadata,
        "models": {
            "current": current_model,
            "stronger": stronger_model_name,
            "tested": sorted({str(row["model"]) for row in all_rows}),
        },
        "access_checks": access_checks,
        "stronger_model_status": stronger_status,
        "summary_rows": summary_rows,
        "comparisons": comparisons,
        "per_query_results": all_rows,
    }


def _write_csv(rows: list[dict[str, Any]], output_path: str | Path) -> str:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output.write_text("")
        return str(output)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return str(output)


def write_strategy_diagnostic_outputs(
    result: dict[str, Any],
    output_dir: str | Path,
) -> dict[str, str]:
    """Write JSON/CSV outputs required by the diagnostic task."""
    base_dir = Path(output_dir)
    base_dir.mkdir(parents=True, exist_ok=True)

    summary_json = base_dir / "summary.json"
    summary_payload = {
        key: value
        for key, value in result.items()
        if key != "per_query_results"
    }
    summary_json.write_text(json.dumps(summary_payload, indent=2))

    return {
        "summary_json": str(summary_json),
        "summary_csv": _write_csv(result["summary_rows"], base_dir / "summary.csv"),
        "per_query_csv": _write_csv(
            result["per_query_results"],
            base_dir / "per_query_results.csv",
        ),
    }


def format_strategy_diagnostic_summary(
    result: dict[str, Any],
    paths: dict[str, str] | None = None,
) -> str:
    """Render a concise terminal summary for the diagnostic run."""
    lines = [
        "--- Strategy Diagnostic ---",
        f"dataset: {result['dataset']['name']}",
        f"queries: {result['dataset']['num_queries']}",
        (
            "models_tested: "
            + (", ".join(result["models"]["tested"]) if result["models"]["tested"] else "none")
        ),
        "",
        "per model x strategy results:",
    ]
    for row in result["summary_rows"]:
        lines.append(
            "  "
            f"{row['model']} | {row['strategy']} | accuracy={row['accuracy']:.4f} | "
            f"total_samples={row['total_samples']} | "
            f"avg_samples/query={row['avg_samples_per_query']:.2f}"
        )

    lines.append("")
    lines.append("improvements:")
    for row in result["comparisons"]["per_model_improvements"]:
        lines.append(
            "  "
            f"{row['model']}: "
            f"reasoning_best_of_3 - reasoning_greedy="
            f"{row['reasoning_best_of_3_minus_reasoning_greedy']:+.4f}, "
            f"structured_sampling_3 - reasoning_best_of_3="
            f"{row['structured_sampling_3_minus_reasoning_best_of_3']:+.4f}"
        )

    stronger_status = result.get("stronger_model_status")
    if stronger_status is not None and stronger_status.get("status") != "tested":
        lines.extend(
            [
                "",
                "stronger model status:",
                "  "
                f"{stronger_status['model']}: status={stronger_status['status']}, "
                f"error_type={stronger_status['error_type']}, "
                f"error={stronger_status['error']}",
            ]
        )

    stronger_effect = result["comparisons"]["stronger_model_effect"]
    if stronger_effect["available"]:
        reasoning_delta = stronger_effect["reasoning_best_of_3_minus_reasoning_greedy"]
        structured_delta = stronger_effect["structured_sampling_3_minus_reasoning_best_of_3"]
        lines.extend(
            [
                "",
                "stronger model effect on extra compute:",
                "  "
                f"reasoning_best_of_3 - reasoning_greedy: "
                f"{reasoning_delta}",
                "  "
                f"structured_sampling_3 - reasoning_best_of_3: "
                f"{structured_delta}",
            ]
        )

    if paths is not None:
        lines.extend(
            [
                "",
                f"summary_json: {paths['summary_json']}",
                f"summary_csv: {paths['summary_csv']}",
                f"per_query_csv: {paths['per_query_csv']}",
            ]
        )
    return "\n".join(lines)
