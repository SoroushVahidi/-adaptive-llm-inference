"""Compatibility wrapper for the original GSM8K model-sampling diagnostic."""

from src.evaluation.strategy_diagnostic import (
    MAX_QUERY_LIMIT,
    PROMPT_DIRECT,
    PROMPT_REASONING,
    STRATEGY_SPECS,
    build_comparison_summary,
    classify_access_error,
    format_strategy_diagnostic_summary,
    run_strategy_diagnostic,
    summarize_rows,
    write_strategy_diagnostic_outputs,
)


def run_model_sampling_diagnostic(config: dict) -> dict:
    """Run the legacy GSM8K diagnostic through the generic strategy path."""
    return run_strategy_diagnostic(config)


def write_model_sampling_diagnostic_outputs(
    result: dict,
    output_dir: str,
) -> dict[str, str]:
    return write_strategy_diagnostic_outputs(result, output_dir)


def format_model_sampling_diagnostic_summary(
    result: dict,
    paths: dict[str, str] | None = None,
) -> str:
    return format_strategy_diagnostic_summary(result, paths)
