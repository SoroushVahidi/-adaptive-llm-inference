"""Compatibility wrapper for the original GSM8K model-sampling diagnostic."""

from src.evaluation.strategy_diagnostic import (
    MAX_QUERY_LIMIT as _MAX_QUERY_LIMIT,
)
from src.evaluation.strategy_diagnostic import (
    PROMPT_DIRECT as _PROMPT_DIRECT,
)
from src.evaluation.strategy_diagnostic import (
    PROMPT_REASONING as _PROMPT_REASONING,
)
from src.evaluation.strategy_diagnostic import (
    STRATEGY_SPECS as _STRATEGY_SPECS,
)
from src.evaluation.strategy_diagnostic import (
    build_comparison_summary as _build_comparison_summary,
)
from src.evaluation.strategy_diagnostic import (
    classify_access_error as _classify_access_error,
)
from src.evaluation.strategy_diagnostic import (
    format_strategy_diagnostic_summary,
    run_strategy_diagnostic,
    write_strategy_diagnostic_outputs,
)
from src.evaluation.strategy_diagnostic import (
    summarize_rows as _summarize_rows,
)

MAX_QUERY_LIMIT = _MAX_QUERY_LIMIT
PROMPT_DIRECT = _PROMPT_DIRECT
PROMPT_REASONING = _PROMPT_REASONING
STRATEGY_SPECS = _STRATEGY_SPECS
build_comparison_summary = _build_comparison_summary
classify_access_error = _classify_access_error
summarize_rows = _summarize_rows


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
