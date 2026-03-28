"""Utilities for loading and validating the action-space catalog."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_ACTION_CATALOG_PATH = Path("configs/action_space_catalog.yaml")

VALID_STATUSES: frozenset[str] = frozenset({"implemented", "partial", "placeholder"})


class ActionCatalogValidationError(ValueError):
    """Raised when the action catalog contains invalid references or values."""


def load_action_catalog(path: str | Path = DEFAULT_ACTION_CATALOG_PATH) -> dict[str, Any]:
    """Load and validate the action-space catalog YAML."""
    catalog_path = Path(path)
    catalog = yaml.safe_load(catalog_path.read_text())
    validate_action_catalog(catalog)
    return catalog


def list_prompt_types(catalog: dict[str, Any]) -> list[str]:
    """Return declared prompt types."""
    return list(catalog["available_components"]["prompt_types"])


def list_stage_structures(catalog: dict[str, Any]) -> list[str]:
    """Return declared stage structures."""
    return list(catalog["available_components"]["stage_structures"])


def list_model_slots(catalog: dict[str, Any]) -> list[str]:
    """Return declared model slots."""
    return list(catalog["available_components"]["model_slots"])


def list_curated_strategies(catalog: dict[str, Any]) -> list[dict[str, Any]]:
    """Return curated strategy definitions."""
    return list(catalog["curated_strategies"])


def validate_action_catalog(catalog: dict[str, Any]) -> None:
    """Validate catalog structure and strategy references."""
    if not isinstance(catalog, dict):
        raise ActionCatalogValidationError("Catalog must be a mapping.")

    components = catalog.get("available_components")
    strategies = catalog.get("curated_strategies")
    if not isinstance(components, dict):
        raise ActionCatalogValidationError("'available_components' must be a mapping.")
    if not isinstance(strategies, list):
        raise ActionCatalogValidationError("'curated_strategies' must be a list.")

    prompt_types = set(components.get("prompt_types", []))
    sample_counts = set(components.get("sample_counts", []))
    stage_structures = set(components.get("stage_structures", []))
    model_slots = set(components.get("model_slots", []))

    _ensure_positive_int_set(sample_counts)

    seen_names: set[str] = set()
    for strategy in strategies:
        if not isinstance(strategy, dict):
            raise ActionCatalogValidationError("Each strategy entry must be a mapping.")

        name = strategy.get("name")
        if not isinstance(name, str) or not name:
            raise ActionCatalogValidationError("Each strategy must have a non-empty string 'name'.")
        if name in seen_names:
            raise ActionCatalogValidationError(f"Duplicate strategy name: {name}")
        seen_names.add(name)

        strategy_prompts = strategy.get("prompt_types", [])
        if not isinstance(strategy_prompts, list) or not strategy_prompts:
            raise ActionCatalogValidationError(
                f"Strategy '{name}' must declare a non-empty list of prompt_types."
            )
        unknown_prompt_types = [p for p in strategy_prompts if p not in prompt_types]
        if unknown_prompt_types:
            raise ActionCatalogValidationError(
                f"Strategy '{name}' references unknown prompt type(s): {unknown_prompt_types}"
            )

        sample_count = strategy.get("sample_count")
        if sample_count not in sample_counts:
            raise ActionCatalogValidationError(
                f"Strategy '{name}' sample_count '{sample_count}' is not declared."
            )
        if not isinstance(sample_count, int) or sample_count <= 0:
            raise ActionCatalogValidationError(
                f"Strategy '{name}' sample_count must be a positive integer."
            )

        stage_structure = strategy.get("stage_structure")
        if stage_structure not in stage_structures:
            raise ActionCatalogValidationError(
                f"Strategy '{name}' stage_structure '{stage_structure}' is not declared."
            )

        model_slot = strategy.get("model_slot")
        if model_slot not in model_slots:
            raise ActionCatalogValidationError(
                f"Strategy '{name}' model_slot '{model_slot}' is not declared."
            )

        status = strategy.get("status")
        if status not in VALID_STATUSES:
            raise ActionCatalogValidationError(
                f"Strategy '{name}' has invalid status '{status}'. "
                f"Must be one of: {sorted(VALID_STATUSES)}"
            )


def _ensure_positive_int_set(values: set[Any]) -> None:
    if not values:
        raise ActionCatalogValidationError("'sample_counts' must not be empty.")
    for value in values:
        if not isinstance(value, int) or value <= 0:
            raise ActionCatalogValidationError(
                "'sample_counts' must contain only positive integers."
            )
