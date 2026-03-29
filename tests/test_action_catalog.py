import pytest

from src.strategies.action_catalog import (
    VALID_STATUSES,
    ActionCatalogValidationError,
    list_curated_strategies,
    list_model_slots,
    list_prompt_types,
    list_stage_structures,
    load_action_catalog,
    validate_action_catalog,
)

# ─────────────────────────────────────────────────────────────────────────────
# Basic structure tests
# ─────────────────────────────────────────────────────────────────────────────


def test_catalog_yaml_loads_successfully():
    catalog = load_action_catalog()
    assert isinstance(catalog, dict)
    assert "available_components" in catalog
    assert "curated_strategies" in catalog


def test_curated_strategies_reference_valid_components():
    catalog = load_action_catalog()
    components = catalog["available_components"]

    valid_prompts = set(components["prompt_types"])
    valid_stages = set(components["stage_structures"])
    valid_models = set(components["model_slots"])
    valid_sample_counts = set(components["sample_counts"])

    for strategy in catalog["curated_strategies"]:
        for prompt_type in strategy["prompt_types"]:
            assert prompt_type in valid_prompts
        assert strategy["stage_structure"] in valid_stages
        assert strategy["model_slot"] in valid_models
        assert strategy["sample_count"] in valid_sample_counts
        assert isinstance(strategy["sample_count"], int)
        assert strategy["sample_count"] > 0


def test_no_duplicate_strategy_names():
    catalog = load_action_catalog()
    names = [strategy["name"] for strategy in catalog["curated_strategies"]]
    assert len(names) == len(set(names))


# ─────────────────────────────────────────────────────────────────────────────
# Status field tests
# ─────────────────────────────────────────────────────────────────────────────


def test_every_strategy_has_valid_status():
    catalog = load_action_catalog()
    for strategy in catalog["curated_strategies"]:
        name = strategy["name"]
        status = strategy.get("status")
        assert status in VALID_STATUSES, (
            f"Strategy '{name}' has invalid or missing status '{status}'"
        )


def test_implemented_strategies_are_expected_subset():
    """Known implemented strategies must carry status='implemented'."""
    catalog = load_action_catalog()
    status_by_name = {s["name"]: s["status"] for s in catalog["curated_strategies"]}

    known_implemented = {
        "direct_greedy",
        "strong_direct",
        "reasoning_greedy",
        "reasoning_best_of_3",
        "self_consistency",
        "self_consistency_3",
        "structured_sampling_3",
        "direct_plus_verify",
        "direct_plus_revise",
        "reasoning_then_revise",
        "direct_plus_critique_plus_final",
        "first_pass_then_hint_guided_reason",
    }
    for name in known_implemented:
        assert name in status_by_name, f"Expected strategy '{name}' not found in catalog"
        assert status_by_name[name] == "implemented", (
            f"Strategy '{name}' should be 'implemented', got '{status_by_name[name]}'"
        )


def test_invalid_status_raises_validation_error():
    catalog = load_action_catalog()
    bad_catalog = {
        "available_components": catalog["available_components"],
        "curated_strategies": [
            {
                "name": "bad_strategy",
                "prompt_types": ["direct"],
                "sample_count": 1,
                "stage_structure": "one_shot",
                "model_slot": "cheap_model",
                "status": "unknown_status",
                "rationale": "Test",
                "expected_cost_tier": "low",
            }
        ],
    }
    with pytest.raises(ActionCatalogValidationError, match="invalid status"):
        validate_action_catalog(bad_catalog)


# ─────────────────────────────────────────────────────────────────────────────
# Core strategy presence tests (all 12 families)
# ─────────────────────────────────────────────────────────────────────────────


def test_expected_core_strategies_exist():
    catalog = load_action_catalog()
    names = {strategy["name"] for strategy in list_curated_strategies(catalog)}

    # Original core set
    original_core = {
        "direct_greedy",
        "reasoning_greedy",
        "reasoning_best_of_3",
        "structured_sampling_3",
        "direct_plus_verify",
        "direct_plus_revise",
        "direct_plus_critique_plus_final",
        "first_pass_then_hint_guided_reason",
        "strong_direct",
        "strong_structured_placeholder",
    }
    assert original_core.issubset(names)


def test_all_required_strategy_families_present():
    catalog = load_action_catalog()
    names = {s["name"] for s in list_curated_strategies(catalog)}

    # A. Cheap / direct baselines
    assert "direct_greedy" in names
    assert "strong_direct" in names

    # B. Reasoning baselines
    assert "reasoning_greedy" in names
    assert "reasoning_best_of_3" in names
    assert "self_consistency" in names

    # C. Structured prompting / diverse sampling
    assert "structured_sampling_3" in names
    assert "direct_plus_double_check" in names

    # D. Sequential correction / self-improvement
    assert "direct_plus_verify" in names
    assert "direct_plus_revise" in names
    assert "direct_plus_critique_plus_final" in names

    # E. Hint-guided reasoning
    assert "first_pass_then_hint_guided_reason" in names

    # F. Token-budget strategies
    assert "token_budget_low" in names
    assert "token_budget_mid" in names
    assert "token_budget_high" in names

    # G. Early-exit strategies
    assert "reasoning_with_early_exit" in names

    # H. Model-routing strategies
    assert "cheap_model_route" in names
    assert "mid_model_route" in names
    assert "strong_model_route" in names
    assert "best_route_style" in names

    # I. Input-adaptive compute strategies
    assert "difficulty_adaptive" in names
    assert "proxy_adaptive" in names

    # J. Verifier-guided search strategies
    assert "reasoning_plus_verifier" in names
    assert "search_plus_process_verifier" in names

    # K. Search-style reasoning
    assert "tree_of_thoughts_style" in names

    # L. Reason-act / tool-interleaving
    assert "react_style" in names


# ─────────────────────────────────────────────────────────────────────────────
# Component listing helpers
# ─────────────────────────────────────────────────────────────────────────────


def test_listing_helpers_return_declared_components():
    catalog = load_action_catalog()

    prompt_types = set(list_prompt_types(catalog))
    stages = set(list_stage_structures(catalog))
    model_slots_set = set(list_model_slots(catalog))

    # Original components still present
    assert "direct" in prompt_types
    assert "hint_guided_reasoning" in prompt_types
    assert "one_shot" in stages
    assert "first_pass_then_hint_guided_reason" in stages

    # New prompt types
    assert "token_budget" in prompt_types
    assert "process_verifier" in prompt_types
    assert "tree_of_thoughts" in prompt_types
    assert "react" in prompt_types

    # New stage structures
    assert "token_budget_constrained" in stages
    assert "early_exit" in stages
    assert "model_routing" in stages
    assert "difficulty_adaptive" in stages
    assert "proxy_adaptive" in stages
    assert "tree_of_thoughts" in stages
    assert "react" in stages

    # Model slots
    assert "cheap_model" in model_slots_set
    assert "middle_model" in model_slots_set
    assert "strong_model" in model_slots_set


def test_sample_counts_include_five():
    """self_consistency uses 5 samples; ensure it is declared."""
    catalog = load_action_catalog()
    sample_counts = set(catalog["available_components"]["sample_counts"])
    assert 5 in sample_counts

