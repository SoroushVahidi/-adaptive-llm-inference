from src.strategies.action_catalog import (
    list_curated_strategies,
    list_prompt_types,
    list_stage_structures,
    load_action_catalog,
)


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


def test_expected_core_strategies_exist():
    catalog = load_action_catalog()
    names = {strategy["name"] for strategy in list_curated_strategies(catalog)}

    expected = {
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

    assert expected.issubset(names)


def test_listing_helpers_return_declared_components():
    catalog = load_action_catalog()

    prompt_types = set(list_prompt_types(catalog))
    stages = set(list_stage_structures(catalog))

    assert "direct" in prompt_types
    assert "hint_guided_reasoning" in prompt_types
    assert "one_shot" in stages
    assert "first_pass_then_hint_guided_reason" in stages
