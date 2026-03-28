# Action Space Catalog

This document defines the expanded **inference action space** used for strategy-expansion experiments.

## 1) Action Definition

We represent each action/strategy as:

\[
a = (\text{prompt\_type}, \text{num\_samples}, \text{stage\_structure}, \text{model\_choice})
\]

- **prompt_type**: what instruction format is used at each stage (e.g., direct vs. reasoning vs. critique).
- **num_samples**: how many candidate generations are produced for a stage.
- **stage_structure**: the sequence/graph of stages (single-shot, verify-after-direct, critique loops, etc.).
- **model_choice**: which capacity tier executes the strategy (cheap, middle, strong).

## 2) Why Factor the Space

Separating prompt, samples, stage, and model gives us a clean experimental abstraction:

- Supports systematic comparisons while changing one axis at a time.
- Makes compute-cost modeling explicit (sample count + stage depth + model tier).
- Keeps implementation modular for allocators and future policy learning.

## 3) Why We Do **Not** Use the Full Cartesian Product

A full Cartesian product over all components is large and inefficient for paper-oriented experiments.

Reasons for a curated subset:

- Many combinations are low-value or redundant.
- Some combinations are implausible in practice (e.g., expensive multi-stage chains on all queries).
- We need interpretable, budget-aware baselines first, not exhaustive search.

Therefore, we maintain a **curated candidate list** in `configs/action_space_catalog.yaml`.

## 4) Available Component Values

### Prompt types
- `direct`
- `reasoning`
- `structured_direct`
- `structured_reasoning`
- `double_check`
- `verify`
- `revise`
- `critique`
- `hint_guided_reasoning`

### Sample counts
- `1`
- `2`
- `3`

### Stage structures
- `one_shot`
- `parallel_vote`
- `direct_then_verify`
- `direct_then_revise`
- `direct_then_critique_then_final`
- `first_pass_then_hint_guided_reason`

### Model slots
- `cheap_model`
- `middle_model` (optional placeholder)
- `strong_model`

## 5) Currently Recommended Curated Strategies

- `direct_greedy`
- `reasoning_greedy`
- `reasoning_best_of_3`
- `structured_sampling_3`
- `direct_plus_verify`
- `direct_plus_revise`
- `direct_plus_critique_plus_final`
- `first_pass_then_hint_guided_reason`
- `strong_direct`
- `strong_structured_placeholder`

Each strategy entry includes prompt type(s), sample count, stage structure, model slot, rationale, and expected cost tier (`low` / `medium` / `high`) in the YAML catalog.
