# Action Space Catalog

This document defines the expanded **inference action space** used for strategy-expansion
experiments in the Adaptive LLM Inference project.

## 1) Action Definition

Each inference strategy is represented as a four-tuple:

```
a = (p, k, s, m)
```

| Symbol | Field | Description |
|--------|-------|-------------|
| `p` | `prompt_types` | Ordered list of prompt families used across stages (e.g., `[direct]`, `[reasoning, verify]`). |
| `k` | `sample_count` | Number of candidate generations produced per stage. |
| `s` | `stage_structure` | The sequence / graph of stages (single-shot, parallel vote, correction chain, search, …). |
| `m` | `model_slot` | Which capacity tier executes the strategy (`cheap_model`, `middle_model`, `strong_model`). |

Two additional fields are tracked for every catalog entry:

- **`status`** — whether the strategy is `implemented`, `partial`, or `placeholder` (see §4).
- **`expected_cost_tier`** — coarse cost estimate: `low`, `medium`, or `high`.

## 2) Why Factor the Space

Separating prompt, samples, stage, and model gives us a clean experimental abstraction:

- Supports systematic comparisons while changing one axis at a time.
- Makes compute-cost modeling explicit (sample count × stage depth × model tier).
- Keeps implementation modular for allocators and future policy learning.

## 3) Why the Catalog Is Broader Than the Implemented Subset

A full Cartesian product over all components is large and inefficient for paper-oriented
experiments.  We therefore maintain a **curated candidate list** in
`configs/action_space_catalog.yaml` that:

- Includes **all major strategy families** found in the literature (§5), even if some are
  only placeholders not yet wired to runnable code.
- Omits low-value or redundant combinations.
- Provides an explicit `status` field so readers and developers know exactly what is ready.

This design lets us reason about the full strategy universe while keeping the runnable
baseline set small enough for budget-constrained experiments.

## 4) Status Levels

| Status | Meaning |
|--------|---------|
| `implemented` | Strategy has an end-to-end runnable path via existing `src/baselines/` or `src/methods/` code. |
| `partial` | Components (prompt or model) exist, but the full execution pipeline is not yet wired together. |
| `placeholder` | Catalog and documentation entry only; no implementation exists yet. |

## 5) Literature-Inspired Strategy Families

The catalog covers twelve families drawn from recent inference-time compute literature:

| Family | Representative strategies | Notes / Literature |
|--------|--------------------------|------------|
| **A. Cheap / direct baselines** | `direct_greedy`, `strong_direct` | Standard greedy decoding baselines |
| **B. Reasoning baselines** | `reasoning_greedy`, `reasoning_best_of_3`, `self_consistency` | Wang et al., 2022 (Self-Consistency) |
| **C. Structured prompting / diverse sampling** | `structured_sampling_3`, `direct_plus_double_check` | Diverse prompt sampling |
| **D. Sequential correction / self-improvement** | `direct_plus_verify`, `direct_plus_revise`, `direct_plus_critique_plus_final` | Self-Refine (Madaan et al., 2023) |
| **E. Hint-guided reasoning** | `first_pass_then_hint_guided_reason` | Internal technique; hint-augmented chain-of-thought |
| **F. Token-budget strategies** | `token_budget_low`, `token_budget_mid`, `token_budget_high` | Budget-constrained inference (exploratory) |
| **G. Early-exit strategies** | `reasoning_with_early_exit` | Early-exit transformers (Schwartz et al., 2020) |
| **H. Model-routing strategies** | `cheap_model_route`, `mid_model_route`, `strong_model_route`, `best_route_style` | BEST-Route (Chen et al., 2023) |
| **I. Input-adaptive compute** | `difficulty_adaptive`, `proxy_adaptive` | Adaptive compute allocation (this project) |
| **J. Verifier-guided search** | `reasoning_plus_verifier`, `search_plus_process_verifier` | PRM / ORM re-ranking (Cobbe et al., 2021; Lightman et al., 2023) |
| **K. Search-style reasoning** | `tree_of_thoughts_style` | Tree of Thoughts (Yao et al., 2023) |
| **L. Reason-act / tool-interleaving** | `react_style` | ReAct (Yao et al., 2022) |

## 6) Available Component Values

### Prompt types
- `direct` — plain answer prompt
- `reasoning` — chain-of-thought prompt
- `structured_direct` — answer with structured output format
- `structured_reasoning` — chain-of-thought with structured output
- `double_check` — explicit double-check / sanity-check prompt
- `verify` — dedicated verification prompt (is the answer correct?)
- `revise` — self-revision prompt (revise your answer)
- `critique` — critique prompt (find flaws in your reasoning)
- `hint_guided_reasoning` — reasoning augmented with a hint
- `token_budget` — prompt with an explicit token-budget constraint
- `process_verifier` — step-level process reward signal
- `tree_of_thoughts` — tree-structured exploration prompt *(placeholder)*
- `react` — reason + act interleaving prompt *(placeholder)*

### Sample counts
- `1` — single sample (greedy / deterministic)
- `2` — two candidates
- `3` — three candidates
- `5` — five candidates (standard self-consistency)

### Stage structures
| Structure | Description |
|-----------|-------------|
| `one_shot` | Single forward pass |
| `parallel_vote` | Multiple independent samples → majority vote |
| `direct_then_verify` | Answer, then verify |
| `direct_then_double_check` | Answer, then double-check *(placeholder)* |
| `direct_then_revise` | Answer, then revise |
| `direct_then_critique_then_final` | Answer → critique → final answer |
| `first_pass_then_hint_guided_reason` | Quick pass → hint-guided re-solve |
| `token_budget_constrained` | Single pass with explicit token budget *(placeholder)* |
| `early_exit` | Stop early on confidence signal *(placeholder)* |
| `model_routing` | Route query to best model tier *(placeholder)* |
| `difficulty_adaptive` | Allocate samples by difficulty *(placeholder)* |
| `proxy_adaptive` | Allocate samples by proxy signal *(placeholder)* |
| `reasoning_plus_verifier` | Samples + outcome verifier re-ranking *(placeholder)* |
| `search_plus_process_verifier` | Search + per-step PRM pruning *(placeholder)* |
| `tree_of_thoughts` | Tree-structured reasoning search *(placeholder)* |
| `react` | Interleaved reason-act loop *(placeholder)* |

### Model slots
- `cheap_model` — lowest-cost model tier
- `middle_model` — intermediate capacity/cost
- `strong_model` — highest-quality model tier

## 7) Currently Implemented Strategies

The following strategies have end-to-end runnable code:

| Strategy | Family | Source |
|----------|--------|--------|
| `direct_greedy` | A | `src/baselines/greedy.py` |
| `strong_direct` | A | `src/baselines/greedy.py` |
| `reasoning_greedy` | B | `src/baselines/greedy.py` |
| `reasoning_best_of_3` | B | `src/baselines/best_of_n.py` |
| `self_consistency` | B | `src/baselines/self_consistency.py` |
| `structured_sampling_3` | C | `src/evaluation/expanded_strategy_eval.py` |
| `direct_plus_verify` | D | `src/evaluation/expanded_strategy_eval.py` |
| `direct_plus_revise` | D | `src/evaluation/expanded_strategy_eval.py` |
| `direct_plus_critique_plus_final` | D | `src/evaluation/expanded_strategy_eval.py` |
| `first_pass_then_hint_guided_reason` | E | `src/evaluation/expanded_strategy_eval.py` |

All other strategies in the catalog are `placeholder` (or `partial`) and are reserved for
future implementation.

