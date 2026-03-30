# Full Repository Audit

**Date:** 2026-03-28  
**Audit scope:** All source files, configs, docs, scripts, and tests in the repository.  
**Audit method:** Manual inspection of every file via `find`, `view`, `bash`; test collection; output-directory check.  
**Note:** A prior audit (`docs/REPO_AUDIT_REPORT.md`, dated 2026-03-27) covered an earlier state with 7 test files / 51 tests. This audit covers the current, substantially expanded state.

---

## 1. Repository Structure

```
.
├── AGENTS.md                   ← agent/CI instructions
├── LICENSE                     ← MIT
├── README.md                   ← project overview and quick-start
├── pyproject.toml              ← package metadata, dev deps, pytest config
├── .gitignore                  ← excludes data/, outputs/, caches
├── configs/                    ← 25 YAML experiment configs
├── docs/                       ← 16 markdown documentation files
├── external/
│   ├── tale/README.md          ← placeholder; no .repo submodule
│   └── best_route/README.md    ← placeholder; no .repo submodule
├── scripts/                    ← 22 Python experiment scripts
├── src/
│   ├── allocators/             ← budget allocation (equal, MCKP, robust, registry)
│   ├── analysis/               ← offline analysis modules
│   ├── baselines/              ← native baselines + external stubs
│   ├── datasets/               ← data loaders + bundled sample
│   ├── evaluation/             ← experiment orchestrators and metrics
│   ├── features/               ← cheap feature extraction layer
│   ├── methods/                ← adaptive method prototypes
│   ├── models/                 ← model interfaces + OpenAI backend + dummy
│   ├── policies/               ← adaptive routing policies v1–v4
│   ├── strategies/             ← action catalog loader/validator
│   └── utils/                  ← config loader, answer extraction
├── tests/                      ← 35 test files (see §6)
├── data/                       ← NOT present (gitignored; downloaded on demand)
└── outputs/                    ← NOT present (gitignored; created on first run)
```

### Top-level components in detail

| Component | Contents | Purpose |
|-----------|----------|---------|
| `src/allocators/` | `base.py`, `equal.py`, `mckp_allocator.py`, `robust_equalized.py`, `registry.py` | Budget allocation: uniform, exact-DP MCKP, robust equalised variant |
| `src/analysis/` | `feature_gap_analysis.py`, `revise_help_feature_analysis.py` | Offline analysis of which features separate routing groups |
| `src/baselines/` | `greedy.py`, `best_of_n.py`, `self_consistency.py`, `external/` stubs | Native inference baselines; stubs for TALE and BEST-Route |
| `src/datasets/` | `gsm8k.py`, `math500.py`, `routing_dataset.py`, `synthetic_ttc.py`, `bundled/gsm8k_test_sample.json` | Dataset loaders for GSM8K, MATH500; routing-dataset assembler; synthetic TTC generator; 20-query bundled sample |
| `src/evaluation/` | 16 evaluation orchestrator files, plus `metrics.py` and `logger.py` | Per-experiment orchestration; JSON/CSV logging; exact-match metrics |
| `src/features/` | `precompute_features.py`, `target_quantity_features.py`, `constraint_violation_features.py` | Three-family feature extraction layer for routing |
| `src/methods/` | `selective_escalation.py`, `mode_then_budget.py` | Adaptive inference-time method prototypes (not final baselines) |
| `src/models/` | `base.py`, `dummy.py`, `openai_llm.py`, `llm_model.py` | Abstract model interface; deterministic dummy; real OpenAI backend |
| `src/policies/` | `adaptive_policy_v1.py` – `adaptive_policy_v4.py`, `router_baseline.py` | Four generations of rule-based adaptive routing policy |
| `src/strategies/` | `action_catalog.py` | YAML catalog loader/validator for the action-space |
| `src/utils/` | `config.py`, `answer_extraction.py` | Config loading; numeric and symbolic answer normalisation |
| `configs/` | 25 YAML files | Experiment configurations (see §5) |
| `scripts/` | 22 Python scripts | Experiment launchers (see §5) |
| `tests/` | 35 test files | Unit and integration tests covering nearly every module |
| `docs/` | 16 markdown files | Research documentation, design specs, experiment logs |
| `external/` | `tale/README.md`, `best_route/README.md` | Placeholder READMEs; no official code cloned |

---

## 2. Datasets

### 2.1 GSM8K

| Field | Value |
|-------|-------|
| Loader module | `src/datasets/gsm8k.py` |
| Source | HuggingFace `openai/gsm8k`, split `test` (auto-download); also `src/datasets/bundled/gsm8k_test_sample.json` (20 queries, offline) |
| Answer type | Numeric (integer / decimal) extracted after `#### ` delimiter |
| Status | **Implemented and runnable**; bundled sample enables fully offline testing |
| Real-run outputs | None in `outputs/` (directory absent / gitignored) |

### 2.2 MATH500

| Field | Value |
|-------|-------|
| Loader module | `src/datasets/math500.py` |
| Source | HuggingFace `HuggingFaceH4/MATH-500`, split `test` |
| Answer type | Symbolic/mixed (LaTeX boxed, fractions, tuples); normalised by `normalize_math_answer` in `src/utils/answer_extraction.py` |
| Status | **Implemented and runnable**; requires network for download |
| Real-run outputs | None in `outputs/` |

### 2.3 Synthetic TTC

| Field | Value |
|-------|-------|
| Loader/generator | `src/datasets/synthetic_ttc.py` |
| Source | Programmatically generated monotone/concave/mixed utility tables |
| Purpose | Budget-allocation simulation experiments (MCKP, equal allocator) |
| Status | **Implemented and runnable**; requires no network or API key |

### 2.4 Routing Dataset

| Field | Value |
|-------|-------|
| Assembler | `src/datasets/routing_dataset.py` |
| Source | Merges query features from `extract_query_features()` with oracle labels from `outputs/oracle_subset_eval/` when available |
| Purpose | Offline supervised / rule-based routing analysis |
| Status | **Implemented**; runs in schema-only mode (no oracle data) or full mode (with oracle outputs) |

---

## 3. Strategies / Action Space

The action space is defined in `configs/action_space_catalog.yaml` and loaded/validated by `src/strategies/action_catalog.py`. Twelve strategy families are documented in `docs/ACTION_SPACE.md`.

### 3.1 Implemented strategies (end-to-end runnable)

| Strategy | Family | Implementation source | Status |
|----------|--------|-----------------------|--------|
| `direct_greedy` | A – Cheap/direct | `src/baselines/greedy.py` | ✅ Implemented |
| `strong_direct` | A – Cheap/direct | `src/baselines/greedy.py` (strong model slot) | ✅ Implemented |
| `reasoning_greedy` | B – Reasoning | `src/baselines/greedy.py` (reasoning prompt) | ✅ Implemented |
| `reasoning_best_of_3` | B – Reasoning | `src/baselines/best_of_n.py` | ✅ Implemented |
| `self_consistency` | B – Reasoning | `src/baselines/self_consistency.py` | ✅ Implemented |
| `structured_sampling_3` | C – Diverse sampling | `src/evaluation/expanded_strategy_eval.py` | ✅ Implemented |
| `direct_plus_verify` | D – Correction | `src/evaluation/expanded_strategy_eval.py` | ✅ Implemented |
| `direct_plus_revise` | D – Correction | `src/evaluation/expanded_strategy_eval.py` | ✅ Implemented |
| `direct_plus_critique_plus_final` | D – Correction | `src/evaluation/expanded_strategy_eval.py` | ✅ Implemented |
| `first_pass_then_hint_guided_reason` | E – Hint-guided | `src/evaluation/expanded_strategy_eval.py` | ✅ Implemented |

### 3.2 Placeholder / partial strategies

The following are defined in `configs/action_space_catalog.yaml` with status `placeholder` or `partial` — no runnable implementation exists:

- `token_budget_low`, `token_budget_mid`, `token_budget_high` (family F)
- `reasoning_with_early_exit` (family G)
- `cheap_model_route`, `mid_model_route`, `strong_model_route`, `best_route_style` (family H)
- `difficulty_adaptive`, `proxy_adaptive` (family I)
- `reasoning_plus_verifier`, `search_plus_process_verifier` (family J)
- `tree_of_thoughts_style` (family K)
- `react_style` (family L)

### 3.3 Adaptive policies (v1–v4)

These are not strategies themselves but **routing policies** that select among the implemented strategies per query at inference time.

| Policy | Source | Trigger for `direct_plus_revise` | Evidence of actual run |
|--------|--------|----------------------------------|------------------------|
| v1 | `src/policies/adaptive_policy_v1.py` | Any unstable first-pass output (too many numbers, parse failure, uncertainty phrases) → fired on 20/20 queries | No `outputs/` files exist |
| v2 | `src/policies/adaptive_policy_v2.py` | Boolean combination of 9 math-specific violation signals (conservative) → fired on 0/20 queries | No `outputs/` files exist |
| v3 | `src/policies/adaptive_policy_v3.py` | Weighted sum of same 9 signals with threshold sweep (conservative/medium/aggressive) | No `outputs/` files exist |
| v4 | `src/policies/adaptive_policy_v4.py` | Constraint-aware signals: answer-type mismatch, target-quantity mismatch, unit mismatch, impossible sign, integer/noninteger, percent/ratio mismatch, answer not in final statement, constraint-word conflict, bound violation | No `outputs/` files exist |

> **Note:** The v1/v2 firing rates documented above (20/20 and 0/20 respectively) are recorded in `docs/ADAPTIVE_POLICY_V3.md` as design context, not from an actual run in this repository.

---

## 4. Features / Routing Signals

### 4.1 Query-only features (`src/features/precompute_features.py`)

13 features computed from question text before any model call:

| Feature | Type | Description |
|---------|------|-------------|
| `question_length_chars` | int | Raw character count |
| `question_length_tokens_approx` | int | Whitespace-split token count |
| `num_numeric_mentions` | int | Count of numeric tokens |
| `num_sentences_approx` | int | Sentence count (split on `.!?`) |
| `has_multi_step_cue` | bool | Matches keywords: *total, remaining, after, left, difference, each, every, altogether, twice, half, percent, ratio, average, consecutive* |
| `has_equation_like_pattern` | bool | Inline arithmetic expression (`3 + 4 = 7`) |
| `has_percent_symbol` | bool | `\d+\s*%` |
| `has_fraction_pattern` | bool | `\d+/\d+` |
| `has_currency_symbol` | bool | `$€£¥₹` |
| `max_numeric_value_approx` | float | Largest numeric value |
| `min_numeric_value_approx` | float | Smallest numeric value |
| `numeric_range_approx` | float | max − min |
| `repeated_number_flag` | bool | Same token appears more than once |

**Status:** Implemented; used in adaptive policies v1–v4 and the router baseline.

### 4.2 Target-quantity / wording-trap features (`src/features/target_quantity_features.py`)

11 boolean features motivated by the feature-gap analysis:

| Feature family | Features |
|----------------|----------|
| Target-type cues | `asks_remaining_or_left`, `asks_total`, `asks_difference`, `asks_rate_or_unit`, `asks_money`, `asks_time` |
| Wording-trap signals | `has_subtraction_trap_verb`, `has_addition_trap_structure`, `has_multi_operation_hint` |
| Answer-risk signals | `likely_intermediate_quantity_ask`, `potential_answer_echo_risk` |

**Status:** Implemented; used in `revise_help_feature_analysis.py` and `feature_gap_analysis.py`. Not yet wired into adaptive policies v1–v4 (v4 uses constraint_violation_features instead).

### 4.3 Constraint-aware violation features (`src/features/constraint_violation_features.py`)

9 features that check question/answer consistency after a first reasoning pass:

| Check family | Features |
|---|---|
| Answer type/format | `answer_type_mismatch_suspected`, `answer_not_mentioned_in_final_statement_suspected`, `integer_expected_but_noninteger_suspected` |
| Quantity/target consistency | `target_quantity_mismatch_suspected`, `constraint_word_conflict_suspected` |
| Unit consistency | `unit_mismatch_suspected`, `percent_or_ratio_mismatch_suspected` |
| Plausibility | `impossible_sign_suspected`, `bound_violation_suspected` |

**Status:** Implemented; used by adaptive policy v4 via `extract_constraint_violation_features()`.

### 4.4 First-pass output features (`src/features/precompute_features.py`)

6 features computed from one cheap inference pass:

| Feature | Type | Description |
|---------|------|-------------|
| `first_pass_parse_success` | bool | Non-empty parsed answer or ≥1 numeric token |
| `first_pass_output_length` | int | Character length of model output |
| `first_pass_has_final_answer_cue` | bool | Output contains *final answer / therefore / the answer is* |
| `first_pass_has_uncertainty_phrase` | bool | Output contains *not sure / uncertain / it depends* |
| `first_pass_num_numeric_mentions` | int | Numeric token count in output |
| `first_pass_empty_or_malformed_flag` | bool | Output is empty or shorter than 3 characters |

**Status:** Implemented; not yet used in adaptive policies (require a model call, so excluded from v1 router baseline).

### 4.5 Router baseline (`src/policies/router_baseline.py`)

Uses all 13 query-only features (§4.1) to train/run:
- Majority baseline (no dependencies)
- Decision tree depth-3 (sklearn or pure-Python fallback)
- Logistic regression binary (sklearn required)

**Status:** Implemented; requires oracle labels from `outputs/routing_dataset/routing_dataset.csv`.

---

## 5. Experiments & Scripts

All scripts require `OPENAI_API_KEY` (for real-LLM experiments) or network access (for HuggingFace downloads) except those using the bundled sample or dummy model. **No `outputs/` directory exists; all real-LLM experiments are blocked.**

### 5.1 Core pipeline experiments

| Script | Config | Dataset | Model | Outputs expected | Status |
|--------|--------|---------|-------|-----------------|--------|
| `run_experiment.py` | `greedy.yaml`, `best_of_n.yaml`, `self_consistency.yaml`, `equal_allocator.yaml` | GSM8K | dummy | `outputs/*.json` | Runnable (dummy model, no API key needed) |
| `run_simulated_allocation.py` | `simulated_equal.yaml`, `simulated_mckp.yaml`, `simulated_sweep.yaml`, `simulated_multi_seed.yaml` | Synthetic TTC | — | `outputs/simulated_*.json` | Runnable (no API key needed) |
| `summarize_simulated_results.py` | — | — | — | `outputs/simulated_summary*.json` | Runnable (reads prior simulated outputs) |

### 5.2 Real-LLM diagnostic experiments

| Script | Config | Dataset | Model | Outputs expected | Status |
|--------|--------|---------|-------|-----------------|--------|
| `run_model_sampling_diagnostic.py` | `model_sampling_diagnostic_gsm8k.yaml` | GSM8K (8 queries) | gpt-4o-mini | `outputs/model_sampling_diagnostic/` | **BLOCKED** (no `OPENAI_API_KEY`) |
| `run_real_llm_experiment.py` | `real_llm_gsm8k.yaml` | GSM8K (20 queries) | gpt-4o-mini | `outputs/real_llm/` | **BLOCKED** |
| `run_real_llm_diagnostic.py` | `real_llm_gsm8k_diagnostic.yaml` | GSM8K | gpt-4o-mini | `outputs/real_llm_diagnostic/` | **BLOCKED** |
| `debug_real_llm_sampling.py` | — | — | gpt-4o-mini | console / `outputs/` | **BLOCKED** |

### 5.3 Strategy diagnostic / expansion experiments

| Script | Config | Dataset | Model | Outputs expected | Status |
|--------|--------|---------|-------|-----------------|--------|
| `run_strategy_diagnostic_math500.py` | `strategy_diagnostic_math500.yaml` | MATH500 (10 queries) | gpt-4o-mini/gpt-4o | `outputs/strategy_diagnostic_math500/` | **BLOCKED** |
| `run_strategy_expansion.py` | `strategy_expansion_gsm8k.yaml` | GSM8K (20 queries) | gpt-4o-mini | `outputs/strategy_expansion/` | **BLOCKED** |
| `run_expanded_strategy_smoke_test.py` | `expanded_strategy_smoke_test_gsm8k.yaml` | GSM8K (20 queries) | gpt-4o-mini | `outputs/expanded_strategy_smoke_test/` | **BLOCKED** |

### 5.4 Oracle subset evaluation

| Script | Config | Dataset | Model | Outputs expected | Status |
|--------|--------|---------|-------|-----------------|--------|
| `run_oracle_subset_eval.py` | `oracle_subset_eval_gsm8k.yaml` | GSM8K bundled (15–20 queries) | gpt-4o-mini | `outputs/oracle_subset_eval/{summary.json, summary.csv, per_query_matrix.csv, oracle_assignments.csv, pairwise_win_matrix.csv}` | **BLOCKED** (per `docs/EXPERIMENT_LOG_ORACLE_SUBSET.md` and `docs/RESULTS_ORACLE_SUBSET.md`) |

> **Provenance note:** `docs/ORACLE_ANALYSIS_SUMMARY.md` contains detailed numeric results (direct_greedy=0.50, reasoning_greedy=0.65, oracle=0.75 over 20 queries). The origin of these numbers is unclear: `docs/RESULTS_ORACLE_SUBSET.md` is marked BLOCKED (dated 2026-03-28) and no `outputs/oracle_subset_eval/` files exist. The ORACLE_ANALYSIS_SUMMARY document may reflect results obtained outside this repository, a prior run that has since been cleaned, or a designed/expected analysis rather than actual API output. **These numbers should not be treated as verified until reproduced.**

### 5.5 Adaptive policy evaluations

| Script | Config | Dataset | Model | Outputs expected | Status |
|--------|--------|---------|-------|-----------------|--------|
| `run_adaptive_policy_eval.py` | `adaptive_policy_gsm8k.yaml` | GSM8K bundled (20 queries) | gpt-4o-mini | `outputs/adaptive_policy_v1/{summary.json, summary.csv, per_query_results.csv}` | **BLOCKED** |
| `run_adaptive_policy_v2_eval.py` | `adaptive_policy_v2_gsm8k.yaml` | GSM8K bundled (20 queries) | gpt-4o-mini | `outputs/adaptive_policy_v2/{...}` | **BLOCKED** |
| `run_adaptive_policy_v3_eval.py` | `adaptive_policy_v3_gsm8k.yaml` | GSM8K bundled (20 queries) | gpt-4o-mini | `outputs/adaptive_policy_v3/{..., threshold_sweep.csv}` | **BLOCKED** |
| `run_adaptive_policy_v4_eval.py` | `adaptive_policy_v4_gsm8k.yaml` | GSM8K bundled (20 queries) | gpt-4o-mini/gpt-4o | `outputs/adaptive_policy_v4/{..., signal_firing_summary.csv}` | **BLOCKED** |

### 5.6 Feature analysis and routing dataset

| Script | Config | Inputs required | Outputs expected | Status |
|--------|--------|----------------|-----------------|--------|
| `run_feature_gap_analysis.py` | — | Oracle outputs (optional; graceful fallback) | `outputs/feature_gap_analysis/{group_feature_summary.csv, missed_revise_cases.csv, pattern_notes.json}` | Partially runnable (runs on empty groups without oracle data) |
| `run_revise_help_feature_analysis.py` | — | Oracle outputs (optional); bundled GSM8K fallback | `outputs/revise_help_feature_analysis/{group_feature_rates.csv, feature_differences.csv, query_feature_table.csv, example_cases.json}` | **Runnable offline** (uses bundled sample as fallback) |
| `build_routing_dataset.py` | — | Oracle outputs (optional; dry-run mode available) | `outputs/routing_dataset/{routing_dataset.csv, routing_dataset_summary.json}` | **Runnable offline** (schema-only mode) |
| `inspect_target_features.py` | — | Bundled GSM8K sample | Console output | **Runnable offline** |

### 5.7 Other experiments

| Script | Config | Notes | Status |
|--------|--------|-------|--------|
| `run_selective_escalation.py` | `selective_escalation_gsm8k.yaml` | Prototype: score-based escalation of top-k queries | **BLOCKED** |
| `run_mode_then_budget.py` | `mode_then_budget_gsm8k.yaml` | Prototype: direct→reasoning mode switch under budget | **BLOCKED** |
| `run_real_budget_sweep.py` | `real_budget_sweep_gsm8k.yaml` | Sweep over budget levels with dummy model | Runnable (dummy model) |
| `run_real_gain_table.py` | `real_gain_table_gsm8k.yaml` | Gain table at sample levels 1–3 | **BLOCKED** (real LLM) |
| `run_router_baseline.py` | — | Requires routing_dataset.csv with oracle labels | **BLOCKED** (requires oracle outputs) |

---

## 6. Test Coverage

35 test files (excluding `__init__.py`) covering essentially every module in `src/`:

| Test file | Modules tested |
|-----------|----------------|
| `test_allocators.py` | `EqualAllocator` |
| `test_mckp_allocator.py` | `MCKPAllocator` |
| `test_robust_allocator.py` | `RobustEqualizedAllocator` |
| `test_simulated_allocation.py` | Synthetic TTC + simulated evaluator + MCKP ≥ equal |
| `test_simulated_aggregate.py` | `simulated_aggregate.py` |
| `test_simulated_analysis.py` | `simulated_analysis.py` |
| `test_baselines.py` | `GreedyBaseline`, `BestOfNBaseline`, `SelfConsistencyBaseline` |
| `test_external_baselines.py` | TALE/BEST-Route not-installed behaviour |
| `test_metrics.py` | `exact_match`, `compute_accuracy` |
| `test_answer_extraction.py` | `extract_numeric_answer` |
| `test_action_catalog.py` | Action catalog loader/validator |
| `test_adaptive_policy.py` | `adaptive_policy_v1` |
| `test_adaptive_policy_v2.py` | `adaptive_policy_v2` |
| `test_adaptive_policy_v3.py` | `adaptive_policy_v3` |
| `test_adaptive_policy_v4.py` | `adaptive_policy_v4` |
| `test_precompute_features.py` | `precompute_features` |
| `test_target_quantity_features.py` | `target_quantity_features` |
| `test_constraint_violation_features.py` | `constraint_violation_features` |
| `test_feature_gap_analysis.py` | `feature_gap_analysis` |
| `test_revise_help_feature_analysis.py` | `revise_help_feature_analysis` |
| `test_routing_dataset.py` | `routing_dataset` assembler |
| `test_router_baseline.py` | `router_baseline` |
| `test_math500_loader.py` | `math500` loader |
| `test_model_sampling_diagnostic.py` | `model_sampling_diagnostic` |
| `test_oracle_subset_eval.py` | `oracle_subset_eval` |
| `test_expanded_strategy_eval.py` | `expanded_strategy_eval` |
| `test_strategy_expansion.py` | `strategy_expansion_eval` |
| `test_strategy_diagnostic.py` | `strategy_diagnostic` |
| `test_openai_llm.py` | `OpenAILLMModel` |
| `test_real_llm_debug.py` | `real_llm_debug` |
| `test_real_llm_diagnostic.py` | `real_llm_diagnostic` |
| `test_real_budget_analysis.py` | `real_budget_analysis` |
| `test_real_gain_table.py` | `real_gain_table` |
| `test_mode_then_budget.py` | `mode_then_budget` |
| `test_selective_escalation.py` | `selective_escalation` |

**pytest is not currently installed** (calling `python3 -m pytest` returns "No module named pytest"). Dependencies must be installed with `pip install -e ".[dev]"` before running the test suite.

---

## 7. Blockers and Missing Evidence

### Critical blockers

| Blocker | Detail |
|---------|--------|
| **No `OPENAI_API_KEY`** | All experiments requiring a real LLM are blocked. `OpenAILLMModel.__init__` raises `ValueError("Missing OPENAI_API_KEY")` before any queries are processed. Confirmed by `docs/EXPERIMENT_LOG_ORACLE_SUBSET.md` and `docs/RESULTS_ORACLE_SUBSET.md`. |
| **No `outputs/` directory** | The outputs directory is gitignored and does not exist in the clone. There are no committed result files — no JSON, CSV, or summary files from any experiment run. |
| **pytest not installed** | `python3 -m pytest` fails with "No module named pytest". The `pip install -e ".[dev]"` step must be run to activate the test suite. |

### Documentation discrepancies

| Issue | Detail |
|-------|--------|
| **ORACLE_ANALYSIS_SUMMARY.md vs RESULTS_ORACLE_SUBSET.md** | `ORACLE_ANALYSIS_SUMMARY.md` contains a detailed per-strategy results table for 20 queries (direct_greedy=0.50, reasoning_greedy=0.65, oracle=0.75). `RESULTS_ORACLE_SUBSET.md` and `EXPERIMENT_LOG_ORACLE_SUBSET.md` both state "BLOCKED" with no numeric results. No output files exist to corroborate either document. The numbers in ORACLE_ANALYSIS_SUMMARY.md cannot be verified from the current repository state. |
| **BASELINE_TRACKER.md is stale** | Marks MCKP as `📋 Planned` but the implementation (`src/allocators/mckp_allocator.py`) and its tests have been present since the prior audit. `RobustEqualizedAllocator` also exists but is not listed in the tracker. |
| **External baselines** | TALE and BEST-Route wrappers exist (`src/baselines/external/`), raise `NotImplementedError`, and their official repos have not been cloned under `external/<name>/.repo`. |

### Missing implementations

- No Snell et al. adaptive-compute baseline
- No TALE integration (official repo not cloned)
- No BEST-Route integration (official repo not cloned)
- No verification baselines (PRM/ORM)
- No difficulty-proxy allocator
- No token-budget, early-exit, tree-of-thoughts, or ReAct strategies (all placeholder)
- Vanilla CoT baseline not implemented (requires real LLM + reasoning prompt; differs from greedy only in prompt)

---

## 8. Next Steps (based only on audited state)

1. **Provide `OPENAI_API_KEY`** — this single step unblocks all real-LLM experiments. Start with `run_oracle_subset_eval.py` on the bundled 20-query sample (cheapest experiment) to produce ground-truth numbers.

2. **Install dev dependencies** — run `pip install -e ".[dev]"` to install pytest and ruff, then run `pytest` to confirm all tests pass and `ruff check --fix src/ tests/ scripts/` to clean up linting.

3. **Update BASELINE_TRACKER.md** — mark MCKP as ✅; add `RobustEqualizedAllocator` row; record current status of TALE/BEST-Route as "stub — official repo not cloned".

4. **Verify or retract ORACLE_ANALYSIS_SUMMARY.md numbers** — reproduce the oracle subset run and compare. If the numbers match, mark the doc as "verified". If they don't, update. Do not leave unverified numeric claims in docs.

5. **Run offline-capable experiments first** — `run_simulated_allocation.py`, `build_routing_dataset.py` (schema-only), `run_revise_help_feature_analysis.py`, and `run_real_budget_sweep.py` (dummy model) all run without an API key. Use these to smoke-test the pipeline.

6. **Clone TALE or BEST-Route** — integrate one external baseline end-to-end to validate the adapter pattern before attempting both.
