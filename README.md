# Adaptive LLM Inference

Adaptive test-time compute allocation for LLM reasoning under budget constraints.

> **New here?** Read [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) for the
> full research goal, paper positioning, baseline families, and implementation plan.
> See [`docs/BASELINE_TRACKER.md`](docs/BASELINE_TRACKER.md) for the status of
> every baseline we plan to compare against.

## Overview

Given a batch of reasoning queries and a fixed compute budget, how should we
allocate test-time compute across queries to maximize accuracy?

This repository provides:

- **Baselines** — greedy, best-of-N, self-consistency (native); TALE, BEST-Route (external wrappers)
- **Allocators** — budget allocation strategies (currently: equal allocation)
- **Evaluation** — exact-match accuracy, per-query logging, compute tracking
- **Experiment runner** — config-driven script tying everything together

Currently uses a dummy model for pipeline validation.  The model interface
supports drop-in replacement with API-based LLMs.

## Project Structure

```
├── src/
│   ├── datasets/          # Dataset loaders (GSM8K, MATH500, routing dataset assembler)
│   ├── models/            # Model interface + dummy implementation
│   ├── baselines/         # Native baselines
│   │   └── external/      # Wrappers for official-code baselines
│   ├── allocators/        # Budget allocation strategies
│   ├── evaluation/        # Metrics and experiment logging
│   ├── features/          # Lightweight query/first-pass feature extraction
│   ├── analysis/          # Post-hoc analysis (feature gap, revise-helps)
│   ├── policies/          # Router baselines and routing policies
│   └── utils/             # Config loading, answer extraction
├── configs/               # YAML experiment configs
├── scripts/               # Experiment runner
├── tests/                 # Unit tests
├── docs/                  # Research documentation
│   ├── PROJECT_CONTEXT.md # ← read this first
│   ├── BASELINE_TRACKER.md
│   ├── ACTION_SPACE.md
│   ├── PRECOMPUTATION_FEATURES.md
│   ├── ROUTING_DATASET.md
│   ├── ROUTER_BASELINE.md
│   └── FEATURE_GAP_ANALYSIS.md
├── external/              # Official code from baseline papers
│   ├── tale/
│   └── best_route/
├── data/                  # Downloaded datasets (gitignored)
└── outputs/               # Experiment results (gitignored)
```

## Setup

```bash
pip install -e ".[dev]"
```

## Quick Start

```bash
# greedy baseline (1 sample per query, 50 GSM8K test queries)
python3 scripts/run_experiment.py --config configs/greedy.yaml

# best-of-N (5 samples per query)
python3 scripts/run_experiment.py --config configs/best_of_n.yaml

# self-consistency (majority vote, 5 samples)
python3 scripts/run_experiment.py --config configs/self_consistency.yaml

# equal allocation under budget constraint
python3 scripts/run_experiment.py --config configs/equal_allocator.yaml
```

Results are saved as JSON in `outputs/`.

## Strategy Expansion Experiment

Compares five inference strategies on a 20-query GSM8K subset using the real
OpenAI API (`gpt-4o-mini` by default).  Strategies:

| Strategy | Description | Samples/query |
|---|---|---|
| `direct_greedy` | Single direct prompt, greedy decode | 1 |
| `reasoning_best_of_3` | Step-by-step prompt, 3 samples, majority vote | 3 |
| `structured_sampling_3` | 3 distinct prompts (direct / step-by-step / solve+check), majority vote | 3 |
| `direct_plus_verify` | Direct answer + rule-based verifier; corrects if verifier rejects | 2 |
| `direct_plus_revise` | Direct answer + self-revision prompt; extract final revised answer | 2 |

```bash
# Requires OPENAI_API_KEY in environment
python3 scripts/run_strategy_expansion.py --config configs/strategy_expansion_gsm8k.yaml
```

Outputs are saved to `outputs/strategy_expansion/` (summary JSON, summary CSV,
per-query CSV).

## Expanded Strategy Smoke-Test

Adds two new prompt families and two new stage structures, compared against the
four inherited strategies on a 20-query GSM8K subset.

| New prompt family | Used in |
|---|---|
| `critique` | `direct_plus_critique_plus_final` |
| `hint-guided reasoning` | `first_pass_then_hint_guided_reason` |

| New stage structure | Stages | Samples/query |
|---|---|---|
| `direct_plus_critique_plus_final` | direct → critique → final-answer | 3 |
| `first_pass_then_hint_guided_reason` | direct → hint-guided re-solve | 2 |

```bash
# Requires OPENAI_API_KEY in environment
python3 scripts/run_expanded_strategy_smoke_test.py \
    --config configs/expanded_strategy_smoke_test_gsm8k.yaml
```

Outputs are saved to `outputs/expanded_strategy_smoke_test/` (summary JSON,
summary CSV, per-query CSV).  When blocked (no API key or no network), a
`summary.json` with `"run_status": "BLOCKED"` is written for traceability.

## Simulated Sweep Diagnostics

The repository also includes a lightweight synthetic analysis flow for studying
when classical allocation optimization helps over equal allocation and how that
advantage changes with noisy utility estimates.

```bash
# run the synthetic budget/noise sweep
python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml

# summarize sweep CSV outputs into a compact table and optional plots
python3 scripts/summarize_simulated_results.py
```

The summary script reads the CSV outputs under `outputs/simulated_sweep/`,
writes `outputs/simulated_sweep/summary_table.csv`, prints key budget/noise
metrics to the terminal, and saves simple plots under
`outputs/simulated_sweep/plots/` when `matplotlib` is available.

For robustness checks across synthetic instance draws, you can also run a
multi-seed version of the sweep:

```bash
# run the sweep across multiple synthetic seeds and aggregate the results
python3 scripts/run_simulated_multi_seed.py --config configs/simulated_multi_seed.yaml
```

This writes per-seed outputs plus aggregated budget/noise summaries under
`outputs/simulated_multi_seed/`, including mean/std utility, mean/std utility
gap, and the fraction of seeds where MCKP beats equal.

## Real-Data Budget Sweeps

For lightweight GSM8K budget sweeps with currently runnable native baselines:

```bash
# run a multi-budget GSM8K sweep with the current local/dummy model path
python3 scripts/run_real_budget_sweep.py --config configs/real_budget_sweep_gsm8k.yaml
```

This writes JSON/CSV summaries under `outputs/real_budget_sweep/`. The current
path uses the existing dummy/local model stack, so the results are useful for
pipeline validation and budget accounting rather than final real-model claims.

For a minimal real-LLM validation run on GSM8K:

```bash
# run greedy and best-of-N with an OpenAI-compatible API backend
python3 scripts/run_real_llm_experiment.py --config configs/real_llm_gsm8k.yaml
```

This writes summary JSON under `outputs/real_llm/` using a real OpenAI backend.

For a focused real-data diagnostic comparing prompt style and small extra
compute on the same GSM8K subset:

```bash
# compare direct vs reasoning-friendly prompting and best-of-3 / self-consistency
python3 scripts/run_real_llm_diagnostic.py --config configs/real_llm_gsm8k_diagnostic.yaml
```

This writes summary JSON plus per-query diagnostic CSV/JSON under
`outputs/real_llm_diagnostic/`.

For a small real-data gain-table / allocation-headroom estimate on GSM8K:

```bash
# estimate empirical per-query success at k=1/2/3 and compare uniform vs oracle
python3 scripts/run_real_gain_table.py --config configs/real_gain_table_gsm8k.yaml
```

This writes per-query gain tables and budget-comparison summaries under
`outputs/real_gain_table/`.

For a first selective-compute prototype on real GSM8K with a real OpenAI model:

```bash
# compare greedy, always-best-of-3, and selective escalation under one sample budget
python3 scripts/run_selective_escalation.py --config configs/selective_escalation_gsm8k.yaml
```

This writes summary JSON and per-query selective-escalation diagnostics under
`outputs/selective_escalation/`.

For a simple hybrid that first chooses direct vs reasoning mode and then
allocates a small reasoning budget only to selected queries:

```bash
# compare direct, reasoning, and a conservative mode-then-budget hybrid
python3 scripts/run_mode_then_budget.py --config configs/mode_then_budget_gsm8k.yaml
```

This writes summary JSON and per-query diagnostics under
`outputs/mode_then_budget/`.

For a tiny real-LLM sampling debug run to inspect sample diversity, parsing
collapse, and selective-escalation gating signals:

```bash
# record repeated raw samples and parsed answers on a 5-query GSM8K subset
python3 scripts/debug_real_llm_sampling.py
```

This writes raw sample traces and a compact sampling debug summary under
`outputs/debug_real_llm/`.

For a small diagnostic that compares the current cheap OpenAI model against one
stronger accessible OpenAI model, while also comparing naive best-of-3 against
simple structured 3-attempt sampling on the same GSM8K subset:

```bash
# compare model tier and small-budget sampling strategy on <=30 GSM8K queries
python3 scripts/run_model_sampling_diagnostic.py --config configs/model_sampling_diagnostic_gsm8k.yaml
```

This writes `summary.json`, `summary.csv`, and `per_query_results.csv` under
`outputs/model_sampling_diagnostic/`, including stronger-model access failures
when the configured stronger model is unavailable.

For the same small strategy comparison on the harder MATH500 benchmark:

```bash
# compare strategy usefulness on <=20 MATH500 queries with real OpenAI models
python3 scripts/run_strategy_diagnostic_math500.py \
    --config configs/strategy_diagnostic_math500.yaml
```

This writes `summary.json`, `summary.csv`, and `per_query_results.csv` under
`outputs/strategy_diagnostic_math500/`.

For an oracle-style subset analysis over the currently implemented GSM8K
strategies:

```bash
# compare direct, reasoning, sampling, and multi-stage strategies on a GSM8K subset
python3 scripts/run_oracle_subset_eval.py \
    --config configs/oracle_subset_eval_gsm8k.yaml
```

This writes `summary.json`, `summary.csv`, `per_query_matrix.csv`,
`oracle_assignments.csv`, and `pairwise_win_matrix.csv` under
`outputs/oracle_subset_eval/`.

For a first rule-based adaptive policy baseline that routes between direct,
reasoning, revise, and rare fallback strategies on GSM8K:

```bash
# run the simple adaptive strategy router on a small GSM8K subset
python3 scripts/run_adaptive_policy_eval.py \
    --config configs/adaptive_policy_gsm8k.yaml
```

This writes `summary.json`, `summary.csv`, and `per_query_results.csv` under
`outputs/adaptive_policy_v1/`.

For an **offline** comparison of adaptive routing policies v4/v5/v6 (no API; uses
documented false-positive fixtures and small recall proxies):

```bash
python3 scripts/run_adaptive_policy_v6_eval.py --config configs/adaptive_policy_v6_offline.yaml
```

Writes `summary.json`, `per_case_results.csv`, and `signal_summary.csv` under
`outputs/adaptive_policy_v6/`. See `docs/ADAPTIVE_POLICY_V6.md`.

For **offline** v5/v6/v7 comparison (false-positive fixtures, recall proxies, real probe snapshot):

```bash
python3 scripts/run_adaptive_policy_v7_eval.py
```

Writes under `outputs/adaptive_policy_v7/`. See `docs/ADAPTIVE_POLICY_V7.md`.

### Real GSM8K routing dataset and policy eval (API)

Build paired outcomes (`reasoning_greedy` + `direct_plus_revise`) and feature columns:

```bash
python3 scripts/run_build_real_routing_dataset.py --subset-size 100
```

Evaluate v5/v6/v7 on `data/real_gsm8k_routing_dataset.csv`:

```bash
python3 scripts/run_real_policy_eval.py
```

Train/evaluate tree ensembles on `revise_helpful` (requires scikit-learn):

```bash
python3 scripts/run_real_routing_model_eval.py
```

See `docs/REAL_GSM8K_ROUTING_STUDY.md` and `docs/REAL_ROUTING_MODEL_RESULTS.md`.

Hard-regime follow-up (MATH500 + mined hard GSM8K):

```bash
python3 scripts/run_select_hard_gsm8k.py --subset-size 100
python3 scripts/run_build_math500_routing_dataset.py --subset-size 100
python3 scripts/run_build_hard_gsm8k_routing_dataset.py
python3 scripts/run_cross_regime_comparison.py
```

See `docs/HARD_REGIME_ROUTING_STUDY.md`.

Next-stage EAAI experiments (oracle routing, budget sweep, `reasoning_then_revise`, baselines, AIME):

See `docs/NEXT_STAGE_EXPERIMENT_RESULTS.md` and scripts `run_reasoning_then_revise_addon.py`, `run_next_stage_postprocess.py`, `run_next_stage_baselines.py`, `run_build_aime_routing_dataset.py`, `run_final_cross_regime_summary.py`.

## Tests & Linting

```bash
pytest
ruff check src/ tests/ scripts/
```

## Config Format

```yaml
dataset:
  name: gsm8k
  split: test
  max_samples: 50

model:
  type: dummy
  correct_prob: 0.3
  seed: 42

baseline: greedy          # greedy | best_of_n | self_consistency
n_samples: 1
budget: 50

output: outputs/results.json
```

## License

MIT
