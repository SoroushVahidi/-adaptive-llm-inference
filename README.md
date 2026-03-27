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
│   ├── datasets/          # Dataset loaders (GSM8K)
│   ├── models/            # Model interface + dummy implementation
│   ├── baselines/         # Native baselines
│   │   └── external/      # Wrappers for official-code baselines
│   ├── allocators/        # Budget allocation strategies
│   ├── evaluation/        # Metrics and experiment logging
│   └── utils/             # Config loading, answer extraction
├── configs/               # YAML experiment configs
├── scripts/               # Experiment runner
├── tests/                 # Unit tests
├── docs/                  # Research documentation
│   ├── PROJECT_CONTEXT.md # ← read this first
│   └── BASELINE_TRACKER.md
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
