# Adaptive LLM Inference

Adaptive test-time compute allocation for LLM reasoning under budget constraints.

## Overview

Given a batch of reasoning queries and a fixed compute budget (total number of LLM samples), how should we allocate samples across queries to maximize accuracy?

This project provides:

- **Baselines** — greedy (1 sample), best-of-N, self-consistency (majority vote)
- **Allocators** — strategies that distribute a global sample budget across queries
- **Evaluation** — exact-match accuracy, per-query logging, compute tracking
- **Experiment runner** — config-driven script that ties everything together

Currently uses a dummy model for pipeline validation. The model interface is designed for drop-in replacement with API-based LLMs.

## Project Structure

```
├── src/
│   ├── datasets/       # Dataset loaders (GSM8K)
│   ├── models/         # Model interface + dummy implementation
│   ├── baselines/      # Greedy, best-of-N, self-consistency
│   ├── allocators/     # Budget allocation strategies
│   ├── evaluation/     # Metrics and experiment logging
│   └── utils/          # Config loading, answer extraction
├── configs/            # YAML experiment configs
├── scripts/            # Experiment runner
├── tests/              # Unit tests
├── data/               # Downloaded datasets (gitignored)
└── outputs/            # Experiment results (gitignored)
```

## Setup

```bash
pip install -e ".[dev]"
```

## Running an Experiment

```bash
# Greedy baseline (1 sample per query)
python scripts/run_experiment.py --config configs/greedy.yaml

# Best-of-N (5 samples per query, majority vote)
python scripts/run_experiment.py --config configs/best_of_n.yaml

# Self-consistency
python scripts/run_experiment.py --config configs/self_consistency.yaml

# Equal allocation with budget constraint
python scripts/run_experiment.py --config configs/equal_allocator.yaml
```

Results are saved as JSON in `outputs/`.

## Running Tests

```bash
pytest
```

## Linting

```bash
ruff check src/ tests/ scripts/
```

## Config Format

Configs are YAML files with these fields:

```yaml
dataset:
  name: gsm8k
  split: test
  max_samples: 50        # limit queries for fast iteration

model:
  type: dummy
  correct_prob: 0.3      # probability the dummy model returns the correct answer
  seed: 42

baseline: greedy         # greedy | best_of_n | self_consistency
n_samples: 1             # samples per query (for non-greedy baselines)
budget: 50               # total sample budget

output: outputs/results.json
```

## License

MIT
