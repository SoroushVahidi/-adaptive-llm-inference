# Project Context — Adaptive LLM Inference

> **Read this file first.**  It contains everything a new contributor or agent
> needs to understand the research goal, positioning, baselines, datasets, and
> implementation plan.

---

## 1. Research Goal

We study **adaptive test-time compute allocation for LLM reasoning under a
limited inference budget**.

**Core question:** Given a batch of reasoning queries and a fixed compute
budget, how should a language model allocate test-time compute across queries
(or reasoning actions) to maximize total answer quality?

"Test-time compute" can mean any of:

- number of sampled reasoning traces (best-of-N)
- token budget per query
- search depth / tree width
- verifier calls
- model-routing decisions (cheap vs. expensive model)
- combinations of the above

## 2. Target Venue

**Engineering Applications of Artificial Intelligence (EAAI)**

The framing is AI/engineering, not pure theory.  Code, documentation, and
paper narrative should reflect applied AI with principled optimization backing.

## 3. Theory + AI Positioning

### Connection to combinatorial optimization

The **offline batch** version of the problem — allocate a discrete sample
budget across N queries to maximize total accuracy — is essentially a
**Multiple-Choice Knapsack Problem (MCKP)**.

- Each query is an "item class."
- Each possible sample count for that query is an "item" in that class.
- The "weight" is the number of samples; the "value" is the expected accuracy
  gain.
- The budget is the knapsack capacity.

This is a known NP-hard problem.  **We do not claim novelty for formulating
a new NP-hard problem.**

### Where the novelty comes from

The contribution is in **AI-specific structure** that makes the problem
interesting and tractable in practice:

| Novelty axis | Description |
|---|---|
| Monotonicity / diminishing returns | Success probability is monotone non-decreasing in compute; often concave (diminishing returns) |
| Prediction-based allocation | Using difficulty proxies or lightweight models to predict per-query compute needs |
| Online arrivals | Queries arrive sequentially; allocation must be decided before seeing future queries |
| Robust allocation | Allocation under noisy or adversarial difficulty predictions |
| Routing + budgeting | Joint decision of which model to use and how much compute to spend |
| Empirical superiority | Demonstrating gains over strong modern AI baselines (not just toy settings) |

### What NOT to claim

- Do not claim discovering a new NP-hard formulation.
- Do not present MCKP as a contribution — present it as the known structure we exploit.
- The contribution is the AI-specific modeling, algorithms, and empirical results.

## 4. Baseline Families

### 4.1 Fixed-compute baselines
| Baseline | Description |
|---|---|
| Direct answer / greedy | Single greedy decode |
| Vanilla chain-of-thought | Single CoT prompt |
| Fixed best-of-N | Sample N traces, pick majority answer |
| Self-consistency | Majority vote over N samples (Wang et al., 2022) |

### 4.2 Adaptive sampling baselines
| Baseline | Description |
|---|---|
| Snell et al. | Adaptive compute allocation via compute-optimal scaling |
| Difficulty-based allocation | Allocate more samples to harder queries |
| Training-free proxy allocation | Use lightweight proxies (entropy, confidence) to allocate |

### 4.3 Adaptive token-budget / stopping baselines
| Baseline | Description |
|---|---|
| TALE | Token-budget-aware LLM reasoning |
| SelfBudgeter | Self-allocated token budgets |
| DEER | Dynamic early exit for reasoning |

### 4.4 Verification-guided baselines
| Baseline | Description |
|---|---|
| Outcome verifiers | Score full solutions and pick the best |
| Process reward models | Score reasoning steps (Rewarding Progress / PRM scaling) |

### 4.5 Routing baselines
| Baseline | Description |
|---|---|
| Cheap-model-only | Always use the small/cheap model |
| Expensive-model-only | Always use the large/expensive model |
| Static threshold routing | Route by fixed confidence threshold |
| BEST-Route | Adaptive routing with test-time optimal compute |

### 4.6 Optimization / TCS baselines
| Baseline | Description |
|---|---|
| MCKP exact / DP | Classical dynamic programming on the knapsack formulation |
| MCKP approximation | FPTAS or greedy approximation |
| Online allocation | Learning-augmented or competitive-ratio allocation (future) |

## 5. Important Papers

These papers should be tracked, cited, and compared against:

1. **Snell et al.** — "Scaling LLM Test-Time Compute Optimally Can Be More Effective than Scaling Model Parameters for Reasoning"
2. **TALE** — "Token-Budget-Aware LLM Reasoning"
3. **BEST-Route** — "Adaptive LLM Routing with Test-Time Optimal Compute"
4. **Rewarding Progress** — "Scaling Automated Process Verifiers for LLM Reasoning"
5. **Training-Free Difficulty Proxies** — "Adaptive Test-Time Compute Allocation via Training-Free Difficulty Proxies"
6. **SelfBudgeter** — Self-allocated token budgets for LLM reasoning
7. **DEER** — Dynamic early exit for efficient LLM reasoning

See `docs/BASELINE_TRACKER.md` for official code availability and reproduction strategy.

## 6. Dataset Plan

### Stage 1 (current)
- **GSM8K** — grade-school math, 8.5K problems, standard reasoning benchmark

### Stage 2 (planned)
- **MATH / MATH500** — harder competition-style math
- **AIME-style subsets** — very hard math competition problems

### Stage 3 (possible)
- Broader reasoning benchmark (e.g., BBH, ARC, or similar)

## 7. Implementation Plan

### Current state (v0.1)

The repository has a working end-to-end pipeline:

- GSM8K dataset loader (auto-download via HuggingFace)
- Abstract model interface + dummy model (for pipeline testing)
- Three native baselines: greedy, best-of-N, self-consistency
- Equal-budget allocator (uniform allocation with remainder distribution)
- Exact-match evaluation with per-query JSON logging
- Config-driven experiment runner

### Near-term roadmap

1. **External baseline wrappers** — integrate TALE and BEST-Route via their
   official code (placeholder wrappers exist under `external/`)
2. **Real model backends** — API-based LLM integration (OpenAI, vLLM, etc.)
3. **Smarter allocators** — difficulty-proxy allocation, MCKP-based allocation
4. **Additional datasets** — MATH/MATH500
5. **Verification baselines** — outcome reward models, process reward models

### Architecture for baselines

The repo supports two baseline types:

- **Native baselines** — implemented directly in `src/baselines/`
- **External baselines** — wrappers around official author code, managed under
  `external/` with thin adapter classes in `src/baselines/external/`

This avoids reimplementing published methods when official code exists.

## 8. Repository Layout

```
├── src/
│   ├── datasets/          # Dataset loaders
│   ├── models/            # Model interface + implementations
│   ├── baselines/         # Native baselines + external wrappers
│   │   └── external/      # Thin wrappers for official-code baselines
│   ├── allocators/        # Budget allocation strategies
│   ├── evaluation/        # Metrics and experiment logging
│   └── utils/             # Config, answer extraction, helpers
├── configs/               # YAML experiment configs
├── scripts/               # Experiment runner scripts
├── tests/                 # Unit tests
├── docs/                  # Research documentation
│   ├── PROJECT_CONTEXT.md # ← this file
│   └── BASELINE_TRACKER.md
├── external/              # Official code from baseline papers
│   ├── tale/
│   └── best_route/
├── data/                  # Downloaded datasets (gitignored)
└── outputs/               # Experiment results (gitignored)
```

## 9. Getting Started

```bash
# install
pip install -e ".[dev]"

# run tests
pytest

# run first experiment (dummy model, 50 GSM8K queries)
python3 scripts/run_experiment.py --config configs/greedy.yaml
```

See `README.md` for more commands and config format.
