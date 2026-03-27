# Baseline Tracker

Status of every baseline we plan to compare against.

> **Policy:** If a baseline has official public code from the authors, prefer
> using or adapting that code.  Only reimplement when official code is
> unavailable or unusable.

## Status Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Done / integrated |
| 🔧 | In progress |
| 📋 | Planned |
| — | Not applicable |

## Tracking Table

| # | Baseline | Paper | Venue / Year | Official Code URL | Official Code Available? | Reproduction Strategy | Status | Notes |
|---|----------|-------|-------------|-------------------|-------------------------|----------------------|--------|-------|
| 1 | Direct answer / greedy | — (standard) | — | — | — | Native impl | ✅ | `src/baselines/greedy.py` |
| 2 | Vanilla CoT | Wei et al. "Chain-of-Thought Prompting Elicits Reasoning in LLMs" | NeurIPS 2022 | — | — | Native impl (prompt-level) | 📋 | Requires real LLM backend; differs from greedy only in prompt |
| 3 | Fixed best-of-N | — (standard) | — | — | — | Native impl | ✅ | `src/baselines/best_of_n.py` |
| 4 | Self-consistency | Wang et al. "Self-Consistency Improves Chain of Thought Reasoning in LLMs" | ICLR 2023 | — | — | Native impl | ✅ | `src/baselines/self_consistency.py` |
| 5 | Snell et al. | "Scaling LLM Test-Time Compute Optimally…" | arXiv 2024 | Unknown | Unknown | TBD — check for code release | 📋 | Key adaptive-compute baseline; search for official repo |
| 6 | TALE | "Token-Budget-Aware LLM Reasoning" | arXiv 2024 | [GitHub](https://github.com/ChenWu98/TALE) | **Yes** | Adapt official code | 📋 | Wrapper stub at `external/tale/`; official repo confirmed public |
| 7 | BEST-Route | "Adaptive LLM Routing with Test-Time Optimal Compute" | arXiv 2024 | [GitHub](https://github.com/best-route/best-route) | **Yes** | Adapt official code | 📋 | Wrapper stub at `external/best_route/`; official repo confirmed public |
| 8 | Rewarding Progress / PRM Scaling | "Scaling Automated Process Verifiers for LLM Reasoning" | arXiv 2024 | Unknown | Unknown | TBD — check for code release | 📋 | Verification-guided baseline |
| 9 | Training-Free Difficulty Proxies | "Adaptive Test-Time Compute Allocation via Training-Free Difficulty Proxies" | arXiv 2024 | Unknown | Unknown | TBD — check for code release | 📋 | Proxy-based adaptive allocation |
| 10 | SelfBudgeter | "SelfBudgeter" | arXiv 2024 | Unknown | Unknown | TBD — check for code release | 📋 | Self-allocated token budgets |
| 11 | DEER | "DEER: Dynamic Early Exit for Efficient LLM Reasoning" | arXiv 2024 | Unknown | Unknown | TBD — check for code release | 📋 | Early-exit / stopping baseline |
| 12 | MCKP optimization baseline | — (classical) | — | — | — | Native impl (DP / greedy approx) | 📋 | Classical knapsack baseline for the batch formulation |

## Notes on Official Code

- **TALE:** Official repo is public. Integration via `external/tale/` + wrapper in `src/baselines/external/tale_wrapper.py`.
- **BEST-Route:** Official repo is public. Integration via `external/best_route/` + wrapper in `src/baselines/external/best_route_wrapper.py`.
- **Snell et al.:** No official code link found yet. Re-check periodically.
- **Rewarding Progress:** No official code link found yet. May require reimplementation of process reward model evaluation.

## How to Add a New Baseline

1. Check for official code; record in this table.
2. If official code exists: clone/submodule under `external/`, write a thin wrapper in `src/baselines/external/`.
3. If no official code: implement natively in `src/baselines/`.
4. Register the baseline name in `scripts/run_experiment.py` and add a config under `configs/`.
5. Update this tracker.
