# Repository Audit Report

**Date:** 2026-03-27
**Branch audited:** `copilot/audit-repository-state`
**Audit method:** Manual inspection of all source files, docs, configs, and tests; full test-suite run; linter run.

---

## 1. Repository Snapshot

### Top-level layout (all present unless noted)

```
.
├── AGENTS.md              ← agent instructions (present)
├── LICENSE                ← MIT (present)
├── README.md              ← project overview (present)
├── pyproject.toml         ← build / dev deps, pytest config (present)
├── .gitignore             ← ignores data/, outputs/, caches
├── configs/               ← 6 YAML experiment configs (present)
├── docs/                  ← 2 of 5 expected docs present (see §3)
├── external/
│   ├── tale/README.md     ← placeholder README only; no .repo
│   └── best_route/README.md ← placeholder README only; no .repo
├── scripts/
│   ├── run_experiment.py  ← real-dataset pipeline
│   └── run_simulated_allocation.py ← synthetic allocation pipeline
├── src/
│   ├── allocators/        ← EqualAllocator, MCKPAllocator, registry
│   ├── baselines/         ← greedy, best_of_n, self_consistency + external stubs
│   ├── datasets/          ← GSM8K loader, synthetic TTC generator
│   ├── evaluation/        ← metrics, logger, simulated evaluator
│   ├── models/            ← abstract Model base, DummyModel
│   └── utils/             ← config loader, answer extraction
├── tests/                 ← 7 test files, 51 tests (all pass)
├── data/                  ← NOT present (gitignored; downloaded on demand)
└── outputs/               ← NOT present (gitignored; created on first run)
```

**Python version:** 3.12 (system python3; no `python` symlink).
**Package:** `adaptive-llm-inference` v0.1.0; installed with `pip install -e ".[dev]"`.

---

## 2. Implemented Components

### Dataset loading
| Component | File | Status |
|-----------|------|--------|
| GSM8K loader | `src/datasets/gsm8k.py` | **Implemented and runnable** — auto-downloads via HuggingFace `datasets`; requires network on first use; normalises `#### <answer>` format |
| Synthetic TTC generator | `src/datasets/synthetic_ttc.py` | **Implemented and runnable** — generates monotone/concave/mixed-difficulty utility tables; supports custom cost vectors and seeds |

### Models
| Component | File | Status |
|-----------|------|--------|
| Abstract `Model` base | `src/models/base.py` | **Implemented** — defines `generate()` and `generate_n()` |
| `DummyModel` | `src/models/dummy.py` | **Implemented and runnable** — stochastic with controllable `correct_prob`; supports `set_ground_truth()` |
| Real LLM backend | *(absent)* | **Not implemented** — only the abstract interface exists |

### Baselines
| Baseline | File | Status |
|----------|------|--------|
| Greedy / direct answer | `src/baselines/greedy.py` | **Implemented and runnable** |
| Best-of-N | `src/baselines/best_of_n.py` | **Implemented and runnable** |
| Self-consistency | `src/baselines/self_consistency.py` | **Implemented and runnable** |
| Vanilla CoT | *(absent)* | **Not implemented** — documented; requires real LLM |
| TALE | `src/baselines/external/tale_wrapper.py` | **Placeholder/wrapper only** — raises `NotImplementedError`; official repo not cloned |
| BEST-Route | `src/baselines/external/best_route_wrapper.py` | **Placeholder/wrapper only** — raises `NotImplementedError`; official repo not cloned |
| Snell et al. | *(absent)* | **Documented only** |
| Rewarding Progress / PRM | *(absent)* | **Documented only** |
| Training-Free Difficulty Proxies | *(absent)* | **Documented only** |
| SelfBudgeter | *(absent)* | **Documented only** |
| DEER | *(absent)* | **Documented only** |

### Allocators
| Allocator | File | Status |
|-----------|------|--------|
| Equal allocator | `src/allocators/equal.py` | **Implemented and runnable** — dual-mode: legacy `allocate(n_queries, budget)` and simulated `allocate(profits, costs, budget)` |
| MCKP allocator | `src/allocators/mckp_allocator.py` | **Implemented and runnable** — exact DP solver with backtracking; input validation; numpy support |
| Allocator registry | `src/allocators/registry.py` | **Implemented** — maps `"equal"` and `"mckp"` to instances |
| Difficulty-proxy allocator | *(absent)* | **Not implemented** |

### Evaluation
| Component | File | Status |
|-----------|------|--------|
| Exact-match metric | `src/evaluation/metrics.py` | **Implemented and runnable** |
| Per-query JSON logger | `src/evaluation/logger.py` | **Implemented and runnable** |
| Simulated evaluator | `src/evaluation/simulated_evaluator.py` | **Implemented and runnable** — orchestrates allocator + utility-table scoring; validates budget feasibility |

### Utils
| Component | File | Status |
|-----------|------|--------|
| YAML/JSON config loader | `src/utils/config.py` | **Implemented** |
| Numeric answer extraction | `src/utils/answer_extraction.py` | **Implemented** |

### Config-driven execution
Six YAML configs present: `greedy.yaml`, `best_of_n.yaml`, `self_consistency.yaml`, `equal_allocator.yaml`, `simulated_equal.yaml`, `simulated_mckp.yaml`. All map to runnable pipelines.

---

## 3. Documentation Status

| File | Present? | Summary |
|------|----------|---------|
| `README.md` | ✅ | Project overview, quick-start commands, config format — accurate and up to date |
| `docs/PROJECT_CONTEXT.md` | ✅ | Research goal, MCKP framing, baseline families, implementation plan, dataset roadmap — comprehensive |
| `docs/BASELINE_TRACKER.md` | ✅ | 12-row table tracking every planned baseline with paper, venue, official code URL, and status |
| `docs/PAPER_NOTES.md` | ❌ **Missing** | Referenced nowhere in code but expected per audit spec |
| `docs/REPRODUCTION_PLAN.md` | ❌ **Missing** | Expected per audit spec; not present |
| `docs/OPEN_QUESTIONS.md` | ❌ **Missing** | Expected per audit spec; not present |

---

## 4. Implemented Baselines / Allocators — Classification

| Name | Classification |
|------|---------------|
| Greedy / direct answer | **Implemented and runnable** |
| Vanilla CoT | **Documented only** (needs real LLM) |
| Best-of-N | **Implemented and runnable** |
| Self-consistency | **Implemented and runnable** |
| Equal allocator | **Implemented and runnable** |
| MCKP allocator | **Implemented and runnable** |
| TALE wrapper | **Placeholder / wrapper only** (official repo not cloned) |
| BEST-Route wrapper | **Placeholder / wrapper only** (official repo not cloned) |
| Snell et al. | **Not present** |
| Rewarding Progress / PRM | **Not present** |
| Training-Free Difficulty Proxies | **Not present** |
| SelfBudgeter | **Not present** |
| DEER | **Not present** |
| Difficulty-proxy allocator | **Not present** |

---

## 5. Runnable Entry Points

### `scripts/run_experiment.py`
- **Purpose:** Load GSM8K, instantiate a model and baseline, optionally apply the equal allocator, evaluate, and save JSON results.
- **Supported baselines:** `greedy`, `best_of_n`, `self_consistency`
- **Supported models:** `dummy` only
- **Run example:** `python3 scripts/run_experiment.py --config configs/greedy.yaml`
- **Output:** Two JSON files — full per-query log and compact summary — written to the path specified in config (default: `outputs/`).
- **Limitation:** No MCKP allocator integration in this script (uses `EqualAllocator` hardcoded); `n_samples` config key interacts with allocation in a non-obvious way.

### `scripts/run_simulated_allocation.py`
- **Purpose:** Generate synthetic utility tables and run an allocator (equal or mckp) against them; save full JSON output.
- **Supported allocators:** `equal`, `mckp`
- **Run example:** `python3 scripts/run_simulated_allocation.py --config configs/simulated_mckp.yaml`
- **Output:** Single JSON file with config, instance metadata, and evaluation results.
- **Status:** Fully runnable without network access.

---

## 6. Test Status

### Test files

| File | What it tests | Passes? |
|------|---------------|---------|
| `tests/test_allocators.py` | `EqualAllocator` legacy mode (4 tests) | ✅ |
| `tests/test_answer_extraction.py` | `extract_numeric_answer` (6 tests) | ✅ |
| `tests/test_baselines.py` | `GreedyBaseline`, `BestOfNBaseline`, `SelfConsistencyBaseline` with `DummyModel` (4 tests) | ✅ |
| `tests/test_external_baselines.py` | `TALEBaseline` and `BESTRouteBaseline` not-installed behaviour (4 tests) | ✅ |
| `tests/test_mckp_allocator.py` | `MCKPAllocator` correctness, budget constraints, invariants, input validation, import surface (20 tests) | ✅ |
| `tests/test_metrics.py` | `exact_match`, `compute_accuracy` (7 tests) | ✅ |
| `tests/test_simulated_allocation.py` | Synthetic TTC generation, simulated evaluator, MCKP ≥ equal on nontrivial instance (6 tests) | ✅ |

**Total: 51 tests — all pass** (`pytest 9.0.2`, 1.59 s).

### Linter (ruff)
Two fixable warnings found:
1. `src/allocators/base.py:30` — W292: no newline at end of file.
2. `tests/test_mckp_allocator.py:7` — I001: unsorted import block.

Neither is a logic error; both are auto-fixable with `ruff check --fix`.

---

## 7. Outputs / Example Results

`outputs/` and `data/` directories are **gitignored and do not exist** in the cloned repo. No result files are committed. There is no evidence of any completed experiment run in the repository.

To generate example results:
```bash
python3 scripts/run_simulated_allocation.py --config configs/simulated_mckp.yaml
# (network-free — no dataset download needed)
```

---

## 8. Gaps and Mismatches Between Docs and Code

### Things docs claim that ARE implemented
- ✅ GSM8K loader (auto-download via HuggingFace)
- ✅ Dummy model with `correct_prob` and `seed`
- ✅ Three native baselines: greedy, best-of-N, self-consistency
- ✅ Equal-budget allocator (including remainder distribution)
- ✅ MCKP allocator (exact DP) — mentioned in PROJECT_CONTEXT and BASELINE_TRACKER as `📋 Planned`, but the implementation **already exists** as `src/allocators/mckp_allocator.py`
- ✅ Exact-match evaluation with per-query JSON logging
- ✅ Config-driven experiment runner
- ✅ Wrapper stubs for TALE and BEST-Route

### Things docs discuss that are NOT yet implemented
- ❌ Vanilla CoT baseline (needs real LLM backend)
- ❌ Real LLM backend (OpenAI API, vLLM, etc.)
- ❌ TALE integration (official repo not cloned)
- ❌ BEST-Route integration (official repo not cloned)
- ❌ Snell et al. baseline
- ❌ Rewarding Progress / PRM baseline
- ❌ Training-Free Difficulty Proxies allocator
- ❌ SelfBudgeter baseline
- ❌ DEER baseline
- ❌ MATH / MATH500 dataset support
- ❌ Difficulty-proxy allocator
- ❌ Online allocation strategy

### Specific mismatches / ambiguities

1. **BASELINE_TRACKER marks MCKP as `📋 Planned`** but the implementation (`src/allocators/mckp_allocator.py`) and its tests (`tests/test_mckp_allocator.py`, 20 tests) are already present and fully passing. The tracker should be updated to `✅`.

2. **`run_experiment.py` hardcodes `EqualAllocator`** for non-greedy baselines. The MCKP allocator exists but is not wired into this script; it is only accessible via `run_simulated_allocation.py`.

3. **`n_samples` vs allocator interaction** in `run_experiment.py` is confusing: the script uses `max(allocated_n, n_samples)`, meaning the config's `n_samples` can silently override the allocator's decision without a warning.

4. **README lists MCKP only as a future item** in the architecture description ("Allocators — budget allocation strategies (currently: equal allocation)") even though MCKP is implemented.

5. **`docs/PAPER_NOTES.md`, `docs/REPRODUCTION_PLAN.md`, `docs/OPEN_QUESTIONS.md`** do not exist, though they may be expected by the project workflow.

6. **No outputs committed** — there is no sanity check that the runner produces correct numbers on a fixed seed. Committing one small example run would help verify reproducibility.

---

## 9. Overall Readiness Assessment

| Dimension | Status | Notes |
|-----------|--------|-------|
| **Documentation** | ⚠️ Partially ready | README and two core docs are solid; three expected docs missing |
| **Prototype** | ✅ Ready | End-to-end pipeline runs with dummy model |
| **Simulation** | ✅ Ready | Synthetic TTC + MCKP/equal allocator pipeline is complete |
| **Real LLM experiment** | ❌ Not ready | No real model backend; GSM8K loader ready but untested with real API |
| **Baseline comparison** | ⚠️ Partial | 3 native baselines work; MCKP works in simulation; all others absent or stub |
| **Paper-ready** | ❌ Not yet | No real LLM results; major baselines (TALE, BEST-Route, Snell et al.) not integrated |

---

## 10. Recommended Next Technical Priorities

1. **Update BASELINE_TRACKER** — Mark MCKP allocator as `✅ Done` (row 12). Also update README and PROJECT_CONTEXT to reflect that MCKP is implemented. (Low effort, high accuracy.)

2. **Fix the two linter warnings** — Run `ruff check --fix src/ tests/ scripts/` to resolve the W292 and I001 issues in `base.py` and `test_mckp_allocator.py`. (Trivial.)

3. **Wire MCKP into `run_experiment.py`** — Add an `allocator` key to the experiment config and route to `MCKPAllocator` when specified. This allows direct comparison of equal vs MCKP allocation on real (or dummy) model outputs. The simulated pipeline already does this correctly and can serve as a template.

4. **Add a real LLM backend** — Implement an `OpenAIModel` (or equivalent) in `src/models/`. This is the single biggest unblocking step toward running real experiments on GSM8K. The abstract `Model` interface is already in place.

5. **Integrate TALE or BEST-Route** — Clone one official external repo into `external/<name>/.repo`, flesh out the wrapper in `src/baselines/external/`, and add an integration test. TALE is the most self-contained and is a good first external baseline to validate the wrapper pattern end-to-end.
