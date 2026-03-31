# Adaptive LLM Inference

**Adaptive test-time compute allocation for LLM reasoning under budget constraints.**

Submitted to *Knowledge-Based Systems*.

> **Reading the paper?**  
> Start with [`MANUSCRIPT_REPRODUCTION.md`](MANUSCRIPT_REPRODUCTION.md) for the
> paper scope, exact regimes, tables/figures, and reproduction commands.

---

## Research Question

> Given a batch of reasoning queries and a fixed inference budget, can a
> lightweight routing policy decide *per query* whether to apply cheap
> single-pass reasoning or costlier self-revision — and at what cost efficiency?

We study binary cheap-vs-revise routing policies evaluated on four 100-query
regimes (GSM8K, Hard-GSM8K ×2, MATH500) using `gpt-4o-mini`.  
The offline allocation problem maps naturally to a **Multiple-Choice Knapsack
Problem (MCKP)**; our contribution is in the AI-specific modelling — lightweight
difficulty features, calibrated routing signals, and policy comparison against
fixed baselines and an oracle ceiling.

### Core paper contribution

- **Binary routing policies** (adaptive\_policy v5–v7): route each query to
  cheap reasoning *or* costlier revision based on query-level difficulty signals.
- **Four evaluation regimes**: GSM8K-random-100, Hard-GSM8K-100, Hard-GSM8K-B2,
  MATH500-100 (all with committed routing datasets).
- **Policy vs oracle vs fixed-cost baselines**: quantifies the accuracy/cost
  trade-off and the gap to the oracle ceiling.
- **Budget-aware analysis**: accuracy-vs-cost curves and budget sweep tables.

All supporting data and pre-computed results are committed in this repository.
Most analysis steps run **offline without any API key**.

---

## Repository Structure

```
├── src/
│   ├── datasets/          # Dataset loaders (GSM8K, MATH500, AIME-2024, GPQA)
│   ├── models/            # Model interface: dummy (offline) + OpenAI-backed
│   ├── baselines/         # Greedy, best-of-N, self-consistency; TALE/BEST-Route wrappers
│   │   └── external/      # Thin wrappers for official-code baselines
│   ├── allocators/        # Budget-allocation strategies (equal, MCKP)
│   ├── evaluation/        # Metrics, per-query logging, strategy evaluators
│   ├── features/          # Query-level difficulty / routing-signal features
│   ├── analysis/          # Post-hoc analysis (feature-gap, revise-helps)
│   ├── policies/          # Routing policies (adaptive_policy v1–v7, router baseline)
│   ├── strategies/        # Action/strategy catalogue
│   ├── paper_artifacts/   # Table / figure export utilities
│   └── utils/             # Config loading, answer extraction
├── configs/               # YAML experiment configurations
├── scripts/               # CLI experiment runner scripts
├── tests/                 # Unit tests (677 collected; pytest: 0 failures; skips may apply)
├── docs/                  # Research documentation (see below)
├── data/                  # Committed routing datasets
├── outputs/               # Committed experiment results
└── external/              # README stubs for TALE / BEST-Route author repos
```

### Key documentation

| File | Purpose |
|------|---------|
| [`MANUSCRIPT_REPRODUCTION.md`](MANUSCRIPT_REPRODUCTION.md) | **Start here** — paper scope, regimes, tables/figures, reproduction commands |
| [`PAPER_ARTIFACT_STATUS.md`](PAPER_ARTIFACT_STATUS.md) | Status of every artifact: main-paper, appendix, exploratory, or incomplete |
| [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) | Full command reference for all manuscript experiments |
| [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) | Dataset provenance and access instructions |
| [`MANUSCRIPT_ARTIFACTS.md`](MANUSCRIPT_ARTIFACTS.md) | Inventory of committed tables and figures |
| [`docs/PROJECT_CONTEXT.md`](docs/PROJECT_CONTEXT.md) | Research goal, MCKP framing, and baseline families |
| [`docs/BASELINE_TRACKER.md`](docs/BASELINE_TRACKER.md) | Status of every comparison baseline |

---

## Installation

Requires **Python ≥ 3.10**.

```bash
pip install -e ".[dev]"
```

Core dependencies: `datasets`, `pyyaml`, `numpy`, `requests`.  
Dev dependencies: `matplotlib`, `scikit-learn`, `pytest`, `ruff`.

---

## Quick Start — Offline (No API Key Required)

These commands use the **dummy model** and run fully offline:

```bash
# Greedy baseline — 1 sample per query, 50 GSM8K test queries
python3 scripts/run_experiment.py --config configs/greedy.yaml

# Best-of-N — 5 samples per query
python3 scripts/run_experiment.py --config configs/best_of_n.yaml

# Self-consistency — majority vote, 5 samples
python3 scripts/run_experiment.py --config configs/self_consistency.yaml

# Equal allocation under budget constraint
python3 scripts/run_experiment.py --config configs/equal_allocator.yaml
```

Results are written as JSON to `outputs/`.

Run the full test suite (offline, tens of seconds on a typical machine):

```bash
pytest
```

---

## Real-Model Experiments (Requires OpenAI API Key)

Copy `.env.example` to `.env` and set `OPENAI_API_KEY` (the file is gitignored).  
After `pip install -e .`, the project loads `.env` automatically in key paths (see `docs/CURSOR_TOKEN_SETUP.md`).

**Interactive shell:** load variables and run commands:

```bash
cp .env.example .env
# fill in OPENAI_API_KEY in .env
export $(grep -v '^#' .env | xargs)
```

**Cursor / automation:** the agent does not inherit your manual `export` in another terminal. Put the key in `.env`, then run tools via the helper so the key is loaded from the repo:

```bash
bash scripts/with_dotenv.sh python3 scripts/test_openai_key.py
bash scripts/with_dotenv.sh python scripts/run_build_real_routing_dataset.py --paired-outcomes --dataset gpqa_diamond --subset-size 5
```

See [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md) for the exact commands that
reproduce the main manuscript results.

---

## Data

### Public benchmark datasets

| Dataset | Source | How to obtain |
|---------|--------|---------------|
| GSM8K | HuggingFace `openai/gsm8k` | Auto-downloaded on first run |
| MATH500 | HuggingFace `lighteval/MATH-Hard` | Auto-downloaded on first run |
| AIME-2024 | Subset in `data/` | Committed (30 problems) |
| GPQA-Diamond | Normalised file in `data/` | Committed (`data/gpqa_diamond_normalized.jsonl`); enriched routing CSV via `docs/GPQA_EVALUATION_STATUS.md` |

### Committed routing datasets

The following derived routing datasets are committed to `data/` and used in all
manuscript experiments.  They were generated by querying `gpt-4o-mini` via the
scripts in `src/data/`; see [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md)
for construction details.

| File | Queries | Regime | Paper |
|------|---------|--------|-------|
| `data/real_gsm8k_routing_dataset_enriched.csv` | 100 | GSM8K random | ✅ Main |
| `data/real_hard_gsm8k_routing_dataset_enriched.csv` | 100 | Hard GSM8K | ✅ Main |
| `data/real_hard_gsm8k_b2_routing_dataset_enriched.csv` | 100 | Hard GSM8K (batch 2) | ✅ Main |
| `data/real_math500_routing_dataset_enriched.csv` | 100 | MATH500 | ✅ Main |
| `data/real_aime2024_routing_dataset.csv` | 30 | AIME-2024 | 🔬 Exploratory (supplementary eval in `outputs/small_pass/`; not main paper) |

See [`DATA_AVAILABILITY.md`](DATA_AVAILABILITY.md) for full provenance.

---

## Committed Results

Key manuscript-supporting results are committed under `outputs/`.  
See [`MANUSCRIPT_ARTIFACTS.md`](MANUSCRIPT_ARTIFACTS.md) and
[`PAPER_ARTIFACT_STATUS.md`](PAPER_ARTIFACT_STATUS.md) for full inventories.

| Path | Contents | Status |
|------|----------|--------|
| `outputs/paper_tables_cleaned/` | Publication-ready table CSVs | ✅ Main paper |
| `outputs/paper_figures_cleaned/` | Publication-ready figures | ✅ Main paper |
| `outputs/paper_tables/cross_regime/final_cross_regime_summary.csv` | Cross-regime routing table | ✅ Main paper |
| `outputs/paper_tables/baselines/` | Baseline strategy comparisons | ✅ Main paper |
| `outputs/paper_tables/oracle_routing/` | Oracle routing upper-bound analysis | ✅ Main paper |
| `outputs/budget_sweep/` | Budget-vs-accuracy sweep CSVs | ✅ Main paper |
| `outputs/real_*_policy_eval/` | Per-regime policy evaluation | ✅ Main paper |
| `outputs/small_pass/` | Exploratory AIME-2024 policy eval + confidence-threshold sweep (offline) | 🔬 Supplementary |
| `outputs/paper_tables_small_pass/` | Tables for AIME + confidence baseline (`scripts/run_small_pass.py`) | 🔬 Supplementary |
| `outputs/baselines/confidence_threshold/` | Confidence-threshold router baseline (`scripts/run_confidence_baseline.py`) | 🔬 Supplementary |

Large raw API response files (`raw_responses.jsonl`) are committed for full
reproducibility traceability but are not intended to be cited directly.

---

## Limitations and Scope

- **Real-model experiments** require a valid `OPENAI_API_KEY` (GPT-4o-mini used
  throughout).  All dummy-model tests and offline policy evaluations run without
  any API key.
- **External baselines** (TALE, BEST-Route): wrapper code exists in
  `src/baselines/external/` but official author repos must be cloned manually
  into `external/<name>/.repo/`.  See `external/tale/README.md` and
  `external/best_route/README.md`.
- **Blocked artifacts**: two tables (simulated allocation sweep, oracle subset)
  noted as BLOCKED in `outputs/paper_tables/export_manifest.json` require
  separate runs; they are not cited as final results.
- **Scope**: all results are for `gpt-4o-mini` on GSM8K, MATH500, and
  Hard-GSM8K regimes.  Generalisation to other models and datasets has not been
  verified in this release.
- **Exploratory content**: policy versions v1–v4, multi-action routing, and
  TALE/BEST-Route wrappers are present but are not part of the core manuscript
  claims.  See [`PAPER_ARTIFACT_STATUS.md`](PAPER_ARTIFACT_STATUS.md) for the
  full status breakdown.

---

## Tests and Linting

```bash
pytest                                   # run all unit tests
ruff check src/ tests/ scripts/          # lint
ruff check --fix src/ tests/ scripts/    # auto-fix lint
```

---

## Config Format

```yaml
dataset:
  name: gsm8k
  split: test
  max_samples: 50

model:
  type: dummy       # or: openai
  correct_prob: 0.3
  seed: 42

baseline: greedy   # greedy | best_of_n | self_consistency
n_samples: 1
budget: 50

output: outputs/results.json
```

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Citation

If you use this code or data in your research, please cite the associated
manuscript (citation details to be added upon acceptance).
