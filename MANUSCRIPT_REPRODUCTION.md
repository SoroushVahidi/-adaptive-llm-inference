# Manuscript Reproduction Guide

> **If you are reading the Knowledge-Based Systems paper, start here.**

This file is the single authoritative reference for reproducing the results
reported in the manuscript on cost-aware adaptive routing for LLM reasoning.

---

## 1. Paper Scope

The manuscript addresses a single, narrow question:

> Given a fixed inference budget and a batch of reasoning queries, can a
> lightweight routing policy decide *per query* whether to apply cheap
> single-pass reasoning or costlier self-revision — and, if so, at what cost
> efficiency gain?

**Core contribution:** Binary cheap-vs-revise routing policies (adaptive\_policy
v5–v7) evaluated against fixed baselines and an oracle ceiling on four
100-query regimes, with budget-aware analysis.

**Model used throughout:** `gpt-4o-mini` (OpenAI API).  
**No GPU required.**

---

## 2. Manuscript Regimes

All main-paper claims are grounded in exactly these four regimes:

| Regime ID | Dataset | n | Data file |
|-----------|---------|---|-----------|
| `gsm8k_random_100` | GSM8K (random 100) | 100 | `data/real_gsm8k_routing_dataset_enriched.csv` |
| `hard_gsm8k_100` | Hard GSM8K (batch 1) | 100 | `data/real_hard_gsm8k_routing_dataset_enriched.csv` |
| `hard_gsm8k_b2` | Hard GSM8K (batch 2) | 100 | `data/real_hard_gsm8k_b2_routing_dataset_enriched.csv` |
| `math500_100` | MATH500 | 100 | `data/real_math500_routing_dataset_enriched.csv` |

> AIME-2024 (30 queries) is present in `data/` but policy evaluation was not
> run on it; it is omitted from main-paper policy tables.

---

## 3. Manuscript Tables and Figures

### Main-paper tables (cleaned, publication-ready)

| Table | Description | File |
|-------|-------------|------|
| Table 1 | Policy evaluation — accuracy, avg cost, revise rate | `outputs/paper_tables_cleaned/main_results_summary.csv` |
| Table 2 | Policy comparison long format (all routes × all regimes) | `outputs/paper_tables/real_routing/real_policy_eval_comparison_long.csv` |
| Table 3 | Cross-regime summary (4 regimes × 7 metrics) | `outputs/paper_tables_cleaned/final_cross_regime_summary_fixed.csv` |
| Table 4 | Oracle routing upper bounds | `outputs/paper_tables_cleaned/oracle_routing_eval.csv` |
| Table 5 | Budget curves — accuracy vs target cost | `outputs/paper_tables_cleaned/budget_curves_all_datasets.csv` |

### Main-paper figures (cleaned)

| Figure | Description | File |
|--------|-------------|------|
| Figure 1 | 2×2 composite: accuracy vs cost, all four regimes | `outputs/paper_figures_cleaned/real_routing/accuracy_vs_cost_2x2_composite.png` |
| Figure 1a | Accuracy vs cost: Hard GSM8K-100 (standalone panel) | `outputs/paper_figures_cleaned/real_routing/accuracy_vs_cost_hard_gsm8k_100.png` |
| Figure 1b | Accuracy vs cost: Hard GSM8K-B2 (standalone panel) | `outputs/paper_figures_cleaned/real_routing/accuracy_vs_cost_hard_gsm8k_b2.png` |
| Figure 2a | Budget curve: Hard GSM8K-B2 | `outputs/paper_figures_cleaned/next_stage/budget_curve_hard_gsm8k_b2.png` |
| Figure 2b | Budget curve: Hard GSM8K-100 | `outputs/paper_figures_cleaned/next_stage/budget_curve_hard_gsm8k_100.png` |

### Appendix-only tables

| Table | Description | File | Note |
|-------|-------------|------|------|
| Table A1 | Baseline strategy rollup (n=15–30) | `outputs/paper_tables_cleaned/baselines_appendix.csv` | Different sample size; not directly comparable to main tables |
| Table A2 | 3-regime cross-regime (without B2) | `outputs/paper_tables_cleaned/cross_regime_summary.csv` | Superseded by Table 3 |

---

## 4. Committed Supporting Outputs

All manuscript-supporting outputs are committed to `outputs/`. The key paths:

```
outputs/
├── paper_tables_cleaned/      ← publication-ready table CSVs (use these)
├── paper_figures_cleaned/     ← publication-ready figures (use these)
├── paper_tables/              ← original export tables (source for cleaned versions)
├── paper_figures/             ← original export figures
├── real_policy_eval/          ← GSM8K policy eval JSON + CSV
├── real_hard_gsm8k_policy_eval/
├── real_hard_gsm8k_b2_policy_eval/
├── real_math500_policy_eval/
├── oracle_routing_eval/       ← oracle upper-bound analysis
├── budget_sweep/              ← budget-vs-accuracy curve CSVs
├── cross_regime_comparison/   ← final cross-regime summary
└── baselines/                 ← baseline strategy comparison JSON
```

Raw LLM response files (`raw_responses.jsonl`) are also committed for full
traceability but are not intended to be cited directly.

---

## 5. Commands to Regenerate Outputs

### Step 0 — Install

```bash
pip install -e ".[dev]"
```

### Step 1 — Offline verification (no API key)

Run the full test suite (offline, ~10 s):

```bash
pytest
# Expected: 612 passed, 5 skipped
```

Run the offline pipeline end-to-end:

```bash
python3 scripts/run_experiment.py --config configs/greedy.yaml
python3 scripts/run_experiment.py --config configs/equal_allocator.yaml
```

### Step 2 — Regenerate policy evaluation tables (offline, uses committed CSVs)

These read the committed enriched CSVs; no API key needed:

```bash
# Evaluate adaptive policies on all committed routing datasets
python3 scripts/run_real_policy_eval.py
# outputs → outputs/real_policy_eval/

# Cross-regime summary table
python3 scripts/run_final_cross_regime_summary.py
# outputs → outputs/cross_regime_comparison/final_cross_regime_summary.csv
#           outputs/paper_tables/cross_regime/final_cross_regime_summary.csv

# Oracle routing upper-bound analysis
python3 scripts/run_oracle_strategy_eval.py \
    --config configs/oracle_strategy_eval_gsm8k.yaml
# outputs → outputs/oracle_routing_eval/

# Budget sweep curves
python3 scripts/run_real_budget_sweep.py \
    --config configs/real_budget_sweep_gsm8k.yaml
# outputs → outputs/real_budget_sweep/
```

### Step 3 — Regenerate paper tables and figures (offline)

```bash
python3 scripts/generate_paper_tables.py
python3 scripts/generate_paper_figures.py
# outputs → outputs/paper_tables/, outputs/paper_figures/
```

---

## 6. Steps Requiring API Access

The following steps make paid OpenAI API calls and **cannot be run without
`OPENAI_API_KEY`**:

| Step | Script | Estimated cost |
|------|--------|---------------|
| Build GSM8K routing dataset (100 q) | `scripts/run_build_real_routing_dataset.py` | ~$0.10–0.30 |
| Build MATH500 routing dataset (100 q) | `scripts/run_build_math500_routing_dataset.py` | ~$0.10–0.30 |
| Build Hard-GSM8K routing dataset (100 q) | `scripts/run_build_hard_gsm8k_routing_dataset.py` | ~$0.10–0.30 |

All four enriched routing CSVs are **already committed** to `data/`, so
reviewers can skip these API steps entirely and use the committed data.

To configure API access:

```bash
cp .env.example .env
# Set OPENAI_API_KEY=sk-... in .env
export $(grep -v '^#' .env | xargs)
```

---

## 7. Exploratory Content Not Part of Core Claims

The following repository content exists but is **not part of the core
manuscript claims**:

| Content | Location | Status |
|---------|----------|--------|
| Policy versions v1–v4 | `src/policies/adaptive_policy_v[1-4].py` | Exploratory; superseded by v5–v7 |
| Multi-action routing (> 2 actions) | `src/evaluation/strategy_expansion_eval.py` | Exploratory; not in main paper |
| TALE / BEST-Route baseline wrappers | `src/baselines/external/`, `external/` | Stubs only; external repos not included |
| Simulated allocation sweep | `scripts/run_simulated_sweep.py`, `configs/simulated_sweep.yaml` | BLOCKED (no committed outputs); not cited as final |
| Oracle subset evaluation (blocked) | `scripts/run_oracle_subset_eval.py` | BLOCKED; not cited as final |
| GPQA-Diamond experiments | `data/gpqa_diamond_normalized.jsonl` | Dataset present but no policy eval committed |
| AIME-2024 policy evaluation | — | Not run; omitted from main tables |
| Consistency benchmark | `data/consistency_benchmark.json` | Diagnostic only |
| Internal working notes | `docs/internal/` | AI-agent planning logs; not for external readers |

See `PAPER_ARTIFACT_STATUS.md` for a complete status table.
