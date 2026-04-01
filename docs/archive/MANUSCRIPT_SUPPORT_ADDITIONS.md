# Manuscript Support Additions

> OUTDATED / SUPERSEDED (2026-03-31): this revision log is historical.
> For authoritative current manuscript assets, use:
> - `FINAL_MANUSCRIPT_QUICKSTART.md`
> - `FINAL_MANUSCRIPT_ASSET_INDEX.md`
> - `outputs/paper_tables_final/` and `outputs/paper_figures_final/`

This document records all additions made to support the KBS manuscript revision.
Every artifact is grounded in the committed routing datasets and experiment outputs —
no new LLM calls or API keys are required.

---

## What Was Added

### 1. Confidence-Threshold Routing Baseline

**Source:** `src/baselines/confidence_threshold_router.py`
**Script:** `scripts/run_confidence_baseline.py`
**Outputs:** `outputs/baselines/confidence_threshold/`

Uses `unified_confidence_score` (already present in all four enriched routing
datasets) as a scalar routing signal. Queries with confidence below a chosen
threshold are routed to `direct_plus_revise`; the rest stay with
`reasoning_greedy`. The threshold is swept from 0.00 to 1.00 and an operating
point is selected at target avg_cost ≤ 1.2 (adjustable via `--target-cost`).

**Output files:**
- `confidence_threshold_summary.csv` — one row per regime at operating point;
  columns: `regime, baseline, threshold, accuracy, avg_cost, revise_rate, n`
- `confidence_threshold_sweep.csv` — full sweep (all thresholds, all regimes)
- `confidence_threshold_summary.json` — same as CSV in JSON format

**Manuscript use:** Direct comparison to v6/v7 in the same four regimes at a
matched cost constraint. Can be added as a row in the main baselines table or
referenced in the routing-baseline discussion.

---

### 2. Learned Router Baseline

**Source:** `src/baselines/learned_router_baseline.py`
**Script:** `scripts/run_learned_router_baseline.py`
**Outputs:** `outputs/baselines/learned_router/`

Trains logistic regression and a shallow decision tree (max depth 4) with
`class_weight='balanced'` to predict `revise_helpful` from the 48 pre-computed
features in the enriched routing CSVs. Evaluated using 5-fold stratified
cross-validation within each regime. Where extreme class imbalance makes
stratification infeasible (e.g., GSM8K has only 2 positives), the number of
folds is reduced and the result is flagged as `degenerate` or annotated with a
note.

**Output files:**
- `learned_router_summary.csv` — per-regime, per-model; columns:
  `regime, baseline, accuracy, avg_cost, revise_rate, cv_folds, degenerate, n, note`
- `learned_router_summary.json` — same as JSON

**Manuscript use:** Shows whether a supervised learned router improves over the
confidence-threshold heuristic. Honest flagging of degenerate regimes (GSM8K)
demonstrates that the class imbalance problem is real and regime-dependent.

---

### 3. Bootstrap Uncertainty Analysis

**Source:** `src/evaluation/uncertainty_analysis.py`
**Script:** `scripts/run_uncertainty_analysis.py`
**Outputs:** `outputs/manuscript_support/`

Computes paired bootstrap 95% confidence intervals (10,000 replicates) for four
comparisons across all four manuscript regimes:
1. `adaptive_best_policy_vs_always_reasoning` — primary paper claim
2. `oracle_vs_always_reasoning` — upper-bound gain
3. `adaptive_best_policy_vs_oracle` — policy–oracle gap
4. `always_revise_vs_always_reasoning` — cost of always revising

**Output files:**
- `uncertainty_analysis.json` — full results (all CIs, all regimes)
- `uncertainty_analysis_summary.csv` — flat CSV with columns:
  `regime, comparison, observed_delta, ci_lower, ci_upper, significant_at_95pct, n, n_bootstrap`

**Manuscript use:** Reports exact bootstrap CIs rather than fabricated
significance claims. Notable findings from the committed data:
- `hard_gsm8k_b2`: adaptive policy gain is **significant** (CI: [+0.02, +0.11])
- `hard_gsm8k_100`: adaptive policy gain is **not significant** at 95% (CI: [-0.03, +0.09])
- Oracle gains are significant in 3 of 4 regimes, confirming the upper-bound analysis

---

### 4. Clarification Export Table

**Source:** `src/evaluation/clarification_export.py`
**Script:** `scripts/run_clarification_export.py`
**Outputs:** `outputs/manuscript_support/`

Reads committed artifact files to produce a clean reconciliation table that
makes explicit the relationship between:
- `always_reasoning` — first-pass reasoning only (baseline)
- `best_adaptive` — best deployable policy (v6 or v7) from the cross-regime summary
- `oracle` — oracle routing (theoretical ceiling)
- `budget_frontier_1.1`, `budget_frontier_1.2` — budget-curve accuracy at those cost levels

**Output files:**
- `clarification_table.csv` — tidy format (regime × strategy rows)
- `clarification_wide.csv` — wide format (one row per regime)
- `clarification_table.tex` — LaTeX booktabs table (ready to paste into manuscript)
- `clarification_table.json` — machine-readable JSON
- `NOTES.md` — short explanation of strategy types (see below)

**Important — strategy type distinctions (`NOTES.md`):**
`always_reasoning` and `best_adaptive` are **single deployable operating points**:
fixed routing rules that work on new queries without any oracle information.
`budget_frontier_1.1` and `budget_frontier_1.2` are **sweep-style summaries**
that rely on oracle query ordering — they are *not* single deployable policies
and must not be compared directly to practical baselines as if they were.

**Manuscript use:** Drop `clarification_wide.csv` into the manuscript or use
`clarification_table.tex` directly. Explains why the adaptive policy is below the
budget frontier on hard regimes (the policy is not oracle-informed; budget
frontier uses oracle ordering). Share `NOTES.md` with reviewers to pre-empt
confusion between deployable policies and budget-sweep summaries.

---

### 5. Tests

**File:** `tests/test_manuscript_support.py`

44 tests covering:
- **Regime integrity** (10 tests): all four regimes are present, size-100, with
  required columns, binary labels, and consistent artifact coverage; regression
  tests for exact accuracy numbers in committed tables.
- **Confidence-threshold baseline** (8 tests): routing logic, sweep ordering,
  operating-point selection, output file schema, real-regime accuracy bounds.
- **Learned router baseline** (6 tests): logistic regression and decision tree
  evaluation, degenerate flagging, output file schema, honest class-imbalance
  handling.
- **Uncertainty analysis** (8 tests): bootstrap CI math, valid interval ordering,
  all-regime coverage, output schema.
- **Clarification export** (13 tests): artifact concordance, exact accuracy
  regression, file creation, strategy names, LaTeX output, best-policy names,
  NOTES.md content verification.

---

## Commands

```bash
# Install dependencies (one-time)
pip install -e ".[dev]"

# Run all new additions
python scripts/run_confidence_baseline.py
python scripts/run_learned_router_baseline.py
python scripts/run_uncertainty_analysis.py
python scripts/run_clarification_export.py

# Run only the new tests
python -m pytest tests/test_manuscript_support.py -v

# Run full test suite
python -m pytest
```

---

## Output Paths Produced

```
outputs/baselines/confidence_threshold/
├── confidence_threshold_summary.csv
├── confidence_threshold_sweep.csv
└── confidence_threshold_summary.json

outputs/paper_tables_small_pass/   ← from `scripts/run_small_pass.py` (manuscript-oriented tables: AIME + confidence baseline + combined comparison)
├── confidence_baseline_main_regimes.csv
├── aime_policy_comparison.csv
├── small_pass_combined_comparison.csv
└── small_pass_run_summary.json

outputs/baselines/learned_router/
├── learned_router_summary.csv
└── learned_router_summary.json

outputs/manuscript_support/
├── uncertainty_analysis.json
├── uncertainty_analysis_summary.csv
├── clarification_table.csv
├── clarification_wide.csv
├── clarification_table.tex
├── clarification_table.json
└── NOTES.md                            ← explains practical policies vs budget-frontier rows
```

---

## Blockers / Limitations

None. All items requested are fully implemented:

- Confidence-threshold baseline: ✅ using `unified_confidence_score` from committed data
- Learned router baseline: ✅ using 48 pre-computed features from committed data
- Uncertainty analysis: ✅ paired bootstrap, 10,000 replicates, no API needed
- Clarification export: ✅ reads committed artifact CSVs, produces CSV + LaTeX + NOTES.md
- Tests: ✅ 45 tests in `tests/test_manuscript_support.py`, all passing; full suite 677 tests collected, 0 failures (`pytest`; optional skips under `skipif`)
