# Canonical Repository Guide

**For:** New readers, reviewers, and manuscript authors  
**Scope:** Adaptive LLM Inference — Knowledge-Based Systems submission  
**Date:** 2026-04-01

---

## Where to Start

| You want to… | Start here |
|--------------|-----------|
| Understand the paper and reproduce the main results | [`MANUSCRIPT_REPRODUCTION.md`](../MANUSCRIPT_REPRODUCTION.md) |
| Find the final manuscript tables and figures | [`FINAL_MANUSCRIPT_QUICKSTART.md`](../FINAL_MANUSCRIPT_QUICKSTART.md) |
| Get a single index of all canonical assets | [`FINAL_MANUSCRIPT_ASSET_INDEX.md`](../FINAL_MANUSCRIPT_ASSET_INDEX.md) |
| Run the full test suite | `pytest` (no API key needed) |
| Regenerate canonical tables and figures offline | `python3 scripts/generate_final_manuscript_artifacts.py` |
| Understand the research framing (MCKP, routing) | [`docs/PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md) |
| Understand the policy versioning story | [`docs/CANONICAL_MANUSCRIPT_DECISIONS.md`](CANONICAL_MANUSCRIPT_DECISIONS.md) |
| Understand what each result means | [`docs/STATE_OF_EVIDENCE.md`](STATE_OF_EVIDENCE.md) |

---

## Source of Truth — Canonical Files

### Manuscript-facing data

| File | Role |
|------|------|
| `data/real_gsm8k_routing_dataset_enriched.csv` | GSM8K routing regime (100 queries) |
| `data/real_hard_gsm8k_routing_dataset_enriched.csv` | Hard-GSM8K routing regime (100 queries) |
| `data/real_hard_gsm8k_b2_routing_dataset_enriched.csv` | Hard-GSM8K batch-2 routing regime (100 queries) |
| `data/real_math500_routing_dataset_enriched.csv` | MATH500 routing regime (100 queries) |

All four datasets were generated with `gpt-4o-mini` via the OpenAI API and are
committed for offline reproducibility.

### Canonical manuscript tables (final)

Located in `outputs/paper_tables_final/`:

| File | Manuscript placement |
|------|----------------------|
| `main_results_summary.csv` | Main results table |
| `cross_regime_summary.csv` | Cross-regime comparison |
| `policy_comparison_main.csv` | Per-policy comparison |
| `oracle_headroom_main.csv` | Oracle headroom |
| `routing_outcome_breakdown_main.csv` | Routing outcome decomposition |
| `budget_curve_main_points.csv` | Budget-curve key points |
| `baseline_comparison_appendix.csv` | Appendix: baseline comparison |
| `statistical_support_main.csv` | Appendix: bootstrap CIs |

> **Regenerate:** `python3 scripts/generate_final_manuscript_artifacts.py`

### Canonical manuscript figures (final)

Located in `outputs/paper_figures_final/`:

| File | Manuscript placement |
|------|----------------------|
| `cross_regime_accuracy_cost.png` | Main figure: accuracy vs cost |
| `routing_headroom_barplot.png` | Main figure: headroom |
| `routing_outcome_stacked_bar.png` | Main figure: outcome decomposition |
| `oracle_gap_barplot.png` | Main figure: oracle gap |
| `budget_curve_main.png` | Main figure: budget curves |
| `adaptive_efficiency_scatter.png` | Main figure: efficiency scatter |
| `graphic_abstract.png` / `.pdf` | Graphic abstract |
| `threshold_tradeoff_curve.png` | Appendix: threshold tradeoff |

> **Regenerate:** `python3 scripts/generate_final_manuscript_artifacts.py`

### Key source code (main paper)

| Module | Role |
|--------|------|
| `src/policies/adaptive_policy_v5.py` | Primary adaptive policy (headline) |
| `src/policies/adaptive_policy_v6.py` | Adaptive policy (main comparison) |
| `src/policies/adaptive_policy_v7.py` | Adaptive policy (hard regimes) |
| `src/allocators/` | Equal and MCKP budget allocation |
| `src/evaluation/` | Per-query logging, metrics, oracle eval |
| `src/features/` | Query-level routing features |
| `src/datasets/` | Dataset loaders (GSM8K, MATH500, AIME, GPQA) |
| `src/baselines/` | Greedy, best-of-N, self-consistency baselines |
| `src/paper_artifacts/` | Table/figure export utilities |

### Key docs (canonical)

| File | Role |
|------|------|
| [`docs/CANONICAL_MANUSCRIPT_DECISIONS.md`](CANONICAL_MANUSCRIPT_DECISIONS.md) | Policy versions, regimes, main-vs-supplementary decisions |
| [`docs/FINAL_CONSISTENCY_AUDIT.md`](FINAL_CONSISTENCY_AUDIT.md) | Consistency issues found and resolved |
| [`docs/FINAL_FIGURE_PROVENANCE.md`](FINAL_FIGURE_PROVENANCE.md) | Input provenance for every final figure |
| [`docs/FINAL_COHERENCE_CHECK.md`](FINAL_COHERENCE_CHECK.md) | Lightweight coherence verification |
| [`docs/FINAL_REPO_STATUS.md`](FINAL_REPO_STATUS.md) | Submission-readiness and safe-claim boundaries |
| [`docs/STATE_OF_EVIDENCE.md`](STATE_OF_EVIDENCE.md) | Honest evidence-level audit with caveats |
| [`docs/BASELINE_TRACKER.md`](BASELINE_TRACKER.md) | Status of every comparison baseline |
| [`docs/MANUSCRIPT_WRITING_AID.md`](MANUSCRIPT_WRITING_AID.md) | Section-by-section writing guide |

---

## What Supports the Main Paper

The main paper is grounded in:

1. **Four evaluation regimes** (each n=100, `gpt-4o-mini`):
   - `gsm8k_random_100`, `hard_gsm8k_100`, `hard_gsm8k_b2`, `math500_100`

2. **Binary routing policies** `adaptive_policy_v5` (primary), `v6`, `v7`
   compared against `reasoning_greedy`, `direct_plus_revise`, and oracle.

3. **Budget-aware analysis** at the canonical cost points: 1.0, 1.1, 1.2, 2.0.

4. **All supporting data and results are committed** to `data/` and `outputs/`;
   no API key is needed for policy evaluation or table/figure regeneration.

---

## What Is Supplementary Only

| Content | Location | Notes |
|---------|----------|-------|
| AIME-2024 exploratory eval | `outputs/small_pass/` | 30-query offline eval; not in main tables |
| GPQA-Diamond data | `data/gpqa_diamond_normalized.jsonl` | No policy eval committed |
| Confidence-threshold baseline | `outputs/baselines/confidence_threshold/` | Supplementary baseline |
| Learned router baseline | `outputs/baselines/learned_router/` | Supplementary baseline |
| Bootstrap uncertainty analysis | Committed in `outputs/paper_tables_final/statistical_support_main.csv` | Appendix |
| Clarification export | `outputs/paper_export/` | Supplementary explanation table |
| Nice-to-have results | `outputs/nice_to_have/` | Supplementary |

---

## What Is Legacy / Exploratory / Internal

| Content | Location | Notes |
|---------|----------|-------|
| Adaptive policy v1–v4 | `src/policies/adaptive_policy_v[1-4].py` | Superseded by v5–v7 |
| Multi-action routing (>2 actions) | `src/evaluation/multi_action_routing.py` | Exploratory; not in paper |
| TALE / BEST-Route wrappers | `src/baselines/external/`, `external/` | Stubs; no official code included |
| Simulated allocation sweep | `scripts/run_simulated_sweep.py` | BLOCKED; not cited |
| Oracle subset eval | `scripts/run_oracle_subset_eval.py` | BLOCKED; not cited |
| Historical intermediate tables | `outputs/paper_tables_cleaned/`, `outputs/paper_tables/` | Superseded by `paper_tables_final/` |
| Historical intermediate figures | `outputs/paper_figures_cleaned/`, `outputs/paper_figures/` | Superseded by `paper_figures_final/` |
| Exploratory docs | ~70 `docs/*.md` files | Working notes; see `docs/README.md` for tier list |
| AI-agent planning logs | `docs/internal/` | Not for external readers |
| Archived superseded root docs | `docs/archive/` | Outdated; retained for provenance |

---

## Key Constraints and Limitations

- **Single model scope:** All results are for `gpt-4o-mini` (OpenAI API).
  Generalisation to other models has not been verified.
- **Small n per regime:** Each regime has exactly 100 queries. Bootstrap CIs
  are reported in `outputs/paper_tables_final/statistical_support_main.csv`.
- **Policy gains are modest on easy regimes:** Routing improvements over
  `reasoning_greedy` are 1–3 pp on GSM8K and MATH500; gains are more substantial
  on hard regimes. Do not overclaim.
- **Blocked artifacts exist:** Simulated allocation sweep and oracle subset
  tables are not committed and are not cited in the manuscript.
- **API required for raw data regeneration:** All four routing datasets require
  an OpenAI API key to regenerate from scratch; the committed CSVs bypass this.

---

## Reproduction Summary

```bash
# 1. Install
pip install -e ".[dev]"

# 2. Verify (offline, no API key)
pytest                                   # 677 tests, 0 failures expected

# 3. Regenerate canonical tables and figures (offline)
python3 scripts/generate_final_manuscript_artifacts.py
# → outputs/paper_tables_final/
# → outputs/paper_figures_final/

# 4. Full offline policy eval (uses committed CSVs)
python3 scripts/run_real_policy_eval.py

# 5. Rebuild routing datasets (requires OPENAI_API_KEY)
# See REPRODUCIBILITY.md Part B for commands
```

See [`MANUSCRIPT_REPRODUCTION.md`](../MANUSCRIPT_REPRODUCTION.md) for the
complete step-by-step reproduction guide.
