# Reproducibility Guide

This document provides exact commands to reproduce the main manuscript results.

---

## Prerequisites

```bash
# 1. Install dependencies
pip install -e ".[dev]"

# 2. (For real-model experiments only) Set OpenAI API key
cp .env.example .env
# edit .env → set OPENAI_API_KEY=sk-...
export $(grep -v '^#' .env | xargs)
```

---

## Part A — Offline Verification (No API Key Required)

The following commands run fully offline using the dummy model or committed
routing-dataset CSVs.  They verify the pipeline end-to-end without any API
cost.

### A1. Run unit tests

```bash
pytest
# Expected: 612 passed, 5 skipped, ~10 s
```

### A2. Baseline pipeline sanity check (dummy model)

```bash
python3 scripts/run_experiment.py --config configs/greedy.yaml
python3 scripts/run_experiment.py --config configs/best_of_n.yaml
python3 scripts/run_experiment.py --config configs/self_consistency.yaml
python3 scripts/run_experiment.py --config configs/equal_allocator.yaml
```

### A3. Offline adaptive policy evaluation (uses committed routing datasets)

Evaluate policy v6/v7 on the committed GSM8K routing dataset (no API needed):

```bash
python3 scripts/run_real_policy_eval.py
# reads: data/real_gsm8k_routing_dataset_enriched.csv
# writes: outputs/real_policy_eval/ (summary.json, policy_comparison.csv)
```

### A4. Offline routing model evaluation (requires scikit-learn)

Train and evaluate a tree ensemble on the committed GSM8K routing dataset:

```bash
python3 scripts/run_real_routing_model_eval.py
# reads: data/real_gsm8k_routing_dataset_enriched.csv
# writes: outputs/real_routing_model/ (model_metrics.csv, feature_importance.csv)
```

### A5. Oracle routing upper-bound analysis

```bash
python3 scripts/run_oracle_strategy_eval.py \
    --config configs/oracle_strategy_eval_gsm8k.yaml
# writes: outputs/oracle_routing_eval/
```

### A6. Budget sweep (committed data → curve CSVs)

```bash
python3 scripts/run_real_budget_sweep.py \
    --config configs/real_budget_sweep_gsm8k.yaml
# writes: outputs/real_budget_sweep/
```

### A7. Cross-regime summary (uses committed enriched CSVs)

```bash
python3 scripts/run_final_cross_regime_summary.py
# reads: data/real_*_routing_dataset_enriched.csv
# writes: outputs/cross_regime_comparison/final_cross_regime_summary.csv
#         outputs/paper_tables/cross_regime/final_cross_regime_summary.csv
```

### A8. Simulated allocation sweep (synthetic, fully offline)

```bash
python3 scripts/run_simulated_sweep.py --config configs/simulated_sweep.yaml
python3 scripts/summarize_simulated_results.py
# writes: outputs/simulated_sweep/
```

---

## Part B — Real-Model Experiments (Requires OPENAI_API_KEY)

> **Warning:** These commands make paid API calls to OpenAI.  Estimated cost
> for the full pipeline is on the order of a few US dollars (GPT-4o-mini rates
> as of 2025).  Costs may change; monitor your usage.

### B1. Build the GSM8K routing dataset (100 queries)

```bash
python3 scripts/run_build_real_routing_dataset.py \
    --paired-outcomes --subset-size 100 \
    --output-dataset-csv data/real_gsm8k_routing_dataset.csv
# writes: data/real_gsm8k_routing_dataset.csv
#         outputs/real_routing_dataset/
```

### B2. Build the MATH500 routing dataset (100 queries)

```bash
python3 scripts/run_build_math500_routing_dataset.py --subset-size 100
# writes: data/real_math500_routing_dataset.csv
#         outputs/real_math500_routing/
```

### B3. Build the Hard-GSM8K routing dataset

```bash
python3 scripts/run_select_hard_gsm8k.py --subset-size 100
python3 scripts/run_build_hard_gsm8k_routing_dataset.py
# writes: data/real_hard_gsm8k_routing_dataset.csv
#         outputs/real_hard_gsm8k_routing/
```

### B4. Enrich datasets with routing features

After running B1–B3, enrich each dataset with query-level features:

```bash
python3 scripts/run_real_policy_eval.py
# also enriches the CSVs to *_enriched.csv variants
```

### B5. Evaluate adaptive routing policies

```bash
python3 scripts/run_real_policy_eval.py
# GSM8K → outputs/real_policy_eval/
```

### B6. Train and evaluate the learned routing model

```bash
python3 scripts/run_real_routing_model_eval.py
# GSM8K → outputs/real_routing_model/
# MATH500 → outputs/real_math500_routing_model/
# Hard-GSM8K → outputs/real_hard_gsm8k_routing_model/
```

### B7. Export paper tables

```bash
python3 scripts/run_paper_table_export.py   # if available
# or run individual export scripts under scripts/
# writes: outputs/paper_tables/
```

---

## Committed Artifacts vs Regenerated

| Artifact | Committed? | Command to regenerate |
|----------|------------|-----------------------|
| `data/real_*_routing_dataset*.csv` | ✅ Yes | B1–B4 above |
| `outputs/paper_tables/cross_regime/final_cross_regime_summary.csv` | ✅ Yes | A7 above |
| `outputs/paper_tables/baselines/` | ✅ Yes | see `scripts/run_strong_baselines.py` |
| `outputs/paper_tables/oracle_routing/` | ✅ Yes | A5 above |
| `outputs/paper_tables/next_stage/` | ✅ Yes | see `scripts/run_next_stage_postprocess.py` |
| `outputs/budget_sweep/` | ✅ Yes | A6 above |
| `outputs/baselines/` | ✅ Yes | see `scripts/run_strong_baselines.py` |
| Simulated-sweep tables | ❌ BLOCKED | A8 above |
| Oracle-subset table | ❌ BLOCKED | A5 above |

See `outputs/paper_tables/export_manifest.json` for the full blocker list.

---

## Environment Details

- Python 3.12 (also compatible with 3.10+)
- Model used in all real experiments: `gpt-4o-mini` via OpenAI API
- scikit-learn ≥ 1.3 required for learned routing model (Part A4, B6)
- No GPU required; all experiments are CPU-only

---

## Linting and Code Style

```bash
ruff check src/ tests/ scripts/
```
