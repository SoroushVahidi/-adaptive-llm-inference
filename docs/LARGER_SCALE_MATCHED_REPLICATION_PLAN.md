# Larger-Scale Matched-Regime Replication Plan

**Date:** 2026-04-01  
**Evidence basis:** Committed repository artifacts only; no new API experiments.  
**Prepared by:** Automated audit from repo analysis

---

## 1. Inferred Canonical Setup

### Canonical Main Regimes (from `FINAL_MANUSCRIPT_QUICKSTART.md` and `docs/CANONICAL_MANUSCRIPT_DECISIONS.md`)

| Regime ID | Description | Data File |
|-----------|-------------|-----------|
| `gsm8k_random_100` | 100 randomly sampled GSM8K test questions | `data/real_gsm8k_routing_dataset_enriched.csv` |
| `hard_gsm8k_100` | Top-100 hardest GSM8K by z-score hardness blend (ranks 0–99) | `data/real_hard_gsm8k_routing_dataset_enriched.csv` |
| `hard_gsm8k_b2` | Ranks 100–199 from same hard selection pool | `data/real_hard_gsm8k_b2_routing_dataset_enriched.csv` |
| `math500_100` | 100 randomly sampled MATH500 test problems | `data/real_math500_routing_dataset_enriched.csv` |

### Canonical Routes

| Route | Description | Implementation |
|-------|-------------|----------------|
| **always-cheap** (`reasoning_greedy`) | Reasoning-first greedy decode, single call | `src/baselines/greedy.py` |
| **always-revise** (`direct_plus_revise`) | Always apply revision pass (2 LLM calls) | `src/evaluation/real_policy_eval.py` |
| **adaptive policy** (`adaptive_policy_v5`) | Primary headline adaptive policy; v6/v7 also reported | `src/policies/adaptive_policy_v5.py` |
| **oracle** | Perfect routing — revise iff `revise_helpful == 1` | `src/evaluation/real_policy_eval.py` |

### Canonical Offline Evaluation Runner

```bash
python3 scripts/run_real_policy_eval.py \
  --dataset-csv data/real_<regime>_routing_dataset_enriched.csv \
  --output-dir outputs/real_<regime>_policy_eval
```

This script requires **no API access** — it reads pre-computed `reasoning_raw`, `reasoning_correct`, `revise_correct`, `revise_helpful` columns from the enriched CSV and applies policy logic offline.

### Canonical Output Locations (Main Paper)

- Final tables: `outputs/paper_tables_final/`
- Final figures: `outputs/paper_figures_final/`
- Per-regime policy eval: `outputs/real_policy_eval/`, `outputs/real_hard_gsm8k_policy_eval/`, `outputs/real_hard_gsm8k_b2_policy_eval/`, `outputs/real_math500_policy_eval/`

---

## 2. Candidate Regimes to Expand

### Target: 300 queries per regime

All four canonical regimes are currently evaluated on exactly **n=100** queries (one gpt-4o-mini call pair per question, committed to data/).

| Regime | Current n | Max Offline n | Max Achievable n | Path to 300 |
|--------|-----------|---------------|-----------------|-------------|
| `gsm8k_random_100` | 100 | 100 | 100 | Requires 200 new API calls |
| `hard_gsm8k_100` | 100 | 200 (b1+b2) | 200 | b2 batch exists with identical protocol |
| `hard_gsm8k_b2` | 100 | 200 (b1+b2) | 200 | See above |
| `math500_100` | 100 | 100 | 100 | Requires 200 new API calls |
| `gpqa_diamond` (extra) | 0 (not canonical) | 198 | 198 | Enriched CSV committed; not main-paper regime |

### Feasible Scale Expansion (Offline, No API)

**Best achievable:** Combine `hard_gsm8k_100` (b1) and `hard_gsm8k_b2` → **n=200** for the hard-GSM8K regime.

**Rationale:** Both batches use identical selection criteria:
- Method: `z_score_blend_question_side`
- Formula: `hardness = z_len + z_num + 0.5*multi_step + 0.5*equation + 0.3*percent + 0.3*fraction + 0.15*tq_count`
- Pool: 1319 GSM8K test items, seed=42
- b1 = ranks 0–99, b2 = ranks 100–199
- Zero ID overlap confirmed (200 distinct questions)

---

## 3. Expected Blockers

| Regime | Blocker | Exact Reason |
|--------|---------|-------------|
| `gsm8k_random_100` → 300 | **API key required** | `data/real_gsm8k_routing_dataset_enriched.csv` contains exactly 100 rows; building 200 more requires `OPENAI_API_KEY` and running `scripts/run_build_real_routing_dataset.py --subset-size 300` |
| `math500_100` → 300 | **API key required** | `data/real_math500_routing_dataset_enriched.csv` contains exactly 100 rows; expanding requires `OPENAI_API_KEY` and running `scripts/run_build_math500_routing_dataset.py --subset-size 300` |
| `hard_gsm8k` → 300 | **API key required** (partial) | b1+b2 gives 200 offline; rows 200–299 require `scripts/run_build_hard_gsm8k_routing_dataset.py --rank-offset 200 --subset-size 100` with active API key |
| `gpqa_diamond` → main paper | **Protocol mismatch** | Policy v5/v6/v7 tuned on GSM8K/MATH; v5 over-revises (91% rate) on GPQA; no canonical policy calibration for GPQA in this manuscript pass |

---

## 4. Exact Commands to Run

### A. Offline expansion already executed (no API needed):

```bash
# Combine hard_gsm8k b1+b2 into 200-row dataset and evaluate
python3 -c "
import csv
b1 = list(csv.DictReader(open('data/real_hard_gsm8k_routing_dataset_enriched.csv')))
b2 = list(csv.DictReader(open('data/real_hard_gsm8k_b2_routing_dataset_enriched.csv')))
rows = b1 + b2
w = csv.DictWriter(open('/tmp/hard_gsm8k_combined.csv','w',newline=''), fieldnames=list(b1[0].keys()))
w.writeheader(); [w.writerow(r) for r in rows]
"

python3 scripts/run_real_policy_eval.py \
  --dataset-csv /tmp/hard_gsm8k_combined.csv \
  --output-dir outputs/expanded_main_regimes/hard_gsm8k_combined_200

# GPQA (198 rows, extra regime)
python3 scripts/run_real_policy_eval.py \
  --dataset-csv data/real_gpqa_diamond_routing_dataset_enriched.csv \
  --output-dir outputs/expanded_main_regimes/gpqa_diamond_198
```

### B. Commands blocked by API key (for future reference):

```bash
# Expand GSM8K random to 300 (requires OPENAI_API_KEY)
python3 scripts/run_build_real_routing_dataset.py \
  --paired-outcomes --subset-size 300 \
  --output-dataset-csv data/real_gsm8k_routing_dataset_enriched_300.csv

# Expand MATH500 to 300 (requires OPENAI_API_KEY)
python3 scripts/run_build_math500_routing_dataset.py --subset-size 300

# Expand hard-GSM8K to 300 (requires OPENAI_API_KEY)
python3 scripts/run_build_hard_gsm8k_routing_dataset.py \
  --subset-size 100 --rank-offset 200
# Then combine b1+b2+b3 for 300 total
```

---

## 5. Execution Notes

- The offline evaluation (`run_real_policy_eval.py`) is fully reproducible from committed data.
- All policy logic (v5/v6/v7) is deterministic and requires no randomness; results are bit-exact reproducible.
- The combined hard_gsm8k n=200 evaluation is the only genuine scale expansion achievable from current committed data.
- Expanding any regime to n=300 requires an active OpenAI API key and approximately 200–400 additional gpt-4o-mini queries per regime.
