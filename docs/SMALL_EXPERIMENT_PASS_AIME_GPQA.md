# Small Experiment Pass: AIME-2024 and GPQA-Diamond

**Date:** 2026-03-30  
**Scope:** Manuscript-strengthening pass — AIME-2024 and GPQA/GPQA-Diamond datasets,
plus confidence/uncertainty threshold router baseline.

---

## 1. Status Audit

### AIME-2024

| Item | Status |
|------|--------|
| Data file | ✅ `data/real_aime2024_routing_dataset.csv` (30 queries) |
| Dataset loader | ✅ `src/datasets/aime2024.py` |
| Routing features | ✅ All 80+ features pre-computed (including `unified_confidence_score`) |
| Model responses | ✅ `reasoning_raw`, `reasoning_correct`, `revise_correct`, `revise_helpful` |
| Policy evaluation | 🔬 Exploratory only — not in main paper (now added in this pass) |
| Policy eval outputs | ✅ Generated: `outputs/small_pass/aime_*` |

**Verdict: present but incomplete → upgraded to manuscript-grade in this pass.**

### GPQA / GPQA-Diamond

| Item | Status |
|------|--------|
| Normalized data file | ✅ `data/gpqa_diamond_normalized.jsonl` (198 questions) |
| Dataset loaders | ✅ `src/datasets/gpqa.py` (official `Idavidrein/gpqa` + `gpqa_diamond`), `src/datasets/gpqa_diamond.py` |
| Hugging Face access | ✅ Use `load_dataset("Idavidrein/gpqa", "gpqa_diamond", ...)` (config required) |
| Enriched routing CSV + policy eval | 🔴 **Not produced (API 401)** — pipeline exists; a **198-query** build was attempted and **every** call failed with **invalid_api_key** (placeholder/invalid `OPENAI_API_KEY`). See **`docs/GPQA_EVALUATION_STATUS.md` §5a**. |

**Verdict:** Dataset access and loaders are **solved**; manuscript-grade routing rows require a **valid** OpenAI key and a clean retry after clearing failed-run checkpoints (see GPQA status doc).

### Confidence/Uncertainty Threshold Router

| Item | Status |
|------|--------|
| Implementation | ✅ `src/baselines/confidence_threshold_router.py` |
| Unit tests | ✅ `tests/test_manuscript_support.py::TestConfidenceThresholdRouter` |
| Main regime outputs | ✅ Generated: `outputs/baselines/confidence_threshold/` |
| AIME outputs | ✅ Generated as part of AIME eval |

**Verdict: present and now manuscript-ready.**

---

## 2. AIME-2024 Results

### Dataset Characteristics

- **Queries:** 30 problems from AIME 2024
- **Model:** GPT-4o-mini (committed responses, no API calls required)
- **Difficulty:** Very high — competition-level mathematics

### Policy Comparison Table

| Route | Accuracy | Avg Cost | Revise Rate | Notes |
|-------|----------|----------|-------------|-------|
| `reasoning_greedy` | **0.133** | 1.00 | 0% | Cheap baseline |
| `direct_plus_revise` | 0.067 | 2.00 | 100% | Always-revise baseline |
| `adaptive_policy_v5` | 0.067 | 1.90 | 90% | Calibrated role + unified error |
| `adaptive_policy_v6` | 0.133 | 1.13 | 13% | v5 + answer confidence filtering |
| `adaptive_policy_v7` | 0.133 | 1.30 | 30% | v6 + extended fixes (main policy) |
| `confidence_threshold` | 0.133 | 1.00 | 0% | Threshold=0.00, target_cost≤1.2 |
| `oracle` | **0.133** | 1.00 | 0% | Upper bound (revise_helpful=0 always) |

Source: `outputs/paper_tables_small_pass/aime_policy_comparison.csv`

### Key Findings

**Finding 1: Revision never helps on AIME-2024.**
`revise_helpful = 0` for all 30 queries. This means the oracle routing strategy
is identical to the cheap baseline — no routing policy can improve over
`reasoning_greedy`. This is an honest and informative result: on competition-level
mathematics, a single GPT-4o-mini pass either succeeds or fails, and a second
pass does not recover the error.

**Finding 2: Policy v5 over-escalates badly.**
Policy v5 routes 90% of AIME queries to revision, achieving only 6.7% accuracy
(worse than cheap baseline). This demonstrates that v5's signals are miscalibrated
for high-difficulty competition math — the problem features that trigger escalation
(number roles, constraint violations, etc.) are ubiquitous in AIME but do not
predict actual revision utility.

**Finding 3: Policies v6/v7 match the oracle at lower cost than v5.**
V6/V7's answer-confidence filtering prevents the over-escalation problem: v6
achieves 13.3% accuracy at cost 1.13, matching cheap baseline accuracy but at
higher cost. V7 revises 30% of queries unnecessarily (accuracy unchanged, cost
rises). The rational operating point is the cheap baseline (cost=1.0).

**Finding 4: The confidence threshold router correctly identifies threshold=0
(revise nobody) as the best operating point.**
This agrees with the oracle. When `revise_helpful` is universally zero, the
optimal routing policy is to never revise — which the confidence sweep selects.

### Interpretation for Manuscript

The AIME-2024 result is an honest **negative result** that strengthens the
manuscript: it shows the routing problem becomes degenerate when revision cannot
help. The main paper's claims are scoped to regimes where revision *can* help
(hard-GSM8K: 12% helpful, hard-GSM8K-b2: 9% helpful) — AIME illustrates the
limits of the approach on tasks beyond the difficulty range where revision helps.

---

## 3. Confidence Threshold Router Results (Main Regimes)

Full results in `outputs/paper_tables_small_pass/confidence_baseline_main_regimes.csv`.
Combined table in `outputs/paper_tables_small_pass/small_pass_combined_comparison.csv`.

| Regime | Cheap Acc | Best Policy Acc | Oracle Acc | Conf-Router Acc | Conf-Router Cost |
|--------|-----------|-----------------|------------|-----------------|------------------|
| `gsm8k_random_100` | 0.90 | **0.92** | 0.92 | 0.92 | 1.11 |
| `hard_gsm8k_100` | 0.79 | 0.82 | 0.91 | **0.89** | 1.13 |
| `hard_gsm8k_b2` | 0.83 | **0.91** | 0.92 | 0.89 | 1.09 |
| `math500_100` | 0.64 | **0.65** | 0.70 | 0.66 | 1.06 |
| `aime2024` | **0.133** | 0.133 | 0.133 | 0.133 | 1.00 |

### Key Finding: Confidence Threshold is a Strong Baseline

- On `hard_gsm8k_100`, the confidence threshold baseline (acc=0.89) **outperforms
  the best adaptive policy** (acc=0.82) at similar cost.
- On `hard_gsm8k_b2`, confidence threshold (0.89) is slightly below best policy (0.91).
- On `gsm8k_random_100` and `math500_100`, it matches the best adaptive policy.

This is an important honest finding: a simple threshold on `unified_confidence_score`
(a pre-computed offline signal) is competitive with the main adaptive policies.
This strengthens the manuscript by showing the paper's results are competitive
against a well-tuned simple baseline.

**Note on confidence threshold advantage:** The operating point is chosen to
maximize accuracy subject to a cost budget, which implicitly optimizes the
threshold using the full dataset. The main adaptive policies use fixed hand-tuned
thresholds and do not optimize for a specific cost budget target. A fair
comparison would show threshold-vs-accuracy curves (see `aime_confidence_sweep.csv`
and the per-regime sweeps in `outputs/baselines/confidence_threshold/`).

---

## 4. GPQA-Diamond: Manuscript pipeline (see dedicated doc)

**Authoritative commands:** `docs/GPQA_EVALUATION_STATUS.md`

**Summary:** `data/gpqa_diamond_normalized.jsonl` is labels-only. Enriched routing
evaluation uses the same paired builder as GSM8K (see `src/data/build_real_routing_dataset.py`,
`dataset="gpqa_diamond"`). **Requires `OPENAI_API_KEY`.**

```bash
python scripts/run_build_real_routing_dataset.py \
  --paired-outcomes \
  --dataset gpqa_diamond \
  --subset-size 198 \
  --output-dataset-csv data/real_gpqa_diamond_routing_dataset_enriched.csv

python scripts/run_real_policy_eval.py \
  --dataset-csv data/real_gpqa_diamond_routing_dataset_enriched.csv \
  --output-dir outputs/real_gpqa_policy_eval
```

**Blockers doc (historical context):** `docs/BLOCKERS_AIME_GPQA_SMALL_PASS.md`

---

## 5. Reproducibility

### Run Everything (No API Key Required)

```bash
# Full small-pass: AIME eval + confidence baseline
python scripts/run_small_pass.py

# Or individually:
python scripts/run_small_pass_aime_eval.py
python scripts/run_confidence_baseline.py --include-aime
```

### Outputs

| File | Description |
|------|-------------|
| `outputs/small_pass/aime_summary.json` | AIME evaluation summary |
| `outputs/small_pass/aime_policy_comparison.csv` | AIME comparison table |
| `outputs/small_pass/aime_confidence_sweep.csv` | AIME confidence threshold sweep |
| `outputs/small_pass/aime_per_query_decisions.csv` | Per-query AIME decisions |
| `outputs/baselines/confidence_threshold/confidence_threshold_sweep.csv` | Full sweep |
| `outputs/baselines/confidence_threshold/confidence_threshold_summary.csv` | Summary |
| `outputs/paper_tables_small_pass/small_pass_combined_comparison.csv` | Combined table |
| `outputs/paper_tables_small_pass/aime_policy_comparison.csv` | AIME table |
| `outputs/paper_tables_small_pass/confidence_baseline_main_regimes.csv` | Conf baseline |

The confidence-threshold sweep CSVs are also written under
`outputs/small_pass/confidence_threshold/` when you run `run_small_pass.py`
(the same numeric content as `scripts/run_confidence_baseline.py`, which writes
to `outputs/baselines/confidence_threshold/`).

### Tests

```bash
python -m pytest tests/test_small_pass.py -v
```
