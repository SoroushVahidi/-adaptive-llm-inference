# Next-Stage Experiment Results (EAAI-oriented)

**Evidence labels:** **measured_now** where numbers come from committed artifacts in this branch; **exploratory_only** for small-N baselines and CV-learned routers; **blocked** only if a step could not run.

---

## 1. What ran successfully

| Component | Status | Artifacts |
|-----------|--------|-----------|
| `reasoning_then_revise` (review pass after frozen reasoning) | **measured_now** | `scripts/run_reasoning_then_revise_addon.py`, enriched `data/*_enriched.csv`, `outputs/reasoning_then_revise/*` |
| Integrated optional RTR in `build_real_routing_dataset` | **implemented** | `include_reasoning_then_revise` on `BuildConfig`; used for AIME build |
| Oracle routing (revise iff `revise_helpful`) | **measured_now** | `outputs/oracle_routing_eval/{dataset}_oracle_summary.json` |
| Budget / Pareto marginal-revise curve | **measured_now** | `outputs/budget_sweep/{dataset}_budget_curve.csv` |
| Cascade (unified_confidence_score thresholds) | **measured_now** | `outputs/next_stage_eval/*/cascade_curve.csv` |
| Baselines ladder | **partial** | `outputs/baselines/gsm8k_baseline_summary.json` (n=30), `hard_gsm8k` (n=30), `math500` (n=15 after cost kill) |
| Hard GSM8K robustness B2 | **measured_now** | `rank_offset=100` slice, `data/real_hard_gsm8k_b2_*`, policy + learned router |
| AIME 2024 (HF `HuggingFaceH4/aime_2024`) | **measured_now** | N=30, `data/real_aime2024_routing_dataset.csv` |
| Final cross-regime CSV | **measured_now** | `outputs/cross_regime_comparison/final_cross_regime_summary.csv` |

**Provider / model:** OpenAI **`gpt-4o-mini`** (greedy 0 for single-sample stages; sample temp 0.7 for self-consistency multi-sample).

---

## 2. What failed or was reduced

| Issue | Details | User fix |
|-------|---------|----------|
| Long-running baseline loop | Initial `for ds in gsm8k hard_gsm8k math500; limit 30` exceeded time budget while **math500** was running (~1M+ seconds wall). Processes **killed**; **gsm8k** and **hard_gsm8k** summaries (n=30) were already written. | Re-run `python3 scripts/run_next_stage_baselines.py --dataset math500 --limit 30` when time allows. |
| MATH500 baseline sample size | Completed **n=15** for ladder after kill (file `outputs/baselines/math500_baseline_summary.json`). | Increase `--limit` for paper runs. |
| HuggingFace rate limits | Unauthenticated HF Hub warnings; datasets still loaded. | Set **`HF_TOKEN`** for faster / more reliable Hub access. |

---

## 3. Key numbers

### 3.1 `reasoning_then_revise` on full 100-query tables (addon script)

| Dataset | RTR accuracy | RTR “helpful” rate (wrong reasoning → fixed) |
|---------|--------------|-----------------------------------------------|
| gsm8k_random100 | **0.93** | 0.03 |
| hard_gsm8k_100 | **0.90** | 0.11 |
| math500_100 | **0.67** | 0.03 |

**Interpretation (measured_now):** On **MATH500**, RTR (**0.67**) beats one-shot reasoning (**0.64**) and **direct_plus_revise** (**0.64**) on the same 100 rows — the gain is small but consistent with “revise-after-reasoning” using full chain-of-thought context. On **hard GSM8K**, RTR (**0.90**) is between reasoning (**0.79**) and oracle-ish always-revise (**0.86** for `direct_plus_revise` in table — note different pipeline).

### 3.2 Oracle routing upper bound (`revise_helpful` oracle)

| Dataset | Oracle accuracy | Avg cost | Revise rate |
|---------|-----------------|----------|-------------|
| gsm8k_random100 | 0.92 | 1.02 | 0.02 |
| math500_100 | 0.70 | 1.06 | 0.06 |
| hard_gsm8k_100 | 0.91 | 1.12 | 0.12 |
| hard_gsm8k_b2 | 0.92 | 1.09 | 0.09 |

**Oracle gap vs best V6/V7:** On random GSM8K the gap is **tiny** (oracle 0.92 vs V6 0.92). On hard GSM8K, oracle **0.91** vs best heuristic **0.82** (V7) shows **headroom** when revise is applied only where it helps.

### 3.3 Budget sweep (marginal gain ordering)

See `outputs/budget_sweep/*_budget_curve.csv`. Example **hard_gsm8k_100**: cost **1.1** achieves **0.89** accuracy vs **0.79** at cost 1.0.

### 3.4 Baselines (exploratory_only — small N)

- **gsm8k** (n=30): reasoning_then_revise **96.7%** @ cost 2; ties elsewhere except beats nothing strongly on this easy slice.
- **hard_gsm8k** (n=30): self_consistency_3/5 **90%** @ cost 3/5 vs reasoning **80%**.
- **math500** (n=15): self-consistency **hurts** vs greedy in this draw; RTR matches greedy **60%**.

### 3.5 AIME 2024 (N=30, measured_now)

Very low exact-match with current extraction: reasoning **13.3%**, `direct_plus_revise` **6.7%**, `reasoning_then_revise` **16.7%**. **revise_helpful** vs `direct_plus_revise` is **0** — the second stage rarely “rescues” under numeric-oriented direct+revise. **exploratory_only** for publication claims without stronger models / symbolic eval.

### 3.6 Hard GSM8K B2 (independent slice, ranks 100–199)

- **revise_helpful:** 9%
- **V6/V7:** both **0.89** @ 1.27 / 1.40 cost
- **Learned router (bagging):** CV F1 **~0.57**, similar spirit to B1

---

## 4. Cross-dataset comparison

See **`outputs/cross_regime_comparison/final_cross_regime_summary.csv`** (uses **enriched** CSVs when present for RTR column).

---

## 5. Cost–accuracy tradeoff

**measured_now:** On **hard GSM8K**, marginal-revise curves and **cascade** thresholds show **clear Pareto moves** (e.g. +10 pp at +0.1 cost in one bucket). **Heuristic V7** improves accuracy over V6 on hard slice **B1** but costs more; on **B2** V6 and V7 **tie**.

---

## 6. Does revise-after-reasoning help (especially MATH500)?

**Yes (small, N=100):** **0.64 → 0.67** vs one-shot reasoning; also **0.64 → 0.67** vs `direct_plus_revise` gold alignment on the same rows. **exploratory_only** for MATH until full 500 and stronger evaluation.

---

## 7. Oracle gap

- **Easy GSM8K:** **Small** — routing barely beats always choosing one strategy.
- **Hard GSM8K:** **Larger** — oracle **~0.91** vs best V6/V7 **~0.82** suggests value in **better revise targeting**, not more features alone.

---

## 8. New / updated entry points

```bash
# Enrich existing routing CSV with one RTR call per row
python3 scripts/run_reasoning_then_revise_addon.py --input-csv data/real_math500_routing_dataset.csv \
  --output-csv data/real_math500_routing_dataset_enriched.csv \
  --summary-json outputs/reasoning_then_revise/math500_rtr_addon_summary.json --mode math

# Oracle + budget + cascade
python3 scripts/run_next_stage_postprocess.py --dataset-key math500_100 \
  --routing-csv data/real_math500_routing_dataset_enriched.csv \
  --policy-summary-json outputs/real_math500_policy_eval/summary.json

# Baselines (expensive on math500)
python3 scripts/run_next_stage_baselines.py --dataset hard_gsm8k --limit 30

# AIME
python3 scripts/run_build_aime_routing_dataset.py --limit 30 --include-reasoning-then-revise

# Final summary table
python3 scripts/run_final_cross_regime_summary.py
```

---

## 9. Requirements

- **`OPENAI_API_KEY`**: required for all live runs (script raises if missing).
- **`datasets`**, **`scikit-learn`** (for learned-router eval): install via `pip install -e ".[dev]"` as in `AGENTS.md`.
- **Internet**: HuggingFace dataset download for MATH500 / GSM8K / AIME when not cached.
