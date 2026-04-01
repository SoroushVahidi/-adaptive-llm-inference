# Dataset and Regime Scale Readiness Audit

**Date:** 2026-04-01  
**Evidence basis:** Committed data files in `data/`, source code inspection, and confirmed run of `run_real_policy_eval.py`

---

## Summary Table

| Regime / Dataset | Current Canonical n | Max Offline n | Max Total n (with API) | Access Method | Blocker if Expansion Fails |
|-----------------|---------------------|---------------|------------------------|---------------|---------------------------|
| `gsm8k_random_100` | 100 | 100 | 1319 (full GSM8K test) | `data/real_gsm8k_routing_dataset_enriched.csv` | API key needed for new queries; script: `scripts/run_build_real_routing_dataset.py` |
| `hard_gsm8k_100` (b1) | 100 | 200 (b1+b2) | 300 (b1+b2+b3) | `data/real_hard_gsm8k_routing_dataset_enriched.csv` | b2 available offline; b3 requires API; script: `scripts/run_build_hard_gsm8k_routing_dataset.py --rank-offset 200` |
| `hard_gsm8k_b2` | 100 | 200 (b1+b2) | 300 (b1+b2+b3) | `data/real_hard_gsm8k_b2_routing_dataset_enriched.csv` | Same as above |
| `math500_100` | 100 | 100 | 500 (full MATH500) | `data/real_math500_routing_dataset_enriched.csv` | API key needed; script: `scripts/run_build_math500_routing_dataset.py --subset-size 500` |
| `gpqa_diamond` (extra) | 0 (not canonical) | 198 | 198 | `data/real_gpqa_diamond_routing_dataset_enriched.csv` | Policy calibration mismatch; v5 over-revises (91% rate) |
| `aime2024` (exploratory) | 30 | 30 | ~30–90 (competition bounds) | `data/real_aime2024_routing_dataset.csv` | Only 30 problems in 2024 AIME; no enriched features pre-computed |

---

## Detailed Notes

### `gsm8k_random_100`

- **Committed data:** `data/real_gsm8k_routing_dataset_enriched.csv` — exactly 100 rows with full enriched features.
- **Offline evaluation:** Fully runnable; `run_real_policy_eval.py` produces all four comparisons.
- **Expansion path:** Run `scripts/run_build_real_routing_dataset.py --paired-outcomes --subset-size 300 --output-dataset-csv data/real_gsm8k_routing_dataset_300.csv` (requires `OPENAI_API_KEY`). The GSM8K test set has 1319 items so headroom is ample.
- **Blocker:** `OPENAI_API_KEY` not available in this environment.

### `hard_gsm8k_100` (Batch 1) and `hard_gsm8k_b2` (Batch 2)

- **Committed data:** Two disjoint 100-row enriched CSVs using identical selection protocol:
  - b1: ranks 0–99 of hardness-sorted GSM8K test (`rank_offset=0`)
  - b2: ranks 100–199 of same sort (`rank_offset=100`)
  - Same formula: `z_len + z_num + 0.5*multi_step + 0.5*equation + 0.3*percent + 0.3*fraction + 0.15*tq_count`
  - Same pool (1319 test items), same seed (42)
- **Max offline n:** 200 (concatenating b1+b2; zero overlap confirmed)
- **Offline evaluation:** Fully runnable on combined 200-row CSV.
- **Expansion to 300:** Requires generating b3 (ranks 200–299) with `run_build_hard_gsm8k_routing_dataset.py --rank-offset 200 --subset-size 100`.
- **Blocker for 300:** `OPENAI_API_KEY` not available.

### `math500_100`

- **Committed data:** `data/real_math500_routing_dataset_enriched.csv` — exactly 100 rows.
- **Offline evaluation:** Fully runnable.
- **Expansion path:** `scripts/run_build_math500_routing_dataset.py --subset-size 300`. MATH500 has 500 questions so headroom exists.
- **Blocker:** `OPENAI_API_KEY` not available.

### `gpqa_diamond` (Extra Regime — Not Canonical Main Paper)

- **Committed data:** `data/real_gpqa_diamond_routing_dataset_enriched.csv` — 198 rows with full enriched features.
- **Offline evaluation:** Runs successfully with `run_real_policy_eval.py`.
- **Key caveat:** Adaptive policy v5 severely over-revises on GPQA (91.4% revise rate), which reflects domain mismatch — the policy's numeric-error signals and role-coverage features are calibrated for arithmetic (GSM8K/MATH) not science MCQ.
- **Protocol parity:** Partial. The binary routing decision (revise vs. greedy) and correctness columns are present and valid. However, adaptive policy thresholds are not calibrated for science-question domains.
- **Max offline n:** 198 (all committed rows).
- **Expansion to full GPQA-Diamond:** Would require API access; full GPQA-Diamond has ~448 questions.

### `aime2024` (Exploratory)

- **Committed data:** `data/real_aime2024_routing_dataset.csv` — 30 rows; no enriched features.
- **Evaluation status:** Small-pass only (`outputs/small_pass/`); not integrated into main paper tables.
- **Max n:** 30 (all AIME 2024 problems). This regime is not expandable without access to AIME 2025 problems.

---

## Scale Readiness Summary

| Regime | Ready for Offline Eval? | Achievable n (Offline) | Can Reach 300? (Offline) |
|--------|------------------------|------------------------|--------------------------|
| `gsm8k_random_100` | ✅ Yes | 100 | ❌ No — API required |
| `hard_gsm8k` (combined) | ✅ Yes | **200** | ❌ No — partial; 100 more need API |
| `math500_100` | ✅ Yes | 100 | ❌ No — API required |
| `gpqa_diamond` | ✅ Yes (with caveats) | 198 | ❌ No — API required, policy mismatch |
| `aime2024` | ⚠️ Partial | 30 | ❌ No — problem set exhausted |

---

## Conclusion

No canonical main-paper regime can be expanded to n=300 from committed data alone. The only meaningful offline scale expansion is the **hard_gsm8k regime from n=100 to n=200** by combining the two committed batches. All further expansion requires an active `OPENAI_API_KEY`.

For the GPQA regime, 198 rows are available offline and produce valid routing metrics, but the policy v5 behaviour on GPQA indicates the current adaptive policy is not calibrated for that domain.
