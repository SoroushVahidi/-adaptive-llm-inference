# Final Pre-Run Audit — Real GSM8K 100-Query Experiment

**Audit date:** 2026-03-29  
**Audited by:** Copilot automated audit  
**Target command:** `python3 scripts/run_build_real_routing_dataset.py --subset-size 100`

---

## 1. Summary Verdict

| Area | Status |
|------|--------|
| `run_build_real_routing_dataset.py` — imports | ✅ Clean |
| `run_build_real_routing_dataset.py` — paths | ✅ Correct |
| `run_build_real_routing_dataset.py` — dataset loader | ✅ HuggingFace + bundled fallback |
| `run_build_real_routing_dataset.py` — output directory creation | ✅ Auto-created |
| `run_build_real_routing_dataset.py` — API key check | ✅ Early exit with clear message |
| `run_build_real_routing_dataset.py` — oracle strategy pipeline | ✅ Uses tested `oracle_subset_eval` |
| `run_build_real_routing_dataset.py` — routing dataset assembly | ✅ Uses tested `routing_dataset` module |
| `run_build_real_routing_dataset.py` — artifact writing | ✅ oracle + routing CSVs + summaries |
| `run_real_routing_model_eval.py` — imports | ✅ Clean |
| `run_real_routing_model_eval.py` — routing CSV guard | ✅ Clear blocker if file missing |
| `run_real_routing_model_eval.py` — oracle label guard | ✅ Blocks if no oracle labels |
| `run_real_routing_model_eval.py` — model fitting | ✅ Uses tested `router_baseline` module |
| `run_real_routing_model_eval.py` — output writing | ✅ JSON + CSV results |
| **Only missing prerequisite** | ⛔ `OPENAI_API_KEY` |

---

## 2. Files Involved

### New scripts (created in this audit)

| File | Role |
|------|------|
| `scripts/run_build_real_routing_dataset.py` | Main 100-query experiment entry point |
| `scripts/run_real_routing_model_eval.py` | Routing model evaluation follow-up |

### Existing modules (verified as ready)

| File | Role |
|------|------|
| `src/datasets/gsm8k.py` | GSM8K loader (HuggingFace + local JSON fallback) |
| `src/datasets/bundled/gsm8k_test_sample.json` | 20-record offline fallback sample |
| `src/datasets/routing_dataset.py` | Routing dataset assembler and CSV writer |
| `src/evaluation/oracle_subset_eval.py` | Oracle strategy runner, summaries, output writer |
| `src/evaluation/strategy_expansion_eval.py` | Individual strategy runners |
| `src/evaluation/expanded_strategy_eval.py` | Additional strategy runners |
| `src/features/precompute_features.py` | Query feature extraction (z(x)) |
| `src/models/openai_llm.py` | OpenAI API wrapper |
| `src/policies/router_baseline.py` | Decision tree / logistic regression baseline |
| `src/utils/answer_extraction.py` | Numeric answer extraction from model output |

### Config files

| File | Role |
|------|------|
| `configs/oracle_subset_eval_gsm8k.yaml` | Reference config for 15-query oracle eval |

---

## 3. What Is Ready

### Offline-ready (no API key needed)

- All imports resolve cleanly (verified with Python syntax check + `ruff`).
- GSM8K loader falls back to `src/datasets/bundled/gsm8k_test_sample.json` when
  HuggingFace is unavailable; 20 records are bundled.
- Output directories (`outputs/real_routing_dataset/`, `outputs/real_router_eval/`)
  are created automatically by the scripts.
- Feature extraction (`extract_query_features`) is pure-Python and offline.
- Routing dataset assembler and CSV writer are offline.
- Router baseline fitting (majority + decision tree + logistic regression) is
  offline once the CSV exists.
- `run_build_real_routing_dataset.py` exits immediately with a clear blocker
  message if `OPENAI_API_KEY` is missing.
- `run_real_routing_model_eval.py` exits immediately with a clear blocker message
  if the routing CSV is missing or has no oracle labels.
- All 492 existing tests pass.
- Both new scripts pass `ruff check`.

### API-dependent (requires `OPENAI_API_KEY`)

- Oracle strategy evaluation (7 strategies × N queries = model calls).
- First-pass features requiring a model call.

---

## 4. What Is Still Fragile

| Fragility | Severity | Notes |
|-----------|----------|-------|
| 20-record bundled fallback | Low | If HuggingFace is offline the run will succeed but with only 20 queries instead of 100. The script prints a warning. |
| No retry / rate-limit handling in `OpenAILLMModel` | Medium | A single `urllib` call with a configurable timeout. Long runs (100 queries × 7 strategies) may hit rate limits silently. Consider adding `--timeout` headroom or running in batches. |
| `reasoning_greedy` strategy not in `CORE_ORACLE_STRATEGIES` list | Low | `reasoning_greedy` has a runner in `oracle_subset_eval.py` but is not included in `CORE_ORACLE_STRATEGIES`. This is pre-existing behaviour; the 100-query run will evaluate the 7 listed core strategies, not `reasoning_greedy`. |
| No checkpoint / resume | Medium | If the run fails mid-way (network, rate limit, timeout), all progress is lost. The entire set of queries must be re-run. For 100 queries this is acceptable; for larger runs it would not be. |
| `per_query_matrix.csv` `correct` column | Low | `load_oracle_files` sums `correct` column per query_id across multiple rows. Requires that `write_oracle_outputs` writes one row per (query, strategy). Verified correct in current code. |
| HuggingFace `datasets` library must be installed | Low | Handled by `pip install -e ".[dev]"`. If not installed, HuggingFace load fails and bundled fallback kicks in. |

---

## 5. Exact Prerequisites Still Missing

**Single blocker: `OPENAI_API_KEY`**

```bash
export OPENAI_API_KEY=sk-<your-key-here>
```

Everything else is in place:
- Code is ready.
- Imports resolve.
- Output directories are auto-created.
- Bundled fallback data is present for offline query loading.
- All tests pass.

---

## 6. Commands to Run When API Access Is Available

### Step 1 — Build the real routing dataset (100 queries)

```bash
export OPENAI_API_KEY=sk-...
python3 scripts/run_build_real_routing_dataset.py --subset-size 100
```

**Estimated API calls:** 100 queries × 7 strategies × avg ~1–3 model calls per
strategy ≈ 700–2100 calls. At `gpt-4o-mini` pricing this is very cheap.

**Expected runtime:** 10–30 minutes depending on API latency.

### Step 2 — Evaluate routing models

```bash
python3 scripts/run_real_routing_model_eval.py
```

No API key required for this step. Runs in seconds.

### Optional: Smaller smoke-test first

```bash
export OPENAI_API_KEY=sk-...
python3 scripts/run_build_real_routing_dataset.py --subset-size 5
python3 scripts/run_real_routing_model_eval.py \
    --routing-csv outputs/real_routing_dataset/routing_dataset.csv
```

### Optional: Custom model or output path

```bash
python3 scripts/run_build_real_routing_dataset.py \
    --subset-size 100 \
    --model gpt-4o-mini \
    --output-dir outputs/real_routing_dataset_v2

python3 scripts/run_real_routing_model_eval.py \
    --routing-csv outputs/real_routing_dataset_v2/routing_dataset.csv \
    --output-dir outputs/real_router_eval_v2
```

---

## 7. Expected Outputs If the Run Succeeds

### After `run_build_real_routing_dataset.py --subset-size 100`

```
outputs/real_routing_dataset/
├── oracle_assignments.csv         (100 rows: question_id, oracle labels)
├── per_query_matrix.csv           (100 × 7 rows: per-strategy correctness)
├── oracle_summary.json            (accuracy per strategy, oracle gap)
├── summary.json                   (oracle eval summary from write_oracle_outputs)
├── summary.csv                    (same, CSV format)
├── pairwise_win_matrix.csv        (7×7 strategy comparison)
├── routing_dataset.csv            (100 rows: features + oracle labels, flat CSV)
└── routing_dataset_summary.json   (column inventory, num_queries: 100,
                                    oracle_labels_available: true)
```

Key values to verify in `routing_dataset_summary.json`:
- `num_queries`: 100
- `oracle_labels_available`: true
- `num_feature_columns`: ≥ 13 (query features) + up to 6 (first-pass features)
- `num_label_columns`: 5

### After `run_real_routing_model_eval.py`

```
outputs/real_router_eval/
├── router_eval_results.json       (all model accuracies, labeled row counts)
├── router_summary.json            (sklearn availability, task summaries)
├── binary_results.json            (direct_already_optimal task)
└── multiclass_results.json        (best_accuracy_strategy task)
```

Key values to verify in `router_eval_results.json`:
- `labeled_rows`: 100 (or close to it)
- `binary_task.results`: at least 1 entry with `accuracy` value
- `multiclass_task.results`: at least 1 entry with `accuracy` value

---

## 8. Blockers and Caveats

1. **`OPENAI_API_KEY` is the only hard blocker.** Both scripts exit cleanly with
   a clear message when this is missing.

2. **Rate limits:** `gpt-4o-mini` default rate limits are generous for this
   workload (~700–2100 calls). If rate limits are hit, the `urllib`-based client
   will raise a `RuntimeError` and the script will exit with a `[BLOCKED]` message.
   Re-run the script; there is no checkpointing.

3. **100 queries vs 20 bundled queries:** The HuggingFace GSM8K test split has
   1319 queries. Loading 100 requires network access on the first run.
   Subsequent runs use the HuggingFace local cache in `data/`. If the network is
   unavailable on first run, the script falls back to the 20-record bundled
   sample and will process only 20 queries.

4. **Router model significance:** With 100 labelled queries, the train/test split
   for the decision tree and logistic regression will be small (e.g., 80 train /
   20 test rows). Accuracy numbers will have high variance. This is noted in
   `docs/CURRENT_STATE_SUMMARY.md` — 100 queries is a meaningful improvement over
   the 20-query bundled sample but still not large enough for robust paper claims.

5. **No `reasoning_greedy` in core oracle strategies:** The 7 core strategies
   evaluated are `direct_greedy`, `reasoning_best_of_3`, `structured_sampling_3`,
   `direct_plus_verify`, `direct_plus_revise`,
   `direct_plus_critique_plus_final`, `first_pass_then_hint_guided_reason`.
   To include `reasoning_greedy`, pass it explicitly:
   ```bash
   python3 scripts/run_build_real_routing_dataset.py --subset-size 100 \
       --strategies direct_greedy reasoning_greedy reasoning_best_of_3 \
           structured_sampling_3 direct_plus_verify direct_plus_revise \
           direct_plus_critique_plus_final first_pass_then_hint_guided_reason
   ```
