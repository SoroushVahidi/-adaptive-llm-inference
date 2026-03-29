# Multi-action supervised routing — results report

## 1. What ran successfully

| Step | Status |
|------|--------|
| **Actions implemented** | `reasoning_greedy`, `direct_plus_revise`, `reasoning_then_revise` (CoT → revise on full trace), `self_consistency_3` (3× CoT + majority vote; ambiguous vote logged) — wired via `MULTI_ACTION_ORACLE_STRATEGIES` in `src/evaluation/oracle_subset_eval.py` and runners in `src/evaluation/strategy_expansion_eval.py`. |
| **Dataset build (real API)** | `python3 scripts/run_build_multi_action_dataset.py --dataset gsm8k_hard --subset-size 15` and `--dataset math500 --subset-size 12` completed with `gpt-4o-mini`. |
| **Artifacts** | CSV paths: `data/multi_action_routing_gsm8k_hard.csv`, `data/multi_action_routing_math500.csv`. Summaries: `outputs/multi_action_oracle/gsm8k_hard_oracle_summary.json`, `outputs/multi_action_oracle/math500_oracle_summary.json`. |
| **Model / policy eval script** | `python3 scripts/run_multi_action_model_eval.py --csv ...` produced `outputs/multi_action_models/gsm8k_hard_model_results.json`, `math500_model_results.json`, and matching `*_policy_simulation.csv`. |
| **Unit tests** | `tests/test_multi_action_routing.py` covers new runners, oracle eval wiring, row assembly, GSM8K `tail_max_samples`. |
| **Hard GSM8K proxy** | `load_gsm8k(..., tail_max_samples=N)` returns the **last N** test problems in fixed order (documented as a tail-slice proxy; not a separate curated “hard” list). |

## 2. What failed or is blocked

### 2.1 Classifier training (degenerate labels)

**Symptom:** For both slices, `best_accuracy_action`, `best_utility_action_lambda_0_10`, and `best_utility_action_lambda_0_25` each had **only one unique value** (`reasoning_greedy`).

**Exact behavior:** `run_multi_action_model_eval.py` prints to stderr, for each target:

`Label ... has fewer than 2 classes: [np.str_('reasoning_greedy')]`

and sets `run_status: SKIPPED` with `skip_reason` in the JSON (classifiers not trained; baseline/oracle policy tables still emitted).

**Root cause:** On every query in these runs, all four actions had the **same** correctness (full ties on accuracy). Tie-breaking then always picks **`reasoning_greedy`** (lowest cost, first in deterministic order among equals).

**Fix:** Run substantially more queries (e.g. 100–500), stratify by difficulty, or filter to queries where actions disagree; then re-run the model script.

### 2.2 Comparison to binary revise routing

No fresh binary revise classifier was re-trained in this pass. **Empirically**, on these slices the multi-action oracle never preferred `direct_plus_revise` over `reasoning_greedy` when accuracy tied — so **learned multi-action routing cannot beat binary revise routing on discriminative labels until actions diverge.**

### 2.3 Self-consistency ambiguity (MATH500)

**Logged:** `self_consistency_ambiguous_queries_total: 3` on 12 MATH500 queries (see `outputs/multi_action_oracle/math500_oracle_summary.json`). Per-query flags are in the CSV columns `self_consistency_3__ambiguous` and `self_consistency_3__tied_values`.

## 3. Label distributions (observed)

- **GSM8K hard tail (n=15):** `best_accuracy_action` = `reasoning_greedy` for all 15 rows; utility labels identical for λ ∈ {0, 0.10, 0.25} because cheaper action wins all utility ties when correctness is equal.
- **MATH500 (n=12):** Same degeneracy: single winning action `reasoning_greedy` for all labels. `tie_counts_best_accuracy` = 12 (all four actions tied on correctness each time).

## 4. Which actions “win” on which datasets

From oracle summaries:

| Dataset | Per-action accuracy | Best-accuracy wins (tie-break) |
|---------|---------------------|--------------------------------|
| gsm8k_hard (15) | All ≈ **0.933** | `reasoning_greedy` (15/15) |
| math500 (12) | rg/dpr/rtr ≈ **0.333**, sc3 **0.25** | `reasoning_greedy` (12/12) — *note: sc3 strictly worse on 3 queries (ambiguous vote)* but still tied “accuracy rank” with others on 9; check per-row CSV for `self_consistency_3__correct`. |

**MATH500 vs GSM8K:** MATH500 is much harder (~33% accuracy vs ~93%) for this model, but **within each slice** the four-action **ranking by correctness was identical** query-by-query in the aggregate summary, so the learned label never switched to revise or self-consistency.

## 5. Cost-aware labels (λ)

Oracle mean utility (choosing best utility action per query):

- **gsm8k_hard:** λ=0 → 0.933; λ=0.10 → 0.833; λ=0.25 → 0.683 (all actions tied on correctness → always pick cost 1).
- **math500:** λ=0 → 0.333; λ=0.10 → 0.233; λ=0.25 → 0.083.

**Conclusion:** On these runs, cost-aware λ>0 **did not change** the selected action vs accuracy-only, because correctness was flat across actions on each query (except where self_consistency_3 differed — still not enough to flip the argmax after tie-breaking in the winning set).

## 6. How to reproduce

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=...

# GSM8K tail (hard proxy)
python3 scripts/run_build_multi_action_dataset.py --dataset gsm8k_hard --subset-size 100

# MATH500
python3 scripts/run_build_multi_action_dataset.py --dataset math500 --subset-size 100

# Train / evaluate (needs ≥2 classes per label for full metrics)
python3 scripts/run_multi_action_model_eval.py \
  --csv data/multi_action_routing_gsm8k_hard.csv --dataset-name gsm8k_hard
```

## 7. Optional: original GSM8K 100

Use `--dataset gsm8k100 --subset-size 100` on the builder script (first 100 test problems, not the tail).

---

**Bottom line:** Pipeline is **end-to-end operational** with real API data on disk (gitignored). **Scientific comparison** of learned multi-action vs binary revise and vs fixed baselines **requires non-degenerate oracle labels** (larger or harder slices where actions disagree).
