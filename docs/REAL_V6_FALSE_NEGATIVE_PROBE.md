# Real OpenAI + V6 scoring probe

**Purpose:** Replace hand-simulated traces for a **small** slice of the false-negative discussion with **one real `reasoning_greedy`-style completion per problem**, then run the **existing** `compute_v6_scores` path (`AdaptivePolicyV6Config` defaults).

**Evidence label:** `measured_now` for the run that produced `results/real_v6_false_negative_probe/` (regenerate locally; `results/` is gitignored).

---

## 1. What was run

- **Script:** `scripts/run_real_v6_false_negative_probe.py`
- **Data:** Seven rows from `src/datasets/bundled/gsm8k_test_sample.json` by id:
  - `gsm8k_test_0` (Natalia), `gsm8k_test_1` (Weng), `gsm8k_test_18` (apples), `gsm8k_test_13` (Jasmine/paperclips), `gsm8k_test_7` (Ken care package), `gsm8k_test_11` (Tobias shoes), `gsm8k_test_12` (Randy trees)
- **Model call:** Same pattern as `run_reasoning_greedy` in `oracle_subset_eval.py` — `OpenAILLMModel` with reasoning system prefix, **temperature 0**, **one** `generate()` per question.
- **Scoring:** `compute_v6_scores(question, raw_output)`; correctness = `v6_parsed_answer` vs gold via `_normalize` / casefold (weekday).

## 2. Model used

- **`gpt-4o-mini`** (default; override with `--model` or `REAL_V6_PROBE_MODEL`)

## 3. Number of examples

- **7**

## 4. How many answers were wrong

- **2** (`gsm8k_test_13`, `gsm8k_test_11`)

## 5. How many wrong answers V6 “missed” (no revise)

- **2** — both wrong cases had **`revise_recommended: false`**.

Definitions used in the script:

- **Wrong:** parsed final (V6 / `extract_math_answer` path) does not match gold.
- **Missed:** wrong **and** `revise_recommended` is false.

## 6. Qualitative analysis of missed cases

### `gsm8k_test_11` (Tobias / shoes)

- **Gold:** 65  
- **Model:** Coherent multi-step trace concluding **95** (incorrect intermediate accounting of savings vs spend).  
- **V6:** `explanation_warning_score = 0`, `answer_error_score = 0`, `final_answer_confident = true` → **no revise**.  
- **Pattern:** Same **methodological** story as the doc’s simulated cases: **internally fluent wrong arithmetic** with **no** constraint hit and **no** `evaluate_candidate` flag on the final number.

### `gsm8k_test_13` (Jasmine / paperclips)

- **Gold:** Sunday  
- **Model:** Trace argues **Sunday** correctly, but the last line is **`Final answer: 7`** (likely “day index” confusion).  
- **V6:** Parsed answer **7**; `final_answer_confident = false` (numeric final on a categorical “which day” question) but **`answer_error_score` still 0** and **`explanation_warning_score` 0** → combo rule does not fire → **no revise**.  
- **Pattern:** **Format / extraction failure** at the final line: wrong token, **low trust**, but **not enough** explanation pressure or answer-error mass to trigger revise.

## 7. Does `answer_error_score = 0` + trust still dominate on real outputs?

**Yes, when the wrong answer is a plausible integer and constraints stay quiet.** The Tobias row is the clearest match to the **“answer_error_score = 0 → primary trust path”** issue from `docs/V6_FALSE_NEGATIVE_ANALYSIS.md`.

The Jasmine row shows a **second** gap: **low `final_answer_confident`** without **high `explanation_warning_score`** still yields **no revise** under current combo thresholds.

---

## Artifacts

| File | Content |
|------|---------|
| `results/real_v6_false_negative_probe/raw_responses.jsonl` | One JSON object per line: question, raw output, V6 fields |
| `results/real_v6_false_negative_probe/scored_results.csv` | Flattened table |
| `results/real_v6_false_negative_probe/summary.json` | Counts and metadata |

---

## How to reproduce

```bash
export OPENAI_API_KEY=...   # never commit
python3 scripts/run_real_v6_false_negative_probe.py
```

Optional: `--model gpt-4o-mini --max-tokens 512 --output-dir results/real_v6_false_negative_probe`

---

## Approximations

- **Tiny N (7):** not a rate estimate.  
- **Bundled JSON only** (no full GSM8K download).  
- **Correctness** uses V6’s parsed final (`extract_math_answer` / numeric path); oracle row’s `extract_numeric_answer` may differ slightly on edge outputs (both stored in JSONL for inspection).
