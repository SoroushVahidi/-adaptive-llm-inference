# Updated recent baselines and datasets report

This document summarizes what was implemented and what was run for the **adaptive test-time compute / routing** project (EAAI-oriented). It accompanies the evaluation code in `src/evaluation/recent_baselines_eval.py` and the runner `scripts/run_recent_baselines_experiment.py`.

## 1. What was successfully implemented

- **Unified static baselines** (same cost proxy as elsewhere: one model call = 1.0):
  - `reasoning_greedy` — one CoT-style sample; numeric vs math extraction by dataset mode.
  - `direct_plus_revise` — direct answer then revise (GSM8K-style numeric path); **math mode** uses a boxed-answer revise prompt and `extract_math_answer` / `normalize_math_answer`.
  - `reasoning_then_revise` — reasoning first, then a second pass that sees the full first response and may correct it.
  - `self_consistency_3` / `self_consistency_5` — multiple reasoning samples, majority vote with deterministic tie-break (sorted order); **ambiguous ties** are logged in per-query metadata (`self_consistency_ambiguous`, `self_consistency_tied_answers`).
  - `always_most_expensive` — alias of self-consistency-5 for the static ladder (same as five-sample vote).
- **Oracle summaries** (`*_oracle_summary.json`):
  - Binary oracle: best of `reasoning_greedy` vs `direct_plus_revise` (cheapest on ties).
  - Multi-action oracle: best among the five actions above (cheapest on ties).
  - Cost-aware utility oracle: maximize `correctness − λ × cost` for λ ∈ {0, 0.1, 0.25}.
- **Lightweight routing baselines** (`*_routing_baseline_summary.json`):
  - **A** — difficulty / parse-failure threshold router → `direct_plus_revise`.
  - **B** — output-aware score (confidence proxy, tail extraction mismatch, short output heuristic) → `self_consistency_3`.
  - **C** — BEST-Route-**inspired** small ladder (`reasoning_greedy` → `reasoning_then_revise` → `self_consistency_3`) from difficulty + confidence.
  - The full **BEST-Route** paper system was **not** reproduced (no official controller code wired here); only the simple inspired baseline above.
- **Datasets**:
  - **Hard GSM8K**: `src/datasets/hard_gsm8k.py` — deterministic top-*k* subset by an offline hardness score over query features (full test split ranked by default).
  - **AIME-style**: `src/datasets/aime2024.py` — loads `math-ai/aime24` (public) into the shared `Query` format; gold from `\boxed{...}` in solutions.
- **`BaselineResult.metadata`**: optional dict for per-query diagnostics (revise-help flags, ambiguity, vote counts).

## 2. What datasets were successfully run (this workspace run)

End-to-end run completed with **OpenAI** and **HuggingFace** access. **Slice sizes for wall-clock** (defaults in script are larger; this run used):

| Dataset      | Queries | Notes |
|-------------|---------|--------|
| GSM8K slice | 20      | Filenames use `gsm8k{N}_*` where `N` is `--gsm8k-slice` (e.g. `gsm8k100_*` when `N=100`). |
| Hard GSM8K  | 20      | Use `--hard-k 100` for top-100 hard subset. |
| MATH500     | 20      | Use `--math500-max 500` for full benchmark. |
| AIME 2024   | 8       | Source: `math-ai/aime24` on HuggingFace; use `--aime-max` to cap. |

Artifacts (gitignored): `outputs/recent_baselines/*.json`, `final_cross_dataset_baseline_summary.csv`, `final_dataset_rollup.csv`.

## 3. What recent baselines were successfully run

All static ladder baselines and oracle computations ran on every **non-blocked** dataset above. **Routing sweeps were not run by default** in the final script (they multiply API calls). To run them on a capped subset:

```bash
python3 scripts/run_recent_baselines_experiment.py --with-routing --routing-max-queries 40
```

## 4. What failed and exact blockers

### GPQA Diamond

- **Source attempted:** HuggingFace `Idavidrein/gpqa`, config `gpqa_diamond`, split `train`.
- **Error type:** `DatasetNotFoundError` (gated dataset).
- **Exact message:** `Dataset 'Idavidrein/gpqa' is a gated dataset on the Hub. You must be authenticated to access it.`
- **Missing resource:** HuggingFace authentication / dataset agreement for the gated repo.
- **How to fix:** Accept the dataset terms on the Hub, then `huggingface-cli login` with a token that has access, and rerun the loader or extend `run_recent_baselines_experiment.py` with a GPQA branch once access works.

### Prior aborted runs (timeout / operator stop)

- An earlier attempt used **`--skip-routing`** while the script still treated “skip” inconsistently with partial writes; the CLI was changed so **routing is off unless `--with-routing`** is passed, avoiding accidental huge sweep cost.

## 5. Cross-dataset comparison (empirical snapshot, this run)

See `outputs/recent_baselines/final_dataset_rollup.csv` and `final_cross_dataset_baseline_summary.csv`. Highlights:

| Dataset     | One-shot (reasoning_greedy) | Best static (name / acc / cost proxy) | Multi oracle acc / cost |
|------------|-----------------------------|----------------------------------------|-------------------------|
| gsm8k20    | 0.90                        | direct_plus_revise / 0.95 / 2.0        | 0.95 / 1.05             |
| hard_gsm8k | 0.90                        | reasoning_greedy / 0.90 / 1.0          | 0.90 / 1.00             |
| math500    | 0.55                        | self_consistency_3 / 0.75 / 3.0      | 0.75 / 1.30             |
| aime2024   | 0.125                       | reasoning_greedy / 0.125 / 1.0         | 0.125 / 1.00            |

**Self-consistency vs revise (MATH500, n=20):** `self_consistency_3` (0.75) **beats** `direct_plus_revise` (0.65) and `reasoning_then_revise` (0.60); extra compute is reflected in cost proxy 3.0 vs 2.0.

## 6. Hard GSM8K vs routing idea

On this **small** slice, one-shot accuracy on the hard subset **did not** fall below the easy slice (both 0.90), so the validation JSON records `hard_is_lower_accuracy: false`. **However**, average hardness **proxies** (numeric range, question length, currency cues) are higher on the hard subset — see `outputs/recent_baselines/hard_gsm8k_validation_summary.json`. **Revise-helpful rate** (direct wrong → revise correct) was slightly higher on hard (0.40) than easy (0.35), which is **directionally** consistent with more “corrective headroom” on harder items, but the sample is too small for strong claims.

**Conclusion:** Keep hard GSM8K as a regime probe, but validate on **larger N** and align the easy comparator slice with the same index protocol if you need a strict difficulty gap in accuracy.

## 7. MATH500 as an action-mismatch regime

Multi-action oracle uses **`self_consistency_3` on 2/20 queries** and `direct_plus_revise` on 2/20, while most queries stay at `reasoning_greedy` under the tie-breaking rule. That pattern is consistent with a regime where **some** queries benefit from stronger actions, but the cheap action is often enough — i.e. **heterogeneous** action value, which is what routing methods target.

## 8. AIME and the hypothesis

On **8** AIME problems, **all** static baselines stayed at **0.125** accuracy except tied self-consistency runs that duplicate the same wrong mode; **oracle** cannot exceed one-shot because no action achieved a correct parsed match in the logged outputs. **High self-consistency ambiguity rates** (many tied votes) indicate **aggregation / normalization** stress for this model and prompt.

**Interpretation:** This supports the hypothesis that **if the expensive action family does not yield correct extractions**, adaptive routing has **no signal** to exploit; AIME here looks **blocked by capability/format** more than by routing structure.

## 9. Self-consistency vs revise

- **MATH500 (this run):** self-consistency-3 **>** revise baselines (see section 5).
- **GSM8K slice:** direct+revise **>** one-shot and matched or beat self-consistency on accuracy at lower or comparable cost — check per-dataset CSV rows.

## 10. Baselines to prioritize for the EAAI paper

Strong candidates to report as **modern fixed-cost** and **upper-bound** references:

1. `reasoning_greedy` (cheap one-shot reasoning).
2. `direct_plus_revise` (sequential self-correction; strong on GSM8K-style numeric word problems).
3. `self_consistency_3` / `self_consistency_5` (standard test-time compute baseline).
4. `reasoning_then_revise` (reasoning-aware correction — distinct from direct-first revise).
5. **Oracle** (binary and multi-action) to show **routing headroom** under known per-query outcomes.
6. **Lightweight threshold routers** (optional) as simple **training-free** routing competitors when run with `--with-routing`.

## Reproduction

```bash
pip install -e ".[dev]"
export OPENAI_API_KEY=...
python3 scripts/run_recent_baselines_experiment.py \
  --output-dir outputs/recent_baselines \
  --gsm8k-slice 100 --hard-k 100 --math500-max 500 --aime-max 30
# Optional routing sweeps (expensive):
#   --with-routing --routing-max-queries 50
```

GPQA requires HF gated access; the script records the blocker in `outputs/recent_baselines/_blockers.json` and adds a rollup row for `gpqa_diamond` when probe fails.
