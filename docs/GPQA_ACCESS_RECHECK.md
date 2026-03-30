# GPQA access recheck (after Hub access approval)

Date: 2026-03-29.  
Environment: same Cursor workspace; **no** model benchmarks; **no** loader code changes.

## 1. Token / account (secret not printed)

| Variable | Present |
|----------|---------|
| `HF_TOKEN` | yes (length 37) |
| `HUGGINGFACE_HUB_TOKEN` | no |

`huggingface_hub.whoami(token=HF_TOKEN)` → account name **`SoroushVahidi`**.

## 2. Official dataset `Idavidrein/gpqa`

### Commands run

```python
from datasets import get_dataset_config_names, load_dataset

get_dataset_config_names("Idavidrein/gpqa")
load_dataset("Idavidrein/gpqa")  # no config
load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train[:2]")
```

### Results

| Step | Outcome |
|------|---------|
| `get_dataset_config_names` | **OK** — configs: `gpqa_extended`, `gpqa_main`, `gpqa_diamond`, `gpqa_experts` |
| `load_dataset("Idavidrein/gpqa")` without config | **Fails** with `ValueError`: *Config name is missing. Please pick one among the available configs: [...]* (not a gating error) |
| Per-config load + `train` split | **OK** for question configs; row counts: **gpqa_extended 546**, **gpqa_main 448**, **gpqa_diamond 198**, **gpqa_experts 60** (experts table has different columns) |
| `load_dataset(..., "gpqa_diamond", split="train[:2]")` | **OK**, `len == 2` |

**Conclusion:** Gating is **resolved** for this token/account; remaining requirement is to **always pass a config name**.

## 3. GPQA Diamond (multiple choice) on official Hub

- **Config:** **`gpqa_diamond`**
- **Split:** **`train`** (only split observed for this config in `datasets`)
- **Rows:** **198**
- **Columns:** 78 fields; the four options are always in dedicated columns:
  - **`Question`** — stem (sometimes includes inline `a)`…`d)` lines; often **no** `(A)`…`(D)` block—options are only in the columns below)
  - **`Correct Answer`**, **`Incorrect Answer 1`**, **`Incorrect Answer 2`**, **`Incorrect Answer 3`** — the four choices as text (correct is always the first column in this schema)
- **Gold for normalization:** index **0** into `(Correct, Incorrect 1–3)`; see §5 for semantics vs exam letter order.

## 4. Fallback `hendrydong/gpqa_diamond_mc` (still works)

```python
load_dataset("hendrydong/gpqa_diamond_mc", split="test[:2]")
```

- **Split:** `test`
- **Rows (full):** 198
- **Columns:** `problem`, `solution`, `domain`
- **Spot check:** same opening text as official **`Question`**; **`solution`** was **`\\boxed{D}`** (letter-in-box), vs official **`Correct Answer`** as **answer string** (e.g. `10^-4 eV`).

## 5. Repo integration (official-first, mirror optional)

**Implemented:** `src/datasets/gpqa.py` loads **`Idavidrein/gpqa`**, **`gpqa_diamond`**, **`train`** by default. You must pass that **config name** (`load_dataset("Idavidrein/gpqa")` without config still raises `ValueError`).

### Official-only normalization (default path)

When `prefer_official=True` and `Idavidrein/gpqa` loads successfully, **`load_gpqa_diamond_mc` does not call `load_dataset` on the mirror**—only the official split is read.

The Hub schema is sufficient **without** the mirror:

| Field | Role |
|-------|------|
| `Question` | Prompt stem (often without `(A)`…`(D)` lines) |
| `Correct Answer` | Gold option text (first of four) |
| `Incorrect Answer 1` … `3` | Three distractors |

Normalized output:

- `choices = (Correct Answer, Incorrect 1, Incorrect 2, Incorrect 3)`
- `answer = 0` **always** on the official path (gold is index 0)

This is **not** the same as “which letter was (A) on the exam”: the release does not shuffle letters in the CSV. For letter-randomized evaluation, shuffle `choices` in experiment code and remap `answer` with a fixed seed.

### Fallback mirror

`hendrydong/gpqa_diamond_mc` / `test` is used **only** if the official dataset fails to load (`prefer_official=False` or Hub error). It parses `(A)`…`(D)` and `\\boxed{letter}`; `answer` is then 0–3 in **A–D presentation order**.

### Row count and schema drift

The loader requires **exactly 198** rows on both splits. If the count changes, normalization **raises** (do not silently continue).

### Optional alignment audit (tests)

`verify_official_mirror_dataset_pair(official, mirror)` loads nothing by itself; pass two `datasets` splits. It asserts equal length, prefix alignment of `Question` vs mirror stem, that the official correct string appears among mirror options, and that the mirror `\\boxed{}` letter matches that slot. Use when both Hub sources are reachable to detect **row reordering** or content drift between releases.

**Cached export:** `data/gpqa_diamond_normalized.jsonl` (198 lines) — tracked via `!data/gpqa_diamond_normalized.jsonl` in `.gitignore`. Regenerate with `write_normalized_gpqa_jsonl()`.

## 6. If access had still failed

N/A — access **succeeded**. Typical residual causes would be wrong token, approval lag, or using `load_dataset` without a config (misread as “broken”).
