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
- **Columns:** 78 fields; MC stem/answers use at least:
  - **`Question`** — full multiple-choice prompt (options embedded in text)
  - **`Correct Answer`** — **string** (e.g. numeric/scientific answer text, **not** a letter A–D in the spot-checked row)
  - **`Incorrect Answer 1`**, **`Incorrect Answer 2`**, **`Incorrect Answer 3`**
- **Correct answer representation:** explicit **`Correct Answer`** column vs three incorrect columns; distractors and wording live in **`Question`** as well.

## 4. Fallback `hendrydong/gpqa_diamond_mc` (still works)

```python
load_dataset("hendrydong/gpqa_diamond_mc", split="test[:2]")
```

- **Split:** `test`
- **Rows (full):** 198
- **Columns:** `problem`, `solution`, `domain`
- **Spot check:** same opening text as official **`Question`**; **`solution`** was **`\\boxed{D}`** (letter-in-box), vs official **`Correct Answer`** as **answer string** (e.g. `10^-4 eV`).

## 5. Repo integration (updated)

**Implemented:** `src/datasets/gpqa.py` loads **`Idavidrein/gpqa`** with **`gpqa_diamond`** / **`train`** by default. You must pass that **config name** (`load_dataset("Idavidrein/gpqa")` without config still raises `ValueError`).

**Normalization:** The official `Question` field often lacks a clean `(A)`…`(D)` block (and `Correct Answer` may be a short digit vs full option text). The loader therefore **also reads** `hendrydong/gpqa_diamond_mc` when the official set loads, to recover the final A–D option block and `\\boxed{letter}` gold; it **permutes** mirror options to match official `Correct Answer` / incorrect columns and checks the boxed letter. If the official Hub load fails entirely, it falls back to **mirror-only** parsing.

**Cached export:** `data/gpqa_diamond_normalized.jsonl` (198 lines) — gitignored except for an explicit `!data/gpqa_diamond_normalized.jsonl` entry. Regenerate with `write_normalized_gpqa_jsonl()` or `python3 -c "from src.datasets.gpqa import write_normalized_gpqa_jsonl; write_normalized_gpqa_jsonl()"`.

## 6. If access had still failed

N/A — access **succeeded**. Typical residual causes would be wrong token, approval lag, or using `load_dataset` without a config (misread as “broken”).
