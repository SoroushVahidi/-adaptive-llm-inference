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

## 5. Repo impact (this checkout)

**No GPQA loader** exists in the Python tree yet (grep only hits docs). A future integration is **not** a drop-in column rename:

- Official: `load_dataset("Idavidrein/gpqa", "gpqa_diamond", split="train")`, map `Question` → prompt, gold from **`Correct Answer`** (and/or parse MC letters from `Question` if the eval expects A/B/C/D).
- Fallback: `load_dataset("hendrydong/gpqa_diamond_mc", split="test")`, gold often **`solution`** (e.g. `\\boxed{letter}`).

**Minimal change when adding support:** new dataset module or config entry with `hub_id="Idavidrein/gpqa"`, `config="gpqa_diamond"`, `split="train"`, plus a small **normalization** step so evaluation uses one gold format (string match vs letter).

## 6. If access had still failed

N/A — access **succeeded**. Typical residual causes would be wrong token, approval lag, or using `load_dataset` without a config (misread as “broken”).
