# GPQA access check (environment diagnostic)

Date: 2026-03-29.  
Scope: dataset / URL access only — no model benchmarks run.

## Token presence (secrets redacted)

| Variable | Present |
|----------|---------|
| `HF_TOKEN` | **yes** (length 37; prefix `hf_`) |
| `HUGGINGFACE_HUB_TOKEN` | **no** |

`huggingface_hub.whoami(token=HF_TOKEN)` succeeded for user **`SoroushVahidi`** (read token **`Wulver-read`**). Full token value was not printed or stored here.

---

## Commands run (summary)

### Environment / Hub identity

```bash
python3 -c "import os; ..."   # print presence + length + masked prefix/suffix only
```

```python
from huggingface_hub import whoami
whoami(token=os.environ["HF_TOKEN"])
```

### Official Hub dataset: `Idavidrein/gpqa`

```python
from datasets import load_dataset, get_dataset_config_names

load_dataset("Idavidrein/gpqa", split="train[:2]")  # also tried with trust_remote_code=True (datasets warns + still fails)
get_dataset_config_names("Idavidrein/gpqa")
```

**Result:** **Not loadable** with the current token.

- **Exception:** `datasets.exceptions.DatasetNotFoundError`
- **Message (faithful summary):** Dataset `Idavidrein/gpqa` is a **gated** dataset on the Hub; visit the dataset page to **ask for access**.

**Note:** `HfApi(token=...).dataset_info("Idavidrein/gpqa")` returned metadata (`private: False`), but **`datasets.load_dataset` still rejects** — consistent with **gating / no grant** for this account, not a missing dataset.

### Fallback Hub dataset (requested name vs actual)

Requested: **`hendrydong/gpqa_diamond_mcon`**

```python
from datasets import load_dataset, get_dataset_config_names
get_dataset_config_names("hendrydong/gpqa_diamond_mcon")
load_dataset("hendrydong/gpqa_diamond_mcon")
```

**Result:** **`DatasetNotFoundError`** — id does not exist on the Hub (typo / renamed).

**Working nearby id:** **`hendrydong/gpqa_diamond_mc`**

```python
from datasets import load_dataset, get_dataset_config_names
get_dataset_config_names("hendrydong/gpqa_diamond_mc")  # -> ['default']
ds = load_dataset("hendrydong/gpqa_diamond_mc")
```

**Result:** **OK**

- **Config:** `default`
- **Split:** `test` (only key in returned `DatasetDict`)
- **Rows:** **198**
- **Columns:** `['solution', 'problem', 'domain']`

### GitHub official repo (raw CSV + ZIP)

**Raw CSV** (common guess path):

```bash
curl -sS -o /dev/null -w "HTTP %{http_code}\n" \
  "https://raw.githubusercontent.com/idavidrein/gpqa/main/dataset.csv"
curl -sS -o /dev/null -w "HTTP %{http_code}\n" \
  "https://github.com/idavidrein/gpqa/raw/main/dataset.csv"
```

**Result:** **HTTP 404** for both. This is a **wrong/missing path**, not authentication; an **HF token does not affect** GitHub raw URLs.

**Repository ZIP** (`main` branch archive):

```bash
curl -sSL -o gpqa.zip "https://github.com/idavidrein/gpqa/archive/refs/heads/main.zip"
unzip -t gpqa.zip
```

**Result:** **HTTP 200**; **`unzip -t` reports all entries OK** — **not password-protected / not encrypted** in the usual zip sense.

The checkout contains **baseline_results/*.csv** etc.; there is **no** `dataset.csv` at repo root in that archive (so the raw URL failure is **link/layout**, not HF).

---

## Checklist (for quick scanning)

| Check | Outcome |
|-------|---------|
| Token present | **yes** (`HF_TOKEN`) |
| `Idavidrein/gpqa` accessible | **no** (gated; `load_dataset` fails) |
| `hendrydong/gpqa_diamond_mcon` accessible | **no** (dataset id not found) |
| `hendrydong/gpqa_diamond_mc` accessible | **yes** (198 rows, split `test`) |
| GitHub raw `dataset.csv` | **404** |
| GitHub ZIP | **downloadable; unencrypted** |

---

## Diagnosis

**HF token did not fix official Hub access; the blocker is still Hub gating** (account lacks accepted access to `Idavidrein/gpqa`, despite a valid read token).

**Separately, GitHub issues are not HF-related:** the tested raw CSV URL returns **404**; the **ZIP is fine** and unencrypted.

---

## Recommended source for this repo *right now*

1. **Practical public fallback:** use Hub id **`hendrydong/gpqa_diamond_mc`** (not `..._mcon`) for a **198-row** multiple-choice–style slice — verify column schema matches your loader (`problem`, `solution`, `domain`).
2. **Official full GPQA on Hub:** after the owning account is **granted access** on [the dataset page](https://huggingface.co/datasets/Idavidrein/gpqa), `load_dataset("Idavidrein/gpqa", ...)` should work with the same `HF_TOKEN` in the environment.

### Minimal code/config change *if* Hub access is granted later

- Set `HF_TOKEN` in the environment (already present here).
- Add a dataset loader config pointing `dataset_name` / hub id to **`Idavidrein/gpqa`** and the correct **config/split** names from `datasets.get_dataset_config_names("Idavidrein/gpqa")` once accessible (this check could not list configs while gated).

No repository code was changed for this diagnostic beyond adding this file.
