# Cursor / automation: API tokens and this repository

**Generated:** 2026-03-30  
**Scope:** Grounded to this repo’s code and a **Cursor agent shell** check on this machine.

---

## 1. How this repo expects secrets (codebase audit)

### Variables used

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | OpenAI Chat Completions (`src/models/openai_llm.py`, `src/models/llm_model.py`) |
| `OPENAI_BASE_URL` | Optional API base (`openai_llm.py`, `llm_model.py`) |
| `HF_TOKEN` | Hugging Face Hub (datasets, gated data; see `src/datasets/gpqa.py` docs) |
| `HUGGINGFACE_HUB_TOKEN` / `HUGGING_FACE_HUB_TOKEN` | Alternative names supported by Hub libraries in some contexts; **this repo’s docs standardize on `HF_TOKEN`** in `.env.example` |

### Mechanisms searched

| Mechanism | Finding |
|-----------|---------|
| **`os.getenv` / `os.environ`** | **Yes** — primary pattern for scripts and models (no automatic `.env` until the change below). |
| **`load_dotenv` / `python-dotenv`** | **Added** — `src/utils/repo_env.py` calls `load_dotenv(<repo>/.env, override=False)` when `python-dotenv` is installed (now a **direct dependency** in `pyproject.toml`). |
| **`.env` file** | Documented in `.env.example`; **listed in `.gitignore`** (`.env`, `.env.*`, with `!.env.example`). |
| **Automatic load before this change** | **No** — Python entrypoints did **not** load `.env` unless you used `export $(grep …)` or `bash scripts/with_dotenv.sh`. |

### Where loading now happens

- **`src/utils/repo_env.py`** — `try_load_repo_dotenv()` (idempotent, `override=False`).
- **`src/models/openai_llm.py`** — calls `try_load_repo_dotenv()` at import time so any code path that uses `OpenAILLMModel` sees variables from `.env`.
- **`scripts/run_build_real_routing_dataset.py`** — calls `try_load_repo_dotenv()` immediately after `sys.path` setup, **before** `OPENAI_API_KEY` checks.
- **`scripts/test_openai_key.py`** — same, so probing the API works without a prior manual `export`.

### Helper script (still valid)

- **`scripts/with_dotenv.sh`** — `source .env` then `exec "$@"`. Use when you want **shell-level** loading without relying on Python (e.g. non-Python tools).

---

## 2. Cursor agent process environment (this checkout, diagnostic shell)

Checked with a **non-printing** shell probe (names, set/unset, length, masked preview only):

| Variable | Present & non-empty? | Masked preview (if set) |
|----------|----------------------|-------------------------|
| `OPENAI_API_KEY` | **No** (unset in the agent shell) | — |
| `HF_TOKEN` | **Yes** | `hf_Y***cL` |
| `HUGGINGFACE_HUB_TOKEN` | **No** | — |
| `HUGGING_FACE_HUB_TOKEN` | **No** | — |

**Interpretation:** The integrated agent shell **does not** automatically inherit your personal terminal `export` or Cursor “account” secrets. A **project `.env`** (or explicit wrapper) is the portable way to supply `OPENAI_API_KEY` to commands the agent runs.

---

## 3. Project `.env` on disk

| Check | Result (this workspace) |
|-------|-------------------------|
| **`.env` exists?** | **Yes** (~1095 bytes at repo root). |
| **Gitignored?** | **Yes** — see `.gitignore` entries for `.env` / `.env.*` with `!.env.example`. |
| **Auto-loaded by Python before this work?** | **No** (documented gap). |
| **Auto-loaded now?** | **Yes**, via `try_load_repo_dotenv()` when dependencies are installed (`pip install -e .` pulls in `python-dotenv`). |

**Never commit `.env`.** Only commit `.env.example`.

---

## 4. Safest setup for automatic experiment runs

### Recommended (this repository)

1. **One-time:** `cp .env.example .env` and set real values for `OPENAI_API_KEY` and (optional) `HF_TOKEN`.
2. **Install deps:** `pip install -e ".[dev]"` (installs `python-dotenv` via the project).
3. **Run experiments** as usual: `python scripts/run_build_real_routing_dataset.py …` — **no per-command `export`** required for keys that live in `.env`, provided the process imports code that triggers `try_load_repo_dotenv()` early enough.

### Shell-only alternative

- `export $(grep -v '^#' .env | xargs)` in the **same** terminal before running commands (does not help the agent unless that shell is what runs the tool).

### Wrapper alternative (no Python)

- `bash scripts/with_dotenv.sh python3 scripts/test_openai_key.py` — still supported.

### Do you need more code changes?

- **Optional:** For scripts that **never** import `src.models.openai_llm` and **never** call `try_load_repo_dotenv`, add one import at the top of that script if you want `.env` without `with_dotenv.sh`. The main experiment driver (`run_build_real_routing_dataset.py`) is already wired.

### Scripts that fail without env

- Anything that checks `os.getenv("OPENAI_API_KEY")` **before** any import that loads `.env` could still fail. The build script and `OpenAILLMModel` path are covered; if you add a new top-level script, follow the same pattern as `run_build_real_routing_dataset.py`.

---

## 5. What you should do right now

### One-time setup

```bash
cd /path/to/adaptive-llm-inference
cp .env.example .env
# Edit .env: set OPENAI_API_KEY=sk-... and optionally HF_TOKEN=hf_...
pip install -e ".[dev]"
```

### Per-terminal setup (only if you refuse `.env` auto-load)

```bash
export OPENAI_API_KEY='sk-...'
export HF_TOKEN='hf-...'
```

### Automatic / persistent setup

- **Project `.env`** (recommended) — gitignored, loaded by `try_load_repo_dotenv()` and by `scripts/with_dotenv.sh`.
- **User shell profile** (`~/.bashrc`) — `export` only if you accept machine-wide secrets (not recommended for shared machines).

### Verify (does not print your key)

```bash
python3 scripts/test_openai_key.py
```

Expect HTTP **200** from `/v1/models` if the key is valid. A **401** means the key was read but rejected by OpenAI (wrong/revoked key), not a loading problem.

---

## 6. Summary table

| Question | Answer |
|----------|--------|
| Does the repo read secrets only from the environment? | **Historically yes**; **now** `.env` is loaded early in key paths via `python-dotenv`. |
| Is `.env` gitignored? | **Yes.** |
| Does Cursor’s agent shell see your manual `export` in another tab? | **No** — use `.env` in the repo or `with_dotenv.sh`. |
| Smallest fix applied in-repo? | **`python-dotenv` dependency** + **`src/utils/repo_env.py`** + hooks in **`openai_llm.py`**, **`run_build_real_routing_dataset.py`**, **`test_openai_key.py`**. |

---

## 7. Rules reminder

- Do **not** commit `.env` or paste full tokens into git-tracked files.
- `override=False` means a **bad** key in `.env` will **not** override a correct key already in the environment (e.g. CI).
