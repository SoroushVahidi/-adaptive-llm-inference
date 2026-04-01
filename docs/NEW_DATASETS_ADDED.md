# New Public Reasoning Datasets Added

This document records the four newly added public reasoning datasets and how
they are stored in a GitHub-safe, offline-agent-friendly way.

## Added Components

- Local-first dataset loaders:
  - `src/datasets/mmlu_pro.py`
  - `src/datasets/musr.py`
  - `src/datasets/strategyqa.py`
  - `src/datasets/bbh.py`
- Deterministic dataset builders:
  - `scripts/build_mmlu_pro_dataset.py`
  - `scripts/build_musr_dataset.py`
  - `scripts/build_strategyqa_dataset.py`
  - `scripts/build_bbh_dataset.py`
- Offline smoke script:
  - `scripts/run_new_dataset_smoke.py`
- Focused loader tests:
  - `tests/test_mmlu_pro_loader.py`
  - `tests/test_musr_loader.py`
  - `tests/test_strategyqa_loader.py`
  - `tests/test_bbh_loader.py`

## Committed Data Artifacts (All GitHub-Safe)

| Path | Rows | Size (bytes) | Committed | Notes |
|---|---:|---:|---|---|
| `data/mmlu_pro_normalized.jsonl` | 12032 | 11141634 | Yes | Full normalized file |
| `data/mmlu_pro_sample.jsonl` | 64 | 48189 | Yes | Smoke-test sample |
| `data/musr_normalized.jsonl` | 756 | 3799620 | Yes | Full normalized file |
| `data/musr_sample.jsonl` | 64 | 378240 | Yes | Smoke-test sample |
| `data/strategyqa_normalized.jsonl` | 2290 | 1238994 | Yes | Full normalized file |
| `data/strategyqa_sample.jsonl` | 64 | 34805 | Yes | Smoke-test sample |
| `data/bbh_normalized.jsonl` | 6511 | 4304416 | Yes | Full normalized file (all tasks) |
| `data/bbh_sample.jsonl` | 64 | 19842 | Yes | Smoke-test sample |

No generated file is near GitHub's 100 MB hard cap; all are well within normal
comfort range for text-based repository artifacts.

## Unified Normalized Schema

Each JSONL row follows the shared schema:

- `dataset`
- `question_id`
- `question`
- `options` (`null` when not multiple-choice)
- `answer`
- `answer_format` (`multiple_choice`, `boolean`, `text`)
- `category` / `task` where applicable
- `source_split`
- `metadata`

Dataset-specific normalization:

- **MMLU-Pro:** multiple-choice, preserves category/source metadata.
- **MuSR:** multiple-choice, preserves subtask (`murder_mysteries`,
  `object_placements`, `team_allocation`) in `category` and `metadata.subtask`.
- **StrategyQA:** answer normalized to lowercase `true`/`false`; decomposition
  fields (`term`, `description`, `facts`) preserved in metadata.
- **BBH:** includes `task` at top-level and in metadata; target answer is
  normalized text for robust comparison.

## Why This Design Is Agent-Friendly

- Local-first loaders work without internet using committed files.
- Deterministic subset selection supports `max_samples` plus `seed`.
- Builders can fully regenerate artifacts when internet is available.
- Smoke tests validate local loading and schema integrity offline.
