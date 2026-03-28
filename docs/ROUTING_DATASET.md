# Routing Dataset

## What Is the Routing Dataset?

The routing dataset is the bridge between raw queries, lightweight features, and
oracle strategy labels.  It is a single flat CSV where every row represents one
query and every column captures either a query-level feature or an oracle-derived
label.

Its purpose is to enable future offline work:

- **Rule-based routing** — e.g. *if `has_multi_step_cue` and
  `num_numeric_mentions > 3`, escalate to `reasoning_best_of_3`*.
- **Supervised routing** — train a tiny classifier mapping z(x) → best strategy.
- **Error analysis** — understand which query properties predict oracle failures.

## How It Connects x, z(x), and Oracle Labels

```
query text  x
    │
    ├── extract_query_features(x)  ──────────────► z_query  (13 features)
    │                                               (always available)
    │
    ├── [optional] extract_first_pass_features(...)► z_fp    (6 features)
    │                                               (requires one model pass)
    │
    └── [optional] oracle_assignments.csv  ────────► y       (5 label cols)
                   per_query_matrix.csv              (requires oracle eval run)
                        │
                        ▼
              routing_dataset.csv  (one row per query, all columns flat)
```

## Output Schema

| Column | Source | Always present? |
|---|---|---|
| `question_id` | query | ✓ |
| `question_text` | query | ✓ |
| `question_length_chars` | z(x) | ✓ |
| `question_length_tokens_approx` | z(x) | ✓ |
| `num_numeric_mentions` | z(x) | ✓ |
| `num_sentences_approx` | z(x) | ✓ |
| `has_multi_step_cue` | z(x) | ✓ |
| `has_equation_like_pattern` | z(x) | ✓ |
| `has_percent_symbol` | z(x) | ✓ |
| `has_fraction_pattern` | z(x) | ✓ |
| `has_currency_symbol` | z(x) | ✓ |
| `max_numeric_value_approx` | z(x) | ✓ |
| `min_numeric_value_approx` | z(x) | ✓ |
| `numeric_range_approx` | z(x) | ✓ |
| `repeated_number_flag` | z(x) | ✓ |
| `first_pass_parse_success` | z_fp | empty when unavailable |
| `first_pass_output_length` | z_fp | empty when unavailable |
| `first_pass_has_final_answer_cue` | z_fp | empty when unavailable |
| `first_pass_has_uncertainty_phrase` | z_fp | empty when unavailable |
| `first_pass_num_numeric_mentions` | z_fp | empty when unavailable |
| `first_pass_empty_or_malformed_flag` | z_fp | empty when unavailable |
| `best_accuracy_strategy` | oracle | empty when unavailable |
| `cheapest_correct_strategy` | oracle | empty when unavailable |
| `direct_already_optimal` | oracle | empty when unavailable |
| `oracle_any_correct` | oracle | empty when unavailable |
| `num_strategies_correct` | oracle | empty when unavailable |
| `oracle_label_available` | assembler | ✓ (True / False) |

## Operating Modes

### Full mode (default)

Oracle CSV files are present — the assembler merges labels from:
- `outputs/oracle_subset_eval/oracle_assignments.csv`
- `outputs/oracle_subset_eval/per_query_matrix.csv`

Every row has `oracle_label_available = True` and all label columns populated.

### Schema-only mode (dry-run / offline)

Oracle files are absent.  Label columns are left empty and every row has
`oracle_label_available = False`.  This mode is always available without any
API key or network access — it lets you inspect the feature schema and pipe it
into downstream tooling even before running the full oracle evaluation.

## What Happens When Oracle Outputs Are Missing?

The assembler never raises an error for missing oracle files.  Instead:

1. `load_oracle_files(dir)` returns an `OracleData` object with `.available = False`
   and `.missing_files` listing the absent paths.
2. `assemble_routing_dataset(queries, oracle_data=None)` fills all oracle columns
   with empty strings and sets `oracle_label_available = False`.
3. The summary JSON records `oracle_labels_available: false` and lists the
   missing inputs under `missing_optional_inputs`.

## Building the Dataset

```bash
# Schema-only (no oracle outputs required)
python3 scripts/build_routing_dataset.py --dry-run

# Full mode (default paths)
python3 scripts/build_routing_dataset.py

# Custom paths
python3 scripts/build_routing_dataset.py \
    --oracle-dir outputs/oracle_subset_eval \
    --output-dir outputs/routing_dataset \
    --max-queries 50
```

Outputs are written to `outputs/routing_dataset/`:
- `routing_dataset.csv` — flat per-query dataset
- `routing_dataset_summary.json` — metadata and column inventory

## Summary JSON Fields

```json
{
  "num_queries": 20,
  "oracle_labels_available": false,
  "num_feature_columns": 19,
  "num_label_columns": 5,
  "feature_columns": ["question_length_chars", ...],
  "label_columns": ["best_accuracy_strategy", ...],
  "source_files": [],
  "missing_optional_inputs": [
    "outputs/oracle_subset_eval/oracle_assignments.csv",
    "outputs/oracle_subset_eval/per_query_matrix.csv"
  ]
}
```

## Connection to Future Routing Work

Once oracle labels are available, the routing dataset can be used to:

1. **Hand-tune thresholds** — plot feature distributions conditioned on the
   oracle best strategy to find decision boundaries.
2. **Train a routing model** — use `best_accuracy_strategy` or
   `cheapest_correct_strategy` as the label and the z(x) columns as features.
3. **Evaluate routing policies** — compare the strategy selected by a rule or
   model against the oracle best strategy.

See `docs/PRECOMPUTATION_FEATURES.md` for a description of each z(x) feature.
