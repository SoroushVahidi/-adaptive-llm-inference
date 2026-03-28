# Precomputation Feature Layer

## Why z(x)?

Before committing to an expensive inference strategy (e.g. best-of-5 with
chain-of-thought), the system can compute a cheap feature vector z(x) from
the query alone.  This vector captures surface-level signals—question length,
numeric density, operator-like patterns, hedging keywords—that correlate with
problem difficulty and the likely benefit of extra compute.

In future work these features can train a lightweight router that maps z(x) →
strategy, replacing hand-tuned heuristics with a data-driven decision.

## Why Must Features Be Cheap?

The whole point of adaptive compute is to *save* resources on easy queries.
Any feature extraction that itself requires a model call would negate that
saving.  All features here are therefore:

- **Regex / string-based** — no tokenizer, no embeddings.
- **O(n) in query length** — fast on any commodity hardware.
- **Stateless** — no external I/O, no caching required.

## Query-Only Features (`extract_query_features`)

These are computed from the raw question string before any model inference.

| Feature | Type | Description |
|---|---|---|
| `question_length_chars` | int | Raw character count |
| `question_length_tokens_approx` | int | Whitespace-split token count |
| `num_numeric_mentions` | int | Count of integer/decimal tokens |
| `num_sentences_approx` | int | Sentence count (split on `.!?`) |
| `has_multi_step_cue` | bool | Keyword match: *total, remaining, after, left, difference, each, every, altogether, twice, half, percent, ratio, average, consecutive* |
| `has_equation_like_pattern` | bool | Inline arithmetic (`3 + 4 = 7`) present |
| `has_percent_symbol` | bool | `\d+\s*%` pattern present |
| `has_fraction_pattern` | bool | `\d+/\d+` pattern present |
| `has_currency_symbol` | bool | `$€£¥₹` present |
| `max_numeric_value_approx` | float | Largest numeric value in text |
| `min_numeric_value_approx` | float | Smallest numeric value in text |
| `numeric_range_approx` | float | max − min of all numeric values |
| `repeated_number_flag` | bool | Same token appears more than once |

## First-Pass-Output Features (`extract_first_pass_features`)

These are computed *after* one cheap inference pass (e.g. greedy decode) and
supplement the query features to decide whether escalation is worthwhile.
They require the original query, the raw model output, and an optional
pre-parsed answer string.

| Feature | Type | Description |
|---|---|---|
| `first_pass_parse_success` | bool | Non-empty parsed answer, or ≥1 numeric token in output |
| `first_pass_output_length` | int | Character length of model output |
| `first_pass_has_final_answer_cue` | bool | Output contains *final answer / therefore / the answer is / …* |
| `first_pass_has_uncertainty_phrase` | bool | Output contains *not sure / uncertain / it depends / …* |
| `first_pass_num_numeric_mentions` | int | Numeric token count in output |
| `first_pass_empty_or_malformed_flag` | bool | Output is empty or shorter than 3 characters |

## How This Supports Strategy Routing

```
query x
  │
  ▼
extract_query_features(x)  ──► z_query
  │
  ▼  (run one cheap greedy pass)
extract_first_pass_features(x, output, parsed)  ──► z_first_pass
  │
  ▼
z = concat(z_query, z_first_pass)
  │
  ▼
router(z)  ──► strategy ∈ {greedy, best-of-N, self-consistency, …}
```

In the current codebase, `z(x)` is not yet wired into the allocator or
strategy router.  The feature layer is provided as a standalone offline
module so that:

1. Features can be logged alongside experiment outputs for later analysis.
2. A supervised router can be trained offline once labelled data is available.
3. Simple hand-tuned rules (e.g. *if `has_multi_step_cue` and
   `num_numeric_mentions > 3`, escalate*) can be prototyped cheaply.

## Example

```python
from src.features import extract_query_features, extract_first_pass_features

q = "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast every day and bakes muffins for her friends every day with 4. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"

z_q = extract_query_features(q)
# {
#   'question_length_chars': 267,
#   'question_length_tokens_approx': 49,
#   'num_numeric_mentions': 5,
#   'num_sentences_approx': 4,
#   'has_multi_step_cue': True,        # 'every', 'remainder'
#   'has_equation_like_pattern': False,
#   'has_percent_symbol': False,
#   'has_fraction_pattern': False,
#   'has_currency_symbol': True,       # '$2'
#   'max_numeric_value_approx': 16.0,
#   'min_numeric_value_approx': 2.0,
#   'numeric_range_approx': 14.0,
#   'repeated_number_flag': False
# }

first_pass_output = "She lays 16 eggs. Eats 3 + bakes 4 = 7. Sells 16 - 7 = 9 eggs. 9 * 2 = 18. Final answer: 18"
z_fp = extract_first_pass_features(q, first_pass_output, parsed_answer="18")
# {
#   'first_pass_parse_success': True,
#   'first_pass_output_length': 91,
#   'first_pass_has_final_answer_cue': True,
#   'first_pass_has_uncertainty_phrase': False,
#   'first_pass_num_numeric_mentions': 10,
#   'first_pass_empty_or_malformed_flag': False
# }
```
