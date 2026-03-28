## Constraint-aware violation features

### Goal

Provide a lightweight question-answer consistency layer for math word problems.

The purpose is not symbolic verification. The purpose is to detect when a
reasoning answer looks like the wrong *kind* of answer for the question.

### Feature taxonomy

The feature layer uses only string, regex, and small arithmetic heuristics.

#### 1. Answer type / format checks

- `answer_type_mismatch_suspected`
  - numeric target question but no numeric final answer
- `answer_not_mentioned_in_final_statement_suspected`
  - a numeric answer is parsed, but the final sentence does not restate it
- `integer_expected_but_noninteger_suspected`
  - count-like questions produce a non-integer answer

#### 2. Quantity / target consistency checks

- `target_quantity_mismatch_suspected`
  - the question asks for a target quantity such as remaining/left/earned, but
    the final statement uses a conflicting quantity cue
- `constraint_word_conflict_suspected`
  - question and final statement use conflicting words like `remaining` vs
    `total`, or `earned` vs `spent`

#### 3. Unit consistency checks

- `unit_mismatch_suspected`
  - the question mentions units/items but the final statement omits them or uses
    conflicting ones
- `percent_or_ratio_mismatch_suspected`
  - percent/fraction style question but final wording does not align with that

#### 4. Simple plausibility checks

- `impossible_sign_suspected`
  - negative answer for count/money/item questions
- `bound_violation_suspected`
  - answer exceeds a simple obvious total from the question when that total can
    be recovered cheaply from the numbers in the text

### Design constraints

- no symbolic solver
- no external math libraries
- no equation extraction
- all signals are heuristic and interpretable

### Intended use

These features are meant for selective escalation:

- keep `reasoning_greedy` when the final answer looks self-consistent
- escalate to `direct_plus_revise` only when question-answer consistency looks
  suspicious

