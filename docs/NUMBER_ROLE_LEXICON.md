# Number-Word and Verb-Role Lexicon

> **Purpose:** Provide a lightweight, reusable support layer for per-number
> role analysis in math word problems.  Two complementary sub-systems are
> provided: a *number-word normalization* utility and a *verb-to-role lexicon*.
> Both rely exclusively on pure regex/string operations — no external NLP
> libraries are required.

---

## 1. Overview

`src/features/number_role_lexicon.py` exposes four public helpers:

```python
from src.features import (
    normalize_number_word,        # single-token lookup
    extract_number_word_matches,  # scan full text for number words
    get_role_cues,                # extract all verb/phrase role cues from text
    classify_local_role_cue,      # dominant role in a short window
)
```

All four functions are entirely offline (no model calls).

---

## 2. Number-Word Normalization

### 2.1 Supported number words

| Surface form | `normalized_value` | `source_type` |
|---|---|---|
| `one` | 1.0 | `number_word` |
| `two` | 2.0 | `number_word` |
| `three` | 3.0 | `number_word` |
| `four` | 4.0 | `number_word` |
| `five` | 5.0 | `number_word` |
| `six` | 6.0 | `number_word` |
| `seven` | 7.0 | `number_word` |
| `eight` | 8.0 | `number_word` |
| `nine` | 9.0 | `number_word` |
| `ten` | 10.0 | `number_word` |
| `eleven` | 11.0 | `number_word` |
| `twelve` | 12.0 | `number_word` |
| `dozen` | 12.0 | `quantity_word` |
| `half` | 0.5 | `fraction_word` |
| `double` | 2.0 | `multiplicative_word` |
| `twice` | 2.0 | `multiplicative_word` |
| `triple` | 3.0 | `multiplicative_word` |

Lookup is case-insensitive; the returned `surface_text` is always lowercased.

### 2.2 `source_type` categories

| Category | Description |
|---|---|
| `number_word` | Cardinal words representing a whole-number count (one–twelve) |
| `quantity_word` | Named quantity groups (`dozen`) |
| `fraction_word` | Fractional multipliers (`half`) |
| `multiplicative_word` | Explicit scaling words (`double`, `twice`, `triple`) |

### 2.3 API

#### `normalize_number_word(token_or_phrase) → NumberWordMatch | None`

Single-token lookup.  Returns `None` for unrecognized inputs.

```python
>>> normalize_number_word("three")
{'surface_text': 'three', 'normalized_value': 3.0, 'source_type': 'number_word'}

>>> normalize_number_word("TWICE")
{'surface_text': 'twice', 'normalized_value': 2.0, 'source_type': 'multiplicative_word'}

>>> normalize_number_word("elephant")
None
```

#### `extract_number_word_matches(text) → list[NumberWordMatch]`

Scans free-form text and returns every number-word occurrence in order.
Each result has the same `surface_text`, `normalized_value`, `source_type`
fields as the single-token lookup.

```python
>>> extract_number_word_matches("She has three apples and twice as many oranges.")
[
  {'surface_text': 'three', 'normalized_value': 3.0, 'source_type': 'number_word'},
  {'surface_text': 'twice', 'normalized_value': 2.0, 'source_type': 'multiplicative_word'},
]
```

---

## 3. Verb-to-Role Lexicon

### 3.1 Supported role cue groups

#### `add` — additive operations

Signals that a quantity is *gained or acquired*.

| Cue phrase / verb |
|---|
| received, receive, receives |
| gained, gain, gains |
| bought, buy, buys |
| added, add, adds |
| found, find, finds |
| got more |
| collected, collect, collects |
| earned, earn, earns |
| picked up, picks up |

#### `subtract` — subtractive operations

Signals that a quantity is *lost, consumed, or transferred away*.

| Cue phrase / verb |
|---|
| spent, spend, spends |
| lost, lose, loses |
| gave away |
| sold, sell, sells |
| used, uses, use up, used up |
| ate, eat, eats, eaten |
| removed, remove, removes |
| donated, donate, donates |
| consumed, consume, consumes |
| gave, give, gives |

#### `multiply_rate` — per-unit / rate operations

Signals that a quantity applies *per unit* or at a rate.

| Cue phrase / word |
|---|
| each |
| every |
| per |
| apiece |
| a piece |

#### `ratio` — scaling / ratio operations

Signals that a quantity is expressed as a *multiple or fraction* of another.

| Cue phrase / word |
|---|
| half |
| double |
| twice |
| triple |

#### `capacity` — ceiling / capacity operations

Signals a *minimum required count* or container-packing problem.

| Cue phrase / word |
|---|
| minimum number of, minimum of |
| at least |
| enough |
| trip(s) |
| box(es) |
| bus(es), busse(s) |
| container(s) |

### 3.2 API

#### `get_role_cues(text) → RoleCueResult`

Returns a dict with one key per role.  Each value is the list of raw matched
surface strings (lowercased) found in *text*.

```python
>>> get_role_cues("She spent $5 and earned $10 each week.")
{
  'add':           ['earned'],
  'subtract':      ['spent'],
  'multiply_rate': ['each'],
  'ratio':         [],
  'capacity':      [],
}
```

#### `classify_local_role_cue(window_text) → str | None`

Returns the dominant role for a short text window, or `None` when no cues
are found.  Ties are broken by priority: `subtract > add > ratio >
multiply_rate > capacity`.

```python
>>> classify_local_role_cue("she spent three dollars on lunch")
'subtract'

>>> classify_local_role_cue("he earned double the usual amount")
'ratio'

>>> classify_local_role_cue("the sky is blue")
None
```

---

## 4. Sample Outputs on Example Strings

### Example A — "remaining" problem

**Input:**
> "Janet had 20 apples. She gave 5 to her friend and ate 3 herself.
>  How many apples does she have left?"

**`extract_number_word_matches`:** *(no written number words — all digits)*
```python
[]
```

**`get_role_cues`:**
```python
{
  'add':           [],
  'subtract':      ['gave', 'ate'],
  'multiply_rate': [],
  'ratio':         [],
  'capacity':      [],
}
```

---

### Example B — "rate × scale" problem

**Input:**
> "She earns double her usual pay. The base pay is $10 per hour."

**`extract_number_word_matches`:**
```python
[{'surface_text': 'double', 'normalized_value': 2.0, 'source_type': 'multiplicative_word'}]
```

**`get_role_cues`:**
```python
{
  'add':           ['earns'],
  'subtract':      [],
  'multiply_rate': ['per'],
  'ratio':         ['double'],
  'capacity':      [],
}
```

---

### Example C — capacity / packing problem

**Input:**
> "A bus can hold 40 passengers. There are 120 students.
>  How many trips does the bus need to make?"

**`extract_number_word_matches`:** *(no written number words)*
```python
[]
```

**`get_role_cues`:**
```python
{
  'add':           [],
  'subtract':      [],
  'multiply_rate': [],
  'ratio':         [],
  'capacity':      ['trips'],
}
```

**`classify_local_role_cue("how many trips does the bus need")`:**
```python
'capacity'
```

---

## 5. Intended Use in Per-Number Role Analysis

This module is intended as a *support layer* for a larger per-number role
analysis pipeline.  The expected integration pattern is:

1. Tokenize or split the problem text into number-bearing spans.
2. For each span, extract a small context window (e.g. ±5 words).
3. Call `classify_local_role_cue(window)` to assign a provisional role to
   each number token.
4. Call `extract_number_word_matches(text)` to surface written quantities
   that do not appear as digits.
5. Combine the per-number roles with the boolean features from
   `extract_target_quantity_features` to build a richer feature vector.

This module does **not** implement the full role-routing policy or the
experimental pipeline.

---

## 6. Limitations

- All matching is **lexical** — surface form only.  Negation (e.g. "did
  *not* spend") is not detected; such a phrase would still fire a `subtract`
  cue.
- Only the 17 number words listed in §2.1 are supported.  Larger cardinals
  (thirteen … one hundred) and ordinals (first, second …) are not included.
- `classify_local_role_cue` uses a fixed tie-breaking priority
  (`subtract > add > ratio > multiply_rate > capacity`) which may not be
  optimal for all problem types.
- Morphological variants not included in the patterns (e.g. unusual
  conjugations) will be missed.
- The capacity cues (`trips`, `boxes`, `buses`) are noun-based, not
  verb-based, and may fire in non-capacity contexts (e.g. "She took three
  trips to the store" where the primary role is counting, not ceiling
  computation).

---

*Module:* `src/features/number_role_lexicon.py`  
*Tests:* `tests/test_number_role_lexicon.py`
