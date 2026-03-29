"""Number-word and verb-role lexicon support for per-number role analysis.

This module provides two lightweight support layers:

1. **Number-word normalization** — maps written quantity words to their
   canonical numeric values and classifies them by ``source_type``.

2. **Verb-to-role lexicon** — maps common verbs and short phrases to expected
   arithmetic roles (add, subtract, multiply/rate, ratio, capacity).

Both layers are implemented with pure regex/string operations.  No external
NLP libraries are required.

See docs/NUMBER_ROLE_LEXICON.md for design rationale, supported keywords, and
limitations.
"""

from __future__ import annotations

import re
from typing import TypedDict

# ---------------------------------------------------------------------------
# Type definitions
# ---------------------------------------------------------------------------


class NumberWordMatch(TypedDict):
    """One resolved number-word span from a piece of text."""

    surface_text: str
    """The exact surface form that was matched (lowercased)."""

    normalized_value: float
    """Numeric equivalent of the surface form."""

    source_type: str
    """Category of the match.  One of:
    ``number_word`` — cardinal words (one … twelve)
    ``multiplicative_word`` — double, triple, twice
    ``fraction_word`` — half
    ``quantity_word`` — dozen
    """


class RoleCueResult(TypedDict):
    """Role-cue extraction result for a piece of text."""

    add: list[str]
    """Surface tokens/phrases that signal an additive role."""

    subtract: list[str]
    """Surface tokens/phrases that signal a subtractive role."""

    multiply_rate: list[str]
    """Surface tokens/phrases that signal a multiplicative / rate role."""

    ratio: list[str]
    """Surface tokens/phrases that signal a ratio / scaling role."""

    capacity: list[str]
    """Surface tokens/phrases that signal a capacity / ceiling role."""


# ---------------------------------------------------------------------------
# Number-word lexicon
# ---------------------------------------------------------------------------

# Maps lowercased surface token → (normalized_value, source_type)
_NUMBER_WORD_MAP: dict[str, tuple[float, str]] = {
    # Cardinal words
    "one": (1.0, "number_word"),
    "two": (2.0, "number_word"),
    "three": (3.0, "number_word"),
    "four": (4.0, "number_word"),
    "five": (5.0, "number_word"),
    "six": (6.0, "number_word"),
    "seven": (7.0, "number_word"),
    "eight": (8.0, "number_word"),
    "nine": (9.0, "number_word"),
    "ten": (10.0, "number_word"),
    "eleven": (11.0, "number_word"),
    "twelve": (12.0, "number_word"),
    # Quantity word
    "dozen": (12.0, "quantity_word"),
    # Fraction word
    "half": (0.5, "fraction_word"),
    # Multiplicative words
    "double": (2.0, "multiplicative_word"),
    "twice": (2.0, "multiplicative_word"),
    "triple": (3.0, "multiplicative_word"),
}

# Build a single compiled pattern that matches any number word as a whole word.
# Ordered longest-first so multi-word tokens (if added later) would win.
_NUMBER_WORD_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in _NUMBER_WORD_MAP) + r")\b",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Verb-to-role lexicon
# ---------------------------------------------------------------------------

# Each role maps to a list of (compiled_pattern, canonical_phrase) pairs.
# Phrases with spaces are matched literally (word-boundary anchored).

_ROLE_PATTERNS: dict[str, list[re.Pattern[str]]] = {
    "add": [
        re.compile(r"\b(?:received|receive|receives)\b", re.IGNORECASE),
        re.compile(r"\b(?:gained|gain|gains)\b", re.IGNORECASE),
        re.compile(r"\b(?:bought|buy|buys)\b", re.IGNORECASE),
        re.compile(r"\b(?:added|add|adds)\b", re.IGNORECASE),
        re.compile(r"\b(?:found|find|finds)\b", re.IGNORECASE),
        re.compile(r"\bgot\s+more\b", re.IGNORECASE),
        re.compile(r"\b(?:collected|collect|collects)\b", re.IGNORECASE),
        re.compile(r"\b(?:earned|earn|earns)\b", re.IGNORECASE),
        re.compile(r"\b(?:picked\s+up|picks\s+up)\b", re.IGNORECASE),
    ],
    "subtract": [
        re.compile(r"\b(?:spent|spend|spends)\b", re.IGNORECASE),
        re.compile(r"\b(?:lost|lose|loses)\b", re.IGNORECASE),
        re.compile(r"\bgave\s+away\b", re.IGNORECASE),
        re.compile(r"\b(?:sold|sell|sells)\b", re.IGNORECASE),
        re.compile(r"\b(?:used|uses|use\s+up|used\s+up)\b", re.IGNORECASE),
        re.compile(r"\b(?:ate|eat|eats|eaten)\b", re.IGNORECASE),
        re.compile(r"\b(?:removed|remove|removes)\b", re.IGNORECASE),
        re.compile(r"\b(?:donated|donate|donates)\b", re.IGNORECASE),
        re.compile(r"\b(?:consumed|consume|consumes)\b", re.IGNORECASE),
        re.compile(r"\b(?:gave|give|gives)\b", re.IGNORECASE),
    ],
    "multiply_rate": [
        re.compile(r"\beach\b", re.IGNORECASE),
        re.compile(r"\bevery\b", re.IGNORECASE),
        re.compile(r"\bper\b", re.IGNORECASE),
        re.compile(r"\bapiece\b", re.IGNORECASE),
        re.compile(r"\ba\s+piece\b", re.IGNORECASE),
    ],
    "ratio": [
        re.compile(r"\bhalf\b", re.IGNORECASE),
        re.compile(r"\bdouble\b", re.IGNORECASE),
        re.compile(r"\btwice\b", re.IGNORECASE),
        re.compile(r"\btriple\b", re.IGNORECASE),
    ],
    "capacity": [
        re.compile(r"\bminimum\s+(?:number\s+of|of)\b", re.IGNORECASE),
        re.compile(r"\bat\s+least\b", re.IGNORECASE),
        re.compile(r"\benough\b", re.IGNORECASE),
        re.compile(r"\btrips?\b", re.IGNORECASE),
        re.compile(r"\bboxes?\b", re.IGNORECASE),
        re.compile(r"\bbuses?\b", re.IGNORECASE),
        re.compile(r"\bbusses?\b", re.IGNORECASE),
        re.compile(r"\bcontainers?\b", re.IGNORECASE),
    ],
}


# ---------------------------------------------------------------------------
# Public API — number-word normalization
# ---------------------------------------------------------------------------


def normalize_number_word(token_or_phrase: str) -> NumberWordMatch | None:
    """Return a normalized representation for a single number-word token.

    The lookup is case-insensitive and requires the input to match one of the
    supported number-word surface forms exactly (ignoring leading/trailing
    whitespace).  Multi-token inputs are accepted but must match a known entry.

    Parameters
    ----------
    token_or_phrase:
        A single word or phrase to look up (e.g. ``"three"``, ``"half"``,
        ``"twice"``).

    Returns
    -------
    NumberWordMatch or None
        Structured dict with ``surface_text``, ``normalized_value``, and
        ``source_type``; or ``None`` if the input is not a recognized
        number word.

    Examples
    --------
    >>> normalize_number_word("three")
    {'surface_text': 'three', 'normalized_value': 3.0, 'source_type': 'number_word'}
    >>> normalize_number_word("TWICE")
    {'surface_text': 'twice', 'normalized_value': 2.0, 'source_type': 'multiplicative_word'}
    >>> normalize_number_word("elephant")
    None
    """
    key = token_or_phrase.strip().lower()
    if key not in _NUMBER_WORD_MAP:
        return None
    value, source_type = _NUMBER_WORD_MAP[key]
    return NumberWordMatch(
        surface_text=key,
        normalized_value=value,
        source_type=source_type,
    )


def extract_number_word_matches(text: str) -> list[NumberWordMatch]:
    """Find all number-word spans in *text* and return normalized matches.

    Each overlapping occurrence is returned as a separate entry.  The order
    of results follows the left-to-right appearance in the text.

    Parameters
    ----------
    text:
        Free-form text to scan (e.g. a GSM8K math word problem).

    Returns
    -------
    list[NumberWordMatch]
        Possibly empty list of structured matches.

    Examples
    --------
    >>> results = extract_number_word_matches("She has three apples and twice as many oranges.")
    >>> [(m['surface_text'], m['normalized_value']) for m in results]
    [('three', 3.0), ('twice', 2.0)]
    """
    if not text:
        return []
    matches: list[NumberWordMatch] = []
    for m in _NUMBER_WORD_PATTERN.finditer(text):
        surface = m.group(0).lower()
        value, source_type = _NUMBER_WORD_MAP[surface]
        matches.append(
            NumberWordMatch(
                surface_text=surface,
                normalized_value=value,
                source_type=source_type,
            )
        )
    return matches


# ---------------------------------------------------------------------------
# Public API — verb-to-role lexicon
# ---------------------------------------------------------------------------


def get_role_cues(text: str) -> RoleCueResult:
    """Scan *text* for verb/phrase cues and return all matched cue strings.

    Returns a dict with one key per arithmetic role.  Each value is a list of
    the raw matched surface strings found in *text* (lowercased).  Lists may
    be empty.

    Parameters
    ----------
    text:
        Free-form text to scan.

    Returns
    -------
    RoleCueResult
        Flat dict with keys ``add``, ``subtract``, ``multiply_rate``,
        ``ratio``, and ``capacity``.

    Examples
    --------
    >>> cues = get_role_cues("She spent $5 and earned $10 each week.")
    >>> cues["subtract"]
    ['spent']
    >>> cues["add"]
    ['earned']
    >>> cues["multiply_rate"]
    ['each']
    """
    result: RoleCueResult = {
        "add": [],
        "subtract": [],
        "multiply_rate": [],
        "ratio": [],
        "capacity": [],
    }
    if not text:
        return result

    for role, patterns in _ROLE_PATTERNS.items():
        seen: set[str] = set()
        for pattern in patterns:
            for m in pattern.finditer(text):
                normalized = m.group(0).lower()
                if normalized not in seen:
                    seen.add(normalized)
                    result[role].append(normalized)  # type: ignore[literal-required]
    return result


def classify_local_role_cue(window_text: str) -> str | None:
    """Return the dominant arithmetic role in a short window of text.

    Counts cue hits per role and returns the role with the most hits.  Ties
    are broken by the order: ``subtract`` > ``add`` > ``ratio`` >
    ``multiply_rate`` > ``capacity``.  Returns ``None`` when no cues are
    found.

    This is intentionally lightweight — designed for 5–15 word windows
    around a number token, not full sentences.

    Parameters
    ----------
    window_text:
        A short text window (e.g. surrounding context of a number).

    Returns
    -------
    str or None
        One of ``"add"``, ``"subtract"``, ``"multiply_rate"``, ``"ratio"``,
        ``"capacity"``; or ``None`` if no role cue is present.

    Examples
    --------
    >>> classify_local_role_cue("she spent three dollars on lunch")
    'subtract'
    >>> classify_local_role_cue("he earned double the usual amount")
    'ratio'
    >>> classify_local_role_cue("the sky is blue")
    None
    """
    cues = get_role_cues(window_text)
    # Tie-break order (index = preference rank)
    priority = ["subtract", "add", "ratio", "multiply_rate", "capacity"]
    best_role: str | None = None
    best_count = 0
    for role in priority:
        count = len(cues[role])  # type: ignore[literal-required]
        if count > best_count:
            best_count = count
            best_role = role
    return best_role
