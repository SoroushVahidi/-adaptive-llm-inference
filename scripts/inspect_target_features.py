#!/usr/bin/env python3
"""Sanity-check target-quantity features on a few example questions.

Usage:
    python3 scripts/inspect_target_features.py

No API key or live model is needed — this is a pure offline diagnostic.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features import extract_query_features, extract_target_quantity_features

# ---------------------------------------------------------------------------
# Example questions (GSM8K-style)
# ---------------------------------------------------------------------------

EXAMPLES = [
    (
        "remaining problem",
        (
            "Janet had 20 apples. She gave 5 to her friend and ate 3 herself. "
            "How many apples does she have left?"
        ),
    ),
    (
        "total earnings problem",
        (
            "Maria earns $15 per hour. She works 8 hours a day for 5 days. "
            "What are her total earnings?"
        ),
    ),
    (
        "rate problem",
        (
            "A machine produces 120 widgets in 4 hours. "
            "How many widgets does it produce per hour?"
        ),
    ),
    (
        "multi-step chain problem",
        (
            "Tom had 50 dollars. He spent $12 on lunch and $8 on a book. "
            "Then he received $5 change from the cashier. "
            "How much money does he have now?"
        ),
    ),
    (
        "difference problem",
        (
            "Alice scored 92 points and Bob scored 74 points. "
            "How many more points did Alice score than Bob?"
        ),
    ),
]

# ---------------------------------------------------------------------------
# Pretty-print helper
# ---------------------------------------------------------------------------

_SECTION_WIDTH = 70


def _print_features(label: str, feats: dict) -> None:
    true_feats = [k for k, v in feats.items() if v]
    false_feats = [k for k, v in feats.items() if not v]
    print(f"  ✓ {'  ✓ '.join(true_feats)}" if true_feats else "  (no features fired)")
    if false_feats:
        print(f"  ✗ {', '.join(false_feats)}")


def main() -> None:
    print("=" * _SECTION_WIDTH)
    print("  Target-Quantity Feature Diagnostic")
    print("=" * _SECTION_WIDTH)

    for name, question in EXAMPLES:
        print(f"\n[{name}]")
        print(f"  Q: {question[:90]}{'...' if len(question) > 90 else ''}")
        print()

        tq = extract_target_quantity_features(question)
        base = extract_query_features(question)

        print("  Target-quantity features:")
        true_keys = [k for k, v in tq.items() if v]
        false_keys = [k for k, v in tq.items() if not v]
        if true_keys:
            for k in true_keys:
                print(f"    ✓  {k}")
        else:
            print("    (none fired)")
        for k in false_keys:
            print(f"    ✗  {k}")

        print()
        print(
            f"  Base features (selected): "
            f"num_numeric={base['num_numeric_mentions']}  "
            f"num_sentences={base['num_sentences_approx']}  "
            f"has_multi_step_cue={base['has_multi_step_cue']}  "
            f"has_currency={base['has_currency_symbol']}"
        )
        print("-" * _SECTION_WIDTH)

    print("\nDone.")


if __name__ == "__main__":
    main()
