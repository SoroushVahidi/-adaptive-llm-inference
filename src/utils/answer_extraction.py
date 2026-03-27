"""Utilities for extracting numeric answers from model output."""

from __future__ import annotations

import re


def extract_numeric_answer(text: str) -> str:
    """Pull the last number from a string, stripping commas.

    This is a simple heuristic used by GSM8K evaluation: the final number in
    the model output is treated as the predicted answer.
    """
    numbers = re.findall(r"-?[\d,]+\.?\d*", text)
    if numbers:
        return numbers[-1].replace(",", "")
    return text.strip()
