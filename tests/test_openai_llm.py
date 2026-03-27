from __future__ import annotations

import os

import pytest

from src.models.openai_llm import OpenAILLMModel


def test_openai_model_requires_api_key() -> None:
    original = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="Missing OPENAI_API_KEY"):
            OpenAILLMModel(model_name="gpt-test")
    finally:
        if original is not None:
            os.environ["OPENAI_API_KEY"] = original


def test_openai_model_generate_n_rejects_non_positive_n() -> None:
    original = os.environ.get("OPENAI_API_KEY")
    os.environ["OPENAI_API_KEY"] = "test-key"
    try:
        model = OpenAILLMModel(model_name="gpt-test")
        with pytest.raises(ValueError, match="n must be positive"):
            model.generate_n("What is 1 + 1?", 0)
    finally:
        if original is None:
            os.environ.pop("OPENAI_API_KEY", None)
        else:
            os.environ["OPENAI_API_KEY"] = original
