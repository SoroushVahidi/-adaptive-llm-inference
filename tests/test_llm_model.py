from __future__ import annotations

import os

import pytest

from src.models.llm_model import LocalStubLLMModel, OpenAICompatibleLLMModel


def test_openai_model_requires_api_key() -> None:
    original = os.environ.pop("OPENAI_API_KEY", None)
    try:
        with pytest.raises(ValueError, match="Missing OpenAI-compatible API key"):
            OpenAICompatibleLLMModel(model_name="gpt-test", api_key=None)
    finally:
        if original is not None:
            os.environ["OPENAI_API_KEY"] = original


def test_local_stub_model_generates_responses() -> None:
    model = LocalStubLLMModel(responses=["The answer is 42.", "The answer is 7."])

    first = model.generate("What is 6 * 7?")
    second = model.generate("What is 3 + 4?")

    assert first == "The answer is 42."
    assert second == "The answer is 7."


def test_local_stub_model_generate_n_uses_base_model_method() -> None:
    model = LocalStubLLMModel(responses=["A", "B", "C"])

    responses = model.generate_n("ignored", 3)

    assert responses == ["A", "B", "C"]
