"""Minimal OpenAI-compatible LLM wrapper for small GSM8K validation runs."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from urllib import error, request

from src.models.base import Model


def _normalize_base_url(base_url: str | None) -> str:
    resolved = (
        base_url
        or os.getenv("OPENAI_BASE_URL")
        or "https://api.openai.com/v1"
    )
    return resolved.rstrip("/")


def _extract_text_from_choice(choice: dict[str, Any]) -> str:
    message = choice.get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text", "")))
        return "\n".join(part for part in parts if part).strip()
    return str(content).strip()


class OpenAICompatibleLLMModel(Model):
    """Call a chat-completions-compatible HTTP API.

    This wrapper is intentionally small. It is meant for validation experiments
    on small GSM8K subsets, not for large-scale inference infrastructure.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str | None = None,
        base_url: str | None = None,
        system_prompt: str | None = None,
        greedy_temperature: float = 0.0,
        sample_temperature: float = 0.7,
        max_tokens: int = 256,
        timeout_seconds: float = 60.0,
    ) -> None:
        resolved_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "Missing OpenAI-compatible API key. Set OPENAI_API_KEY or provide api_key."
            )

        self.model_name = str(model_name)
        self.api_key = resolved_key
        self.base_url = _normalize_base_url(base_url)
        self.system_prompt = (
            system_prompt
            or "Solve the math problem carefully. End with only the final numeric answer."
        )
        self.greedy_temperature = float(greedy_temperature)
        self.sample_temperature = float(sample_temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)

    def _build_messages(self, question: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    "Solve this GSM8K-style math problem.\n"
                    "Return your reasoning briefly and finish with the final numeric answer.\n\n"
                    f"Question: {question}"
                ),
            },
        ]

    def _chat_completions(
        self,
        question: str,
        n: int,
        temperature: float,
    ) -> list[str]:
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(question),
            "temperature": float(temperature),
            "n": int(n),
            "max_tokens": self.max_tokens,
        }
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            url=f"{self.base_url}/chat/completions",
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with request.urlopen(req, timeout=self.timeout_seconds) as response:
                response_body = response.read().decode("utf-8")
        except error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(
                "OpenAI-compatible request failed with HTTP "
                f"{exc.code} at {self.base_url}/chat/completions: {error_body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                f"OpenAI-compatible request failed for {self.base_url}/chat/completions: {exc}"
            ) from exc

        parsed = json.loads(response_body)
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                "OpenAI-compatible response did not contain any choices: "
                f"{response_body}"
            )

        return [_extract_text_from_choice(choice) for choice in choices]

    def generate(self, question: str) -> str:
        return self._chat_completions(
            question=question,
            n=1,
            temperature=self.greedy_temperature,
        )[0]

    def generate_n(self, question: str, n: int) -> list[str]:
        if n <= 0:
            raise ValueError("n must be positive")
        return self._chat_completions(
            question=question,
            n=n,
            temperature=self.sample_temperature,
        )


class LocalStubLLMModel(Model):
    """Small local fallback for tests or manual dry-runs."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self._responses = (
            list(responses)
            if responses is not None
            else ["The answer is 0.", "The answer is 1.", "The answer is 2."]
        )
        self._cursor = 0

    def generate(self, question: str) -> str:  # noqa: ARG002
        response = self._responses[self._cursor % len(self._responses)]
        self._cursor += 1
        return response


def load_api_key_from_file(path: str | None) -> str | None:
    """Optional helper for loading an API key from a small local file."""
    if path is None:
        return None
    return Path(path).read_text().strip()
