"""Minimal OpenAI chat-completions wrapper for small GSM8K experiments."""

from __future__ import annotations

import json
import os
from typing import Any
from urllib import error, request

from src.models.base import Model


def _normalize_base_url(base_url: str | None) -> str:
    resolved = base_url or os.getenv("OPENAI_BASE_URL") or "https://api.openai.com/v1"
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


class OpenAILLMModel(Model):
    """OpenAI-backed model with a minimal generate / generate_n interface."""

    def __init__(
        self,
        model_name: str,
        base_url: str | None = None,
        greedy_temperature: float = 0.0,
        sample_temperature: float = 0.7,
        max_tokens: int = 128,
        timeout_seconds: float = 60.0,
        prompt_prefix: str | None = None,
    ) -> None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY")

        self.model_name = str(model_name)
        self.api_key = api_key
        self.base_url = _normalize_base_url(base_url)
        self.greedy_temperature = float(greedy_temperature)
        self.sample_temperature = float(sample_temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_seconds = float(timeout_seconds)
        self.prompt_prefix = (
            prompt_prefix
            or "Answer the following question. Give only the final numeric answer."
        )

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        return [
            {"role": "system", "content": self.prompt_prefix},
            {"role": "user", "content": prompt},
        ]

    def debug_sampling_config(self) -> dict[str, Any]:
        """Return the effective sampling configuration for debugging."""
        return {
            "model_name": self.model_name,
            "base_url": self.base_url,
            "greedy_temperature": self.greedy_temperature,
            "sample_temperature": self.sample_temperature,
            "max_tokens": self.max_tokens,
            "timeout_seconds": self.timeout_seconds,
            "prompt_prefix": self.prompt_prefix,
        }

    def _request_completions(self, prompt: str, n: int, temperature: float) -> list[str]:
        payload = {
            "model": self.model_name,
            "messages": self._build_messages(prompt),
            "temperature": float(temperature),
            "n": int(n),
            "max_tokens": int(self.max_tokens),
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
                f"OpenAI API error HTTP {exc.code}: {error_body}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(f"OpenAI API request failed: {exc}") from exc

        parsed = json.loads(response_body)
        choices = parsed.get("choices")
        if not isinstance(choices, list) or not choices:
            raise RuntimeError(
                "OpenAI API response did not contain choices: "
                f"{response_body}"
            )

        return [_extract_text_from_choice(choice) for choice in choices]

    def generate(self, prompt: str) -> str:
        return self._request_completions(
            prompt=prompt,
            n=1,
            temperature=self.greedy_temperature,
        )[0]

    def generate_n(self, prompt: str, n: int) -> list[str]:
        if n <= 0:
            raise ValueError("n must be positive")
        return self._request_completions(
            prompt=prompt,
            n=n,
            temperature=self.sample_temperature,
        )

    def generate_with_temperature(self, prompt: str, n: int, temperature: float) -> list[str]:
        """Expose a small override hook for experiments with mixed sampling stages."""
        if n <= 0:
            raise ValueError("n must be positive")
        return self._request_completions(
            prompt=prompt,
            n=n,
            temperature=temperature,
        )

    def with_prompt_prefix(self, prompt_prefix: str) -> "OpenAILLMModel":
        """Create a shallow copy that changes only the prompt prefix."""
        cloned = OpenAILLMModel(
            model_name=self.model_name,
            base_url=self.base_url,
            greedy_temperature=self.greedy_temperature,
            sample_temperature=self.sample_temperature,
            max_tokens=self.max_tokens,
            timeout_seconds=self.timeout_seconds,
            prompt_prefix=prompt_prefix,
        )
        return cloned
