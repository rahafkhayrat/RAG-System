from __future__ import annotations

from typing import Protocol


class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        ...


class StubLLM:
    def complete(self, system: str, user: str) -> str:
        return "[stub] no LLM configured"


class OpenAICompatibleLLM:
    """Works with any OpenAI-compatible endpoint (OpenAI, Azure-style gateways, vLLM, etc.)."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
    ) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._base_url = base_url
        self._api_key = api_key
        self._model = model
        self._temperature = temperature

    def complete(self, system: str, user: str) -> str:
        from openai import OpenAI

        client = OpenAI(base_url=self._base_url, api_key=self._api_key)
        response = client.chat.completions.create(
            model=self._model,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()
