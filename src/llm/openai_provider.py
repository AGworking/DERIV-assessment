"""OpenAI LLM provider — the real-LLM path behind the feature flag.

Activated when `LLM_PROVIDER=openai`. Requires `OPENAI_API_KEY` to be set
(either in the environment or in .env). Optional `OPENAI_MODEL` overrides
the default model (`gpt-4o-mini`).

Note: the `openai` SDK is imported inside `__init__` so the mock path never
touches that dependency at import time.
"""

from __future__ import annotations

import os

from src.llm.base import LLMProvider


DEFAULT_MODEL = "gpt-4o-mini"


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        # Strip defensively — .env values can pick up trailing newlines /
        # quoting noise that HTTPX would otherwise reject in the auth header.
        self.model = (os.environ.get("OPENAI_MODEL") or DEFAULT_MODEL).strip()
        api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set in the environment."
            )

        from openai import OpenAI  # local import keeps mock path dep-free
        self._client = OpenAI(api_key=api_key)

    def generate(self, system: str, user: str) -> str:
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
        )
        return resp.choices[0].message.content or "{}"
