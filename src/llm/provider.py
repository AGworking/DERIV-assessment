"""LLM provider abstraction.

Default provider is `mock` (no network, no keys, deterministic).
Setting LLM_PROVIDER=openai + OPENAI_API_KEY switches to real OpenAI calls.

Every provider exposes the same surface: `.generate(system, user)` returns a
dict with `text`, `provider`, `model`. The caller (src/draft.py) is responsible
for parsing the JSON the LLM emits and for logging the call.
"""

from __future__ import annotations

import json
import os
import re
from abc import ABC, abstractmethod


class LLMProvider(ABC):
    name: str = "abstract"
    model: str = "n/a"

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Return the raw text the LLM emitted."""


# ---------------------------------------------------------------------------
# Mock provider
# ---------------------------------------------------------------------------

class MockProvider(LLMProvider):
    """Builds a grounded reply by templating the retrieved KB content.

    The mock provider is intentionally simple but it stays inside the
    retrieval evidence — exactly the contract we ask of a real LLM.
    It receives the same system/user prompt the real provider sees, so we can
    still emit a JSON object with the right shape.

    The user payload is a JSON string the caller built — we parse it back out
    so the templated reply can reference subject + KB content directly.
    """

    name = "mock"
    model = "mock-grounded-templater-v1"

    _SUMMARY_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")

    def generate(self, system: str, user: str) -> str:
        try:
            payload = json.loads(user)
        except json.JSONDecodeError:
            # Defensive: still return something valid for the caller to parse.
            return json.dumps({
                "draft_reply": "Thank you for contacting support. A human agent will follow up shortly.",
                "used_article_ids": [],
                "confidence": "low",
            })

        return self._build_reply(payload)

    def _build_reply(self, payload: dict) -> str:
        subject = payload.get("subject", "").strip()
        snippets = payload.get("snippets", [])

        if not snippets:
            return json.dumps({
                "draft_reply": (
                    "Thanks for reaching out. We need a little more information to help — "
                    "could you share the relevant transaction reference and a screenshot of the issue?"
                ),
                "used_article_ids": [],
                "confidence": "low",
            })

        # Use the top-1 snippet as the primary source. We may cite up to two.
        primary = snippets[0]
        cited_ids = [primary["article_id"]]
        if len(snippets) > 1 and snippets[1].get("score", 0.0) >= 0.10:
            cited_ids.append(snippets[1]["article_id"])

        primary_summary = self._summarise(primary["content"])
        citation_blob = ", ".join(cited_ids)

        body = (
            f"Hi,\n\nThanks for reaching out about \"{subject}\". "
            f"{primary_summary} "
            f"If the issue persists after trying this, please reply with any "
            f"reference numbers or screenshots and we'll escalate.\n\n"
            f"Best regards,\nSupport Team\n\n"
            f"(Reference: {citation_blob})"
        )

        confidence = "high" if primary.get("score", 0.0) >= 0.3 else "medium"

        return json.dumps({
            "draft_reply": body,
            "used_article_ids": cited_ids,
            "confidence": confidence,
        })

    def _summarise(self, content: str) -> str:
        """First 1-2 sentences of the KB content — short, grounded, no invention."""
        sentences = self._SUMMARY_SENTENCE_RE.split(content.strip())
        return " ".join(sentences[:2]).strip()


# ---------------------------------------------------------------------------
# OpenAI provider (real LLM, behind feature flag)
# ---------------------------------------------------------------------------

class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self) -> None:
        self.model = os.environ.get("OPENAI_MODEL", "gpt-4o-mini")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "LLM_PROVIDER=openai but OPENAI_API_KEY is not set in the environment."
            )
        # Import inside __init__ so 'mock' mode never imports openai.
        from openai import OpenAI

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


# ---------------------------------------------------------------------------
# Factory (feature flag)
# ---------------------------------------------------------------------------

def get_provider() -> LLMProvider:
    flag = os.environ.get("LLM_PROVIDER", "mock").strip().lower()
    if flag == "openai":
        return OpenAIProvider()
    # Default and fallback: mock — works on a clean checkout with no keys.
    return MockProvider()
