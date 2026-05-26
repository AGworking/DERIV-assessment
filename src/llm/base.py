"""Abstract LLM provider interface.

Every concrete provider (MockProvider, OpenAIProvider, ...) exposes the same
surface: `.generate(system, user)` returns the raw text the model emitted. The
caller (the draft / self-check / clarify stages) is responsible for parsing
that text and for logging the call.

Keeping the interface this minimal means adding a new provider — Anthropic,
Azure OpenAI, a local Ollama instance — is a single new file that subclasses
LLMProvider and a one-line addition to the factory in `provider.py`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Base class for every LLM backend the pipeline can talk to."""

    #: short identifier written into llm_calls.jsonl (e.g. "mock", "openai")
    name: str = "abstract"

    #: model name written into llm_calls.jsonl (e.g. "gpt-4o-mini")
    model: str = "n/a"

    @abstractmethod
    def generate(self, system: str, user: str) -> str:
        """Return the raw text the LLM emitted.

        `system` is the system prompt. `user` is a JSON-serialised payload
        the calling stage built (with the ticket, retrieved KB snippets, and
        a `task` discriminator like "draft" / "self_check" / "clarify").
        """
