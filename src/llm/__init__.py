"""LLM abstraction.

- `base.LLMProvider`            — abstract interface every backend implements
- `mock.MockProvider`           — deterministic templater, no network
- `openai_provider.OpenAIProvider` — real OpenAI calls (requires OPENAI_API_KEY)
- `provider.get_provider()`     — factory driven by the LLM_PROVIDER env var
- `logger.log_call(...)`        — append one record per LLM call to llm_calls.jsonl
"""

from src.llm.base import LLMProvider
from src.llm.provider import get_provider

__all__ = ["LLMProvider", "get_provider"]
