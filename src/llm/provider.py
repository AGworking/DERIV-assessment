"""LLM provider factory — picks a backend based on the LLM_PROVIDER env var.

Adding a new provider is a three-line change:
    1. Create `src/llm/<name>_provider.py` with a class subclassing LLMProvider.
    2. Import it here.
    3. Add another `if flag == "<name>": return <Name>Provider()` branch.

The default is the mock provider so a clean checkout with no API keys still
runs the full pipeline end-to-end.
"""

from __future__ import annotations

import os

from src.llm.base import LLMProvider
from src.llm.mock import MockProvider
from src.llm.openai_provider import OpenAIProvider


def get_provider() -> LLMProvider:
    flag = (os.environ.get("LLM_PROVIDER") or "mock").strip().lower()
    if flag == "openai":
        return OpenAIProvider()
    return MockProvider()
