"""Deterministic pipeline stages.

Pure Python — no LLM calls. These stages decide the route for every ticket
BEFORE any LLM is consulted. Modules in this package must not import from
src.llm.* — that boundary is what makes the safety guarantee inspectable.
"""
