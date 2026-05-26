"""Pipeline stages that call the LLM.

Each module here only runs on tickets that the deterministic layer
(src.deterministic.route) has explicitly cleared. The actual LLM transport
lives in src/llm/base.py + src/llm/mock.py + src/llm/openai_provider.py;
this subpackage holds the higher-level prompting + parsing logic.
"""
