"""Append-only logger for every LLM call.

Each record is one JSON object per line in artifacts/llm_calls.jsonl, with the
shape required by the spec: stage, ticket_id, timestamp, provider, model,
prompt_hash, input_artifacts, output_artifact.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from src.paths import LLM_CALLS, ensure_artifacts_dir


def prompt_hash(text: str) -> str:
    """Stable short hash for a prompt so we can match calls to their input."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log_call(
    *,
    stage: str,
    ticket_id: str | None,
    provider: str,
    model: str,
    prompt_hash_value: str,
    input_artifacts: Iterable[str | Path],
    output_artifact: str | Path,
) -> None:
    ensure_artifacts_dir()
    record = {
        "stage": stage,
        "ticket_id": ticket_id,
        "timestamp": now_iso(),
        "provider": provider,
        "model": model,
        "prompt_hash": prompt_hash_value,
        "input_artifacts": [str(p) for p in input_artifacts],
        "output_artifact": str(output_artifact),
    }
    with LLM_CALLS.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, ensure_ascii=False) + "\n")


def reset_log() -> None:
    """Truncate llm_calls.jsonl at the start of a pipeline run.

    Without this, repeated runs would accumulate old records and the
    'one record per call' validation invariant would fail.
    """
    ensure_artifacts_dir()
    LLM_CALLS.write_text("", encoding="utf-8")
