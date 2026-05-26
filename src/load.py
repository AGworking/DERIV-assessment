"""Stage: load + validate input fixtures.

Deterministic. No LLM calls here. Validation failures raise loudly so the
pipeline halts before any downstream stage runs on bad data.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from src.paths import KB_IN, TICKETS_IN


REQUIRED_TICKET_FIELDS = {
    "ticket_id",
    "created_at",
    "channel",
    "language",
    "customer_tier",
    "subject",
    "message",
    "attachments",
    "metadata",
}

REQUIRED_KB_FIELDS = {
    "article_id",
    "title",
    "category",
    "tags",
    "content",
    "safe_for_ai",
}


class ValidationError(Exception):
    """Raised when an input file fails schema validation."""


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise ValidationError(f"Required input file not found: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValidationError(f"{path.name} is not valid JSON: {exc}") from exc


def _validate_tickets(tickets: Any) -> list[dict]:
    if not isinstance(tickets, list):
        raise ValidationError("tickets.json must be a JSON array")

    seen_ids: set[str] = set()
    for idx, ticket in enumerate(tickets):
        if not isinstance(ticket, dict):
            raise ValidationError(f"Ticket #{idx} is not an object")

        missing = REQUIRED_TICKET_FIELDS - ticket.keys()
        if missing:
            raise ValidationError(
                f"Ticket #{idx} ({ticket.get('ticket_id', '?')}) missing fields: {sorted(missing)}"
            )

        tid = ticket["ticket_id"]
        if not isinstance(tid, str) or not tid.strip():
            raise ValidationError(f"Ticket #{idx} has empty ticket_id")
        if tid in seen_ids:
            raise ValidationError(f"Duplicate ticket_id: {tid}")
        seen_ids.add(tid)

        try:
            datetime.fromisoformat(ticket["created_at"].replace("Z", "+00:00"))
        except (ValueError, AttributeError) as exc:
            raise ValidationError(
                f"Ticket {tid} has unparseable created_at: {ticket.get('created_at')!r}"
            ) from exc

        if not isinstance(ticket["message"], str) or not ticket["message"].strip():
            raise ValidationError(f"Ticket {tid} has empty message")

        if not isinstance(ticket["language"], str) or not ticket["language"].strip():
            raise ValidationError(f"Ticket {tid} has empty language")

        if not isinstance(ticket["attachments"], list):
            raise ValidationError(f"Ticket {tid} attachments must be a list")

        if not isinstance(ticket["metadata"], dict):
            raise ValidationError(f"Ticket {tid} metadata must be an object")

    return tickets


def _validate_kb(articles: Any) -> list[dict]:
    if not isinstance(articles, list):
        raise ValidationError("kb.json must be a JSON array")

    seen_ids: set[str] = set()
    for idx, article in enumerate(articles):
        if not isinstance(article, dict):
            raise ValidationError(f"KB article #{idx} is not an object")

        missing = REQUIRED_KB_FIELDS - article.keys()
        if missing:
            raise ValidationError(
                f"KB article #{idx} ({article.get('article_id', '?')}) missing fields: {sorted(missing)}"
            )

        aid = article["article_id"]
        if not isinstance(aid, str) or not aid.strip():
            raise ValidationError(f"KB article #{idx} has empty article_id")
        if aid in seen_ids:
            raise ValidationError(f"Duplicate article_id: {aid}")
        seen_ids.add(aid)

        if not isinstance(article["content"], str) or not article["content"].strip():
            raise ValidationError(f"KB article {aid} has empty content")

        if not isinstance(article["tags"], list):
            raise ValidationError(f"KB article {aid} tags must be a list")

        if not isinstance(article["safe_for_ai"], bool):
            raise ValidationError(f"KB article {aid} safe_for_ai must be boolean")

    return articles


def load_inputs() -> tuple[list[dict], list[dict]]:
    """Read + validate tickets.json and kb.json. Halts the pipeline on any error."""
    tickets = _validate_tickets(_read_json(TICKETS_IN))
    articles = _validate_kb(_read_json(KB_IN))
    return tickets, articles
