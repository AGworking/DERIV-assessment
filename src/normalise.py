"""Stage: normalise validated tickets into a consistent internal representation.

Adds lowercased text fields for downstream retrieval / classification,
plus derived fields (text length, attachment count, logged_in flag).
"""

from __future__ import annotations

import json
import re

from src.paths import TICKETS_NORMALISED, ensure_artifacts_dir

_WS_RE = re.compile(r"\s+")


def _clean(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def normalise_tickets(tickets: list[dict]) -> list[dict]:
    out: list[dict] = []
    for t in tickets:
        subject = _clean(t.get("subject", ""))
        message = _clean(t.get("message", ""))
        combined = f"{subject}. {message}".strip()

        metadata = t.get("metadata") or {}
        logged_in = bool(metadata.get("logged_in", False))

        out.append({
            "ticket_id": t["ticket_id"],
            "created_at": t["created_at"],
            "channel": t.get("channel"),
            "language": t.get("language"),
            "customer_tier": t.get("customer_tier"),
            "country": metadata.get("country"),
            "logged_in": logged_in,
            "subject": subject,
            "message": message,
            "subject_lower": subject.lower(),
            "message_lower": message.lower(),
            "text_lower": combined.lower(),
            "text_length": len(combined),
            "attachment_count": len(t.get("attachments") or []),
            "attachments": list(t.get("attachments") or []),
        })

    ensure_artifacts_dir()
    TICKETS_NORMALISED.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out
