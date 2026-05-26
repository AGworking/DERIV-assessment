"""Stretch stage: clarification drafts for INSUFFICIENT_CONTEXT tickets.

When a ticket is too thin for the deterministic safety gate to act on
(e.g. "help please"), instead of letting the LLM guess what the customer wants,
we ask the LLM to draft a short clarification question template the human
agent can send out as-is or lightly edit.
"""

from __future__ import annotations

import json

from src.llm.logger import log_call, prompt_hash
from src.llm.provider import get_provider
from src.config.paths import (
    CLARIFICATIONS, FINAL_ROUTES, TICKETS_IN, ensure_artifacts_dir,
)


SYSTEM_PROMPT = (
    "You are drafting a short clarification question to send to a customer "
    "whose support ticket did not contain enough information for us to act on. "
    "Be polite, concise, and ask for the specific details we would need to "
    "help (e.g. an error message, a transaction reference, a screenshot). "
    "Do not assume the nature of the problem. "
    "Respond with JSON: {\"clarification_question\": str, "
    "\"suggested_info_to_request\": [str]}."
)


def _coerce(raw: str, ticket_id: str) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "ticket_id": ticket_id,
            "clarification_question": raw.strip()[:400] or "(no clarification text generated)",
            "suggested_info_to_request": [],
            "parse_error": True,
        }

    return {
        "ticket_id": ticket_id,
        "clarification_question": str(obj.get("clarification_question", "")).strip(),
        "suggested_info_to_request": [
            str(x) for x in (obj.get("suggested_info_to_request") or [])
        ],
    }


def generate_clarifications(
    final_routes: list[dict], tickets: list[dict],
) -> list[dict]:
    tickets_by_id = {t["ticket_id"]: t for t in tickets}
    provider = get_provider()

    out: list[dict] = []
    for route in final_routes:
        if route["final_route"] != "INSUFFICIENT_CONTEXT":
            continue

        tid = route["ticket_id"]
        ticket = tickets_by_id[tid]

        payload = {
            "task": "clarify",
            "ticket_id": tid,
            "subject": ticket.get("subject", ""),
            "message": ticket.get("message", ""),
            "channel": ticket.get("channel"),
        }
        user_str = json.dumps(payload, ensure_ascii=False)

        raw = provider.generate(SYSTEM_PROMPT, user_str)
        out.append(_coerce(raw, tid))

        log_call(
            stage="clarification_generation",
            ticket_id=tid,
            provider=provider.name,
            model=provider.model,
            prompt_hash_value=prompt_hash(SYSTEM_PROMPT + "\n" + user_str),
            input_artifacts=[TICKETS_IN, FINAL_ROUTES],
            output_artifact=CLARIFICATIONS,
        )

    ensure_artifacts_dir()
    CLARIFICATIONS.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out
