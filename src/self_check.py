"""Stretch stage: LLM self-check on every auto-generated draft.

Asks the LLM (or the mock) whether each draft contains unsupported claims,
missing caveats, or advice not present in the retrieved evidence.

Crucial constraint from the spec:
    "Do not let the LLM silently overwrite the original draft."

So we write the review into a separate artifact (draft_reviews.json) and never
mutate drafts.json. Operations can then decide whether to ship, edit, or block
each draft based on the review.
"""

from __future__ import annotations

import json

from src.llm.logger import log_call, prompt_hash
from src.llm.provider import get_provider
from src.paths import (
    DRAFT_REVIEWS, DRAFTS, KB_IN, RETRIEVAL, TICKETS_IN, ensure_artifacts_dir,
)


SYSTEM_PROMPT = (
    "You are reviewing a draft customer-support reply for grounding errors. "
    "You will be given the original ticket, the retrieved KB snippets the "
    "drafter saw, and the draft itself. Your job is to flag any of the following:\n"
    "  - claims or facts not supported by the snippets\n"
    "  - missing caveats (e.g. processing time ranges that the snippet says exist)\n"
    "  - cited article_ids that are not in the retrieved evidence\n"
    "  - promises of irreversible actions (refunds, account deletion)\n"
    "Respond with JSON: {\"supported\": bool, \"issues\": [str], "
    "\"recommendation\": \"ship\" | \"edit\" | \"block\"}. "
    "If there are no issues, supported=true, issues=[], recommendation=\"ship\"."
)


def _snippets_for(ticket_id: str, retrieval: list[dict], kb_by_id: dict) -> list[dict]:
    for r in retrieval:
        if r["ticket_id"] == ticket_id:
            return [
                {
                    "article_id": c["article_id"],
                    "title": kb_by_id[c["article_id"]]["title"],
                    "content": kb_by_id[c["article_id"]]["content"],
                    "safe_for_ai": c["safe_for_ai"],
                }
                for c in r["candidates"]
            ]
    return []


def _coerce_review(raw: str, ticket_id: str) -> dict:
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError:
        return {
            "ticket_id": ticket_id,
            "supported": False,
            "issues": ["Self-check LLM returned unparseable output."],
            "recommendation": "edit",
            "parse_error": True,
        }

    return {
        "ticket_id": ticket_id,
        "supported": bool(obj.get("supported", False)),
        "issues": [str(x) for x in (obj.get("issues") or [])],
        "recommendation": str(obj.get("recommendation", "edit")).lower(),
    }


def review_drafts(
    drafts: list[dict], retrieval: list[dict], tickets: list[dict],
    kb_articles: list[dict],
) -> list[dict]:
    kb_by_id = {a["article_id"]: a for a in kb_articles}
    tickets_by_id = {t["ticket_id"]: t for t in tickets}
    provider = get_provider()

    reviews: list[dict] = []
    for draft in drafts:
        tid = draft["ticket_id"]
        ticket = tickets_by_id[tid]
        snippets = _snippets_for(tid, retrieval, kb_by_id)

        payload = {
            "task": "self_check",
            "ticket_id": tid,
            "subject": ticket.get("subject", ""),
            "message": ticket.get("message", ""),
            "snippets": snippets,
            "draft_reply": draft.get("draft_reply", ""),
            "used_article_ids": draft.get("used_article_ids", []),
        }
        user_str = json.dumps(payload, ensure_ascii=False)

        raw = provider.generate(SYSTEM_PROMPT, user_str)
        review = _coerce_review(raw, tid)
        reviews.append(review)

        log_call(
            stage="draft_self_check",
            ticket_id=tid,
            provider=provider.name,
            model=provider.model,
            prompt_hash_value=prompt_hash(SYSTEM_PROMPT + "\n" + user_str),
            input_artifacts=[TICKETS_IN, KB_IN, RETRIEVAL, DRAFTS],
            output_artifact=DRAFT_REVIEWS,
        )

    ensure_artifacts_dir()
    DRAFT_REVIEWS.write_text(
        json.dumps(reviews, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return reviews
