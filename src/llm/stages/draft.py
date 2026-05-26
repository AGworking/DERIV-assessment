"""Stage: LLM draft generation for AUTO_DRAFT tickets only.

We deliberately gate the LLM call on the deterministic final route. The LLM
never sees tickets that are escalated, insufficient, or sent to human review —
that's the whole point of the safety layer being upstream.

Each call:
  1. Builds a constrained prompt referencing only the top-3 KB snippets
     for this ticket (grounding).
  2. Sends it through the configured provider (mock by default, OpenAI behind
     LLM_PROVIDER=openai).
  3. Parses the JSON output into {draft_reply, used_article_ids, confidence}.
  4. Appends one record to artifacts/llm_calls.jsonl.

If the LLM returns malformed JSON, we capture the raw output and mark the
ticket as low confidence so a human still sees it before sending.
"""

from __future__ import annotations

import json
from typing import Iterable

from src.llm.logger import log_call, prompt_hash
from src.llm.provider import get_provider
from src.config.paths import (
    DRAFTS, FINAL_ROUTES, KB_IN, RETRIEVAL, TICKETS_IN, ensure_artifacts_dir,
)


SYSTEM_PROMPT = (
    "You are a customer support agent drafting reply emails. "
    "You MUST follow these rules without exception:\n"
    "1. Use ONLY the knowledge base snippets provided in the user message. "
    "Do not invent policies, timeframes, or actions that are not in the snippets.\n"
    "2. Cite the article_ids you used in the 'used_article_ids' field.\n"
    "3. If the snippets don't fully cover the question, ask the customer for "
    "the specific missing detail (e.g. transaction reference, error screenshot).\n"
    "4. Keep the tone professional, calm, and concise (under ~150 words).\n"
    "5. Do not promise refunds, account deletions, or any irreversible action.\n"
    "6. Respond in JSON only with this exact shape:\n"
    "   {\"draft_reply\": str, \"used_article_ids\": [str], "
    "\"confidence\": \"low\" | \"medium\" | \"high\"}"
)


def _build_user_prompt(ticket: dict, snippets: list[dict]) -> dict:
    """Compact payload passed to the LLM. Both providers parse this same shape."""
    return {
        "ticket_id": ticket["ticket_id"],
        "subject": ticket.get("subject", ""),
        "message": ticket.get("message", ""),
        "customer_tier": ticket.get("customer_tier"),
        "snippets": [
            {
                "article_id": s["article_id"],
                "title": s["title"],
                "content": s["content"],
                "score": s.get("score"),
            }
            for s in snippets
        ],
    }


def _snippets_for(ticket_id: str, retrieval: list[dict], kb_by_id: dict) -> list[dict]:
    for r in retrieval:
        if r["ticket_id"] == ticket_id:
            out = []
            for c in r["candidates"]:
                article = kb_by_id[c["article_id"]]
                out.append({
                    "article_id": c["article_id"],
                    "title": article["title"],
                    "content": article["content"],
                    "score": c["score"],
                    "safe_for_ai": c["safe_for_ai"],
                })
            return out
    return []


def _coerce_draft(raw_text: str, ticket_id: str) -> dict:
    """Parse the LLM response. If invalid JSON, return a low-confidence shell."""
    try:
        obj = json.loads(raw_text)
    except json.JSONDecodeError:
        return {
            "ticket_id": ticket_id,
            "draft_reply": raw_text.strip()[:600] or "(LLM returned no parseable content.)",
            "used_article_ids": [],
            "confidence": "low",
            "parse_error": True,
        }

    return {
        "ticket_id": ticket_id,
        "draft_reply": str(obj.get("draft_reply", "")).strip(),
        "used_article_ids": [str(x) for x in (obj.get("used_article_ids") or [])],
        "confidence": str(obj.get("confidence", "medium")).lower(),
    }


def generate_drafts(
    final_routes: list[dict],
    retrieval: list[dict],
    tickets: list[dict],
    kb_articles: list[dict],
) -> list[dict]:
    kb_by_id = {a["article_id"]: a for a in kb_articles}
    tickets_by_id = {t["ticket_id"]: t for t in tickets}

    provider = get_provider()
    drafts: list[dict] = []

    for route in final_routes:
        if route["final_route"] != "AUTO_DRAFT":
            continue

        tid = route["ticket_id"]
        ticket = tickets_by_id[tid]
        # Only pass through safe_for_ai snippets — defence in depth.
        snippets = [
            s for s in _snippets_for(tid, retrieval, kb_by_id)
            if s.get("safe_for_ai", True)
        ]

        user_payload = _build_user_prompt(ticket, snippets)
        user_str = json.dumps(user_payload, ensure_ascii=False)

        raw = provider.generate(SYSTEM_PROMPT, user_str)
        draft = _coerce_draft(raw, tid)
        drafts.append(draft)

        log_call(
            stage="draft_generation",
            ticket_id=tid,
            provider=provider.name,
            model=provider.model,
            prompt_hash_value=prompt_hash(SYSTEM_PROMPT + "\n" + user_str),
            input_artifacts=[TICKETS_IN, KB_IN, RETRIEVAL, FINAL_ROUTES],
            output_artifact=DRAFTS,
        )

    ensure_artifacts_dir()
    DRAFTS.write_text(
        json.dumps(drafts, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return drafts
