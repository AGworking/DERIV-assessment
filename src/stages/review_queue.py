"""Stage: produce the human review queue for all non-AUTO_DRAFT tickets.

Each entry includes a concise summary, the top retrieved evidence, and a
recommended next action so a human agent can pick up the ticket without
re-reading the entire chain.
"""

from __future__ import annotations

import json

from src.config.paths import REVIEW_QUEUE, ensure_artifacts_dir


# Map (intent, route) → recommended action. Falls back to a per-route default.
INTENT_RECOMMENDATIONS = {
    ("account_closure", "ESCALATE_POLICY"): "Escalate to the policy specialist for account closure: confirm balance is withdrawn before closure and do not promise immediate deletion.",
    ("refund", "ESCALATE_POLICY"): "Escalate to the finance team for refund review with the transaction reference.",
    ("fraud_security", "ESCALATE_RISK"): "Escalate immediately to the security team to review for possible account freeze and incident investigation.",
}

ROUTE_DEFAULT_RECOMMENDATIONS = {
    "ESCALATE_POLICY": "Escalate to the policy specialist — AI drafting is blocked because the relevant KB article is marked safe_for_ai=false.",
    "ESCALATE_RISK": "Escalate to the security/risk team — this ticket carries a risk signal that AI must not address.",
    "HUMAN_REVIEW": "Review manually — retrieval evidence was weak or the rule layer marked the ticket as borderline.",
    "INSUFFICIENT_CONTEXT": "Reply to the customer with a clarification request asking for the specific missing detail (transaction reference, error screenshot, exact error message).",
}


def _summarise(ticket: dict) -> str:
    subject = ticket.get("subject") or ""
    message = ticket.get("message") or ""
    snippet = message[:160] + ("…" if len(message) > 160 else "")
    return f"{subject} — {snippet}".strip(" —")


def _recommendation(intent: str, route: str) -> str:
    key = (intent, route)
    return INTENT_RECOMMENDATIONS.get(key) or ROUTE_DEFAULT_RECOMMENDATIONS.get(
        route, "Manual review by a support agent."
    )


def build_review_queue(
    final_routes: list[dict], retrieval: list[dict], tickets: list[dict],
) -> list[dict]:
    tickets_by_id = {t["ticket_id"]: t for t in tickets}
    retrieval_by_id = {r["ticket_id"]: r for r in retrieval}

    out: list[dict] = []
    for route in final_routes:
        if route["final_route"] == "AUTO_DRAFT":
            continue

        tid = route["ticket_id"]
        ticket = tickets_by_id[tid]
        candidates = retrieval_by_id[tid]["candidates"]
        top = candidates[0] if candidates else None

        out.append({
            "ticket_id": tid,
            "final_route": route["final_route"],
            "intent": route["intent"],
            "urgency": route["urgency"],
            "summary": _summarise(ticket),
            "reason": route["reason"],
            "top_evidence": (
                {
                    "article_id": top["article_id"],
                    "title": top["title"],
                    "score": top["score"],
                    "safe_for_ai": top["safe_for_ai"],
                }
                if top is not None
                else None
            ),
            "recommended_next_action": _recommendation(route["intent"], route["final_route"]),
        })

    ensure_artifacts_dir()
    REVIEW_QUEUE.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out
