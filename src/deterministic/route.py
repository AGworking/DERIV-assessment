"""Stage: safety gate + final routing.

Combines deterministic rules (src/deterministic/classify.py) with retrieval
evidence (src/deterministic/retrieve.py) to produce the FINAL route for every
ticket, with a human-readable reason and a list of top article IDs.

This stage is deterministic. The LLM is downstream of it.

Decision precedence (first matching rule wins):

  1. Rules say the ticket is too thin to act on        -> INSUFFICIENT_CONTEXT
  2. Intent is fraud / unauthorized / security          -> ESCALATE_RISK
  3. The top KB article is policy-restricted
     (safe_for_ai = False) AND the rule layer agrees
     the ticket touches a policy topic                  -> ESCALATE_POLICY
  4. Rules already chose ESCALATE_POLICY / ESCALATE_RISK -> keep
  5. The top retrieval score is below the floor
     (weak evidence — we don't want the LLM to
     hallucinate to fill the gap)                       -> HUMAN_REVIEW
  6. Top article is safe_for_ai AND rules permitted     -> AUTO_DRAFT
  7. Anything else                                      -> HUMAN_REVIEW
"""

from __future__ import annotations

import json

from src.config.paths import FINAL_ROUTES, ensure_artifacts_dir


# Below this top-1 cosine similarity we treat retrieval as too weak to ground
# an AI reply. Tuned against the actual fixture range (genuine hits sit at
# 0.14–0.55; junk hits are 0.0–0.05).
RETRIEVAL_SCORE_FLOOR = 0.10


def _decide(classification: dict, candidates: list[dict]) -> tuple[str, str]:
    intent = classification["intent"]
    prelim = classification["preliminary_route"]
    policy = classification["contains_policy_topic"]
    needs_more = classification["needs_more_context"]

    top = candidates[0] if candidates else None
    top_score = top["score"] if top else 0.0
    top_safe = bool(top["safe_for_ai"]) if top else False
    top_id = top["article_id"] if top else None

    # 1. Insufficient context — rules already said the message is too thin.
    if needs_more:
        return (
            "INSUFFICIENT_CONTEXT",
            f"Message is too short or generic to act on (rules flagged needs_more_context). "
            f"Top retrieval score {top_score:.3f} (article {top_id}) below useful threshold.",
        )

    # 2. Fraud / security incident — never let the LLM near it.
    if intent == "fraud_security":
        return (
            "ESCALATE_RISK",
            f"Ticket flagged as fraud_security by rules; top KB hit '{top_id}' "
            f"({'safe_for_ai' if top_safe else 'policy-restricted'}). "
            f"Security incidents must be handled by a human.",
        )

    # 3. Policy-restricted top hit + rules concur the ticket is policy-loaded.
    if top is not None and (not top_safe) and policy:
        return (
            "ESCALATE_POLICY",
            f"Top retrieved article '{top_id}' is marked safe_for_ai=False and "
            f"the rule layer detected a policy topic (intent={intent}). "
            f"AI drafting blocked by policy.",
        )

    # 4. Rules already escalated for policy/risk reasons — keep that decision.
    if prelim in {"ESCALATE_POLICY", "ESCALATE_RISK"}:
        return (
            prelim,
            f"Rule-layer escalation retained (intent={intent}, policy={policy}). "
            f"Top retrieved article '{top_id}' (score {top_score:.3f}).",
        )

    # 5. Weak retrieval — better to ask a human than hallucinate.
    if top_score < RETRIEVAL_SCORE_FLOOR:
        return (
            "HUMAN_REVIEW",
            f"Retrieval evidence is weak (top score {top_score:.3f} < floor "
            f"{RETRIEVAL_SCORE_FLOOR}). Sending to a human reviewer rather than "
            f"risk an unsupported AI reply.",
        )

    # 6. Safe to auto-draft.
    if top_safe and prelim == "AUTO_DRAFT":
        return (
            "AUTO_DRAFT",
            f"Rule layer cleared the ticket (intent={intent}, non-policy, "
            f"sufficient context) and top article '{top_id}' is safe_for_ai "
            f"with score {top_score:.3f} >= floor {RETRIEVAL_SCORE_FLOOR}.",
        )

    # 7. Fallback — anything ambiguous becomes a human review.
    return (
        "HUMAN_REVIEW",
        f"Did not meet AUTO_DRAFT criteria (top_safe={top_safe}, "
        f"top_score={top_score:.3f}, preliminary={prelim}). Routing to human review.",
    )


def decide_final_routes(
    classifications: list[dict], retrieval: list[dict],
) -> list[dict]:
    cls_by_id = {c["ticket_id"]: c for c in classifications}
    out: list[dict] = []

    for r in retrieval:
        tid = r["ticket_id"]
        classification = cls_by_id[tid]
        candidates = r["candidates"]

        final_route, reason = _decide(classification, candidates)
        out.append({
            "ticket_id": tid,
            "intent": classification["intent"],
            "urgency": classification["urgency"],
            "preliminary_route": classification["preliminary_route"],
            "final_route": final_route,
            "reason": reason,
            "top_article_ids": [c["article_id"] for c in candidates],
        })

    ensure_artifacts_dir()
    FINAL_ROUTES.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out
