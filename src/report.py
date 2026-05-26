"""Stage: produce ops_report.md — the human-readable run summary.

Sections required by the spec:
  - Executive Summary
  - Route Distribution
  - Retrieval Quality Notes
  - Auto-Drafted Tickets
  - Human Review / Escalation Queue
  - Safety Constraints
  - Known Limitations
"""

from __future__ import annotations

from collections import Counter
from statistics import mean

from src.paths import OPS_REPORT, ensure_artifacts_dir


def _route_distribution_table(final_routes: list[dict]) -> str:
    hist = Counter(r["final_route"] for r in final_routes)
    total = sum(hist.values()) or 1
    rows = ["| Route | Count | Share |", "|---|---:|---:|"]
    for route, count in sorted(hist.items(), key=lambda kv: -kv[1]):
        rows.append(f"| `{route}` | {count} | {count/total:.0%} |")
    rows.append(f"| **Total** | **{total}** | **100%** |")
    return "\n".join(rows)


def _retrieval_quality(retrieval: list[dict]) -> str:
    top_scores = [r["candidates"][0]["score"] for r in retrieval if r["candidates"]]
    if not top_scores:
        return "_No retrieval results to summarise._"
    avg = mean(top_scores)
    weakest = min(top_scores)
    strong = sum(1 for s in top_scores if s >= 0.10)
    weak = len(top_scores) - strong
    return (
        f"- Average top-1 cosine similarity: **{avg:.3f}**\n"
        f"- Weakest top-1 score: **{weakest:.3f}**\n"
        f"- Tickets with usable retrieval (top-1 >= 0.10): **{strong}/{len(top_scores)}**\n"
        f"- Tickets with weak retrieval (top-1 < 0.10): **{weak}/{len(top_scores)}** — these are routed to `INSUFFICIENT_CONTEXT` or `HUMAN_REVIEW`."
    )


def _draft_section(drafts: list[dict], final_routes: list[dict]) -> str:
    by_id = {d["ticket_id"]: d for d in drafts}
    lines = ["| Ticket | Intent | Confidence | Cited Articles |", "|---|---|---|---|"]
    for r in final_routes:
        if r["final_route"] != "AUTO_DRAFT":
            continue
        d = by_id.get(r["ticket_id"])
        if not d:
            continue
        cited = ", ".join(d["used_article_ids"]) or "—"
        lines.append(f"| `{r['ticket_id']}` | {r['intent']} | {d['confidence']} | {cited} |")
    if len(lines) == 2:
        return "_No tickets were auto-drafted._"
    return "\n".join(lines)


def _review_section(review_queue: list[dict]) -> str:
    if not review_queue:
        return "_No tickets in the review queue._"
    lines = [
        "| Ticket | Route | Intent | Recommended Next Action |",
        "|---|---|---|---|",
    ]
    for item in review_queue:
        lines.append(
            f"| `{item['ticket_id']}` | `{item['final_route']}` | "
            f"{item['intent']} | {item['recommended_next_action']} |"
        )
    return "\n".join(lines)


def render_report(
    *,
    final_routes: list[dict],
    retrieval: list[dict],
    drafts: list[dict],
    review_queue: list[dict],
    provider_name: str,
    provider_model: str,
) -> str:
    total = len(final_routes)
    auto = sum(1 for r in final_routes if r["final_route"] == "AUTO_DRAFT")
    escalated = sum(
        1 for r in final_routes
        if r["final_route"] in {"ESCALATE_POLICY", "ESCALATE_RISK"}
    )

    md = f"""# Support Copilot — Ops Report

## Executive Summary

- **Total tickets processed:** {total}
- **Auto-drafted by AI:** {auto}
- **Escalated (policy or risk):** {escalated}
- **Routed for human review or clarification:** {total - auto - escalated}
- **LLM provider used for drafting:** `{provider_name}` (model: `{provider_model}`)

The deterministic safety layer (rules + retrieval + safety gate) decided every
final route before any LLM call was issued. The LLM was only invoked for the
{auto} tickets that the safety layer cleared as `AUTO_DRAFT`.

## Route Distribution

{_route_distribution_table(final_routes)}

## Retrieval Quality Notes

{_retrieval_quality(retrieval)}

The retrieval score floor of **0.10** is used by the safety gate to demote
tickets with weak evidence to `HUMAN_REVIEW` rather than risk an
unsupported AI reply.

## Auto-Drafted Tickets

{_draft_section(drafts, final_routes)}

Every auto-drafted reply must cite the article IDs it relied on. The mock
provider templates the KB content directly so the citation is always
grounded; the OpenAI provider is prompted to do the same.

## Human Review / Escalation Queue

{_review_section(review_queue)}

## Safety Constraints

The pipeline blocks the LLM from drafting whenever **any** of the following
hold (all enforced in deterministic Python, not in the prompt):

1. Rule layer detects a policy topic (account closure, refund, chargeback,
   fraud, suspected unauthorized access).
2. Top retrieved KB article is marked `safe_for_ai = false`.
3. Ticket intent is `fraud_security` — always escalated to risk, regardless of
   retrieval.
4. Message is too short or generic to act on (`needs_more_context`).
5. Top retrieval cosine similarity is below `0.10` — evidence is too weak to
   ground a confident reply.

The LLM is never the sole decider of routing.

## Known Limitations

- The rule layer relies on English-keyword regexes; non-English tickets would
  need separate patterns (the `language` field is normalised but not yet
  branched on).
- The mock LLM provider templates retrieved KB content rather than truly
  composing a reply. It is safe-by-construction but not as fluent as a real
  LLM would be. Set `LLM_PROVIDER=openai` (with `OPENAI_API_KEY`) to switch.
- TF-IDF retrieval is exact-token based. Synonyms (e.g. "topup" vs "deposit")
  must be present in the KB tags to match. An embedding-based retriever
  would improve recall but is out of scope for this exercise.
"""
    ensure_artifacts_dir()
    OPS_REPORT.write_text(md, encoding="utf-8")
    return md
