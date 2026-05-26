"""Mock LLM provider.

Builds grounded outputs by templating retrieved KB content directly. No network
calls, no API key required, fully deterministic — the default provider so the
pipeline runs end-to-end on a clean checkout.

The mock dispatches on `payload["task"]` so the same `.generate()` call handles
all three downstream stages:
    - draft         (src/llm/stages/draft.py)
    - self_check    (src/llm/stages/self_check.py)
    - clarify       (src/llm/stages/clarify.py)

For the real LLM path (LLMProvider=openai), see openai_provider.py.
"""

from __future__ import annotations

import json
import re

from src.llm.base import LLMProvider


_SUMMARY_SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class MockProvider(LLMProvider):
    name = "mock"
    model = "mock-grounded-templater-v1"

    def generate(self, system: str, user: str) -> str:
        try:
            payload = json.loads(user)
        except json.JSONDecodeError:
            return json.dumps({
                "draft_reply": "Thank you for contacting support. A human agent will follow up shortly.",
                "used_article_ids": [],
                "confidence": "low",
            })

        task = payload.get("task", "draft")
        if task == "self_check":
            return self._build_self_check(payload)
        if task == "clarify":
            return self._build_clarification(payload)
        return self._build_reply(payload)

    # -- task: draft ------------------------------------------------------

    def _build_reply(self, payload: dict) -> str:
        subject = (payload.get("subject") or "").strip()
        snippets = payload.get("snippets") or []

        if not snippets:
            return json.dumps({
                "draft_reply": (
                    "Thanks for reaching out. We need a little more information to help — "
                    "could you share the relevant transaction reference and a screenshot of the issue?"
                ),
                "used_article_ids": [],
                "confidence": "low",
            })

        primary = snippets[0]
        cited_ids = [primary["article_id"]]
        if len(snippets) > 1 and snippets[1].get("score", 0.0) >= 0.10:
            cited_ids.append(snippets[1]["article_id"])

        summary = self._summarise(primary["content"])
        body = (
            f"Hi,\n\nThanks for reaching out about \"{subject}\". {summary} "
            "If the issue persists after trying this, please reply with any "
            "reference numbers or screenshots and we'll escalate.\n\n"
            "Best regards,\nSupport Team\n\n"
            f"(Reference: {', '.join(cited_ids)})"
        )
        confidence = "high" if primary.get("score", 0.0) >= 0.3 else "medium"

        return json.dumps({
            "draft_reply": body,
            "used_article_ids": cited_ids,
            "confidence": confidence,
        })

    # -- task: self_check -------------------------------------------------

    def _build_self_check(self, payload: dict) -> str:
        """Conservative deterministic self-check.

        The mock can't reason semantically about a draft, so it only flags
        structural problems (empty draft, no citations, citations outside the
        retrieved set, and a few forbidden promise keywords). Real semantic
        grading happens when LLM_PROVIDER=openai is enabled.
        """
        draft = payload.get("draft_reply", "") or ""
        cited = set(payload.get("used_article_ids") or [])
        allowed_ids = {s["article_id"] for s in (payload.get("snippets") or [])}

        issues: list[str] = []
        if not draft.strip():
            issues.append("Draft is empty.")
        if not cited:
            issues.append("Draft cites no article_ids — every AI reply must reference at least one KB source.")
        unknown = cited - allowed_ids
        if unknown:
            issues.append(
                f"Draft cites article_ids that were not in the retrieved evidence: {sorted(unknown)}."
            )

        forbidden = [
            ("refund", "Draft mentions 'refund' — refunds require finance team review and cannot be promised."),
            ("immediately", "Draft promises 'immediate' action — escalations may be required first."),
            ("guarantee", "Draft uses the word 'guarantee' — avoid absolute promises."),
        ]
        lower = draft.lower()
        for word, msg in forbidden:
            if word in lower:
                issues.append(msg)

        supported = not issues
        return json.dumps({
            "supported": supported,
            "issues": issues,
            "recommendation": "ship" if supported else "edit",
        })

    # -- task: clarify ----------------------------------------------------

    def _build_clarification(self, payload: dict) -> str:
        subject = (payload.get("subject") or "").strip()
        message = (payload.get("message") or "").strip()

        return json.dumps({
            "clarification_question": (
                "Thanks for reaching out. We need a bit more information to help — "
                "could you share what you were trying to do, any error message you saw, "
                "and a reference number or screenshot if available?"
            ),
            "suggested_info_to_request": [
                "exact error message (if any)",
                "transaction reference or order ID",
                "screenshot of the issue",
                "the email address on file",
            ],
            "context_subject": subject,
            "context_message": message[:200],
        })

    # -- helpers ----------------------------------------------------------

    @staticmethod
    def _summarise(content: str) -> str:
        """First 1-2 sentences of the KB content — short, grounded, no invention."""
        sentences = _SUMMARY_SENTENCE_RE.split(content.strip())
        return " ".join(sentences[:2]).strip()
