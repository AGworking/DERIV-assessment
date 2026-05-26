"""Stage: deterministic, rule-based classification.

Produces:
  - intent (deposit / account_closure / kyc / password_reset / login_2fa /
            withdrawal / refund / fraud_security / other)
  - urgency (low / medium / high)
  - contains_policy_topic (bool)
  - needs_human (bool)
  - needs_more_context (bool)
  - preliminary_route (one of the five final routes)

This is intentionally non-LLM. The safety gate (src/route.py) refines
the preliminary route using retrieval evidence.

The keyword groups are deliberately broad-but-specific English phrases so
that swapped-in fixtures still trigger correctly.
"""

from __future__ import annotations

import json
import re
from typing import Iterable

from src.paths import ROUTING, ensure_artifacts_dir


# Keyword groups per intent. We match whole-word patterns against text_lower.
INTENT_PATTERNS: dict[str, list[str]] = {
    "account_closure": [
        r"\bclose\s+my\s+account\b", r"\bclose\s+account\b",
        r"\bdelete\s+my\s+account\b", r"\bdelete\s+account\b",
        r"\baccount\s+deletion\b", r"\baccount\s+closure\b",
        r"\bremove\s+my\s+account\b", r"\bdeactivate\s+my\s+account\b",
    ],
    "refund": [
        r"\brefund\b", r"\bmoney\s+back\b", r"\bchargeback\b", r"\breimburse\b",
    ],
    "fraud_security": [
        r"\bfraud\b", r"\bunauthori[sz]ed\b", r"\bhacked\b", r"\bstolen\b",
        r"\bsuspicious\b", r"\bcompromis(?:ed|e)\b", r"\bphish(?:ing)?\b",
        r"\bsomeone\s+(?:accessed|used|logged)\b", r"\bfreeze\s+my\s+account\b",
    ],
    "kyc": [
        r"\bkyc\b", r"\bverify\s+(?:my\s+)?identit(?:y|ies)\b",
        r"\bidentity\s+verification\b", r"\bdocument\s+upload\b",
        r"\bupload(?:ing)?\s+(?:my\s+)?(?:document|id|passport)\b",
        r"\bunsupported\s+format\b", r"\bverification\s+(?:fail|error)\b",
        r"\bverify\s+identity\b", r"\bcan'?t\s+verify\b",
    ],
    "password_reset": [
        r"\bforgot\s+(?:my\s+)?password\b", r"\bpassword\s+reset\b",
        r"\breset\s+(?:my\s+)?password\b", r"\breset\s+email\b",
        r"\bcan'?t\s+log\s*in\b.*\bpassword\b",
    ],
    "login_2fa": [
        r"\b2fa\b", r"\btwo[-\s]?factor\b", r"\bauthenticator\b",
        r"\bsms\s+code\b", r"\bverification\s+code\b", r"\bauth\s+code\b",
        r"\bone[-\s]?time\s+(?:password|code)\b", r"\botp\b",
    ],
    "withdrawal": [
        r"\bwithdraw(?:al|ing|n)?\b", r"\bpayout\b", r"\bcash\s*out\b",
        r"\bwithdraw\s+(?:my\s+)?(?:funds|money|balance)\b",
    ],
    "deposit": [
        r"\bdeposit\b", r"\bbank\s+transfer\b", r"\btransfer(?:red)?\b.*\b(?:funds|money|balance|usd|eur|gbp)\b",
        r"\bbalance\s+(?:still\s+)?(?:says|shows|is)\s*0\b",
        r"\bfunds\s+not\s+(?:showing|appearing|received)\b",
        r"\bmoney\s+not\s+(?:showing|appearing)\b",
        r"\btopup\b", r"\btop[-\s]?up\b",
    ],
}

# Words that bump urgency upward.
URGENCY_HIGH_PATTERNS = [
    r"\burgent(?:ly)?\b", r"\bimmediat(?:e|ely)\b", r"\basap\b",
    r"\bemergency\b", r"\bcritical\b", r"\bnow\b",
    r"\bright\s+now\b", r"\bplease\s+help\s+(?:me\s+)?now\b",
]
URGENCY_MEDIUM_PATTERNS = [
    r"\bstill\s+(?:not|hasn'?t|haven'?t)\b",
    r"\bnot\s+working\b", r"\bkeeps?\s+failing\b",
    r"\b(?:\d+|two|three|four|five|six)\s+(?:hour|day|week)s?\s+ago\b",
    r"\bsince\s+(?:yesterday|last\s+week)\b",
]

# Policy keyword set drives contains_policy_topic regardless of intent
# (in case the regex misses but the topic is still policy-loaded).
POLICY_KEYWORDS = [
    "close my account", "delete my account", "account deletion", "account closure",
    "refund", "chargeback", "money back",
    "fraud", "unauthorized", "unauthorised", "hacked", "stolen",
    "compromised", "phishing", "freeze my account",
]

# Minimum useful message length. Anything under this is too thin to act on
# without clarification.
SHORT_TEXT_THRESHOLD = 30

# Generic non-actionable phrases — if the message is basically only these,
# we treat it as missing context regardless of length.
GENERIC_PHRASES = {
    "help", "help please", "please help", "hi", "hello", "broken",
    "not working", "it's broken", "its broken", "doesn't work", "doesnt work",
    "fix this", "fix it", "please fix", "support",
}


def _any_match(patterns: Iterable[str], text: str) -> bool:
    return any(re.search(p, text) for p in patterns)


def _detect_intent(text_lower: str) -> str:
    # Order matters: more specific / higher-priority topics first.
    priority = [
        "fraud_security",
        "account_closure",
        "refund",
        "kyc",
        "password_reset",
        "login_2fa",
        "withdrawal",
        "deposit",
    ]
    for intent in priority:
        if _any_match(INTENT_PATTERNS[intent], text_lower):
            return intent
    return "other"


def _detect_urgency(text_lower: str, intent: str, customer_tier: str | None) -> str:
    if _any_match(URGENCY_HIGH_PATTERNS, text_lower):
        return "high"
    if intent == "fraud_security":
        # Fraud reports are always at minimum medium urgency.
        return "high"
    if _any_match(URGENCY_MEDIUM_PATTERNS, text_lower):
        # Slight VIP bump.
        if (customer_tier or "").lower() == "vip":
            return "high"
        return "medium"
    if (customer_tier or "").lower() == "vip":
        return "medium"
    return "low"


def _contains_policy_topic(text_lower: str, intent: str) -> bool:
    if intent in {"account_closure", "refund", "fraud_security"}:
        return True
    return any(kw in text_lower for kw in POLICY_KEYWORDS)


def _needs_more_context(message_lower: str, text_length: int, intent: str) -> bool:
    stripped = message_lower.strip().rstrip(".!?")
    if stripped in GENERIC_PHRASES:
        return True
    if text_length < SHORT_TEXT_THRESHOLD and intent == "other":
        return True
    return False


def _preliminary_route(
    *, intent: str, urgency: str, policy: bool,
    needs_human: bool, needs_more_context: bool,
) -> str:
    if needs_more_context:
        return "INSUFFICIENT_CONTEXT"
    if intent == "fraud_security" or (policy and intent == "fraud_security"):
        return "ESCALATE_RISK"
    if policy:
        return "ESCALATE_POLICY"
    if needs_human:
        return "HUMAN_REVIEW"
    return "AUTO_DRAFT"


def classify_tickets(normalised: list[dict]) -> list[dict]:
    out: list[dict] = []
    for t in normalised:
        text_lower = t["text_lower"]
        intent = _detect_intent(text_lower)
        urgency = _detect_urgency(text_lower, intent, t.get("customer_tier"))
        policy = _contains_policy_topic(text_lower, intent)
        more_ctx = _needs_more_context(t["message_lower"], t["text_length"], intent)

        needs_human = policy or intent == "fraud_security" or (
            urgency == "high" and intent in {"other", "withdrawal"}
        )

        prelim = _preliminary_route(
            intent=intent, urgency=urgency, policy=policy,
            needs_human=needs_human, needs_more_context=more_ctx,
        )

        out.append({
            "ticket_id": t["ticket_id"],
            "intent": intent,
            "urgency": urgency,
            "contains_policy_topic": policy,
            "needs_human": needs_human,
            "needs_more_context": more_ctx,
            "preliminary_route": prelim,
        })

    ensure_artifacts_dir()
    ROUTING.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out
