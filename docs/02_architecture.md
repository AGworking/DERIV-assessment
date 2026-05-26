# Architecture & Design Notes

Companion to `README.md`. This document goes one level deeper into *why* the pipeline is shaped this way, and walks one ticket end-to-end so the data flow is concrete.

---

## Design principle: the LLM is downstream of the decision

The single most important architectural choice in this codebase is that **every routing decision is made by deterministic Python before the LLM runs**. The LLM is not asked "what should I do with this ticket?" It is only asked, after we've already decided to draft a reply, "given these KB snippets, write a draft."

This matters because:

- **Auditability.** Every `final_route` decision in `artifacts/final_routes.json` has a human-readable `reason` field naming the rule that fired. We can re-derive the decision from the code without re-running the LLM.
- **Replayability.** A deterministic decision is the same every run. The LLM provider can be swapped (mock ↔ openai) and the **routes stay identical** — only the *quality* of the AI-drafted replies changes.
- **Safety.** A non-deterministic decider is impossible to audit retroactively. By isolating the LLM to drafting + critique, we can prove via code review that no policy-restricted ticket can ever reach the drafting stage.

Concretely: a refund request (`safe_for_ai: false` on the KB article) is rejected by `src/deterministic/route.py:_decide`, **before** `src/llm/stages/draft.py` even iterates that ticket. The LLM never sees it.

The folder split makes this contract enforceable: nothing under `src/deterministic/` may import from `src/llm/*`. If a future change ever tried to consult the LLM in the routing decision, the import would have to cross the package boundary — a code-review smell that's much harder to miss than a subtle behavioural regression.

---

## The five final routes

| Route | Triggered by | Goes to |
|---|---|---|
| `AUTO_DRAFT` | rules clear + top KB article is `safe_for_ai` + retrieval score ≥ 0.10 | `drafts.json` + self-check |
| `HUMAN_REVIEW` | rules clear but retrieval is too weak (top score < 0.10) | `review_queue.json` |
| `ESCALATE_POLICY` | top KB hit `safe_for_ai: false` AND rules detect policy topic | `review_queue.json` with finance/policy recommendation |
| `ESCALATE_RISK` | rules detect fraud / security intent | `review_queue.json` with security recommendation |
| `INSUFFICIENT_CONTEXT` | message too short or contains only generic phrases | `clarifications.json` + `review_queue.json` |

The decision logic is at `src/deterministic/route.py:_decide`, written as a numbered if/elif precedence list. First match wins.

---

## Worked example: ticket `t_002` ("Please close my account now")

Following one ticket through every stage.

### Stage 1 + 2: load + validate (`src/deterministic/load.py`)

```json
{
  "ticket_id": "t_002",
  "created_at": "2026-05-01T09:18:00Z",
  "channel": "web_form",
  "language": "en",
  "customer_tier": "standard",
  "subject": "Please close my account now",
  "message": "I want my account deleted immediately. Also tell me what happens to my remaining balance.",
  "attachments": [],
  "metadata": {"country": "GB", "logged_in": true}
}
```

Required fields present, `ticket_id` unique, `created_at` parses, `message` non-empty → **passes**.

### Stage 3: normalise (`src/deterministic/normalise.py`)

Adds lowercase versions for downstream regex / TF-IDF, plus derived fields:

```json
{
  "ticket_id": "t_002",
  "subject_lower": "please close my account now",
  "message_lower": "i want my account deleted immediately. also tell me what happens to my remaining balance.",
  "text_lower": "please close my account now. i want my account deleted immediately...",
  "text_length": 124,
  "attachment_count": 0,
  "logged_in": true
}
```

### Stage 4: rule-based classify (`src/deterministic/classify.py`)

Regex over `text_lower` against `INTENT_PATTERNS`. Priority order matters here — `account_closure` is checked before `deposit`:

- matches `\bclose\s+my\s+account\b` → **intent = `account_closure`**
- matches `\bimmediat(?:e|ely)\b` → **urgency = `high`**
- intent is in the policy intent set → **contains_policy_topic = true**
- text length 124 >> threshold → **needs_more_context = false**
- policy_topic OR fraud → **needs_human = true**
- precedence: `policy AND not fraud` → **preliminary_route = `ESCALATE_POLICY`**

Written to `routing.json`.

### Stage 5 + 6: TF-IDF retrieval (`src/deterministic/retrieve.py`)

The vectoriser is fit on the union of all KB documents and all ticket documents so the IDF vocabulary covers both. For `t_002`:

| Rank | Article | Score | Matched terms |
|---|---|---|---|
| 1 | `kb_002` "Account deletion and closure requests" | 0.1415 | account, balance, close, remaining |
| 2 | `kb_008` "Suspected fraud or unauthorized account access" | 0.0554 | account, immediately |
| 3 | `kb_001` "Bank transfer deposits" | 0.0210 | balance |

Critically, `kb_002.safe_for_ai = false` (it's a policy topic, handled by humans). That flag travels into `retrieval.json` so the safety gate can use it directly.

### Stage 7: safety gate (`src/deterministic/route.py`)

Walking the precedence list:

1. `needs_more_context`? No.
2. `intent == fraud_security`? No.
3. Top article `safe_for_ai = false` AND rules detected policy topic? **YES** → route is `ESCALATE_POLICY`.

The reason field that gets written:

> *"Top retrieved article 'kb_002' is marked safe_for_ai=False and the rule layer detected a policy topic (intent=account_closure). AI drafting blocked by policy."*

This is the safety gate **catching the same hazard from two independent angles** — both the rule layer (intent=account_closure) and the retrieval layer (top KB hit is `safe_for_ai=false`) agree, and the reason makes that visible.

### Stage 8: drafting — **skipped**

`src/llm/stages/draft.py:generate_drafts` iterates `final_routes` and filters `if route["final_route"] != "AUTO_DRAFT": continue`. `t_002` is not in the AUTO_DRAFT set, so no LLM call happens for it. Verifiable in `llm_calls.jsonl` — no record with `"ticket_id": "t_002"`.

### Stage 9: review queue (`src/deterministic/review_queue.py`)

`t_002` enters `review_queue.json` with the per-intent recommendation:

> *"Escalate to the policy specialist for account closure: confirm balance is withdrawn before closure and do not promise immediate deletion."*

### Final result

The customer's account-closure request reaches a human policy specialist with all the context attached, and the AI has not had any chance to make a promise it shouldn't. Auditing the decision is one open of `final_routes.json` away.

---

## Retrieval: why TF-IDF cosine?

The spec accepts "token overlap, BM25-style scoring, TF-IDF cosine similarity, embeddings if locally available, or another inspectable method." I picked TF-IDF cosine via `sklearn` because:

- **Inspectable.** The score is the cosine of two sparse vectors and the matched-terms list shows *which* words drove the match. A reviewer can derive the same number with pen and paper.
- **Zero infrastructure.** No model download, no API call, no cache. One `pip install scikit-learn` and it works on a clean checkout.
- **Robust to swapped fixtures.** As long as the new tickets and KB share English vocabulary, retrieval still works. No hardcoded ticket IDs or wording anywhere in `src/deterministic/retrieve.py`.

The TF-IDF index is fit on the **union** of KB documents and ticket documents (`src/deterministic/retrieve.py` ~line 80). This is important — if we fit on the KB alone, a ticket-only word like "USD" or "topup" wouldn't appear in the vocabulary at all. Fitting on the union gives every meaningful word an IDF score.

Title and tags are concatenated **twice** when building each KB document — a cheap way to up-weight them relative to the article body. Matches in the title are more discriminating than matches in flavour text.

The retrieval score floor (0.10) used by the safety gate was set against the actual fixture range: genuine hits sit at 0.14–0.55, junk hits cluster at 0.00–0.05.

---

## Why mock + OpenAI behind a feature flag

The spec says: *"Any LLM provider or local model may be used, but the solution should still be understandable if API keys are not available. A mock mode or fallback is acceptable."*

A naive interpretation is "make it fall back to mock if the API call fails." That's worse than no fallback — it hides errors. The pipeline does this instead:

- The provider is chosen explicitly at startup via `LLM_PROVIDER` (default `mock`).
- The factory `src/llm/provider.py:get_provider()` returns the right class.
- The OpenAI provider raises immediately if `OPENAI_API_KEY` is missing — no silent degradation.
- Both providers emit JSON in the same schema, so the calling stage doesn't care which one ran.
- Both paths log to `llm_calls.jsonl` with the `provider` field set, so the evaluator can see which one was used.

Adding Anthropic or any other provider is one new file at `src/llm/<name>_provider.py` plus a one-line factory branch.

---

## What's intentionally not in scope

- **Multilingual support.** Only the `language` field is normalised — the rules are English-keyword based.
- **Real database / queue.** Everything is file-based. Sufficient for a 60-minute exercise and easy to test.
- **Pagination / streaming.** The pipeline reads all tickets into memory and processes them in one go. Fine up to thousands of tickets; would need refactoring for millions.
- **A learned classifier or embedding retriever.** TF-IDF is enough at this fixture size and stays inspectable.
- **Caching the TF-IDF index between runs.** Re-fitting per run is fast enough and avoids stale-cache bugs when the KB changes.
