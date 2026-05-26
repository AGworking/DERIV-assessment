# Problem Understanding — Support Ticket Copilot Pipeline

## TL;DR

Build a **deterministic, replayable pipeline** that takes inbound support tickets + a KB, and decides — in code, not via the LLM — which tickets the AI is allowed to draft a reply for. The LLM only drafts (and optionally critiques) responses for tickets the deterministic layer has already cleared. Every intermediate decision is written to disk so the evaluator can inspect *why* each ticket was routed the way it was.

The evaluator will run this from a **clean checkout** and may **replace the fixtures** with same-schema data, so we cannot hardcode IDs, ticket text, or "known good" answers.

---

## What the Evaluator is Really Testing

> "The evaluator cares about engineering judgment: what should be deterministic vs LLM-assisted, how you prevent hallucinated advice, how you handle missing data and ambiguity, how you structure artifacts so the system is inspectable and rerunnable."

So the grading axes are:

1. **Determinism boundary** — validation / routing / retrieval / safety gates must be plain code. The LLM is downstream, never the decider.
2. **Hallucination control** — drafts must be constrained to retrieved KB content, and must cite article IDs.
3. **Inspectability** — every stage writes a JSON artifact. Nothing important lives only in memory.
4. **Replayability** — runs from a clean checkout, fixtures swappable, no hardcoded ticket assumptions.
5. **Safety gates that actually fire** — e.g. account deletion (`kb_002.safe_for_ai = false`) must NOT be auto-drafted, even if rules + retrieval would otherwise allow it.

---

## Inputs

- `tickets.json` — array of ticket objects (id, created_at, channel, language, customer_tier, subject, message, attachments, metadata).
- `kb.json` — array of KB articles (article_id, title, category, tags, content, **safe_for_ai** flag).

The `safe_for_ai` flag on KB articles is the most important signal in the entire pipeline — it's the policy-level "this topic is off-limits to the AI" switch.

---

## Pipeline Stages (must enforce in code, in order)

```
INIT
  → INPUTS_LOADED          # read files
  → DATA_VALIDATED         # schema + uniqueness checks
  → TICKETS_NORMALISED     # lowercased text, length, attachment count, logged_in flag
  → RULES_CLASSIFIED       # intent, urgency, policy topic, needs_human, needs_more_context, preliminary_route
  → KB_INDEXED             # build retrieval index (TF-IDF or token-overlap)
  → CANDIDATES_RETRIEVED   # top-3 KB articles per ticket with scores
  → SAFETY_DECISIONS_APPLIED  # combine rules + retrieval → final route
  → DRAFTS_GENERATED       # LLM call ONLY for AUTO_DRAFT tickets
  → REVIEW_QUEUE_BUILT     # everything not AUTO_DRAFT
  → REPORT_GENERATED       # ops_report.md
  → VALIDATION_COMPLETE    # validate.py runs checks
  → RESULTS_FINALISED
```

**Critical rule:** draft generation MUST come after deterministic validation, routing, retrieval, and safety. The LLM cannot be in the routing decision path.

---

## Final Routes (exactly one per ticket)

| Route | When |
|---|---|
| `AUTO_DRAFT` | Safe, well-scoped, evidence is `safe_for_ai`, ticket is non-policy with enough context |
| `HUMAN_REVIEW` | Looks ok but retrieval is weak or ticket is borderline |
| `ESCALATE_POLICY` | Policy-sensitive (account closure, refunds, legal) — top KB hit is `safe_for_ai: false` |
| `ESCALATE_RISK` | High urgency + ambiguous + risky language (fraud, chargeback, security) |
| `INSUFFICIENT_CONTEXT` | Message too short / missing crucial info (no transaction ID, no error text, etc.) |

---

## Decomposition into Parts (build in this order)

### Part A — Skeleton & I/O (~5 min)
- `pipeline.py` entrypoint that walks the stages.
- Read `tickets.json` and `kb.json`, fail loudly on parse errors.
- Stage 1 + 2: validation (unique IDs, required fields, parseable timestamps) + normalisation (`tickets_normalized.json`).

### Part B — Deterministic Routing (~10 min)
- Keyword/regex rules for `intent` (deposit, account_closure, kyc, password_reset, billing, other).
- Keyword rules for `urgency` (high if "immediately", "urgent", "fraud", high-tier customer + complaint, etc.).
- `contains_policy_topic` = matches policy keywords (delete, close account, refund, legal, chargeback).
- `needs_more_context` = message too short, or missing required entities for the intent.
- `needs_human` = OR of policy_topic + risk markers.
- Write `routing.json` with `preliminary_route`.

### Part C — KB Retrieval (~10 min)
- Tokenize KB content + tags + title. Build TF-IDF over them (sklearn) or fall back to weighted token-overlap if sklearn unavailable.
- For each ticket, score every KB article; take top 3 with `article_id`, `score`, `matched_terms`, `safe_for_ai`.
- Write `retrieval.json`. **Important:** must not be hardcoded — score against actual ticket tokens.

### Part D — Safety Gate / Final Routing (~5 min)
- Combine rules + retrieval to produce the final route + human-readable reason.
- Rules of thumb:
  - If `top.safe_for_ai == false` AND policy_topic → `ESCALATE_POLICY`.
  - If `needs_more_context` → `INSUFFICIENT_CONTEXT`.
  - If urgency=high + risk signals → `ESCALATE_RISK`.
  - If top score below threshold → `HUMAN_REVIEW` (weak retrieval).
  - Else if top is `safe_for_ai: true` and rules permit → `AUTO_DRAFT`.
- Write `final_routes.json`.

### Part E — LLM Draft Generation (~8 min)
- For AUTO_DRAFT tickets only, build a constrained prompt that includes:
  - the ticket subject + message
  - the retrieved KB snippets with their IDs
  - explicit instructions: cite article IDs, do not promise actions not in KB, professional tone, ask clarifying question if needed.
- Output schema: `{ticket_id, draft_reply, used_article_ids[], confidence}`.
- **Provider abstraction**: support real LLM (Anthropic) AND a mock mode that produces grounded canned drafts using retrieved KB content directly. Mock mode is what lets the evaluator run without keys.
- Append one record per call to `llm_calls.jsonl`.

### Part F — Review Queue + Ops Report (~5 min)
- `review_queue.json` for every non-AUTO_DRAFT ticket: ticket_id, final_route, summary, top evidence, recommended next action.
- `ops_report.md`: human-readable summary with the required sections (Executive Summary, Route Distribution, Retrieval Quality Notes, Auto-Drafted, Review/Escalation, Safety Constraints, Known Limitations).

### Part G — Validation (~8 min)
- `validate.py` checks:
  - all required artifacts exist and are valid JSON
  - ticket IDs are unique
  - every ticket has exactly one final route
  - drafts exist only for AUTO_DRAFT tickets
  - non-AUTO_DRAFT tickets are in review_queue
  - retrieval entries include scores + article_ids
  - route reasons non-empty
  - `llm_calls.jsonl` has one record per draft call
  - if `expected_routes.json` exists, compute + print accuracy.

### Part H — Stretch (only if time remains)
- Self-check LLM pass → `draft_reviews.json`
- `expected_routes.json` with ~8 labelled tickets + accuracy check in validate.py
- `clarifications.json` for INSUFFICIENT_CONTEXT tickets

---

## Time Budget (60 min)

| Block | Minutes |
|---|---|
| Skeleton + I/O + validation + normalisation (A) | 5 |
| Rules classification (B) | 10 |
| KB retrieval (C) | 10 |
| Safety gate / final routing (D) | 5 |
| LLM drafts + mock fallback + llm_calls.jsonl (E) | 8 |
| Review queue + ops report (F) | 5 |
| validate.py (G) | 8 |
| Stretch goals (H) | 5 |
| Buffer / fix-up | 4 |

If we run short, the stretch items get dropped before the core artifacts.

---

## Risks & How We Mitigate

| Risk | Mitigation |
|---|---|
| Evaluator swaps fixtures → our rules break | Rules are keyword/regex based on *content*, not IDs. KB retrieval scores against actual tokens. |
| No API key available at eval time | Mock LLM provider that constructs grounded drafts by templating retrieved KB content. Still logs to `llm_calls.jsonl` with `provider: "mock"`. |
| LLM hallucinates outside KB | Prompt constrains to retrieved snippets; output must cite `used_article_ids`; self-check pass (if we add it) flags unsupported claims. |
| safe_for_ai=false gets auto-drafted anyway | Hard rule in safety gate: if any top-3 retrieved article touching a policy topic is `safe_for_ai: false`, route is `ESCALATE_POLICY`. |
| INSUFFICIENT_CONTEXT misfires on legitimate short messages | Use both length AND missing-entity checks per intent (e.g. deposit intent without any amount/time reference). |

---

## Decisions Locked In (from user)

| Decision | Choice |
|---|---|
| **Language** | Python 3 |
| **Retrieval** | TF-IDF cosine similarity via `scikit-learn` |
| **LLM mode** | **Feature-flag driven.** Default = mock provider (runs with zero env setup). If `LLM_PROVIDER=openai` and `OPENAI_API_KEY` is set, switch to OpenAI calls. Both paths log to `llm_calls.jsonl`. |
| **Stretch goals** | Attempt **all three** if time permits, in order: (1) self-check pass → `draft_reviews.json`, (2) eval set + accuracy → `expected_routes.json`, (3) clarification drafts → `clarifications.json`. Drop from the back if we run short. |
| **Fixtures** | Use the exact sample `tickets.json` / `kb.json` from the problem statement as starter fixtures. Add 4 more synthetic tickets to reach 8 for `expected_routes.json`. |

## Provider Abstraction Plan

```
llm/provider.py
  - get_provider() -> reads LLM_PROVIDER env var
      "mock"   (default): MockProvider — templates retrieved KB into a grounded reply
      "openai": OpenAIProvider — uses openai SDK + OPENAI_API_KEY
  - All providers expose: .draft(prompt, system) and .review(prompt, system)
  - All calls go through a logger that appends to llm_calls.jsonl
```

This is the feature flag the user asked for.

## Repo Layout (planned)

```
DERIV/
  tickets.json
  kb.json
  pipeline.py            # main entrypoint, walks stages
  validate.py            # validation command
  src/
    __init__.py
    load.py              # stage 1+2: load + validate
    normalise.py         # stage 3
    classify.py          # stage 4: deterministic rules
    retrieve.py          # stage 5+6: TF-IDF index + top-3
    route.py             # stage 7: safety gate + final route
    draft.py             # stage 8: LLM drafts (gated to AUTO_DRAFT only)
    review_queue.py      # stage 9
    report.py            # stage 10: ops_report.md
    llm/
      __init__.py
      provider.py        # MockProvider + OpenAIProvider behind a flag
      logger.py          # append-only llm_calls.jsonl
  artifacts/             # all generated JSON lands here (gitignored)
    tickets_normalized.json
    routing.json
    retrieval.json
    final_routes.json
    drafts.json
    review_queue.json
    ops_report.md
    llm_calls.jsonl
    draft_reviews.json       # stretch
    clarifications.json      # stretch
  expected_routes.json       # stretch — hand-labelled
  docs/
    01_understanding.md  # this file
  README.md
  requirements.txt
```

## What I'll Do Next

Once you say go, the build order is:
1. Scaffold repo + `requirements.txt` + sample fixtures.
2. Parts A → G from the decomposition above.
3. Stretch H if time allows.
4. End by running the pipeline + `validate.py` to confirm artifacts.
