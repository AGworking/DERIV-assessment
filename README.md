# Support Ticket Copilot Pipeline

A replayable pipeline that ingests customer-support tickets, classifies each into a deterministic action path, retrieves grounded knowledge-base snippets, drafts a constrained AI response **only when safe**, and produces an ops review queue for human agents.

The deterministic layers — validation, classification, retrieval, safety gates — are plain Python. The LLM is only used downstream, for drafting and self-critique. **The LLM never makes routing decisions.**

---

## Quick start

```powershell
# 1. Create the conda env
conda create -n deriv python=3.11 -y
conda activate deriv

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) Configure the LLM provider
cp .env.example .env
# Edit .env — leave LLM_PROVIDER=mock for a zero-config run,
# or set LLM_PROVIDER=openai + OPENAI_API_KEY to use real OpenAI.

# 4. Run the pipeline
python pipeline.py

# 5. Verify every artifact invariant
python validate.py
```

A clean checkout with no `.env` and no API key runs end-to-end on the **mock provider** — no network calls required.

---

## What it does

Given two input files at the project root:

- `tickets.json` — a batch of inbound support tickets
- `kb.json` — a small knowledge base of articles, each flagged `safe_for_ai: true|false`

…the pipeline produces:

| Artifact | What it is |
|---|---|
| `artifacts/tickets_normalized.json` | normalised tickets with lowercased text, lengths, attachment counts |
| `artifacts/routing.json` | rule-based intent, urgency, policy flags, preliminary route |
| `artifacts/retrieval.json` | top-3 KB articles per ticket via TF-IDF cosine, with scores + matched terms |
| `artifacts/final_routes.json` | final route per ticket + human-readable reason |
| `artifacts/drafts.json` | AI-drafted replies for AUTO_DRAFT tickets only |
| `artifacts/review_queue.json` | every non-AUTO_DRAFT ticket with a recommended next action |
| `artifacts/draft_reviews.json` | second LLM pass critiquing each draft (never overwrites it) |
| `artifacts/clarifications.json` | clarification-question drafts for INSUFFICIENT_CONTEXT tickets |
| `artifacts/ops_report.md` | human-readable run summary |
| `artifacts/llm_calls.jsonl` | one JSON record per LLM call (stage, ticket, provider, model, prompt_hash, …) |

Every ticket ends in **exactly one** of these five final states:

| Route | Meaning |
|---|---|
| `AUTO_DRAFT` | Safe to draft with AI — non-policy, sufficient context, `safe_for_ai` evidence |
| `HUMAN_REVIEW` | Borderline — retrieval weak or signals mixed |
| `ESCALATE_POLICY` | Policy-sensitive (account closure, refunds, legal); top KB article is `safe_for_ai=false` |
| `ESCALATE_RISK` | High urgency + risk markers (fraud, security incident) |
| `INSUFFICIENT_CONTEXT` | Message too short or missing required entities |

---

## Pipeline architecture

```
                    tickets.json                         kb.json
                         │                                  │
                         ▼                                  │
                 ┌──────────────┐                           │
                 │   load +     │  (schema, unique IDs,     │
                 │   validate   │   parseable timestamps)   │
                 └──────┬───────┘                           │
                        │                                   │
                        ▼                                   │
                 ┌──────────────┐                           │
                 │  normalise   │  → tickets_normalized.json│
                 └──────┬───────┘                           │
                        │                                   │
                        ▼                                   │
                 ┌──────────────┐                           │
                 │  classify    │  (intent, urgency,        │
                 │  (rule-based)│   policy_topic,           │
                 │              │   needs_more_context)     │
                 └──────┬───────┘  → routing.json           │
                        │                                   │
                        │            ┌─────────────────┐    │
                        │            │  TF-IDF index   │ ◀──┘
                        │            └────────┬────────┘
                        │                     │
                        └────────┐   ┌────────┘
                                 ▼   ▼
                          ┌──────────────┐
                          │   retrieve   │  top-3 per ticket
                          │  (cosine sim)│  → retrieval.json
                          └──────┬───────┘
                                 │
                                 ▼
                          ┌──────────────┐
                          │ safety gate  │  combine rules + retrieval
                          │  + routing   │  + safe_for_ai checks
                          └──────┬───────┘  → final_routes.json
                                 │
              ┌──────────────────┼─────────────────┐
              │                  │                 │
              ▼                  ▼                 ▼
    ┌────────────────┐ ┌──────────────────┐ ┌────────────────┐
    │   AUTO_DRAFT?  │ │ INSUFFICIENT_CTX │ │ everything else│
    └────────┬───────┘ └──────┬───────────┘ └────────┬───────┘
             │                │                      │
             ▼                ▼                      ▼
    ┌────────────────┐ ┌──────────────────┐ ┌────────────────┐
    │ LLM: draft     │ │ LLM: clarify     │ │ review queue   │
    │ ↓              │ │ ↓                │ │ (no LLM call)  │
    │ self-check     │ │ clarifications   │ │                │
    └────────┬───────┘ └──────┬───────────┘ └────────┬───────┘
             ▼                ▼                      ▼
        drafts.json    clarifications.json    review_queue.json
             │                                       │
             └──────────────┬────────────────────────┘
                            ▼
                      ops_report.md
                      llm_calls.jsonl
```

The boxes above the bottom four cells are **deterministic**: every decision they make is the result of a function over the inputs and could be reproduced by a human reading the code. The LLM only ever runs inside the three bottom cells, and only on tickets the deterministic layer has explicitly cleared.

---

## Configuration: `.env`

The LLM provider is controlled by the `LLM_PROVIDER` feature flag.

| Variable | Required when | Default | Purpose |
|---|---|---|---|
| `LLM_PROVIDER` | always optional | `mock` | feature flag: `mock` or `openai` |
| `OPENAI_API_KEY` | `LLM_PROVIDER=openai` | — | OpenAI auth |
| `OPENAI_MODEL` | optional | `gpt-4o-mini` | model override for the OpenAI path |

Copy `.env.example` to `.env` and edit. The `.env` file is gitignored.

The pipeline also accepts these as real environment variables — `.env` is just a convenience. The loader in `src/config/env.py` falls back to a stdlib parser if `python-dotenv` isn't installed.

---

## Project layout

```
DERIV/
├── README.md                       this file
├── pipeline.py                     orchestrator — walks every stage in order
├── validate.py                     20 invariant checks; exits non-zero on failure
├── requirements.txt
├── .env / .env.example / .gitignore
├── tickets.json / kb.json          input fixtures (evaluator may swap these)
├── expected_routes.json            optional labelled set; validate.py reports accuracy
├── docs/
│   ├── 01_understanding.md         initial problem analysis + decisions
│   └── 02_architecture.md          deeper architecture + extensibility notes
├── src/
│   ├── config/                     boring infrastructure
│   │   ├── env.py                  .env loader (with stdlib fallback)
│   │   └── paths.py                every artifact path in one place
│   ├── deterministic/              ★ pure-Python pipeline stages — NO LLM calls
│   │   ├── load.py                 schema + uniqueness validation
│   │   ├── normalise.py            lowercased text, lengths, derived flags
│   │   ├── classify.py             keyword/regex rules → intent + urgency + flags
│   │   ├── retrieve.py             sklearn TF-IDF cosine over KB, top-3 per ticket
│   │   ├── route.py                safety gate — combines rules + retrieval → final route
│   │   ├── review_queue.py         every non-AUTO_DRAFT ticket + next action
│   │   └── report.py               renders ops_report.md
│   └── llm/                        ★ everything that touches the LLM lives here
│       ├── base.py                 LLMProvider ABC (one method: generate)
│       ├── mock.py                 MockProvider — deterministic templater
│       ├── openai_provider.py      OpenAIProvider — real OpenAI calls
│       ├── provider.py             get_provider() factory (reads LLM_PROVIDER)
│       ├── logger.py               append-only writer for llm_calls.jsonl
│       └── stages/                 pipeline stages that issue LLM calls
│           ├── draft.py            LLM draft for AUTO_DRAFT tickets only
│           ├── self_check.py       second LLM pass critiquing each draft
│           └── clarify.py          clarification questions for INSUFFICIENT_CONTEXT
└── artifacts/                      all generated files (gitignored)

The folder boundary is the safety contract: nothing under `src/deterministic/`
may import from `src/llm/*`. The pipeline orchestrator (`pipeline.py`) runs
every deterministic stage first, only then asks `get_provider()` for an LLM,
and only then runs the `src/llm/stages/*` modules — and only on tickets the
deterministic layer cleared.
```

---

## Safety design

The pipeline blocks the LLM from drafting whenever **any** of these hold — all enforced in deterministic Python, not in a prompt:

1. The rule layer flags a **policy topic** (account closure, refund, chargeback, fraud, suspected unauthorized access).
2. The top retrieved KB article is marked `safe_for_ai: false`.
3. The ticket intent is `fraud_security` — always escalated to risk, regardless of retrieval.
4. The message is too short or generic to act on (`needs_more_context`).
5. The top retrieval cosine similarity is below `0.10` — evidence is too weak to ground a confident reply.

These checks happen in `src/deterministic/ + src/llm/stages/classify.py` and `src/deterministic/ + src/llm/stages/route.py` *before* any LLM call. The LLM is downstream of the decision, not part of it.

When the LLM does draft, the prompt:
- includes only the retrieved KB snippets (no other context)
- forbids inventing policies, timeframes, or actions outside those snippets
- requires citing `article_id`s
- requires a clarifying question instead of guessing if information is missing
- explicitly forbids promising refunds, account deletions, or other irreversible actions

The self-check stage (`src/deterministic/ + src/llm/stages/self_check.py`) runs a *second* LLM pass that critiques the first draft, looking for unsupported claims or missing caveats. The review is written to `draft_reviews.json` — it never overwrites the original draft.

---

## Validation

`python validate.py` runs 20 checks and exits non-zero on the first failure:

- all 10 required artifacts exist and parse as JSON
- ticket IDs are unique
- every ticket has exactly one final route from the allowed set
- drafts exist only for AUTO_DRAFT tickets
- every non-AUTO_DRAFT ticket appears in the review queue
- every retrieval entry has `article_id` + `score`
- every final route has a non-empty reason
- `llm_calls.jsonl` has one record per LLM call (drafts + self-checks + clarifications)
- if `expected_routes.json` exists, route accuracy is computed and printed (currently **10/10**)

The check on the LLM call count means a partial run that crashed mid-pipeline will fail validation — desirable for CI.

---

## Replaying with swapped fixtures

The evaluator may replace `tickets.json` and `kb.json` with same-schema data. The pipeline does not depend on:

- specific ticket IDs (every stage scores against ticket *content*, not IDs)
- specific ticket wording (rules are broad-but-specific English keyword patterns)
- hardcoded route answers (`final_routes.json` is regenerated every run)

Re-running `python pipeline.py` against a swapped fixture set will produce a fresh set of artifacts. Re-running `python validate.py` will still pass as long as the input shapes are valid.

---

## Extending the system

**Add a new intent**: edit `INTENT_PATTERNS` in `src/deterministic/ + src/llm/stages/classify.py`. Add the new key to the priority list at the bottom of `_detect_intent` if order matters.

**Add a new KB article**: append to `kb.json`. The TF-IDF index is rebuilt every run, so the new article is picked up automatically. Set `safe_for_ai: false` for policy topics.

**Add a new LLM provider** (e.g. Anthropic, Azure OpenAI, local Ollama):
1. Create `src/llm/<name>_provider.py` with a class subclassing `LLMProvider`.
2. Implement `.generate(system, user) -> str` returning the model's raw text.
3. Add a branch in `src/llm/provider.py:get_provider()`.

Three small files, no other code touched. The drafts / self-check / clarification stages will all use the new provider automatically.

**Tighten the safety policy**: the route decision precedence lives at the top of `src/deterministic/ + src/llm/stages/route.py:_decide` as a numbered comment block. Add or reorder rules there; every change shows up in `final_routes.json[*].reason` so it's auditable.

---

## License / attribution

Built as a take-home exercise. No external proprietary data sources used.
