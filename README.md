# Support Ticket Copilot Pipeline

Replayable pipeline that ingests customer-support tickets, classifies each into a deterministic action path, retrieves grounded knowledge snippets, drafts a constrained AI response only when safe, and produces an ops review queue for human agents.

The deterministic layers (validation, routing, retrieval, safety gates) are plain Python. The LLM is only used downstream for drafting and self-critique — it never makes routing decisions.

See `docs/01_understanding.md` for the full problem analysis, decisions, and build plan.

## Quick start

```bash
pip install -r requirements.txt
python pipeline.py     # produces all artifacts under artifacts/
python validate.py     # checks artifacts are well-formed
```

## LLM mode

- **Default**: `MockProvider` — produces grounded drafts by templating retrieved KB content. Runs on a clean checkout with zero env setup.
- **OpenAI**: set `LLM_PROVIDER=openai` and `OPENAI_API_KEY=…` to switch to real OpenAI calls.

Both paths log every call to `artifacts/llm_calls.jsonl`.

## Final routes

| Route | Meaning |
|---|---|
| `AUTO_DRAFT` | Safe to draft with AI — non-policy, sufficient context, `safe_for_ai` evidence |
| `HUMAN_REVIEW` | Borderline — retrieval weak or signals mixed |
| `ESCALATE_POLICY` | Policy-sensitive (account closure, refunds, legal) |
| `ESCALATE_RISK` | High urgency + risk markers (fraud, security) |
| `INSUFFICIENT_CONTEXT` | Message too short or missing required entities |

## Artifacts

All written under `artifacts/` and regenerated on every run:

- `tickets_normalized.json`
- `routing.json`
- `retrieval.json`
- `final_routes.json`
- `drafts.json`
- `review_queue.json`
- `ops_report.md`
- `llm_calls.jsonl`
- `draft_reviews.json` *(stretch)*
- `clarifications.json` *(stretch)*
