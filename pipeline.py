"""Support ticket copilot pipeline — single entrypoint.

Walks the stages in the required order:

    INIT
     -> INPUTS_LOADED
     -> DATA_VALIDATED
     -> TICKETS_NORMALISED
     -> RULES_CLASSIFIED          (TODO Part B)
     -> KB_INDEXED                (TODO Part C)
     -> CANDIDATES_RETRIEVED      (TODO Part C)
     -> SAFETY_DECISIONS_APPLIED  (TODO Part D)
     -> DRAFTS_GENERATED          (TODO Part E)
     -> REVIEW_QUEUE_BUILT        (TODO Part F)
     -> REPORT_GENERATED          (TODO Part F)
     -> VALIDATION_COMPLETE       (validate.py)
     -> RESULTS_FINALISED

Deterministic stages (validation, classification, retrieval, safety gates)
run before any LLM call. The LLM is only invoked downstream for drafting.
"""

from __future__ import annotations

import sys

from src.env import load_env

# Load .env BEFORE importing any module that reads env vars at import time.
load_env()

from src.clarify import generate_clarifications
from src.classify import classify_tickets
from src.draft import generate_drafts
from src.llm.logger import reset_log
from src.llm.provider import get_provider
from src.load import load_inputs, ValidationError
from src.normalise import normalise_tickets
from src.paths import ensure_artifacts_dir
from src.report import render_report
from src.retrieve import retrieve_candidates
from src.review_queue import build_review_queue
from src.route import decide_final_routes
from src.self_check import review_drafts


def _log(stage: str, msg: str = "") -> None:
    suffix = f" — {msg}" if msg else ""
    print(f"[{stage}]{suffix}")


def main() -> int:
    ensure_artifacts_dir()
    reset_log()  # truncate llm_calls.jsonl so each run starts clean

    _log("INIT", "starting pipeline")

    try:
        tickets, articles = load_inputs()
    except ValidationError as exc:
        print(f"[DATA_VALIDATED] FAILED: {exc}", file=sys.stderr)
        return 2
    _log("INPUTS_LOADED", f"{len(tickets)} tickets, {len(articles)} kb articles")
    _log("DATA_VALIDATED", "schema + uniqueness checks passed")

    normalised = normalise_tickets(tickets)
    _log("TICKETS_NORMALISED", f"wrote {len(normalised)} normalised tickets")

    classifications = classify_tickets(normalised)
    from collections import Counter
    route_hist = Counter(c["preliminary_route"] for c in classifications)
    _log("RULES_CLASSIFIED", f"preliminary route hist = {dict(route_hist)}")

    _log("KB_INDEXED", f"indexing {len(articles)} kb articles via TF-IDF")
    retrieval = retrieve_candidates(normalised, articles)
    _log("CANDIDATES_RETRIEVED", f"top-{retrieval[0]['top_k']} per ticket, wrote retrieval.json")

    final_routes = decide_final_routes(classifications, retrieval)
    final_hist = Counter(r["final_route"] for r in final_routes)
    _log("SAFETY_DECISIONS_APPLIED", f"final route hist = {dict(final_hist)}")

    provider = get_provider()
    _log("LLM_PROVIDER", f"using provider={provider.name}, model={provider.model}")
    drafts = generate_drafts(final_routes, retrieval, tickets, articles)
    _log("DRAFTS_GENERATED", f"drafted {len(drafts)} AUTO_DRAFT replies")

    review_queue = build_review_queue(final_routes, retrieval, tickets)
    _log("REVIEW_QUEUE_BUILT", f"{len(review_queue)} tickets queued for human review")

    # --- Stretch goals ---
    draft_reviews = review_drafts(drafts, retrieval, tickets, articles)
    unsupported = sum(1 for r in draft_reviews if not r.get("supported"))
    _log(
        "DRAFTS_SELF_CHECKED",
        f"{len(draft_reviews)} drafts reviewed, {unsupported} flagged for edit/block",
    )

    clarifications = generate_clarifications(final_routes, tickets)
    _log(
        "CLARIFICATIONS_GENERATED",
        f"{len(clarifications)} clarification drafts for INSUFFICIENT_CONTEXT tickets",
    )

    render_report(
        final_routes=final_routes, retrieval=retrieval, drafts=drafts,
        review_queue=review_queue,
        provider_name=provider.name, provider_model=provider.model,
    )
    _log("REPORT_GENERATED", "wrote artifacts/ops_report.md")

    _log("VALIDATION_COMPLETE", "run `python validate.py` to verify artifact invariants")
    _log("RESULTS_FINALISED", "pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
