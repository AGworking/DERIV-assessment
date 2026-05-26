"""Support ticket copilot pipeline — single entrypoint.

Walks every deterministic stage in order, then drafts and self-critiques the
eligible tickets via the configured LLM provider. Each stage writes its
output to `artifacts/` so every step is independently inspectable.

The deterministic path (validate, normalise, classify, retrieve, route)
decides what happens to every ticket BEFORE any LLM call. The LLM is only
used downstream, for drafting and self-checking — never for routing.

Run:
    python pipeline.py        # uses LLM_PROVIDER from .env (default: mock)
    python validate.py        # verifies every artifact invariant

Stages emitted (one line each):
    INIT  ->  INPUTS_LOADED  ->  DATA_VALIDATED  ->  TICKETS_NORMALISED
        ->  RULES_CLASSIFIED  ->  KB_INDEXED  ->  CANDIDATES_RETRIEVED
        ->  SAFETY_DECISIONS_APPLIED  ->  LLM_PROVIDER  ->  DRAFTS_GENERATED
        ->  REVIEW_QUEUE_BUILT  ->  DRAFTS_SELF_CHECKED
        ->  CLARIFICATIONS_GENERATED  ->  REPORT_GENERATED
        ->  VALIDATION_COMPLETE  ->  RESULTS_FINALISED
"""

from __future__ import annotations

import sys
from collections import Counter

# Load .env BEFORE importing anything that reads env vars at import time.
from src.config.env import load_env

load_env()

from src.config.paths import ensure_artifacts_dir
from src.llm.logger import reset_log
from src.llm.provider import get_provider
from src.stages.classify import classify_tickets
from src.stages.clarify import generate_clarifications
from src.stages.draft import generate_drafts
from src.stages.load import ValidationError, load_inputs
from src.stages.normalise import normalise_tickets
from src.stages.report import render_report
from src.stages.retrieve import retrieve_candidates
from src.stages.review_queue import build_review_queue
from src.stages.route import decide_final_routes
from src.stages.self_check import review_drafts


def _stage(name: str, msg: str = "") -> None:
    """Print one line per stage so progress is obvious from the terminal."""
    if msg:
        print(f"[{name}] — {msg}")
    else:
        print(f"[{name}]")


def main() -> int:
    ensure_artifacts_dir()
    reset_log()  # truncate llm_calls.jsonl so each run starts clean

    _stage("INIT", "starting pipeline")

    # --- Stage 1+2: load + validate ----------------------------------------
    try:
        tickets, articles = load_inputs()
    except ValidationError as exc:
        print(f"[DATA_VALIDATED] FAILED: {exc}", file=sys.stderr)
        return 2
    _stage("INPUTS_LOADED", f"{len(tickets)} tickets, {len(articles)} kb articles")
    _stage("DATA_VALIDATED", "schema + uniqueness checks passed")

    # --- Stage 3: normalise ------------------------------------------------
    normalised = normalise_tickets(tickets)
    _stage("TICKETS_NORMALISED", f"wrote {len(normalised)} normalised tickets")

    # --- Stage 4: rule-based classification --------------------------------
    classifications = classify_tickets(normalised)
    prelim_hist = Counter(c["preliminary_route"] for c in classifications)
    _stage("RULES_CLASSIFIED", f"preliminary route hist = {dict(prelim_hist)}")

    # --- Stage 5+6: TF-IDF retrieval ---------------------------------------
    _stage("KB_INDEXED", f"indexing {len(articles)} kb articles via TF-IDF")
    retrieval = retrieve_candidates(normalised, articles)
    _stage("CANDIDATES_RETRIEVED", f"top-{retrieval[0]['top_k']} per ticket")

    # --- Stage 7: safety gate + final routing ------------------------------
    final_routes = decide_final_routes(classifications, retrieval)
    final_hist = Counter(r["final_route"] for r in final_routes)
    _stage("SAFETY_DECISIONS_APPLIED", f"final route hist = {dict(final_hist)}")

    # --- Stage 8: LLM drafts (AUTO_DRAFT only) -----------------------------
    provider = get_provider()
    _stage("LLM_PROVIDER", f"provider={provider.name}, model={provider.model}")
    drafts = generate_drafts(final_routes, retrieval, tickets, articles)
    _stage("DRAFTS_GENERATED", f"drafted {len(drafts)} AUTO_DRAFT replies")

    # --- Stage 9: review queue ---------------------------------------------
    review_queue = build_review_queue(final_routes, retrieval, tickets)
    _stage("REVIEW_QUEUE_BUILT", f"{len(review_queue)} tickets queued for human review")

    # --- Stretch: self-check + clarifications ------------------------------
    draft_reviews = review_drafts(drafts, retrieval, tickets, articles)
    flagged = sum(1 for r in draft_reviews if not r.get("supported"))
    _stage("DRAFTS_SELF_CHECKED", f"{len(draft_reviews)} reviewed, {flagged} flagged for edit/block")

    clarifications = generate_clarifications(final_routes, tickets)
    _stage("CLARIFICATIONS_GENERATED", f"{len(clarifications)} clarification drafts for INSUFFICIENT_CONTEXT tickets")

    # --- Stage 10: ops report ----------------------------------------------
    render_report(
        final_routes=final_routes,
        retrieval=retrieval,
        drafts=drafts,
        review_queue=review_queue,
        provider_name=provider.name,
        provider_model=provider.model,
    )
    _stage("REPORT_GENERATED", "wrote artifacts/ops_report.md")

    _stage("VALIDATION_COMPLETE", "run `python validate.py` to verify artifact invariants")
    _stage("RESULTS_FINALISED", "pipeline complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
