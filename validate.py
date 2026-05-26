"""Validation command for the support copilot pipeline artifacts.

Run after `python pipeline.py`. Exits 0 if every invariant holds, non-zero if
anything is wrong. Each check prints a single line so failures are obvious in
CI logs.

Checks (mirrors the spec's VALIDATION REQUIREMENTS section):
  - required artifacts exist
  - JSON files parse
  - ticket IDs are unique
  - every ticket has exactly one final route
  - drafts exist only for AUTO_DRAFT tickets
  - non-AUTO_DRAFT tickets appear in the review queue
  - retrieval entries include scores + article IDs
  - route reasons non-empty
  - llm_calls.jsonl has one record per LLM call
  - if expected_routes.json exists, compute + print accuracy
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from src.config.paths import (
    CLARIFICATIONS, DRAFT_REVIEWS, DRAFTS, EXPECTED_ROUTES, FINAL_ROUTES,
    KB_IN, LLM_CALLS, OPS_REPORT, RETRIEVAL, REVIEW_QUEUE, TICKETS_IN,
    TICKETS_NORMALISED, ROUTING,
)


REQUIRED_ARTIFACTS = [
    TICKETS_IN, KB_IN,
    TICKETS_NORMALISED, ROUTING, RETRIEVAL, FINAL_ROUTES,
    DRAFTS, REVIEW_QUEUE, OPS_REPORT, LLM_CALLS,
]


def _ok(msg: str) -> None:
    print(f"  PASS  {msg}")


def _fail(msg: str, errors: list[str]) -> None:
    print(f"  FAIL  {msg}")
    errors.append(msg)


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    errors: list[str] = []
    print("Validating pipeline artifacts...\n")

    # 1. Required files exist.
    for path in REQUIRED_ARTIFACTS:
        if path.exists():
            _ok(f"artifact present: {path.name}")
        else:
            _fail(f"artifact missing: {path}", errors)

    if errors:
        # Bail early; later checks all assume the files are there.
        print("\nValidation FAILED — required artifacts missing.")
        return 1

    # 2. JSON parses.
    try:
        tickets = _load_json(TICKETS_IN)
        kb = _load_json(KB_IN)
        normalised = _load_json(TICKETS_NORMALISED)
        routing = _load_json(ROUTING)
        retrieval = _load_json(RETRIEVAL)
        final_routes = _load_json(FINAL_ROUTES)
        drafts = _load_json(DRAFTS)
        review_queue = _load_json(REVIEW_QUEUE)
        _ok("all JSON artifacts parsed cleanly")
    except json.JSONDecodeError as exc:
        _fail(f"JSON parse failure: {exc}", errors)
        print("\nValidation FAILED.")
        return 1

    # 3. Ticket IDs unique.
    ticket_ids = [t["ticket_id"] for t in tickets]
    dup = [tid for tid, n in Counter(ticket_ids).items() if n > 1]
    if dup:
        _fail(f"duplicate ticket_ids: {dup}", errors)
    else:
        _ok(f"{len(ticket_ids)} ticket_ids unique")

    # 4. Every ticket has exactly one final route.
    routed_ids = [r["ticket_id"] for r in final_routes]
    if sorted(routed_ids) != sorted(ticket_ids):
        _fail("final_routes ticket set does not match tickets.json", errors)
    elif len(routed_ids) != len(set(routed_ids)):
        _fail("final_routes contains duplicate ticket_ids", errors)
    else:
        _ok("every ticket has exactly one entry in final_routes")

    valid_routes = {
        "AUTO_DRAFT", "HUMAN_REVIEW", "ESCALATE_POLICY",
        "ESCALATE_RISK", "INSUFFICIENT_CONTEXT",
    }
    bad = [r["ticket_id"] for r in final_routes if r["final_route"] not in valid_routes]
    if bad:
        _fail(f"unknown final_route values for: {bad}", errors)
    else:
        _ok("all final_route values are within the allowed set")

    # 5. Drafts exist only for AUTO_DRAFT tickets.
    auto_ids = {r["ticket_id"] for r in final_routes if r["final_route"] == "AUTO_DRAFT"}
    draft_ids = {d["ticket_id"] for d in drafts}
    if draft_ids != auto_ids:
        extra = sorted(draft_ids - auto_ids)
        missing = sorted(auto_ids - draft_ids)
        _fail(
            f"drafts do not match AUTO_DRAFT set (extra: {extra}, missing: {missing})",
            errors,
        )
    else:
        _ok(f"drafts exist for exactly the {len(auto_ids)} AUTO_DRAFT tickets")

    # 6. Non-AUTO_DRAFT tickets appear in the review queue.
    non_auto = {r["ticket_id"] for r in final_routes if r["final_route"] != "AUTO_DRAFT"}
    queue_ids = {q["ticket_id"] for q in review_queue}
    if queue_ids != non_auto:
        extra = sorted(queue_ids - non_auto)
        missing = sorted(non_auto - queue_ids)
        _fail(
            f"review queue mismatch (extra: {extra}, missing: {missing})",
            errors,
        )
    else:
        _ok(f"review queue contains all {len(non_auto)} non-AUTO_DRAFT tickets")

    # 7. Retrieval entries include scores + article IDs.
    bad_retr = []
    for r in retrieval:
        for c in r.get("candidates", []):
            if "article_id" not in c or "score" not in c:
                bad_retr.append(r["ticket_id"])
                break
    if bad_retr:
        _fail(f"retrieval entries missing fields for: {bad_retr}", errors)
    else:
        _ok("every retrieval candidate has article_id + score")

    # 8. Route reasons non-empty.
    empty = [r["ticket_id"] for r in final_routes if not (r.get("reason") or "").strip()]
    if empty:
        _fail(f"empty reason for tickets: {empty}", errors)
    else:
        _ok("every final route has a non-empty reason")

    # 9. llm_calls.jsonl has one record per LLM call.
    llm_records = [
        json.loads(line)
        for line in LLM_CALLS.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_calls = len(drafts)  # one per AUTO_DRAFT
    # Stretch contributions (if enabled) bump the expected count.
    if DRAFT_REVIEWS.exists():
        reviews = _load_json(DRAFT_REVIEWS)
        expected_calls += len(reviews)
    if CLARIFICATIONS.exists():
        clars = _load_json(CLARIFICATIONS)
        expected_calls += len(clars)

    if len(llm_records) != expected_calls:
        _fail(
            f"llm_calls.jsonl has {len(llm_records)} records, expected {expected_calls}",
            errors,
        )
    else:
        _ok(f"llm_calls.jsonl contains {len(llm_records)} records (one per LLM call)")

    # 10. Optional: expected_routes.json → accuracy.
    if EXPECTED_ROUTES.exists():
        try:
            expected = _load_json(EXPECTED_ROUTES)
            expected_by_id = {e["ticket_id"]: e["expected_route"] for e in expected}
            actual_by_id = {r["ticket_id"]: r["final_route"] for r in final_routes}
            common = expected_by_id.keys() & actual_by_id.keys()
            if not common:
                _fail("expected_routes.json present but no ticket_ids overlap with run output", errors)
            else:
                correct = sum(1 for tid in common if expected_by_id[tid] == actual_by_id[tid])
                total = len(common)
                accuracy = correct / total
                _ok(f"route accuracy vs expected_routes.json: {correct}/{total} = {accuracy:.0%}")
                mismatches = [
                    (tid, expected_by_id[tid], actual_by_id[tid])
                    for tid in sorted(common)
                    if expected_by_id[tid] != actual_by_id[tid]
                ]
                for tid, exp, act in mismatches:
                    print(f"        mismatch {tid}: expected={exp}, actual={act}")
        except Exception as exc:
            _fail(f"failed to evaluate expected_routes.json: {exc}", errors)

    print()
    if errors:
        print(f"Validation FAILED — {len(errors)} error(s).")
        return 1
    print("Validation PASSED.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
