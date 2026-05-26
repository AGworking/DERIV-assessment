"""Centralised file paths so every stage agrees on where artifacts live.

ROOT is the project root (the folder containing tickets.json, kb.json,
pipeline.py, etc.). All inputs sit at ROOT; all generated artifacts sit
under ROOT/artifacts/.
"""

from pathlib import Path

# This file lives at <ROOT>/src/config/paths.py, so the project root is
# three parents up.
ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = ROOT / "artifacts"

# Inputs
TICKETS_IN = ROOT / "tickets.json"
KB_IN = ROOT / "kb.json"
EXPECTED_ROUTES = ROOT / "expected_routes.json"

# Artifacts (regenerated every run)
TICKETS_NORMALISED = ARTIFACTS / "tickets_normalized.json"
ROUTING = ARTIFACTS / "routing.json"
RETRIEVAL = ARTIFACTS / "retrieval.json"
FINAL_ROUTES = ARTIFACTS / "final_routes.json"
DRAFTS = ARTIFACTS / "drafts.json"
REVIEW_QUEUE = ARTIFACTS / "review_queue.json"
OPS_REPORT = ARTIFACTS / "ops_report.md"
LLM_CALLS = ARTIFACTS / "llm_calls.jsonl"
DRAFT_REVIEWS = ARTIFACTS / "draft_reviews.json"
CLARIFICATIONS = ARTIFACTS / "clarifications.json"


def ensure_artifacts_dir() -> None:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
