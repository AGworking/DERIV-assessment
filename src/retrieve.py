"""Stage: deterministic KB retrieval.

TF-IDF cosine similarity over (title + tags + content) for each KB article.
Returns the top-3 articles per ticket with a score, the terms that overlapped,
and the article's safe_for_ai flag (so the safety gate can use it directly).
"""

from __future__ import annotations

import json
import re
from typing import Iterable

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.paths import RETRIEVAL, ensure_artifacts_dir


TOP_K = 3

# Light stopword list — sklearn's built-in 'english' is removed in recent versions
# unless you opt in. Keeping our own keeps the dependency footprint small.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "i", "you", "he", "she", "we", "they", "it", "my", "your", "our", "their",
    "me", "us", "them", "this", "that", "these", "those", "to", "of", "in",
    "on", "at", "by", "for", "with", "as", "from", "into", "about", "what",
    "when", "where", "why", "how", "please", "still", "also",
}

_TOKEN_RE = re.compile(r"[a-z][a-z0-9_]+")


def _tokenise(text: str) -> list[str]:
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 2]


def _kb_document(article: dict) -> str:
    """Concatenate the fields we want indexed for each KB article.

    Tags and title are repeated to give them more weight than the content body.
    """
    tags = " ".join(article.get("tags") or [])
    title = article.get("title", "")
    content = article.get("content", "")
    return f"{title} {title} {tags} {tags} {content}"


def _matched_terms(
    ticket_tokens: set[str], article_tokens: set[str],
) -> list[str]:
    """Return the lowercased terms that appear in both the ticket and the article.

    These are not the only signal driving the score (TF-IDF does that), but they
    give a human-readable explanation in retrieval.json.
    """
    overlap = ticket_tokens & article_tokens
    # Stable order, capped so the artifact stays readable.
    return sorted(overlap)[:10]


def retrieve_candidates(
    normalised_tickets: list[dict], kb_articles: list[dict],
) -> list[dict]:
    if not kb_articles:
        raise ValueError("KB is empty — cannot run retrieval")

    kb_docs = [_kb_document(a) for a in kb_articles]
    ticket_docs = [t["text_lower"] for t in normalised_tickets]

    vectoriser = TfidfVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b[a-z][a-z0-9_]{2,}\b",
        stop_words=list(_STOPWORDS),
        ngram_range=(1, 2),
        min_df=1,
    )
    # Fit on the combined corpus so TF-IDF "knows" about both ticket and KB
    # vocabulary; then split the resulting matrix.
    all_docs = kb_docs + ticket_docs
    matrix = vectoriser.fit_transform(all_docs)
    kb_matrix = matrix[: len(kb_docs)]
    ticket_matrix = matrix[len(kb_docs) :]

    # Pre-compute token sets per article for the explanation column.
    article_tokens = [set(_tokenise(d)) for d in kb_docs]

    out: list[dict] = []
    for i, ticket in enumerate(normalised_tickets):
        sims = cosine_similarity(ticket_matrix[i], kb_matrix).ravel()
        top_idx = np.argsort(-sims)[:TOP_K]
        ticket_tokens = set(_tokenise(ticket["text_lower"]))

        candidates = []
        for idx in top_idx:
            article = kb_articles[int(idx)]
            score = float(sims[int(idx)])
            candidates.append({
                "article_id": article["article_id"],
                "title": article["title"],
                "score": round(score, 4),
                "matched_terms": _matched_terms(ticket_tokens, article_tokens[int(idx)]),
                "safe_for_ai": bool(article["safe_for_ai"]),
            })

        out.append({
            "ticket_id": ticket["ticket_id"],
            "top_k": TOP_K,
            "candidates": candidates,
        })

    ensure_artifacts_dir()
    RETRIEVAL.write_text(
        json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return out
