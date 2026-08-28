"""
Vector embedding retrieval for VISION Learning.

retrieve_knowledge(query, top_k=5) →  list of relevant KnowledgeItem snippets
Used by agent.py to augment context when a query is knowledge-relevant.
"""
import logging
import math
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Embedding helper
# ---------------------------------------------------------------------------

def embed_text(text: str) -> list[float]:
    """
    Embed text using the existing nomic-embed-text Ollama model.
    Reuses the existing ai.services.embeddings module.
    """
    from ai.services.embeddings import get_embedding
    return get_embedding(text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Numerically stable cosine similarity between two float vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = math.sqrt(sum(x * x for x in a))
    mag_b = math.sqrt(sum(x * x for x in b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)


# ---------------------------------------------------------------------------
# Query relevance heuristic (fast, no embedding needed)
# ---------------------------------------------------------------------------

_STABLE_PATTERNS = (
    r"^what is (a |an |the )?",
    r"^define ",
    r"^explain ",
    r"^how does .+ work",
    r"^what does .+ mean",
)

import re
_STABLE_RE = re.compile("|".join(_STABLE_PATTERNS), re.IGNORECASE)

# Keywords that suggest the user wants current / up-to-date information
_KNOWLEDGE_TRIGGER_WORDS = {
    "latest", "new", "recent", "current", "update", "release", "version",
    "news", "announced", "just", "2024", "2025", "2026", "changed",
    "deprecated", "broke", "breaking", "migrate", "upgrade", "security",
    "vulnerability", "cve", "patch", "advisory", "changelog",
}


def is_knowledge_relevant(query: str) -> bool:
    """
    Heuristic: should we search the knowledge base for this query?
    Returns False for stable factual questions that don't need current info.
    Returns True for queries about recent events, versions, changes, etc.
    """
    if not query:
        return False
    q_lower = query.lower()
    # Fast-pass on knowledge trigger words
    words = set(re.findall(r"\w+", q_lower))
    if words & _KNOWLEDGE_TRIGGER_WORDS:
        return True
    # Skip for stable definitional questions
    if _STABLE_RE.match(query.strip()):
        return False
    # If query is long and technical, probably useful to check KB
    if len(query.split()) > 8:
        return True
    return False


# ---------------------------------------------------------------------------
# Main retrieval function
# ---------------------------------------------------------------------------

def retrieve_knowledge(
    query: str,
    top_k: int = 5,
    min_similarity: float = 0.60,
    categories: Optional[list] = None,
) -> list[dict]:
    """
    Embed the query and find the top-k most relevant active KnowledgeItems.
    Returns a list of dicts: {title, summary, source_url, quality_score, category}

    Falls back gracefully if embedding or DB fails.
    """
    from learning.models import KnowledgeItem

    try:
        query_emb = embed_text(query)
    except Exception as e:
        logger.debug("[LEARNING] Embedding failed for retrieval: %s", e)
        return []

    # Load active items that have embeddings, ordered by quality desc
    qs = KnowledgeItem.objects.filter(
        status="active",
        admin_approved=True,
        embedding__isnull=False,
    )
    if categories:
        qs = qs.filter(category__in=categories)

    # Stream values to avoid loading all ORM objects into memory
    candidates = list(qs.values("id", "title", "summary", "source_url", "quality_score", "category", "embedding")[:2000])

    if not candidates:
        return []

    scored = []
    for c in candidates:
        try:
            sim = cosine_similarity(query_emb, c["embedding"])
            if sim >= min_similarity:
                scored.append((sim, c))
        except Exception:
            continue

    scored.sort(key=lambda x: x[0], reverse=True)
    results = []
    for sim, c in scored[:top_k]:
        results.append({
            "title": c["title"],
            "summary": c["summary"],
            "source_url": c["source_url"],
            "quality_score": c["quality_score"],
            "category": c["category"],
            "similarity": round(sim, 3),
        })

    logger.debug("[LEARNING] Retrieved %d knowledge items for query (top sim=%.2f)",
                 len(results), scored[0][0] if scored else 0)
    return results
