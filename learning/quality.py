"""
Quality scoring pipeline for VISION Learning.

Each raw document is scored across three dimensions:
  - Authority (0-100): based on domain and source tier
  - Relevance (0-100): keyword match to technology/coding topics
  - Freshness  (0-100): exponential decay from publish date

Final quality_score = weighted average (authority 40% + relevance 40% + freshness 20%)
Items below settings.min_quality_score are rejected.
"""
import math
import re
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Authority scoring — domain whitelist tiers
# ---------------------------------------------------------------------------

# Tier 1: official, authoritative (score 90-100)
TIER1_DOMAINS = {
    "python.org", "docs.python.org",
    "developer.mozilla.org", "tc39.es",
    "reactjs.org", "react.dev", "nextjs.org",
    "nodejs.org", "npmjs.com",
    "rust-lang.org", "go.dev", "golang.org",
    "postgresql.org", "sqlite.org", "redis.io", "mongodb.com",
    "github.com/ollama", "ollama.com",
    "tensorflow.org", "pytorch.org", "huggingface.co",
    "w3.org", "ietf.org", "rfc-editor.org",
    "nist.gov", "cve.mitre.org", "nvd.nist.gov",
    "arxiv.org",
    "docs.djangoproject.com",
    "tailwindcss.com", "vitejs.dev",
    "kubernetes.io", "docker.com",
    "cloud.google.com", "aws.amazon.com", "azure.microsoft.com",
    "developer.apple.com", "developer.android.com",
}

# Tier 2: reputable publications (score 70-85)
TIER2_DOMAINS = {
    "github.com", "github.blog",
    "stackoverflow.com", "blog.stackoverflow.com",
    "techcrunch.com", "theverge.com", "wired.com",
    "arstechnica.com", "zdnet.com",
    "thenewstack.io", "infoq.com",
    "smashingmagazine.com", "css-tricks.com",
    "changelog.com", "devto", "dev.to",
    "hackernews.com", "news.ycombinator.com",
    "reddit.com/r/programming",
    "medium.com", "substack.com",
}

# Everything else defaults to Tier 3 (score 40-55)


def score_authority(url: str, source_authority_score: Optional[int] = None) -> int:
    """Return 0-100 authority score for a URL."""
    if source_authority_score is not None:
        return source_authority_score
    if not url:
        return 40
    url_lower = url.lower()
    for domain in TIER1_DOMAINS:
        if domain in url_lower:
            return 95
    for domain in TIER2_DOMAINS:
        if domain in url_lower:
            return 72
    return 45


# ---------------------------------------------------------------------------
# Relevance scoring — keyword matching to tech/coding topics
# ---------------------------------------------------------------------------

TECH_KEYWORDS = {
    # Programming languages
    "python", "javascript", "typescript", "rust", "go", "golang", "java", "c++",
    "ruby", "php", "swift", "kotlin", "scala", "haskell",
    # Web/frontend
    "react", "vue", "angular", "next.js", "nextjs", "tailwind", "css", "html",
    "svelte", "vite", "webpack", "babel",
    # Backend / infra
    "node", "django", "fastapi", "flask", "express", "rails",
    "docker", "kubernetes", "terraform", "ansible", "nginx",
    # Databases
    "postgresql", "postgres", "mysql", "sqlite", "mongodb", "redis",
    "elasticsearch", "cassandra", "dynamodb",
    # AI / ML
    "llm", "rag", "ollama", "langchain", "openai", "gpt", "claude", "gemini",
    "machine learning", "neural network", "transformer", "embedding",
    "artificial intelligence", "computer vision",
    # General tech
    "api", "rest", "graphql", "grpc", "microservice", "serverless",
    "security", "vulnerability", "cve", "exploit", "authentication",
    "performance", "optimization", "algorithm", "data structure",
    "open source", "release", "update", "version", "framework",
}


def score_relevance(text: str) -> int:
    """Return 0-100 relevance score based on keyword density."""
    if not text:
        return 0
    text_lower = text.lower()
    words = set(re.findall(r"\w+", text_lower))
    # multi-word keyword matching
    full_text_for_phrases = text_lower

    hits = 0
    for kw in TECH_KEYWORDS:
        if " " in kw:
            if kw in full_text_for_phrases:
                hits += 2
        elif kw in words:
            hits += 1

    # Cap: 20+ hits = 100, 0 hits = 0; square-root scaled
    score = min(100, int(math.sqrt(max(hits, 0)) * 22))
    return score


# ---------------------------------------------------------------------------
# Freshness scoring — time-decay from publish date
# ---------------------------------------------------------------------------

# Days to reach 0 freshness (documents older than this get 0)
FRESHNESS_DECAY_DAYS = 365


def score_freshness(published_at: Optional[datetime]) -> int:
    """Return 0-100 freshness score. Recent = 100, 1-year-old = 0."""
    if published_at is None:
        return 60  # unknown date — give neutral score
    now = datetime.now(tz=timezone.utc)
    if published_at.tzinfo is None:
        published_at = published_at.replace(tzinfo=timezone.utc)
    age_days = max(0, (now - published_at).days)
    if age_days == 0:
        return 100
    score = max(0, int(100 * (1 - age_days / FRESHNESS_DECAY_DAYS)))
    return score


# ---------------------------------------------------------------------------
# Composite quality score
# ---------------------------------------------------------------------------

def compute_quality(authority: int, relevance: int, freshness: int) -> int:
    """
    Weighted composite score:
      authority 40% + relevance 40% + freshness 20%
    """
    return int(authority * 0.4 + relevance * 0.4 + freshness * 0.2)


def classify_confidence(quality_score: int) -> str:
    if quality_score >= 80:
        return "high"
    elif quality_score >= 55:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def is_duplicate(content_hash: str) -> bool:
    """Check if a KnowledgeItem with this hash already exists."""
    from learning.models import KnowledgeItem
    return KnowledgeItem.objects.filter(content_hash=content_hash).exists()


# ---------------------------------------------------------------------------
# Contradiction detection
# ---------------------------------------------------------------------------

def find_contradictions(summary: str, category: str, similarity_threshold: float = 0.92) -> list:
    """
    Find existing KnowledgeItems semantically very similar but with diverging summaries.
    Returns a list of conflicting items (may be empty).

    Strategy: cosine similarity ≥ threshold on embedding → flag as potential conflict.
    The calling code must decide whether to store both as CONFLICT or supersede.
    """
    from learning.models import KnowledgeItem
    from learning.retrieval import cosine_similarity, embed_text

    try:
        query_emb = embed_text(summary)
    except Exception:
        return []

    candidates = list(
        KnowledgeItem.objects.filter(
            category=category, status="active"
        ).exclude(embedding=None).values("id", "summary", "embedding")[:500]
    )

    conflicts = []
    for c in candidates:
        try:
            sim = cosine_similarity(query_emb, c["embedding"])
            if sim >= similarity_threshold:
                # High similarity in embedding but check if summaries differ meaningfully
                conflicts.append(str(c["id"]))
        except Exception:
            continue
    return conflicts
