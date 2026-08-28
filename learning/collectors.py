"""
Information collectors for VISION Learning.

Each collector fetches raw content from a source and returns a list of RawDocument
objects. The pipeline in pipeline.py then processes these through quality scoring.

Collectors:
  - collect_rss_feeds()   — Parses registered RSS/Atom feeds
  - collect_doc_urls()    — Fetches registered documentation/release URLs
  - collect_admin_sources() — Pulls active admin-registered sources from DB
"""
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structure passed between collectors and the pipeline
# ---------------------------------------------------------------------------

@dataclass
class RawDocument:
    title: str
    content: str
    url: str = ""
    source_name: str = ""
    published_at: Optional[datetime] = None
    source_type: str = "url"
    authority_score: Optional[int] = None
    category: str = "general"
    tags: list = field(default_factory=list)
    source_id: Optional[str] = None  # KnowledgeSource.id if from DB


# ---------------------------------------------------------------------------
# Default built-in sources
# ---------------------------------------------------------------------------

DEFAULT_RSS_FEEDS = [
    {
        "name": "Hacker News Top Stories",
        "url": "https://news.ycombinator.com/rss",
        "category": "technology",
        "tags": ["tech", "programming"],
    },
    {
        "name": "GitHub Blog",
        "url": "https://github.blog/feed/",
        "category": "technology",
        "tags": ["github", "open-source"],
    },
    {
        "name": "Python Insider",
        "url": "https://blog.python.org/feeds/posts/default",
        "category": "programming",
        "tags": ["python"],
    },
    {
        "name": "React Blog",
        "url": "https://react.dev/blog/rss.xml",
        "category": "web_dev",
        "tags": ["react", "javascript"],
    },
    {
        "name": "Node.js Blog",
        "url": "https://nodejs.org/en/feed/blog.xml",
        "category": "web_dev",
        "tags": ["nodejs", "javascript"],
    },
    {
        "name": "Rust Blog",
        "url": "https://blog.rust-lang.org/feed.xml",
        "category": "programming",
        "tags": ["rust"],
    },
    {
        "name": "Go Blog",
        "url": "https://go.dev/blog/feed.atom",
        "category": "programming",
        "tags": ["go", "golang"],
    },
    {
        "name": "PostgreSQL News",
        "url": "https://www.postgresql.org/news.rss",
        "category": "databases",
        "tags": ["postgresql", "database"],
    },
    {
        "name": "The New Stack",
        "url": "https://thenewstack.io/feed/",
        "category": "technology",
        "tags": ["cloud", "devops", "kubernetes"],
    },
    {
        "name": "Security CVE Feed",
        "url": "https://nvd.nist.gov/feeds/xml/cve/misc/nvd-rss.xml",
        "category": "security",
        "tags": ["cve", "security", "vulnerability"],
    },
]

DEFAULT_DOC_URLS = [
    # Release pages — lightweight, mostly text
    {"name": "Python Changelog", "url": "https://docs.python.org/3/whatsnew/changelog.html", "category": "programming", "tags": ["python"]},
    {"name": "Django Releases", "url": "https://www.djangoproject.com/weblog/", "category": "web_dev", "tags": ["django", "python"]},
    {"name": "Ollama GitHub Releases", "url": "https://github.com/ollama/ollama/releases", "category": "ai_ml", "tags": ["ollama", "llm"]},
    {"name": "Hugging Face Blog", "url": "https://huggingface.co/blog", "category": "ai_ml", "tags": ["llm", "ai", "huggingface"]},
]


# ---------------------------------------------------------------------------
# RSS collector
# ---------------------------------------------------------------------------

def _parse_rss(url: str, max_items: int = 20) -> list[dict]:
    """
    Parse an RSS/Atom feed. Returns list of {title, url, content, published_at}.
    Uses built-in xml.etree only — no feedparser dependency required.
    """
    import xml.etree.ElementTree as ET
    import urllib.request

    NS = {
        "atom": "http://www.w3.org/2005/Atom",
        "content": "http://purl.org/rss/1.0/modules/content/",
        "dc": "http://purl.org/dc/elements/1.1/",
    }

    def _get_text(el, tag, default=""):
        child = el.find(tag)
        return (child.text or "").strip() if child is not None else default

    def _parse_date(s: str) -> Optional[datetime]:
        if not s:
            return None
        # Try RFC 2822 and ISO 8601
        for fmt in ["%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%SZ"]:
            try:
                return datetime.strptime(s.strip(), fmt)
            except ValueError:
                continue
        return None

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VISION-Learning/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            raw = resp.read()
        root = ET.fromstring(raw)
    except Exception as e:
        logger.warning("RSS fetch failed %s: %s", url, e)
        return []

    items = []
    # RSS 2.0: root/channel/item
    for item in root.findall(".//item")[:max_items]:
        title = _get_text(item, "title")
        link = _get_text(item, "link")
        desc = _get_text(item, "description") or _get_text(item, "content:encoded")
        pub_str = _get_text(item, "pubDate") or _get_text(item, "dc:date")
        items.append({"title": title, "url": link, "content": desc, "published_at": _parse_date(pub_str)})

    # Atom: entry elements
    if not items:
        for entry in root.findall(".//atom:entry", NS)[:max_items]:
            title_el = entry.find("atom:title", NS)
            link_el = entry.find("atom:link", NS)
            summary_el = entry.find("atom:summary", NS) or entry.find("atom:content", NS)
            updated_el = entry.find("atom:updated", NS)
            title = (title_el.text or "") if title_el is not None else ""
            link = (link_el.get("href") or "") if link_el is not None else ""
            content = (summary_el.text or "") if summary_el is not None else ""
            pub_str = (updated_el.text or "") if updated_el is not None else ""
            items.append({"title": title, "url": link, "content": content, "published_at": _parse_date(pub_str)})

    return items


def collect_rss_feeds(enabled_categories: Optional[set] = None) -> list[RawDocument]:
    """Fetch all configured RSS feeds and return RawDocuments."""
    docs = []
    for feed in DEFAULT_RSS_FEEDS:
        if enabled_categories and feed["category"] not in enabled_categories:
            continue
        try:
            items = _parse_rss(feed["url"])
            for it in items:
                content = it.get("content") or it.get("title") or ""
                if not content.strip():
                    continue
                docs.append(RawDocument(
                    title=it.get("title") or feed["name"],
                    content=content,
                    url=it.get("url") or feed["url"],
                    source_name=feed["name"],
                    published_at=it.get("published_at"),
                    source_type="rss",
                    category=feed["category"],
                    tags=feed.get("tags", []),
                ))
            time.sleep(0.3)  # be polite between requests
        except Exception as e:
            logger.error("RSS collect error for %s: %s", feed["name"], e)
    return docs


# ---------------------------------------------------------------------------
# URL / documentation collector
# ---------------------------------------------------------------------------

def _fetch_url_text(url: str, max_chars: int = 8000) -> str:
    """Fetch a URL and return plain text (strips HTML tags)."""
    import urllib.request
    import re as _re

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "VISION-Learning/1.0"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw = resp.read(max_chars * 4).decode("utf-8", errors="ignore")
        # Strip HTML
        text = _re.sub(r"<[^>]+>", " ", raw)
        text = _re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]
    except Exception as e:
        logger.warning("URL fetch failed %s: %s", url, e)
        return ""


def collect_doc_urls(enabled_categories: Optional[set] = None) -> list[RawDocument]:
    """Fetch registered documentation URLs."""
    docs = []
    for doc in DEFAULT_DOC_URLS:
        if enabled_categories and doc["category"] not in enabled_categories:
            continue
        content = _fetch_url_text(doc["url"])
        if not content:
            continue
        docs.append(RawDocument(
            title=doc["name"],
            content=content,
            url=doc["url"],
            source_name=doc["name"],
            source_type="url",
            category=doc["category"],
            tags=doc.get("tags", []),
        ))
        time.sleep(0.5)
    return docs


# ---------------------------------------------------------------------------
# Admin-registered sources from DB
# ---------------------------------------------------------------------------

def collect_admin_sources(enabled_categories: Optional[set] = None) -> list[RawDocument]:
    """Fetch sources registered by the admin in the KnowledgeSource table."""
    from learning.models import KnowledgeSource, SourceType
    docs = []
    qs = KnowledgeSource.objects.filter(is_active=True)
    if enabled_categories:
        qs = qs.filter(category__in=enabled_categories)

    for src in qs:
        try:
            if src.source_type == SourceType.RSS:
                items = _parse_rss(src.url, max_items=15)
                for it in items:
                    content = it.get("content") or it.get("title") or ""
                    if not content.strip():
                        continue
                    docs.append(RawDocument(
                        title=it.get("title") or src.name,
                        content=content,
                        url=it.get("url") or src.url,
                        source_name=src.name,
                        published_at=it.get("published_at"),
                        source_type="rss",
                        authority_score=src.authority_score,
                        category=src.category,
                        tags=src.tags,
                        source_id=str(src.id),
                    ))
            elif src.source_type == SourceType.URL:
                content = _fetch_url_text(src.url)
                if content:
                    docs.append(RawDocument(
                        title=src.name,
                        content=content,
                        url=src.url,
                        source_name=src.name,
                        source_type="url",
                        authority_score=src.authority_score,
                        category=src.category,
                        tags=src.tags,
                        source_id=str(src.id),
                    ))
        except Exception as e:
            logger.error("Admin source collect error for %s: %s", src.name, e)
        time.sleep(0.3)
    return docs


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def collect_all(settings_obj) -> list[RawDocument]:
    """
    Collect from all enabled sources, respecting per-category toggles
    from the LearningSettings singleton.
    """
    enabled = set()
    if settings_obj.news_enabled:
        enabled.update(["technology", "science", "general"])
    if settings_obj.technology_enabled:
        enabled.update(["technology", "devops"])
    if settings_obj.coding_enabled:
        enabled.update(["programming", "web_dev", "databases"])
    if settings_obj.security_enabled:
        enabled.add("security")
    if settings_obj.ai_ml_enabled:
        enabled.add("ai_ml")

    docs = []
    docs.extend(collect_rss_feeds(enabled_categories=enabled))
    docs.extend(collect_doc_urls(enabled_categories=enabled))
    docs.extend(collect_admin_sources(enabled_categories=enabled))
    logger.info("[LEARNING] Collected %d raw documents from all sources", len(docs))
    return docs
