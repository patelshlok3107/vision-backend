"""
VISION Learning Pipeline — end-to-end orchestration.

process_document(doc, run, settings) → None | KnowledgeItem
  1. Deduplicate (content hash)
  2. Score quality (authority + relevance + freshness)
  3. Reject below min_quality_score
  4. Summarise via Ollama (concise, 150-word max)
  5. Embed via nomic-embed-text
  6. Detect semantic contradictions with existing items
  7. Save KnowledgeItem (or update if existing superseded)
  8. Fire admin notification for critical items

run_pipeline(run_id) → stats dict
  Orchestrates a full collection + processing cycle.
"""
import logging
import time
from datetime import timezone

from django.utils import timezone as dj_timezone

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-document processing
# ---------------------------------------------------------------------------

def _summarise(content: str, title: str) -> str:
    """
    Use Ollama to generate a concise factual summary (≤ 150 words).
    Falls back to the first 300 chars of content if Ollama is unavailable.
    """
    from ai.services.ollama_client import client, OllamaError

    prompt = (
        f"Summarize the following content in at most 150 words. "
        f"Be factual, concise, and technical. Only output the summary.\n\n"
        f"Title: {title}\n\n"
        f"Content:\n{content[:3000]}"
    )
    try:
        summary = client.chat(
            [{"role": "user", "content": prompt}],
            temperature=0.1,
            stream=False,
        )
        return (summary or "").strip()[:1200]
    except OllamaError as e:
        logger.warning("[LEARNING] Summarisation failed (%s), using truncation", e)
        return content[:300].strip()


def process_document(doc, run, settings_obj) -> bool:
    """
    Process a single RawDocument through the quality pipeline.
    Returns True if item was saved, False if rejected/skipped.
    """
    from learning.models import KnowledgeItem, KnowledgeSource, ItemStatus
    from learning.quality import (
        score_authority, score_relevance, score_freshness,
        compute_quality, classify_confidence, is_duplicate,
        find_contradictions, TIER1_DOMAINS,
    )
    from learning.retrieval import embed_text

    content = (doc.content or "").strip()
    if not content or len(content) < 50:
        run.items_rejected += 1
        return False

    content_hash = KnowledgeItem.make_hash(content)

    # 1. Deduplication
    if is_duplicate(content_hash):
        run.duplicates_skipped += 1
        return False

    # 2. Quality scoring
    authority = score_authority(doc.url, doc.authority_score)
    relevance = score_relevance(content + " " + doc.title)
    freshness = score_freshness(doc.published_at)
    quality = compute_quality(authority, relevance, freshness)

    if quality < settings_obj.min_quality_score:
        logger.debug("[LEARNING] Rejected (Q=%d < %d): %s", quality, settings_obj.min_quality_score, doc.title[:60])
        run.items_rejected += 1
        return False

    # 3. Summarise
    summary = _summarise(content, doc.title)
    if not summary:
        run.items_rejected += 1
        return False

    # 4. Embed
    try:
        embedding = embed_text(summary)
    except Exception as e:
        logger.warning("[LEARNING] Embedding failed for %s: %s", doc.title[:50], e)
        embedding = None

    # 5. Contradiction detection
    conflicts = []
    if embedding:
        try:
            conflicts = find_contradictions(summary, doc.category)
        except Exception:
            pass

    # 6. Determine status
    status = ItemStatus.ACTIVE if settings_obj.auto_approve else ItemStatus.PENDING
    if conflicts:
        status = ItemStatus.CONFLICT
        run.conflicts_detected += 1

    # 7. Source FK (if from admin KnowledgeSource)
    source_obj = None
    if doc.source_id:
        try:
            source_obj = KnowledgeSource.objects.get(id=doc.source_id)
        except KnowledgeSource.DoesNotExist:
            pass

    item = KnowledgeItem(
        title=doc.title[:500],
        summary=summary,
        content=content,
        content_hash=content_hash,
        source=source_obj,
        source_url=doc.url[:1000] if doc.url else "",
        source_name=doc.source_name[:255] if doc.source_name else "",
        authority_score=authority,
        category=doc.category,
        tags=doc.tags or [],
        quality_score=quality,
        relevance_score=relevance,
        freshness_score=freshness,
        confidence=classify_confidence(quality),
        status=status,
        admin_approved=settings_obj.auto_approve,
        published_at=doc.published_at,
        embedding=embedding,
        learning_run=run,
    )
    if conflicts:
        item.conflict_note = f"Semantically similar to items: {', '.join(conflicts[:3])}"

    try:
        item.save()
    except Exception as e:
        logger.error("[LEARNING] Failed to save KnowledgeItem: %s", e)
        return False

    run.items_added += 1

    # Update source last_fetched_at
    if source_obj:
        source_obj.last_fetched_at = dj_timezone.now()
        source_obj.save(update_fields=["last_fetched_at"])

    # Update per-category breakdown
    cat = doc.category
    run.category_breakdown[cat] = run.category_breakdown.get(cat, 0) + 1

    # 8. Fire notification for critical items (security vulns, major releases)
    _maybe_notify(item, run)

    return True


def _maybe_notify(item, run):
    """Create a LearningNotification for important items."""
    from learning.models import LearningNotification, NotificationSeverity

    title_lower = item.title.lower()
    severity = None
    if any(kw in title_lower for kw in ["critical", "cve", "vulnerability", "zero-day", "exploit", "rce"]):
        severity = NotificationSeverity.CRITICAL
    elif any(kw in title_lower for kw in ["security", "advisory", "patch", "release", "major"]):
        severity = NotificationSeverity.WARNING
    elif any(kw in title_lower for kw in ["announced", "launched", "new model", "new llm"]):
        severity = NotificationSeverity.INFO

    if severity:
        LearningNotification.objects.create(
            learning_run=run,
            severity=severity,
            title=item.title[:255],
            body=item.summary[:1000],
            knowledge_item=item,
        )


# ---------------------------------------------------------------------------
# Full pipeline orchestration
# ---------------------------------------------------------------------------

def run_pipeline(run_id: str) -> dict:
    """
    Full collection + processing cycle for a LearningRun.
    Called from the Celery task.
    """
    from learning.models import LearningRun, LearningSettings, RunStatus
    from learning.collectors import collect_all

    try:
        run = LearningRun.objects.get(id=run_id)
    except LearningRun.DoesNotExist:
        logger.error("[LEARNING] LearningRun %s not found", run_id)
        return {"error": "run not found"}

    settings_obj = LearningSettings.get()
    run.status = RunStatus.RUNNING
    run.save(update_fields=["status"])

    try:
        # Collect
        docs = collect_all(settings_obj)
        run.documents_fetched = len(docs)
        run.sources_processed = len(set(d.source_name for d in docs))

        # Process each document
        for doc in docs:
            try:
                process_document(doc, run, settings_obj)
            except Exception as e:
                logger.error("[LEARNING] Error processing doc '%s': %s", doc.title[:40], e)
            # Periodically save running stats
            if (run.items_added + run.items_rejected) % 10 == 0:
                run.save(update_fields=[
                    "documents_fetched", "sources_processed",
                    "items_added", "items_rejected", "duplicates_skipped",
                    "conflicts_detected", "category_breakdown",
                ])

        run.status = RunStatus.SUCCESS
    except Exception as e:
        run.status = RunStatus.FAILED
        run.error_log = str(e)
        logger.exception("[LEARNING] Pipeline failed: %s", e)
    finally:
        run.finished_at = dj_timezone.now()
        run.save()

    return {
        "status": run.status,
        "documents_fetched": run.documents_fetched,
        "items_added": run.items_added,
        "items_rejected": run.items_rejected,
        "duplicates_skipped": run.duplicates_skipped,
        "conflicts_detected": run.conflicts_detected,
    }
