"""
Celery tasks for VISION Learning.

Tasks:
  run_daily_learning()          — Scheduled daily pipeline run
  run_manual_learning()         — Admin-triggered run (selected sources or all)
  process_admin_upload(id)      — Process a single AdminTrainingExample
  run_coding_benchmark()        — Run coding quality benchmark
"""
import logging

from celery import shared_task

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="learning.run_daily_learning", max_retries=1)
def run_daily_learning(self):
    """Scheduled daily learning pipeline. Respects LearningSettings.enabled."""
    from learning.models import LearningRun, LearningSettings, RunStatus
    from learning.pipeline import run_pipeline
    from learning.benchmark import run_coding_benchmark_sync

    settings_obj = LearningSettings.get()
    if not settings_obj.enabled or settings_obj.schedule_type == "disabled":
        logger.info("[LEARNING] Daily learning skipped — disabled in settings")
        return {"skipped": True, "reason": "disabled"}

    logger.info("[LEARNING] Starting daily learning run")
    run = LearningRun.objects.create(triggered_by="scheduler")

    try:
        stats = run_pipeline(str(run.id))
    except Exception as exc:
        logger.exception("[LEARNING] Daily pipeline failed: %s", exc)
        run.status = RunStatus.FAILED
        run.error_log = str(exc)
        run.save()
        self.retry(exc=exc, countdown=300)
        return

    # Optionally run benchmark after successful run
    try:
        benchmark = run_coding_benchmark_sync(quick=True)
        run.refresh_from_db()
        run.benchmark_results = benchmark
        run.save(update_fields=["benchmark_results"])
        _check_benchmark_regression(run, benchmark, settings_obj)
    except Exception as e:
        logger.warning("[LEARNING] Benchmark after daily run failed: %s", e)

    logger.info("[LEARNING] Daily run complete: %s", stats)
    return stats


@shared_task(bind=True, name="learning.run_manual_learning")
def run_manual_learning(self, source_ids=None):
    """Admin-triggered run. source_ids can limit to specific KnowledgeSources."""
    from learning.models import LearningRun, RunStatus
    from learning.pipeline import run_pipeline

    logger.info("[LEARNING] Starting manual learning run (source_ids=%s)", source_ids)
    run = LearningRun.objects.create(triggered_by="admin")
    try:
        stats = run_pipeline(str(run.id))
    except Exception as exc:
        logger.exception("[LEARNING] Manual pipeline failed: %s", exc)
        run.status = RunStatus.FAILED
        run.error_log = str(exc)
        run.save()
        return {"error": str(exc)}

    logger.info("[LEARNING] Manual run complete: %s", stats)
    return stats


@shared_task(name="learning.process_admin_upload")
def process_admin_upload(training_example_id: str):
    """
    Process an AdminTrainingExample — generate summary, embed, save as KnowledgeItem.
    Called after admin previews and approves an upload.
    """
    import uuid
    from django.utils import timezone
    from learning.models import (
        AdminTrainingExample, KnowledgeItem, LearningSettings, ItemStatus
    )
    from learning.quality import score_relevance, classify_confidence
    from learning.retrieval import embed_text

    try:
        example = AdminTrainingExample.objects.get(id=training_example_id)
    except AdminTrainingExample.DoesNotExist:
        logger.error("[LEARNING] AdminTrainingExample %s not found", training_example_id)
        return

    if not example.approved:
        logger.info("[LEARNING] Example %s not approved yet — skipping", training_example_id)
        return

    settings_obj = LearningSettings.get()
    content = f"Prompt: {example.prompt}\n\nAnswer: {example.answer}"
    content_hash = KnowledgeItem.make_hash(content)

    if KnowledgeItem.objects.filter(content_hash=content_hash).exists():
        logger.info("[LEARNING] Admin upload already in KB: %s", training_example_id)
        return

    try:
        embedding = embed_text(content)
    except Exception:
        embedding = None

    relevance = score_relevance(content)
    quality = min(100, int(relevance * 0.5 + 50))  # admin content gets base 50 + relevance bonus

    item = KnowledgeItem.objects.create(
        title=example.prompt[:120] + "..." if len(example.prompt) > 120 else example.prompt,
        summary=example.preview_summary or content[:400],
        content=content,
        content_hash=content_hash,
        source_name="Admin Training",
        authority_score=90,  # admin-provided content gets high authority
        category=example.category,
        subcategory=example.subcategory,
        tags=example.tags,
        quality_score=quality,
        relevance_score=relevance,
        freshness_score=100,
        confidence="high",
        status=ItemStatus.ACTIVE,
        admin_approved=True,
        embedding=embedding,
    )
    example.knowledge_item = item
    example.save(update_fields=["knowledge_item"])
    logger.info("[LEARNING] Admin upload %s → KnowledgeItem %s", training_example_id, item.id)


@shared_task(name="learning.run_coding_benchmark")
def run_coding_benchmark():
    """Celery wrapper for the coding benchmark. Stores result in latest run."""
    from learning.benchmark import run_coding_benchmark_sync
    from learning.models import LearningRun

    results = run_coding_benchmark_sync(quick=False)

    # Attach to most recent successful run if available
    latest = LearningRun.objects.filter(status="success").first()
    if latest:
        latest.benchmark_results = results
        latest.save(update_fields=["benchmark_results"])

    return results


# ---------------------------------------------------------------------------
# Private: benchmark regression check
# ---------------------------------------------------------------------------

def _check_benchmark_regression(run, new_results, settings_obj):
    """Compare new benchmark against previous run. Notify on regression > 5%."""
    from learning.models import LearningRun, LearningNotification, NotificationSeverity

    prev_run = (
        LearningRun.objects.filter(status="success", benchmark_results__isnull=False)
        .exclude(id=run.id)
        .first()
    )
    if not prev_run or not prev_run.benchmark_results:
        return

    prev_overall = prev_run.benchmark_results.get("overall", 0)
    new_overall = new_results.get("overall", 0)
    delta = new_overall - prev_overall

    run.previous_benchmark = prev_run.benchmark_results
    if delta < -5:
        run.benchmark_regression = True
        LearningNotification.objects.create(
            learning_run=run,
            severity=NotificationSeverity.WARNING,
            title=f"⚠ Coding benchmark decreased: {prev_overall:.0f}% → {new_overall:.0f}%",
            body=(
                f"The learning update caused a {abs(delta):.1f}% drop in the coding benchmark. "
                f"Consider reviewing the new knowledge items added in this run."
            ),
        )
        logger.warning("[LEARNING] Benchmark regression: %.1f%% → %.1f%%", prev_overall, new_overall)
    run.save(update_fields=["previous_benchmark", "benchmark_regression"])
