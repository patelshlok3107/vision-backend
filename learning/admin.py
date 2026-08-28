"""Django Admin for VISION Learning."""
from django.contrib import admin
from django.utils import timezone
from django.utils.html import format_html

from .models import (
    KnowledgeSource, KnowledgeItem, LearningRun,
    LearningNotification, LearningSettings, AdminTrainingExample
)


@admin.register(KnowledgeSource)
class KnowledgeSourceAdmin(admin.ModelAdmin):
    list_display = ["name", "source_type", "authority_tier", "authority_score", "category", "is_active", "last_fetched_at"]
    list_filter = ["source_type", "authority_tier", "category", "is_active"]
    search_fields = ["name", "url"]
    list_editable = ["is_active", "authority_score"]
    fieldsets = [
        (None, {"fields": ["name", "url", "source_type", "authority_tier", "authority_score"]}),
        ("Classification", {"fields": ["category", "tags"]}),
        ("Schedule", {"fields": ["is_active", "fetch_interval_hours"]}),
        ("Notes", {"fields": ["notes"]}),
    ]


@admin.register(KnowledgeItem)
class KnowledgeItemAdmin(admin.ModelAdmin):
    list_display = [
        "title_short", "category", "subcategory", "quality_score",
        "confidence", "status", "collected_at", "source_name",
    ]
    list_filter = ["status", "category", "confidence", "admin_approved"]
    search_fields = ["title", "summary", "source_name", "tags"]
    readonly_fields = [
        "content_hash", "embedding_dims", "collected_at",
        "quality_score", "relevance_score", "freshness_score",
        "authority_score", "conflict_note",
    ]
    fieldsets = [
        ("Content", {"fields": ["title", "summary", "content", "content_hash"]}),
        ("Source", {"fields": ["source", "source_url", "source_name", "authority_score", "published_at"]}),
        ("Classification", {"fields": ["category", "subcategory", "tags", "version"]}),
        ("Quality", {"fields": ["quality_score", "relevance_score", "freshness_score", "confidence"]}),
        ("Lifecycle", {"fields": ["status", "admin_approved", "rejection_reason", "superseded_by", "conflict_note"]}),
        ("Embedding", {"fields": ["embedding_dims"]}),
    ]
    actions = ["approve_items", "reject_items", "mark_superseded"]

    def title_short(self, obj):
        return obj.title[:70] + "…" if len(obj.title) > 70 else obj.title
    title_short.short_description = "Title"

    def embedding_dims(self, obj):
        if obj.embedding:
            return f"{len(obj.embedding)} dims"
        return "No embedding"
    embedding_dims.short_description = "Embedding"

    @admin.action(description="Approve selected items")
    def approve_items(self, request, queryset):
        queryset.update(admin_approved=True, status="active")

    @admin.action(description="Reject selected items")
    def reject_items(self, request, queryset):
        queryset.update(status="rejected", rejection_reason="Admin rejected")

    @admin.action(description="Mark selected as superseded")
    def mark_superseded(self, request, queryset):
        queryset.update(status="superseded")


@admin.register(LearningRun)
class LearningRunAdmin(admin.ModelAdmin):
    list_display = [
        "started_at", "triggered_by", "status", "duration_display",
        "items_added", "items_rejected", "duplicates_skipped", "benchmark_overall",
    ]
    list_filter = ["status", "triggered_by", "benchmark_regression"]
    readonly_fields = [
        "started_at", "finished_at", "duration_display",
        "sources_processed", "documents_fetched", "items_added",
        "items_updated", "items_rejected", "duplicates_skipped",
        "conflicts_detected", "category_breakdown",
        "benchmark_results", "previous_benchmark",
        "benchmark_regression", "error_log",
    ]
    actions = ["trigger_rollback"]

    def duration_display(self, obj):
        d = obj.duration_seconds
        if d is None:
            return "—"
        return f"{d}s"
    duration_display.short_description = "Duration"

    def benchmark_overall(self, obj):
        if obj.benchmark_results:
            score = obj.benchmark_results.get("overall", "—")
            if obj.benchmark_regression:
                return format_html('<span style="color:red">⚠ {}%</span>', score)
            return f"{score}%"
        return "—"
    benchmark_overall.short_description = "Benchmark"

    @admin.action(description="⚠ Rollback selected run (deactivates items from this run)")
    def trigger_rollback(self, request, queryset):
        from .models import KnowledgeItem, ItemStatus
        for run in queryset:
            KnowledgeItem.objects.filter(learning_run=run).update(
                status=ItemStatus.REJECTED,
                rejection_reason=f"Rolled back by admin from run {run.id}"
            )
        self.message_user(request, f"Rolled back {queryset.count()} run(s).")


@admin.register(LearningNotification)
class LearningNotificationAdmin(admin.ModelAdmin):
    list_display = ["severity", "title", "is_read", "created_at"]
    list_filter = ["severity", "is_read"]
    actions = ["mark_read"]

    @admin.action(description="Mark as read")
    def mark_read(self, request, queryset):
        queryset.update(is_read=True)


@admin.register(LearningSettings)
class LearningSettingsAdmin(admin.ModelAdmin):
    fieldsets = [
        ("Master", {"fields": ["enabled", "schedule_type", "schedule_hour", "schedule_day"]}),
        ("Topic Toggles", {"fields": ["news_enabled", "technology_enabled", "coding_enabled", "security_enabled", "ai_ml_enabled"]}),
        ("Quality", {"fields": ["min_quality_score", "auto_approve"]}),
        ("Regression Guard", {"fields": ["reject_on_benchmark_regression"]}),
        ("Fine-tuning (Level 3 — Off by default)", {"fields": ["auto_finetune_enabled"]}),
    ]

    def has_add_permission(self, request):
        return not LearningSettings.objects.exists()

    def has_delete_permission(self, request, obj=None):
        return False


@admin.register(AdminTrainingExample)
class AdminTrainingExampleAdmin(admin.ModelAdmin):
    list_display = ["prompt_short", "quality", "category", "approved", "created_at"]
    list_filter = ["quality", "category", "approved"]
    search_fields = ["prompt", "answer", "tags"]
    readonly_fields = ["preview_summary", "knowledge_item", "approved_at"]
    actions = ["approve_examples"]

    def prompt_short(self, obj):
        return obj.prompt[:70] + "…" if len(obj.prompt) > 70 else obj.prompt
    prompt_short.short_description = "Prompt"

    @admin.action(description="Approve and process selected examples")
    def approve_examples(self, request, queryset):
        from .tasks import process_admin_upload
        from django.utils import timezone
        for ex in queryset.filter(approved=False):
            ex.approved = True
            ex.approved_at = timezone.now()
            ex.save(update_fields=["approved", "approved_at"])
            process_admin_upload.delay(str(ex.id))
        self.message_user(request, f"Approved and queued {queryset.count()} example(s) for processing.")
