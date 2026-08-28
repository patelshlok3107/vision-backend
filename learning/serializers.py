"""DRF serializers for the VISION Learning REST API."""
from rest_framework import serializers

from .models import (
    KnowledgeSource, KnowledgeItem, LearningRun,
    LearningNotification, LearningSettings, AdminTrainingExample
)


class KnowledgeSourceSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeSource
        fields = ["id", "name", "url", "source_type", "authority_tier", "authority_score",
                  "category", "tags", "is_active", "last_fetched_at", "fetch_interval_hours"]


class KnowledgeItemListSerializer(serializers.ModelSerializer):
    """Lightweight serializer for list views."""
    class Meta:
        model = KnowledgeItem
        fields = [
            "id", "title", "summary", "category", "subcategory", "tags",
            "quality_score", "confidence", "status", "source_name",
            "source_url", "published_at", "collected_at",
        ]


class KnowledgeItemDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = KnowledgeItem
        exclude = ["embedding", "content"]  # large fields excluded from API responses


class LearningRunSerializer(serializers.ModelSerializer):
    duration_seconds = serializers.ReadOnlyField()
    benchmark_overall = serializers.SerializerMethodField()

    class Meta:
        model = LearningRun
        fields = [
            "id", "triggered_by", "started_at", "finished_at", "status",
            "sources_processed", "documents_fetched",
            "items_added", "items_updated", "items_rejected",
            "duplicates_skipped", "conflicts_detected",
            "category_breakdown", "benchmark_results", "previous_benchmark",
            "benchmark_regression", "duration_seconds", "benchmark_overall",
            "error_log",
        ]

    def get_benchmark_overall(self, obj):
        if obj.benchmark_results:
            return obj.benchmark_results.get("overall")
        return None


class LearningNotificationSerializer(serializers.ModelSerializer):
    class Meta:
        model = LearningNotification
        fields = ["id", "severity", "title", "body", "is_read", "created_at"]


class LearningSettingsSerializer(serializers.ModelSerializer):
    class Meta:
        model = LearningSettings
        exclude = ["id", "updated_at"]


class AdminTrainingExampleSerializer(serializers.ModelSerializer):
    class Meta:
        model = AdminTrainingExample
        fields = [
            "id", "prompt", "answer", "quality", "reason",
            "category", "subcategory", "tags", "source_description",
            "approved", "preview_summary", "created_at",
        ]
        read_only_fields = ["approved", "preview_summary", "created_at"]


class DashboardSerializer(serializers.Serializer):
    """Combined dashboard stats serializer."""
    enabled = serializers.BooleanField()
    schedule_type = serializers.CharField()
    schedule_hour = serializers.IntegerField()
    total_knowledge_items = serializers.IntegerField()
    active_items = serializers.IntegerField()
    pending_items = serializers.IntegerField()
    total_sources = serializers.IntegerField()
    unread_notifications = serializers.IntegerField()
    last_run = LearningRunSerializer(allow_null=True)
    category_breakdown = serializers.DictField()
    latest_benchmark = serializers.DictField(allow_null=True)
