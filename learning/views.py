"""REST API views for VISION Learning."""
import logging

from django.db.models import Count, Sum
from rest_framework import status
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAdminUser, IsAuthenticated
from rest_framework.response import Response
from rest_framework.views import APIView

from .models import (
    KnowledgeItem, KnowledgeSource, LearningRun,
    LearningNotification, LearningSettings, AdminTrainingExample, ItemStatus
)
from .serializers import (
    DashboardSerializer, KnowledgeItemListSerializer, KnowledgeItemDetailSerializer,
    KnowledgeSourceSerializer, LearningRunSerializer, LearningNotificationSerializer,
    LearningSettingsSerializer, AdminTrainingExampleSerializer,
)

logger = logging.getLogger(__name__)


class DashboardView(APIView):
    """GET /api/learning/dashboard/ — aggregated stats for the admin dashboard."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        settings_obj = LearningSettings.get()
        last_run = LearningRun.objects.first()

        # Category breakdown from all active items
        cat_qs = (
            KnowledgeItem.objects
            .filter(status=ItemStatus.ACTIVE)
            .values("category")
            .annotate(count=Count("id"))
        )
        breakdown = {row["category"]: row["count"] for row in cat_qs}

        latest_benchmark = None
        bench_run = LearningRun.objects.filter(benchmark_results__isnull=False).first()
        if bench_run:
            latest_benchmark = bench_run.benchmark_results

        data = {
            "enabled": settings_obj.enabled,
            "schedule_type": settings_obj.schedule_type,
            "schedule_hour": settings_obj.schedule_hour,
            "total_knowledge_items": KnowledgeItem.objects.count(),
            "active_items": KnowledgeItem.objects.filter(status=ItemStatus.ACTIVE).count(),
            "pending_items": KnowledgeItem.objects.filter(status=ItemStatus.PENDING).count(),
            "total_sources": KnowledgeSource.objects.filter(is_active=True).count(),
            "unread_notifications": LearningNotification.objects.filter(is_read=False).count(),
            "last_run": LearningRunSerializer(last_run).data if last_run else None,
            "category_breakdown": breakdown,
            "latest_benchmark": latest_benchmark,
        }
        return Response(data)


class RunListView(APIView):
    """GET /api/learning/runs/ — paginated run history."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = LearningRun.objects.all()[:50]
        return Response(LearningRunSerializer(qs, many=True).data)


class RunDetailView(APIView):
    """GET /api/learning/runs/<id>/"""
    permission_classes = [IsAuthenticated]

    def get(self, request, run_id):
        try:
            run = LearningRun.objects.get(id=run_id)
        except LearningRun.DoesNotExist:
            return Response({"error": "Not found"}, status=404)
        return Response(LearningRunSerializer(run).data)


class TriggerRunView(APIView):
    """POST /api/learning/runs/trigger/ — manually start a learning run."""
    permission_classes = [IsAdminUser]

    def post(self, request):
        from .tasks import run_manual_learning
        result = run_manual_learning.delay()
        return Response({"status": "queued", "task_id": result.id}, status=202)


class RollbackRunView(APIView):
    """POST /api/learning/rollback/<run_id>/ — deactivate all items from a run."""
    permission_classes = [IsAdminUser]

    def post(self, request, run_id):
        try:
            run = LearningRun.objects.get(id=run_id)
        except LearningRun.DoesNotExist:
            return Response({"error": "Not found"}, status=404)
        count = KnowledgeItem.objects.filter(learning_run=run).update(
            status=ItemStatus.REJECTED,
            rejection_reason=f"Rolled back by admin via API"
        )
        return Response({"rolled_back_items": count})


class KnowledgeItemListView(APIView):
    """GET /api/learning/items/ — browse knowledge items."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = KnowledgeItem.objects.exclude(status=ItemStatus.REJECTED)
        category = request.query_params.get("category")
        search = request.query_params.get("q")
        if category:
            qs = qs.filter(category=category)
        if search:
            qs = qs.filter(title__icontains=search) | qs.filter(summary__icontains=search)
        qs = qs[:100]
        return Response(KnowledgeItemListSerializer(qs, many=True).data)


class KnowledgeItemRejectView(APIView):
    """POST /api/learning/items/<id>/reject/"""
    permission_classes = [IsAdminUser]

    def post(self, request, item_id):
        try:
            item = KnowledgeItem.objects.get(id=item_id)
        except KnowledgeItem.DoesNotExist:
            return Response({"error": "Not found"}, status=404)
        reason = request.data.get("reason", "Admin rejected")
        item.status = ItemStatus.REJECTED
        item.rejection_reason = reason[:255]
        item.save(update_fields=["status", "rejection_reason"])
        return Response({"status": "rejected"})


class NotificationsView(APIView):
    """GET /api/learning/notifications/ — unread admin notifications."""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        qs = LearningNotification.objects.filter(is_read=False)[:30]
        return Response(LearningNotificationSerializer(qs, many=True).data)

    def patch(self, request):
        """Mark all as read."""
        LearningNotification.objects.filter(is_read=False).update(is_read=True)
        return Response({"status": "ok"})


class SettingsView(APIView):
    """GET/PATCH /api/learning/settings/"""
    permission_classes = [IsAdminUser]

    def get(self, request):
        obj = LearningSettings.get()
        return Response(LearningSettingsSerializer(obj).data)

    def patch(self, request):
        obj = LearningSettings.get()
        serializer = LearningSettingsSerializer(obj, data=request.data, partial=True)
        if serializer.is_valid():
            serializer.save()
            # Immediately reschedule Celery beat
            _reschedule_beat(serializer.validated_data)
            return Response(serializer.data)
        return Response(serializer.errors, status=400)


class TrainingUploadView(APIView):
    """POST /api/learning/training/upload/ — admin uploads prompt/answer example."""
    permission_classes = [IsAdminUser]

    def post(self, request):
        from ai.services.ollama_client import client, OllamaError

        serializer = AdminTrainingExampleSerializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=400)

        # Generate preview before saving
        prompt = request.data.get("prompt", "")
        answer = request.data.get("answer", "")
        preview = ""
        try:
            preview = client.chat(
                [{"role": "user", "content": f"Summarize this Q&A in 3 sentences max:\n\nQ: {prompt}\n\nA: {answer[:1000]}"}],
                temperature=0.1, stream=False,
            ) or ""
        except OllamaError:
            preview = (prompt[:80] + "…") if len(prompt) > 80 else prompt

        example = serializer.save(preview_summary=preview.strip()[:500])
        return Response({
            "id": str(example.id),
            "preview_summary": example.preview_summary,
            "category": example.category,
            "tags": example.tags,
            "message": "Preview generated. Call /approve/ to add to Knowledge Base.",
        }, status=201)


class TrainingApproveView(APIView):
    """POST /api/learning/training/<id>/approve/"""
    permission_classes = [IsAdminUser]

    def post(self, request, example_id):
        from django.utils import timezone
        from .tasks import process_admin_upload

        try:
            example = AdminTrainingExample.objects.get(id=example_id)
        except AdminTrainingExample.DoesNotExist:
            return Response({"error": "Not found"}, status=404)

        example.approved = True
        example.approved_at = timezone.now()
        example.save(update_fields=["approved", "approved_at"])
        process_admin_upload.delay(str(example.id))
        return Response({"status": "approved_and_queued"})


class BenchmarkView(APIView):
    """GET /api/learning/benchmark/latest/"""
    permission_classes = [IsAuthenticated]

    def get(self, request):
        run = LearningRun.objects.filter(benchmark_results__isnull=False).first()
        if not run:
            return Response({"message": "No benchmark results yet."})
        return Response({
            "results": run.benchmark_results,
            "previous": run.previous_benchmark,
            "regression": run.benchmark_regression,
            "run_date": run.started_at,
        })

    def post(self, request):
        """POST to trigger a fresh benchmark run."""
        from .tasks import run_coding_benchmark
        result = run_coding_benchmark.delay()
        return Response({"status": "queued", "task_id": result.id}, status=202)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _reschedule_beat(settings_data):
    """No-op placeholder — Celery beat reschedule would need app-level crontab update."""
    pass
