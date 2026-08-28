"""URL routing for the VISION Learning API."""
from django.urls import path
from . import views

urlpatterns = [
    path("dashboard/", views.DashboardView.as_view(), name="learning-dashboard"),
    path("runs/", views.RunListView.as_view(), name="learning-runs"),
    path("runs/trigger/", views.TriggerRunView.as_view(), name="learning-trigger"),
    path("runs/<uuid:run_id>/", views.RunDetailView.as_view(), name="learning-run-detail"),
    path("rollback/<uuid:run_id>/", views.RollbackRunView.as_view(), name="learning-rollback"),
    path("items/", views.KnowledgeItemListView.as_view(), name="learning-items"),
    path("items/<uuid:item_id>/reject/", views.KnowledgeItemRejectView.as_view(), name="learning-item-reject"),
    path("notifications/", views.NotificationsView.as_view(), name="learning-notifications"),
    path("settings/", views.SettingsView.as_view(), name="learning-settings"),
    path("training/upload/", views.TrainingUploadView.as_view(), name="learning-training-upload"),
    path("training/<uuid:example_id>/approve/", views.TrainingApproveView.as_view(), name="learning-training-approve"),
    path("benchmark/latest/", views.BenchmarkView.as_view(), name="learning-benchmark"),
]
