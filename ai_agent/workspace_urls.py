from django.urls import path
from .workspace_views import WorkspaceListView, WorkspaceReadView, WorkspaceWriteView, WorkspaceDeleteView

urlpatterns = [
    path('list/', WorkspaceListView.as_view(), name='workspace-list'),
    path('read/', WorkspaceReadView.as_view(), name='workspace-read'),
    path('write/', WorkspaceWriteView.as_view(), name='workspace-write'),
    path('delete/', WorkspaceDeleteView.as_view(), name='workspace-delete'),
]
