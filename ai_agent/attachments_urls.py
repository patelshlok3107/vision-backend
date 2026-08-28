from django.urls import path
from .attachments_views import AttachmentUploadView, AttachmentServeView, AttachmentListView

urlpatterns = [
    path('conversations/<uuid:conversation_id>/attachments/', AttachmentUploadView.as_view(), name='attachment-upload'),
    path('conversations/<uuid:conversation_id>/attachments/list/', AttachmentListView.as_view(), name='attachment-list'),
    path('attachments/<uuid:attachment_id>/', AttachmentServeView.as_view(), name='attachment-serve'),
]
