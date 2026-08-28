from django.urls import path
from .conversations_views import (
    ConversationListCreateView,
    ConversationDetailViewV2,
    ConversationArchiveView,
    ConversationRenameView,
    MessageListCreateView,
    ConversationSearchView,
)

urlpatterns = [
    path('', ConversationListCreateView.as_view(), name='conv-list-create'),
    path('search/', ConversationSearchView.as_view(), name='conv-search'),
    path('<uuid:conversation_id>/', ConversationDetailViewV2.as_view(), name='conv-detail'),
    path('<uuid:conversation_id>/messages/', MessageListCreateView.as_view(), name='conv-messages'),
    path('<uuid:conversation_id>/archive/', ConversationArchiveView.as_view(), name='conv-archive'),
    path('<uuid:conversation_id>/rename/', ConversationRenameView.as_view(), name='conv-rename'),
]
