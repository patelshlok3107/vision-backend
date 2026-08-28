from django.urls import path
from .memory_views import MemoryListCreateView, MemoryDetailView, MemoryClearView, MemoryRetrieveView

urlpatterns = [
    path('', MemoryListCreateView.as_view(), name='memory-list-create'),
    path('retrieve/', MemoryRetrieveView.as_view(), name='memory-retrieve'),
    path('clear/', MemoryClearView.as_view(), name='memory-clear'),
    path('<uuid:memory_id>/', MemoryDetailView.as_view(), name='memory-detail'),
]
