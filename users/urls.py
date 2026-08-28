from django.urls import path
from .views import (
    RegisterView, MeView, ProfileView, AvatarUploadView,
    SettingsView, ChangePasswordView, DataStatsView,
    ExportDataView, DeleteAccountView, ClearHistoryView,
)

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('me/', MeView.as_view(), name='me'),
    path('profile/', ProfileView.as_view(), name='profile'),
    path('avatar/', AvatarUploadView.as_view(), name='avatar'),
    path('settings/', SettingsView.as_view(), name='settings'),
    path('change-password/', ChangePasswordView.as_view(), name='change-password'),
    path('data/stats/', DataStatsView.as_view(), name='data-stats'),
    path('data/export/', ExportDataView.as_view(), name='data-export'),
    path('data/clear-history/', ClearHistoryView.as_view(), name='clear-history'),
    path('delete-account/', DeleteAccountView.as_view(), name='delete-account'),
]
