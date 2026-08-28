"""
URL configuration for config project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.2/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.http import JsonResponse
from django.urls import path, include
from rest_framework_simplejwt.views import (
    TokenObtainPairView,
    TokenRefreshView,
)
from django.conf import settings as _dj_settings
from ai.services.ollama_client import client as _ollama_client


def health_view(request):
    return JsonResponse({"status": "ok", "service": "VISION Backend"})


def ai_health_alias(request):
    """GET /api/health/ai — alias that actually tests the AI provider."""
    try:
        h = _ollama_client.healthCheck()
        ok = h.get("ollama", {}).get("connected") and h.get("textModel", {}).get("installed")
        provider = h.get("ollama", {}).get("baseUrl", "unknown")
        # normalize provider name
        if "groq" in str(provider).lower():
            prov_name = "groq"
        elif "openai" in str(provider).lower():
            prov_name = "openai"
        else:
            prov_name = "ollama"
        model = h.get("textModel", {}).get("name", "")
        if ok:
            return JsonResponse({"status": "ok", "provider": prov_name, "model": model, "reachable": True})
        err = h.get("ollama", {}).get("error") or "AI service unreachable"
        return JsonResponse({"status": "error", "provider": prov_name, "model": model, "reachable": False, "error": err}, status=503)
    except Exception as e:
        return JsonResponse({"status": "error", "provider": "unknown", "model": "", "reachable": False, "error": str(e)[:300]}, status=503)


urlpatterns = [
    path('health', health_view, name='health'),
    path('health/', health_view, name='health-slash'),
    path('api/health/ai', ai_health_alias, name='ai-health-alias'),
    path('api/health/ai/', ai_health_alias, name='ai-health-alias-slash'),
    path('admin/', admin.site.urls),
    
    # Auth Endpoints
    path('api/auth/login/', TokenObtainPairView.as_view(), name='token_obtain_pair'),
    path('api/auth/refresh/', TokenRefreshView.as_view(), name='token_refresh'),
    path('api/auth/', include('users.urls')),
    
    # VISION AI — simplified
    path('api/ai/', include('ai_agent.urls')),
    path('api/conversations/', include('ai_agent.conversations_urls')),
    path('api/memory/', include('ai_agent.memory_urls')),
    path('api/workspace/', include('ai_agent.workspace_urls')),
    path('api/', include('ai_agent.attachments_urls')),
    
    # Continuous Learning
    path('api/learning/', include('learning.urls')),
]

from django.conf import settings
from django.conf.urls.static import static
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
