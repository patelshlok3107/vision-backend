from django.urls import path
from .views import (
    AIChatView,
    ai_chat_async_view,
    AIHealthView,
    AISettingsView,
    AIUsageView,
    AIVoiceView,
    AIPerformanceView,
    AIVisionTestView,
    AIRouterView,
)

from django.http import StreamingHttpResponse
import time as _t_debug
import asyncio

async def _debug_stream(request):
    async def gen():
        for i in range(5):
            yield f'{{"type":"token","content":"chunk{i} "}}\n'
            await asyncio.sleep(0.5)
        yield '{"type":"done"}\n'
    resp = StreamingHttpResponse(gen(), content_type='application/x-ndjson')
    resp['Cache-Control'] = 'no-cache'
    resp['X-Accel-Buffering'] = 'no'
    return resp

urlpatterns = [
    path('chat/', ai_chat_async_view, name='ai-chat'),
    path('chat-sync/', AIChatView.as_view(), name='ai-chat-sync'),
    path('debug-stream/', _debug_stream, name='debug-stream'),
    path('health/', AIHealthView.as_view(), name='ai-health'),
    path('settings/', AISettingsView.as_view(), name='ai-settings'),
    path('usage/', AIUsageView.as_view(), name='ai-usage'),
    path('tts/', AIVoiceView.as_view(), name='ai-tts'),
    path('performance/', AIPerformanceView.as_view(), name='ai-performance'),
    path('vision/test/', AIVisionTestView.as_view(), name='vision-test'),
    path('router/', AIRouterView.as_view(), name='ai-router'),
]
