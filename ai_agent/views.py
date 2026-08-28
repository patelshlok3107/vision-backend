"""
Updated ai_agent views:
- SemanticSearchView: real Ollama embeddings
- AIChatView: delegates to VisionAgent
- AIHealthView: Ollama + model health check
- AISettingsView: returns non-sensitive AI config
- AIUsageView: local inference usage stats
"""

import json
import logging
import traceback
import asyncio
from asgiref.sync import sync_to_async
from django.http import StreamingHttpResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated, AllowAny
from django.conf import settings
from django.db.models import Count, Avg, Sum, Q

from .models import Conversation, Message, AIUsageLog
from ai.services.ollama_client import client as ollama_client
from ai.services.agent import VisionAgent

logger = logging.getLogger(__name__)


class AIChatView(APIView):
    permission_classes = [AllowAny]

    def post(self, request):
        import time as _t
        import logging as _log
        _t0 = _t.perf_counter()
        _logger = _log.getLogger("vision.chat")
        # Request tracing — every request gets a unique ID for end-to-end tracing
        _req_id = request.data.get('request_id') if hasattr(request, 'data') and isinstance(request.data, dict) else None
        if not _req_id:
            _req_id = request.headers.get('X-Request-ID', '') or f"req_{int(_t0*1000)}"
        _logger.info("[CHAT:%s] ▶ Request received: content_type=%s", _req_id, request.content_type)
        # Support both JSON and multipart (for images)
        if request.content_type and 'multipart' in request.content_type:
            message = (request.data.get('message') or request.data.get('content') or '').strip()
            conversation_id = request.data.get('conversation_id')
            attachment_ids = request.data.getlist('attachment_ids') or request.data.getlist('attachments')
            if not attachment_ids and request.data.get('attachment_ids'):
                attachment_ids = [request.data.get('attachment_ids')]
            files = request.FILES.getlist('images') or request.FILES.getlist('file') or []
            if 'image' in request.FILES:
                files = [request.FILES['image']]
        else:
            message = (request.data.get('message') or request.data.get('content') or '').strip()
            conversation_id = request.data.get('conversation_id')
            attachment_ids = request.data.get('attachment_ids') or request.data.get('attachments') or []
            if isinstance(attachment_ids, str):
                attachment_ids = [attachment_ids]
            files = []
        # Phase 1: mode + memory toggle (works for both JSON and multipart)
        try:
            mode = (request.data.get('mode') or request.data.get('think_mode') or "").strip().lower()
        except:
            mode = ""
        memory_enabled_raw = request.data.get('memory_enabled', request.data.get('memory', True))
        if isinstance(memory_enabled_raw, str):
            memory_enabled = memory_enabled_raw.lower() not in ("0", "false", "off", "no")
        else:
            memory_enabled = bool(memory_enabled_raw) if memory_enabled_raw is not None else True
        _logger.info("[CHAT] Conversation ID: %s", conversation_id)
        _logger.info("[CHAT] Mode=%s memory_enabled=%s", mode, memory_enabled)
        _logger.info("[CHAT] Image detected: %s (files=%d, ids=%s)", bool(files or attachment_ids), len(files), attachment_ids)
        _logger.info("[CHAT] User prompt: %s", (message or "")[:200])
        for f in files:
            _logger.info("[CHAT] Image MIME type: %s name=%s size=%d", getattr(f, 'content_type', 'unknown'), getattr(f, 'name', 'unknown'), getattr(f, 'size', 0))

        # allow image-only messages
        if not message and not files and not attachment_ids:
            return Response({'error': 'Message or image required'}, status=400)
        if not message:
            message = "What's in this image?"  # default for image-only per spec §16

        # Reject external AI if LOCAL_AI_ONLY is set
        if getattr(settings, 'LOCAL_AI_ONLY', True) and getattr(settings, 'AI_PROVIDER', 'ollama') != 'ollama':
            return Response(
                {'error': 'External AI providers are disabled. LOCAL_AI_ONLY is active.'},
                status=403
            )

        # Determine guest vs authenticated for sync view
        is_guest = not request.user or not request.user.is_authenticated
        if is_guest:
            # Check Authorization header manually for JWT if DRF didn't authenticate
            auth_header = request.headers.get('Authorization', '')
            if auth_header.startswith('Bearer '):
                # token present but invalid - reject
                from rest_framework_simplejwt.authentication import JWTAuthentication as _J
                try:
                    _a = _J()
                    _res = _a.authenticate(request)
                    if _res:
                        is_guest = False
                    else:
                        # invalid token treated as guest? but header present -> error
                        pass
                except Exception:
                    pass
            if is_guest:
                if len(message) > 4000:
                    return Response({'detail': 'Guest message too long. Please sign in.'}, status=413)
                if mode in ('agent','think'):
                    mode = 'auto'
                memory_enabled = False
                # Guest direct streaming without DB
                def _guest_sync_stream():
                    import json as _j
                    import base64 as _b64
                    from ai.services.prompts import SIMPLE_CHAT_SYSTEM_PROMPT, VISION_SYSTEM_PROMPT
                    from django.utils import timezone as _tz
                    b64s = []
                    # process files for guest (files already parsed)
                    for f in files[:3]:
                        from .attachments_views import validate_image, process_image
                        err = validate_image(f)
                        if err:
                            yield _j.dumps({"type":"error","content": err}) + "\n"
                            yield _j.dumps({"type":"done","conversation_id": None}) + "\n"
                            return
                        try:
                            buf, w, h = process_image(f)
                            buf.seek(0)
                            b64s.append(_b64.b64encode(buf.read()).decode('utf-8'))
                        except Exception as ve:
                            yield _j.dumps({"type":"error","content": str(ve)}) + "\n"
                            yield _j.dumps({"type":"done","conversation_id": None}) + "\n"
                            return
                    has_img = len(b64s) > 0
                    if has_img:
                        ok, err_msg = ollama_client.validate_vision_model()
                        if not ok:
                            yield _j.dumps({"type":"error","content": err_msg}) + "\n"
                            yield _j.dumps({"type":"done","conversation_id": None}) + "\n"
                            return
                        system_prompt = VISION_SYSTEM_PROMPT.format(today=_tz.now().isoformat())
                    else:
                        system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT
                    msgs = [{"role":"system","content": system_prompt}]
                    # guest_history support
                    gh = request.data.get('guest_history') or []
                    if isinstance(gh, list):
                        for m in gh[-6:]:
                            if isinstance(m, dict) and m.get('role') in ('user','assistant'):
                                msgs.append({"role": m['role'], "content": str(m['content'])[:1500]})
                    entry = {"role":"user","content": message or "What's in this image?"}
                    if b64s:
                        entry["images"] = b64s
                    msgs.append(entry)
                    yield _j.dumps({"type":"stream_start","content":{"path":"guest","mode":"guest"}}) + "\n"
                    yield _j.dumps({"type":"status","content":"VISION is thinking..."}) + "\n"
                    try:
                        resp_stream = ollama_client.chat(msgs, temperature=0.4, stream=True, num_predict=512, num_ctx=2048, is_vision=has_img)
                        for line in resp_stream.iter_lines(decode_unicode=True):
                            if not line: continue
                            try: chunk = _j.loads(line)
                            except: continue
                            token = chunk.get("message",{}).get("content","")
                            if token:
                                yield _j.dumps({"type":"token","content": token}) + "\n"
                        import uuid as _uu
                        yield _j.dumps({"type":"done","conversation_id": f"guest_{_uu.uuid4().hex[:8]}"}) + "\n"
                    except Exception as exc:
                        yield _j.dumps({"type":"error","content": str(exc)[:500]}) + "\n"
                resp = StreamingHttpResponse(_guest_sync_stream(), content_type='application/x-ndjson')
                resp['Cache-Control'] = 'no-cache, no-transform'
                resp['X-Accel-Buffering'] = 'no'
                resp['X-Request-ID'] = _req_id
                return resp

        # Resolve or create conversation — CRITICAL PATH: keep this under 15ms
        import time as _t2
        _t_conv_start = _t2.perf_counter()
        conversation = None
        if conversation_id:
            try:
                # No select_related needed — we already have request.user; only need conversation fields
                conversation = Conversation.objects.only('id','title','conversation_summary','user_id','updated_at').get(id=conversation_id, user=request.user)
            except Conversation.DoesNotExist:
                pass

        if not conversation:
            _logger.info("[CHAT:%s] Creating new conversation", _req_id)
            conversation = Conversation.objects.create(
                user=request.user,
                title=Conversation.generate_title(message or "Image analysis")
            )
        else:
            _logger.info("[CHAT:%s] Using existing conversation %s", _req_id, conversation.id)
        _t_conv_ms = int((_t2.perf_counter() - _t_conv_start)*1000)
        _t_recv_ms = int((_t2.perf_counter() - _t0)*1000)
        _logger.info("[PERF:%s] Backend received: %dms | Conversation ready: %dms", _req_id, _t_recv_ms, _t_conv_ms)

        # If direct files were sent with this chat request, create attachments now
        if files:
            from .attachments_views import validate_image, process_image
            from .models import Attachment
            for f in files[:5]:
                _logger.info("[CHAT] Validating image %s", f.name)
                err = validate_image(f)
                if err:
                    _logger.warning("[CHAT] Image validation failed: %s", err)
                    return Response({"error": err}, status=400)
                try:
                    _logger.info("[CHAT] Processing image %s", f.name)
                    buf, w, h = process_image(f)
                    _logger.info("[CHAT] Image processed w=%d h=%d", w, h)
                except ValueError as ve:
                    _logger.warning("[CHAT] Image processing failed: %s", ve)
                    return Response({"error": str(ve)}, status=400)
                att = Attachment.objects.create(conversation=conversation, user=request.user, file_name=f.name, mime_type=f.content_type, file_size=f.size, width=w, height=h)
                att.file.save(f.name, buf, save=True)
                _logger.info("[CHAT] Image stored id=%s", att.id)
                attachment_ids.append(str(att.id))
        _logger.info("[CHAT:%s] Final attachment_ids=%s", _req_id, attachment_ids)
        _t_ctx_ready = int((_t.perf_counter() - _t0)*1000)
        _logger.info("[PERF:%s] Request started: 0ms | Backend received: %dms | Context ready: %dms | Ollama request: %dms", _req_id, _t_recv_ms, _t_ctx_ready, _t_ctx_ready)

        agent = VisionAgent(user=request.user)
        try:
            _logger.info("[CHAT:%s] Starting agent stream mode=%s memory=%s", _req_id, mode, memory_enabled)
            stream_gen = agent.chat_stream(message, conversation=conversation, attachment_ids=attachment_ids, mode=mode, memory_enabled=memory_enabled, request_id=_req_id, t0=_t0)
            resp = StreamingHttpResponse(stream_gen, content_type='application/x-ndjson')
            # Critical: prevent any proxy or middleware from buffering the stream
            resp['Cache-Control'] = 'no-cache, no-transform'
            resp['X-Accel-Buffering'] = 'no'
            resp['X-Request-ID'] = _req_id
            return resp
        except Exception as exc:
            tb = traceback.format_exc()
            logger.error("AIChatView unhandled error: %s\n%s", exc, tb)
            return Response({'error': str(exc), 'traceback': tb}, status=500)


# ── ASYNC STREAMING VIEW (fixes Daphne buffering: sync StreamingHttpResponse buffers, async streams per-chunk) ──
from django.views.decorators.csrf import csrf_exempt
import json as _json

@csrf_exempt
async def ai_chat_async_view(request):
    """
    Async streaming chat endpoint for Daphne/ASGI.
    Uses async generator + run_in_executor to yield per-token without buffering.
    Handles JWT auth via sync_to_async.
    """
    import time as _t
    import logging as _log
    _t0 = _t.perf_counter()
    _logger = _log.getLogger("vision.chat.async")
    if request.method != 'POST':
        from django.http import JsonResponse
        return JsonResponse({'error': 'Method not allowed'}, status=405)

    # Parse body
    try:
        if request.content_type and 'multipart' in request.content_type:
            # For multipart, need to read via sync_to_async
            def _parse_multipart():
                import json as _j
                # Django's request.POST and FILES are sync
                data = {}
                data['message'] = (request.POST.get('message') or request.POST.get('content') or '').strip()
                data['conversation_id'] = request.POST.get('conversation_id')
                data['mode'] = (request.POST.get('mode') or request.POST.get('think_mode') or '').strip().lower()
                data['memory_enabled'] = request.POST.get('memory_enabled', 'true')
                data['attachment_ids'] = request.POST.getlist('attachment_ids') or request.POST.getlist('attachments')
                data['request_id'] = request.POST.get('request_id')
                # guest_history for ephemeral guest sessions (JSON string)
                gh_raw = request.POST.get('guest_history') or request.POST.get('history') or ""
                if gh_raw:
                    try:
                        data['guest_history'] = _j.loads(gh_raw) if isinstance(gh_raw, str) else gh_raw
                    except:
                        data['guest_history'] = []
                else:
                    data['guest_history'] = []
                files = request.FILES.getlist('images') or request.FILES.getlist('file') or []
                if 'image' in request.FILES:
                    files = [request.FILES['image']]
                return data, files
            _data, files = await sync_to_async(_parse_multipart, thread_sensitive=True)()
            message = _data.get('message', '')
            conversation_id = _data.get('conversation_id')
            mode = _data.get('mode', '')
            memory_enabled_raw = _data.get('memory_enabled', True)
            attachment_ids = _data.get('attachment_ids') or []
            request_id = _data.get('request_id')
        else:
            body = await sync_to_async(lambda: request.body)()
            _data = _json.loads(body.decode() or '{}') if body else {}
            message = (_data.get('message') or _data.get('content') or '').strip()
            conversation_id = _data.get('conversation_id')
            mode = (_data.get('mode') or _data.get('think_mode') or '').strip().lower()
            memory_enabled_raw = _data.get('memory_enabled', _data.get('memory', True))
            attachment_ids = _data.get('attachment_ids') or _data.get('attachments') or []
            if isinstance(attachment_ids, str):
                attachment_ids = [attachment_ids]
            files = []
            request_id = _data.get('request_id')
    except Exception as _e:
        from django.http import JsonResponse
        return JsonResponse({'error': f'Invalid request: {_e}'}, status=400)

    if isinstance(memory_enabled_raw, str):
        memory_enabled = memory_enabled_raw.lower() not in ("0", "false", "off", "no")
    else:
        memory_enabled = bool(memory_enabled_raw) if memory_enabled_raw is not None else True

    _req_id = request_id or request.headers.get('X-Request-ID', '') or f"req_{int(_t0*1000)}"
    _t_after_parse = _t.perf_counter()
    _logger.info("[CHAT-ASYNC:%s] ▶ Request received (parse %.0fms)", _req_id, (_t_after_parse-_t0)*1000)

    if not message and not files and not attachment_ids:
        from django.http import JsonResponse
        return JsonResponse({'error': 'Message or image required'}, status=400)
    if not message:
        message = "What's in this image?"

    if getattr(settings, 'LOCAL_AI_ONLY', True) and getattr(settings, 'AI_PROVIDER', 'ollama') != 'ollama':
        from django.http import JsonResponse
        return JsonResponse({'error': 'External AI providers are disabled.'}, status=403)

    # ── Auth (sync DB via sync_to_async) - guest allowed ──
    from rest_framework_simplejwt.authentication import JWTAuthentication
    _auth = JWTAuthentication()
    _user = None
    _is_guest = False
    try:
        _auth_result = await sync_to_async(_auth.authenticate)(request)
    except Exception as _ae:
        # Treat auth error as guest if no valid token, else error
        _auth_result = None
        auth_header = request.headers.get('Authorization', '')
        if auth_header.startswith('Bearer ') and len(auth_header) > 10:
            from django.http import JsonResponse
            return JsonResponse({'detail': f'Authentication failed: {_ae}'}, status=401)
    if not _auth_result:
        _is_guest = True
        _user = None
        # Guest rate limiting - simple in-memory check (20 per minute per IP)
        try:
            import time as _rt
            from collections import defaultdict
            if not hasattr(ai_chat_async_view, '_guest_hits'):
                ai_chat_async_view._guest_hits = defaultdict(list)
            _ip = request.META.get('REMOTE_ADDR', 'unknown')
            now = _rt.time()
            hits = ai_chat_async_view._guest_hits[_ip]
            hits[:] = [t for t in hits if now - t < 60]
            if len(hits) >= 20:
                from django.http import JsonResponse
                return JsonResponse({'detail': 'Guest rate limit exceeded. Please sign in for unlimited access.'}, status=429)
            hits.append(now)
            # Guest: enforce basic mode only, no memory, limit message size
            if len(message) > 4000:
                from django.http import JsonResponse
                return JsonResponse({'detail': 'Guest message too long. Please sign in for longer conversations.'}, status=413)
            # Guest cannot use agent/think heavy modes - downgrade to auto
            if mode in ('agent', 'think'):
                mode = 'auto'
            memory_enabled = False
        except Exception:
            pass
    else:
        _user, _validated_token = _auth_result

    if _is_guest:
        # ── GUEST STREAMING PATH - no DB, direct Ollama ──
        # Handle image files for guest (base64 without DB persistence)
        import base64 as _b64
        guest_b64s = []
        if files:
            from .attachments_views import validate_image, process_image
            for f in files[:3]:  # limit guest to 3 images
                err = await sync_to_async(validate_image)(f)
                if err:
                    from django.http import JsonResponse
                    return JsonResponse({"error": err}, status=400)
                try:
                    buf, w, h = await sync_to_async(process_image)(f)
                    # read buf to base64
                    def _read_buf():
                        buf.seek(0)
                        return _b64.b64encode(buf.read()).decode('utf-8')
                    b64 = await sync_to_async(_read_buf)()
                    guest_b64s.append(b64)
                except ValueError as ve:
                    from django.http import JsonResponse
                    return JsonResponse({"error": str(ve)}, status=400)
        # Also handle attachment_ids for guest? ignore (guest has no DB attachments)
        has_image = len(guest_b64s) > 0
        if has_image:
            ok, err_msg = ollama_client.validate_vision_model()
            if not ok:
                async def _guest_err():
                    yield _json.dumps({"type": "error", "content": err_msg}) + "\n"
                    yield _json.dumps({"type": "done", "conversation_id": None}) + "\n"
                resp = StreamingHttpResponse(_guest_err(), content_type='application/x-ndjson')
                resp['Cache-Control'] = 'no-cache, no-transform'
                resp['X-Accel-Buffering'] = 'no'
                resp['X-Request-ID'] = _req_id
                return resp

        # Guest history support - accept guest_history from request if provided
        guest_history = _data.get('guest_history') or _data.get('history') or []
        # Build simple guest messages
        from ai.services.prompts import SIMPLE_CHAT_SYSTEM_PROMPT, VISION_SYSTEM_PROMPT
        from django.utils import timezone as _tz
        # Use simple prompt for guest (fast), unless image
        if has_image:
            now = _tz.now()
            system_prompt = VISION_SYSTEM_PROMPT.format(today=now.isoformat())
        else:
            system_prompt = SIMPLE_CHAT_SYSTEM_PROMPT
        messages = [{"role": "system", "content": system_prompt}]
        # Include limited guest history (last 6 messages, truncate)
        if isinstance(guest_history, list):
            for m in guest_history[-6:]:
                if isinstance(m, dict) and m.get('role') in ('user','assistant') and m.get('content'):
                    c = str(m['content'])[:1500]
                    messages.append({"role": m['role'], "content": c})
        user_entry = {"role": "user", "content": message if message.strip() else "What's in this image?"}
        if guest_b64s:
            user_entry["images"] = guest_b64s
        messages.append(user_entry)

        # Guest uses restricted model params
        from django.conf import settings as _gs
        guest_num_predict = 512 if not has_image else 1024
        guest_num_ctx = 2048 if not has_image else 4096

        async def _guest_stream():
            # stream_start
            yield _json.dumps({"type": "stream_start", "content": {"path": "guest", "mode": "guest"}}) + "\n"
            yield _json.dumps({"type": "status", "content": "VISION is thinking..."}) + "\n"
            try:
                # Run Ollama streaming in thread
                loop = asyncio.get_event_loop()
                def _open_stream():
                    return ollama_client.chat(messages, temperature=0.4, stream=True, num_predict=guest_num_predict, num_ctx=guest_num_ctx, is_vision=has_image)
                resp_stream = await loop.run_in_executor(None, _open_stream)
                first = True
                full = []
                for line in resp_stream.iter_lines(decode_unicode=True):
                    if not line:
                        continue
                    try:
                        chunk = _json.loads(line)
                    except Exception:
                        continue
                    token = chunk.get("message", {}).get("content", "")
                    if not token:
                        continue
                    if first:
                        first = False
                    full.append(token)
                    yield _json.dumps({"type": "token", "content": token}) + "\n"
                    await asyncio.sleep(0)
                # done with guest id
                import uuid as _uuid
                guest_conv_id = f"guest_{_uuid.uuid4().hex[:8]}"
                yield _json.dumps({"type": "done", "conversation_id": guest_conv_id}) + "\n"
            except Exception as exc:
                err = str(exc)
                if "connect" in err.lower() or "ollama" in err.lower():
                    yield _json.dumps({"type": "error", "content": "VISION is temporarily unavailable. Please try again."}) + "\n"
                else:
                    yield _json.dumps({"type": "error", "content": err[:500]}) + "\n"

        resp = StreamingHttpResponse(_guest_stream(), content_type='application/x-ndjson')
        resp['Cache-Control'] = 'no-cache, no-transform'
        resp['X-Accel-Buffering'] = 'no'
        resp['X-Request-ID'] = _req_id
        return resp

    # ── AUTHENTICATED PATH ──
    _t_conv_start = _t.perf_counter()
    conversation = None
    if conversation_id:
        try:
            conversation = await sync_to_async(lambda: Conversation.objects.only('id','title','conversation_summary','user_id','updated_at').get(id=conversation_id, user=_user))()
        except Conversation.DoesNotExist:
            pass
        except Exception:
            pass
    if not conversation:
        conversation = await sync_to_async(lambda: Conversation.objects.create(user=_user, title=Conversation.generate_title(message or "Image analysis")))()
    _t_conv_ms = int((_t.perf_counter() - _t_conv_start)*1000)
    _t_recv_ms = int((_t.perf_counter() - _t0)*1000)
    _logger.info("[PERF-ASYNC:%s] Backend received: %dms | Conversation ready: %dms | Auth+Parse: %.0fms", _req_id, _t_recv_ms, _t_conv_ms, (_t_conv_start-_t0)*1000)

    # ── Files (if any) ──
    if files:
        from .attachments_views import validate_image, process_image
        from .models import Attachment
        for f in files[:5]:
            err = await sync_to_async(validate_image)(f)
            if err:
                from django.http import JsonResponse
                return JsonResponse({"error": err}, status=400)
            try:
                buf, w, h = await sync_to_async(process_image)(f)
            except ValueError as ve:
                from django.http import JsonResponse
                return JsonResponse({"error": str(ve)}, status=400)
            def _create_att():
                att = Attachment.objects.create(conversation=conversation, user=_user, file_name=f.name, mime_type=f.content_type, file_size=f.size, width=w, height=h)
                att.file.save(f.name, buf, save=True)
                return att
            att = await sync_to_async(_create_att, thread_sensitive=True)()
            attachment_ids.append(str(att.id))

    _t_ctx_ready = int((_t.perf_counter() - _t0)*1000)
    print(f"[ASYNC] context ready {_t_ctx_ready}ms", flush=True)
    _logger.info("[PERF-ASYNC:%s] Context ready: %dms | Ollama request: %dms", _req_id, _t_ctx_ready, _t_ctx_ready)

    # ── Stream via async generator that wraps sync VisionAgent.chat_stream ──
    agent = VisionAgent(user=_user)

    async def _async_stream():
        sync_gen = agent.chat_stream(message, conversation=conversation, attachment_ids=attachment_ids, mode=mode, memory_enabled=memory_enabled, request_id=_req_id, t0=_t0)
        loop = asyncio.get_event_loop()
        def _next():
            try:
                return next(sync_gen)
            except StopIteration:
                return None
        while True:
            chunk = await loop.run_in_executor(None, _next)
            if chunk is None:
                break
            yield chunk
            await asyncio.sleep(0)

    resp = StreamingHttpResponse(_async_stream(), content_type='application/x-ndjson')
    resp['Cache-Control'] = 'no-cache, no-transform'
    resp['X-Accel-Buffering'] = 'no'
    resp['X-Request-ID'] = _req_id
    return resp


class AIHealthView(APIView):
    permission_classes = [AllowAny]

    def get(self, request):
        import logging as _lg
        _logger = _lg.getLogger("vision.health")
        _logger.info("[HEALTH] AI health check: provider=%s base=%s", getattr(settings, 'AI_PROVIDER', ''), getattr(settings, 'OLLAMA_BASE_URL', ''))
        health = ollama_client.healthCheck()
        legacy_ok = health["ollama"]["connected"] and health["textModel"]["installed"]
        status_code = 200 if legacy_ok else 503
        health["status"] = "healthy" if legacy_ok else "unhealthy"
        # Normalize provider for response
        base = str(health.get("ollama", {}).get("baseUrl", "")).lower()
        if "groq" in base or getattr(settings, 'AI_PROVIDER', '').lower() == "groq":
            health["provider"] = "groq"
        elif "openai" in base or getattr(settings, 'AI_PROVIDER', '').lower() == "openai":
            health["provider"] = "openai"
        else:
            health["provider"] = "ollama"
        # Add spec-compliant top-level fields
        health["reachable"] = bool(health["ollama"]["connected"])
        health["model"] = health.get("textModel", {}).get("name", "")
        if not legacy_ok:
            _logger.warning("[AI ERROR] Provider=%s reachable=%s error=%s", health["provider"], health["reachable"], health.get("ollama", {}).get("error", "unknown"))
        return Response(health, status=status_code)


class AIVisionTestView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        # Test vision model with a tiny 1x1 png
        from ai.services.ollama_client import client as oc
        health = oc.healthCheck()
        if not health["ollama"]["connected"]:
            return Response({"success": False, "error": f"VISION can't connect to Ollama. Make sure Ollama is running at {health['ollama']['baseUrl']}."})
        vm = health["visionModel"]
        if not vm["configured"]:
            return Response({"success": False, "error": "No vision-capable Ollama model is configured. Configure OLLAMA_VISION_MODEL in your environment settings. Example: OLLAMA_VISION_MODEL=llava"})
        if not vm["installed"]:
            return Response({"success": False, "error": f"The configured VISION model {vm['name']} isn't installed. Run ollama pull {vm['name']} and try again."})
        if not vm["capable"]:
            return Response({"success": False, "error": f"The configured model {vm['name']} doesn't support image analysis. Select a vision-capable Ollama model."})
        # Try a minimal vision request
        try:
            import base64
            # 1x1 red png
            tiny = base64.b64encode(b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x01\x01\x01\x00\x18\xdd\x8d\xb0\x00\x00\x00\x00IEND\xaeB`\x82').decode()
            msgs = [{"role": "user", "content": "What do you see?", "images": [tiny]}]
            resp = oc.chat(msgs, is_vision=True)
            return Response({"success": True, "response": resp[:500]})
        except Exception as exc:
            return Response({"success": False, "error": str(exc)})


class AISettingsView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        health = ollama_client.healthCheck()
        return Response({
            'provider': settings.AI_PROVIDER,
            'local_only': settings.LOCAL_AI_ONLY,
            'ollama_url': settings.OLLAMA_BASE_URL,
            'model': getattr(settings, 'OLLAMA_TEXT_MODEL', getattr(settings, 'OLLAMA_MODEL', '')),
            'text_model': health["textModel"]["name"],
            'embedding_model': settings.OLLAMA_EMBEDDING_MODEL,
            'embedding_dim': settings.OLLAMA_EMBEDDING_DIM,
            'vision_model': health["visionModel"]["name"],
            'vision_enabled': getattr(settings, 'OLLAMA_VISION_ENABLED', True),
            'vision_available': health["visionModel"]["installed"] and health["visionModel"]["capable"],
            'vision_installed': health["visionModel"]["installed"],
            'vision_capable': health["visionModel"]["capable"],
            'vision_configured': health["visionModel"]["configured"],
            'health': health,
        })


class AIUsageView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        logs = AIUsageLog.objects.filter(user=request.user)
        total = logs.count()
        successful = logs.filter(success=True).count()
        failed = logs.filter(success=False).count()
        avg_latency = logs.aggregate(avg=Avg('latency_ms'))['avg'] or 0
        avg_ttft = logs.exclude(ttft_ms__isnull=True).aggregate(avg=Avg('ttft_ms'))['avg'] or 0
        tool_calls = logs.filter(request_type='chat_with_tool').count()
        embeddings = logs.filter(request_type='embedding').count()

        return Response({
            'model': settings.OLLAMA_MODEL,
            'provider': 'ollama',
            'local_inference': True,
            'external_api_usage': 'None — Local inference only',
            'total_requests': total,
            'successful': successful,
            'failed': failed,
            'average_latency_ms': round(avg_latency, 1),
            'average_ttft_ms': round(avg_ttft, 1),
            'tool_calls': tool_calls,
            'embedding_operations': embeddings,
        })


class ConversationListView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        conversations = Conversation.objects.filter(user=request.user)[:20]
        data = [
            {'id': str(c.id), 'title': c.title, 'updated_at': c.updated_at.isoformat()}
            for c in conversations
        ]
        return Response({'conversations': data})


class ConversationDetailView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, conversation_id):
        try:
            conversation = Conversation.objects.get(id=conversation_id, user=request.user)
        except Conversation.DoesNotExist:
            return Response({'error': 'Not found'}, status=404)

        messages = Message.objects.filter(conversation=conversation).exclude(role='tool')
        data = [
            {'role': m.role, 'content': m.content, 'timestamp': m.timestamp.isoformat()}
            for m in messages
        ]
        return Response({
            'id': str(conversation.id),
            'title': conversation.title,
            'messages': data,
        })

class AIVoiceView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request):
        text = request.data.get('text', '')
        if not text:
            return Response({'error': 'No text provided'}, status=400)
            
        from ai.services.tts import generate_voice_file
        try:
            url_path = generate_voice_file(text)
            return Response({
                'text': text,
                'url': request.build_absolute_uri(url_path)
            })
        except Exception as e:
            return Response({'error': str(e)}, status=500)


class AIPerformanceView(APIView):
    """
    Development-only endpoint returning real measured latency metrics
    and Ollama hardware status. Do NOT expose publicly in production.
    """
    permission_classes = [IsAuthenticated]

    def get(self, request):
        import time

        result = {
            "ollama": {
                "reachable": False,
                "model_loaded": False,
                "model": settings.OLLAMA_MODEL,
                "gpu_acceleration": "unknown",
                "hardware": "unknown",
            },
            "latency": {
                "average_ttft_ms": None,
                "average_response_ms": None,
                "status": "NO DATA",
                "recent_requests": [],
            },
            "config": {
                "keep_alive": getattr(settings, "OLLAMA_KEEP_ALIVE", "30m"),
                "num_predict": getattr(settings, "OLLAMA_NUM_PREDICT", 256),
                "num_ctx": getattr(settings, "OLLAMA_NUM_CTX", 4096),
                "temperature": getattr(settings, "OLLAMA_TEMPERATURE", 0.2),
            }
        }

        # --- Ollama health probe ---
        try:
            health = ollama_client.is_healthy()
            result["ollama"]["reachable"] = health.get("status") == "healthy"

            # Query /api/ps to check if the model is actively loaded in memory
            ps_resp = ollama_client._get("/api/ps")
            ps_data = ps_resp.json()
            running_models = [m.get("name", "") for m in ps_data.get("models", [])]
            model_short = settings.OLLAMA_MODEL.split(":")[0]
            result["ollama"]["model_loaded"] = any(model_short in m for m in running_models)

            # Check GPU details from running model entries
            for m in ps_data.get("models", []):
                if model_short in m.get("name", ""):
                    details = m.get("details", {})
                    size_vram = m.get("size_vram", 0)
                    size_total = m.get("size", 1)
                    gpu_pct = round((size_vram / size_total) * 100) if size_total else 0
                    result["ollama"]["gpu_acceleration"] = f"YES ({gpu_pct}% in VRAM)" if size_vram > 0 else "NO (CPU only)"
                    result["ollama"]["hardware"] = details
                    break
        except Exception as exc:
            result["ollama"]["error"] = str(exc)

        # --- Latency metrics from DB ---
        try:
            logs = AIUsageLog.objects.filter(user=request.user, request_type__startswith="chat")
            avg_ttft = logs.exclude(ttft_ms__isnull=True).aggregate(avg=Avg('ttft_ms'))['avg']
            avg_latency = logs.aggregate(avg=Avg('latency_ms'))['avg']

            result["latency"]["average_ttft_ms"] = round(avg_ttft, 1) if avg_ttft else None
            result["latency"]["average_response_ms"] = round(avg_latency, 1) if avg_latency else None

            # Status classification
            if avg_latency is not None:
                if avg_latency < 3000:
                    result["latency"]["status"] = "FAST"
                elif avg_latency < 5000:
                    result["latency"]["status"] = "GOOD"
                elif avg_latency < 8000:
                    result["latency"]["status"] = "SLOW"
                else:
                    result["latency"]["status"] = "CRITICAL"

            # Last 5 requests
            recent = logs.order_by('-created_at').values(
                'request_type', 'latency_ms', 'ttft_ms', 'success', 'created_at'
            )[:5]
            result["latency"]["recent_requests"] = [
                {
                    "type": r['request_type'],
                    "latency_ms": r['latency_ms'],
                    "ttft_ms": r['ttft_ms'],
                    "success": r['success'],
                    "timestamp": r['created_at'].isoformat() if r['created_at'] else None,
                }
                for r in recent
            ]
        except Exception as exc:
            result["latency"]["error"] = str(exc)

        return Response(result)


class AIRouterView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        from ai.services.router import get_available_modes
        try:
            data = get_available_modes()
            return Response(data)
        except Exception as exc:
            return Response({"error": str(exc)}, status=500)

