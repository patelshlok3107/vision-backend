"""
Production Conversation / Message API — PostgreSQL source of truth.
All endpoints require authentication and enforce conversation.user == request.user.
"""
from django.db.models import Count, Q
from django.shortcuts import get_object_or_404
from django.utils import timezone
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from .models import Conversation, Message


def _serialize_conversation(c):
    return {
        "id": str(c.id),
        "title": c.title,
        "created_at": c.created_at.isoformat() if c.created_at else None,
        "updated_at": c.updated_at.isoformat() if c.updated_at else None,
        "last_message_at": c.last_message_at.isoformat() if c.last_message_at else None,
        "is_archived": c.is_archived,
        "message_count": getattr(c, "message_count", c.messages.count()),
    }


def _serialize_message(m):
    atts = getattr(m, 'prefetched_attachments', None) or list(m.attachments.all()) if hasattr(m, 'attachments') else []
    # Fallback query if not prefetched
    if not atts and hasattr(m, 'attachments'):
        try:
            atts = list(m.attachments.all())
        except: atts=[]
    return {
        "id": str(m.id),
        "conversation_id": str(m.conversation_id),
        "role": m.role,
        "content": m.content,
        "tool_name": getattr(m, 'tool_name', ''),
        "tool_args": getattr(m, 'tool_args', {}),
        "tool_result": getattr(m, 'tool_result', ''),
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
        "metadata": m.metadata or {},
        "attachments": [{"id": str(a.id), "file_name": a.file_name, "mime_type": a.mime_type, "url": f"/api/attachments/{a.id}/", "width": a.width, "height": a.height} for a in atts],
    }


class ConversationListCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        # Query params: is_archived, limit, offset
        is_archived = request.query_params.get("is_archived")
        limit = int(request.query_params.get("limit", 50))
        offset = int(request.query_params.get("offset", 0))
        limit = min(limit, 100)

        qs = Conversation.objects.filter(user=request.user).annotate(message_count=Count("messages"))
        if is_archived is not None:
            qs = qs.filter(is_archived=(is_archived.lower() in ("1", "true", "yes")))
        else:
            # default hide archived unless explicitly requested
            if request.query_params.get("include_archived") != "true":
                qs = qs.filter(is_archived=False)
        qs = qs.order_by("-updated_at")[offset:offset + limit]
        return Response([_serialize_conversation(c) for c in qs])

    def post(self, request):
        # Create new conversation — title from first message or explicit
        title = request.data.get("title") or "New Conversation"
        first_message = request.data.get("first_message")
        if first_message and title == "New Conversation":
            title = Conversation.generate_title(first_message)
        conv = Conversation.objects.create(user=request.user, title=title[:255])
        # Optionally create first USER message immediately
        if first_message:
            Message.objects.create(conversation=conv, role=Message.Role.USER, content=first_message, metadata={})
            conv.last_message_at = timezone.now()
            conv.save(update_fields=["last_message_at", "updated_at"])
        return Response(_serialize_conversation(conv), status=status.HTTP_201_CREATED)


class ConversationDetailViewV2(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, conversation_id):
        conv = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        conv.message_count = conv.messages.count()
        return Response(_serialize_conversation(conv))

    def patch(self, request, conversation_id):
        conv = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        if "title" in request.data:
            conv.title = str(request.data["title"])[:255] or conv.title
        if "is_archived" in request.data:
            conv.is_archived = bool(request.data["is_archived"])
        if "conversation_summary" in request.data:
            conv.conversation_summary = str(request.data["conversation_summary"])[:5000]
        conv.save()
        conv.message_count = conv.messages.count()
        return Response(_serialize_conversation(conv))

    def delete(self, request, conversation_id):
        conv = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        conv.delete()
        return Response({"deleted": True})


class ConversationArchiveView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, conversation_id):
        conv = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        # toggle if no body, else set
        if "is_archived" in request.data:
            conv.is_archived = bool(request.data["is_archived"])
        else:
            conv.is_archived = not conv.is_archived
        conv.save(update_fields=["is_archived", "updated_at"])
        return Response(_serialize_conversation(conv))


class ConversationRenameView(APIView):
    permission_classes = [IsAuthenticated]

    def post(self, request, conversation_id):
        conv = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        title = str(request.data.get("title") or request.data.get("name") or "").strip()
        if not title:
            return Response({"error": "Title required"}, status=400)
        conv.title = title[:255]
        conv.save(update_fields=["title", "updated_at"])
        return Response(_serialize_conversation(conv))


class MessageListCreateView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request, conversation_id):
        conv = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        limit = int(request.query_params.get("limit", 50))
        offset = int(request.query_params.get("offset", 0))
        limit = min(limit, 200)
        total = conv.messages.count()
        qs = conv.messages.prefetch_related("attachments").order_by("-created_at")[offset:offset + limit]
        msgs = list(reversed(list(qs)))
        return Response({
            "conversation_id": str(conv.id),
            "total": total,
            "limit": limit,
            "offset": offset,
            "messages": [_serialize_message(m) for m in msgs],
        })

    def post(self, request, conversation_id):
        conv = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        role = request.data.get("role", Message.Role.USER)
        content = str(request.data.get("content") or "").strip()
        metadata = request.data.get("metadata") or {}
        if not content:
            return Response({"error": "Content required"}, status=400)
        if role not in dict(Message.Role.choices):
            role = Message.Role.USER
        msg = Message.objects.create(conversation=conv, role=role, content=content, metadata=metadata)
        # update conv timestamps / title if first message
        conv.last_message_at = timezone.now()
        if conv.messages.count() == 1 and conv.title == "New Conversation":
            conv.title = Conversation.generate_title(content)
        conv.save(update_fields=["last_message_at", "updated_at", "title"])
        return Response(_serialize_message(msg), status=201)


class ConversationSearchView(APIView):
    permission_classes = [IsAuthenticated]

    def get(self, request):
        q = (request.query_params.get("q") or "").strip()
        if not q:
            return Response([])
        # search title + messages content, backend-side, no full download
        convs = Conversation.objects.filter(user=request.user).filter(
            Q(title__icontains=q) | Q(messages__content__icontains=q)
        ).distinct().order_by("-updated_at")[:20]
        # annotate matched snippet? keep simple
        return Response([_serialize_conversation(c) for c in convs])
