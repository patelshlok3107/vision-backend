"""
Memory API — Phase 1 Persistent Memory (transparent, controllable).
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from django.shortcuts import get_object_or_404
from .models import Memory

def _ser(m):
    return {
        "id": str(m.id),
        "category": m.category,
        "content": m.content,
        "importance": m.importance,
        "is_pinned": m.is_pinned,
        "source_conversation": str(m.source_conversation_id) if m.source_conversation_id else None,
        "created_at": m.created_at.isoformat() if m.created_at else None,
        "updated_at": m.updated_at.isoformat() if m.updated_at else None,
    }

class MemoryListCreateView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        category = request.query_params.get("category")
        qs = Memory.objects.filter(user=request.user)
        if category:
            qs = qs.filter(category=category)
        qs = qs.order_by("-is_pinned", "-importance", "-updated_at")[:100]
        return Response([_ser(m) for m in qs])

    def post(self, request):
        content = (request.data.get("content") or "").strip()
        if not content:
            return Response({"error": "Content required"}, status=400)
        if len(content) > 2000:
            return Response({"error": "Content too long (max 2000)"}, status=400)
        category = request.data.get("category") or Memory.Category.FACT
        if category not in dict(Memory.Category.choices):
            category = Memory.Category.FACT
        importance = int(request.data.get("importance") or 1)
        importance = max(1, min(5, importance))
        is_pinned = bool(request.data.get("is_pinned", False))
        m = Memory.objects.create(
            user=request.user,
            category=category,
            content=content,
            importance=importance,
            is_pinned=is_pinned,
        )
        return Response(_ser(m), status=201)

class MemoryDetailView(APIView):
    permission_classes = [IsAuthenticated]
    def patch(self, request, memory_id):
        m = get_object_or_404(Memory, id=memory_id, user=request.user)
        if "content" in request.data:
            c = str(request.data["content"]).strip()
            if not c:
                return Response({"error": "Content cannot be empty"}, status=400)
            m.content = c[:2000]
        if "category" in request.data and request.data["category"] in dict(Memory.Category.choices):
            m.category = request.data["category"]
        if "importance" in request.data:
            m.importance = max(1, min(5, int(request.data["importance"])))
        if "is_pinned" in request.data:
            m.is_pinned = bool(request.data["is_pinned"])
        m.save()
        return Response(_ser(m))

    def delete(self, request, memory_id):
        m = get_object_or_404(Memory, id=memory_id, user=request.user)
        m.delete()
        return Response({"deleted": True})

class MemoryClearView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        # Forget everything
        deleted, _ = Memory.objects.filter(user=request.user).delete()
        return Response({"deleted": deleted})

class MemoryRetrieveView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        # Simple keyword search for transparency/debugging ?q=python
        q = (request.query_params.get("q") or "").strip()
        qs = Memory.objects.filter(user=request.user)
        if q:
            qs = qs.filter(content__icontains=q)
        qs = qs.order_by("-is_pinned", "-importance", "-updated_at")[:20]
        return Response([_ser(m) for m in qs])
