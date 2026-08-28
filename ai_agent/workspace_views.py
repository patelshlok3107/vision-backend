"""
Workspace Views — Phase 2 direct file management (no LLM).
Sandboxed per-user, complements tool calling.
"""
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from ai.services.tools import _safe_path, _user_workspace
from pathlib import Path

class WorkspaceListView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        path = request.query_params.get("path", "")
        try:
            from ai.services.tools import tool_filesystem_list
            data = tool_filesystem_list(request.user, path=path)
            return Response(data)
        except Exception as e:
            return Response({"error": str(e)}, status=400)

class WorkspaceReadView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request):
        path = request.query_params.get("path", "")
        if not path:
            return Response({"error": "path required"}, status=400)
        try:
            from ai.services.tools import tool_filesystem_read
            data = tool_filesystem_read(request.user, path=path)
            return Response(data)
        except Exception as e:
            return Response({"error": str(e)}, status=400)

class WorkspaceWriteView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        path = request.data.get("path", "")
        content = request.data.get("content", "")
        if not path:
            return Response({"error": "path required"}, status=400)
        try:
            from ai.services.tools import tool_filesystem_write
            data = tool_filesystem_write(request.user, path=path, content=content)
            return Response(data)
        except Exception as e:
            return Response({"error": str(e)}, status=400)

class WorkspaceDeleteView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request):
        path = request.data.get("path", "")
        if not path:
            return Response({"error": "path required"}, status=400)
        try:
            from ai.services.tools import tool_filesystem_delete
            data = tool_filesystem_delete(request.user, path=path)
            return Response(data)
        except Exception as e:
            return Response({"error": str(e)}, status=400)
