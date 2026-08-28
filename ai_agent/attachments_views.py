import base64
import io
from django.conf import settings
from django.shortcuts import get_object_or_404
from django.http import FileResponse, Http404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework import status
from PIL import Image
from .models import Attachment, Conversation, Message
import time
import logging

logger = logging.getLogger(__name__)

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/jpg"}
MAX_SIZE = getattr(settings, "VISION_MAX_IMAGE_SIZE_MB", 10) * 1024 * 1024
MAX_IMAGES = getattr(settings, "VISION_MAX_IMAGES_PER_MESSAGE", 5)
MAX_DIM = getattr(settings, "VISION_MAX_IMAGE_DIMENSION", 2048)

def validate_image(file):
    if file.content_type not in ALLOWED_MIME:
        return f"Unsupported type {file.content_type}. Use JPG, PNG, WEBP, GIF."
    if file.size > MAX_SIZE:
        return f"This image is too large. Please choose an image under {settings.VISION_MAX_IMAGE_SIZE_MB} MB."
    return None

def process_image(file):
    """Resize if excessively large, return bytes + dimensions."""
    start = time.time()
    try:
        img = Image.open(file)
        img.verify()
        file.seek(0)
        img = Image.open(file)
        w, h = img.size
        # Convert to RGB if needed for JPEG
        if img.mode in ("RGBA", "P", "CMYK"):
            img = img.convert("RGB")
        
        resized = False
        # Resize if larger than MAX_DIM
        if max(w, h) > MAX_DIM:
            ratio = MAX_DIM / max(w, h)
            new_size = (int(w * ratio), int(h * ratio))
            img = img.resize(new_size, Image.LANCZOS)
            w, h = new_size
            resized = True
            
        # Re-encode with compression
        buf = io.BytesIO()
        fmt = "JPEG" if file.content_type in ("image/jpeg", "image/jpg") else "PNG" if file.content_type=="image/png" else "WEBP" if file.content_type=="image/webp" else "PNG"
        
        # Adaptive quality for large files vs small files
        if fmt == "JPEG":
            quality = 85 if (file.size > 2 * 1024 * 1024 or resized) else 93
            img.save(buf, format=fmt, quality=quality, optimize=True)
        else:
            img.save(buf, format=fmt, optimize=True)
        buf.seek(0)
        
        elapsed = int((time.time() - start) * 1000)
        processed_size = buf.getbuffer().nbytes
        logger.debug("[VISION Image] %s: %d bytes -> %d bytes (%dms)", file.name, file.size, processed_size, elapsed)
        
        return buf, w, h
    except Exception as e:
        logger.error("[VISION Image Error] %s", e)
        raise ValueError(f"That file doesn't appear to be a valid image: {e}")

class AttachmentUploadView(APIView):
    permission_classes = [IsAuthenticated]
    def post(self, request, conversation_id):
        conv = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        files = request.FILES.getlist("images") or request.FILES.getlist("file") or []
        if not files:
            # also support single 'image'
            if "image" in request.FILES:
                files = [request.FILES["image"]]
        if not files:
            return Response({"error": "No images provided"}, status=400)
        if len(files) > MAX_IMAGES:
            return Response({"error": f"Maximum {MAX_IMAGES} images per message"}, status=400)
        # check existing attachments count for this conversation? just per-request limit
        attachments = []
        for f in files:
            err = validate_image(f)
            if err:
                return Response({"error": err}, status=400)
            try:
                buf, w, h = process_image(f)
            except ValueError as ve:
                return Response({"error": str(ve)}, status=400)
            att = Attachment.objects.create(
                conversation=conv,
                user=request.user,
                file_name=f.name,
                mime_type=f.content_type,
                file_size=f.size,
                width=w, height=h,
            )
            # save processed bytes
            att.file.save(f.name, buf, save=True)
            attachments.append(att)
        return Response([{
            "id": str(a.id),
            "file_name": a.file_name,
            "mime_type": a.mime_type,
            "url": f"/api/attachments/{a.id}/",
            "width": a.width,
            "height": a.height,
        } for a in attachments], status=201)

class AttachmentServeView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, attachment_id):
        att = get_object_or_404(Attachment, id=attachment_id)
        # Enforce private access
        if att.user != request.user:
            # also allow via conversation ownership
            if not att.conversation or att.conversation.user != request.user:
                if not att.message or att.message.conversation.user != request.user:
                    return Response({"error": "Forbidden"}, status=403)
        if not att.file:
            raise Http404
        return FileResponse(att.file.open("rb"), content_type=att.mime_type)

class AttachmentListView(APIView):
    permission_classes = [IsAuthenticated]
    def get(self, request, conversation_id):
        conv = get_object_or_404(Conversation, id=conversation_id, user=request.user)
        atts = Attachment.objects.filter(conversation=conv).order_by("created_at")
        return Response([{
            "id": str(a.id),
            "file_name": a.file_name,
            "mime_type": a.mime_type,
            "url": f"/api/attachments/{a.id}/",
            "width": a.width,
            "height": a.height,
            "message_id": str(a.message_id) if a.message_id else None,
        } for a in atts])
