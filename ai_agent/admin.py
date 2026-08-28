from django.contrib import admin
from .models import Conversation, Message, Attachment, Memory, AIUsageLog

@admin.register(Memory)
class MemoryAdmin(admin.ModelAdmin):
    list_display = ("user", "category", "content", "importance", "is_pinned", "updated_at")
    list_filter = ("category", "is_pinned")
    search_fields = ("content",)

@admin.register(Conversation)
class ConversationAdmin(admin.ModelAdmin):
    list_display = ("title", "user", "updated_at")

@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ("conversation", "role", "content", "created_at")

@admin.register(Attachment)
class AttachmentAdmin(admin.ModelAdmin):
    list_display = ("file_name", "user", "mime_type")

@admin.register(AIUsageLog)
class AIUsageLogAdmin(admin.ModelAdmin):
    list_display = ("user", "model", "request_type", "latency_ms", "success")
