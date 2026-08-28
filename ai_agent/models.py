from django.db import models
from django.conf import settings
import uuid


def _gen_title(text: str) -> str:
    """Lightweight title from first user message, 30-45 chars, no LLM call."""
    t = text.strip().replace("\n", " ")
    # remove leading boilerplate like "hey vision," etc would be done by agent, keep simple truncation
    if len(t) <= 45:
        return t
    cut = t[:45].rsplit(" ", 1)[0]
    return cut.strip() or t[:45]


class Conversation(models.Model):
    """A chat session between a user and VISION — source of truth for chat history."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='conversations', db_index=True)
    title = models.CharField(max_length=255, default='New Conversation')
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, db_index=True)
    last_message_at = models.DateTimeField(null=True, blank=True, db_index=True)
    is_archived = models.BooleanField(default=False, db_index=True)
    conversation_summary = models.TextField(blank=True, help_text="Rolling summary for long conversations")

    class Meta:
        ordering = ['-updated_at']
        indexes = [
            models.Index(fields=['user', 'is_archived', 'updated_at']),
            models.Index(fields=['user', 'updated_at']),
            models.Index(fields=['user', 'is_archived', 'last_message_at']),
        ]

    def __str__(self):
        return f"Conversation({self.user.email}, {self.title})"

    @staticmethod
    def generate_title(first_message: str) -> str:
        return _gen_title(first_message)


class Message(models.Model):
    """A single message in a conversation."""
    class Role(models.TextChoices):
        USER = 'user', 'User'
        ASSISTANT = 'assistant', 'Assistant'
        SYSTEM = 'system', 'System'
        TOOL = 'tool', 'Tool'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='messages', db_index=True)
    role = models.CharField(max_length=20, choices=Role.choices, db_index=True)
    content = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True, null=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True, null=True)
    # JSON metadata for future: model, latency, tool_calls, tokens etc.
    metadata = models.JSONField(default=dict, blank=True)

    # Legacy tool fields kept for backward compat (mirror into metadata)
    tool_name = models.CharField(max_length=100, blank=True)
    tool_args = models.JSONField(default=dict, blank=True)
    tool_result = models.TextField(blank=True)

    # Keep old timestamp field as alias for created_at for compat
    timestamp = models.DateTimeField(auto_now_add=True, null=True)

    class Meta:
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['conversation', 'created_at']),
            models.Index(fields=['conversation', 'role']),
            models.Index(fields=['role']),
        ]

    def __str__(self):
        return f"Message({self.role}, {self.content[:50]})"


class Attachment(models.Model):
    """Image/file attached to a Message — persisted with conversation."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name='attachments', null=True, blank=True)
    # For pre-message upload, attachment is linked after message creation via temp upload
    conversation = models.ForeignKey(Conversation, on_delete=models.CASCADE, related_name='attachments', null=True, blank=True)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='attachments')
    file = models.FileField(upload_to='attachments/%Y/%m/%d/')
    file_name = models.CharField(max_length=255)
    mime_type = models.CharField(max_length=100)
    file_size = models.PositiveIntegerField()
    width = models.PositiveIntegerField(null=True, blank=True)
    height = models.PositiveIntegerField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['created_at']
        indexes = [
            models.Index(fields=['user', 'conversation']),
            models.Index(fields=['message']),
        ]

    def __str__(self):
        return f"Attachment({self.file_name}, {self.mime_type})"


class Memory(models.Model):
    """Long-term memory — transparent, controllable persistent facts per user (Phase 1)."""
    class Category(models.TextChoices):
        PREFERENCE = 'preference', 'Preference'
        PROJECT = 'project', 'Project'
        FACT = 'fact', 'Fact'
        INSTRUCTION = 'instruction', 'Instruction'

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name='memories', db_index=True)
    category = models.CharField(max_length=20, choices=Category.choices, default=Category.FACT, db_index=True)
    content = models.TextField(help_text="What VISION remembers")
    source_conversation = models.ForeignKey(Conversation, on_delete=models.SET_NULL, null=True, blank=True, related_name='memory_sources')
    importance = models.PositiveSmallIntegerField(default=1, help_text="1-5, higher = surfaced first")
    is_pinned = models.BooleanField(default=False, help_text="Pinned memories are always injected")
    created_at = models.DateTimeField(auto_now_add=True, db_index=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ['-is_pinned', '-importance', '-updated_at']
        indexes = [
            models.Index(fields=['user', 'category']),
            models.Index(fields=['user', 'is_pinned']),
            models.Index(fields=['user', 'updated_at']),
        ]
        verbose_name_plural = 'Memories'

    def __str__(self):
        return f"Memory({self.user.email}, {self.category}, {self.content[:40]})"


class AIUsageLog(models.Model):
    """Tracks local AI inference events without storing sensitive content."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.SET_NULL, null=True, related_name='ai_logs')
    model = models.CharField(max_length=100)
    request_type = models.CharField(max_length=50)
    latency_ms = models.PositiveIntegerField(default=0)
    ttft_ms = models.PositiveIntegerField(null=True, blank=True)
    success = models.BooleanField(default=True)
    error_type = models.CharField(max_length=100, blank=True)
    tool_name = models.CharField(max_length=100, blank=True)
    timestamp = models.DateTimeField(auto_now_add=True)
    created_at = models.DateTimeField(auto_now_add=True, null=True)

    class Meta:
        ordering = ['-timestamp']
