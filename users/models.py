from django.contrib.auth.models import AbstractUser
from django.db import models
import uuid

class User(AbstractUser):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    email = models.EmailField(unique=True)

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['username']

    def __str__(self):
        return self.email

class Profile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='profile')
    bio = models.TextField(blank=True, null=True)
    avatar = models.ImageField(upload_to='avatars/%Y/%m/', null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    def __str__(self):
        return f"{self.user.email} Profile"


class UserSettings(models.Model):
    """Persisted per-user preferences — mirrors frontend localStorage but server-backed when logged in."""
    user = models.OneToOneField(User, on_delete=models.CASCADE, related_name='settings')
    # General
    language = models.CharField(max_length=10, default='en')
    default_mode = models.CharField(max_length=20, default='auto')
    enter_to_send = models.BooleanField(default=True)
    show_suggested_prompts = models.BooleanField(default=True)
    auto_scroll = models.BooleanField(default=True)
    confirm_delete = models.BooleanField(default=True)
    # Appearance
    theme = models.CharField(max_length=10, default='system')  # dark | light | system
    chat_density = models.CharField(max_length=10, default='comfortable')  # comfortable | compact
    animations = models.BooleanField(default=True)
    reduce_motion = models.BooleanField(default=False)
    font_size = models.CharField(max_length=10, default='medium')  # small | medium | large
    # Voice
    voice_enabled = models.BooleanField(default=True)
    voice_id = models.CharField(max_length=50, default='vision-male')
    speech_speed = models.CharField(max_length=10, default='1x')
    autoplay_voice = models.BooleanField(default=False)
    # AI
    chat_model = models.CharField(max_length=100, blank=True, default='')
    code_model = models.CharField(max_length=100, blank=True, default='')
    vision_model = models.CharField(max_length=100, blank=True, default='')
    reasoning_model = models.CharField(max_length=100, blank=True, default='')
    agent_model = models.CharField(max_length=100, blank=True, default='')
    temperature = models.FloatField(default=0.2)
    context_length = models.IntegerField(default=8192)
    streaming = models.BooleanField(default=True)
    show_generation_status = models.BooleanField(default=True)
    # Performance
    fast_mode = models.BooleanField(default=True)
    use_routing = models.BooleanField(default=True)
    keep_warm = models.BooleanField(default=True)
    max_tokens = models.IntegerField(default=2048)
    # Privacy
    chat_history_enabled = models.BooleanField(default=True)
    memory_enabled = models.BooleanField(default=True)
    save_files = models.BooleanField(default=True)
    use_history_context = models.BooleanField(default=True)
    analytics = models.BooleanField(default=False)
    personalization = models.BooleanField(default=True)
    # Notifications
    notif_ai_complete = models.BooleanField(default=True)
    notif_agent_complete = models.BooleanField(default=True)
    notif_build_complete = models.BooleanField(default=True)
    notif_research_complete = models.BooleanField(default=True)
    notif_system = models.BooleanField(default=True)
    notif_email = models.BooleanField(default=False)
    updated_at = models.DateTimeField(auto_now=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Settings({self.user.email})"
