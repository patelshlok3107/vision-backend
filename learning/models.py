"""
VISION Learning — Database models.

Models:
  KnowledgeSource     — Registered source (URL / RSS / admin paste) with authority score
  KnowledgeItem       — Core knowledge unit with embedding, quality score, freshness metadata
  LearningRun         — Audit log for every pipeline execution
  LearningNotification — Admin alerts for high-priority events
  LearningSettings    — Singleton: admin-controlled toggles and schedule
  AdminTrainingExample — Admin-supplied prompt/answer pairs for quality examples
"""
import hashlib
import uuid

from django.db import models
from django.utils import timezone


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

class SourceType(models.TextChoices):
    RSS = "rss", "RSS Feed"
    URL = "url", "URL / Documentation"
    ADMIN_PASTE = "admin_paste", "Admin Paste"
    ADMIN_FILE = "admin_file", "Admin File Upload"


class AuthorityTier(models.TextChoices):
    OFFICIAL = "official", "Official (Tier 1)"
    REPUTABLE = "reputable", "Reputable Publication (Tier 2)"
    COMMUNITY = "community", "Community / Blog (Tier 3)"
    UNKNOWN = "unknown", "Unknown"


class KnowledgeSource(models.Model):
    """A registered information source — RSS feed, documentation URL, or admin paste."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    url = models.URLField(max_length=1000, blank=True)
    source_type = models.CharField(max_length=20, choices=SourceType.choices, default=SourceType.URL)
    authority_tier = models.CharField(max_length=20, choices=AuthorityTier.choices, default=AuthorityTier.COMMUNITY)
    authority_score = models.PositiveSmallIntegerField(
        default=50,
        help_text="0–100. Official docs ≈ 95, reputable press ≈ 75, blogs ≈ 50."
    )
    category = models.CharField(max_length=100, default="general")
    tags = models.JSONField(default=list, blank=True)
    is_active = models.BooleanField(default=True)
    last_fetched_at = models.DateTimeField(null=True, blank=True)
    fetch_interval_hours = models.PositiveSmallIntegerField(default=24)
    created_at = models.DateTimeField(auto_now_add=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-authority_score", "name"]
        verbose_name = "Knowledge Source"

    def __str__(self):
        return f"{self.name} [{self.get_authority_tier_display()}]"


# ---------------------------------------------------------------------------
# Knowledge items
# ---------------------------------------------------------------------------

class ItemStatus(models.TextChoices):
    ACTIVE = "active", "Active"
    SUPERSEDED = "superseded", "Superseded by newer"
    REJECTED = "rejected", "Rejected"
    PENDING = "pending", "Pending admin approval"
    CONFLICT = "conflict", "Conflicting sources"


class KnowledgeCategory(models.TextChoices):
    PROGRAMMING = "programming", "Programming"
    WEB_DEV = "web_dev", "Web Development"
    DATABASES = "databases", "Databases"
    AI_ML = "ai_ml", "AI & Machine Learning"
    SECURITY = "security", "Security"
    DEVOPS = "devops", "DevOps & Cloud"
    TECHNOLOGY = "technology", "Technology News"
    SCIENCE = "science", "Science"
    GENERAL = "general", "General"


class KnowledgeItem(models.Model):
    """
    A single vetted knowledge unit stored in VISION's knowledge base.
    The embedding field stores a JSON-serialised float list for cosine-similarity lookup.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    title = models.CharField(max_length=500)
    summary = models.TextField(help_text="Concise LLM-generated summary (≤ 200 words)")
    content = models.TextField(help_text="Full source content (stored for re-processing)")
    content_hash = models.CharField(
        max_length=64, unique=True, db_index=True,
        help_text="SHA-256 of stripped content — deduplication key"
    )

    # Source metadata
    source = models.ForeignKey(
        KnowledgeSource, null=True, blank=True,
        on_delete=models.SET_NULL, related_name="items"
    )
    source_url = models.URLField(max_length=1000, blank=True)
    source_name = models.CharField(max_length=255, blank=True)
    authority_score = models.PositiveSmallIntegerField(default=50)

    # Classification
    category = models.CharField(max_length=50, choices=KnowledgeCategory.choices, default=KnowledgeCategory.GENERAL)
    subcategory = models.CharField(max_length=100, blank=True, help_text="e.g. 'JavaScript', 'React', 'PostgreSQL'")
    tags = models.JSONField(default=list, blank=True)

    # Quality metrics
    quality_score = models.PositiveSmallIntegerField(default=0, help_text="Composite 0–100 score")
    relevance_score = models.PositiveSmallIntegerField(default=0)
    freshness_score = models.PositiveSmallIntegerField(default=0)
    confidence = models.CharField(
        max_length=10, default="medium",
        choices=[("high", "High"), ("medium", "Medium"), ("low", "Low")]
    )

    # Lifecycle
    status = models.CharField(max_length=20, choices=ItemStatus.choices, default=ItemStatus.ACTIVE, db_index=True)
    superseded_by = models.ForeignKey(
        "self", null=True, blank=True, on_delete=models.SET_NULL, related_name="supersedes"
    )
    conflict_note = models.TextField(blank=True, help_text="Describes any contradicting information found")
    rejection_reason = models.CharField(max_length=255, blank=True)

    # Timestamps & freshness
    published_at = models.DateTimeField(null=True, blank=True)
    collected_at = models.DateTimeField(default=timezone.now)
    last_verified_at = models.DateTimeField(null=True, blank=True)
    version = models.CharField(max_length=50, blank=True, help_text="e.g. 'React 18.3.0'")

    # Vector embedding (JSON list of floats) for cosine similarity
    embedding = models.JSONField(null=True, blank=True)

    # Admin control
    admin_approved = models.BooleanField(default=True, help_text="False until admin approves pending items")
    learning_run = models.ForeignKey(
        "LearningRun", null=True, blank=True, on_delete=models.SET_NULL, related_name="items"
    )

    class Meta:
        ordering = ["-quality_score", "-collected_at"]
        indexes = [
            models.Index(fields=["status", "category"]),
            models.Index(fields=["status", "quality_score"]),
            models.Index(fields=["category", "collected_at"]),
            models.Index(fields=["content_hash"]),
        ]
        verbose_name = "Knowledge Item"

    def __str__(self):
        return f"[{self.category}] {self.title[:60]} (Q:{self.quality_score})"

    @staticmethod
    def make_hash(content: str) -> str:
        return hashlib.sha256(content.strip().encode()).hexdigest()


# ---------------------------------------------------------------------------
# Learning run history
# ---------------------------------------------------------------------------

class RunStatus(models.TextChoices):
    RUNNING = "running", "Running"
    SUCCESS = "success", "Success"
    FAILED = "failed", "Failed"
    PARTIAL = "partial", "Partial (some errors)"


class LearningRun(models.Model):
    """Audit trail for every pipeline execution — daily automatic or manual."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    triggered_by = models.CharField(
        max_length=20, default="scheduler",
        choices=[("scheduler", "Scheduler"), ("admin", "Admin"), ("api", "API")]
    )
    started_at = models.DateTimeField(default=timezone.now)
    finished_at = models.DateTimeField(null=True, blank=True)
    status = models.CharField(max_length=20, choices=RunStatus.choices, default=RunStatus.RUNNING, db_index=True)

    # Aggregated stats
    sources_processed = models.PositiveIntegerField(default=0)
    documents_fetched = models.PositiveIntegerField(default=0)
    items_added = models.PositiveIntegerField(default=0)
    items_updated = models.PositiveIntegerField(default=0)
    items_rejected = models.PositiveIntegerField(default=0)
    duplicates_skipped = models.PositiveIntegerField(default=0)
    conflicts_detected = models.PositiveIntegerField(default=0)

    # Per-category breakdown stored as JSON {category: count}
    category_breakdown = models.JSONField(default=dict, blank=True)

    # Benchmark results JSON: {category: score, overall: score}
    benchmark_results = models.JSONField(null=True, blank=True)
    previous_benchmark = models.JSONField(null=True, blank=True)

    # Quality gate — did benchmark regress?
    benchmark_regression = models.BooleanField(default=False)
    error_log = models.TextField(blank=True)
    notes = models.TextField(blank=True)

    class Meta:
        ordering = ["-started_at"]
        verbose_name = "Learning Run"

    def __str__(self):
        dt = self.started_at.strftime("%Y-%m-%d %H:%M")
        return f"LearningRun {dt} [{self.status}] +{self.items_added} items"

    @property
    def duration_seconds(self):
        if self.finished_at:
            return int((self.finished_at - self.started_at).total_seconds())
        return None


# ---------------------------------------------------------------------------
# Admin notifications
# ---------------------------------------------------------------------------

class NotificationSeverity(models.TextChoices):
    INFO = "info", "Info"
    WARNING = "warning", "Warning"
    CRITICAL = "critical", "Critical"


class LearningNotification(models.Model):
    """High-priority alerts surfaced to the admin after a learning run."""
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    learning_run = models.ForeignKey(LearningRun, null=True, blank=True, on_delete=models.SET_NULL, related_name="notifications")
    severity = models.CharField(max_length=10, choices=NotificationSeverity.choices, default=NotificationSeverity.INFO)
    title = models.CharField(max_length=255)
    body = models.TextField()
    is_read = models.BooleanField(default=False)
    created_at = models.DateTimeField(auto_now_add=True)
    knowledge_item = models.ForeignKey(KnowledgeItem, null=True, blank=True, on_delete=models.SET_NULL)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Learning Notification"

    def __str__(self):
        return f"[{self.severity.upper()}] {self.title}"


# ---------------------------------------------------------------------------
# Admin settings (singleton)
# ---------------------------------------------------------------------------

class LearningSettings(models.Model):
    """
    Singleton model — only one row should exist (id=1).
    Call LearningSettings.get() to retrieve it.
    """
    # Master switch
    enabled = models.BooleanField(default=True)

    # Schedule
    schedule_type = models.CharField(
        max_length=10, default="daily",
        choices=[("daily", "Daily"), ("weekly", "Weekly"), ("manual", "Manual only"), ("disabled", "Disabled")]
    )
    schedule_hour = models.PositiveSmallIntegerField(default=2, help_text="UTC hour for daily run (0–23)")
    schedule_day = models.PositiveSmallIntegerField(null=True, blank=True, help_text="Day of week for weekly (0=Mon)")

    # Per-category toggles
    news_enabled = models.BooleanField(default=True)
    technology_enabled = models.BooleanField(default=True)
    coding_enabled = models.BooleanField(default=True)
    security_enabled = models.BooleanField(default=True)
    ai_ml_enabled = models.BooleanField(default=True)

    # Quality thresholds
    min_quality_score = models.PositiveSmallIntegerField(default=60)
    auto_approve = models.BooleanField(default=True, help_text="Automatically approve items above quality threshold")

    # Fine-tuning (always off by default — Level 3)
    auto_finetune_enabled = models.BooleanField(default=False)

    # Benchmark regression guard
    reject_on_benchmark_regression = models.BooleanField(default=True)

    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        verbose_name = "Learning Settings"
        verbose_name_plural = "Learning Settings"

    def __str__(self):
        return f"LearningSettings (schedule={self.schedule_type}, hour={self.schedule_hour}:00 UTC)"

    @classmethod
    def get(cls) -> "LearningSettings":
        obj, _ = cls.objects.get_or_create(id=1)
        return obj


# ---------------------------------------------------------------------------
# Admin training examples
# ---------------------------------------------------------------------------

class ExampleQuality(models.TextChoices):
    GOOD = "good", "Good (preferred)"
    BAD = "bad", "Bad (anti-pattern)"
    NEUTRAL = "neutral", "Neutral"


class AdminTrainingExample(models.Model):
    """
    Admin-supplied prompt/answer example. Used for knowledge enrichment and
    optional future fine-tuning dataset construction.
    NOT automatically applied to model weights.
    """
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    prompt = models.TextField(help_text="Example user request / question")
    answer = models.TextField(help_text="Ideal answer or code")
    quality = models.CharField(max_length=10, choices=ExampleQuality.choices, default=ExampleQuality.GOOD)
    reason = models.TextField(blank=True, help_text="Why is this a good / bad answer?")
    category = models.CharField(max_length=50, choices=KnowledgeCategory.choices, default=KnowledgeCategory.PROGRAMMING)
    subcategory = models.CharField(max_length=100, blank=True)
    tags = models.JSONField(default=list, blank=True)
    source_description = models.CharField(max_length=255, default="Admin upload")

    # Approval workflow
    approved = models.BooleanField(default=False)
    approved_at = models.DateTimeField(null=True, blank=True)
    preview_summary = models.TextField(blank=True, help_text="Auto-generated preview shown before approval")

    # If added to KB as a KnowledgeItem
    knowledge_item = models.OneToOneField(
        KnowledgeItem, null=True, blank=True, on_delete=models.SET_NULL, related_name="training_example"
    )

    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "Admin Training Example"

    def __str__(self):
        return f"[{self.quality.upper()}] {self.prompt[:60]} ({'approved' if self.approved else 'pending'})"
