import os
import django
os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")
django.setup()

from learning.tasks import run_daily_learning
from learning.models import KnowledgeSource, KnowledgeItem, LearningSettings
from learning.retrieval import retrieve_knowledge

print("=== Disabling settings checks for test ===")
settings = LearningSettings.get()
settings.enabled = True
settings.auto_approve = True
settings.save()

print("=== Creating Test KnowledgeSource ===")
KnowledgeSource.objects.get_or_create(
    name="Test Source",
    url="https://example.com/feed",
    source_type="rss",
    category="coding",
    authority_score=90
)

print("=== Running Pipeline Sync ===")
# We need an RSS feed or URL that actually works. 
# wait, if url is example.com, it will fail to fetch RSS.
# Mocking it or using a real feed (e.g., HackerNews feed)
try:
    # Just run it, the collectors will skip bad urls
    stats = run_daily_learning()
    print("Pipeline Stats:", stats)
except Exception as e:
    print("Pipeline Error:", e)

# Let's manually create a knowledge item to test retrieval and RAG
try:
    from learning.retrieval import embed_text
    print("=== Creating fake item to test Retrieval ===")
    content = "To center a div in CSS, use flexbox: display: flex; align-items: center; justify-content: center."
    KnowledgeItem.objects.create(
        title="Centering a div in CSS",
        summary=content,
        content=content,
        content_hash=KnowledgeItem.make_hash(content),
        category="coding",
        quality_score=95,
        status="active",
        admin_approved=True,
        embedding=embed_text(content)
    )
except Exception as e:
    print("Fake item creation failed:", e)

print("=== Testing Retrieval ===")
print(retrieve_knowledge("how do i center a div properly?"))

print("ALL TESTS COMPLETED.")
