"""
Memory service — Phase 1 persistent memory retrieval & auto-extraction.
Gracefully degraded: if no memory, no impact. Transparent & controllable.
"""
import re
import logging
from django.db.models import Q

logger = logging.getLogger(__name__)

# Heuristic patterns for auto-extraction (conservative) — Phase 1 improved to capture multi-word
_PATTERNS = [
    # My favorite X is Y — capture up to 4 words for X, rest Y
    (re.compile(r"my favorite (.+?) is ([^.!\n]+)", re.I), "preference", lambda m: f"Favorite {m.group(1).strip()}: {m.group(2).strip()}"),
    (re.compile(r"i prefer ([^.!\n]+)", re.I), "preference", lambda m: f"Prefers {m.group(1).strip()}"),
    (re.compile(r"remember that (.+)", re.I), "fact", lambda m: m.group(1).strip()),
    (re.compile(r"my (\w+) is ([^.!\n]+)", re.I), "fact", lambda m: f"{m.group(1)} is {m.group(2).strip()}"),
    (re.compile(r"my favorite language is ([^.!\n]+)", re.I), "preference", lambda m: f"Favorite language: {m.group(1).strip()}"),
]

def retrieve_memories(user, query: str, limit=6, memory_enabled=True) -> list:
    """Retrieve relevant memories for query. Simple keyword + pinned priority for Phase 1."""
    if not memory_enabled or not user or not query:
        return []
    try:
        from ai_agent.models import Memory
        # Always include pinned
        pinned = list(Memory.objects.filter(user=user, is_pinned=True).order_by("-importance", "-updated_at")[:3])
        # Keyword match for remainder
        remaining = limit - len(pinned)
        if remaining <= 0:
            return pinned
        qs = Memory.objects.filter(user=user)
        # Exclude pinned already
        pinned_ids = {m.id for m in pinned}
        # If query short, just return recent
        if len(query.strip()) < 3:
            recent = list(qs.exclude(id__in=pinned_ids).order_by("-updated_at")[:remaining])
            return pinned + recent
        # Filter by icontains on query tokens
        tokens = [t for t in re.split(r"\W+", query.lower()) if len(t) > 2][:5]
        if tokens:
            q_obj = Q()
            for t in tokens:
                q_obj |= Q(content__icontains=t)
            matched = list(qs.filter(q_obj).exclude(id__in=pinned_ids).order_by("-is_pinned", "-importance", "-updated_at")[:remaining*2])
            # Simple rank by token overlap
            def score(m):
                c = m.content.lower()
                return sum(1 for t in tokens if t in c) * 2 + m.importance
            matched.sort(key=score, reverse=True)
            matched = matched[:remaining]
            # If not enough, fill with recent
            if len(matched) < remaining:
                extra = list(qs.exclude(id__in=pinned_ids).exclude(id__in=[m.id for m in matched]).order_by("-updated_at")[:remaining - len(matched)])
                matched = matched + extra
            return pinned + matched
        else:
            recent = list(qs.exclude(id__in=pinned_ids).order_by("-updated_at")[:remaining])
            return pinned + recent
    except Exception as exc:
        logger.warning("Memory retrieve failed: %s", exc)
        return []

def format_memories(memories: list) -> str:
    if not memories:
        return ""
    lines = []
    for m in memories:
        cat = m.category if hasattr(m, "category") else "fact"
        lines.append(f"- [{cat}] {m.content}")
    return "VISION remembers:\n" + "\n".join(lines)

def try_extract_memories(user, user_message: str, conversation=None) -> list:
    """Heuristic auto-extract: conservative, only explicit statements. Returns created memories."""
    if not user or not user_message:
        return []
    # Respect memory toggle if passed via request - checked at view level, but default to True here
    # Heuristic only for explicit remember/preference
    created = []
    try:
        from ai_agent.models import Memory
        text = user_message.strip()
        # Avoid extracting from short greetings
        if len(text) < 15:
            return []
        # Check each pattern, create at most 1 per message to avoid spam
        for pat, category, fmt in _PATTERNS:
            m = pat.search(text)
            if m:
                content = fmt(m)[:500].strip()
                # Dedupe: don't create if very similar exists
                exists = Memory.objects.filter(user=user, content__iexact=content).exists()
                if exists:
                    break
                mem = Memory.objects.create(
                    user=user,
                    category=category,
                    content=content,
                    source_conversation=conversation,
                    importance=3,
                )
                created.append(mem)
                break
        # Also handle "My project is ..." or "I'm working on ..."
        if not created and re.search(r"(i'm working on|my project is|current project)", text, re.I):
            content = text[:300].strip()
            if len(content) > 20:
                exists = Memory.objects.filter(user=user, content__icontains=content[:40]).exists()
                if not exists:
                    mem = Memory.objects.create(
                        user=user,
                        category=Memory.Category.PROJECT,
                        content=content,
                        source_conversation=conversation,
                        importance=2,
                    )
                    created.append(mem)
    except Exception as exc:
        logger.warning("Memory extract failed: %s", exc)
    return created
