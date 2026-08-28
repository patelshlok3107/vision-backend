"""
Model Router — Phase 1: fast / think / vision routing (VISION spec #5).
Local-first: picks installed Ollama model, never calls external API.
"""
import re
from django.conf import settings
from .ollama_client import client

# Think mode enum exposed to frontend — Phase 4: Auto is default intelligent
MODES = ["auto", "fast", "think", "vision", "code", "agent"]

# Keyword sets for intent classification
_COMPLEX_KEYWORDS = [
    "think step", "step by step", "think carefully", "deep think",
    "let me think", "reason about", "chain of thought", "cognitive",
    "analyze deeply", "work through", "solve this", "prove",
]

_CODE_KEYWORDS = [
    "code", "bug", "function", "error", "stack trace", "python", "javascript",
    "typescript", "fix", "refactor", "package.json", "e-commerce", "ecommerce",
    "react", "build app", "generate code", "website", "component", "api",
    "implement", "deploy", "database", "sql", "docker", "terminal",
    "npm", "pip", "import ", "def ", "class ", "async ", "await",
    "```", "src/", "node_modules", "debug", "compile", "runtime",
    "html", "css", "c program", "landing page", "dashboard", "script",
    "write a", "create a", "build a", "generate a",
]

_AGENT_KEYWORDS = [
    "open chrome", "search for", "browse", "navigate", "find my project",
    "list files", "read file", "write file", "run command", "control",
    "research", "compare products", "investigate", "reverse image",
    "download", "upload", "execute", "shell", "bash", "powershell",
    "click", "scroll", "type into", "open app",
]

_PERSONAL_KEYWORDS = [
    "my project", "remember when", "yesterday", "favorite", "previous",
    "we discussed", "my preference", "last time", "earlier today",
    "before we", "as i mentioned", "my code", "my file", "our conversation",
    "i told you", "you said", "based on what we",
]

_ATTACHMENT_KEYWORDS = [
    "this image", "the picture", "that screenshot", "the photo",
    "in the image", "look at this", "analyze this image", "describe this",
    "the diagram", "the chart", "the graph", "attached", "uploaded",
]

def get_router_config():
    """Return router config from settings — all local models."""
    return {
        "fast_model": getattr(settings, "OLLAMA_FAST_MODEL", "") or getattr(settings, "OLLAMA_TEXT_MODEL", "") or getattr(settings, "OLLAMA_MODEL", "llama3"),
        "think_model": getattr(settings, "OLLAMA_THINK_MODEL", "") or getattr(settings, "OLLAMA_TEXT_MODEL", "") or getattr(settings, "OLLAMA_MODEL", "llama3"),
        "code_model": getattr(settings, "OLLAMA_CODE_MODEL", "") or getattr(settings, "OLLAMA_TEXT_MODEL", "") or getattr(settings, "OLLAMA_MODEL", "llama3"),
        "vision_model": getattr(settings, "OLLAMA_VISION_MODEL", ""),
        "text_model": getattr(settings, "OLLAMA_TEXT_MODEL", "") or getattr(settings, "OLLAMA_MODEL", "llama3"),
    }

def classify_intent(message: str, has_image: bool, explicit_mode: str) -> dict:
    """
    Classify intent and return full tuning dict.
    Keys: mode, skip_memory, skip_rag, is_simple, num_ctx, num_predict, temperature, keep_alive,
          top_k, top_p, repeat_penalty, is_ultra_short
    """
    msg_low = (message or "").lower().strip()
    msg_len = len(message or "")
    mode = (explicit_mode or "").lower().strip()
    if mode not in MODES:
        mode = "auto"

    # Explicit mode wins first
    if mode == "fast":
        pass
    elif mode == "think":
        pass
    elif mode == "code":
        pass
    elif mode == "agent":
        pass
    elif mode == "vision":
        pass
    else:
        # Auto mode: figure out if it needs complex capabilities
        # Code detection — explicit markers
        code_score = 0
        for kw in _CODE_KEYWORDS:
            if kw in msg_low:
                code_score += 1
        has_code_markers = any(c in msg_low for c in ["```", "def ", "import ", "async ", "function ", "const ", "class ", "interface ", "npm", "src/", "node_modules"])
        if code_score >= 2 or (code_score >= 1 and has_code_markers) or (code_score >= 1 and msg_len > 60):
            mode = "code"

        # Agent detection
        if mode == "auto":
            agent_score = sum(1 for kw in _AGENT_KEYWORDS if kw in msg_low)
            if agent_score >= 1:
                mode = "agent"

        # Think detection — only if explicitly asks for reasoning AND long enough
        if mode == "auto":
            has_complex_kw = any(kw in msg_low for kw in _COMPLEX_KEYWORDS)
            think_kw = ["explain in depth", "compare and contrast", "detailed analysis", "architecture", "quantum", "philosophical", "in detail"]
            has_think_kw = any(kw in msg_low for kw in think_kw)
            if (has_complex_kw or (has_think_kw and msg_len > 80)) and msg_len > 60:
                mode = "think"

        # DEFAULT: if nothing matched complex → FAST
        if mode == "auto":
            mode = "fast"

    # Vision always wins when image present (override explicit if needed)
    if has_image:
        mode = "vision"

    # ULTRA_FAST determination — AGGRESSIVE: 90% of short queries should hit this
    skip_memory = False
    skip_rag = False
    is_simple = False
    is_ultra_short = msg_len < 50

    ultra_fast_eligible = (
        msg_len < 150
        and mode == "fast"
        and not has_image
    )

    if ultra_fast_eligible:
        no_complex_request = not any(kw in msg_low for kw in (_COMPLEX_KEYWORDS + _CODE_KEYWORDS + _AGENT_KEYWORDS))
        no_personal_ref = not any(kw in msg_low for kw in _PERSONAL_KEYWORDS)
        # Only skip memory if definitely no personal context needed
        if no_complex_request and no_personal_ref:
            is_simple = True
            skip_memory = True
            skip_rag = True

    # Tune parameters based on mode
    if is_simple:
        # Even more aggressive for ultra-short greetings/questions
        # CPU hardware: 20 tok/s, so 64 tokens = ~3s total, 96 = ~4.5s. Keep simple answers <4s.
        if is_ultra_short and msg_len < 25:
            num_ctx = 512
            num_predict = 64
        else:
            num_ctx = getattr(settings, "ULTRA_FAST_NUM_CTX", 1024)
            # Use 96 for normal ultra_fast (was 256, too large for CPU 8B)
            _ultra_pred = getattr(settings, "ULTRA_FAST_NUM_PREDICT", 256)
            num_predict = 96 if _ultra_pred > 120 else _ultra_pred
        temperature = getattr(settings, "ULTRA_FAST_TEMPERATURE", 0.4)
        top_k = getattr(settings, "ULTRA_FAST_TOP_K", 20)
        top_p = getattr(settings, "ULTRA_FAST_TOP_P", 0.9)
        repeat_penalty = getattr(settings, "ULTRA_FAST_REPEAT_PENALTY", 1.1)
    elif mode == "think":
        num_ctx = getattr(settings, "THINK_NUM_CTX", 12288)
        num_predict = getattr(settings, "THINK_NUM_PREDICT", 3072)
        temperature = 0.1
        top_k = 40
        top_p = 0.95
        repeat_penalty = 1.1
    elif mode == "code":
        # FAST CODE MODE: dynamic limits based on request size — critical for 5s target on CPU
        msg_len_code = len(msg_low)
        is_small_code = msg_len_code < 120 and not any(x in msg_low for x in ["website", "ecommerce", "e-commerce", "landing page", "dashboard", "e commerce", "full stack", "entire project"])
        is_medium_code = not is_small_code and msg_len_code < 250 and not any(x in msg_low for x in ["ecommerce", "full stack", "entire project", "create a website with"])
        if is_small_code:
            num_ctx = 4096
            num_predict = 512
        elif is_medium_code:
            num_ctx = 6144
            num_predict = 1024
        else:
            # Large website/project — still cap at 8192 ctx / 2048 predict (was 16384/4096, too slow)
            num_ctx = 8192
            num_predict = 2048
        # Allow override via settings only if smaller than dynamic
        cfg_ctx = getattr(settings, "CODE_NUM_CTX", 16384)
        cfg_pred = getattr(settings, "CODE_NUM_PREDICT", 4096)
        # Never exceed dynamic for small/medium to keep 5s target
        if is_small_code:
            num_ctx = min(num_ctx, cfg_ctx if cfg_ctx < 5000 else num_ctx)
            num_predict = min(num_predict, cfg_pred if cfg_pred < 800 else num_predict)
        temperature = 0.05
        top_k = 30
        top_p = 0.9
        repeat_penalty = 1.05
        # Code never needs RAG or memory — force skip
        skip_memory = True
        skip_rag = True
    elif mode == "agent":
        num_ctx = getattr(settings, "AGENT_NUM_CTX", 12288)
        num_predict = getattr(settings, "AGENT_NUM_PREDICT", 2048)
        temperature = 0.2
        top_k = 40
        top_p = 0.95
        repeat_penalty = 1.1
    elif mode == "vision":
        num_ctx = 8192
        num_predict = 1024
        temperature = 0.2
        top_k = 40
        top_p = 0.95
        repeat_penalty = 1.1
    else:
        num_ctx = getattr(settings, "NORMAL_NUM_CTX", 4096)
        num_predict = getattr(settings, "NORMAL_NUM_PREDICT", 1024)
        temperature = 0.2
        top_k = 40
        top_p = 0.95
        repeat_penalty = 1.1

    keep_alive = getattr(settings, "OLLAMA_KEEP_ALIVE", "2h")

    return {
        "mode": mode,
        "skip_memory": skip_memory,
        "skip_rag": skip_rag,
        "is_simple": is_simple,
        "is_ultra_short": is_ultra_short,
        "num_ctx": num_ctx,
        "num_predict": num_predict,
        "temperature": temperature,
        "keep_alive": keep_alive,
        "top_k": top_k,
        "top_p": top_p,
        "repeat_penalty": repeat_penalty,
    }


def resolve_model(mode: str, has_image: bool, message: str = "") -> tuple[str, str]:
    """
    Returns (model_name, resolved_mode).
    Phase 4: Auto routes to FAST by default unless explicit code/think/agent hints.
    Priority:
      1. has_image -> vision
      2. explicit mode (fast/think/vision/code/agent)
      3. auto -> FAST unless explicit complex hints (code, agent, think)
    """
    cfg = get_router_config()
    mode = (mode or "").lower().strip()
    if mode not in MODES:
        mode = "auto"

    # Vision always wins when image present
    if has_image:
        if cfg["vision_model"]:
            return cfg["vision_model"], "vision"
        return cfg["text_model"], "vision"

    if mode == "fast":
        return cfg["fast_model"] or cfg["text_model"], "fast"
    if mode == "think":
        return cfg["think_model"] or cfg["text_model"], "think"
    if mode == "vision":
        return cfg["vision_model"] or cfg["text_model"], "vision"
    if mode == "code":
        return cfg["code_model"] or cfg["text_model"], "code"
    if mode == "agent":
        return cfg["think_model"] or cfg["text_model"], "agent"

    # Auto — DEFAULT = FAST unless explicit complex markers
    msg_low = (message or "").lower()
    msg_len = len(message or "")

    # Code: only if STRONG markers (multi-keyword or has code syntax)
    code_score = sum(1 for kw in _CODE_KEYWORDS if kw in msg_low)
    has_code_markers = any(c in msg_low for c in ["```", "def ", "import ", "async ", "function ", "const ", "class ", "npm", "src/", "node_modules"])
    if code_score >= 2 or (code_score >= 1 and has_code_markers) or (code_score >= 1 and msg_len > 100):
        return cfg["code_model"] or cfg["text_model"], "code"

    # Agent: any explicit agent keyword
    if any(kw in msg_low for kw in _AGENT_KEYWORDS):
        return cfg["think_model"] or cfg["text_model"], "agent"

    # Think: only if explicitly requests deep reasoning AND long enough
    has_complex_kw = any(kw in msg_low for kw in _COMPLEX_KEYWORDS)
    think_triggers = ["explain in depth", "compare and contrast", "detailed analysis", "architecture design", "quantum mechanics", "philosophical"]
    if (has_complex_kw or any(t in msg_low for t in think_triggers)) and msg_len > 80:
        return cfg["think_model"] or cfg["text_model"], "think"

    # DEFAULT: FAST mode (90%+ of short questions hit this)
    return cfg["fast_model"] or cfg["text_model"], "fast"

def get_available_modes():
    cfg = get_router_config()
    health = client.healthCheck()
    return {
        "modes": MODES,
        "config": cfg,
        "health": health,
        "available": {
            "vision": bool(health["visionModel"]["installed"] and health["visionModel"]["capable"]),
            "fast": bool(health["textModel"]["installed"]),
            "think": bool(health["textModel"]["installed"]),
            "code": bool(client.model_exists(cfg["code_model"]) if cfg["code_model"] else health["textModel"]["installed"]),
            "agent": bool(health["textModel"]["installed"]),
        }
    }
