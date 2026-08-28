"""
VISION Grok/Groq Keep-Alive Service
Keeps the free Grok/Groq API warm by sending a lightweight ping every ~14 minutes.
- Server-side only, no browser dependency
- No chat history, no notifications, no visible messages
- Single scheduler per process, configurable via env
- Survives failures with backoff, never crashes backend
"""
import logging
import os
import threading
import time

logger = logging.getLogger("vision.keepalive")

_lock = threading.Lock()
_instance = None
_thread = None
_stop_event = threading.Event()

# Config — read from Django settings or env, with GROK/GROQ alias support
def _get_config():
    try:
        from django.conf import settings as s
        enabled = getattr(s, "GROK_KEEPALIVE_ENABLED", None)
        if enabled is None:
            enabled = getattr(s, "GROQ_KEEPALIVE_ENABLED", True)
        interval_ms = getattr(s, "GROK_KEEPALIVE_INTERVAL_MS", None)
        if interval_ms is None:
            interval_ms = getattr(s, "GROQ_KEEPALIVE_INTERVAL_MS", 840000)
        # Parse string like "14m"
        if isinstance(interval_ms, str):
            v = interval_ms.strip().lower()
            if v.endswith("m"):
                interval_ms = int(v[:-1]) * 60 * 1000
            elif v.endswith("s"):
                interval_ms = int(v[:-1]) * 1000
            else:
                interval_ms = int(v)
        return bool(enabled), int(interval_ms)
    except Exception:
        # Fallback to env
        enabled_str = os.environ.get("GROK_KEEPALIVE_ENABLED", os.environ.get("GROQ_KEEPALIVE_ENABLED", "true"))
        enabled = enabled_str.lower() not in ("0", "false", "off", "no")
        interval_str = os.environ.get("GROK_KEEPALIVE_INTERVAL_MS", os.environ.get("GROQ_KEEPALIVE_INTERVAL_MS", os.environ.get("GROK_KEEPALIVE_INTERVAL", "840000")))
        try:
            if isinstance(interval_str, str) and interval_str.lower().endswith("m"):
                interval_ms = int(interval_str[:-1]) * 60 * 1000
            elif isinstance(interval_str, str) and interval_str.lower().endswith("s"):
                interval_ms = int(interval_str[:-1]) * 1000
            else:
                interval_ms = int(interval_str)
        except:
            interval_ms = 840000
        return enabled, interval_ms


def _do_ping():
    """Send lightweight ping to Grok/Groq — 1 token, no history, no side effects."""
    try:
        from ai.services.ollama_client import client
        # Check if any cloud provider is configured
        client._refresh_keys()
        has_key = bool(client._grok_key or client._groq_key or client._openai_key)
        if not has_key:
            logger.info("[VISION] Grok keep-alive skipped — no API key configured")
            return False
        # Determine provider for logging
        provider = "grok" if client._grok_key and client._provider in ("grok", "xai") else "groq" if client._groq_key else "openai"
        if "grok" in client.base_url.lower() or "x.ai" in client.base_url.lower():
            provider = "grok"
        elif "groq" in client.base_url.lower():
            provider = "groq"
        logger.info("[VISION] Grok keep-alive ping sent (provider=%s)", provider)
        # Lightweight non-streaming request: "ping" with max_tokens=1, temperature 0
        # This consumes ~1-2 tokens total, minimal cost
        try:
            # Use client._groq_chat directly for cloud, or client.chat for generic
            # We call client.chat with minimal payload to ensure correct routing
            resp = client.chat(
                messages=[{"role": "user", "content": "ping"}],
                temperature=0.0,
                stream=False,
                num_predict=1,
                num_ctx=64,
            )
            # resp is string for non-streaming, or FakeResp
            logger.info("[VISION] Grok keep-alive successful (provider=%s)", provider)
            return True
        except Exception as e:
            # Handle expected API errors without crashing
            err = str(e)[:300]
            logger.warning("[VISION] Grok keep-alive failed (provider=%s): %s", provider, err)
            return False
    except Exception as e:
        logger.warning("[VISION] Grok keep-alive failed: %s", str(e)[:300])
        return False


def _loop(interval_sec: float):
    """Background loop — runs every interval_sec seconds, with backoff on failure."""
    logger.info("[VISION] Grok keep-alive started (interval=%ds)", int(interval_sec))
    # Initial delay is full interval (don't ping immediately on startup if warmup already did)
    # But we want first ping after interval, not immediately
    consecutive_failures = 0
    while not _stop_event.is_set():
        # Wait for interval, but allow early exit
        if _stop_event.wait(timeout=interval_sec):
            break
        if _stop_event.is_set():
            break
        success = _do_ping()
        if not success:
            consecutive_failures += 1
            # Backoff: wait 60s * failures (capped at 5 min) before next normal interval
            # But don't spam — just log and continue; next interval will retry
            backoff = min(60 * consecutive_failures, 300)
            if consecutive_failures <= 3:
                logger.info("[VISION] Grok keep-alive backoff %ds (failure #%d)", backoff, consecutive_failures)
                # Short backoff sleep, but interruptible
                if _stop_event.wait(timeout=backoff):
                    break
            else:
                # After 3 failures, just continue normal schedule, but don't spam logs
                if consecutive_failures == 4:
                    logger.warning("[VISION] Grok keep-alive failing repeatedly — will continue retrying every interval")
        else:
            consecutive_failures = 0


def start_keepalive():
    """Start singleton keep-alive scheduler — safe to call multiple times."""
    global _thread, _instance
    with _lock:
        if _thread and _thread.is_alive():
            logger.info("[VISION] Grok keep-alive already running")
            return _thread
        enabled, interval_ms = _get_config()
        if not enabled:
            logger.info("[VISION] Grok keep-alive disabled (GROK_KEEPALIVE_ENABLED=false)")
            return None
        # Enforce minimum interval 60s to prevent accidental tight loop
        if interval_ms < 60000:
            logger.warning("[VISION] Grok keep-alive interval too short (%dms), clamping to 60000ms", interval_ms)
            interval_ms = 60000
        interval_sec = interval_ms / 1000.0
        # Prevent duplicate start on reload — check env flag
        _stop_event.clear()
        _thread = threading.Thread(target=_loop, args=(interval_sec,), daemon=True, name="grok-keepalive")
        _thread.start()
        _instance = _thread
        return _thread


def stop_keepalive():
    """Stop scheduler — for tests or shutdown."""
    global _thread
    _stop_event.set()
    if _thread and _thread.is_alive():
        _thread.join(timeout=2)
    _thread = None


def trigger_ping():
    """Manual trigger for Render cron or internal endpoint — runs ping synchronously."""
    logger.info("[VISION] Grok keep-alive ping sent (manual trigger)")
    ok = _do_ping()
    if ok:
        logger.info("[VISION] Grok keep-alive successful (manual)")
    else:
        logger.warning("[VISION] Grok keep-alive failed (manual)")
    return ok


def get_status():
    """Return scheduler status for health checks."""
    enabled, interval_ms = _get_config()
    alive = _thread.is_alive() if _thread else False
    return {
        "enabled": enabled,
        "interval_ms": interval_ms,
        "running": alive,
        "provider": "grok/groq",
    }
