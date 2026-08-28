"""
OllamaClient — single reusable service for Vision.
Do NOT create ad-hoc HTTP requests elsewhere.

Timeout strategy:
  - Streaming requests:     timeout=(CONNECT_TIMEOUT, None)
      → only the TCP connect is bounded; token reads never time out
      → Ollama can take as long as it needs to generate each chunk
  - Non-streaming requests: timeout=(CONNECT_TIMEOUT, read_timeout)
      → both connect + full response are bounded
"""
import logging
import time
import requests
from django.conf import settings

logger = logging.getLogger(__name__)

# How long we wait for the TCP connection to Ollama to open (seconds).
# Keep short — if Ollama isn't up yet this should fail fast.
CONNECT_TIMEOUT = getattr(settings, "OLLAMA_CONNECT_TIMEOUT", 10)


class OllamaError(Exception): pass
class OllamaUnavailableError(OllamaError): pass
class OllamaModelNotFoundError(OllamaError): pass
class OllamaTimeoutError(OllamaError): pass


VISION_FAMILIES = {
    "llava", "bakllava", "moondream", "minicpm-v",
    "qwen2-vl", "llava-phi3", "llava-llama3",
}


class OllamaClient:
    def __init__(self):
        self.base_url = settings.OLLAMA_BASE_URL.rstrip("/")
        self.text_model = getattr(settings, "OLLAMA_TEXT_MODEL", "") or getattr(settings, "OLLAMA_MODEL", "")
        self.vision_model = getattr(settings, "OLLAMA_VISION_MODEL", "")
        self.embedding_model = settings.OLLAMA_EMBEDDING_MODEL
        # Read timeouts (non-streaming only — streaming uses None)
        self.text_read_timeout = getattr(settings, "OLLAMA_TEXT_TIMEOUT", 90000) / 1000.0
        self.vision_read_timeout = getattr(settings, "OLLAMA_VISION_TIMEOUT", 300000) / 1000.0
        self.session = requests.Session()
        # Connection pooling: keep-alive + properly sized pool for sub-10ms connect
        from requests.adapters import HTTPAdapter
        adapter = HTTPAdapter(pool_connections=10, pool_maxsize=20, max_retries=0)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        # Warm: ensure TCP connection pre-established
        self.session.headers.update({"Connection": "keep-alive"})

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _post(self, path: str, payload: dict, timeout=None, stream: bool = False):
        """
        POST to Ollama.

        For streaming requests: timeout=(CONNECT_TIMEOUT, None)
            — only the TCP handshake is bounded; reads never time out so
              Ollama can stream tokens as slowly as needed.

        For non-streaming requests: timeout=(CONNECT_TIMEOUT, read_s)
            — both connect and full response read are bounded.
        """
        url = f"{self.base_url}{path}"

        if stream:
            # Never apply a total-response timeout to a streaming call.
            effective_timeout = (CONNECT_TIMEOUT, None)
        elif timeout is not None:
            # Caller-supplied (e.g. /api/show health checks)
            effective_timeout = (CONNECT_TIMEOUT, timeout) if isinstance(timeout, (int, float)) else timeout
        else:
            effective_timeout = (CONNECT_TIMEOUT, self.text_read_timeout)

        try:
            resp = self.session.post(url, json=payload, timeout=effective_timeout, stream=stream)
        except requests.exceptions.ConnectionError as exc:
            raise OllamaUnavailableError(
                f"VISION can't connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running."
            ) from exc
        except requests.exceptions.Timeout as exc:
            # For non-streaming this is a real timeout; for streaming it
            # means the initial TCP connect timed out (Ollama is down).
            raise OllamaTimeoutError(
                "Ollama connection timed out. The local model may need more time "
                f"to load, or Ollama is not running at {self.base_url}."
            ) from exc

        if resp.status_code == 404:
            model = payload.get("model", "unknown")
            raise OllamaModelNotFoundError(model)
        if not resp.ok:
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}: {resp.text[:300]}")
        return resp

    def _get(self, path: str, timeout: float = 10):
        url = f"{self.base_url}{path}"
        try:
            resp = self.session.get(url, timeout=(CONNECT_TIMEOUT, timeout))
        except requests.exceptions.ConnectionError as exc:
            raise OllamaUnavailableError(
                f"VISION can't connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running."
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise OllamaTimeoutError("Ollama health check timed out.") from exc
        if not resp.ok:
            raise OllamaError(f"Ollama returned HTTP {resp.status_code}")
        return resp

    # ── Public API ───────────────────────────────────────────────────────────

    def list_models(self) -> list[dict]:
        """Return raw model dicts from /api/tags."""
        try:
            resp = self._get("/api/tags")
            return resp.json().get("models", [])
        except OllamaUnavailableError:
            raise
        except Exception as exc:
            raise OllamaError(str(exc))

    def model_exists(self, name: str) -> bool:
        if not name:
            return False
        try:
            models = self.list_models()
            short = name.split(":")[0].lower()
            for m in models:
                n = m.get("name", "").lower()
                if n == name.lower() or n.startswith(short + ":") or n.split(":")[0] == short:
                    return True
            return False
        except OllamaUnavailableError:
            return False

    def get_model_info(self, name: str) -> dict | None:
        """Call /api/show to get capabilities/family."""
        if not name:
            return None
        try:
            resp = self._post("/api/show", {"name": name}, timeout=10)
            return resp.json()
        except Exception:
            return None

    def is_vision_capable(self, name: str) -> bool:
        if not name:
            return False
        info = self.get_model_info(name)
        if info:
            caps = [c.lower() for c in info.get("capabilities", [])]
            if "vision" in caps:
                return True
            details = info.get("details", {})
            family = details.get("family", "").lower()
            families = [f.lower() for f in details.get("families", [])] + [family]
            if any(f in VISION_FAMILIES for f in families):
                return True
        # Fallback: check tags list
        try:
            for m in self.list_models():
                n = m.get("name", "").lower()
                if n == name.lower() or n.split(":")[0] == name.split(":")[0].lower():
                    caps = [c.lower() for c in m.get("capabilities", [])]
                    if "vision" in caps:
                        return True
                    details = m.get("details", {})
                    fam = details.get("family", "").lower()
                    if fam in VISION_FAMILIES:
                        return True
        except Exception:
            pass
        # Model name heuristic
        low = name.lower()
        if any(v in low for v in ["llava", "bakllava", "moondream", "minicpm-v", "qwen2-vl"]):
            return True
        return False

    def validate_vision_model(self) -> tuple[bool, str]:
        """
        Validate the configured vision model before sending an image.
        Returns (ok, error_message). Used for early error surfacing.
        """
        if not self.vision_model:
            return False, (
                "No vision model is configured. "
                "Set OLLAMA_VISION_MODEL in your .env file (e.g. OLLAMA_VISION_MODEL=llava)."
            )
        if not self.model_exists(self.vision_model):
            return False, (
                f"VISION vision model is unavailable.\n"
                f"Model: {self.vision_model}\n"
                f"Please install it with: ollama pull {self.vision_model}"
            )
        if not self.is_vision_capable(self.vision_model):
            return False, (
                f"The configured model '{self.vision_model}' does not support image analysis.\n"
                f"Please configure a vision-capable model such as llava, minicpm-v, or qwen2-vl."
            )
        return True, ""

    def healthCheck(self) -> dict:
        """Structured health per spec — no sensitive env exposure."""
        base = self.base_url
        result = {
            "ollama": {"connected": False, "baseUrl": base},
            "textModel": {"name": self.text_model or "", "installed": False},
            "visionModel": {
                "name": self.vision_model or "",
                "installed": False,
                "capable": False,
                "configured": bool(self.vision_model),
            },
        }
        try:
            self._get("/api/tags", timeout=5)
            result["ollama"]["connected"] = True
        except OllamaUnavailableError as e:
            result["ollama"]["error"] = str(e)
            return result
        except Exception as e:
            result["ollama"]["error"] = str(e)
            return result

        if self.text_model:
            result["textModel"]["installed"] = self.model_exists(self.text_model)
        if self.vision_model:
            result["visionModel"]["installed"] = self.model_exists(self.vision_model)
            if result["visionModel"]["installed"]:
                result["visionModel"]["capable"] = self.is_vision_capable(self.vision_model)
        return result

    def chat(
        self,
        messages: list[dict],
        temperature: float = 0.2,
        stream: bool = False,
        model: str | None = None,
        is_vision: bool = False,
        num_predict: int | None = None,
        num_ctx: int | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repeat_penalty: float | None = None,
        keep_alive: str | None = None,
    ):
        use_model = model
        if not use_model:
            use_model = self.vision_model if is_vision else self.text_model
        if not use_model:
            raise OllamaModelNotFoundError("__no_model_configured__")

        base_predict = getattr(settings, "OLLAMA_NUM_PREDICT", 2048)
        base_ctx = getattr(settings, "OLLAMA_NUM_CTX", 8192)
        ka = keep_alive if keep_alive else getattr(settings, "OLLAMA_KEEP_ALIVE", "2h")

        options: dict = {
            "temperature": temperature,
            "num_predict": num_predict if num_predict is not None else base_predict,
            "num_ctx": num_ctx if num_ctx is not None else base_ctx,
        }
        if top_k is not None:
            options["top_k"] = top_k
        if top_p is not None:
            options["top_p"] = top_p
        if repeat_penalty is not None:
            options["repeat_penalty"] = repeat_penalty

        payload = {
            "model": use_model,
            "messages": messages,
            "stream": stream,
            "keep_alive": ka,
            "options": options,
        }

        start = time.time()
        try:
            # For streaming: _post uses (CONNECT_TIMEOUT, None) — no read timeout
            # For non-streaming: _post uses (CONNECT_TIMEOUT, vision/text_read_timeout)
            if stream:
                resp = self._post("/api/chat", payload, stream=True)
            else:
                read_t = self.vision_read_timeout if is_vision or use_model == self.vision_model else self.text_read_timeout
                resp = self._post("/api/chat", payload, timeout=read_t, stream=False)
        except OllamaModelNotFoundError as exc:
            failed_model = str(exc) if str(exc) not in ("__no_model_configured__", "") else use_model
            if failed_model == "__no_model_configured__":
                raise OllamaError(
                    "No Ollama model is configured. "
                    "Set OLLAMA_TEXT_MODEL or OLLAMA_VISION_MODEL in your .env file."
                )
            if not self.model_exists(failed_model):
                raise OllamaError(
                    f"The configured model '{failed_model}' isn't installed. "
                    f"Run: ollama pull {failed_model}"
                )
            if is_vision and not self.is_vision_capable(failed_model):
                raise OllamaError(
                    f"The configured model '{failed_model}' doesn't support image analysis. "
                    "Please configure a vision-capable model."
                )
            raise OllamaError(
                f"The configured model '{failed_model}' isn't installed. "
                f"Run: ollama pull {failed_model}"
            )
        except (OllamaUnavailableError, OllamaTimeoutError):
            raise

        latency = int((time.time() - start) * 1000)
        if stream:
            logger.debug("[Ollama] Stream opened model=%s connect=%dms", use_model, latency)
            return resp
        data = resp.json()
        content = data.get("message", {}).get("content", "")
        logger.debug("[Ollama] chat latency=%dms model=%s", latency, use_model)
        return content

    def warm_model(self, model_name: str | None = None) -> bool:
        """
        Ping the model with a tiny 1-token inference to ensure it is loaded in
        memory (kept warm via OLLAMA_KEEP_ALIVE). Returns True on success.
        Does not call this method on every request — use at startup / idle time.
        """
        use_model = model_name or self.text_model
        if not use_model:
            return False
        payload = {
            "model": use_model,
            "messages": [{"role": "user", "content": "Hi"}],
            "stream": False,
            "keep_alive": getattr(settings, "OLLAMA_KEEP_ALIVE", "2h"),
            "options": {
                "temperature": 0.0,
                "num_predict": 1,
                "num_ctx": 512,
            },
        }
        try:
            resp = self._post("/api/chat", payload, timeout=30, stream=False)
            _ = resp.json()
            logger.info("[OLLAMA] Warm model %s success (keep_alive=%s)", use_model, getattr(settings, "OLLAMA_KEEP_ALIVE", "2h"))
            return True
        except Exception as exc:
            logger.warning("[OLLAMA] Warm model %s failed: %s", use_model, exc)
            return False

    # ── Backward compat aliases ──────────────────────────────────────────────
    def list_models_names(self): return [m.get("name", "") for m in self.list_models()]

    def is_healthy(self):
        h = self.healthCheck()
        ok = h["ollama"]["connected"] and h["textModel"]["installed"]
        return {
            "provider": "ollama", "status": "healthy" if ok else "unhealthy",
            "model": self.text_model, "embedding_model": self.embedding_model,
            "ollama_url": self.base_url, "local_only": getattr(settings, "LOCAL_AI_ONLY", True),
            "error": None if ok else "Check /api/ai/health for details",
        }

    def embed(self, text: str) -> list[float]:
        payload = {"model": self.embedding_model, "prompt": text}
        try:
            resp = self._post("/api/embeddings", payload, timeout=self.text_read_timeout)
        except OllamaModelNotFoundError:
            raise OllamaModelNotFoundError(self.embedding_model)
        data = resp.json()
        emb = data.get("embedding")
        if not emb:
            raise OllamaError("Ollama returned an empty embedding.")
        return emb

    def visionChat(self, messages, **kw): return self.chat(messages, is_vision=True, **kw)


client = OllamaClient()
