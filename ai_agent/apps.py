from django.apps import AppConfig


class AiAgentConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'ai_agent'

    def ready(self):
        import sys
        import threading
        if 'runserver' not in sys.argv and 'daphne' not in sys.argv[0]:
            return
        def startup_check():
            import time
            time.sleep(2)
            from ai.services.ollama_client import client
            import logging
            logger = logging.getLogger(__name__)
            try:
                h = client.healthCheck()
                logger.info("\nVISION AI\n" + "-"*24)
                logger.info("Ollama:        %s %s", "✓ Connected" if h["ollama"]["connected"] else "✗ Offline", h["ollama"]["baseUrl"])
                logger.info("Text Model:    %s %s", "✓" if h["textModel"]["installed"] else "✗", h["textModel"]["name"] or "(not configured)")
                vm = h["visionModel"]
                if not vm["configured"]:
                    logger.info("Vision Model:  ○ Not configured (set OLLAMA_VISION_MODEL)")
                elif vm["installed"] and vm["capable"]:
                    logger.info("Vision Model:  ✓ %s", vm["name"])
                elif vm["installed"]:
                    logger.info("Vision Model:  ✗ %s not vision-capable", vm["name"])
                else:
                    logger.info("Vision Model:  ✗ %s not installed (ollama pull %s)", vm["name"], vm["name"])
                logger.info("-"*24)
                ready = h["ollama"]["connected"] and h["textModel"]["installed"] and (not vm["configured"] or (vm["installed"] and vm["capable"]))
                logger.info("Vision:        %s", "✓ Ready" if ready else "○ Unavailable")
                # Warmup text model only if installed — use tiny 1-token warm to avoid 30s stall
                if h["textModel"]["installed"]:
                    try:
                        logger.info("Warming up Ollama text model into VRAM (keep_alive=%s)...", h.get("config", {}).get("keep_alive", "2h"))
                        # Use warm_model with 1 token, 512 ctx — completes in <2s even on CPU
                        ok = client.warm_model()
                        if ok:
                            logger.info("Ollama warmup complete — model kept warm.")
                        else:
                            # Fallback tiny chat
                            client.chat([{"role": "user", "content": "Hi"}], temperature=0.0, num_predict=1, num_ctx=512)
                            logger.info("Ollama warmup complete (fallback).")
                    except Exception as exc:
                        logger.warning(f"Ollama warmup failed: {exc}")
                # Warm code model if different (fast code needs separate warm)
                try:
                    from django.conf import settings as _s
                    code_model = getattr(_s, "OLLAMA_CODE_MODEL", "") or getattr(_s, "OLLAMA_TEXT_MODEL", "")
                    text_model = getattr(_s, "OLLAMA_TEXT_MODEL", "")
                    if code_model and code_model != text_model and client.model_exists(code_model):
                        logger.info("Warming up Ollama code model %s into VRAM...", code_model)
                        ok2 = client.warm_model(code_model)
                        logger.info("Code model warmup: %s", "✓" if ok2 else "✗")
                except Exception as exc:
                    logger.warning(f"Code model warmup failed: {exc}")
            except Exception as exc:
                logger.warning(f"VISION startup check failed: {exc}")
        threading.Thread(target=startup_check, daemon=True).start()
