"""
Centralized embedding generation using the local Ollama embedding model.
Import get_embedding() anywhere you need a vector — do not call ollama_client directly.
"""

import logging
from .ollama_client import client, OllamaError

logger = logging.getLogger(__name__)


def get_embedding(text: str) -> list[float] | None:
    """
    Generate an embedding for `text` using the configured local Ollama model.

    Returns:
        A list of floats, or None if the model is unavailable.
    """
    if not text or not text.strip():
        return None
    try:
        # Truncate to avoid exceeding context window for embedding models
        truncated = text[:8000]
        return client.embed(truncated)
    except OllamaError as exc:
        logger.error("Embedding generation failed: %s", exc)
        return None
