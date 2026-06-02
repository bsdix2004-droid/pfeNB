"""Optional sentence-level embeddings using sentence-transformers."""
from __future__ import annotations

from typing import List, Optional, Sequence

from app.ocr_pipeline.ocr_config import APP_CONFIG
from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)


class EmbeddingEngine:
    """Lazy-loaded sentence embedding engine."""

    def __init__(self, model_name: str | None = None) -> None:
        self._model_name = model_name or APP_CONFIG.embedding.model_name
        self._model = None
        self._available: Optional[bool] = None

    def embed(self, texts: Sequence[str]) -> List[List[float]]:
        """Return one embedding vector per input text, or empty vectors."""
        model = self._load()
        if model is None:
            return [[] for _ in texts]
        try:
            vectors = model.encode(list(texts), convert_to_numpy=True)
            return [vector.tolist() for vector in vectors]
        except Exception as exc:
            LOGGER.warning("embedding failed: %s", exc)
            return [[] for _ in texts]

    def embed_one(self, text: str) -> List[float]:
        result = self.embed([text])
        return result[0] if result else []

    def is_available(self) -> bool:
        return self._load() is not None

    def _load(self):
        if self._available is False:
            return None
        if self._model is not None:
            return self._model
        if not APP_CONFIG.embedding.enabled:
            self._available = False
            return None
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self._model = SentenceTransformer(self._model_name, local_files_only=True)
            self._available = True
            LOGGER.info("Embedding model loaded: %s", self._model_name)
        except ImportError:
            self._available = False
            LOGGER.warning("sentence-transformers not installed - embeddings disabled")
        except Exception as exc:
            self._available = False
            LOGGER.warning("Could not load embedding model: %s", exc)
        return self._model


_ENGINE: Optional[EmbeddingEngine] = None


def get_engine() -> EmbeddingEngine:
    """Return the process-wide embedding engine."""
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = EmbeddingEngine()
    return _ENGINE

