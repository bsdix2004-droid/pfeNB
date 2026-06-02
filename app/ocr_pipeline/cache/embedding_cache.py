"""Simple disk-backed embedding cache keyed by SHA-256 of input text."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Optional

from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)
_CACHE_DIR = Path("output") / ".embedding_cache"


class EmbeddingCache:
    """Persist and retrieve embedding vectors by input-text hash."""

    def __init__(self, cache_dir: Path = _CACHE_DIR) -> None:
        self._dir = cache_dir
        self._dir.mkdir(parents=True, exist_ok=True)

    def get(self, text: str) -> Optional[List[float]]:
        path = self._path(text)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            LOGGER.warning("Cache read failed: %s", exc)
            return None

    def set(self, text: str, embedding: List[float]) -> None:
        try:
            self._path(text).write_text(json.dumps(embedding, ensure_ascii=False), encoding="utf-8")
        except Exception as exc:
            LOGGER.warning("Cache write failed: %s", exc)

    def _path(self, text: str) -> Path:
        key = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return self._dir / f"{key}.json"

