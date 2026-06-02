"""Semantic retrieval over embedded document regions."""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from app.ocr_pipeline.ocr_config import APP_CONFIG
from app.ocr_pipeline.core.models.region import SemanticRegion
from app.ocr_pipeline.semantic.embeddings import EmbeddingEngine, get_engine
from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)


class RegionRetriever:
    """Build a searchable index over SemanticRegion objects."""

    def __init__(self, engine: Optional[EmbeddingEngine] = None) -> None:
        self._engine = engine or get_engine()
        self._regions: List[SemanticRegion] = []
        self._matrix: Optional[np.ndarray] = None
        self._faiss_index = None

    def index(self, regions: Sequence[SemanticRegion]) -> None:
        """Embed all regions and build the search index."""
        self._regions = list(regions)
        texts = [region.text for region in self._regions]

        if not self._engine.is_available() or not texts:
            LOGGER.warning("Embeddings unavailable - retrieval will return empty results")
            self._matrix = None
            return

        vectors = self._engine.embed(texts)
        if not any(vectors):
            self._matrix = None
            return

        matrix = np.array(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self._matrix = matrix / norms

        if len(self._regions) > APP_CONFIG.embedding.faiss_threshold:
            self._build_faiss()

    def query(self, text: str, top_k: int = 5) -> List[Tuple[SemanticRegion, float]]:
        """Return top_k (region, score) pairs sorted by descending similarity."""
        if self._matrix is None or not self._regions:
            return []

        q_vec = self._engine.embed_one(text)
        if not q_vec:
            return []

        q = np.array(q_vec, dtype=np.float32)
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm

        if self._faiss_index is not None:
            return self._faiss_query(q, top_k)

        return self._numpy_query(q, top_k)

    def _numpy_query(self, q: np.ndarray, top_k: int) -> List[Tuple[SemanticRegion, float]]:
        scores = self._matrix @ q  # type: ignore[operator]
        top_idx = np.argsort(scores)[::-1][:top_k]
        return [(self._regions[int(index)], float(scores[int(index)])) for index in top_idx]

    def _build_faiss(self) -> None:
        try:
            import faiss  # type: ignore

            dim = self._matrix.shape[1]  # type: ignore[union-attr]
            index = faiss.IndexFlatIP(dim)
            index.add(self._matrix)  # type: ignore[union-attr]
            self._faiss_index = index
            LOGGER.info("FAISS index built (%d regions)", len(self._regions))
        except ImportError:
            return
        except Exception as exc:
            LOGGER.warning("FAISS index build failed, falling back to numpy: %s", exc)

    def _faiss_query(self, q: np.ndarray, top_k: int) -> List[Tuple[SemanticRegion, float]]:
        try:
            scores, indices = self._faiss_index.search(q.reshape(1, -1), top_k)  # type: ignore[union-attr]
            return [
                (self._regions[int(index)], float(score))
                for score, index in zip(scores[0], indices[0])
                if index >= 0
            ]
        except Exception as exc:
            LOGGER.warning("FAISS query failed, falling back to numpy: %s", exc)
            self._faiss_index = None
            return self._numpy_query(q, top_k)

