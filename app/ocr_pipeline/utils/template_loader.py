"""Small cached loader for repository HTML templates."""
from __future__ import annotations

from pathlib import Path
from typing import Dict

from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)


class TemplateLoader:
    """Load HTML templates from the templates directory."""

    _cache: Dict[str, str] = {}
    _dir = Path(__file__).resolve().parents[1] / "templates"

    @classmethod
    def load(cls, name: str) -> str:
        if name not in cls._cache:
            try:
                cls._cache[name] = (cls._dir / name).read_text(encoding="utf-8")
            except Exception as exc:
                LOGGER.warning("Could not load template %s: %s", name, exc)
                cls._cache[name] = "<html><body>Template missing</body></html>"
        return cls._cache[name]

