"""Language detection helpers used by OCR and semantic reconstruction."""

from __future__ import annotations

from app.ocr_pipeline.language.detector import LanguageDetector, LanguageProfile, detect_language

__all__ = ["LanguageDetector", "LanguageProfile", "detect_language"]

