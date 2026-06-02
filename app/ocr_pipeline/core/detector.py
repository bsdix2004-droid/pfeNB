"""Document type and language detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from app.config import get_settings, Settings
from app.ocr_pipeline.classifiers.document_classifier import ClassificationResult, DocumentClassifier
from app.ocr_pipeline.core.language_identifier import LanguageIdentificationResult, LanguageIdentifier


@dataclass(slots=True)
class DocumentInfo:
    doc_type: str
    language: str
    direction: str
    confidence: float
    language_probabilities: Dict[str, float] = field(default_factory=dict)
    classification_scores: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, List[str]] = field(default_factory=dict)


class DocumentDetector:
    """Detect document classification and linguistic direction."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.OCR_PIPELINE
        self.language_identifier = LanguageIdentifier(self.config.language)
        self.classifier = DocumentClassifier(self.settings)

    def detect(self, image: np.ndarray | None, raw_text: str) -> DocumentInfo:
        if not raw_text or not raw_text.strip():
            return DocumentInfo("unknown", "unknown", "ltr", 0.0)

        lang_res = self.language_identifier.identify(raw_text)
        classification = self.classifier.classify(raw_text, lang_res.primary_language, image)

        doc_type = classification.doc_type
        confidence = classification.confidence
        
        # Fallback to unknown if confidence is too low
        min_conf = self.config.detector.min_confidence if self.config else 0.20
        if confidence < min_conf:
            doc_type = "unknown"

        return DocumentInfo(
            doc_type,
            lang_res.primary_language,
            lang_res.direction,
            confidence,
            language_probabilities=lang_res.probabilities,
            classification_scores=classification.scores,
            evidence=classification.evidence,
        )

    def detect_with_language(
        self,
        image: np.ndarray | None,
        raw_text: str,
        lang_result: LanguageIdentificationResult,
    ) -> DocumentInfo:
        """
        Classification-only path that reuses already computed language data.
        """
        if not raw_text or not raw_text.strip():
            return DocumentInfo("unknown", lang_result.primary_language, lang_result.direction, 0.0)
        classification = self.classifier.classify(raw_text, lang_result.primary_language, image)
        return DocumentInfo(
            classification.doc_type,
            lang_result.primary_language,
            lang_result.direction,
            classification.confidence,
            language_probabilities=lang_result.probabilities,
            classification_scores=classification.scores,
            evidence=classification.evidence,
        )

    def _detect_language(self, text: str) -> Tuple[str, str]:
        res = self.language_identifier.identify(text)
        return res.primary_language, res.direction

    def _detect_type(self, text: str, lang: str) -> Tuple[str, float]:
        result: ClassificationResult = self.classifier.classify(text, lang)
        return result.doc_type, result.confidence

