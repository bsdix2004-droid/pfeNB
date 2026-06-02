"""Rule-based document type classification."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import cv2
import numpy as np

from app.config import get_settings, Settings
from app.ocr_pipeline.core.constants import CV_MARKERS, INVOICE_MARKERS, LEGAL_MARKERS
from app.ocr_pipeline.utils.logger import get_logger


LOGGER = get_logger(__name__)

PATTERNS: Dict[str, List[str]] = {
    "cv": [
        *sorted(CV_MARKERS),
        "experience",
        "education",
        "skills",
        "work history",
        "curriculum vitae",
        "resume",
        "profile",
        "expérience",
        "formation",
        "compétences",
        "خبرة",
        "مهارات",
        "تعليم",
        "السيرة الذاتية",
    ],
    "invoice": [
        *sorted(INVOICE_MARKERS),
        "invoice",
        "total",
        "amount due",
        "payment",
        "subtotal",
        "vat",
        "tax",
        "facture",
        "montant",
        "tva",
        "devis",
        "فاتورة",
        "المبلغ الإجمالي",
        "الدفع",
    ],
    "contract": [
        *sorted(LEGAL_MARKERS),
        "agreement",
        "terms and conditions",
        "signed",
        "party",
        "parties",
        "contract",
        "contrat",
        "signature",
        "عقد",
        "اتفاقية",
        "الطرف",
    ],
    "id_card": [
        "national id",
        "identity card",
        "date of birth",
        "nationality",
        "carte d'identité",
        "cin",
        "رقم الهوية",
        "تاريخ الميلاد",
        "الجنسية",
    ],
    "form": [
        "please fill",
        "checkbox",
        "signature required",
        "form",
        "application",
        "formulaire",
        "استمارة",
        "يرجى تعبئة",
    ],
    "signature": [
        "signature",
        "signed by",
        "authorized by",
        "signatory",
        "signature requise",
        "signé par",
        "approved by",
        "توقيع",
        "الإمضاء",
        "ختم",
    ],
    "legal": [
        *sorted(LEGAL_MARKERS),
        "court",
        "article",
        "regulation",
        "law",
        "judgment",
        "tribunal",
        "loi",
        "المحكمة",
        "قانون",
        "مرسوم",
    ],
    "administrative": [
        "ministry",
        "department",
        "official document",
        "reference number",
        "certificate",
        "ministère",
        "attestation",
        "وزارة",
        "إدارة",
        "وثيقة رسمية",
    ],
}


@dataclass(slots=True)
class ClassificationResult:
    doc_type: str
    confidence: float
    scores: Dict[str, float] = field(default_factory=dict)
    evidence: Dict[str, List[str]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        return {
            "document_type": self.doc_type,
            "confidence": round(self.confidence, 4),
            "scores": {key: round(value, 4) for key, value in self.scores.items()},
            "evidence": self.evidence,
        }


def classify_document(full_text: str) -> str:
    text_lower = full_text.lower()
    scores = {doc_type: 0 for doc_type in PATTERNS}

    for doc_type, keywords in PATTERNS.items():
        for keyword in keywords:
            if keyword.lower() in text_lower:
                scores[doc_type] += 1

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "unknown"


class DocumentClassifier:
    """Thin wrapper used by the pipeline's detector."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.OCR_PIPELINE

    def classify(self, text: str, language: str = "unknown", image: object | None = None) -> ClassificationResult:
        text_lower = text.lower()
        raw_scores: Dict[str, float] = {}
        evidence: Dict[str, List[str]] = {}

        for doc_type, keywords in PATTERNS.items():
            hits = [keyword for keyword in keywords if keyword.lower() in text_lower]
            raw_scores[doc_type] = float(len(hits))
            if hits:
                evidence[doc_type] = hits[:8]

        visual_scores, visual_evidence = self._visual_scores(image)
        for doc_type, score in visual_scores.items():
            raw_scores[doc_type] = raw_scores.get(doc_type, 0.0) + score
            if visual_evidence.get(doc_type):
                evidence.setdefault(doc_type, []).extend(visual_evidence[doc_type])

        if not raw_scores or max(raw_scores.values()) <= 0:
            return ClassificationResult("unknown", 0.0, raw_scores, evidence)

        total = sum(raw_scores.values()) or 1.0
        scores = {key: value / total for key, value in raw_scores.items()}
        doc_type = max(scores, key=scores.get)
        return ClassificationResult(doc_type, scores[doc_type], scores, evidence)

    def _visual_scores(self, image: object | None) -> tuple[Dict[str, float], Dict[str, List[str]]]:
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return {}, {}

        scores: Dict[str, float] = {}
        evidence: Dict[str, List[str]] = {}
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
            height, width = gray.shape[:2]
            if height <= 0 or width <= 0:
                return {}, {}

            if image.ndim == 3 and self._has_colored_sidebar(image):
                scores["cv"] = scores.get("cv", 0.0) + 1.25
                evidence.setdefault("cv", []).append("colored_sidebar_layout")

            line_density = self._line_density(gray)
            if line_density > 0.018:
                scores["invoice"] = scores.get("invoice", 0.0) + 0.6
                scores["form"] = scores.get("form", 0.0) + 0.6
                evidence.setdefault("invoice", []).append("table_or_form_lines")
                evidence.setdefault("form", []).append("table_or_form_lines")
        except Exception as exc:
            LOGGER.warning("visual document classification failed: %s", exc)
            return {}, {}
        return scores, evidence

    def _has_colored_sidebar(self, image: np.ndarray) -> bool:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        height, width = hsv.shape[:2]
        left = hsv[:, : max(1, int(width * 0.34))]
        body = hsv[:, int(width * 0.40) :]
        if left.size == 0 or body.size == 0:
            return False
        left_sat = float(np.mean(left[:, :, 1]))
        body_sat = float(np.mean(body[:, :, 1]))
        left_val = float(np.mean(left[:, :, 2]))
        body_val = float(np.mean(body[:, :, 2]))
        return (left_sat - body_sat > 18.0 or abs(left_val - body_val) > 28.0) and left_sat > 22.0

    def _line_density(self, gray: np.ndarray) -> float:
        edges = cv2.Canny(gray, 60, 160)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(12, gray.shape[1] // 35), 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, gray.shape[0] // 45)))
        horizontal = cv2.morphologyEx(edges, cv2.MORPH_OPEN, horizontal_kernel)
        vertical = cv2.morphologyEx(edges, cv2.MORPH_OPEN, vertical_kernel)
        return float(np.mean((horizontal > 0) | (vertical > 0)))

