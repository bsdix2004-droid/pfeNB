"""Lightweight rule-based language fallback.

The production language path lives in :mod:`core.language_identifier`: it tries
fastText first, then CLD3. This module is intentionally dependency-free so OCR
line annotation and tests still work when optional models are missing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Dict


_RTL_LANGS = frozenset({"ar", "fa", "he", "ur", "yi"})
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
_FRENCH_HINTS = {
    "à",
    "â",
    "ç",
    "é",
    "è",
    "ê",
    "ë",
    "î",
    "ï",
    "ô",
    "ù",
    "û",
    "ü",
    "œ",
    "formation",
    "expérience",
    "compétences",
    "facture",
    "montant",
    "paiement",
    "téléphone",
    "adresse",
    "courriel",
}
_ENGLISH_HINTS = {
    "invoice",
    "total",
    "payment",
    "amount",
    "experience",
    "education",
    "skills",
    "profile",
    "phone",
    "email",
    "address",
}
_ARABIC_HINTS = {
    "فاتورة",
    "المبلغ",
    "الدفع",
    "خبرة",
    "مهارات",
    "تعليم",
    "الهاتف",
    "البريد",
}


@dataclass(slots=True)
class LanguageProfile:
    """Result of dependency-free language detection."""

    language: str
    direction: str
    confidence: float
    probabilities: Dict[str, float] = field(default_factory=dict)
    method: str = "rule_based"


class LanguageDetector:
    """Detect broad document language from scripts and common document words."""

    def detect(self, text: str) -> LanguageProfile:
        normalized = " ".join((text or "").strip().split())
        if not normalized:
            return LanguageProfile("unknown", "ltr", 0.0, {"unknown": 1.0})

        scores = self._scores(normalized)
        present = {lang: score for lang, score in scores.items() if score > 0.0}
        if not present:
            return LanguageProfile("unknown", "ltr", 0.0, {"unknown": 1.0})

        total = sum(present.values()) or 1.0
        probabilities = {lang: round(score / total, 4) for lang, score in present.items()}
        ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
        primary, confidence = ordered[0]

        # Mixed documents are common in invoices and CVs. Mark as mixed when a
        # secondary language is genuinely present rather than a tiny OCR artifact.
        if len(ordered) > 1 and ordered[1][1] >= 0.18 and confidence <= 0.82:
            return LanguageProfile("mixed", "ltr", round(confidence, 4), probabilities)

        return LanguageProfile(primary, _direction(primary), round(confidence, 4), probabilities)

    def _scores(self, text: str) -> Dict[str, float]:
        lower = text.lower()
        latin_count = len(_LATIN_RE.findall(text))
        arabic_count = len(_ARABIC_RE.findall(text))
        total_chars = max(1, latin_count + arabic_count)

        scores: Dict[str, float] = {
            "ar": arabic_count / total_chars,
            "fr": 0.0,
            "en": 0.0,
        }
        if latin_count:
            # Default Latin script to English unless French hints dominate.
            scores["en"] += latin_count / total_chars * 0.55
            scores["fr"] += latin_count / total_chars * 0.45

        for hint in _FRENCH_HINTS:
            if hint in lower:
                scores["fr"] += 0.18
        for hint in _ENGLISH_HINTS:
            if hint in lower:
                scores["en"] += 0.16
        for hint in _ARABIC_HINTS:
            if hint in lower:
                scores["ar"] += 0.24

        # Diacritics are a strong French signal for this project's current
        # European CV/invoice use cases.
        if re.search(r"[À-ÖØ-öø-ÿ]", text):
            scores["fr"] += 0.35
        return scores


def detect_language(text: str) -> str:
    """Return the best language code for one text fragment."""

    return LanguageDetector().detect(text).language


def _direction(lang: str) -> str:
    return "rtl" if lang.lower() in _RTL_LANGS else "ltr"

