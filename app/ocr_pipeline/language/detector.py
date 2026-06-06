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
_CJK_RE = re.compile(r"[\u4e00-\u9fff\u3000-\u303f]")
_LATIN_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]")
_FRENCH_HINTS = {
    "à", "â", "ç", "é", "è", "ê", "ë", "î", "ï", "ô", "ù", "û", "ü", "œ",
    "formation", "expérience", "compétences", "facture", "montant",
    "paiement", "téléphone", "adresse", "courriel",
}
_ENGLISH_HINTS = {
    "invoice", "total", "payment", "amount", "experience",
    "education", "skills", "profile", "phone", "email", "address",
}
_ARABIC_HINTS = {
    "فاتورة", "المبلغ", "الدفع", "خبرة", "مهارات", "تعليم", "الهاتف", "البريد",
}
_CHINESE_HINTS = {
    "发票", "简历", "教育", "经验", "技能", "电话", "邮箱", "地址", "总额",
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
    """Detect document language from scripts and common document words.

    Returns the single best language (fr / en / ar / zh) — never "mixed".
    """

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

        return LanguageProfile(primary, _direction(primary), round(confidence, 4), probabilities)

    def _scores(self, text: str) -> Dict[str, float]:
        lower = text.lower()
        latin_count = len(_LATIN_RE.findall(text))
        arabic_count = len(_ARABIC_RE.findall(text))
        cjk_count = len(_CJK_RE.findall(text))
        total_script = max(1, latin_count + arabic_count + cjk_count)

        scores: Dict[str, float] = {
            "ar": arabic_count / total_script,
            "fr": 0.0,
            "en": 0.0,
            "zh": cjk_count / total_script,
        }

        if latin_count:
            scores["en"] += latin_count / total_script * 0.55
            scores["fr"] += latin_count / total_script * 0.45

        for hint in _FRENCH_HINTS:
            if hint in lower:
                scores["fr"] += 0.18
        for hint in _ENGLISH_HINTS:
            if hint in lower:
                scores["en"] += 0.16
        for hint in _ARABIC_HINTS:
            if hint in lower:
                scores["ar"] += 0.24
        for hint in _CHINESE_HINTS:
            if hint in lower:
                scores["zh"] += 0.20

        if re.search(r"[À-ÖØ-öø-ÿ]", text):
            scores["fr"] += 0.35

        return scores


def detect_language(text: str) -> str:
    """Return the best language code for one text fragment."""

    return LanguageDetector().detect(text).language


def _direction(lang: str) -> str:
    return "rtl" if lang.lower() in _RTL_LANGS else "ltr"

