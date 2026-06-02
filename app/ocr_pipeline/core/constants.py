"""Shared constants and compiled patterns for document understanding."""
from __future__ import annotations

import re
from typing import Dict, Tuple

from app.ocr_pipeline.core.block_types import BlockType


SECTION_ALIASES: Dict[str, Tuple[str, ...]] = {
    "summary": ("summary", "profile", "objective", "about me", "professional summary", "profil"),
    "experience": ("experience", "work experience", "employment", "work history", "professional experience", "expérience", "experiences"),
    "education": ("education", "academic", "formation", "academic background"),
    "skills": ("skills", "technical skills", "professional skills", "competences", "compétences", "competencies", "tools"),
    "languages": ("languages", "language", "langues", "langue"),
    "contact": ("contact", "contact information", "details", "coordonnées", "personal details"),
    "strengths": ("strengths", "achievements", "highlights"),
    "hobbies": ("hobbies", "interests"),
    "invoice_items": ("description", "item", "items", "services", "qty", "quantity"),
    "totals": ("subtotal", "tax", "vat", "total", "amount due", "balance due"),
    "legal_clause": ("whereas", "definitions", "term", "confidentiality", "obligations", "governing law"),
}

SECTION_TYPES = {
    "summary": BlockType.SUMMARY_SECTION,
    "experience": BlockType.EXPERIENCE_SECTION,
    "education": BlockType.EDUCATION_SECTION,
    "skills": BlockType.SKILLS_SECTION,
    "languages": BlockType.LANGUAGES_SECTION,
    "contact": BlockType.CONTACT_SECTION,
    "strengths": BlockType.STRENGTHS_SECTION,
    "hobbies": BlockType.HOBBIES_SECTION,
    "invoice_items": BlockType.TABLE_SECTION,
    "totals": BlockType.TOTALS_SECTION,
    "legal_clause": BlockType.LEGAL_CLAUSE,
}

EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+?\d[\d\s().-]{6,}\d)")
URL_RE = re.compile(r"(?:https?://)?(?:www\.)?[A-Za-z0-9.-]+\.[A-Za-z]{2,}(?:/[^\s]*)?")
DATE_RANGE_RE = re.compile(
    r"(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+)?\d{4}"
    r"\s*(?:-|to|\u2013|\u2014)\s*(?:(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+)?(?:\d{4}|Present|Current|Ongoing)",
    re.IGNORECASE,
)
DATE_RE = re.compile(
    r"\b(?:\d{4}[-/]\d{1,2}[-/]\d{1,2}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})\b",
    re.IGNORECASE,
)
BULLET_RE = re.compile(r"^\s*(?:[-*]|\u2022|[0-9]+[.)])\s*")

SECTION_WORDS = {
    "summary",
    "profile",
    "profil",
    "experience",
    "experiences",
    "expérience",
    "work experience",
    "education",
    "formation",
    "skills",
    "compétences",
    "competences",
    "languages",
    "langues",
    "contact",
    "strengths",
    "hobbies",
}

INVOICE_MARKERS = frozenset({
    "invoice", "facture", "total", "subtotal", "vat", "tax", "amount due",
    "balance due", "payment", "فاتورة", "المبلغ", "الضريبة",
})
CV_MARKERS = frozenset({
    "experience", "education", "skills", "curriculum vitae", "resume",
    "expérience", "formation", "compétences", "خبرة", "مهارات", "تعليم",
})
LEGAL_MARKERS = frozenset({
    "agreement", "contract", "parties", "whereas", "obligations",
    "governing law", "contrat", "عقد", "اتفاقية",
})

