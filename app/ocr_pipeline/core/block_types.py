"""Canonical block type constants - single source of truth."""
from __future__ import annotations

from enum import Enum


class BlockType(str, Enum):
    PARAGRAPH = "paragraph"
    HEADING = "heading"
    LIST_ITEM = "list_item"
    KEY_VALUE = "key_value"
    TABLE_SECTION = "table_section"
    EXPERIENCE_SECTION = "experience_section"
    EDUCATION_SECTION = "education_section"
    SKILLS_SECTION = "skills_section"
    LANGUAGES_SECTION = "languages_section"
    CONTACT_SECTION = "contact_section"
    SUMMARY_SECTION = "summary_section"
    STRENGTHS_SECTION = "strengths_section"
    HOBBIES_SECTION = "hobbies_section"
    TOTALS_SECTION = "totals_section"
    LEGAL_CLAUSE = "legal_clause"
    SIGNATURE = "signature"
    HEADER = "header"
    FOOTER = "footer"
    FIGURE = "figure"
    UNKNOWN = "unknown"
    TABLE = "table"
    STAMP = "stamp"
    LINE_ITEMS = "line_items"
    INVOICE_ITEMS_SECTION = "invoice_items_section"

