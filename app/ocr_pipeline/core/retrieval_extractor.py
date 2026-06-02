"""Retrieval-based field extraction over semantic document regions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Sequence

from app.ocr_pipeline.core.constants import DATE_RE, EMAIL_RE, PHONE_RE
from app.ocr_pipeline.core.models.region import SemanticRegion
from app.ocr_pipeline.semantic.retrieval import RegionRetriever

_QUERIES: Dict[str, str] = {
    "total": "invoice total amount due grand total",
    "vat": "VAT tax GST amount percentage",
    "invoice_number": "invoice number reference ID",
    "issue_date": "invoice date issued on",
    "due_date": "payment due date deadline",
    "seller": "seller vendor supplier company from",
    "buyer": "buyer client bill to customer",
    "person_name": "full name person applicant candidate",
    "email": "email address contact",
    "phone": "phone number telephone mobile",
}


@dataclass(slots=True)
class RetrievalExtractionResult:
    fields: Dict[str, Any] = field(default_factory=dict)
    sources: Dict[str, str] = field(default_factory=dict)
    method: str = "retrieval"
    available: bool = False


class RetrievalExtractor:
    """Extract fields by semantic search over indexed document regions."""

    def __init__(self) -> None:
        self._retriever = RegionRetriever()

    def index(self, regions: Sequence[SemanticRegion]) -> None:
        self._retriever.index(regions)

    def extract_invoice(self) -> RetrievalExtractionResult:
        result = RetrievalExtractionResult(available=self._retriever._matrix is not None)
        for field_name in ("total", "vat", "invoice_number", "issue_date", "due_date", "seller", "buyer"):
            value, region_id = self._find_field(field_name)
            if value:
                result.fields[field_name] = value
                result.sources[field_name] = region_id
        return result

    def extract_cv(self) -> RetrievalExtractionResult:
        result = RetrievalExtractionResult(available=self._retriever._matrix is not None)
        for field_name in ("person_name", "email", "phone"):
            value, region_id = self._find_field(field_name)
            if value:
                result.fields[field_name] = value
                result.sources[field_name] = region_id
        return result

    def _find_field(self, field_name: str) -> tuple[Any, str]:
        query = _QUERIES.get(field_name, field_name)
        hits = self._retriever.query(query, top_k=3)
        for region, score in hits:
            if score < 0.25:
                continue
            value = _extract_value(field_name, region.text)
            if value:
                return value, region.id
        return None, ""


def _extract_value(field_name: str, text: str) -> Any:
    """Apply targeted extraction to one region's text."""
    if field_name == "email":
        match = EMAIL_RE.search(text)
        return match.group(0) if match else None
    if field_name == "phone":
        match = PHONE_RE.search(text)
        return match.group(0) if match else None
    if field_name in {"issue_date", "due_date"}:
        match = DATE_RE.search(text)
        return match.group(0) if match else None
    if field_name in {"total", "vat"}:
        import re

        match = re.search(r"[\d,. ]+(?:EUR|USD|GBP|TND|MAD|DZD|€|\$|£)?", text)
        return match.group(0).strip() if match else None
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[0] if lines else None

