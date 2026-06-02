"""Dedicated field extraction step after document classification."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence

from app.ocr_pipeline.core.block_types import BlockType
from app.ocr_pipeline.core.detector import DocumentInfo
from app.ocr_pipeline.core.models.region import SemanticRegion
from app.ocr_pipeline.core.ocr_engine import TextBlock
from app.ocr_pipeline.core.retrieval_extractor import RetrievalExtractor
from app.ocr_pipeline.core.schema_extractor import StructuredSchemaExtractor
from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class FieldExtractionResult:
    fields: Dict[str, Any] = field(default_factory=dict)
    field_count: int = 0
    strategy: str = "deterministic"
    warnings: List[str] = field(default_factory=list)
    retrieval_fields: Dict[str, Any] = field(default_factory=dict)

    @property
    def total(self) -> str | None:
        return self.fields.get("total") or (self.fields.get("totals") or {}).get("total") or self.retrieval_fields.get("total")

    @property
    def vat(self) -> str | None:
        return self.fields.get("vat") or (self.fields.get("totals") or {}).get("tax") or self.retrieval_fields.get("vat")

    @property
    def dates(self) -> List[str]:
        raw = self.fields.get("dates") or self.fields.get("issue_date") or self.fields.get("effective_date") or []
        return raw if isinstance(raw, list) else ([raw] if raw else [])

    @property
    def names(self) -> List[str]:
        out: List[str] = []
        person = self.fields.get("person") or {}
        if isinstance(person, dict) and person.get("name"):
            out.append(str(person["name"]))
        for party_key in ("buyer", "seller"):
            party = self.fields.get(party_key) or {}
            if isinstance(party, dict) and party.get("name"):
                out.append(str(party["name"]))
        return out


class FieldExtractor:
    """Run deterministic field extraction appropriate to the document type."""

    def __init__(self) -> None:
        self._schema = StructuredSchemaExtractor()
        self._retrieval = RetrievalExtractor()

    def extract(
        self,
        blocks: Sequence[TextBlock],
        doc_info: DocumentInfo,
    ) -> FieldExtractionResult:
        try:
            # 1. Deterministic extraction (current primary)
            raw = self._schema.extract(blocks, doc_info)
            count = _count_non_null(raw)
            
            # 2. Retrieval extraction (new semantic path)
            retrieval_data = {}
            try:
                regions = [SemanticRegion.from_text_block(b) for b in blocks]
                self._retrieval.index(regions)
                if doc_info.doc_type == "invoice":
                    retrieval_res = self._retrieval.extract_invoice()
                    retrieval_data = retrieval_res.fields
                elif doc_info.doc_type == "cv":
                    retrieval_res = self._retrieval.extract_cv()
                    retrieval_data = retrieval_res.fields
            except Exception as e:
                LOGGER.warning("Retrieval extraction failed: %s", e)

            return FieldExtractionResult(
                fields=raw,
                field_count=count,
                strategy="deterministic+retrieval" if retrieval_data else "deterministic",
                retrieval_fields=retrieval_data,
            )
        except Exception as exc:
            LOGGER.warning("field extraction failed for type=%s: %s", doc_info.doc_type, exc)
            return FieldExtractionResult(warnings=[str(exc)])


def _count_non_null(data: Any, depth: int = 0) -> int:
    if depth > 5:
        return 0
    if isinstance(data, dict):
        return sum(_count_non_null(value, depth + 1) for value in data.values())
    if isinstance(data, list):
        return sum(_count_non_null(value, depth + 1) for value in data)
    return 1 if data not in (None, "", [], {}) else 0

