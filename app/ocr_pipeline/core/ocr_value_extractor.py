"""OCR-first candidate field extraction.

This layer extracts values directly from verified OCR lines before AI runs.
AI receives these candidates as source-of-truth evidence and may only organize,
validate, or reject them.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Dict, Iterable, List, Sequence

from app.ocr_pipeline.core.ocr_engine import TextBlock


EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,}\d{2,4}")
URL_RE = re.compile(r"(?:https?://)?(?:www\.)[\w.-]+\.[A-Za-z]{2,}\S*", re.IGNORECASE)
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|"
    r"(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{4}|"
    r"\d{4})\b",
    re.IGNORECASE,
)
MONEY_RE = re.compile(r"(?:€|\$|£)?\s*\d[\d\s,.]*(?:€|\$|£|EUR|USD|GBP|TND|MAD|DZD)?", re.IGNORECASE)
KEY_VALUE_RE = re.compile(r"^\s*([^:]{2,80})\s*:\s*(.{1,300})\s*$")


@dataclass(slots=True)
class OCRCandidateField:
    field_name: str
    raw_value: str
    normalized_value: str | None
    confidence: float
    source_ocr_line: str
    evidence_text: str
    bbox: list[float]
    page_number: int
    block_id: int
    block_type: str
    extraction_source: str
    data_type: str = "string"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class OCRValueExtractor:
    """Extract grounded candidate values from semantic OCR blocks."""

    def extract(self, blocks: Sequence[TextBlock], page_number: int = 1) -> List[OCRCandidateField]:
        candidates: List[OCRCandidateField] = []
        for block in blocks:
            for index, line in enumerate(block.lines):
                text = line.strip()
                if not text:
                    continue
                confidence = self._line_confidence(block, index)
                bbox = self._line_bbox(block, index)
                if confidence < 0.70:
                    continue
                candidates.extend(self._extract_line_candidates(block, index, text, confidence, bbox, page_number))
            if block.block_type == "table_section":
                candidates.extend(self._extract_table_candidates(block, page_number))
        return self._dedupe(candidates)

    def _extract_line_candidates(
        self,
        block: TextBlock,
        index: int,
        text: str,
        confidence: float,
        bbox: list[float],
        page_number: int,
    ) -> List[OCRCandidateField]:
        candidates: List[OCRCandidateField] = []
        for match in EMAIL_RE.finditer(text):
            candidates.append(self._candidate("email", match.group(0), confidence, text, bbox, page_number, block, "ocr_pattern", "email"))
        for match in PHONE_RE.finditer(text):
            value = match.group(0).strip()
            if len(re.sub(r"\D", "", value)) >= 7:
                candidates.append(self._candidate("phone", value, confidence, text, bbox, page_number, block, "ocr_pattern", "phone"))
        for match in URL_RE.finditer(text):
            candidates.append(self._candidate("url", match.group(0), confidence, text, bbox, page_number, block, "ocr_pattern", "url"))
        for match in DATE_RE.finditer(text):
            candidates.append(self._candidate("date", match.group(0), confidence, text, bbox, page_number, block, "ocr_pattern", "date"))

        kv = KEY_VALUE_RE.match(text)
        if kv:
            label = self._field_name(kv.group(1))
            value = kv.group(2).strip()
            candidates.append(self._candidate(label, value, confidence, text, bbox, page_number, block, "ocr_key_value", self._data_type(value)))

        lower = text.lower()
        if any(key in lower for key in ("invoice", "facture", "reference", "référence", "ref", "n°", "no.")):
            invoice_id = self._invoice_reference(text)
            if invoice_id:
                candidates.append(self._candidate("invoice_number", invoice_id, confidence, text, bbox, page_number, block, "ocr_pattern", "identifier"))

        money = self._money_value(text)
        if money:
            field_name = self._amount_field_name(text)
            candidates.append(self._candidate(field_name, money, confidence, text, bbox, page_number, block, "ocr_pattern", "amount"))
        return candidates

    def _extract_table_candidates(self, block: TextBlock, page_number: int) -> List[OCRCandidateField]:
        candidates: List[OCRCandidateField] = []
        for index, line in enumerate(block.lines):
            parts = [part.strip() for part in re.split(r"\s{2,}|\t|\|", line) if part.strip()]
            if len(parts) < 2:
                continue
            bbox = self._line_bbox(block, index)
            confidence = self._line_confidence(block, index)
            for part in parts:
                money = self._money_value(part)
                if money:
                    candidates.append(self._candidate("table_amount", money, confidence, line, bbox, page_number, block, "ocr_table", "amount"))
        return candidates

    def _candidate(
        self,
        field_name: str,
        value: str,
        confidence: float,
        evidence: str,
        bbox: list[float],
        page_number: int,
        block: TextBlock,
        source: str,
        data_type: str,
    ) -> OCRCandidateField:
        return OCRCandidateField(
            field_name=field_name,
            raw_value=value.strip(),
            normalized_value=self._normalize(value, data_type),
            confidence=round(float(confidence), 4),
            source_ocr_line=evidence,
            evidence_text=evidence,
            bbox=bbox,
            page_number=page_number,
            block_id=int(block.id),
            block_type=str(block.block_type),
            extraction_source=source,
            data_type=data_type,
        )

    def _field_name(self, label: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
        mapping = {
            "e_mail": "email",
            "mail": "email",
            "telephone": "phone",
            "tel": "phone",
            "tva": "tax",
            "vat": "tax",
            "sub_total": "subtotal",
            "sous_total": "subtotal",
            "grand_total": "total",
            "amount_due": "total",
            "montant_total": "total",
            "invoice_no": "invoice_number",
            "invoice_number": "invoice_number",
            "facture": "invoice_number",
        }
        return mapping.get(normalized, normalized or "field")

    def _amount_field_name(self, text: str) -> str:
        lower = text.lower()
        if "sub" in lower or "sous-total" in lower or "sous total" in lower:
            return "subtotal"
        if "tax" in lower or "vat" in lower or "tva" in lower:
            return "tax"
        if "total" in lower or "montant" in lower or "amount due" in lower:
            return "total"
        return "amount"

    def _invoice_reference(self, text: str) -> str | None:
        match = re.search(r"(?:invoice|facture|ref(?:erence)?|n[°o.]*)\s*[:#-]?\s*([A-Za-z0-9][A-Za-z0-9._/-]{2,})", text, re.I)
        return match.group(1).strip() if match else None

    def _money_value(self, text: str) -> str | None:
        matches = [match.group(0).strip() for match in MONEY_RE.finditer(text)]
        matches = [value for value in matches if len(re.sub(r"\D", "", value)) >= 1]
        return matches[-1] if matches else None

    def _data_type(self, value: str) -> str:
        if EMAIL_RE.fullmatch(value.strip()):
            return "email"
        if URL_RE.fullmatch(value.strip()):
            return "url"
        if MONEY_RE.search(value):
            return "amount"
        if DATE_RE.search(value):
            return "date"
        return "string"

    def _normalize(self, value: str, data_type: str) -> str | None:
        clean = value.strip()
        if data_type == "email":
            return clean.lower()
        if data_type == "phone":
            return re.sub(r"\s+", " ", clean)
        if data_type == "amount":
            number = re.sub(r"[^\d,.-]", "", clean).replace(",", ".")
            if number.count(".") > 1:
                number = number.replace(".", "", number.count(".") - 1)
            return number or None
        return clean

    def _line_confidence(self, block: TextBlock, index: int) -> float:
        if index < len(block.line_confidences):
            return float(block.line_confidences[index])
        return float(block.avg_confidence)

    def _line_bbox(self, block: TextBlock, index: int) -> list[float]:
        if index < len(block.line_bboxes):
            return [float(value) for value in block.line_bboxes[index]]
        return [float(value) for value in block.bbox]

    def _dedupe(self, candidates: Iterable[OCRCandidateField]) -> List[OCRCandidateField]:
        output: List[OCRCandidateField] = []
        seen: set[tuple[str, str, int]] = set()
        for candidate in candidates:
            key = (candidate.field_name, candidate.raw_value.lower(), candidate.page_number)
            if key in seen:
                continue
            seen.add(key)
            output.append(candidate)
        return output
