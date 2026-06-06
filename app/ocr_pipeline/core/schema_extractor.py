"""Deterministic schema extraction from verified semantic OCR blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Sequence
import re
import unicodedata

from app.ocr_pipeline.core.block_types import BlockType
from app.ocr_pipeline.core.constants import BULLET_RE, DATE_RE, DATE_RANGE_RE, EMAIL_RE, PHONE_RE, SECTION_WORDS, URL_RE
from app.ocr_pipeline.core.detector import DocumentInfo
from app.ocr_pipeline.core.ocr_engine import TextBlock
from app.ocr_pipeline.utils.collections import unique_ordered


@dataclass(slots=True)
class ExtractedLine:
    text: str
    block_type: str
    block_id: int
    confidence: float


class StructuredSchemaExtractor:
    """Build strict JSON without inventing values outside OCR text."""

    def extract(self, blocks: Sequence[TextBlock], doc_info: DocumentInfo) -> Dict[str, Any]:
        lines = self._lines(blocks)
        doc_type = doc_info.doc_type
        if doc_type == "cv":
            payload = self._extract_cv(blocks, lines)
        elif doc_type == "invoice":
            payload = self._extract_invoice(blocks, lines)
        elif doc_type in {"contract", "legal"}:
            payload = self._extract_legal(blocks, lines, doc_type)
        else:
            payload = self._extract_generic(blocks, lines, doc_type)
        payload["_source"] = {
            "document_type_confidence": round(float(doc_info.confidence), 4),
            "line_count": len(lines),
            "block_count": len(blocks),
            "strategy": "deterministic_ocr_schema",
        }
        return payload

    def _extract_cv(self, blocks: Sequence[TextBlock], lines: Sequence[ExtractedLine]) -> Dict[str, Any]:
        plain_lines = [line.text for line in lines]
        header = self._header_lines(plain_lines)
        name = self._person_name(header)
        headline = self._headline(header, name)
        contact = self._contact(plain_lines, blocks)

        return {
            "document_type": "cv",
            "person": {
                "name": name,
                "headline": headline,
            },
            "contact": contact,
            "summary": self._section_text(blocks, {BlockType.SUMMARY_SECTION}),
            "skills": self._list_section(blocks, {BlockType.SKILLS_SECTION}),
            "experience": self._experience_entries(blocks),
            "education": self._education_entries(blocks),
            "languages": self._list_section(blocks, {BlockType.LANGUAGES_SECTION}),
            "strengths": self._strengths(blocks),
            "raw_sections": self._raw_sections(blocks),
        }

    def _extract_invoice(self, blocks: Sequence[TextBlock], lines: Sequence[ExtractedLine]) -> Dict[str, Any]:
        plain_lines = [line.text for line in lines]
        key_values = self._key_values(plain_lines)
        invoice_number = self._first_key(key_values, ("invoice no", "invoice number", "invoice", "inv"))
        totals = {
            "subtotal": self._first_key(key_values, ("subtotal", "sub total")),
            "tax": self._first_key(key_values, ("tax", "vat", "gst", "hst")),
            "total": self._first_key(key_values, ("total", "amount due", "balance due", "grand total")),
        }
        return {
            "document_type": "invoice",
            "invoice_number": invoice_number,
            "issue_date": self._first_key(key_values, ("date", "issue date", "invoice date")),
            "due_date": self._first_key(key_values, ("due date",)),
            "seller": self._party_after_label(plain_lines, ("from", "seller", "vendor", "supplier")),
            "buyer": self._party_after_label(plain_lines, ("billed to", "bill to", "customer", "client")),
            "line_items": self._invoice_items(blocks),
            "totals": totals,
            "taxes": {"raw": totals["tax"]},
            "payment": {
                "status": self._first_key(key_values, ("payment status", "status")),
                "terms": self._first_key(key_values, ("payment terms", "terms")),
            },
            "verified_fields": key_values,
        }

    def _extract_legal(self, blocks: Sequence[TextBlock], lines: Sequence[ExtractedLine], doc_type: str) -> Dict[str, Any]:
        plain_lines = [line.text for line in lines]
        title = next((line for line in plain_lines[:10] if len(line.split()) >= 2), None)
        return {
            "document_type": doc_type,
            "title": title,
            "reference": self._match_line(plain_lines, ("reference", "agreement ref", "contract ref")),
            "effective_date": self._first_regex(plain_lines, DATE_RE),
            "parties": self._legal_parties(plain_lines),
            "clauses": [
                {"heading": block.lines[0], "text": block.text}
                for block in blocks
                if block.block_type in {BlockType.LEGAL_CLAUSE, BlockType.PARAGRAPH} and len(block.text) > 80
            ][:24],
            "signatures": [line for line in plain_lines if "signature" in line.lower() or "signed" in line.lower()],
        }

    def _extract_generic(self, blocks: Sequence[TextBlock], lines: Sequence[ExtractedLine], doc_type: str) -> Dict[str, Any]:
        plain_lines = [line.text for line in lines]
        return {
            "document_type": doc_type,
            "title": next((line for line in plain_lines if len(line.split()) >= 2), None),
            "fields": self._key_values(plain_lines),
            "important_lines": plain_lines[:25],
            "raw_sections": self._raw_sections(blocks),
        }

    def _lines(self, blocks: Sequence[TextBlock]) -> List[ExtractedLine]:
        output: List[ExtractedLine] = []
        for block in sorted(blocks, key=lambda item: item.reading_order):
            for index, text in enumerate(block.lines):
                confidence = block.line_confidences[index] if index < len(block.line_confidences) else block.avg_confidence
                output.append(ExtractedLine(text.strip(), block.block_type, block.id, float(confidence)))
        return [line for line in output if line.text]

    def _header_lines(self, lines: Sequence[str]) -> List[str]:
        header: List[str] = []
        for line in lines[:12]:
            if self._is_section_heading(line):
                break
            header.append(line)
        return header

    def _person_name(self, header: Sequence[str]) -> str | None:
        for line in header:
            clean = line.strip()
            if not clean or self._has_contact_signal(clean) or re.search(r"\d", clean):
                continue
            words = clean.split()
            if 2 <= len(words) <= 4 and len(clean) <= 60:
                return clean
        return None

    def _headline(self, header: Sequence[str], name: str | None) -> str | None:
        for line in header:
            clean = line.strip()
            if clean and clean != name and not self._has_contact_signal(clean) and not re.search(r"\d", clean):
                return clean
        return None

    def _contact(self, lines: Sequence[str], blocks: Sequence[TextBlock]) -> Dict[str, Any]:
        emails = self._unique(match.group(0) for line in lines for match in EMAIL_RE.finditer(line))
        phones = self._unique(match.group(0).strip() for line in lines for match in PHONE_RE.finditer(line))
        urls = self._unique(
            match.group(0)
            for line in lines
            if not EMAIL_RE.search(line)
            for match in URL_RE.finditer(line)
        )
        contact_blocks = [block for block in blocks if block.block_type == BlockType.CONTACT_SECTION]
        contact_lines = [line for block in contact_blocks for line in block.lines if not self._is_section_heading(line)]
        location = next(
            (
                line
                for line in contact_lines + list(lines[:12])
                if "," in line and not EMAIL_RE.search(line) and not DATE_RANGE_RE.search(line)
            ),
            None,
        )
        return {
            "email": emails[0] if emails else None,
            "phone": phones[0] if phones else None,
            "urls": urls,
            "location": location,
            "raw_contact_lines": contact_lines,
        }

    def _section_text(self, blocks: Sequence[TextBlock], block_types: set[str]) -> str | None:
        parts = []
        for block in blocks:
            if block.block_type in block_types:
                lines = [line for line in block.lines if not self._is_section_heading(line)]
                parts.extend(lines)
        return "\n".join(parts).strip() or None

    def _list_section(self, blocks: Sequence[TextBlock], block_types: set[str]) -> List[str]:
        values: List[str] = []
        for block in blocks:
            if block.block_type not in block_types:
                continue
            for line in block.lines:
                clean = BULLET_RE.sub("", line).strip()
                if clean and not self._is_section_heading(clean):
                    values.extend(self._split_compact_list(clean))
        return self._unique(values)

    def _strengths(self, blocks: Sequence[TextBlock]) -> List[Dict[str, str]]:
        output = []
        for block in blocks:
            if block.block_type != BlockType.STRENGTHS_SECTION:
                continue
            lines = [line for line in block.lines if not self._is_section_heading(line)]
            i = 0
            while i < len(lines):
                title = lines[i]
                detail = lines[i + 1] if i + 1 < len(lines) else ""
                output.append({"title": title, "description": detail})
                i += 2
        return output

    def _experience_entries(self, blocks: Sequence[TextBlock]) -> List[Dict[str, Any]]:
        lines = []
        for block in blocks:
            if block.block_type == BlockType.EXPERIENCE_SECTION:
                lines.extend(line for line in block.lines if not self._is_section_heading(line))
        return self._entry_blocks(lines, education=False)

    def _education_entries(self, blocks: Sequence[TextBlock]) -> List[Dict[str, Any]]:
        lines = []
        for block in blocks:
            if block.block_type == BlockType.EDUCATION_SECTION:
                lines.extend(line for line in block.lines if not self._is_section_heading(line))
        return self._entry_blocks(lines, education=True)

    def _entry_blocks(self, lines: Sequence[str], education: bool) -> List[Dict[str, Any]]:
        entries: List[Dict[str, Any]] = []
        current: Dict[str, Any] | None = None

        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            date = self._first_regex([line], DATE_RANGE_RE)
            bullet = bool(BULLET_RE.match(line))
            clean_bullet = BULLET_RE.sub("", line).strip()

            organization_key = "institution" if education else "organization"
            current_is_complete = bool(current and current.get("date_range") and current.get(organization_key))
            current_has_body = bool(current and (current.get("description") or current.get("bullets")))
            current_ready_for_next = current_is_complete or current_has_body
            looks_like_title = self._looks_like_entry_title(line, education=education)
            starts_entry = (
                not bullet
                and not date
                and len(line) <= 90
                and (
                    current is None
                    or (current_ready_for_next and looks_like_title)
                )
            )
            if starts_entry:
                if current:
                    entries.append(current)
                current = {
                    "degree" if education else "title": line,
                    "institution" if education else "organization": None,
                    "date_range": None,
                    "location": None,
                    "description": [],
                    "bullets": [],
                }
                continue

            if current is None:
                current = {
                    "degree" if education else "title": None,
                    "institution" if education else "organization": None,
                    "date_range": None,
                    "location": None,
                    "description": [],
                    "bullets": [],
                }

            if date:
                current["date_range"] = date
                location = self._location_fragment(line.replace(date, ""))
                if location:
                    current["location"] = location
            elif bullet:
                current["bullets"].append(clean_bullet)
            elif current.get("institution" if education else "organization") is None:
                current["institution" if education else "organization"] = line
            else:
                current["description"].append(line)

        if current:
            entries.append(current)
        return entries

    def _invoice_items(self, blocks: Sequence[TextBlock]) -> List[Dict[str, Any]]:
        item_blocks = [block for block in blocks if block.block_type in {BlockType.TABLE_SECTION, BlockType.INVOICE_ITEMS_SECTION}]
        rows = []
        for block in item_blocks:
            table_data = block.metadata.get("table_data") if isinstance(block.metadata, dict) else None
            if table_data and table_data.get("col_count", 0) >= 2 and table_data.get("rows"):
                for row in table_data["rows"]:
                    entry: Dict[str, Any] = {}
                    for i, cell in enumerate(row):
                        col_name = table_data["columns"][i].lower().strip() if i < len(table_data["columns"]) else f"col_{i}"
                        entry[col_name] = cell
                    if any(v.strip() for v in entry.values()):
                        entry["raw"] = " | ".join(row)
                        rows.append(entry)
                continue
            for line in block.lines:
                if self._is_section_heading(line):
                    continue
                rows.append({"raw": line})
        return rows

    def _key_values(self, lines: Sequence[str]) -> Dict[str, str]:
        fields: Dict[str, str] = {}
        for line in lines:
            if ":" not in line:
                continue
            label, value = line.split(":", 1)
            key = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_")
            value = value.strip()
            if key and value:
                fields[self._unique_key(fields, key)] = value
        return fields

    def _first_key(self, fields: Dict[str, str], keys: Iterable[str]) -> str | None:
        normalized = {key.replace("_", " "): value for key, value in fields.items()}
        for expected in keys:
            for key, value in normalized.items():
                if key == expected:
                    return value
        for expected in keys:
            for key, value in normalized.items():
                if expected in key:
                    return value
        return None

    def _party_after_label(self, lines: Sequence[str], labels: Iterable[str]) -> Dict[str, Any]:
        lowered = [line.lower() for line in lines]
        for index, lower in enumerate(lowered):
            if any(label in lower for label in labels):
                party_lines = []
                for line in lines[index + 1 : index + 7]:
                    if self._is_section_heading(line):
                        break
                    party_lines.append(line)
                return {"name": party_lines[0] if party_lines else None, "raw_lines": party_lines}
        return {"name": None, "raw_lines": []}

    def _legal_parties(self, lines: Sequence[str]) -> List[str]:
        parties = []
        for line in lines:
            lower = line.lower()
            if "party a" in lower or "party b" in lower or " by and between" in lower or "between:" in lower:
                parties.append(line)
        return parties[:12]

    def _raw_sections(self, blocks: Sequence[TextBlock]) -> List[Dict[str, Any]]:
        return [
            {
                "block_id": block.id,
                "type": block.block_type,
                "text": block.text,
                "confidence": round(float(block.avg_confidence), 4),
            }
            for block in sorted(blocks, key=lambda item: item.reading_order)
        ]

    def _match_line(self, lines: Sequence[str], needles: Iterable[str]) -> str | None:
        for line in lines:
            lower = line.lower()
            if any(needle in lower for needle in needles):
                return line
        return None

    def _first_regex(self, lines: Sequence[str], pattern: re.Pattern[str]) -> str | None:
        for line in lines:
            match = pattern.search(line)
            if match:
                return match.group(0)
        return None

    def _split_compact_list(self, text: str) -> List[str]:
        pieces = re.split(r"\s{2,}|[|;,]", text)
        if len(pieces) > 1:
            return [piece.strip() for piece in pieces if piece.strip()]
        return [text.strip()]

    def _location_fragment(self, text: str) -> str | None:
        cleaned = re.sub(r"^[,;:\s-]+|[,;:\s-]+$", "", text.strip())
        return cleaned or None

    def _looks_like_entry_title(self, text: str, education: bool = False) -> bool:
        words = [word for word in re.split(r"\s+", text.strip()) if word]
        if not words or len(words) > 10:
            return False
        lower = text.lower()
        if "," in text or lower.endswith((".", ";", ":")):
            return False
        if education and re.search(r"\b(bachelor|master|science|degree|licence|diploma|phd|doctor|university)\b", lower):
            return True
        role_words = (
            "designer",
            "engineer",
            "developer",
            "manager",
            "director",
            "consultant",
            "analyst",
            "specialist",
            "assistant",
            "officer",
            "architect",
            "researcher",
            "intern",
            "lead",
            "head",
        )
        action_starters = ("led", "conducted", "collaborated", "created", "built", "managed", "reduced", "increased", "designed")
        if not education and lower.split()[0] in action_starters:
            return False
        title_like = 0
        for word in words:
            stripped = re.sub(r"[^A-Za-zÀ-ÖØ-öø-ÿ]", "", word)
            if not stripped:
                continue
            if stripped.isupper() or stripped[:1].isupper():
                title_like += 1
        ratio = title_like / max(1, len(words))
        if education:
            return ratio >= 0.55
        return any(role in lower for role in role_words) or (ratio >= 0.78 and len(words) <= 5)

    def _is_section_heading(self, line: str) -> bool:
        normalized = self._normalize_label(line)
        return normalized in SECTION_WORDS

    def _normalize_label(self, text: str) -> str:
        decomposed = unicodedata.normalize("NFKD", text.lower())
        asciiish = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
        normalized = re.sub(r"[^a-z0-9 ]+", " ", asciiish)
        return re.sub(r"\s+", " ", normalized).strip()

    def _has_contact_signal(self, line: str) -> bool:
        return bool(EMAIL_RE.search(line) or PHONE_RE.search(line) or "linkedin" in line.lower() or "@" in line)

    def _unique(self, values: Iterable[str]) -> List[str]:
        return unique_ordered(values)

    def _unique_key(self, fields: Dict[str, str], key: str) -> str:
        if key not in fields:
            return key
        suffix = 2
        while f"{key}_{suffix}" in fields:
            suffix += 1
        return f"{key}_{suffix}"

