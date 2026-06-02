"""Clean client-facing JSON exports.

The review layer keeps bboxes, confidence, source providers, and warnings. This
exporter intentionally emits only useful business data, while keeping grounding
evidence available to callers that need audit output.
"""

from __future__ import annotations

from dataclasses import asdict
import json
import re
from typing import Any, Dict, Iterable, List, Sequence

from app.ocr_pipeline.ocr_config import APP_CONFIG
from app.ocr_pipeline.core.detector import DocumentInfo
from app.ocr_pipeline.core.models import DocumentSource, ExtractionEvidence, ExtractionField, FinalExport, TableBlock
from app.ocr_pipeline.core.ocr_engine import TextBlock
from app.ocr_pipeline.core.structurer import parse_and_validate
from app.ocr_pipeline.core.llm_client import OllamaGenerateClient
from app.ocr_pipeline.validation.grounding import GroundingChecker


EMAIL_RE = re.compile(r"[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:\+\d{1,3}[\s.-]?)?(?:\(?\d{2,4}\)?[\s.-]?){2,}\d{2,4}")
URL_RE = re.compile(r"(?:https?://)?(?:www\.)[\w.-]+\.[A-Za-z]{2,}\S*")
DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}|"
    r"(?:janvier|février|fevrier|mars|avril|mai|juin|juillet|août|aout|septembre|octobre|novembre|décembre|decembre|"
    r"january|february|march|april|may|june|july|august|september|october|november|december)"
    r"\s+\d{4})\b",
    re.IGNORECASE,
)
DATE_RANGE_RE = re.compile(
    rf"{DATE_RE.pattern}\s*(?:-|–|—|to|à|au|jusqu.?à)\s*(?:{DATE_RE.pattern}|en poste actuellement|present|currently)",
    re.IGNORECASE,
)
MONEY_RE = re.compile(r"(?:€|\$|£)?\s*\d[\d\s,.]*(?:€|\$|£|EUR|USD|GBP|TND|MAD|DZD)?", re.IGNORECASE)


class CleanJSONExporter:
    """Create clean, grounded JSON exports from normalized documents or OCR results."""

    FIELD_LABELS: Dict[str, str] = {
        "personal_info.name": "Full Name",
        "personal_info.contact.phone": "Phone",
        "personal_info.contact.email": "Email",
        "personal_info.contact.urls": "URLs",
        "personal_info.hobbies": "Hobbies",
        "education.formation": "Education",
        "work_experience": "Work Experience",
        "skills": "Skills",
        "languages": "Languages",
        "certifications": "Certifications",
        "projects": "Projects",
        "vendor.name": "Vendor Name",
        "vendor.attention_of": "Attention To",
        "client.phone": "Client Phone",
        "client.email": "Client Email",
        "client.address": "Client Address",
        "items": "Line Items",
        "totals.subtotal": "Subtotal",
        "totals.tax": "Tax / VAT",
        "totals.total": "Total",
        "payment.method": "Payment Method",
        "payment.conditions": "Payment Terms",
        "payment.account_number": "Account Number",
        "payment.due_date": "Due Date",
        "title": "Document Title",
        "contacts": "Contact Information",
        "dates": "Dates",
        "amounts": "Amounts",
        "key_values": "Key-Value Fields",
        "tables": "Tables",
        "sections": "Sections",
    }

    def export_document(self, document: DocumentSource) -> FinalExport:
        doc_type = (document.document_type or "unknown").lower()
        if doc_type == "cv":
            return self._export_cv(document)
        if doc_type == "invoice":
            return self._export_invoice(document)
        return self._export_generic(document)

    def export_pipeline_result(self, result: Any) -> FinalExport:
        document = self._document_from_pipeline_result(result)
        return self.export_document(document)

    def _export_cv(self, document: DocumentSource) -> FinalExport:
        lines = _nonempty_lines(document.text)
        sections = _sections(lines)
        contact = self._contact(lines)
        name = self._candidate_name(lines, contact)
        data = {
            "personal_info": {
                "name": name,
                "contact": contact,
                "hobbies": self._list_from_sections(sections, ("hobbies", "centres d'intérêt", "interests")),
            },
            "education": {
                "formation": self._education_entries(sections),
            },
            "work_experience": self._work_entries(sections),
            "skills": {item.lower(): True for item in self._list_from_sections(sections, ("skills", "compétences", "competences"))},
            "languages": self._language_entries(sections),
            "certifications": self._certification_entries(sections),
            "projects": self._project_entries(sections),
        }
        evidence = self._evidence_for_data(data, document)
        return FinalExport(
            document_type="CV",
            document_name=document.document_name,
            data=data,
            fields=self._build_fields(data, evidence),
            evidence=evidence,
            warnings=list(document.warnings),
        )

    def _export_invoice(self, document: DocumentSource) -> FinalExport:
        lines = _nonempty_lines(document.text)
        key_values = _key_values(lines)
        contact = self._contact(lines)
        items = self._invoice_items(document.tables if hasattr(document, "tables") else _document_tables(document))
        totals = self._invoice_totals(lines, items, key_values)
        data = {
            "vendor": {
                "name": self._party_name(lines, ("vendor", "seller", "from", "émetteur", "fournisseur")) or self._first_company_like(lines),
                "attention_of": self._line_containing(lines, ("attention", "à l'attention", "a l'attention")),
            },
            "client": {
                "phone": contact.get("phone"),
                "email": contact.get("email"),
                "address": self._address(lines),
            },
            "items": items,
            "totals": totals,
            "payment": {
                "method": self._payment_line(lines),
                "conditions": self._first_value(key_values, ("conditions", "terms", "payment_terms", "conditions_de_paiement")),
                "account_number": self._account_number(lines),
                "due_date": self._first_value(key_values, ("due_date", "date_d_echeance", "échéance", "echeance"))
                or self._line_containing(lines, ("sous 30 jours", "due", "échéance", "echeance")),
            },
        }
        evidence = self._evidence_for_data(data, document)
        return FinalExport(
            document_type="invoice",
            data=data,
            fields=self._build_fields(data, evidence),
            evidence=evidence,
            warnings=list(document.warnings),
        )

    def _export_generic(self, document: DocumentSource) -> FinalExport:
        ai_data = self._try_ai_inferred_export(document)
        data = ai_data or self._grounded_generic_data(document)
        grounded = GroundingChecker().verify(data, document.text)
        warnings = list(document.warnings)
        if not grounded["is_grounded"]:
            data = _remove_ungrounded(data, grounded["field_status"])
            warnings.append("Some inferred export fields were removed because they were not grounded in extracted text.")
        evidence = self._evidence_for_data(data, document)
        return FinalExport(
            document_type=document.document_type or "unknown",
            document_name=document.document_name,
            data=data,
            fields=self._build_fields(data, evidence),
            evidence=evidence,
            warnings=warnings,
        )

    def _document_from_pipeline_result(self, result: Any) -> DocumentSource:
        ocr = getattr(result, "ocr_data", {}) or {}
        document = DocumentSource(
            source_file=str(getattr(result, "source_file", "")),
            file_type="image",
            document_name=_stem(str(getattr(result, "source_file", ""))),
            language=ocr.get("language", getattr(getattr(result, "doc_info", None), "language", "unknown")),
            direction=getattr(getattr(result, "doc_info", None), "direction", "ltr"),
            document_type=ocr.get("document_type", getattr(getattr(result, "doc_info", None), "doc_type", "unknown")),
            text=ocr.get("full_text", ""),
            warnings=list(getattr(result, "errors", []) or []),
            metadata={"source": "ocr_pipeline", "output_dir": getattr(result, "output_dir", "")},
        )
        tables = []
        for block in ocr.get("semantic_blocks", []) or []:
            if not isinstance(block, dict):
                continue
            if str(block.get("block_type", "")).lower() == "table_section":
                rows = [[line] for line in block.get("lines", []) or []]
                tables.append(TableBlock(page=1, rows=rows, bbox=list(block.get("bbox", [])), source="ocr_semantic_table"))
        document.metadata["tables"] = [asdict(table) for table in tables]
        return document

    def _build_fields(self, data: Dict[str, Any], evidence: List[ExtractionEvidence]) -> List[ExtractionField]:
        evidence_map: Dict[str, ExtractionEvidence] = {}
        for e in evidence:
            evidence_map[e.field_path] = e
        fields: List[ExtractionField] = []
        for path, value in _flatten(data).items():
            if value in (None, "", [], {}):
                continue
            ev = evidence_map.get(path)
            status = "verified" if ev is not None else "suggested"
            fields.append(
                ExtractionField(
                    id=path,
                    label=self.FIELD_LABELS.get(path, path.replace("_", " ").replace(".", " > ").title()),
                    value=value,
                    confidence=ev.confidence if ev else 0.5,
                    source=ev.source if ev else "deterministic",
                    status=status,
                    editable=True,
                    bbox=list(ev.bbox) if ev and ev.bbox else [],
                    alternatives=[],
                )
            )
        return fields

    @property
    def tables(self) -> List[TableBlock]:
        return []

    def _contact(self, lines: Sequence[str]) -> Dict[str, Any]:
        text = "\n".join(lines)
        email = _first_match(EMAIL_RE, text)
        phone = _first_match(PHONE_RE, text)
        urls = _unique(URL_RE.findall(text))
        contact: Dict[str, Any] = {
            "phone": phone,
            "email": email,
        }
        skype = next((line for line in lines if "skype" in line.lower()), None)
        if skype:
            contact["skype"] = skype
        if urls:
            contact["urls"] = urls
        return contact

    def _candidate_name(self, lines: Sequence[str], contact: Dict[str, Any]) -> str | None:
        for line in lines[:12]:
            if EMAIL_RE.search(line) or PHONE_RE.search(line) or "linkedin" in line.lower():
                continue
            words = line.split()
            if 2 <= len(words) <= 4 and len(line) <= 70 and not any(char.isdigit() for char in line):
                return line
        return None

    def _list_from_sections(self, sections: Dict[str, List[str]], labels: Sequence[str]) -> List[str]:
        values: List[str] = []
        for label in labels:
            for line in sections.get(_norm(label), []):
                values.extend(_split_list_line(line))
        return _unique(value for value in values if not _looks_like_heading(value))

    def _education_entries(self, sections: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        lines = _section_lines(sections, ("education", "formation", "études", "etudes"))
        entries = []
        for group in _entry_groups(lines):
            date_values = _date_ranges(group)
            entry_lines = [line for line in group if line not in date_values]
            entries.append(
                {
                    "name": entry_lines[0] if entry_lines else None,
                    "university": entry_lines[1] if len(entry_lines) > 1 else None,
                    "dates": date_values,
                }
            )
        return entries

    def _work_entries(self, sections: Dict[str, List[str]]) -> List[Dict[str, Any]]:
        lines = _section_lines(sections, ("experience", "work experience", "expérience", "expériences", "emploi"))
        entries = []
        for group in _entry_groups(lines):
            date_values = _date_ranges(group)
            clean = [line for line in group if line not in date_values]
            responsibilities = [line for line in clean[3:] if len(line) > 12]
            entries.append(
                {
                    "company": clean[0] if clean else None,
                    "position": clean[1] if len(clean) > 1 else None,
                    "location": clean[2] if len(clean) > 2 and _looks_location(clean[2]) else None,
                    "dates": date_values,
                    "responsibilities": responsibilities,
                }
            )
        return entries

    def _language_entries(self, sections: Dict[str, List[str]]) -> List[Dict[str, str]]:
        lines = self._list_from_sections(sections, ("languages", "langues"))
        entries: List[Dict[str, str]] = []
        for line in lines:
            if ":" in line:
                name, level = line.split(":", 1)
            elif "-" in line:
                name, level = line.split("-", 1)
            else:
                parts = line.split()
                name, level = parts[0], " ".join(parts[1:])
            if name.strip():
                entries.append({"name": name.strip(), "level": level.strip()})
        return entries

    def _certification_entries(self, sections: Dict[str, List[str]]) -> List[Dict[str, str]]:
        lines = _section_lines(sections, ("certifications", "certificats", "certification"))
        entries = []
        for line in lines:
            if " - " in line:
                name, provider = line.split(" - ", 1)
            elif "," in line:
                name, provider = line.split(",", 1)
            else:
                name, provider = line, ""
            if name.strip():
                entries.append({"name": name.strip(), "provider": provider.strip()})
        return entries

    def _project_entries(self, sections: Dict[str, List[str]]) -> List[Dict[str, str]]:
        lines = _section_lines(sections, ("projects", "projets"))
        entries = []
        for group in _entry_groups(lines):
            if group:
                entries.append({"name": group[0], "description": " ".join(group[1:])})
        return entries

    def _invoice_items(self, tables: Sequence[TableBlock]) -> List[Dict[str, Any]]:
        best = max(tables, key=lambda table: len(table.rows), default=None)
        if not best or len(best.rows) < 2:
            return []
        header = [_norm(str(cell)) for cell in best.rows[0]]
        desc_idx = _first_header(header, ("description", "designation", "article", "item", "service"))
        price_idx = _first_header(header, ("price", "prix", "unit", "unitaire"))
        qty_idx = _first_header(header, ("qty", "quantity", "quantité", "quantite"))
        total_idx = _first_header(header, ("total", "montant"))
        if desc_idx is None:
            desc_idx = 0
        items: List[Dict[str, Any]] = []
        for row in best.rows[1:]:
            if not any(str(cell).strip() for cell in row):
                continue
            item = {
                "description": _cell(row, desc_idx),
                "price": _number(_cell(row, price_idx)),
                "quantity": _number(_cell(row, qty_idx)),
                "total": _number(_cell(row, total_idx)),
            }
            if item["description"]:
                items.append(item)
        return items

    def _invoice_totals(self, lines: Sequence[str], items: Sequence[Dict[str, Any]], key_values: Dict[str, str]) -> Dict[str, Any]:
        subtotal = self._amount_from_keys(key_values, ("subtotal", "sous_total", "sub_total"))
        tax = self._amount_from_keys(key_values, ("tax", "vat", "tva"))
        total = self._amount_from_keys(key_values, ("total", "grand_total", "amount_due", "montant_total"))
        if subtotal is None:
            subtotal = sum(float(item.get("total") or 0) for item in items) or None
        return {
            "subtotal": subtotal,
            "tax": tax,
            "total": total if total is not None else (subtotal + tax if subtotal is not None and tax is not None else None),
        }

    def _amount_from_keys(self, key_values: Dict[str, str], keys: Sequence[str]) -> float | int | None:
        raw = self._first_value(key_values, keys)
        return _number(raw)

    def _first_value(self, key_values: Dict[str, str], keys: Sequence[str]) -> str | None:
        for key in keys:
            norm_key = _norm(key).replace(" ", "_")
            if norm_key in key_values:
                return key_values[norm_key]
        for actual, value in key_values.items():
            if any(_norm(key).replace(" ", "_") in actual for key in keys):
                return value
        return None

    def _party_name(self, lines: Sequence[str], labels: Sequence[str]) -> str | None:
        normalized_labels = tuple(_norm(label) for label in labels)
        for index, line in enumerate(lines):
            if any(label in _norm(line) for label in normalized_labels):
                for candidate in lines[index + 1 : index + 4]:
                    if candidate and not EMAIL_RE.search(candidate) and not PHONE_RE.search(candidate):
                        return candidate
        return None

    def _first_company_like(self, lines: Sequence[str]) -> str | None:
        for line in lines[:12]:
            if len(line) <= 70 and not EMAIL_RE.search(line) and not PHONE_RE.search(line) and not MONEY_RE.search(line):
                if line.isupper() or any(word in line.lower() for word in ("sarl", "ltd", "inc", "company", "naudin")):
                    return line
        return None

    def _line_containing(self, lines: Sequence[str], needles: Sequence[str]) -> str | None:
        lowered = tuple(needle.lower() for needle in needles)
        return next((line for line in lines if any(needle in line.lower() for needle in lowered)), None)

    def _address(self, lines: Sequence[str]) -> Dict[str, str]:
        street = next((line for line in lines if re.search(r"\b(st\.|street|rue|avenue|av\.|bd|boulevard)\b", line, re.I)), None)
        city = next((line for line in lines if re.search(r"\b(city|ville|paris|toulouse|lyon|any city)\b", line, re.I)), None)
        return {"street": street, "city": city}

    def _payment_line(self, lines: Sequence[str]) -> str | None:
        return self._line_containing(lines, ("paiement", "payment", "payable", "ordre de"))

    def _account_number(self, lines: Sequence[str]) -> str | None:
        for line in lines:
            if re.search(r"\b\d{4}\s+\d{4}\s+\d{4}", line):
                return re.search(r"\b[\d ]{14,}\b", line).group(0).strip()  # type: ignore[union-attr]
        return None

    def _grounded_generic_data(self, document: DocumentSource) -> Dict[str, Any]:
        lines = _nonempty_lines(document.text)
        return {
            "title": lines[0] if lines else None,
            "contacts": self._contact(lines),
            "dates": _unique(match.group(0) for line in lines for match in DATE_RE.finditer(line)),
            "amounts": _unique(match.group(0).strip() for line in lines for match in MONEY_RE.finditer(line)),
            "key_values": _key_values(lines),
            "tables": [{"rows": table.rows, "source": table.source} for table in _document_tables(document)],
            "sections": _sections(lines),
        }

    def _try_ai_inferred_export(self, document: DocumentSource) -> Dict[str, Any]:
        if not APP_CONFIG.qwen.enabled or not document.text.strip():
            return {}
        client = OllamaGenerateClient(APP_CONFIG.qwen)
        if not client.is_available(autostart=False):
            return {}
        prompt = (
            "Create a clean JSON object for this document. Use a document-specific schema. "
            "Only copy values that appear verbatim in the source text. Omit uncertain fields. "
            "Return JSON only.\n\nSOURCE TEXT:\n"
            + document.text[:6000]
        )
        response = client.generate(prompt)
        if not response.ok:
            return {}
        parsed = parse_and_validate(response.text, document.text)
        if parsed.get("error"):
            return {}
        return {key: value for key, value in parsed.items() if not key.startswith("_")}

    def _evidence_for_data(self, data: Dict[str, Any], document: DocumentSource) -> List[ExtractionEvidence]:
        evidence: List[ExtractionEvidence] = []
        source_text = document.text.lower()
        for path, value in _flatten(data).items():
            if value in (None, "", [], {}):
                continue
            text_value = str(value)
            if len(text_value) < 2:
                continue
            if _norm_text(text_value) and _norm_text(text_value) not in _norm_text(source_text):
                continue
            region = _find_region_for_value(document, text_value)
            evidence.append(
                ExtractionEvidence(
                    field_path=path,
                    value=value,
                    source=region.source if region else "document_text",
                    page=region.page if region else None,
                    bbox=list(region.bbox) if region else [],
                    confidence=region.confidence if region else 1.0,
                )
            )
        return evidence


def _document_tables(document: DocumentSource) -> List[TableBlock]:
    tables: List[TableBlock] = []
    for page in document.pages:
        tables.extend(page.tables)
    for item in document.metadata.get("tables", []) if isinstance(document.metadata, dict) else []:
        if isinstance(item, TableBlock):
            tables.append(item)
        elif isinstance(item, dict):
            tables.append(
                TableBlock(
                    page=int(item.get("page", 1)),
                    rows=item.get("rows", []) if isinstance(item.get("rows"), list) else [],
                    bbox=item.get("bbox", []) if isinstance(item.get("bbox"), list) else [],
                    source=str(item.get("source", "metadata_table")),
                    confidence=float(item.get("confidence", 1.0) or 1.0),
                    metadata=item.get("metadata", {}) if isinstance(item.get("metadata"), dict) else {},
                )
            )
    return tables


def _nonempty_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _sections(lines: Sequence[str]) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = "body"
    for line in lines:
        label = _norm(line)
        if _looks_like_heading(line):
            current = label
            sections.setdefault(current, [])
            continue
        sections.setdefault(current, []).append(line)
    return sections


def _section_lines(sections: Dict[str, List[str]], labels: Sequence[str]) -> List[str]:
    output: List[str] = []
    normalized = tuple(_norm(label) for label in labels)
    for key, lines in sections.items():
        if any(label in key or key in label for label in normalized):
            output.extend(lines)
    return output


def _looks_like_heading(line: str) -> bool:
    clean = _norm(line)
    headings = {
        "profile",
        "summary",
        "experience",
        "work experience",
        "education",
        "formation",
        "skills",
        "competences",
        "compétences",
        "languages",
        "langues",
        "certifications",
        "projects",
        "projets",
        "hobbies",
        "centres d interet",
        "centres d'intérêt",
        "payment",
        "paiement",
    }
    return clean in headings or (len(line.split()) <= 3 and line.isupper() and len(line) > 3)


def _key_values(lines: Sequence[str]) -> Dict[str, str]:
    fields: Dict[str, str] = {}
    for line in lines:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        norm_key = _norm(key).replace(" ", "_")
        if norm_key and value.strip():
            fields[norm_key] = value.strip()
    return fields


def _entry_groups(lines: Sequence[str]) -> List[List[str]]:
    groups: List[List[str]] = []
    current: List[str] = []
    for line in lines:
        if DATE_RANGE_RE.search(line) and current:
            current.append(line)
            groups.append(current)
            current = []
            continue
        if len(current) >= 5 and not line.startswith(("-", "•", "*")):
            groups.append(current)
            current = [line]
        else:
            current.append(line)
    if current:
        groups.append(current)
    return [group for group in groups if any(group)]


def _date_ranges(lines: Sequence[str]) -> List[str]:
    ranges = []
    for line in lines:
        match = DATE_RANGE_RE.search(line)
        if match:
            ranges.append(match.group(0))
    if ranges:
        return ranges
    return _unique(match.group(0) for line in lines for match in DATE_RE.finditer(line))


def _split_list_line(line: str) -> List[str]:
    clean = re.sub(r"^[•*\-+\s]+", "", line).strip()
    if not clean:
        return []
    parts = re.split(r"\s{2,}|[;,|]", clean)
    return [part.strip() for part in parts if part.strip()]


def _looks_location(line: str) -> bool:
    return "," in line or bool(re.search(r"\b(paris|toulouse|stockholm|amsterdam|city|france|suède|suede)\b", line, re.I))


def _first_match(pattern: re.Pattern[str], text: str) -> str | None:
    match = pattern.search(text)
    return match.group(0).strip() if match else None


def _first_header(header: Sequence[str], names: Sequence[str]) -> int | None:
    for name in names:
        normalized = _norm(name)
        for index, value in enumerate(header):
            if normalized in value:
                return index
    return None


def _cell(row: Sequence[Any], index: int | None) -> str:
    if index is None or index < 0 or index >= len(row):
        return ""
    return str(row[index]).strip()


def _number(value: Any) -> float | int | None:
    if value is None:
        return None
    text = str(value)
    match = MONEY_RE.search(text)
    if not match:
        return None
    clean = re.sub(r"[^\d,.-]", "", match.group(0)).replace(",", ".")
    if clean.count(".") > 1:
        clean = clean.replace(".", "", clean.count(".") - 1)
    try:
        number = float(clean)
    except ValueError:
        return None
    return int(number) if number.is_integer() else number


def _flatten(data: Any, prefix: str = "") -> Dict[str, Any]:
    if isinstance(data, dict):
        output: Dict[str, Any] = {}
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(value, path))
        return output
    if isinstance(data, list):
        output = {}
        for index, value in enumerate(data):
            output.update(_flatten(value, f"{prefix}[{index}]"))
        return output
    return {prefix: data}


def _find_region_for_value(document: DocumentSource, value: str) -> Any | None:
    normalized = _norm_text(value)
    if not normalized:
        return None
    for page in document.pages:
        for region in page.review_regions:
            if normalized in _norm_text(region.text):
                return region
    return None


def _remove_ungrounded(data: Any, statuses: Dict[str, bool], prefix: str = "") -> Any:
    if isinstance(data, dict):
        output = {}
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            cleaned = _remove_ungrounded(value, statuses, path)
            if cleaned not in (None, "", [], {}):
                output[key] = cleaned
        return output
    if isinstance(data, list):
        output = []
        for index, value in enumerate(data):
            cleaned = _remove_ungrounded(value, statuses, f"{prefix}[{index}]")
            if cleaned not in (None, "", [], {}):
                output.append(cleaned)
        return output
    if prefix in statuses and not statuses[prefix]:
        return None
    return data


def _unique(values: Iterable[str]) -> List[str]:
    output: List[str] = []
    seen: set[str] = set()
    for value in values:
        clean = str(value).strip()
        key = clean.lower()
        if clean and key not in seen:
            seen.add(key)
            output.append(clean)
    return output


def _norm(value: str) -> str:
    import unicodedata

    decomposed = unicodedata.normalize("NFKD", str(value).lower())
    asciiish = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", asciiish)).strip()


def _norm_text(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value).lower())


def _stem(path_text: str) -> str:
    from pathlib import Path

    return Path(path_text).stem or "document"

