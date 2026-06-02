"""Build strict summary JSON from extracted text and validated entities."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
import re

from app.ocr_pipeline.core.detector import DocumentInfo
from app.ocr_pipeline.core.ocr_engine import TextBlock


class SummaryEngine:
    """Build summary payloads without inventing document values."""

    def build(
        self,
        blocks: Sequence[TextBlock],
        doc_info: DocumentInfo,
        structured_data: Dict[str, Any],
        vision_data: Dict[str, Any] | None = None,
    ) -> Dict[str, Any]:
        full_text = self.cleaned_text(blocks)
        warnings = structured_data.get("_hallucination_warnings", []) if isinstance(structured_data, dict) else []
        return {
            "document_type": doc_info.doc_type,
            "language": doc_info.language,
            "purpose": self._purpose(doc_info.doc_type),
            "summary": self._factual_summary(full_text),
            "structured_extraction": structured_data if isinstance(structured_data, dict) else {},
            "key_entities": self._key_entities(structured_data),
            "important_points": self._important_points(full_text),
            "confidence": round(self._avg_confidence(blocks), 4),
            "_hallucination_warnings": warnings,
        }

    def cleaned_text(self, blocks: Sequence[TextBlock]) -> str:
        return "\n".join(block.text.strip() for block in sorted(blocks, key=lambda item: item.reading_order) if block.text.strip())

    def _purpose(self, doc_type: str) -> str | None:
        purposes = {
            "cv": "Job application curriculum vitae",
            "invoice": "Payment or billing document",
            "contract": "Agreement terms document",
            "id_card": "Identity document",
            "form": "Information collection form",
            "legal": "Legal or regulatory document",
            "administrative": "Administrative document",
        }
        return purposes.get(doc_type)

    def _factual_summary(self, full_text: str) -> str:
        lines = [line.strip() for line in full_text.splitlines() if line.strip()]
        if not lines:
            return ""
        selected = lines[:3]
        return " ".join(selected)[:700]

    def _important_points(self, full_text: str) -> List[str]:
        lines = [line.strip() for line in full_text.splitlines() if len(line.strip()) > 3]
        scored = sorted(lines, key=lambda line: (self._line_score(line), -len(line)), reverse=True)
        output: List[str] = []
        seen: set[str] = set()
        for line in scored:
            key = line.lower()
            if key in seen:
                continue
            seen.add(key)
            output.append(line)
            if len(output) >= 8:
                break
        return output

    def _line_score(self, line: str) -> int:
        score = 0
        if re.search(r"[\w.+-]+@[\w.-]+\.\w+", line):
            score += 3
        if re.search(r"\d", line):
            score += 2
        if re.search(r"(total|invoice|facture|experience|education|contract|agreement|date)", line.lower()):
            score += 2
        if ":" in line:
            score += 1
        return score

    def _key_entities(self, structured_data: Dict[str, Any]) -> Dict[str, Any]:
        entities = {
            "person_name": None,
            "contact": {"phone": None, "email": None, "address": None},
            "experience": [],
            "education": [],
            "skills": [],
            "languages_spoken": [],
        }
        if not isinstance(structured_data, dict):
            return entities

        personal = structured_data.get("personal_info") if isinstance(structured_data.get("personal_info"), dict) else {}
        person = structured_data.get("person") if isinstance(structured_data.get("person"), dict) else {}
        contact = structured_data.get("contact") if isinstance(structured_data.get("contact"), dict) else {}
        entities["person_name"] = (
            structured_data.get("full_name")
            or structured_data.get("name")
            or structured_data.get("customer")
            or structured_data.get("customer_name")
            or personal.get("name")
            or person.get("name")
        )
        entities["contact"]["phone"] = structured_data.get("phone") or personal.get("phone") or contact.get("phone")
        entities["contact"]["email"] = (
            structured_data.get("email")
            or structured_data.get("email_address")
            or personal.get("email")
            or contact.get("email")
        )
        entities["contact"]["address"] = structured_data.get("address") or personal.get("address") or contact.get("location")
        entities["experience"] = self._as_list(structured_data.get("experience"))
        entities["education"] = self._as_list(structured_data.get("education"))
        entities["skills"] = self._as_list(structured_data.get("skills"))
        entities["languages_spoken"] = self._as_list(structured_data.get("languages") or structured_data.get("languages_spoken"))
        return entities

    def _as_list(self, value: Any) -> List[Any]:
        if isinstance(value, list):
            return value
        if value in (None, "", {}):
            return []
        return [value]

    def _avg_confidence(self, blocks: Sequence[TextBlock]) -> float:
        if not blocks:
            return 0.0
        return sum(block.avg_confidence for block in blocks) / len(blocks)

