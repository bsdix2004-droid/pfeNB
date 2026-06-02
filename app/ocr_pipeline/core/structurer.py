"""Qwen2.5 structuring through local Ollama only."""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any, Dict, List

from app.config import QwenConfig
from app.ocr_pipeline.core.detector import DocumentInfo
from app.ocr_pipeline.core.llm_client import OllamaGenerateClient
from app.ocr_pipeline.core.ocr_engine import TextBlock
from app.ocr_pipeline.utils.logger import get_logger


LOGGER = get_logger(__name__)


class PromptLoader:
    """Load prompt templates from the prompts directory."""

    _cache: Dict[str, str] = {}
    _prompts_dir = Path(__file__).resolve().parents[1] / "prompts"

    @classmethod
    def system(cls) -> str:
        return cls._load("base_system.txt")

    @classmethod
    def for_doc_type(cls, doc_type: str) -> str:
        filename = f"{doc_type}.txt"
        if (cls._prompts_dir / filename).exists():
            return cls._load(filename)
        return cls._load("generic.txt")

    @classmethod
    def _load(cls, filename: str) -> str:
        if filename not in cls._cache:
            path = cls._prompts_dir / filename
            try:
                cls._cache[filename] = path.read_text(encoding="utf-8")
            except Exception as exc:
                LOGGER.warning("Could not load prompt %s: %s", filename, exc)
                cls._cache[filename] = ""
        return cls._cache[filename]


QWEN_SYSTEM_PROMPT = PromptLoader.system()


class QwenStructurer:
    """Structures semantic blocks into JSON and rejects unsupported values."""

    def __init__(self, config: QwenConfig) -> None:
        self.config = config
        self.client = OllamaGenerateClient(config)

    def is_available(self) -> bool:
        return self.client.is_available(autostart=False)

    def structure(
        self,
        blocks: List[TextBlock],
        doc_info: DocumentInfo,
        candidate_fields: List[Dict[str, Any]] | None = None,
        layout_info: Dict[str, Any] | None = None,
        correction_examples: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        if not blocks or not self.config.enabled:
            return {}

        blocks_text, excluded_lines = self._verified_blocks_text(blocks)
        candidate_fields = list(candidate_fields or [])
        layout_info = layout_info or {}
        correction_examples = list(correction_examples or [])
        if not blocks_text.strip():
            return {
                "error": "No high-confidence OCR text available for AI structuring",
                "_excluded_low_confidence_lines": excluded_lines,
            }

        exact_data = self._exact_extraction(blocks_text)
        system_prompt = self._strict_system_prompt(PromptLoader.system())
        doc_prompt = PromptLoader.for_doc_type(doc_info.doc_type)
        user_prompt = self._build_user_prompt(
            doc_prompt=doc_prompt,
            verified_text=blocks_text,
            candidate_fields=candidate_fields,
            doc_info=doc_info,
            layout_info=layout_info,
            correction_examples=correction_examples,
        )

        response = self.client.generate(user_prompt, system=system_prompt)
        if not response.ok:
            LOGGER.debug("Qwen structuring failed: %s", response.error)
            return {
                **exact_data,
                "_qwen_error": response.error,
                "_excluded_low_confidence_lines": excluded_lines,
                "_ocr_candidate_fields": candidate_fields,
            }
        qwen_data = parse_and_validate(response.text, blocks_text, candidate_fields=candidate_fields)
        return {
            **exact_data,
            "_qwen_extraction": qwen_data,
            "_hallucination_warnings": qwen_data.get("_hallucination_warnings", []) if isinstance(qwen_data, dict) else [],
            "_excluded_low_confidence_lines": excluded_lines,
            "_ocr_candidate_fields": candidate_fields,
        }

    def _strict_system_prompt(self, base_prompt: str) -> str:
        rules = """

STRICT OCR-GROUNDED EXTRACTION RULES:
- Output valid JSON only.
- Use only values copied from OCR candidate fields or verified OCR text.
- Every non-null extracted field must include evidence_text and confidence.
- Preserve the raw OCR value. Do not silently normalize away OCR evidence.
- If evidence is missing, set value to null and requires_review to true.
- Do not guess names, totals, dates, addresses, invoice numbers, or missing fields.
- Unsupported or hallucinated values must be null.
- You are allowed to organize, group, label, and validate OCR-derived values only.
"""
        return (base_prompt or "") + rules

    def _build_user_prompt(
        self,
        doc_prompt: str,
        verified_text: str,
        candidate_fields: List[Dict[str, Any]],
        doc_info: DocumentInfo,
        layout_info: Dict[str, Any],
        correction_examples: List[Dict[str, Any]],
    ) -> str:
        payload = {
            "document_type": doc_info.doc_type,
            "language": doc_info.language,
            "layout_info": layout_info,
            "ocr_candidate_fields": candidate_fields,
            "previous_correction_examples": correction_examples,
            "verified_ocr_text": verified_text,
        }
        prefix = doc_prompt.replace("{verified_text}", verified_text)
        return (
            prefix
            + "\n\nOCR SOURCE PACKAGE (authoritative):\n"
            + json.dumps(payload, ensure_ascii=False, indent=2)
            + "\n\nReturn only JSON. Every value must be copied from the OCR source package."
        )

    def _verified_blocks_text(self, blocks: List[TextBlock]) -> tuple[str, List[Dict[str, Any]]]:
        parts: List[str] = []
        excluded: List[Dict[str, Any]] = []

        for block in sorted(blocks, key=lambda item: item.reading_order):
            verified_lines: List[str] = []
            for index, line in enumerate(block.lines):
                confidence = block.line_confidences[index] if index < len(block.line_confidences) else block.avg_confidence
                if confidence >= self.config.min_ocr_confidence:
                    verified_lines.append(line)
                else:
                    excluded.append(
                        {
                            "block_id": block.id,
                            "line": line,
                            "confidence": round(float(confidence), 4),
                            "reason": "below_minimum_ocr_confidence",
                        }
                    )

            if verified_lines:
                parts.append(f"[{block.block_type.upper()} - block {block.id}]\n" + "\n".join(verified_lines))

        return "\n\n".join(parts), excluded

    def _exact_extraction(self, blocks_text: str) -> Dict[str, Any]:
        fields: Dict[str, str] = {}
        lines: List[str] = []

        for raw_line in blocks_text.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("["):
                continue
            lines.append(line)
            if ":" not in line:
                continue

            label, value = line.split(":", 1)
            value = value.strip()
            if not value:
                continue
            key = re.sub(r"[^a-z0-9]+", "_", label.strip().lower()).strip("_") or "field"
            unique_key = key
            suffix = 2
            while unique_key in fields:
                unique_key = f"{key}_{suffix}"
                suffix += 1
            fields[unique_key] = value

        return {
            **fields,
            "verified_fields": fields,
            "verified_text_lines": lines,
        }


def parse_and_validate(
    raw_output: str,
    source_text: str,
    candidate_fields: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    clean = raw_output.strip()
    if clean.startswith("```"):
        clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
        clean = re.sub(r"```$", "", clean).strip()

    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        return {"error": "Qwen returned invalid JSON", "raw": raw_output}

    if not isinstance(data, dict):
        return {"error": "Qwen returned JSON that is not an object", "raw": data}
    return validate_no_hallucination(data, source_text, candidate_fields=candidate_fields or [])


def validate_no_hallucination(
    data: Dict[str, Any],
    source_text: str,
    candidate_fields: List[Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    flagged: List[str] = []
    source_lower = source_text.lower()
    candidate_values = {
        str(item.get("raw_value", "")).strip().lower()
        for item in candidate_fields or []
        if str(item.get("raw_value", "")).strip()
    }

    def check(obj: Any, path: str = "") -> Any:
        if isinstance(obj, dict):
            return {key: check(value, f"{path}.{key}" if path else key) for key, value in obj.items()}
        if isinstance(obj, list):
            return [check(item, f"{path}[]") for item in obj]
        if isinstance(obj, bool):
            flagged.append(path)
            return None
        if isinstance(obj, (int, float)):
            flagged.append(path)
            return None
        if isinstance(obj, str):
            value = obj.strip()
            lower_value = value.lower()
            if value and lower_value not in source_lower and lower_value not in candidate_values:
                flagged.append(path)
                return None
        return obj

    cleaned = check(data)
    if isinstance(cleaned, dict):
        cleaned["_hallucination_warnings"] = flagged
    return cleaned if isinstance(cleaned, dict) else {"value": cleaned, "_hallucination_warnings": flagged}

