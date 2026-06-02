"""AI orchestration for deterministic extraction and local Qwen validation."""

from __future__ import annotations

from dataclasses import dataclass, field
from dataclasses import asdict
from pathlib import Path
import json
import re
import subprocess
import sys
import tempfile
from typing import Any, Dict, List

from app.ocr_pipeline.ocr_config import AppConfig
from app.ocr_pipeline.core.detector import DocumentInfo
from app.ocr_pipeline.core.field_extractor import FieldExtractionResult
from app.ocr_pipeline.core.ocr_engine import TextBlock
from app.ocr_pipeline.core.ocr_value_extractor import OCRValueExtractor
from app.ocr_pipeline.core.schema_extractor import StructuredSchemaExtractor
from app.ocr_pipeline.core.structurer import QwenStructurer, PromptLoader
from app.ocr_pipeline.utils.logger import get_logger


LOGGER = get_logger(__name__)


DOC_TYPE_DISPLAY = {
    "cv": "CV / Resume",
    "invoice": "Invoice",
    "contract": "Contract",
    "legal": "Legal Document",
    "id_card": "ID Card",
    "form": "Form",
    "administrative": "Administrative",
    "signature": "Signature",
    "unknown": "Unknown",
}


@dataclass(slots=True)
class AIOrchestrationResult:
    structured_data: Dict[str, Any] = field(default_factory=dict)
    providers: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    ai_classification: Dict[str, Any] = field(default_factory=dict)


class DocumentAIOrchestrator:
    """Qwen AI extraction as primary — deterministic rules fill gaps only."""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.schema_extractor = StructuredSchemaExtractor()
        self.value_extractor = OCRValueExtractor()
        self.qwen = QwenStructurer(config.qwen)

    def analyze(
        self,
        blocks: List[TextBlock],
        doc_info: DocumentInfo,
        extracted_fields: FieldExtractionResult | None = None,
        layout_info: Dict[str, Any] | None = None,
        correction_examples: List[Dict[str, Any]] | None = None,
    ) -> AIOrchestrationResult:
        candidate_fields = [item.to_dict() for item in self.value_extractor.extract(blocks)]
        qwen_available = self.qwen.is_available()

        # 0. Qwen AI classification — PRIMARY, overrides rule-based
        ai_classification: Dict[str, Any] = {}
        if qwen_available and self.config.qwen.enabled:
            ai_classification = self._classify_with_qwen(blocks)
            if ai_classification.get("document_type") and ai_classification["document_type"] != doc_info.doc_type:
                old_type = doc_info.doc_type
                doc_info.doc_type = ai_classification["document_type"]
                LOGGER.info(
                    "Document type corrected by AI: %s → %s (confidence: %s)",
                    old_type,
                    doc_info.doc_type,
                    ai_classification.get("confidence", "N/A"),
                )

        # 1. Qwen AI extraction — PRIMARY
        qwen_data = (
            self._run_qwen_isolated(
                blocks,
                doc_info,
                candidate_fields=candidate_fields,
                layout_info=layout_info or {},
                correction_examples=correction_examples or [],
            )
            if self.config.qwen.enabled
            else {}
        )

        # 2. Deterministic extraction — FALLBACK only
        deterministic = self.schema_extractor.extract(blocks, doc_info)
        deterministic.setdefault("_ocr_candidate_fields", candidate_fields)
        if extracted_fields and extracted_fields.fields:
            deterministic.update(extracted_fields.fields)

        # 3. BASELINE: Qwen extraction — deterministic fills gaps Qwen left null/missing
        qwen_fields = {}
        qwen_failed = False
        if isinstance(qwen_data, dict):
            qwen_fields = {
                k: v for k, v in qwen_data.get("_qwen_extraction", {}).items()
                if not k.startswith("_") and v not in (None, "", [], {})
            }
            qwen_failed = bool(qwen_data.get("_qwen_error") or qwen_data.get("error"))
        combined = {**qwen_fields}
        for key, value in deterministic.items():
            if key.startswith("_"):
                continue
            if key not in combined or combined[key] in (None, "", [], {}):
                combined[key] = value

        # Audit trail — both sources preserved
        combined["_qwen_extraction"] = qwen_data
        combined["_deterministic_fallback"] = deterministic
        combined["_ai_classification"] = ai_classification
        combined["_source"] = {
            "primary": "qwen_ollama" if qwen_fields else ("deterministic_fallback" if qwen_failed else "none"),
            "fallback_used": not bool(qwen_fields),
        }
        warnings = qwen_data.get("_hallucination_warnings", []) if isinstance(qwen_data, dict) else []
        combined["_hallucination_warnings"] = warnings

        result = AIOrchestrationResult(
            providers={
                "structured_extraction": {
                    "provider": "qwen_ollama",
                    "model": self.config.qwen.model,
                    "enabled": self.config.qwen.enabled,
                    "available": qwen_available,
                    "status": "pending" if self.config.qwen.enabled else "skipped",
                },
                "deterministic_schema": {
                    "provider": "local_rules",
                    "enabled": True,
                    "available": True,
                    "status": "fallback",
                    "role": "fills gaps where Qwen returned null",
                },
            }
        )

        if isinstance(qwen_data, dict):
            qwen_error = qwen_data.get("_qwen_error") or qwen_data.get("error")
            if qwen_error:
                result.errors.append(str(qwen_error))
                result.providers["structured_extraction"]["status"] = "warning"
            else:
                result.providers["structured_extraction"]["status"] = "ok" if qwen_data else "skipped"
            result.providers["structured_extraction"]["available"] = self.qwen.is_available()

        result.structured_data = combined
        result.ai_classification = ai_classification
        return result

    def _classify_with_qwen(self, blocks: List[TextBlock]) -> Dict[str, Any]:
        """Ask Qwen to classify the document type, returns override or empty."""
        if not blocks:
            return {}
        verify_text = "\n".join(
            line for block in blocks for idx, line in enumerate(block.lines)
            if idx < len(block.line_confidences) and block.line_confidences[idx] >= 0.7
        )[:4000]
        if not verify_text.strip():
            return {}
        classify_prompt = PromptLoader.for_doc_type("classify")
        prompt = classify_prompt.replace("{verified_text}", verify_text)
        try:
            response = self.qwen.client.generate(prompt, system="Return only valid JSON. No explanation.")
        except Exception:
            return {}
        if not response.ok or not response.text:
            return {}
        try:
            clean = response.text.strip()
            if clean.startswith("```"):
                clean = re.sub(r"^```[a-zA-Z]*\n?", "", clean)
                clean = re.sub(r"```$", "", clean).strip()
            data = json.loads(clean)
        except (json.JSONDecodeError, ValueError):
            return {}
        doc_type = str(data.get("document_type", "")).lower().strip()
        valid_types = {"cv", "invoice", "contract", "legal", "id_card", "form", "administrative", "signature", "unknown"}
        if doc_type not in valid_types:
            return {}
        return {
            "document_type": doc_type,
            "confidence": float(data.get("confidence", 0.0) or 0.0),
            "reason": str(data.get("reason", "")),
        }

    def refine_document_info(self, detected: DocumentInfo, understanding: Dict[str, Any] | None = None) -> DocumentInfo:
        return detected

    def _run_qwen_isolated(
        self,
        blocks: List[TextBlock],
        doc_info: DocumentInfo,
        candidate_fields: List[Dict[str, Any]] | None = None,
        layout_info: Dict[str, Any] | None = None,
        correction_examples: List[Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        if not blocks:
            return {}
        with tempfile.TemporaryDirectory(prefix="docai_qwen_") as temp_dir:
            input_path = Path(temp_dir) / "qwen_input.json"
            output_path = Path(temp_dir) / "qwen_output.json"
            payload = {
                "doc_info": asdict(doc_info),
                "blocks": [asdict(block) for block in blocks],
                "candidate_fields": list(candidate_fields or []),
                "layout_info": layout_info or {},
                "correction_examples": list(correction_examples or []),
            }
            input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
            command = [sys.executable, "-m", "app.ocr_pipeline.core.qwen_worker", str(input_path), str(output_path)]
            try:
                completed = subprocess.run(
                    command,
                    cwd=str(Path(__file__).resolve().parents[3]),
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    timeout=max(10, int(self.config.qwen.timeout_sec) + 10),
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return {"_qwen_error": f"Qwen worker timeout after {self.config.qwen.timeout_sec}s"}
            if completed.returncode != 0 or not output_path.exists():
                return {"_qwen_error": f"Qwen worker exited with code {completed.returncode}"}
            try:
                data = json.loads(output_path.read_text(encoding="utf-8"))
            except Exception as exc:
                return {"_qwen_error": f"Invalid Qwen worker output: {exc}"}
            return data if isinstance(data, dict) else {"_qwen_error": "Qwen worker output was not a JSON object"}


