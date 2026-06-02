"""Isolated local Qwen worker."""

from __future__ import annotations

import sys
from typing import Any, Dict

from app.ocr_pipeline.ocr_config import APP_CONFIG
from app.ocr_pipeline.core.block_types import BlockType
from app.ocr_pipeline.core.detector import DocumentInfo
from app.ocr_pipeline.core.ocr_engine import TextBlock
from app.ocr_pipeline.core.structurer import QwenStructurer


def main(argv: list[str] | None = None) -> int:
    from app.ocr_pipeline.utils.worker_utils import run_worker

    return run_worker(argv or sys.argv, 2, _handle)


def _handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    doc_info = _doc_info(payload.get("doc_info", {}))
    blocks = [_text_block(item) for item in payload.get("blocks", []) if isinstance(item, dict)]
    candidate_fields = payload.get("candidate_fields", [])
    layout_info = payload.get("layout_info", {})
    correction_examples = payload.get("correction_examples", [])
    return QwenStructurer(APP_CONFIG.qwen).structure(
        blocks,
        doc_info,
        candidate_fields=candidate_fields if isinstance(candidate_fields, list) else [],
        layout_info=layout_info if isinstance(layout_info, dict) else {},
        correction_examples=correction_examples if isinstance(correction_examples, list) else [],
    )


def _doc_info(data: Dict[str, Any]) -> DocumentInfo:
    return DocumentInfo(
        doc_type=str(data.get("doc_type", "unknown")),
        language=str(data.get("language", "unknown")),
        direction=str(data.get("direction", "ltr")),
        confidence=float(data.get("confidence", 0.0)),
        language_probabilities=data.get("language_probabilities", {}) if isinstance(data.get("language_probabilities"), dict) else {},
        classification_scores=data.get("classification_scores", {}) if isinstance(data.get("classification_scores"), dict) else {},
        evidence=data.get("evidence", {}) if isinstance(data.get("evidence"), dict) else {},
    )


def _text_block(data: Dict[str, Any]) -> TextBlock:
    return TextBlock(
        id=int(data.get("id", 0)),
        text=str(data.get("text", "")),
        lines=[str(item) for item in data.get("lines", [])],
        bbox=tuple(data.get("bbox", (0, 0, 0, 0))),
        avg_confidence=float(data.get("avg_confidence", 0.0)),
        reading_order=int(data.get("reading_order", 0)),
        block_type=str(data.get("block_type", BlockType.PARAGRAPH)),
        line_bboxes=[tuple(item) for item in data.get("line_bboxes", [])],
        line_confidences=[float(item) for item in data.get("line_confidences", [])],
        language=str(data.get("language", "unknown")),
        metadata=data.get("metadata", {}) if isinstance(data.get("metadata"), dict) else {},
    )


if __name__ == "__main__":
    raise SystemExit(main())

