"""Isolated OCR worker so Paddle crashes cannot kill the main pipeline."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
import sys

import cv2

from app.ocr_pipeline.ocr_config import APP_CONFIG
from app.ocr_pipeline.core.detector import DocumentInfo
from app.ocr_pipeline.core.ocr_engine import PaddleOCREngine, OCRCandidateResult
from app.ocr_pipeline.semantic.reconstructor import OCRLine


def main(argv: list[str] | None = None) -> int:
    from app.ocr_pipeline.utils.worker_utils import run_worker

    return run_worker(argv or sys.argv, 2, _handle)


def _handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    candidates = _load_candidates(payload.get("candidates", []))
    if not candidates:
        return {"error": "No OCR candidates"}

    doc_payload = payload.get("doc_info", {})
    doc_info = DocumentInfo(
        str(doc_payload.get("doc_type", "unknown")),
        str(doc_payload.get("language", "en")),
        str(doc_payload.get("direction", "ltr")),
        float(doc_payload.get("confidence", 0.0) or 0.0),
    )

    layout_blocks = payload.get("layout_blocks")
    scripts = payload.get("scripts")
    result = PaddleOCREngine(APP_CONFIG.ocr).extract_best(
        candidates,
        doc_info,
        layout_blocks=layout_blocks if isinstance(layout_blocks, list) else None,
        scripts=scripts if isinstance(scripts, list) else None,
    )
    return _result_to_dict(result)


def _load_candidates(items: Any) -> List[SimpleNamespace]:
    candidates: List[SimpleNamespace] = []
    if not isinstance(items, list):
        return candidates
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        path = Path(str(item.get("path", "")))
        image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if image is None or image.size == 0:
            continue
        candidates.append(SimpleNamespace(name=str(item.get("name") or f"candidate_{index}"), image=image))
    return candidates


def _result_to_dict(result: OCRCandidateResult) -> Dict[str, Any]:
    return {
        "candidate_name": result.candidate_name,
        "candidate_index": result.candidate_index,
        "score": result.score,
        "avg_confidence": result.avg_confidence,
        "raw_line_count": result.raw_line_count,
        "char_count": result.char_count,
        "engine_version": result.engine_version,
        "languages_used": result.languages_used,
        "fallback_languages_skipped": result.fallback_languages_skipped,
        "lines": [_line_to_dict(line) for line in result.lines],
    }


def _line_to_dict(line: OCRLine) -> Dict[str, Any]:
    return {
        "text": line.text,
        "polygon": [[float(x), float(y)] for x, y in line.polygon],
        "bbox": [int(value) for value in line.bbox],
        "confidence": float(line.confidence),
        "source_index": int(line.source_index),
        "language": line.language,
        "low_confidence": bool(line.low_confidence),
    }


if __name__ == "__main__":
    raise SystemExit(main())

