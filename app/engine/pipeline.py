"""
app/engine/pipeline.py

Thin orchestration layer between Celery tasks and the OCR pipeline.

This is the "engine" boundary where we:
  - receive a trace_id from the caller (API/Celery)
  - set up a per-run pipeline logger
  - run the intelligent document analyzer (app/ocr_pipeline)
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from app.ocr_pipeline.core.smart_pipeline import SmartDocumentPipeline, SmartDocumentResult
from app.utils.logging import get_logger, log_stage


def run_document_pipeline(
    *,
    document_bytes: bytes,
    filename: str,
    trace_id: str,
    run_ocr_fallback: bool = True,
) -> SmartDocumentResult:
    """
    Run the intelligent document analyzer on bytes.

    Notes:
    - The OCR pipeline expects a local path, so we materialize a temp file.
    - Results are returned as a SmartDocumentResult; persistence is handled by the caller.
    """

    suffix = Path(filename).suffix or ".bin"
    with log_stage(trace_id, "engine.write_temp_input"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
            f.write(document_bytes)
            temp_path = Path(f.name)

    try:
        pipeline = SmartDocumentPipeline()
        with log_stage(trace_id, "engine.run_pipeline"):
            result = pipeline.process(temp_path, run_ocr_fallback=run_ocr_fallback, trace_id=trace_id)
        get_logger(trace_id).bind(stage="engine").info(
            "pipeline_completed",
            status=result.status,
            output_dir=result.output_dir,
            duration_ms=result.processing_time_ms,
        )
        return result
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception as exc:  # pragma: no cover
            get_logger(trace_id).bind(stage="engine.cleanup").warning("temp_cleanup_failed", error=str(exc))

