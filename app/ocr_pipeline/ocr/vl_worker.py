"""Isolated PaddleOCR-VL worker.

This module is run in a subprocess so PaddleOCR-VL crashes or hangs cannot stop
the main OCR pipeline from writing outputs.
"""

from __future__ import annotations

import sys
from typing import Any, Dict

from app.ocr_pipeline.ocr.paddle_vl import _create_paddleocr_vl_engine, _parse_vl_result


def main(argv: list[str] | None = None) -> int:
    from app.ocr_pipeline.utils.worker_utils import run_worker

    return run_worker(argv or sys.argv, 2, _handle)


def _handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    input_path = str(payload.get("image_path", ""))
    use_orientation = bool(payload.get("use_orientation", False))
    use_unwarping = bool(payload.get("use_unwarping", False))
    try:
        from app.config import LayoutConfig
        from paddleocr import PaddleOCRVL  # type: ignore

        config = LayoutConfig(
            paddleocr_vl_use_orientation=use_orientation,
            paddleocr_vl_use_unwarping=use_unwarping,
        )
        engine = _create_paddleocr_vl_engine(PaddleOCRVL, config)
        raw = engine.predict(input_path) if hasattr(engine, "predict") else engine(input_path)
        return {"source": "paddleocr_vl", "available": True, "result": _parse_vl_result(raw)}
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        return {"source": "paddleocr_vl", "available": False, "reason": f"{type(exc).__name__}: {exc}", "result": []}


if __name__ == "__main__":
    raise SystemExit(main())

