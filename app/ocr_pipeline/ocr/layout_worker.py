"""Isolated layout worker so Paddle layout crashes cannot stop the pipeline."""

from __future__ import annotations

from typing import Any, Dict
import sys

import cv2

from app.ocr_pipeline.ocr.paddle_vl import run_layout_analysis


def main(argv: list[str] | None = None) -> int:
    from app.ocr_pipeline.utils.worker_utils import run_worker

    return run_worker(argv or sys.argv, 2, _handle)


def _handle(payload: Dict[str, Any]) -> Dict[str, Any]:
    input_path = str(payload.get("image_path", ""))
    use_gpu = bool(payload.get("use_gpu", False))
    try:
        image = cv2.imread(input_path, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            return {
                "source": "layout_detection",
                "available": False,
                "reason": "could not read worker input image",
                "detected_regions": [],
            }
        return run_layout_analysis(image, use_gpu=use_gpu, engine=None)
    except BaseException as exc:
        if isinstance(exc, KeyboardInterrupt):
            raise
        return {
            "source": "layout_detection",
            "available": False,
            "reason": f"{type(exc).__name__}: {exc}",
            "detected_regions": [],
        }


if __name__ == "__main__":
    raise SystemExit(main())

