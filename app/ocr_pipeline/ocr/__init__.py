"""Optional OCR/layout model adapters."""

from app.ocr_pipeline.ocr.paddle_vl import PaddleOCRVLAdapter, run_layout_analysis, run_paddleocr_vl

__all__ = ["PaddleOCRVLAdapter", "run_layout_analysis", "run_paddleocr_vl"]

