"""Optional OCR/layout model adapters."""

from app.ocr_pipeline.ocr.paddle_vl import LayoutAdapter, run_layout_analysis

__all__ = ["LayoutAdapter", "run_layout_analysis"]

