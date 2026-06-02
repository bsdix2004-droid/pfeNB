"""
OCR Ensemble - Merges and scores multiple OCR engine results.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence
import numpy as np

from app.ocr_pipeline.core.ocr_engine import TextBlock, OCRCandidateResult
from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)

class OCREnsemble:
    """Combines multiple OCR candidates into one best-guess result."""

    def merge(self, candidates: Sequence[OCRCandidateResult]) -> OCRCandidateResult:
        """
        Simple ensemble strategy:
        1. Pick the candidate with the highest overall score.
        2. In future: character-level voting or region-level merging.
        """
        if not candidates:
            raise ValueError("No OCR candidates provided to ensemble")

        # Pick the best by score (which includes confidence and density)
        best = max(candidates, key=lambda c: c.score)
        
        LOGGER.info(
            "Ensemble selected '%s' (score=%.4f, conf=%.4f)",
            best.candidate_name, best.score, best.avg_confidence
        )
        
        return best

    def score_consensus(self, candidates: Sequence[OCRCandidateResult]) -> float:
        """Calculate how much the engines agree with each other."""
        if len(candidates) < 2:
            return 1.0
            
        # Placeholder for text-similarity based consensus scoring
        return 0.85

