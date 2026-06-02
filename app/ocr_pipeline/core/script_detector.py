"""
Script detection for selecting OCR language routes before OCR runs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import cv2
import numpy as np

from app.config import ScriptDetectorConfig
from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)

_ARABIC_LO, _ARABIC_HI = 0x0600, 0x06FF
_CJK_RANGES = [(0x4E00, 0x9FFF), (0x3040, 0x30FF), (0xAC00, 0xD7AF)]
_LATIN_LO, _LATIN_HI = 0x0020, 0x024F


@dataclass(slots=True)
class ScriptDetectionResult:
    scripts_found: List[str]
    primary_script: str
    has_arabic: bool
    has_latin: bool
    has_cjk: bool
    confidence: float
    method: str


class ScriptDetector:
    """Detect writing scripts present in a document image."""

    def __init__(self, config: ScriptDetectorConfig | None = None) -> None:
        self.config = config or ScriptDetectorConfig()

    def detect(
        self,
        image: np.ndarray,
        layout_blocks: Sequence[Dict[str, Any]] | None = None,
    ) -> ScriptDetectionResult:
        hint_text = _extract_layout_text(layout_blocks or [])
        unicode_scores: Dict[str, float] = _unicode_scores(hint_text) if hint_text.strip() else {}
        visual_scores: Dict[str, float] = _visual_scores(image)

        merged: Dict[str, float] = {}
        for script, score in unicode_scores.items():
            merged[script] = merged.get(script, 0.0) + score * self.config.unicode_weight
        for script, score in visual_scores.items():
            merged[script] = merged.get(script, 0.0) + score * self.config.visual_weight

        if not merged:
            merged = {"latin": 1.0}

        total = sum(merged.values()) or 1.0
        primary = max(merged, key=merged.get)
        scripts_found = [script for script, value in merged.items() if value / total >= self.config.min_script_ratio]
        method = "combined" if hint_text.strip() else "visual"

        return ScriptDetectionResult(
            scripts_found=scripts_found,
            primary_script=primary,
            has_arabic="arabic" in scripts_found,
            has_latin="latin" in scripts_found,
            has_cjk="cjk" in scripts_found,
            confidence=round(merged[primary] / total, 4),
            method=method,
        )


def _unicode_scores(text: str) -> Dict[str, float]:
    counts: Dict[str, int] = {"arabic": 0, "latin": 0, "cjk": 0}
    for ch in text:
        cp = ord(ch)
        if _ARABIC_LO <= cp <= _ARABIC_HI:
            counts["arabic"] += 1
        elif any(lo <= cp <= hi for lo, hi in _CJK_RANGES):
            counts["cjk"] += 1
        elif _LATIN_LO <= cp <= _LATIN_HI and ch.isalpha():
            counts["latin"] += 1
    total = sum(counts.values()) or 1
    return {key: value / total for key, value in counts.items() if value > 0}


def _visual_scores(image: np.ndarray) -> Dict[str, float]:
    """Classify script from connected-component aspect ratios."""
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        n_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        ratios: List[float] = []
        for i in range(1, n_labels):
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            area = stats[i, cv2.CC_STAT_AREA]
            if height < 5 or area < 15:
                continue
            ratios.append(width / max(height, 1))
        if not ratios:
            return {"latin": 1.0}
        avg = float(np.mean(ratios))
        if avg > 2.5:
            return {"arabic": 0.70, "latin": 0.30}
        if 0.8 <= avg <= 1.3:
            return {"cjk": 0.60, "latin": 0.40}
        return {"latin": 1.0}
    except Exception as exc:
        LOGGER.warning("visual script detection failed: %s", exc)
        return {"latin": 1.0}


def _extract_layout_text(layout_blocks: Sequence[Dict[str, Any]]) -> str:
    parts: List[str] = []
    for block in layout_blocks:
        content = block.get("content", [])
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict):
                    parts.append(str(item.get("text", "")))
        elif isinstance(content, str):
            parts.append(content)
    return " ".join(parts)

