"""
Converts raw OCR/Layout blocks into structured SemanticRegion objects.
"""
from __future__ import annotations

from typing import Any, Dict, List, Sequence
import uuid

from app.ocr_pipeline.core.models.region import SemanticRegion
from app.ocr_pipeline.core.models.bbox import BBox
from app.ocr_pipeline.core.block_types import BlockType
from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)

class RegionBuilder:
    """Builds SemanticRegion objects from various sources."""

    def build_from_ocr(self, ocr_blocks: Sequence[Any], page_num: int = 1) -> List[SemanticRegion]:
        """Convert OCR engine blocks to SemanticRegions."""
        regions = []
        for block in ocr_blocks:
            try:
                # Use the existing from_text_block helper
                region = SemanticRegion.from_text_block(block, page=page_num)
                regions.append(region)
            except Exception as exc:
                LOGGER.warning("Failed to build region from block: %s", exc)
        return regions

    def build_from_layout(self, layout_data: Sequence[Dict[str, Any]], page_num: int = 1) -> List[SemanticRegion]:
        """Convert Layout Analysis regions to SemanticRegions."""
        regions = []
        for item in layout_data:
            try:
                bbox_raw = item.get("bbox", [0, 0, 0, 0])
                bbox = BBox.from_tuple(tuple(bbox_raw))
                
                label = item.get("label", "unknown").lower()
                try:
                    block_type = BlockType(label)
                except ValueError:
                    block_type = BlockType.UNKNOWN

                region = SemanticRegion(
                    id=str(uuid.uuid4()),
                    page=page_num,
                    bbox=bbox,
                    text=str(item.get("text", "")),
                    block_type=block_type,
                    confidence=float(item.get("confidence", 0.0)),
                    metadata=item.get("metadata", {})
                )
                regions.append(region)
            except Exception as exc:
                LOGGER.warning("Failed to build region from layout item: %s", exc)
        return regions

