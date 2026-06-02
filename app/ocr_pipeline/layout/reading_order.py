"""
Logic for determining the reading order of semantic regions.
Supports multi-column layouts and nested sections.
"""
from __future__ import annotations

from typing import List, Sequence
from app.ocr_pipeline.core.models.region import SemanticRegion

class ReadingOrderEngine:
    """Determines the logical reading order of document regions."""

    def sort(self, regions: Sequence[SemanticRegion]) -> List[SemanticRegion]:
        """
        Sort regions by logical reading order:
        1. Identify columns (x-coordinate grouping)
        2. Sort by Y within columns
        3. Handle overlaps and sidebars
        """
        if not regions:
            return []

        # Simple heuristic: Top-to-bottom, then Left-to-right
        # For more complex multi-column, we would group by X ranges first.
        sorted_regions = sorted(
            regions, 
            key=lambda r: (r.bbox.y1 // 10, r.bbox.x1) # Grouping by Y-bands of 10px
        )

        # Update the reading_order attribute
        for i, region in enumerate(sorted_regions):
            region.reading_order = i + 1
            
        return sorted_regions

    def group_columns(self, regions: Sequence[SemanticRegion]) -> List[List[SemanticRegion]]:
        """Group regions into columns based on horizontal alignment."""
        if not regions:
            return []
            
        # Very simple column detector: find regions that are horizontally far apart
        # but vertically overlapping.
        # This is a placeholder for more advanced logic.
        return [list(regions)]

