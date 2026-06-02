"""
Table structure recognition and cell grouping.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from app.ocr_pipeline.core.models.region import SemanticRegion
from app.ocr_pipeline.core.models.bbox import BBox
from app.ocr_pipeline.core.block_types import BlockType

@dataclass(slots=True)
class TableCell:
    region: SemanticRegion
    row_index: int
    col_index: int
    row_span: int = 1
    col_span: int = 1

class TableUnderstanding:
    """Analyzes regions within a table_section to reconstruct rows and columns."""

    def analyze_table(self, table_region: SemanticRegion, cell_regions: Sequence[SemanticRegion]) -> Dict[str, Any]:
        """Group sub-regions into a grid structure."""
        if not cell_regions:
            return {"rows": [], "cols_count": 0}

        # 1. Sort cells by Y, then X
        sorted_cells = sorted(cell_regions, key=lambda r: (r.bbox.y1, r.bbox.x1))

        # 2. Identify rows (regions with similar Y)
        rows: List[List[SemanticRegion]] = []
        if sorted_cells:
            current_row = [sorted_cells[0]]
            for i in range(1, len(sorted_cells)):
                prev = sorted_cells[i-1]
                curr = sorted_cells[i]
                
                # If vertical distance is small, same row
                if abs(curr.bbox.y1 - prev.bbox.y1) < 15:
                    current_row.append(curr)
                else:
                    rows.append(sorted(current_row, key=lambda r: r.bbox.x1))
                    current_row = [curr]
            rows.append(sorted(current_row, key=lambda r: r.bbox.x1))

        return {
            "rows": [[r.text for r in row] for row in rows],
            "row_count": len(rows),
            "max_cols": max(len(row) for row in rows) if rows else 0
        }

