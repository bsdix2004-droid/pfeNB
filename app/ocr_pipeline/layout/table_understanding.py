"""Table structure recognition and column-aware parsing for OCR tables."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple
import re

from app.ocr_pipeline.core.ocr_engine import TextBlock


def parse_ocr_table(block: TextBlock) -> Dict[str, Any]:
    """Parse a table_section TextBlock into structured columns and rows.

    Uses line bboxes to detect column boundaries when each cell is a
    separate OCR line.  Falls back to whitespace splitting (2+ spaces /
    tabs) for row-level OCR lines.
    """
    if not block.lines:
        return {"columns": [], "rows": [], "row_count": 0, "col_count": 0}

    line_bboxes = getattr(block, "line_bboxes", None) or []
    lines = [line.strip() for line in block.lines if line.strip()]
    if not lines:
        return {"columns": [], "rows": [], "row_count": 0, "col_count": 0}

    bbox_ok = len(line_bboxes) == len(block.lines) and _looks_like_cells(line_bboxes)

    if bbox_ok:
        return _parse_by_bbox(lines, line_bboxes, block.lines)
    else:
        return _parse_by_spacing(lines)


def _looks_like_cells(bboxes: List[Tuple[int, int, int, int]]) -> bool:
    """Heuristic: cells are narrow relative to page width (<=30% each)."""
    if len(bboxes) < 2:
        return False
    widths = [b[2] - b[0] for b in bboxes if b[2] > b[0]]
    if not widths:
        return False
    # If the average width is small, these are individual cells
    return (sum(widths) / len(widths)) < 400


def _parse_by_bbox(
    lines: List[str],
    bboxes: List[Tuple[int, int, int, int]],
    raw_lines: List[str],
) -> Dict[str, Any]:
    """Group OCR lines into columns by their X-position clusters."""
    # Collect all x-centers
    centers = [(b[0] + b[2]) / 2 for b in bboxes]
    x1_vals = [b[0] for b in bboxes]
    x2_vals = [b[2] for b in bboxes]

    # Build column boundaries from unique x-starts and x-ends
    all_x = sorted(set(x1_vals + x2_vals))
    # Merge close boundaries (within 15px)
    merged = _cluster(all_x, 15)
    boundaries = merged  # these are the column separator x-positions

    if len(boundaries) < 2:
        return _parse_by_spacing(lines)

    # For each line, determine which column it belongs to by its center
    col_for_line = []
    for c in centers:
        col = 0
        for i in range(1, len(boundaries)):
            if c >= boundaries[i - 1] and c < boundaries[i]:
                col = i - 1
                break
        col_for_line.append(col)

    # Group into rows by Y-position
    y_positions = [(b[1] + b[3]) / 2 for b in bboxes]
    row_groups = _group_rows(y_positions, 20)

    # Build grid
    grid: List[List[str]] = []
    for row_indices in row_groups:
        row = [""] * (len(boundaries) - 1)
        for idx in row_indices:
            col = col_for_line[idx]
            if 0 <= col < len(row):
                if row[col]:
                    row[col] += " "
                row[col] += raw_lines[idx].strip()
        grid.append([cell.strip() for cell in row])

    if not grid:
        return _parse_by_spacing(lines)

    return _build_result(grid)


def _parse_by_spacing(lines: List[str]) -> Dict[str, Any]:
    """Split table rows by 2+ spaces or tabs."""
    parsed = [re.split(r"\s{2,}|\t", line) for line in lines]
    parsed = [[cell.strip() for cell in row] for row in parsed if any(cell.strip() for cell in row)]
    return _build_result(parsed)


def _build_result(rows: List[List[str]]) -> Dict[str, Any]:
    if not rows:
        return {"columns": [], "rows": [], "row_count": 0, "col_count": 0}
    col_count = max(len(r) for r in rows)
    # Normalise all rows to col_count
    normalised = [row + [""] * (col_count - len(row)) for row in rows]
    return {
        "columns": normalised[0],
        "rows": normalised[1:] if len(normalised) > 1 else [],
        "row_count": max(0, len(normalised) - 1),
        "col_count": col_count,
    }


def _cluster(values: List[float], threshold: float) -> List[float]:
    """Merge close values into cluster centroids."""
    if not values:
        return []
    clusters = [[values[0]]]
    for v in values[1:]:
        if v - clusters[-1][-1] <= threshold:
            clusters[-1].append(v)
        else:
            clusters.append([v])
    return [sum(c) / len(c) for c in clusters]


def _group_rows(ys: List[float], threshold: float) -> List[List[int]]:
    """Group line indices by Y proximity."""
    indexed = sorted(enumerate(ys), key=lambda x: x[1])
    groups: List[List[int]] = []
    if not indexed:
        return groups
    current = [indexed[0][0]]
    current_y = indexed[0][1]
    for idx, y in indexed[1:]:
        if abs(y - current_y) <= threshold:
            current.append(idx)
        else:
            groups.append(sorted(current))
            current = [idx]
            current_y = y
    groups.append(sorted(current))
    return groups
