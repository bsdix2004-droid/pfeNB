"""Semantic reconstruction: OCR lines to layout-aware document blocks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Sequence, Tuple
import re
import unicodedata

from app.ocr_pipeline.ocr_config import (
    COL_SEP_BALANCE_WEIGHT,
    COL_SEP_CROSSING_PENALTY,
    COL_SEP_GAP_WEIGHT,
    COL_SEP_MIN_GAP_ABS,
    COL_SEP_MIN_GAP_RATIO,
    COL_SEP_SEARCH_START_RATIO,
    COL_SEP_SEARCH_STOP_RATIO,
    COL_SEP_SIDEBAR_GROWTH_RATIO,
    COL_SEP_SIDEBAR_SPAN_RATIO,
    COL_SEP_SIDEBAR_X_RATIO,
)
from app.ocr_pipeline.core.block_types import BlockType
from app.ocr_pipeline.core.constants import BULLET_RE, DATE_RANGE_RE as DATE_RE, SECTION_ALIASES, SECTION_TYPES
from app.ocr_pipeline.language.detector import detect_language


@dataclass(slots=True)
class OCRLine:
    text: str
    polygon: List[Tuple[float, float]]
    bbox: Tuple[int, int, int, int]
    confidence: float
    source_index: int = 0
    language: str = "unknown"
    low_confidence: bool = False

    @property
    def x(self) -> int:
        return self.bbox[0]

    @property
    def y(self) -> int:
        return self.bbox[1]

    @property
    def right(self) -> int:
        return self.bbox[2]

    @property
    def bottom(self) -> int:
        return self.bbox[3]

    @property
    def width(self) -> int:
        return max(1, self.right - self.x)

    @property
    def height(self) -> int:
        return max(1, self.bottom - self.y)

    @property
    def center_x(self) -> float:
        return self.x + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.y + self.height / 2.0

    def to_dict(self, line_index: int | None = None) -> Dict[str, object]:
        data: Dict[str, object] = {
            "text": self.text,
            "bbox": list(self.bbox),
            "confidence": round(float(self.confidence), 4),
            "low_confidence": self.low_confidence,
            "language": self.language,
        }
        if line_index is not None:
            data["line_index"] = line_index
        return data


@dataclass(slots=True)
class SemanticBlockCandidate:
    text: str
    lines: List[OCRLine]
    bbox: Tuple[int, int, int, int]
    avg_confidence: float
    block_type: str = BlockType.PARAGRAPH
    metadata: Dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ColumnGroup:
    lines: List[OCRLine]
    column_id: int
    role: str = "column"


def build_semantic_blocks(
    ocr_lines: Sequence[OCRLine],
    layout_blocks: Sequence[Dict[str, object]] | None = None,
    page_shape: Tuple[int, int] | None = None,
    direction: str = "ltr",
) -> List[SemanticBlockCandidate]:
    lines = [line for line in ocr_lines if line.text and line.text.strip()]
    if not lines:
        return []
    ordered = order_lines_geometry(lines, page_shape=page_shape, direction=direction)
    if layout_blocks:
        blocks = group_by_layout(ordered, layout_blocks)
    else:
        blocks = group_by_geometry(ordered, page_shape=page_shape, direction=direction)
    return assign_reading_order(blocks, page_shape=page_shape)


def group_by_proximity(ocr_lines: Sequence[OCRLine]) -> List[SemanticBlockCandidate]:
    return group_by_geometry(ocr_lines)


def group_by_geometry(
    ocr_lines: Sequence[OCRLine],
    page_shape: Tuple[int, int] | None = None,
    direction: str = "ltr",
) -> List[SemanticBlockCandidate]:
    if not ocr_lines:
        return []

    columns = split_columns_with_roles(ocr_lines, page_shape=page_shape)
    blocks: List[SemanticBlockCandidate] = []
    for column in columns:
        ordered = sorted(column.lines, key=lambda line: (line.y, -line.right if direction == "rtl" else line.x))
        blocks.extend(group_column_lines(ordered, column_id=column.column_id, column_role=column.role, page_shape=page_shape))
    return blocks


def split_columns(
    lines: Sequence[OCRLine],
    page_shape: Tuple[int, int] | None = None,
) -> List[List[OCRLine]]:
    return [column.lines for column in split_columns_with_roles(lines, page_shape=page_shape)]


def split_columns_with_roles(
    lines: Sequence[OCRLine],
    page_shape: Tuple[int, int] | None = None,
) -> List[ColumnGroup]:
    if len(lines) < 8:
        return [ColumnGroup(list(lines), 0, "main")]

    page_width = page_shape[1] if page_shape else max(line.right for line in lines)
    separator = find_column_separator(lines, page_width)

    if separator is None:
        centers = sorted(line.center_x for line in lines)
        gaps = [(centers[i + 1] - centers[i], i) for i in range(len(centers) - 1)]
        if not gaps:
            return [ColumnGroup(list(lines), 0, "main")]

        largest_gap, gap_index = max(gaps, key=lambda item: item[0])
        min_gap = max(90.0, page_width * 0.10)
        if largest_gap < min_gap:
            return [ColumnGroup(list(lines), 0, "main")]
        separator = (centers[gap_index] + centers[gap_index + 1]) / 2.0

    left = [line for line in lines if line.center_x <= separator]
    right = [line for line in lines if line.center_x > separator]
    if len(left) < 3 or len(right) < 3:
        return [ColumnGroup(list(lines), 0, "main")]

    left_role, right_role = classify_column_roles(left, right, page_width)
    return [ColumnGroup(left, 0, left_role), ColumnGroup(right, 1, right_role)]


def find_column_separator(lines: Sequence[OCRLine], page_width: float) -> float | None:
    """Find a low-occupancy vertical lane that separates sidebar/main columns."""
    if not lines or page_width <= 0:
        return None

    start = page_width * COL_SEP_SEARCH_START_RATIO
    stop = page_width * COL_SEP_SEARCH_STOP_RATIO
    best_separator: float | None = None
    best_score = float("-inf")

    for step in range(41):
        separator = start + (stop - start) * (step / 40.0)
        left = [line for line in lines if line.center_x <= separator]
        right = [line for line in lines if line.center_x > separator]
        if len(left) < 3 or len(right) < 3:
            continue

        crossing_weight = 0.0
        for line in lines:
            if line.x < separator < line.right:
                crossing_weight += COL_SEP_BALANCE_WEIGHT if line.width > page_width * 0.62 else 1.0
        crossing_ratio = crossing_weight / max(1.0, float(len(lines)))
        gap = min(line.x for line in right) - max(line.right for line in left)
        if gap < max(COL_SEP_MIN_GAP_ABS, page_width * COL_SEP_MIN_GAP_RATIO):
            continue

        balance = min(len(left), len(right)) / max(len(left), len(right))
        score = (gap / page_width) * COL_SEP_GAP_WEIGHT + balance * COL_SEP_BALANCE_WEIGHT - crossing_ratio * COL_SEP_CROSSING_PENALTY
        if score > best_score:
            best_score = score
            best_separator = separator

    return best_separator


def classify_column_roles(left: Sequence[OCRLine], right: Sequence[OCRLine], page_width: float) -> Tuple[str, str]:
    left_span = _column_span(left)
    right_span = _column_span(right)
    right_min_x = min(line.x for line in right)
    left_text = "\n".join(line.text.lower() for line in left)
    sidebar_keywords = ("contact", "skills", "languages", "hobbies", "address", "phone", "email", "coordonnées", "compétences")
    left_has_sidebar_signals = sum(1 for keyword in sidebar_keywords if keyword in left_text) >= 2

    if (
        left_span < page_width * COL_SEP_SIDEBAR_SPAN_RATIO
        and right_span > left_span * COL_SEP_SIDEBAR_GROWTH_RATIO
        and right_min_x > page_width * COL_SEP_SIDEBAR_X_RATIO
        and left_has_sidebar_signals
    ):
        return "sidebar", "main"
    return "column", "column"


def group_column_lines(
    lines: Sequence[OCRLine],
    column_id: int = 0,
    column_role: str = "column",
    page_shape: Tuple[int, int] | None = None,
) -> List[SemanticBlockCandidate]:
    if not lines:
        return []

    heights = sorted(line.height for line in lines)
    median_height = heights[len(heights) // 2]
    blocks: List[SemanticBlockCandidate] = []
    current: List[OCRLine] = []
    current_section: str | None = None

    for line in lines:
        section = detect_section(line.text)
        if not current:
            current = [line]
            current_section = section
            continue

        previous = current[-1]
        vertical_gap = line.y - previous.bottom
        same_row = abs(line.center_y - previous.center_y) <= max(previous.height, line.height) * 0.65
        new_section = section is not None
        large_gap = vertical_gap > max(median_height * 1.8, 34)
        alignment_reset = (
            vertical_gap > median_height * 0.7
            and abs(line.x - min(item.x for item in current)) > max(60, median_height * 2.5)
            and not BULLET_RE.match(line.text)
            and not same_row
        )

        if new_section or large_gap or alignment_reset:
            blocks.append(make_block(current, len(blocks) + 1, column_id=column_id, column_role=column_role, forced_section=current_section))
            current = [line]
            current_section = section
        else:
            current.append(line)
            current_section = current_section or section

    if current:
        blocks.append(make_block(current, len(blocks) + 1, column_id=column_id, column_role=column_role, forced_section=current_section))

    return merge_continuations(blocks, page_shape=page_shape)


def merge_continuations(
    blocks: Sequence[SemanticBlockCandidate],
    page_shape: Tuple[int, int] | None = None,
) -> List[SemanticBlockCandidate]:
    if not blocks:
        return []

    merged: List[SemanticBlockCandidate] = []
    for block in blocks:
        if not merged:
            merged.append(block)
            continue
        previous = merged[-1]
        prev_section = str(previous.metadata.get("section", ""))
        should_merge = (
            prev_section in {"experience", "education", "summary", "legal_clause"}
            and block.block_type in {BlockType.PARAGRAPH, BlockType.LIST_ITEM, BlockType.TABLE_SECTION, BlockType.KEY_VALUE}
            and block.bbox[1] - previous.bbox[3] < max(60, _median_height(previous.lines) * 2.2)
        )
        if should_merge:
            lines = previous.lines + block.lines
            merged[-1] = make_block(
                lines,
                int(previous.metadata.get("block_id", len(merged))),
                column_id=int(previous.metadata.get("column_id", 0)),
                column_role=str(previous.metadata.get("column_role", "column")),
                forced_section=prev_section,
            )
        else:
            merged.append(block)
    return merged


def group_by_layout(
    ocr_lines: Sequence[OCRLine],
    layout_blocks: Sequence[Dict[str, object]],
) -> List[SemanticBlockCandidate]:
    result_blocks: List[SemanticBlockCandidate] = []
    used_line_ids: set[int] = set()

    sorted_regions = sorted(layout_blocks, key=lambda region: _region_sort_key(region))
    for region in sorted_regions:
        bbox = region.get("bbox") if isinstance(region, dict) else None
        if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
            continue
        rx1, ry1, rx2, ry2 = [int(v) for v in bbox[:4]]
        lines_in_region: List[OCRLine] = []

        for line in ocr_lines:
            if id(line) in used_line_ids:
                continue
            cx = (line.x + line.right) / 2
            cy = (line.y + line.bottom) / 2
            if rx1 <= cx <= rx2 and ry1 <= cy <= ry2:
                lines_in_region.append(line)
                used_line_ids.add(id(line))

        if lines_in_region:
            lines_in_region.sort(key=lambda line: (line.y, line.x))
            block = make_block(lines_in_region, len(result_blocks) + 1)
            region_type = str(region.get("region_type") or region.get("type") or block.block_type).lower()
            block.block_type = normalize_region_type(region_type, block.text)
            block.metadata["layout_source"] = "vision"
            result_blocks.append(block)

    remaining = [line for line in ocr_lines if id(line) not in used_line_ids]
    if remaining:
        extra_blocks = group_by_geometry(remaining)
        for block in extra_blocks:
            block.metadata["layout_source"] = "geometry_fallback"
            result_blocks.append(block)

    return result_blocks


def make_block(
    lines: Sequence[OCRLine],
    block_id: int,
    column_id: int = 0,
    column_role: str = "column",
    forced_section: str | None = None,
) -> SemanticBlockCandidate:
    ordered = sorted(lines, key=lambda line: (line.y, line.x))
    text = "\n".join(line.text for line in ordered)
    x1 = min(line.x for line in ordered)
    y1 = min(line.y for line in ordered)
    x2 = max(line.right for line in ordered)
    y2 = max(line.bottom for line in ordered)
    avg_conf = sum(line.confidence for line in ordered) / len(ordered)
    section = forced_section or detect_section(ordered[0].text) or detect_section(text)
    block_type = classify_block(text, ordered, section)

    return SemanticBlockCandidate(
        text=text,
        lines=list(ordered),
        bbox=(x1, y1, x2, y2),
        avg_confidence=round(float(avg_conf), 4),
        block_type=block_type,
        metadata={
            "block_id": block_id,
            "line_count": len(ordered),
            "low_confidence_count": sum(1 for line in ordered if line.low_confidence),
            "language": detect_language(text),
            "column_id": column_id,
            "column_role": column_role,
            "section": section,
            "font_scale": round(float(_median_height(ordered)), 3),
            "contains_date_range": bool(DATE_RE.search(text)),
        },
    )


def classify_block(text: str, lines: Sequence[OCRLine], section: str | None = None) -> str:
    stripped = text.strip()
    if section:
        return SECTION_TYPES.get(section, f"{section}_section")

    avg_height = sum(line.height for line in lines) / len(lines)
    words = stripped.split()
    if len(words) <= 8 and (avg_height > 22 or stripped.isupper()):
        return BlockType.HEADING
    if all(BULLET_RE.match(line.text) for line in lines if line.text.strip()):
        return BlockType.LIST_ITEM
    if len(lines) >= 3 and _looks_tabular(lines):
        return BlockType.TABLE_SECTION
    if re.search(r":\s+\S", stripped) or re.search(r"\.{4,}", stripped):
        return BlockType.KEY_VALUE
    return BlockType.PARAGRAPH


def detect_section(text: str) -> str | None:
    normalized = _normalize_label(text)
    if not normalized:
        return None
    for section, aliases in SECTION_ALIASES.items():
        for alias in aliases:
            alias_norm = _normalize_label(alias)
            if normalized == alias_norm or normalized.startswith(alias_norm + " "):
                return section
    return None


def normalize_region_type(region_type: str, text: str) -> str:
    section = detect_section(text)
    if section:
        return SECTION_TYPES.get(section, region_type)
    mapping = {
        "title": BlockType.HEADING,
        "text": BlockType.PARAGRAPH,
        "table": BlockType.TABLE_SECTION,
        "figure": BlockType.FIGURE,
        "header": BlockType.HEADER,
        "footer": BlockType.FOOTER,
        "stamp": BlockType.SIGNATURE,
        "seal": BlockType.SIGNATURE,
        "signature": BlockType.SIGNATURE,
    }
    return mapping.get(region_type.lower(), region_type.lower())


def order_lines_geometry(
    lines: Sequence[OCRLine],
    page_shape: Tuple[int, int] | None = None,
    direction: str = "ltr",
) -> List[OCRLine]:
    columns = split_columns(lines, page_shape=page_shape)
    ordered: List[OCRLine] = []
    for column in columns:
        ordered.extend(sorted(column, key=lambda line: (line.y, -line.right if direction == "rtl" else line.x)))
    return ordered


def assign_reading_order(
    blocks: Sequence[SemanticBlockCandidate],
    page_shape: Tuple[int, int] | None = None,
) -> List[SemanticBlockCandidate]:
    if not blocks:
        return []

    page_height = page_shape[0] if page_shape else max(block.bbox[3] for block in blocks)
    page_width = page_shape[1] if page_shape else max(block.bbox[2] for block in blocks)
    has_sidebar = any(block.metadata.get("column_role") == "sidebar" for block in blocks)

    def key(block: SemanticBlockCandidate) -> Tuple[int, int, int, int]:
        top_band = block.bbox[1] < page_height * 0.16
        likely_title = block.block_type == BlockType.HEADING and block.bbox[2] - block.bbox[0] > page_width * 0.18
        title_rank = 0 if top_band and likely_title else 1
        column_id = int(block.metadata.get("column_id", 0))
        role = str(block.metadata.get("column_role", "column"))
        if has_sidebar:
            role_rank = 2 if role == "sidebar" else 1
        else:
            role_rank = column_id + 1
        return (title_rank, role_rank, block.bbox[1], block.bbox[0])

    ordered = sorted(blocks, key=key)
    for index, block in enumerate(ordered, 1):
        block.metadata["block_id"] = index
        block.metadata["reading_order"] = index
    return ordered


def _looks_tabular(lines: Sequence[OCRLine]) -> bool:
    x_positions = sorted({round(line.x / 25) * 25 for line in lines})
    return len(x_positions) >= 3 and any(re.search(r"\d", line.text) for line in lines)


def _median_height(lines: Sequence[OCRLine]) -> float:
    values = sorted(line.height for line in lines)
    return float(values[len(values) // 2]) if values else 0.0


def _column_span(lines: Sequence[OCRLine]) -> float:
    if not lines:
        return 0.0
    return float(max(line.right for line in lines) - min(line.x for line in lines))


def _region_sort_key(region: Dict[str, object]) -> Tuple[int, int]:
    bbox = region.get("bbox", [])
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return int(bbox[1]), int(bbox[0])
    return 0, 0


def _normalize_label(text: str) -> str:
    decomposed = unicodedata.normalize("NFKD", text.lower())
    asciiish = "".join(ch for ch in decomposed if not unicodedata.combining(ch))
    normalized = re.sub(r"[^a-z0-9 ]+", " ", asciiish)
    return re.sub(r"\s+", " ", normalized).strip()


class SemanticReconstructor:
    """OCR-line ordering and block reconstruction."""

    def reconstruct(
        self,
        lines: Sequence[OCRLine],
        direction: str = "ltr",
        page_shape: Tuple[int, int] | None = None,
        layout_blocks: Sequence[Dict[str, object]] | None = None,
    ) -> List[SemanticBlockCandidate]:
        return build_semantic_blocks(lines, layout_blocks=layout_blocks, page_shape=page_shape, direction=direction)

    def order_lines(self, lines: List[OCRLine], direction: str = "ltr") -> List[OCRLine]:
        if direction == "rtl":
            return sorted(lines, key=lambda line: (line.y, -line.right))
        return order_lines_geometry(lines, direction=direction)

    def clean_text(self, blocks: Sequence[SemanticBlockCandidate]) -> str:
        return "\n\n".join(block.text.strip() for block in blocks if block.text.strip())

