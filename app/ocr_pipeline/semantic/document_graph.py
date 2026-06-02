"""Document graph over semantic document regions."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import groupby
from typing import Dict, Iterator, List, Optional, Sequence

from app.ocr_pipeline.core.models.region import SemanticRegion
from app.ocr_pipeline.utils.logger import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class GraphEdge:
    source_id: str
    target_id: str
    relation: str
    weight: float = 1.0


class DocumentGraph:
    """Lightweight adjacency-list graph over SemanticRegions."""

    def __init__(self) -> None:
        self._regions: Dict[str, SemanticRegion] = {}
        self._edges: List[GraphEdge] = []

    def build(self, regions: Sequence[SemanticRegion]) -> None:
        self._regions = {region.id: region for region in regions}
        self._edges.clear()
        self._add_reading_order_edges(regions)
        self._add_spatial_edges(regions)
        self._add_section_edges(regions)
        self._populate_neighbors()
        LOGGER.info("Document graph: %d nodes, %d edges", len(self._regions), len(self._edges))

    def neighbors(self, region_id: str, relation: str | None = None) -> List[SemanticRegion]:
        results: List[SemanticRegion] = []
        for edge in self._edges:
            if edge.source_id == region_id and (relation is None or edge.relation == relation):
                target = self._regions.get(edge.target_id)
                if target:
                    results.append(target)
        return results

    def edges_of(self, region_id: str) -> List[GraphEdge]:
        return [edge for edge in self._edges if edge.source_id == region_id]

    def get(self, region_id: str) -> Optional[SemanticRegion]:
        return self._regions.get(region_id)

    def __iter__(self) -> Iterator[SemanticRegion]:
        return iter(self._regions.values())

    def __len__(self) -> int:
        return len(self._regions)

    def _add_reading_order_edges(self, regions: Sequence[SemanticRegion]) -> None:
        ordered = sorted(regions, key=lambda region: region.reading_order)
        for index in range(len(ordered) - 1):
            self._edges.append(GraphEdge(ordered[index].id, ordered[index + 1].id, "next_in_order"))

    def _add_spatial_edges(self, regions: Sequence[SemanticRegion]) -> None:
        for index, first in enumerate(regions):
            for second in regions[index + 1:]:
                relation = _spatial_relation(first, second)
                if relation:
                    self._edges.append(GraphEdge(first.id, second.id, relation))
                    self._edges.append(GraphEdge(second.id, first.id, _inverse(relation)))

    def _add_section_edges(self, regions: Sequence[SemanticRegion]) -> None:
        keyfn = lambda region: region.region_type.value.split("_")[0]
        sorted_regions = sorted(regions, key=keyfn)
        for _, group in groupby(sorted_regions, key=keyfn):
            group_list = list(group)
            for first, second in zip(group_list, group_list[1:]):
                self._edges.append(GraphEdge(first.id, second.id, "same_section", weight=0.8))

    def _populate_neighbors(self) -> None:
        for region in self._regions.values():
            region.neighbors = [edge.target_id for edge in self._edges if edge.source_id == region.id]


def _spatial_relation(first: SemanticRegion, second: SemanticRegion) -> Optional[str]:
    ax1, ay1, ax2, ay2 = first.bbox.to_tuple()
    bx1, by1, bx2, by2 = second.bbox.to_tuple()
    a_cy = (ay1 + ay2) / 2
    b_cy = (by1 + by2) / 2
    a_cx = (ax1 + ax2) / 2
    b_cx = (bx1 + bx2) / 2
    dy = abs(a_cy - b_cy)
    dx = abs(a_cx - b_cx)
    if dy > dx * 1.5:
        return "above" if a_cy < b_cy else "below"
    if dx > dy * 1.5:
        return "left_of" if a_cx < b_cx else "right_of"
    return None


def _inverse(relation: str) -> str:
    return {
        "above": "below",
        "below": "above",
        "left_of": "right_of",
        "right_of": "left_of",
    }.get(relation, relation)

