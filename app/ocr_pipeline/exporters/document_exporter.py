"""Assemble the three strict JSON output files."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from html import escape
from typing import Any, Dict, List, Sequence

from app.ocr_pipeline.core.detector import DocumentInfo
from app.ocr_pipeline.core.ocr_engine import OCRCandidateResult, TextBlock
from app.ocr_pipeline.utils.template_loader import TemplateLoader


class DocumentExporter:
    """Build document_ocr.json and document_vision.json payloads."""

    def build_ocr_output(
        self,
        source: Path,
        doc_info: DocumentInfo,
        blocks: Sequence[TextBlock],
        preprocessing: Dict[str, Any],
        ocr_selection: OCRCandidateResult,
        layout_data: Dict[str, Any],
        original_size: List[int],
        preprocessed_size: List[int],
        low_confidence_threshold: float = 0.75,
    ) -> Dict[str, Any]:
        avg_conf = sum(block.avg_confidence for block in blocks) / len(blocks) if blocks else 0.0
        total_lines = len(ocr_selection.lines)
        low_confidence_count = sum(1 for line in ocr_selection.lines if line.low_confidence)
        high_confidence_count = total_lines - low_confidence_count
        semantic_blocks = [self.block_to_dict(block) for block in blocks]
        uncertain_lines = [
            line.to_dict(index)
            for index, line in enumerate(ocr_selection.lines)
            if line.low_confidence
        ]
        selected_index = int(preprocessing.get("selected_index", 0)) if isinstance(preprocessing.get("selected_index", 0), int) else 0
        preprocessing_candidates = preprocessing.get("candidates", [])
        selected_candidate = (
            preprocessing_candidates[selected_index]
            if isinstance(preprocessing_candidates, list) and 0 <= selected_index < len(preprocessing_candidates)
            else {}
        )
        semantic_text = self._semantic_text(blocks)
        engine_name = "SVG vector text extractor" if ocr_selection.engine_version == "svg_text_extractor" else "PP-OCRv5"
        return {
            "document_type": doc_info.doc_type,
            "language": doc_info.language,
            "document_confidence": round(float(doc_info.confidence), 4),
            "language_probabilities": doc_info.language_probabilities,
            "classification_scores": doc_info.classification_scores,
            "classification_evidence": doc_info.evidence,
            "processing_timestamp": datetime.now(timezone.utc).isoformat(),
            "image_metadata": {
                "source_file": source.name,
                "original_size": original_size,
                "preprocessed_size": preprocessed_size,
                "quality_score": round(float(selected_candidate.get("score", 0.0)), 4) if isinstance(selected_candidate, dict) else 0.0,
            },
            "ocr_metadata": {
                "engine": engine_name,
                "engine_version": ocr_selection.engine_version,
                "total_lines": total_lines,
                "high_confidence_lines": high_confidence_count,
                "total_blocks": len(blocks),
                "char_count": ocr_selection.char_count,
                "avg_confidence": round(float(avg_conf), 4),
                "candidate_selection_score": round(float(ocr_selection.score), 4),
                "low_confidence_count": low_confidence_count,
                "low_confidence_threshold": low_confidence_threshold,
                "selected_preprocessing_candidate": ocr_selection.candidate_name,
                "requires_review": low_confidence_count > 0,
            },
            "preprocessing": preprocessing,
            "layout_source": layout_data.get("vision_source", "pp_doclayout"),
            "ocr_lines": [line.to_dict(index) for index, line in enumerate(ocr_selection.lines)],
            "uncertain_ocr_lines": uncertain_lines,
            "semantic_blocks": semantic_blocks,
            "full_text": semantic_text,
            "verified_text": self._semantic_text(blocks, verified_only=True, threshold=low_confidence_threshold),
            "raw_ocr_text_geometry_order": "\n".join(line.text for line in sorted(ocr_selection.lines, key=lambda item: (item.y, item.x))),
        }

    def build_vision_output(self, layout_data: Dict[str, Any], blocks: Sequence[TextBlock] | None = None) -> Dict[str, Any]:
        regions = layout_data.get("detected_regions", []) if isinstance(layout_data, dict) else []
        semantic_regions = [self.block_to_region(block) for block in blocks or []]
        detected = [
            {
                "region_type": region.get("region_type", "unknown"),
                "bbox": region.get("bbox", []),
                "content_preview": self._content_preview(region),
            }
            for region in regions
            if isinstance(region, dict)
        ]
        if not detected:
            detected = semantic_regions
        document_layout = layout_data.get("document_layout", "unknown") if isinstance(layout_data, dict) else "unknown"
        if document_layout == "unknown" and detected:
            document_layout = self._infer_document_layout(detected)
        return {
            "vision_source": layout_data.get("vision_source", "pp_doclayout") if isinstance(layout_data, dict) else "pp_doclayout",
            "available": bool(layout_data.get("available", False) or semantic_regions) if isinstance(layout_data, dict) else bool(semantic_regions),
            "document_layout": document_layout,
            "detected_regions": detected,
            "semantic_regions": semantic_regions,
            "layout_summary": layout_data.get("layout_summary", "") if isinstance(layout_data, dict) else "",
        }

    def block_to_dict(self, block: TextBlock) -> Dict[str, Any]:
        return {
            "block_id": block.id,
            "block_type": block.block_type,
            "text": block.text,
            "bbox": list(block.bbox),
            "reading_order": block.reading_order,
            "lines": list(block.lines),
            "confidence_avg": round(float(block.avg_confidence), 4),
            "low_confidence_count": int(block.metadata.get("low_confidence_count", 0)),
            "requires_review": int(block.metadata.get("low_confidence_count", 0)) > 0,
            "language": block.language,
            "metadata": dict(block.metadata),
        }

    def block_to_region(self, block: TextBlock) -> Dict[str, Any]:
        return {
            "region_type": block.block_type,
            "bbox": list(block.bbox),
            "text": block.text,
            "confidence": round(float(block.avg_confidence), 4),
            "reading_order": block.reading_order,
            "column_role": block.metadata.get("column_role", "column"),
            "content_preview": block.text.replace("\n", " ")[:160],
        }

    def _content_preview(self, region: Dict[str, Any]) -> str:
        content = region.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict):
                return str(first.get("text", ""))[:120]
            return str(first)[:120]
        if isinstance(content, dict):
            return str(content)[:120]
        return ""

    def _infer_document_layout(self, regions: Sequence[Dict[str, Any]]) -> str:
        if any(region.get("column_role") == "sidebar" for region in regions):
            return "sidebar-main"
        centers = []
        for region in regions:
            bbox = region.get("bbox", [])
            if isinstance(bbox, list) and len(bbox) == 4:
                centers.append((float(bbox[0]) + float(bbox[2])) / 2.0)
        if len(centers) >= 4 and max(centers) - min(centers) > 420:
            return "two-column"
        return "single-column"

    def _semantic_text(self, blocks: Sequence[TextBlock], verified_only: bool = False, threshold: float = 0.75) -> str:
        parts: List[str] = []
        for block in sorted(blocks, key=lambda item: item.reading_order):
            lines: List[str] = []
            for index, line in enumerate(block.lines):
                confidence = block.line_confidences[index] if index < len(block.line_confidences) else block.avg_confidence
                if verified_only and confidence < threshold:
                    continue
                if line.strip():
                    lines.append(line)
            if lines:
                parts.append("\n".join(lines))
        return "\n\n".join(parts)

    def build_layout_viewer_html(
        self,
        image_filename: str,
        blocks: Sequence[TextBlock],
        image_size: Sequence[int],
    ) -> str:
        width = int(image_size[0]) if image_size else 1
        height = int(image_size[1]) if len(image_size) > 1 else 1
        boxes = []
        for block in blocks:
            x1, y1, x2, y2 = block.bbox
            left = x1 / max(width, 1) * 100
            top = y1 / max(height, 1) * 100
            box_width = max(0, x2 - x1) / max(width, 1) * 100
            box_height = max(0, y2 - y1) / max(height, 1) * 100
            boxes.append(
                f'<button class="box {escape(block.block_type)}" '
                f'style="left:{left:.4f}%;top:{top:.4f}%;width:{box_width:.4f}%;height:{box_height:.4f}%;" '
                f'data-title="Block {block.id} - {escape(block.block_type)}" '
                f'data-confidence="{block.avg_confidence:.3f}" '
                f'data-text="{escape(block.text)}">'
                f'<span>{block.id}</span></button>'
            )

        return TemplateLoader.load("layout_viewer.html").format(
            width=width,
            image_filename=escape(image_filename),
            boxes="".join(boxes),
        )

