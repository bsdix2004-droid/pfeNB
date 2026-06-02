"""PP-OCRv5 text extraction with bbox/confidence preservation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence, Tuple
import math

import cv2
import numpy as np

from app.config import OCREngineConfig
from app.ocr_pipeline.ocr_config import (
    OCR_CLOSE_X_RATIO,
    OCR_CLOSE_Y_RATIO,
    OCR_IOU_DUPLICATE_THRESHOLD,
    OCR_SCORE_CONFIDENCE_WEIGHT,
    OCR_SCORE_DENSITY_LOG_BASE,
    OCR_SCORE_DENSITY_WEIGHT,
)
from app.ocr_pipeline.core.block_types import BlockType
from app.ocr_pipeline.core.detector import DocumentInfo
from app.ocr_pipeline.language.detector import detect_language
from app.ocr_pipeline.semantic.reconstructor import OCRLine, SemanticBlockCandidate, SemanticReconstructor
from app.ocr_pipeline.utils.logger import get_logger, suppress_external_output


LOGGER = get_logger(__name__)

_SCRIPT_TO_PADDLE_LANGS: Dict[str, List[str]] = {
    "arabic": ["ar"],
    "latin": ["en", "fr"],
    "cjk": ["ch"],
}


def _scripts_to_languages(
    scripts: Sequence[str],
    fallback: List[str],
) -> List[str]:
    """Map detected script names to PaddleOCR language codes."""
    seen: set[str] = set()
    result: List[str] = []
    for script in scripts:
        for lang in _SCRIPT_TO_PADDLE_LANGS.get(script, []):
            if lang not in seen:
                seen.add(lang)
                result.append(lang)
    return result or list(fallback)


@dataclass(slots=True)
class TextBlock:
    id: int
    text: str
    lines: List[str]
    bbox: Tuple[int, int, int, int]
    avg_confidence: float
    reading_order: int
    block_type: str = BlockType.PARAGRAPH
    line_bboxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    line_confidences: List[float] = field(default_factory=list)
    language: str = "unknown"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class OCRCandidateResult:
    candidate_name: str
    candidate_index: int
    blocks: List[TextBlock]
    lines: List[OCRLine]
    score: float
    avg_confidence: float
    raw_line_count: int
    char_count: int
    engine_version: str
    languages_used: List[str] = field(default_factory=list)
    fallback_languages_skipped: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_name": self.candidate_name,
            "candidate_index": self.candidate_index,
            "score": round(self.score, 4),
            "avg_confidence": round(self.avg_confidence, 4),
            "raw_line_count": self.raw_line_count,
            "char_count": self.char_count,
            "engine_version": self.engine_version,
            "languages_used": list(self.languages_used),
            "fallback_languages_skipped": list(self.fallback_languages_skipped),
        }


class PaddleOCREngine:
    """Runs English/French and Arabic PaddleOCR passes and keeps exact OCR text."""

    def __init__(self, config: OCREngineConfig) -> None:
        self.config = config
        self._engines: Dict[str, Any] = {}
        self._paddleocr_cls: Any | None = None
        self.reconstructor = SemanticReconstructor()

    def extract(self, image: np.ndarray, doc_info: DocumentInfo) -> List[TextBlock]:
        return self._extract_single(image, doc_info, "selected", 0).blocks

    def extract_best(
        self,
        candidates: Sequence[Any],
        doc_info: DocumentInfo,
        layout_blocks: Sequence[Dict[str, object]] | None = None,
        scripts: Sequence[str] | None = None,
    ) -> OCRCandidateResult:
        if not candidates:
            raise ValueError("No OCR candidates provided")

        limit = max(1, min(self.config.max_ocr_candidates, len(candidates))) if self.config.candidate_selection else 1
        primary_languages = [self.config.languages[0]] if self.config.languages else ["en"]
        if scripts:
            full_languages = _scripts_to_languages(scripts, list(self.config.languages))
        else:
            full_languages = list(self.config.languages or primary_languages)
        selection_languages = primary_languages if len(full_languages) > 1 and limit > 1 else full_languages

        results: List[OCRCandidateResult] = []
        for index, candidate in enumerate(candidates[:limit]):
            image = getattr(candidate, "image", candidate)
            name = str(getattr(candidate, "name", f"candidate_{index}"))
            try:
                results.append(self._extract_single(image, doc_info, name, index, layout_blocks, languages=selection_languages))
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                LOGGER.debug("OCR candidate '%s' failed: %s", name, exc)

        if not results:
            raise RuntimeError("All OCR candidates failed")

        best = max(results, key=lambda item: item.score)
        if selection_languages != full_languages and self._needs_multilingual_fallback(best):
            candidate = candidates[best.candidate_index]
            image = getattr(candidate, "image", candidate)
            best = self._extract_single(
                image,
                doc_info,
                best.candidate_name,
                best.candidate_index,
                layout_blocks,
                languages=full_languages,
            )
        elif selection_languages != full_languages:
            best.fallback_languages_skipped = [lang for lang in full_languages if lang not in selection_languages]
        LOGGER.info("OCR selected '%s' with %s lines", best.candidate_name, best.raw_line_count)
        return best

    def rebuild_blocks(
        self,
        result: OCRCandidateResult,
        doc_info: DocumentInfo,
        image_shape: Tuple[int, int],
        layout_blocks: Sequence[Dict[str, object]] | None = None,
    ) -> OCRCandidateResult:
        semantic_blocks = self.reconstructor.reconstruct(
            result.lines,
            direction=doc_info.direction,
            page_shape=image_shape[:2],
            layout_blocks=layout_blocks,
        )
        result.blocks = self._to_text_blocks(semantic_blocks)
        result.char_count = sum(len(block.text) for block in result.blocks)
        result.score = self._score_result(result.blocks, result.avg_confidence, result.raw_line_count, result.char_count)
        return result

    def _extract_single(
        self,
        image: np.ndarray,
        doc_info: DocumentInfo,
        candidate_name: str,
        candidate_index: int,
        layout_blocks: Sequence[Dict[str, object]] | None = None,
        languages: Sequence[str] | None = None,
    ) -> OCRCandidateResult:
        requested_languages = list(languages or self.config.languages)
        raw_results = []
        for language in requested_languages:
            try:
                engine = self._get_engine(language)
                raw_results.append(self._run_ocr(engine, image))
            except BaseException as exc:
                if isinstance(exc, KeyboardInterrupt):
                    raise
                LOGGER.debug("OCR pass failed for lang=%s on candidate '%s': %s", language, candidate_name, exc)

        all_lines: List[OCRLine] = []
        for raw_result in raw_results:
            all_lines.extend(self._parse_result(raw_result))

        lines = [
            line
            for line in self._merge_ocr_passes(all_lines)
            if line.confidence >= self.config.min_confidence
        ]
        for index, line in enumerate(lines):
            line.source_index = index
            line.language = detect_language(line.text)
            line.low_confidence = line.confidence < self.config.low_confidence_threshold

        semantic_blocks = self.reconstructor.reconstruct(
            lines,
            direction=doc_info.direction,
            page_shape=image.shape[:2],
            layout_blocks=layout_blocks,
        )
        blocks = self._to_text_blocks(semantic_blocks)
        avg_conf = float(np.mean([line.confidence for line in lines])) if lines else 0.0
        char_count = sum(len(block.text) for block in blocks)
        score = self._score_result(blocks, avg_conf, len(lines), char_count)
        return OCRCandidateResult(
            candidate_name=candidate_name,
            candidate_index=candidate_index,
            blocks=blocks,
            lines=lines,
            score=score,
            avg_confidence=avg_conf,
            raw_line_count=len(lines),
            char_count=char_count,
            engine_version=self._engine_version_label(),
            languages_used=requested_languages,
        )

    def _get_engine(self, lang: str) -> Any:
        cache_key = f"{lang}:{self.config.ocr_version}:cpu:{self.config.use_gpu}"
        if cache_key in self._engines:
            return self._engines[cache_key]

        paddleocr_cls = self._load_paddleocr()
        legacy_kwargs = {
            "use_angle_cls": self.config.use_angle_cls,
            "lang": lang,
            "use_gpu": False,
            "show_log": False,
        }
        modern_kwargs = {
            "lang": lang,
            "ocr_version": self.config.ocr_version,
            "device": "gpu:0" if self.config.use_gpu else "cpu",
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": self.config.use_angle_cls,
        }

        try:
            with suppress_external_output():
                engine = paddleocr_cls(**modern_kwargs)
        except TypeError:
            with suppress_external_output():
                engine = paddleocr_cls(**legacy_kwargs)
        except Exception as exc:
            try:
                with suppress_external_output():
                    engine = paddleocr_cls(**legacy_kwargs)
            except Exception as legacy_exc:
                raise RuntimeError(f"PaddleOCR initialization failed for lang={lang}: {exc}; {legacy_exc}") from legacy_exc

        self._engines[cache_key] = engine
        return engine

    def _load_paddleocr(self) -> Any:
        if self._paddleocr_cls is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore
            except Exception as exc:
                raise RuntimeError("PaddleOCR is not installed or failed to import") from exc
            self._paddleocr_cls = PaddleOCR
        return self._paddleocr_cls

    def _run_ocr(self, engine: Any, image: np.ndarray) -> Any:
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        with suppress_external_output():
            if hasattr(engine, "predict"):
                return engine.predict(image)
            return engine.ocr(image, cls=self.config.use_angle_cls)

    def _parse_result(self, result: Any) -> List[OCRLine]:
        if not result:
            return []
        modern = self._parse_modern_result(result)
        if modern:
            return modern
        return self._parse_legacy_result(result)

    def _parse_modern_result(self, result: Any) -> List[OCRLine]:
        records = result if isinstance(result, list) else [result]
        lines: List[OCRLine] = []
        source_index = 0

        for record in records:
            payload = self._record_to_mapping(record)
            if not payload:
                continue
            texts = self._first_present(payload, ("rec_texts", "texts", "ocr_texts"))
            scores = self._first_present(payload, ("rec_scores", "scores", "ocr_scores"))
            polygons = self._first_present(payload, ("rec_polys", "dt_polys", "polys"))
            boxes = self._first_present(payload, ("rec_boxes", "boxes"))
            if texts is None:
                continue

            texts_list = list(texts)
            scores_list = self._safe_list(scores, default=1.0, size=len(texts_list))
            polygons_list = self._safe_list(polygons, default=None, size=len(texts_list))
            boxes_list = self._safe_list(boxes, default=None, size=len(texts_list))

            for text, score, polygon, box in zip(texts_list, scores_list, polygons_list, boxes_list):
                if text is None:
                    continue
                poly = self._coerce_polygon(polygon, box)
                if not poly:
                    continue
                line = OCRLine(
                    text=str(text),
                    polygon=poly,
                    bbox=self._polygon_to_bbox(poly),
                    confidence=self._coerce_confidence(score),
                    source_index=source_index,
                )
                lines.append(line)
                source_index += 1
        return lines

    def _parse_legacy_result(self, result: Any) -> List[OCRLine]:
        pages = result if isinstance(result, list) else [result]
        if pages and isinstance(pages[0], list) and pages[0] and self._looks_like_legacy_line(pages[0][0]):
            page_lines = pages[0]
        else:
            page_lines = pages

        lines: List[OCRLine] = []
        for index, item in enumerate(page_lines):
            if not self._looks_like_legacy_line(item):
                continue
            polygon_raw, text_conf = item
            text, confidence = text_conf[0], text_conf[1]
            polygon = self._coerce_polygon(polygon_raw, None)
            if not polygon:
                continue
            lines.append(
                OCRLine(
                    text=str(text),
                    polygon=polygon,
                    bbox=self._polygon_to_bbox(polygon),
                    confidence=self._coerce_confidence(confidence),
                    source_index=index,
                )
            )
        return lines

    def _merge_ocr_passes(self, lines: Sequence[OCRLine]) -> List[OCRLine]:
        merged: List[OCRLine] = []
        for line in sorted(lines, key=lambda item: (item.y, item.x)):
            duplicate_index = self._find_duplicate(line, merged)
            if duplicate_index is None:
                merged.append(line)
                continue
            existing = merged[duplicate_index]
            if (line.confidence, len(line.text)) > (existing.confidence, len(existing.text)):
                merged[duplicate_index] = line
        return sorted(merged, key=lambda item: (item.y, item.x))

    def _find_duplicate(self, line: OCRLine, existing_lines: Sequence[OCRLine]) -> int | None:
        for index, existing in enumerate(existing_lines):
            if self._bbox_iou(line.bbox, existing.bbox) >= OCR_IOU_DUPLICATE_THRESHOLD:
                return index
            close_y = abs(line.center_y - existing.center_y) <= max(line.height, existing.height) * OCR_CLOSE_Y_RATIO
            close_x = abs(line.center_x - existing.center_x) <= max(line.width, existing.width) * OCR_CLOSE_X_RATIO
            if close_y and close_x:
                return index
        return None

    def _bbox_iou(self, a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
        ax1, ay1, ax2, ay2 = a
        bx1, by1, bx2, by2 = b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
        area_b = max(1, (bx2 - bx1) * (by2 - by1))
        return inter / max(1, area_a + area_b - inter)

    def _record_to_mapping(self, record: Any) -> Dict[str, Any]:
        if isinstance(record, dict):
            payload = record.get("res", record)
            return payload if isinstance(payload, dict) else {}
        for attr in ("json", "to_dict", "dict"):
            value = getattr(record, attr, None)
            if isinstance(value, dict):
                payload = value.get("res", value)
                return payload if isinstance(payload, dict) else {}
            if callable(value):
                try:
                    mapped = value()
                except TypeError:
                    continue
                if isinstance(mapped, dict):
                    payload = mapped.get("res", mapped)
                    return payload if isinstance(payload, dict) else {}
        payload = getattr(record, "res", None)
        return payload if isinstance(payload, dict) else {}

    def _first_present(self, payload: Dict[str, Any], keys: Iterable[str]) -> Any:
        for key in keys:
            if key in payload and payload[key] is not None:
                return payload[key]
        return None

    def _safe_list(self, value: Any, default: Any, size: int) -> List[Any]:
        if value is None:
            return [default for _ in range(size)]
        if isinstance(value, np.ndarray):
            value = value.tolist()
        values = list(value) if isinstance(value, (list, tuple)) else [value]
        if len(values) < size:
            values.extend([default for _ in range(size - len(values))])
        return values[:size]

    def _looks_like_legacy_line(self, item: Any) -> bool:
        return (
            isinstance(item, (list, tuple))
            and len(item) >= 2
            and isinstance(item[1], (list, tuple))
            and len(item[1]) >= 2
        )

    def _coerce_polygon(self, polygon: Any, box: Any) -> List[Tuple[float, float]]:
        if polygon is not None:
            arr = np.asarray(polygon, dtype=float)
            if arr.ndim == 2 and arr.shape[0] >= 4 and arr.shape[1] >= 2:
                return [(float(x), float(y)) for x, y in arr[:4, :2]]
        if box is not None:
            values = np.asarray(box, dtype=float).flatten().tolist()
            if len(values) >= 4:
                x1, y1, x2, y2 = values[:4]
                return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
        return []

    def _polygon_to_bbox(self, polygon: Sequence[Tuple[float, float]]) -> Tuple[int, int, int, int]:
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))

    def _coerce_confidence(self, value: Any) -> float:
        try:
            conf = float(value)
        except (TypeError, ValueError):
            conf = 1.0
        if conf > 1.0 and conf <= 100.0:
            conf = conf / 100.0
        if math.isnan(conf):
            return 0.0
        return max(0.0, min(1.0, conf))

    def _to_text_blocks(self, blocks: Sequence[SemanticBlockCandidate]) -> List[TextBlock]:
        text_blocks: List[TextBlock] = []
        for index, block in enumerate(blocks, 1):
            language = detect_language(block.text)
            text_blocks.append(
                TextBlock(
                    id=index,
                    text=block.text,
                    lines=[line.text for line in block.lines],
                    bbox=block.bbox,
                    avg_confidence=block.avg_confidence,
                    reading_order=index,
                    block_type=block.block_type,
                    line_bboxes=[line.bbox for line in block.lines],
                    line_confidences=[line.confidence for line in block.lines],
                    language=language,
                    metadata=block.metadata,
                )
            )
        return text_blocks

    def _score_result(self, blocks: Sequence[TextBlock], avg_conf: float, raw_line_count: int, char_count: int) -> float:
        if not blocks or raw_line_count == 0:
            return 0.0
        density = min(1.0, math.log(max(char_count, 1), OCR_SCORE_DENSITY_LOG_BASE))
        return max(0.0, avg_conf * OCR_SCORE_CONFIDENCE_WEIGHT + density * OCR_SCORE_DENSITY_WEIGHT)

    def _needs_multilingual_fallback(self, result: OCRCandidateResult) -> bool:
        if not self.config.multilingual_fallback_enabled:
            return False
        return (
            result.raw_line_count < self.config.multilingual_fallback_min_lines
            or result.char_count < self.config.multilingual_fallback_min_chars
            or result.avg_confidence < self.config.multilingual_fallback_min_confidence
        )

    def _engine_version_label(self) -> str:
        try:
            import paddleocr  # type: ignore

            package_version = str(getattr(paddleocr, "__version__", "unknown"))
        except Exception as exc:
            LOGGER.warning("paddleocr version detection failed: %s", exc)
            package_version = "unknown"
        return f"{self.config.ocr_version} / paddleocr-{package_version}"

    def cleaned_text(self, blocks: Sequence[TextBlock]) -> str:
        return "\n\n".join(block.text.strip() for block in blocks if block.text.strip())

