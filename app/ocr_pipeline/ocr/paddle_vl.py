"""PPStructure and PaddleOCR-VL layout analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List
import json
import subprocess
import sys
import tempfile

import cv2
import numpy as np

from app.config import LayoutConfig
from app.ocr_pipeline.core.block_types import BlockType
from app.ocr_pipeline.utils.logger import get_logger, suppress_external_output


LOGGER = get_logger(__name__)


def _bbox_to_xyxy(value: Any) -> List[int]:
    if isinstance(value, dict):
        if {"x1", "y1", "x2", "y2"}.issubset(value):
            return [int(float(value[key])) for key in ("x1", "y1", "x2", "y2")]
        if {"x", "y", "width", "height"}.issubset(value):
            x1 = int(float(value["x"]))
            y1 = int(float(value["y"]))
            return [x1, y1, x1 + int(float(value["width"])), y1 + int(float(value["height"]))]
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    nums = [int(float(v)) for v in value[:4]]
    x1, y1, x2, y2 = nums
    if x2 < x1 or y2 < y1:
        x2 = x1 + max(0, x2)
        y2 = y1 + max(0, y2)
    return [x1, y1, x2, y2]


def run_layout_analysis(image: np.ndarray, use_gpu: bool = False, engine: Any | None = None) -> Dict[str, Any]:
    """Use PPStructure to detect document regions."""
    try:
        if engine is None:
            engine = _create_layout_engine(use_gpu=use_gpu)

        with suppress_external_output():
            result = engine.predict(image) if hasattr(engine, "predict") else engine(image)
    except Exception as exc:
        LOGGER.debug("PPStructure analysis failed: %s", exc)
        return {"source": "ppstructure", "available": False, "reason": str(exc), "detected_regions": []}

    layout_blocks = _parse_layout_regions(result)
    source = _layout_engine_source(engine)

    return {"source": source, "available": True, "detected_regions": layout_blocks}


def run_layout_analysis_isolated(image: np.ndarray, config: LayoutConfig) -> Dict[str, Any]:
    """Run PPStructure/LayoutDetection outside the main process."""
    with tempfile.TemporaryDirectory(prefix="docai_layout_") as temp_dir:
        input_path = Path(temp_dir) / "input.png"
        output_path = Path(temp_dir) / "result.json"
        payload_path = Path(temp_dir) / "payload.json"
        cv2.imwrite(str(input_path), image)
        payload_path.write_text(
            json.dumps({"image_path": str(input_path), "use_gpu": bool(config.ppstructure_use_gpu)}, ensure_ascii=False),
            encoding="utf-8",
        )
        command = [sys.executable, "-m", "app.ocr_pipeline.ocr.layout_worker", str(payload_path), str(output_path)]
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[3]),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=config.layout_detection_timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "source": "layout_detection",
                "available": False,
                "reason": f"timeout after {config.layout_detection_timeout_sec}s",
                "detected_regions": [],
            }
        if completed.returncode != 0 or not output_path.exists():
            return {
                "source": "layout_detection",
                "available": False,
                "reason": f"worker exited with code {completed.returncode}",
                "detected_regions": [],
            }
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "source": "layout_detection",
                "available": False,
                "reason": f"invalid worker output: {exc}",
                "detected_regions": [],
            }
        return payload if isinstance(payload, dict) else {
            "source": "layout_detection",
            "available": False,
            "reason": "worker output was not a JSON object",
            "detected_regions": [],
        }


def run_paddleocr_vl(image: np.ndarray, config: LayoutConfig | None = None, engine: Any | None = None) -> Dict[str, Any]:
    """Use PaddleOCR-VL when installed; caller falls back to PPStructure output."""
    temp_path = _write_temp_image(image)
    try:
        if engine is None:
            from paddleocr import PaddleOCRVL  # type: ignore

            engine = _create_paddleocr_vl_engine(PaddleOCRVL, config)
        with suppress_external_output():
            raw = engine.predict(str(temp_path)) if hasattr(engine, "predict") else engine(str(temp_path))
        return {"source": "paddleocr_vl", "available": True, "result": _parse_vl_result(raw)}
    except Exception as exc:
        LOGGER.debug("PaddleOCR-VL analysis failed: %s", exc)
        return {"source": "paddleocr_vl", "available": False, "reason": str(exc), "result": []}
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception as exc:
            LOGGER.warning("temporary layout image cleanup failed: %s", exc)


def run_paddleocr_vl_isolated(image: np.ndarray, config: LayoutConfig) -> Dict[str, Any]:
    """Run PaddleOCR-VL outside the main process so crashes cannot kill OCR."""
    with tempfile.TemporaryDirectory(prefix="docai_vl_") as temp_dir:
        input_path = Path(temp_dir) / "input.png"
        output_path = Path(temp_dir) / "result.json"
        payload_path = Path(temp_dir) / "payload.json"
        cv2.imwrite(str(input_path), image)
        payload_path.write_text(
            json.dumps(
                {
                    "image_path": str(input_path),
                    "use_orientation": bool(config.paddleocr_vl_use_orientation),
                    "use_unwarping": bool(config.paddleocr_vl_use_unwarping),
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        command = [sys.executable, "-m", "app.ocr_pipeline.ocr.vl_worker", str(payload_path), str(output_path)]
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[3]),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=config.paddleocr_vl_timeout_sec,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "source": "paddleocr_vl",
                "available": False,
                "reason": f"timeout after {config.paddleocr_vl_timeout_sec}s",
                "result": [],
            }
        if completed.returncode != 0 or not output_path.exists():
            return {
                "source": "paddleocr_vl",
                "available": False,
                "reason": f"worker exited with code {completed.returncode}",
                "result": [],
            }
        try:
            return json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {"source": "paddleocr_vl", "available": False, "reason": f"invalid worker output: {exc}", "result": []}


class PaddleOCRVLAdapter:
    """Pipeline adapter that returns a unified vision/layout payload."""

    def __init__(self, config: LayoutConfig) -> None:
        self.config = config
        self._ppstructure_engine: Any | None = None
        self._paddleocr_vl_engine: Any | None = None

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        ppstructure = (
            run_layout_analysis_isolated(image, self.config)
            if self.config.ppstructure_enabled
            else {"source": "ppstructure", "available": False, "reason": "disabled", "detected_regions": []}
        )
        vl = (
            run_paddleocr_vl_isolated(image, self.config)
            if self.config.paddleocr_vl_enabled
            else {"source": "paddleocr_vl", "available": False, "reason": "disabled", "result": []}
        )

        pp_regions = [region for region in ppstructure.get("detected_regions", []) if isinstance(region, dict)]
        vl_regions = _vl_elements_to_regions(vl.get("result", []))
        if vl.get("available") and vl_regions:
            source = "paddleocr_vl"
            regions = vl_regions
        elif ppstructure.get("available"):
            source = str(ppstructure.get("source", "ppstructure"))
            regions = pp_regions
        elif vl.get("available"):
            source = "paddleocr_vl"
            regions = []
        else:
            source = "none"
            regions = []

        return {
            "vision_source": source,
            "available": bool(vl.get("available") or ppstructure.get("available")),
            "paddleocr_vl": vl,
            "paddleocr_vl_regions": vl_regions,
            "ppstructure": ppstructure,
            "detected_regions": regions,
            "document_layout": infer_document_layout(regions),
            "layout_summary": summarize_layout(regions),
        }

    def _get_ppstructure_engine(self) -> Any | None:
        if self._ppstructure_engine is not None:
            return self._ppstructure_engine
        try:
            self._ppstructure_engine = _create_layout_engine(use_gpu=self.config.ppstructure_use_gpu)
        except Exception as exc:
            LOGGER.debug("PPStructure is unavailable: %s", exc)
            return None
        return self._ppstructure_engine

    def _get_paddleocr_vl_engine(self) -> Any | None:
        if self._paddleocr_vl_engine is not None:
            return self._paddleocr_vl_engine
        try:
            from paddleocr import PaddleOCRVL  # type: ignore

            self._paddleocr_vl_engine = _create_paddleocr_vl_engine(PaddleOCRVL, self.config)
        except Exception as exc:
            LOGGER.debug("PaddleOCR-VL is unavailable: %s", exc)
            return None
        return self._paddleocr_vl_engine


def infer_document_layout(regions: List[Dict[str, Any]]) -> str:
    if not regions:
        return "unknown"
    if any(str(region.get("region_type", "")).lower() == "table" for region in regions):
        return "table-based"
    centers = []
    for region in regions:
        bbox = region.get("bbox", [])
        if isinstance(bbox, list) and len(bbox) == 4:
            centers.append((bbox[0] + bbox[2]) / 2)
    if len(centers) >= 4:
        span = max(centers) - min(centers)
        if span > 450:
            return "two-column"
    return "single-column"


def summarize_layout(regions: List[Dict[str, Any]]) -> str:
    if not regions:
        return "No layout regions were detected."
    counts: Dict[str, int] = {}
    for region in regions:
        key = str(region.get("region_type", "unknown")).lower()
        counts[key] = counts.get(key, 0) + 1
    parts = [f"{count} {kind}" for kind, count in sorted(counts.items())]
    return "Detected " + ", ".join(parts) + " region(s)."


def _create_layout_engine(use_gpu: bool = False) -> Any:
    try:
        from paddleocr import PPStructure  # type: ignore

        try:
            with suppress_external_output():
                return PPStructure(show_log=False, use_gpu=use_gpu)
        except TypeError:
            with suppress_external_output():
                return PPStructure(show_log=False)
    except ImportError:
        from paddleocr import LayoutDetection  # type: ignore

        with suppress_external_output():
            return LayoutDetection()


def _create_paddleocr_vl_engine(paddleocr_vl_cls: Any, config: LayoutConfig | None = None) -> Any:
    cfg = config or LayoutConfig()
    kwargs = {
        "pipeline_version": "v1.5",
        "use_doc_orientation_classify": cfg.paddleocr_vl_use_orientation,
        "use_doc_unwarping": cfg.paddleocr_vl_use_unwarping,
        "use_layout_detection": True,
        "use_chart_recognition": False,
        "use_seal_recognition": False,
        "use_ocr_for_image_block": False,
    }
    try:
        with suppress_external_output():
            return paddleocr_vl_cls(**kwargs)
    except TypeError:
        with suppress_external_output():
            return paddleocr_vl_cls(pipeline_version="v1.5")


def _layout_engine_source(engine: Any) -> str:
    name = engine.__class__.__name__.lower()
    if "layout" in name:
        return "layout_detection"
    return "ppstructure"


def _parse_layout_regions(result: Any) -> List[Dict[str, Any]]:
    layout_blocks: List[Dict[str, Any]] = []
    records = result if isinstance(result, list) else [result]
    for region in records or []:
        if isinstance(region, dict) and ("type" in region or "bbox" in region):
            block = {
                "region_type": str(region.get("type", "unknown")).lower(),
                "bbox": _bbox_to_xyxy(region.get("bbox", [])),
                "content": [],
            }
            content = region.get("res")
            if isinstance(content, list):
                for line in content:
                    parsed = _parse_region_line(line)
                    if parsed:
                        block["content"].append(parsed)
            elif isinstance(content, dict):
                block["content"] = content
            layout_blocks.append(block)
            continue

        payload = _to_mapping(region)
        payload = payload.get("res", payload) if isinstance(payload, dict) else {}
        boxes = payload.get("boxes") if isinstance(payload, dict) else None
        if isinstance(boxes, list):
            for item in boxes:
                if not isinstance(item, dict):
                    continue
                bbox = _bbox_to_xyxy(item.get("coordinate") or item.get("bbox") or item.get("box"))
                if not bbox:
                    continue
                layout_blocks.append(
                    {
                        "region_type": str(item.get("label", item.get("type", "unknown"))).lower(),
                        "bbox": bbox,
                        "content": [],
                        "confidence": round(float(item.get("score", 0.0)), 4),
                    }
                )
    return layout_blocks


def _parse_region_line(line: Any) -> Dict[str, Any] | None:
    if not isinstance(line, (list, tuple)) or len(line) != 2:
        return None
    bbox_pts, text_conf = line
    if not isinstance(text_conf, (list, tuple)) or len(text_conf) < 2:
        return None
    text, conf = text_conf[0], text_conf[1]
    return {"text": str(text), "confidence": round(float(conf), 4), "bbox": _points_to_bbox(bbox_pts)}


def _points_to_bbox(points: Any) -> List[int]:
    arr = np.asarray(points, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 4 or arr.shape[1] < 2:
        return []
    xs = arr[:, 0]
    ys = arr[:, 1]
    return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]


def _parse_vl_result(raw: Any) -> List[Dict[str, Any]]:
    records = raw if isinstance(raw, list) else [raw]
    elements: List[Dict[str, Any]] = []
    for record in records:
        mapping = _to_mapping(record)
        payload = mapping.get("res", mapping)
        if not isinstance(payload, dict):
            continue
        for key in ("layout_parsing_result", "parsing_res_list", "elements", "blocks"):
            value = payload.get(key)
            if isinstance(value, list):
                elements.extend(item for item in value if isinstance(item, dict))
    return elements


def _vl_elements_to_regions(elements: Any) -> List[Dict[str, Any]]:
    if not isinstance(elements, list):
        return []

    regions: List[Dict[str, Any]] = []
    for element in elements:
        if not isinstance(element, dict):
            continue
        bbox = _extract_bbox(element)
        if not bbox:
            continue
        region_type = normalize_region_type(str(
            element.get("region_type")
            or element.get("type")
            or element.get("label")
            or element.get("category")
            or element.get("block_type")
            or "unknown"
        ).lower())
        text = element.get("text") or element.get("content") or element.get("markdown") or ""
        regions.append({"region_type": region_type, "bbox": bbox, "content": [{"text": str(text), "confidence": 1.0}]})
    return regions


def _extract_bbox(element: Dict[str, Any]) -> List[int]:
    for key in ("bbox", "box", "coordinate", "coordinates", "rect"):
        bbox = _bbox_to_xyxy(element.get(key))
        if bbox:
            return bbox

    for key in ("poly", "polygon", "points"):
        value = element.get(key)
        if value is None:
            continue
        bbox = _points_to_bbox(value)
        if bbox:
            return bbox
    return []


def _to_mapping(record: Any) -> Dict[str, Any]:
    if isinstance(record, dict):
        return record
    for attr in ("json", "to_dict", "dict"):
        value = getattr(record, attr, None)
        if isinstance(value, dict):
            return value
        if callable(value):
            try:
                mapped = value()
            except TypeError:
                continue
            if isinstance(mapped, dict):
                return mapped
    payload = getattr(record, "res", None)
    return payload if isinstance(payload, dict) else {}


def normalize_region_type(region_type: str) -> str:
    """Normalize PaddleOCR-VL region labels to canonical block types."""
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


def _write_temp_image(image: np.ndarray) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="docai_layout_", suffix=".png", delete=False)
    path = Path(handle.name)
    handle.close()
    cv2.imwrite(str(path), image)
    return path


