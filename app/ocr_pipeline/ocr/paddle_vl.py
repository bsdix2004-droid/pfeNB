"""PP-DocLayoutV3 layout analysis — replaces PPStructure + PaddleOCR-VL."""

from __future__ import annotations

from typing import Any, Dict, List

import cv2
import numpy as np

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
    """Use PP-DocLayoutV3 to detect document regions."""
    try:
        if engine is None:
            engine = _create_layout_engine(use_gpu=use_gpu)
        with suppress_external_output():
            result = engine(image) if not hasattr(engine, "predict") else engine.predict(image)
    except Exception as exc:
        LOGGER.debug("PP-DocLayoutV3 analysis failed: %s", exc)
        return {"source": "pp_doclayout", "available": False, "reason": str(exc), "detected_regions": []}

    layout_blocks = _parse_layout_regions(result)
    return {"source": "pp_doclayout", "available": True, "detected_regions": layout_blocks}


def run_layout_analysis_isolated(image: np.ndarray, use_gpu: bool) -> Dict[str, Any]:
    """Run PP-DocLayoutV3 outside the main process."""
    import json
    import subprocess
    import sys
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory(prefix="docai_layout_") as temp_dir:
        input_path = Path(temp_dir) / "input.png"
        output_path = Path(temp_dir) / "result.json"
        payload_path = Path(temp_dir) / "payload.json"
        cv2.imwrite(str(input_path), image)
        payload_path.write_text(
            json.dumps({"image_path": str(input_path), "use_gpu": bool(use_gpu)}, ensure_ascii=False),
            encoding="utf-8",
        )
        command = [sys.executable, "-m", "app.ocr_pipeline.ocr.layout_worker", str(payload_path), str(output_path)]
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[3]),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=300,
                check=False,
            )
        except subprocess.TimeoutExpired:
            return {
                "source": "pp_doclayout",
                "available": False,
                "reason": "timeout after 300s",
                "detected_regions": [],
            }
        if completed.returncode != 0 or not output_path.exists():
            return {
                "source": "pp_doclayout",
                "available": False,
                "reason": f"worker exited with code {completed.returncode}",
                "detected_regions": [],
            }
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "source": "pp_doclayout",
                "available": False,
                "reason": f"invalid worker output: {exc}",
                "detected_regions": [],
            }
        return payload if isinstance(payload, dict) else {
            "source": "pp_doclayout",
            "available": False,
            "reason": "worker output was not a JSON object",
            "detected_regions": [],
        }


class LayoutAdapter:
    """Pipeline adapter that returns a unified vision/layout payload via PP-DocLayoutV3."""

    def __init__(self, use_gpu: bool = False) -> None:
        self._use_gpu = use_gpu
        self._engine: Any | None = None

    def analyze(self, image: np.ndarray) -> Dict[str, Any]:
        result = run_layout_analysis_isolated(image, self._use_gpu)
        regions = [r for r in result.get("detected_regions", []) if isinstance(r, dict)]
        return {
            "vision_source": "pp_doclayout",
            "available": result.get("available", False),
            "detected_regions": regions,
            "document_layout": infer_document_layout(regions),
            "layout_summary": summarize_layout(regions),
        }

    def _get_engine(self) -> Any | None:
        if self._engine is not None:
            return self._engine
        try:
            self._engine = _create_layout_engine(use_gpu=self._use_gpu)
        except Exception as exc:
            LOGGER.debug("PP-DocLayoutV3 is unavailable: %s", exc)
            return None
        return self._engine


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
    from paddleocr import PPDocLayout  # type: ignore

    with suppress_external_output():
        return PPDocLayout(use_gpu=use_gpu)


def _parse_layout_regions(result: Any) -> List[Dict[str, Any]]:
    """Parse PP-DocLayoutV3 result into unified region dicts."""
    records = result if isinstance(result, list) else [result]
    regions: List[Dict[str, Any]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        region_type = normalize_region_type(str(item.get("type", "unknown")))
        bbox = _bbox_to_xyxy(item.get("bbox", []))
        if not bbox:
            continue
        regions.append({
            "region_type": region_type,
            "bbox": bbox,
            "content": [],
            "confidence": round(float(item.get("score", 0.0)), 4),
        })
    return regions


def normalize_region_type(region_type: str) -> str:
    """Normalize PP-DocLayoutV3 region labels to canonical block types."""
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
