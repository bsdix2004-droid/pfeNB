"""Image loading, validation, and drawing utilities."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Sequence, Tuple

import cv2
import numpy as np

from app.config import OCRInputConfig as InputConfig
from app.ocr_pipeline.ocr_config import APP_CONFIG
from app.ocr_pipeline.core.ocr_engine import TextBlock
from app.ocr_pipeline.utils.logger import get_logger


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class LoadedImage:
    image: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk after safety validation."""
    return load_image_with_metadata(path).image


def load_image_with_metadata(path: str, config: InputConfig | None = None) -> LoadedImage:
    """Load raster images or SVGs with strict file and resolution limits."""
    cfg = config or APP_CONFIG.input
    source = Path(path)
    metadata = _validate_file(source, cfg)
    suffix = source.suffix.lower()

    if suffix == ".svg":
        image, render_meta = _load_svg(source, cfg)
        metadata.update(render_meta)
    else:
        image, raster_meta = _load_raster(source)
        metadata.update(raster_meta)

    if image is None or image.size == 0:
        raise ValueError(f"Could not decode image: {source}")

    height, width = image.shape[:2]
    image, normalized = _normalize_large_image(image, cfg)
    final_height, final_width = image.shape[:2]
    _validate_dimensions(final_width, final_height, cfg)

    metadata.update(
        {
            "width": int(width),
            "height": int(height),
            "final_width": int(final_width),
            "final_height": int(final_height),
            "pixels": int(width * height),
            "normalized": normalized,
        }
    )
    if suffix == ".svg":
        vector_lines = extract_svg_text_lines(source, final_width, final_height)
        metadata["vector_text_lines"] = vector_lines
        metadata["vector_text_line_count"] = len(vector_lines)
    return LoadedImage(image=image, metadata=metadata)


def draw_blocks(image: np.ndarray, blocks: List[TextBlock], type_colors: dict) -> np.ndarray:
    """Annotate image with colored bounding boxes and labels."""
    annotated = image.copy()
    for block in blocks:
        x1, y1, x2, y2 = block.bbox
        block_type = getattr(block, "block_type", "text")
        color = type_colors.get(block_type, (128, 128, 128))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        label = f"ID:{block.id} {block_type}"
        cv2.putText(annotated, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return annotated


def save_image(image: np.ndarray, path: str | Path) -> None:
    """Save image to path."""
    cv2.imwrite(str(path), image)


def _validate_file(source: Path, config: InputConfig) -> Dict[str, Any]:
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Input file not found: {source}")

    suffix = source.suffix.lower()
    allowed = {item.lower() for item in config.allowed_extensions}
    if suffix not in allowed:
        raise ValueError(f"Unsupported input type '{suffix}'. Allowed: {', '.join(sorted(allowed))}")

    file_size = source.stat().st_size
    max_bytes = int(config.max_file_size_mb * 1024 * 1024)
    if file_size <= 0:
        raise ValueError(f"Input file is empty: {source}")
    if file_size > max_bytes:
        raise ValueError(f"Input file is too large: {file_size / (1024 * 1024):.1f} MB > {config.max_file_size_mb} MB")

    return {
        "source_file_size_bytes": int(file_size),
        "source_file_size_mb": round(file_size / (1024 * 1024), 3),
        "extension": suffix,
    }


def _load_raster(source: Path) -> tuple[np.ndarray, Dict[str, Any]]:
    image = cv2.imread(str(source), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not decode raster image: {source}")
    return image, {"loader": "opencv"}


def _load_svg(source: Path, config: InputConfig) -> tuple[np.ndarray, Dict[str, Any]]:
    browser = _browser_executable()
    if browser:
        image = _rasterize_svg_with_browser(source, browser, config)
        return image, {"loader": Path(browser).name, "svg_render_width": config.svg_render_width}

    try:
        import cairosvg  # type: ignore

        png_bytes = cairosvg.svg2png(url=str(source), output_width=config.svg_render_width)
        array = np.frombuffer(png_bytes, dtype=np.uint8)
        image = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if image is not None:
            return image, {"loader": "cairosvg", "svg_render_width": config.svg_render_width}
    except Exception as exc:
        LOGGER.warning("cairosvg SVG rasterization failed: %s", exc)

    raise RuntimeError("SVG input requires Google Chrome, Microsoft Edge, or a working CairoSVG runtime.")


def _validate_dimensions(width: int, height: int, config: InputConfig) -> None:
    if width < config.min_width or height < config.min_height:
        raise ValueError(
            f"Input resolution is too small: {width}x{height}. Minimum is {config.min_width}x{config.min_height}."
        )
    if width > config.max_width or height > config.max_height:
        raise ValueError(
            f"Input resolution is too large: {width}x{height}. Maximum is {config.max_width}x{config.max_height}."
        )
    pixels = width * height
    if pixels > config.max_pixels:
        raise ValueError(f"Input has too many pixels: {pixels:,}. Maximum is {config.max_pixels:,}.")


def _normalize_large_image(image: np.ndarray, config: InputConfig) -> tuple[np.ndarray, bool]:
    height, width = image.shape[:2]
    long_edge = max(width, height)
    if long_edge <= config.normalize_max_long_edge:
        return image, False
    scale = config.normalize_max_long_edge / long_edge
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return resized, True


def _browser_executable() -> str | None:
    candidates = [
        os.environ.get("CHROME_EXE"),
        os.environ.get("EDGE_EXE"),
        str(Path(os.environ.get("ProgramFiles", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(Path(os.environ.get("ProgramFiles(x86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe"),
        str(Path(os.environ.get("ProgramFiles", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
        str(Path(os.environ.get("ProgramFiles(x86)", "")) / "Microsoft" / "Edge" / "Application" / "msedge.exe"),
        shutil.which("chrome"),
        shutil.which("msedge"),
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return str(candidate)
    return None


def _rasterize_svg_with_browser(source: Path, browser: str, config: InputConfig) -> np.ndarray:
    width, height = _svg_render_size(source, config.svg_render_width)
    with tempfile.TemporaryDirectory(prefix="docai_svg_") as temp_dir:
        output = Path(temp_dir) / "rendered.png"
        command = [
            browser,
            "--headless=new",
            "--disable-gpu",
            "--hide-scrollbars",
            f"--window-size={width},{height}",
            f"--screenshot={output}",
            source.resolve().as_uri(),
        ]
        completed = subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=60, check=False)
        if completed.returncode != 0 or not output.exists():
            raise RuntimeError(f"Browser SVG rasterization failed with exit code {completed.returncode}")
        image = cv2.imread(str(output), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not decode browser-rendered SVG: {source}")
        return image


def _svg_render_size(source: Path, target_width: int) -> tuple[int, int]:
    text = source.read_text(encoding="utf-8", errors="ignore")[:4096]
    viewbox = re.search(r'viewBox=["\']\s*[-.\d]+\s+[-.\d]+\s+([.\d]+)\s+([.\d]+)\s*["\']', text)
    if viewbox:
        width = float(viewbox.group(1))
        height = float(viewbox.group(2))
    else:
        width_match = re.search(r'width=["\']([.\d]+)', text)
        height_match = re.search(r'height=["\']([.\d]+)', text)
        width = float(width_match.group(1)) if width_match else 595.0
        height = float(height_match.group(1)) if height_match else 842.0
    ratio = height / max(width, 1.0)
    render_width = max(320, int(target_width))
    render_height = max(320, int(render_width * ratio))
    return render_width, render_height


def extract_svg_text_lines(source: Path, final_width: int, final_height: int) -> List[Dict[str, Any]]:
    """Extract embedded SVG text as high-confidence OCR-like lines."""
    try:
        tree = ET.parse(source)
    except Exception as exc:
        LOGGER.warning("SVG text extraction parse failed: %s", exc)
        return []

    root = tree.getroot()
    view_x, view_y, view_width, view_height = _svg_viewbox(root)
    if view_width <= 0 or view_height <= 0 or final_width <= 0 or final_height <= 0:
        return []

    font_sizes = _svg_font_sizes(root)
    fragments: List[Dict[str, Any]] = []
    _collect_svg_text_fragments(root, _identity_matrix(), font_sizes, fragments)

    scale_x = final_width / view_width
    scale_y = final_height / view_height
    scaled: List[Dict[str, Any]] = []
    for fragment in fragments:
        text = _clean_svg_text(str(fragment.get("text", "")))
        if not text:
            continue
        x = (float(fragment["x"]) - view_x) * scale_x
        y = (float(fragment["y"]) - view_y) * scale_y
        font_height = max(8.0, float(fragment["font_height"]) * scale_y)
        width = max(font_height * 0.7, _estimate_svg_text_width(text, font_height))
        bbox = (
            int(max(0, round(x))),
            int(max(0, round(y - font_height * 1.05))),
            int(min(final_width, round(x + width))),
            int(min(final_height, round(y + font_height * 0.25))),
        )
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            continue
        scaled.append(
            {
                "text": text,
                "x": float(x),
                "baseline_y": float(y),
                "font_height": float(font_height),
                "bbox": bbox,
            }
        )

    return _merge_svg_fragments_into_lines(scaled, final_width, final_height)


def _svg_viewbox(root: ET.Element) -> Tuple[float, float, float, float]:
    viewbox = root.attrib.get("viewBox") or root.attrib.get("viewbox")
    if viewbox:
        values = _numbers(viewbox)
        if len(values) >= 4:
            return values[0], values[1], values[2], values[3]

    width = _svg_length(root.attrib.get("width"), 595.0)
    height = _svg_length(root.attrib.get("height"), 842.0)
    return 0.0, 0.0, width, height


def _svg_font_sizes(root: ET.Element) -> Dict[str, float]:
    font_sizes: Dict[str, float] = {}
    for element in root.iter():
        if _local_name(element.tag) != "style":
            continue
        style_text = "".join(element.itertext())
        for match in re.finditer(r"\.([A-Za-z0-9_-]+)\s*\{[^}]*font-size\s*:\s*([0-9.]+)px", style_text):
            try:
                font_sizes[match.group(1)] = float(match.group(2))
            except ValueError:
                continue
    return font_sizes


def _collect_svg_text_fragments(
    element: ET.Element,
    parent_matrix: Tuple[float, float, float, float, float, float],
    font_sizes: Dict[str, float],
    fragments: List[Dict[str, Any]],
) -> None:
    matrix = _multiply_matrices(parent_matrix, _parse_svg_transform(element.attrib.get("transform", "")))
    if _local_name(element.tag) == "text":
        text = _clean_svg_text("".join(element.itertext()))
        if text:
            x, y = _svg_text_anchor(element)
            tx, ty = _transform_point(matrix, x, y)
            fragments.append(
                {
                    "text": text,
                    "x": tx,
                    "y": ty,
                    "font_height": _svg_effective_font_height(element, matrix, font_sizes),
                }
            )
        return

    for child in list(element):
        _collect_svg_text_fragments(child, matrix, font_sizes, fragments)


def _svg_text_anchor(element: ET.Element) -> Tuple[float, float]:
    x_values = _numbers(element.attrib.get("x", ""))
    y_values = _numbers(element.attrib.get("y", ""))
    return (x_values[0] if x_values else 0.0, y_values[0] if y_values else 0.0)


def _svg_effective_font_height(
    element: ET.Element,
    matrix: Tuple[float, float, float, float, float, float],
    font_sizes: Dict[str, float],
) -> float:
    font_size = _svg_length(element.attrib.get("font-size"), 12.0)
    for child in element.iter():
        class_attr = child.attrib.get("class", "")
        for class_name in class_attr.split():
            if class_name in font_sizes:
                font_size = font_sizes[class_name]
    _x0, y0 = _transform_point(matrix, 0.0, 0.0)
    _x1, y1 = _transform_point(matrix, 0.0, 1.0)
    return max(6.0, abs(y1 - y0) * font_size)


def _merge_svg_fragments_into_lines(
    fragments: Sequence[Dict[str, Any]],
    final_width: int,
    final_height: int,
) -> List[Dict[str, Any]]:
    if not fragments:
        return []

    ordered = sorted(fragments, key=lambda item: (float(item["baseline_y"]), float(item["x"])))
    rows: List[List[Dict[str, Any]]] = []
    for fragment in ordered:
        placed = False
        for row in rows:
            row_y = sum(float(item["baseline_y"]) for item in row) / len(row)
            row_height = max(float(item["font_height"]) for item in row)
            if abs(float(fragment["baseline_y"]) - row_y) <= max(5.0, row_height * 0.48):
                row.append(fragment)
                placed = True
                break
        if not placed:
            rows.append([fragment])

    lines: List[Dict[str, Any]] = []
    for row in rows:
        row.sort(key=lambda item: float(item["x"]))
        current: List[Dict[str, Any]] = []
        for fragment in row:
            if not current:
                current = [fragment]
                continue
            prev_bbox = current[-1]["bbox"]
            gap = int(fragment["bbox"][0]) - int(prev_bbox[2])
            height = max(float(item["font_height"]) for item in current + [fragment])
            max_gap = max(90.0, final_width * 0.08, height * 4.0)
            if gap <= max_gap:
                current.append(fragment)
            else:
                lines.append(_make_svg_line(current, final_width, final_height, len(lines)))
                current = [fragment]
        if current:
            lines.append(_make_svg_line(current, final_width, final_height, len(lines)))

    lines = [line for line in lines if line["text"]]
    return sorted(lines, key=lambda item: (item["bbox"][1], item["bbox"][0]))


def _make_svg_line(
    fragments: Sequence[Dict[str, Any]],
    final_width: int,
    final_height: int,
    index: int,
) -> Dict[str, Any]:
    text = _join_svg_text_parts([str(item["text"]) for item in fragments])
    x1 = min(int(item["bbox"][0]) for item in fragments)
    y1 = min(int(item["bbox"][1]) for item in fragments)
    x2 = max(int(item["bbox"][2]) for item in fragments)
    y2 = max(int(item["bbox"][3]) for item in fragments)
    x1 = max(0, min(final_width, x1))
    x2 = max(x1 + 1, min(final_width, x2))
    y1 = max(0, min(final_height, y1))
    y2 = max(y1 + 1, min(final_height, y2))
    return {
        "text": text,
        "bbox": [x1, y1, x2, y2],
        "confidence": 1.0,
        "source_index": index,
        "source": "svg_text",
    }


def _join_svg_text_parts(parts: Sequence[str]) -> str:
    text = ""
    for raw_part in parts:
        part = _clean_svg_text(raw_part)
        if not part:
            continue
        if not text:
            text = part
            continue
        if part in {",", ".", ";", ":", ")", "]", "}"}:
            text = text.rstrip() + part
        elif part in {"-", "–", "—"}:
            text = text.rstrip() + f" {part} "
        elif text.endswith(("(", "[", "{", "/", "-")):
            text += part
        else:
            text = text.rstrip() + " " + part.lstrip()
    return re.sub(r"\s+", " ", text).strip()


def _clean_svg_text(text: str) -> str:
    text = text.replace("\u00a0", " ").replace("\ufeff", "")
    return re.sub(r"\s+", " ", text).strip()


def _estimate_svg_text_width(text: str, font_height: float) -> float:
    width = 0.0
    for char in text:
        if char.isspace():
            width += font_height * 0.30
        elif char in ".,:;|!":
            width += font_height * 0.22
        elif char in "MW@#%":
            width += font_height * 0.78
        else:
            width += font_height * 0.52
    return max(font_height * 0.7, width)


def _parse_svg_transform(transform: str) -> Tuple[float, float, float, float, float, float]:
    matrix = _identity_matrix()
    for name, args in re.findall(r"([A-Za-z]+)\(([^)]*)\)", transform or ""):
        values = _numbers(args)
        local = _identity_matrix()
        if name == "matrix" and len(values) >= 6:
            local = (values[0], values[1], values[2], values[3], values[4], values[5])
        elif name == "translate" and values:
            local = (1.0, 0.0, 0.0, 1.0, values[0], values[1] if len(values) > 1 else 0.0)
        elif name == "scale" and values:
            sx = values[0]
            sy = values[1] if len(values) > 1 else sx
            local = (sx, 0.0, 0.0, sy, 0.0, 0.0)
        elif name == "rotate" and values:
            angle = np.deg2rad(values[0])
            cos_v = float(np.cos(angle))
            sin_v = float(np.sin(angle))
            rotation = (cos_v, sin_v, -sin_v, cos_v, 0.0, 0.0)
            if len(values) >= 3:
                cx, cy = values[1], values[2]
                local = _multiply_matrices(
                    _multiply_matrices((1.0, 0.0, 0.0, 1.0, cx, cy), rotation),
                    (1.0, 0.0, 0.0, 1.0, -cx, -cy),
                )
            else:
                local = rotation
        matrix = _multiply_matrices(matrix, local)
    return matrix


def _identity_matrix() -> Tuple[float, float, float, float, float, float]:
    return 1.0, 0.0, 0.0, 1.0, 0.0, 0.0


def _multiply_matrices(
    left: Tuple[float, float, float, float, float, float],
    right: Tuple[float, float, float, float, float, float],
) -> Tuple[float, float, float, float, float, float]:
    a1, b1, c1, d1, e1, f1 = left
    a2, b2, c2, d2, e2, f2 = right
    return (
        a1 * a2 + c1 * b2,
        b1 * a2 + d1 * b2,
        a1 * c2 + c1 * d2,
        b1 * c2 + d1 * d2,
        a1 * e2 + c1 * f2 + e1,
        b1 * e2 + d1 * f2 + f1,
    )


def _transform_point(
    matrix: Tuple[float, float, float, float, float, float],
    x: float,
    y: float,
) -> Tuple[float, float]:
    a, b, c, d, e, f = matrix
    return a * x + c * y + e, b * x + d * y + f


def _svg_length(value: str | None, default: float) -> float:
    if not value:
        return default
    values = _numbers(value)
    return values[0] if values else default


def _numbers(value: str) -> List[float]:
    numbers: List[float] = []
    for match in re.finditer(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value or ""):
        try:
            numbers.append(float(match.group(0)))
        except ValueError:
            continue
    return numbers


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag

