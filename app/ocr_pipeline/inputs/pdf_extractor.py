"""Native-first PDF extraction.

This module extracts searchable PDF text and tables without rasterizing pages
for OCR. Page preview rendering is only for the frontend review UI and scanned
page fallback.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List
import re

from app.ocr_pipeline.core.models import DocumentPage, DocumentSource, ReviewRegion, TableBlock, TextSpan
from app.ocr_pipeline.utils.file_utils import ensure_directory
from app.ocr_pipeline.utils.logger import get_logger


LOGGER = get_logger(__name__)


class NativePDFExtractor:
    """Extract native PDF spans, tables, and review regions."""

    def __init__(self, min_text_chars_for_native: int = 24, preview_zoom: float = 1.8) -> None:
        self.min_text_chars_for_native = min_text_chars_for_native
        self.preview_zoom = preview_zoom

    def extract(
        self,
        pdf_path: str | Path,
        output_dir: str | Path | None = None,
        render_previews: bool = True,
    ) -> DocumentSource:
        path = Path(pdf_path)
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"NativePDFExtractor only accepts PDF files: {path}")
        pymupdf = _load_pymupdf()
        if pymupdf is None:
            raise RuntimeError("PyMuPDF is required for native PDF extraction. Install pymupdf.")

        preview_dir = ensure_directory(Path(output_dir) / "previews") if output_dir and render_previews else None
        document = DocumentSource.from_path(path, "pdf")
        document.metadata["native_pdf"] = {"engine": "pymupdf", "render_previews": bool(preview_dir)}

        pdf = pymupdf.open(str(path))
        try:
            pages_data: List[Dict[str, Any]] = []
            has_pymupdf_tables = False
            for page_index, page in enumerate(pdf):
                page_number = page_index + 1
                pymupdf_tables = self._extract_pymupdf_tables(page, page_number)
                if pymupdf_tables:
                    has_pymupdf_tables = True
                pages_data.append({
                    "number": page_number,
                    "width": float(page.rect.width),
                    "height": float(page.rect.height),
                    "spans": self._extract_spans(page, page_number),
                    "tables": pymupdf_tables,
                })
            plumber_tables = self._extract_pdfplumber_tables(path) if has_pymupdf_tables else {}
            camelot_tables = self._extract_camelot_tables(path) if has_pymupdf_tables else {}
            for pd in pages_data:
                page_number = pd["number"]
                width = pd["width"]
                height = pd["height"]
                spans = pd["spans"]
                page_text = "\n".join(span.text for span in spans if span.text.strip())
                tables = pd["tables"] + plumber_tables.get(page_number, []) + camelot_tables.get(page_number, [])
                review_regions = self._review_regions(spans, tables)
                preview_image = self._render_preview(pdf[page_number - 1], preview_dir, page_number) if preview_dir else ""
                needs_ocr = len(re.sub(r"\s+", "", page_text)) < self.min_text_chars_for_native
                document.pages.append(
                    DocumentPage(
                        number=page_number,
                        width=width,
                        height=height,
                        text=page_text,
                        spans=spans,
                        tables=tables,
                        review_regions=review_regions,
                        preview_image=preview_image,
                        extraction_source="native_pdf",
                        needs_ocr=needs_ocr,
                        metadata={
                            "native_text_chars": len(page_text),
                            "span_count": len(spans),
                            "table_count": len(tables),
                        },
                    )
                )
        finally:
            pdf.close()

        document.text = "\n\n".join(page.text for page in document.pages if page.text.strip())
        if any(page.needs_ocr for page in document.pages):
            document.warnings.append("Some PDF pages have little or no native text and need OCR fallback.")
        return document

    def _extract_spans(self, page: Any, page_number: int) -> List[TextSpan]:
        spans: List[TextSpan] = []
        try:
            payload = page.get_text("dict")
        except Exception as exc:
            LOGGER.warning("PyMuPDF span extraction failed on page %s: %s", page_number, exc)
            return spans

        for block in payload.get("blocks", []):
            if not isinstance(block, dict):
                continue
            for line in block.get("lines", []) or []:
                if not isinstance(line, dict):
                    continue
                line_text_parts: List[str] = []
                line_bbox: List[float] | None = None
                for span in line.get("spans", []) or []:
                    if not isinstance(span, dict):
                        continue
                    text = str(span.get("text", "")).strip()
                    if not text:
                        continue
                    bbox = _bbox(span.get("bbox"))
                    if not bbox:
                        continue
                    line_text_parts.append(text)
                    line_bbox = _union_bbox(line_bbox, bbox)
                if line_text_parts and line_bbox:
                    spans.append(
                        TextSpan(
                            text=" ".join(line_text_parts).strip(),
                            page=page_number,
                            bbox=line_bbox,
                            confidence=1.0,
                            source="pymupdf_text",
                            metadata={"kind": "line"},
                        )
                    )
        if not spans:
            spans.extend(self._extract_words(page, page_number))
        return spans

    def _extract_words(self, page: Any, page_number: int) -> List[TextSpan]:
        spans: List[TextSpan] = []
        try:
            words = page.get_text("words")
        except Exception:
            return spans
        for item in words or []:
            if not isinstance(item, (list, tuple)) or len(item) < 5:
                continue
            text = str(item[4]).strip()
            if not text:
                continue
            spans.append(
                TextSpan(
                    text=text,
                    page=page_number,
                    bbox=[float(item[0]), float(item[1]), float(item[2]), float(item[3])],
                    confidence=1.0,
                    source="pymupdf_word",
                    metadata={"kind": "word"},
                )
            )
        return spans

    def _extract_pymupdf_tables(self, page: Any, page_number: int) -> List[TableBlock]:
        if not hasattr(page, "find_tables"):
            return []
        tables: List[TableBlock] = []
        try:
            found = page.find_tables()
        except Exception as exc:
            LOGGER.debug("PyMuPDF table extraction failed on page %s: %s", page_number, exc)
            return tables
        for index, table in enumerate(getattr(found, "tables", []) or []):
            try:
                rows = table.extract()
            except Exception:
                rows = []
            if not rows:
                continue
            tables.append(
                TableBlock(
                    page=page_number,
                    rows=_clean_rows(rows),
                    bbox=_bbox(getattr(table, "bbox", [])),
                    source="pymupdf_table",
                    confidence=1.0,
                    metadata={"table_index": index},
                )
            )
        return tables

    def _extract_pdfplumber_tables(self, path: Path) -> Dict[int, List[TableBlock]]:
        try:
            import pdfplumber  # type: ignore
        except Exception:
            return {}
        output: Dict[int, List[TableBlock]] = {}
        try:
            with pdfplumber.open(str(path)) as pdf:
                for index, page in enumerate(pdf.pages, 1):
                    blocks: List[TableBlock] = []
                    try:
                        tables = page.find_tables()
                    except Exception:
                        tables = []
                    for table_index, table in enumerate(tables or []):
                        try:
                            rows = table.extract()
                        except Exception:
                            rows = []
                        if rows:
                            blocks.append(
                                TableBlock(
                                    page=index,
                                    rows=_clean_rows(rows),
                                    bbox=[float(table.bbox[0]), float(table.bbox[1]), float(table.bbox[2]), float(table.bbox[3])]
                                    if getattr(table, "bbox", None)
                                    else [],
                                    source="pdfplumber_table",
                                    confidence=1.0,
                                    metadata={"table_index": table_index},
                                )
                            )
                    if blocks:
                        output[index] = blocks
        except Exception as exc:
            LOGGER.debug("pdfplumber table extraction failed: %s", exc)
        return output

    def _extract_camelot_tables(self, path: Path) -> Dict[int, List[TableBlock]]:
        try:
            import camelot  # type: ignore
        except Exception:
            return {}
        output: Dict[int, List[TableBlock]] = {}
        for flavor in ("lattice", "stream"):
            try:
                tables = camelot.read_pdf(str(path), pages="all", flavor=flavor)
            except Exception as exc:
                LOGGER.debug("Camelot %s extraction failed: %s", flavor, exc)
                continue
            for table_index, table in enumerate(tables or []):
                try:
                    rows = table.df.fillna("").values.tolist()
                    page_number = int(getattr(table, "page", 1) or 1)
                    accuracy = float(getattr(table, "accuracy", 0.0) or 0.0)
                except Exception:
                    continue
                if rows:
                    output.setdefault(page_number, []).append(
                        TableBlock(
                            page=page_number,
                            rows=_clean_rows(rows),
                            bbox=[],
                            source=f"camelot_{flavor}",
                            confidence=accuracy / 100.0 if accuracy > 1.0 else accuracy,
                            metadata={"table_index": table_index, "flavor": flavor},
                        )
                    )
        return output

    def _render_preview(self, page: Any, preview_dir: Path, page_number: int) -> str:
        try:
            pymupdf = _load_pymupdf()
            matrix = pymupdf.Matrix(self.preview_zoom, self.preview_zoom)
            pixmap = page.get_pixmap(matrix=matrix, alpha=False)
            output = preview_dir / f"page_{page_number}.png"
            pixmap.save(str(output))
            return str(output)
        except Exception as exc:
            LOGGER.warning("PDF preview rendering failed on page %s: %s", page_number, exc)
            return ""

    def _review_regions(self, spans: Iterable[TextSpan], tables: Iterable[TableBlock]) -> List[ReviewRegion]:
        regions: List[ReviewRegion] = []
        for index, span in enumerate(spans):
            regions.append(
                ReviewRegion(
                    id=f"p{span.page}-text-{index}",
                    page=span.page,
                    bbox=list(span.bbox),
                    text=span.text,
                    region_type="text",
                    confidence=span.confidence,
                    source=span.source,
                    metadata=dict(span.metadata),
                )
            )
        for index, table in enumerate(tables):
            regions.append(
                ReviewRegion(
                    id=f"p{table.page}-table-{index}",
                    page=table.page,
                    bbox=list(table.bbox),
                    text=_table_preview(table.rows),
                    region_type="table",
                    confidence=table.confidence,
                    source=table.source,
                    metadata={"row_count": len(table.rows), **table.metadata},
                )
            )
        return regions


def _load_pymupdf() -> Any | None:
    try:
        import pymupdf  # type: ignore

        return pymupdf
    except Exception:
        try:
            import fitz  # type: ignore

            return fitz
        except Exception:
            return None


def _bbox(value: Any) -> List[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return []
    try:
        return [float(value[0]), float(value[1]), float(value[2]), float(value[3])]
    except Exception:
        return []


def _union_bbox(current: List[float] | None, bbox: List[float]) -> List[float]:
    if current is None:
        return list(bbox)
    return [
        min(float(current[0]), float(bbox[0])),
        min(float(current[1]), float(bbox[1])),
        max(float(current[2]), float(bbox[2])),
        max(float(current[3]), float(bbox[3])),
    ]


def _clean_rows(rows: Iterable[Iterable[Any]]) -> List[List[Any]]:
    clean: List[List[Any]] = []
    for row in rows:
        cells = ["" if cell is None else str(cell).strip() for cell in row]
        if any(cells):
            clean.append(cells)
    return clean


def _table_preview(rows: List[List[Any]]) -> str:
    return "\n".join(" | ".join(str(cell) for cell in row) for row in rows[:8])

