"""Smart provider-based document pipeline for backend uploads."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import re
import time
from typing import Any, Dict, List

from app.config import get_settings
from app.ocr_pipeline.classifiers.document_classifier import DocumentClassifier
from app.ocr_pipeline.core.detector import DocumentInfo
from app.ocr_pipeline.core.language_identifier import LanguageIdentifier
from app.ocr_pipeline.core.models import DocumentPage, DocumentSource, FinalExport, ReviewRegion, TableBlock, TextSpan
from app.ocr_pipeline.core.pipeline import DocumentPipeline, PipelineResult
from app.ocr_pipeline.exporters.clean_json_exporter import CleanJSONExporter
from app.ocr_pipeline.inputs.pdf_extractor import NativePDFExtractor
from app.ocr_pipeline.utils.file_utils import ensure_directory, save_json


@dataclass(slots=True)
class SmartDocumentResult:
    document_id: str
    source_file: str
    output_dir: str
    status: str
    document: DocumentSource
    final_export: FinalExport
    output_files: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    processing_time_ms: int = 0


class SmartDocumentPipeline:
    """Route PDFs through native extraction and images through the OCR pipeline."""

    def __init__(self, settings: Any = None) -> None:
        self.settings = settings or get_settings()
        self.config = self.settings.OCR_PIPELINE
        self.language_identifier = LanguageIdentifier(self.config.language)
        self.classifier = DocumentClassifier(self.settings)
        self.image_pipeline = DocumentPipeline(self.config)
        self.pdf_extractor = NativePDFExtractor()
        self.clean_exporter = CleanJSONExporter()

    def process(
        self,
        source_path: str | Path,
        run_ocr_fallback: bool = True,
        trace_id: str | None = None,
    ) -> SmartDocumentResult:
        started = time.perf_counter()
        source = Path(source_path)
        output_dir = self._create_output_dir(source)
        suffix = source.suffix.lower()

        if suffix == ".pdf":
            document = self._process_pdf(source, output_dir, run_ocr_fallback=run_ocr_fallback, trace_id=trace_id)
            final_export = self.clean_exporter.export_document(document)
        else:
            image_result = self.image_pipeline.run(str(source), verbose=False, trace_id=trace_id)
            document = self._document_from_image_result(image_result)
            final_export = self.clean_exporter.export_document(document)

        status = "completed_with_warnings" if document.warnings or final_export.warnings else "completed"
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        result = SmartDocumentResult(
            document_id=output_dir.name,
            source_file=str(source),
            output_dir=str(output_dir),
            status=status,
            document=document,
            final_export=final_export,
            warnings=list(dict.fromkeys(document.warnings + final_export.warnings)),
            processing_time_ms=elapsed_ms,
        )
        result.output_files = self._write_outputs(result)
        return result

    def _process_pdf(self, source: Path, output_dir: Path, run_ocr_fallback: bool, trace_id: str | None = None) -> DocumentSource:
        document = self.pdf_extractor.extract(source, output_dir=output_dir, render_previews=True)
        self._classify_document(document)
        if run_ocr_fallback:
            self._run_pdf_ocr_fallback(document, trace_id=trace_id)
            self._classify_document(document)
        return document

    def _classify_document(self, document: DocumentSource) -> None:
        text = document.text or "\n\n".join(page.text for page in document.pages)
        document.text = text
        lang_result = self.language_identifier.identify(text)
        classification = self.classifier.classify(text, lang_result.primary_language, image=None)
        doc_type = classification.doc_type

        # Table-heavy PDFs with invoice signals should not remain generic just
        # because the table text was split across cells.
        lower = text.lower()
        table_count = sum(len(page.tables) for page in document.pages)
        if doc_type in {"unknown", "form"} and table_count and re.search(r"\b(invoice|facture|total|vat|tva|amount|montant)\b", lower):
            doc_type = "invoice"
        if doc_type == "unknown" and re.search(r"\b(experience|expérience|education|formation|skills|compétences)\b", lower):
            doc_type = "cv"

        document.language = lang_result.primary_language
        document.direction = lang_result.direction
        document.document_type = doc_type
        document.metadata["language"] = {
            "confidence": lang_result.confidence,
            "probabilities": lang_result.probabilities,
            "method": lang_result.method,
        }
        document.metadata["classification"] = {
            "confidence": classification.confidence,
            "scores": classification.scores,
            "evidence": classification.evidence,
        }

    def _run_pdf_ocr_fallback(self, document: DocumentSource, trace_id: str | None = None) -> None:
        for page in document.pages:
            if not page.needs_ocr:
                continue
            if not page.preview_image:
                document.warnings.append(f"Page {page.number} needs OCR fallback but has no rendered preview.")
                continue
            result = self.image_pipeline.run(page.preview_image, verbose=False, trace_id=trace_id)
            if not result.ocr_data:
                document.warnings.append(f"OCR fallback failed for PDF page {page.number}: {'; '.join(result.errors)}")
                continue
            self._merge_ocr_page_result(document, page, result)

    def _merge_ocr_page_result(self, document: DocumentSource, page: DocumentPage, result: PipelineResult) -> None:
        ocr = result.ocr_data
        ocr_lines = ocr.get("ocr_lines", []) if isinstance(ocr, dict) else []
        spans: List[TextSpan] = []
        regions: List[ReviewRegion] = []
        for index, item in enumerate(ocr_lines):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            bbox = item.get("bbox", [])
            if not text or not isinstance(bbox, list) or len(bbox) < 4:
                continue
            spans.append(
                TextSpan(
                    text=text,
                    page=page.number,
                    bbox=[float(value) for value in bbox[:4]],
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    source="paddleocr_pdf_fallback",
                    language=str(item.get("language", document.language)),
                )
            )
            regions.append(
                ReviewRegion(
                    id=f"p{page.number}-ocr-{index}",
                    page=page.number,
                    bbox=[float(value) for value in bbox[:4]],
                    text=text,
                    region_type="text",
                    confidence=float(item.get("confidence", 0.0) or 0.0),
                    source="paddleocr_pdf_fallback",
                )
            )
        if spans:
            page.spans.extend(spans)
            page.review_regions.extend(regions)
            page.text = "\n".join([page.text, "\n".join(span.text for span in spans)]).strip()
            page.extraction_source = "native_pdf+ocr_fallback"
            page.needs_ocr = False
            document.text = "\n\n".join(p.text for p in document.pages if p.text.strip())
        if result.errors:
            document.warnings.extend(result.errors)

    def _document_from_image_result(self, result: PipelineResult) -> DocumentSource:
        ocr = result.ocr_data or {}
        doc_info: DocumentInfo = result.doc_info
        source = Path(result.source_file)
        metadata = ocr.get("image_metadata", {}) if isinstance(ocr, dict) else {}
        width, height = _size(metadata.get("preprocessed_size") or metadata.get("original_size") or [0, 0])
        page = DocumentPage(
            number=1,
            width=width,
            height=height,
            text=ocr.get("full_text", "") if isinstance(ocr, dict) else "",
            preview_image=(result.output_files or {}).get("preprocessed_image", ""),
            extraction_source="ocr_pipeline",
            metadata={"layout_source": ocr.get("layout_source") if isinstance(ocr, dict) else ""},
        )
        for index, line in enumerate(ocr.get("ocr_lines", []) if isinstance(ocr, dict) else []):
            if not isinstance(line, dict):
                continue
            bbox = line.get("bbox", [])
            text = str(line.get("text", "")).strip()
            if not text or not isinstance(bbox, list) or len(bbox) < 4:
                continue
            span = TextSpan(
                text=text,
                page=1,
                bbox=[float(value) for value in bbox[:4]],
                confidence=float(line.get("confidence", 0.0) or 0.0),
                source="paddleocr",
                language=str(line.get("language", doc_info.language)),
            )
            page.spans.append(span)
            page.review_regions.append(
                ReviewRegion(
                    id=f"p1-ocr-{index}",
                    page=1,
                    bbox=span.bbox,
                    text=text,
                    region_type="text",
                    confidence=span.confidence,
                    source=span.source,
                )
            )
        for block in ocr.get("semantic_blocks", []) if isinstance(ocr, dict) else []:
            if not isinstance(block, dict) or block.get("block_type") != "table_section":
                continue
            page.tables.append(
                TableBlock(
                    page=1,
                    rows=[[line] for line in block.get("lines", []) if str(line).strip()],
                    bbox=[float(value) for value in block.get("bbox", [])[:4]] if isinstance(block.get("bbox"), list) else [],
                    source="ocr_semantic_table",
                    confidence=float(block.get("confidence_avg", 0.0) or 0.0),
                )
            )

        return DocumentSource(
            source_file=result.source_file,
            file_type=source.suffix.lower().lstrip(".") or "image",
            document_name=source.stem,
            pages=[page],
            language=doc_info.language,
            direction=doc_info.direction,
            document_type=doc_info.doc_type,
            text=page.text,
            warnings=list(result.errors),
            metadata={
                "source": "ocr_pipeline",
                "output_dir": result.output_dir,
                "classification": {
                    "confidence": doc_info.confidence,
                    "scores": doc_info.classification_scores,
                    "evidence": doc_info.evidence,
                },
                "output_files": result.output_files,
            },
        )

    def _write_outputs(self, result: SmartDocumentResult) -> Dict[str, str]:
        output_dir = ensure_directory(result.output_dir)
        files = {
            "review_json": output_dir / "document_review.json",
            "final_export_json": output_dir / "final_export.json",
            "audit_export_json": output_dir / "final_export.audit.json",
            "smart_report_json": output_dir / "smart_pipeline_report.json",
        }
        save_json(result.document, files["review_json"])
        save_json(result.final_export.clean_dict(), files["final_export_json"])
        save_json(result.final_export.audit_dict(), files["audit_export_json"])
        save_json(
            {
                "document_id": result.document_id,
                "status": result.status,
                "source_file": result.source_file,
                "output_dir": result.output_dir,
                "processing_time_ms": result.processing_time_ms,
                "warnings": result.warnings,
                "document": {
                    "type": result.document.document_type,
                    "language": result.document.language,
                    "page_count": result.document.page_count,
                },
                "outputs": {key: str(path) for key, path in files.items()},
            },
            files["smart_report_json"],
        )
        return {key: str(path) for key, path in files.items()}

    def _create_output_dir(self, source: Path) -> Path:
        base = ensure_directory(Path(self.config.output_dir) / "runs")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = re.sub(r"[^A-Za-z0-9._-]+", "_", source.stem).strip("._-").lower() or "document"
        candidate = base / f"{stem}_{timestamp}_smart"
        suffix = 2
        while candidate.exists():
            candidate = base / f"{stem}_{timestamp}_smart_{suffix}"
            suffix += 1
        return ensure_directory(candidate)


def _size(value: Any) -> tuple[float, float]:
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return float(value[0]), float(value[1])
    return 0.0, 0.0

