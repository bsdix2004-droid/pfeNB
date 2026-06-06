"""Pipeline orchestration for local document intelligence."""

from __future__ import annotations

import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
from html import escape
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import cv2
import numpy as np

from app.config import OCRPipelineConfig, get_settings
from app.ocr_pipeline.core.ai_orchestrator import DocumentAIOrchestrator, AIOrchestrationResult
from app.ocr_pipeline.core.detector import DocumentDetector, DocumentInfo
from app.ocr_pipeline.core.language_identifier import LanguageIdentifier
from app.ocr_pipeline.core.models import Document, Page, SemanticRegion
from app.ocr_pipeline.core.block_types import BlockType
from app.ocr_pipeline.core.ocr_engine import OCRCandidateResult, PaddleOCREngine, TextBlock
from app.ocr_pipeline.core.script_detector import ScriptDetector
from app.ocr_pipeline.exporters.document_exporter import DocumentExporter
from app.ocr_pipeline.layout.reading_order import ReadingOrderEngine
from app.ocr_pipeline.layout.region_builder import RegionBuilder
from app.ocr_pipeline.layout.table_understanding import parse_ocr_table
from app.ocr_pipeline.ocr.ensemble import OCREnsemble
from app.ocr_pipeline.ocr.paddle_vl import LayoutAdapter
from app.ocr_pipeline.preprocessing.pipeline import DocumentPreprocessor, PreprocessingResult
from app.ocr_pipeline.semantic.document_graph import DocumentGraph
from app.ocr_pipeline.semantic.reconstructor import OCRLine
from app.ocr_pipeline.summarization.summary_engine import SummaryEngine
from app.ocr_pipeline.utils.file_utils import ensure_directory, save_json
from app.ocr_pipeline.utils.image_utils import load_image_with_metadata, save_image
from app.ocr_pipeline.utils.logger import get_logger
from app.utils.logging import get_logger as get_trace_logger
from structlog.contextvars import bind_contextvars, unbind_contextvars
from app.ocr_pipeline.utils.template_loader import TemplateLoader
from app.ocr_pipeline.validation.grounding import GroundingChecker


LOGGER = get_logger(__name__)


@dataclass(slots=True)
class PipelineResult:
    source_file: str
    doc_info: DocumentInfo
    ocr_data: Dict[str, Any]
    vision_data: Dict[str, Any]
    summary_data: Dict[str, Any]
    processing_time_ms: int
    document: Optional[Document] = None
    errors: List[str] = field(default_factory=list)
    output_dir: str = ""
    output_files: Dict[str, str] = field(default_factory=dict)
    steps: List[Dict[str, Any]] = field(default_factory=list)
    tools: Dict[str, Any] = field(default_factory=dict)


class PipelineProgress:
    """Pipeline progress collector + logger (no prints; JSON logs only)."""

    def __init__(self, verbose: bool = True) -> None:
        self.verbose = verbose
        self.steps: List[Dict[str, Any]] = []

    def header(self, source: Path, output_dir: Path) -> None:
        LOGGER.info(
            "pipeline_header",
            source_file=str(source),
            output_dir=str(output_dir),
        )

    def step(self, name: str) -> "PipelineStep":
        return PipelineStep(self, name)

    def finish(self, name: str, status: str, detail: str, elapsed_ms: int) -> None:
        status = status.upper()
        item = {
            "name": name,
            "status": status,
            "detail": detail,
            "elapsed_ms": elapsed_ms,
        }
        self.steps.append(item)
        LOGGER.info(
            "pipeline_step",
            step=name,
            status=status,
            detail=detail,
            duration_ms=elapsed_ms,
        )

    def footer(self, status: str, output_files: Dict[str, str], elapsed_ms: int) -> None:
        LOGGER.info(
            "pipeline_footer",
            status=status,
            duration_ms=elapsed_ms,
            output_files=output_files,
        )


class PipelineStep:
    def __init__(self, progress: PipelineProgress, name: str) -> None:
        self.progress = progress
        self.name = name
        self.detail = ""
        self.status = "OK"
        self.started = 0.0

    def __enter__(self) -> "PipelineStep":
        self.started = time.perf_counter()
        stage = "pipeline." + re.sub(r"[^a-z0-9]+", "_", self.name.strip().lower()).strip("_")
        bind_contextvars(stage=stage)
        LOGGER.info("step_start", step=self.name)
        return self

    def done(self, detail: str = "", status: str = "OK") -> None:
        self.detail = detail
        self.status = status

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> bool:
        elapsed_ms = int((time.perf_counter() - self.started) * 1000)
        if exc is not None:
            self.progress.finish(self.name, "FAIL", str(exc), elapsed_ms)
            try:
                unbind_contextvars("stage")
            except Exception:
                pass
            return False
        self.progress.finish(self.name, self.status, self.detail, elapsed_ms)
        try:
            unbind_contextvars("stage")
        except Exception:
            pass
        return False


class DocumentPipeline:
    """Image -> preprocessing -> OCR/layout -> semantics -> AI checks -> outputs."""

    def __init__(self, config: OCRPipelineConfig | None = None) -> None:
        self.config: OCRPipelineConfig = config or get_settings().OCR_PIPELINE
        self.preprocessor = DocumentPreprocessor(config.preprocessing)
        self.detector = DocumentDetector(get_settings())
        self.script_detector = ScriptDetector(config.script)
        self.language_identifier = LanguageIdentifier(config.language)
        self.ocr_engine = PaddleOCREngine(config.ocr)
        self.ocr_ensemble = OCREnsemble()
        self.layout_adapter = LayoutAdapter(use_gpu=config.layout.use_gpu)
        self.region_builder = RegionBuilder()
        self.reading_order = ReadingOrderEngine()
        # parse_ocr_table() is imported and called on table_section blocks after OCR
        self.ai_orchestrator = DocumentAIOrchestrator(config)
        self.grounding_checker = GroundingChecker()
        self.summary_engine = SummaryEngine()
        self.exporter = DocumentExporter()

    def run(self, image_path: str, verbose: bool = True, trace_id: str | None = None) -> PipelineResult:
        start_time = time.perf_counter()
        # Bind trace_id once for the whole run so that all nested module logs inherit it.
        get_trace_logger(trace_id)
        source = Path(image_path)
        output_dir = self._create_output_dir(source)
        progress = PipelineProgress(verbose=verbose)
        progress.header(source, output_dir)
        errors: List[str] = []
        input_metadata: Dict[str, Any] = {}

        try:
            with progress.step("Load image") as step:
                loaded = load_image_with_metadata(str(source), self.config.input)
                image = loaded.image
                input_metadata = loaded.metadata
                original_size = [int(image.shape[1]), int(image.shape[0])]
                size_mb = input_metadata.get("source_file_size_mb", 0)
                loader = input_metadata.get("loader", "opencv")
                normalized = " normalized" if input_metadata.get("normalized") else ""
                step.done(f"{original_size[0]}x{original_size[1]}, {size_mb} MB, {loader}{normalized}")

            with progress.step("Preprocess image") as step:
                preprocessing_result = self.preprocessor.process_with_metadata(image)
                clean_image = preprocessing_result.image
                selected = preprocessing_result.selected_candidate.name
                step.done(f"{len(preprocessing_result.candidates)} candidates, selected {selected}")

            layout_image = self._as_bgr(clean_image)

            with progress.step("Layout detection") as step:
                layout_data = self.layout_adapter.analyze(layout_image)
                layout_blocks = layout_data.get("detected_regions", [])
                layout_status = "OK" if layout_data.get("available") else "WARN"
                step.done(
                    f"{layout_data.get('vision_source', 'none')} - {len(layout_blocks)} regions",
                    status=layout_status,
                )

            with progress.step("Script detection") as step:
                script_result = self.script_detector.detect(clean_image, layout_blocks)
                step.done(
                    f"scripts: {', '.join(script_result.scripts_found)} "
                    f"({script_result.method}, {script_result.confidence:.0%})"
                )

            with progress.step("OCR text extraction") as step:
                initial_info = DocumentInfo("unknown", "en", "ltr", 0.0)
                vector_lines = input_metadata.get("vector_text_lines", [])
                if isinstance(vector_lines, list) and vector_lines:
                    ocr_selection = self._vector_text_ocr_selection(vector_lines)
                    preprocessing_result.select_candidate(0)
                else:
                    ocr_selection = self._extract_best_ocr_isolated(
                        preprocessing_result,
                        initial_info,
                        scripts=script_result.scripts_found,
                        layout_blocks=layout_blocks,
                    )
                    preprocessing_result.select_candidate(ocr_selection.candidate_index)
                clean_image = preprocessing_result.image
                preprocessed_size = [int(clean_image.shape[1]), int(clean_image.shape[0])]
                full_text = self.ocr_engine.cleaned_text(ocr_selection.blocks) or "\n".join(
                    line.text for line in ocr_selection.lines
                )
                langs = ",".join(ocr_selection.languages_used or self.config.ocr.languages)
                skipped = ""
                if ocr_selection.fallback_languages_skipped:
                    skipped = f"; fallback skipped: {','.join(ocr_selection.fallback_languages_skipped)}"
                status = "OK" if ocr_selection.raw_line_count else "WARN"
                step.done(
                    f"{ocr_selection.raw_line_count} lines, {ocr_selection.avg_confidence:.1%} confidence, languages {langs}{skipped}",
                    status=status,
                )

            with progress.step("Language identification") as step:
                lang_result = self.language_identifier.identify(full_text)
                step.done(
                    f"{lang_result.primary_language} via {lang_result.method} "
                    f"({lang_result.confidence:.0%} confidence)"
                )

            with progress.step("Document classification") as step:
                doc_info = self.detector.detect_with_language(layout_image, full_text, lang_result)
                ocr_selection = self.ocr_engine.rebuild_blocks(
                    ocr_selection,
                    doc_info,
                    clean_image.shape[:2],
                    layout_blocks=layout_blocks,
                )
                blocks = ocr_selection.blocks
                step.done(f"{doc_info.doc_type}, {doc_info.language}, {len(blocks)} blocks")

            with progress.step("Table parsing") as step:
                table_count = 0
                for block in blocks:
                    if block.block_type in (BlockType.TABLE_SECTION, BlockType.INVOICE_ITEMS_SECTION):
                        parsed = parse_ocr_table(block)
                        if parsed.get("col_count", 0) > 0:
                            block.metadata["table_data"] = parsed
                            table_count += 1
                step.done(f"{table_count} tables parsed")

            with progress.step("AI extraction (primary)") as step:
                ai_result = self.ai_orchestrator.analyze(
                    blocks,
                    doc_info,
                    layout_info=layout_data,
                )
                
                if ai_result.structured_data:
                    grounding_report = self.grounding_checker.verify(ai_result.structured_data, full_text)
                    if not grounding_report["is_grounded"]:
                        LOGGER.warning("AI extraction contains potential hallucinations: %s", 
                                       grounding_report["field_status"])
                
                if ai_result.errors:
                    errors.extend(ai_result.errors)
                ai_status = "WARN" if ai_result.errors else "OK"
                step.done(self._ai_detail(ai_result), status=ai_status)

            with progress.step("Semantic graph construction") as step:
                regions = self.region_builder.build_from_ocr(blocks)
                
                regions = self.reading_order.sort(regions)
                
                graph = DocumentGraph()
                graph.build(regions)
                
                page = Page(number=1, width=preprocessed_size[0], height=preprocessed_size[1], regions=regions)
                document = Document(pages=[page], metadata=input_metadata, graph=graph)
                step.done(f"{len(regions)} semantic regions, {len(graph._edges)} relations")

            with progress.step("Build JSON and HTML outputs") as step:
                vision_output = self.exporter.build_vision_output(layout_data, blocks)
                summary_data = self.summary_engine.build(blocks, doc_info, ai_result.structured_data, vision_output)
                ocr_output = self.exporter.build_ocr_output(
                    source=source,
                    doc_info=doc_info,
                    blocks=blocks,
                    preprocessing=preprocessing_result.to_dict(),
                    ocr_selection=ocr_selection,
                    layout_data=layout_data,
                    original_size=original_size,
                    preprocessed_size=preprocessed_size,
                    low_confidence_threshold=self.config.ocr.low_confidence_threshold,
                )
                output_files = self._write_core_files(
                    output_dir,
                    ocr_output,
                    vision_output,
                    summary_data,
                    clean_image,
                    blocks,
                    preprocessed_size,
                )
                step.done(str(output_dir))

            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            tools = self._tool_status(preprocessing_result, ocr_selection, layout_data, ai_result)
            status = "completed_with_warnings" if errors else "completed"
            report = self._build_report(
                status=status,
                source=source,
                output_dir=output_dir,
                output_files=output_files,
                steps=progress.steps,
                tools=tools,
                input_metadata=input_metadata,
                elapsed_ms=elapsed_ms,
                errors=errors,
                doc_info=doc_info,
                ocr_selection=ocr_selection,
            )
            self._write_report_files(output_dir, output_files, report)
            self._update_latest_outputs(output_files)
            progress.footer(status, output_files, elapsed_ms)

            return PipelineResult(
                source_file=str(source),
                doc_info=doc_info,
                ocr_data=ocr_output,
                vision_data=vision_output,
                summary_data=summary_data,
                processing_time_ms=elapsed_ms,
                document=document,
                errors=errors,
                output_dir=str(output_dir),
                output_files=output_files,
                steps=progress.steps,
                tools=tools,
            )
        except BaseException as exc:
            if isinstance(exc, KeyboardInterrupt):
                raise
            LOGGER.error("Pipeline failed: %s", exc, exc_info=True)
            errors.append(str(exc))
            elapsed_ms = int((time.perf_counter() - start_time) * 1000)
            output_files = self._write_failure_files(output_dir, source, errors, progress.steps, elapsed_ms)
            progress.footer("failed", output_files, elapsed_ms)
            return PipelineResult(
                source_file=str(source),
                doc_info=DocumentInfo("unknown", "unknown", "ltr", 0.0),
                ocr_data={},
                vision_data={},
                summary_data={},
                processing_time_ms=elapsed_ms,
                errors=errors,
                output_dir=str(output_dir),
                output_files=output_files,
                steps=progress.steps,
                tools={},
            )

    def run_batch(self, image_paths: Sequence[str], verbose: bool = False) -> List[PipelineResult]:
        return [self.run(path, verbose=verbose) for path in image_paths]

    async def run_async(self, image_path: str, verbose: bool = False) -> PipelineResult:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.run, image_path, verbose)

    async def run_batch_async(self, image_paths: Sequence[str], verbose: bool = False) -> List[PipelineResult]:
        results: List[PipelineResult] = []
        for path in image_paths:
            results.append(await self.run_async(path, verbose=verbose))
        return results

    def _write_core_files(
        self,
        output_dir: Path,
        ocr_data: Dict[str, Any],
        vision_data: Dict[str, Any],
        summary_data: Dict[str, Any],
        clean_image: np.ndarray,
        blocks: List[TextBlock],
        preprocessed_size: List[int],
    ) -> Dict[str, str]:
        ensure_directory(output_dir)
        paths = {
            "ocr_json": output_dir / "document_ocr.json",
            "summary_json": output_dir / "document_summary.json",
            "vision_json": output_dir / "document_vision.json",
            "preprocessed_image": output_dir / "document_preprocessed.png",
            "layout_viewer": output_dir / "layout_viewer.html",
        }
        save_json(ocr_data, paths["ocr_json"])
        save_json(summary_data, paths["summary_json"])
        save_json(vision_data, paths["vision_json"])
        save_image(clean_image, paths["preprocessed_image"])
        viewer = self.exporter.build_layout_viewer_html("document_preprocessed.png", blocks, preprocessed_size)
        paths["layout_viewer"].write_text(viewer, encoding="utf-8")
        LOGGER.info("Output saved to %s", output_dir)
        return {key: str(path) for key, path in paths.items()}

    def _write_report_files(self, output_dir: Path, output_files: Dict[str, str], report: Dict[str, Any]) -> None:
        report_json = output_dir / "pipeline_report.json"
        report_html = output_dir / "pipeline_report.html"
        output_files["pipeline_report_json"] = str(report_json)
        output_files["pipeline_report_html"] = str(report_html)
        report["outputs"] = dict(output_files)
        save_json(report, report_json)
        report_html.write_text(self._build_report_html(report), encoding="utf-8")

    def _write_failure_files(
        self,
        output_dir: Path,
        source: Path,
        errors: List[str],
        steps: List[Dict[str, Any]],
        elapsed_ms: int,
    ) -> Dict[str, str]:
        ensure_directory(output_dir)
        failure_payload = {
            "status": "failed",
            "source_file": str(source),
            "errors": errors,
            "processing_time_ms": elapsed_ms,
        }
        files = {
            "ocr_json": output_dir / "document_ocr.json",
            "summary_json": output_dir / "document_summary.json",
            "vision_json": output_dir / "document_vision.json",
            "layout_viewer": output_dir / "layout_viewer.html",
        }
        save_json(failure_payload, files["ocr_json"])
        save_json(failure_payload, files["summary_json"])
        save_json(failure_payload, files["vision_json"])
        files["layout_viewer"].write_text(self._empty_layout_html(errors), encoding="utf-8")
        output_files = {key: str(path) for key, path in files.items()}
        report = self._build_report(
            status="failed",
            source=source,
            output_dir=output_dir,
            output_files=output_files,
            steps=steps,
            tools={},
            input_metadata={},
            elapsed_ms=elapsed_ms,
            errors=errors,
            doc_info=DocumentInfo("unknown", "unknown", "ltr", 0.0),
            ocr_selection=None,
        )
        self._write_report_files(output_dir, output_files, report)
        self._update_latest_outputs(output_files)
        return output_files

    def _build_report(
        self,
        status: str,
        source: Path,
        output_dir: Path,
        output_files: Dict[str, str],
        steps: List[Dict[str, Any]],
        tools: Dict[str, Any],
        input_metadata: Dict[str, Any],
        elapsed_ms: int,
        errors: List[str],
        doc_info: DocumentInfo,
        ocr_selection: OCRCandidateResult | None,
    ) -> Dict[str, Any]:
        return {
            "status": status,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "source_file": str(source),
            "input": input_metadata,
            "output_dir": str(output_dir),
            "processing_time_ms": elapsed_ms,
            "document": {
                "type": doc_info.doc_type,
                "language": doc_info.language,
                "confidence": round(float(doc_info.confidence), 4),
            },
            "ocr": ocr_selection.to_dict() if ocr_selection else {},
            "steps": steps,
            "tools": tools,
            "outputs": dict(output_files),
            "warnings": errors,
        }

    def _tool_status(
        self,
        preprocessing: PreprocessingResult,
        ocr_selection: OCRCandidateResult,
        layout_data: Dict[str, Any],
        ai_result: AIOrchestrationResult,
    ) -> Dict[str, Any]:
        return {
            "preprocessing": {
                "name": "OpenCV adaptive preprocessing",
                "status": "ok",
                "selected_candidate": preprocessing.selected_candidate.name,
                "candidate_count": len(preprocessing.candidates),
            },
            "ocr": {
                "name": "SVG vector text extractor"
                if ocr_selection.engine_version == "svg_text_extractor"
                else "PaddleOCR PP-OCRv5",
                "status": "ok" if ocr_selection.raw_line_count else "warning",
                "engine_version": ocr_selection.engine_version,
                "languages_used": ocr_selection.languages_used,
                "fallback_languages_skipped": ocr_selection.fallback_languages_skipped,
                "lines": ocr_selection.raw_line_count,
                "avg_confidence": round(float(ocr_selection.avg_confidence), 4),
            },
            "layout": self._layout_tool_status("PP-DocLayoutV3", layout_data, enabled=self.config.layout.enabled),
            "qwen_ollama": ai_result.providers.get("structured_extraction", {}),
            "deterministic_schema": ai_result.providers.get("deterministic_schema", {}),
        }

    def _layout_tool_status(self, name: str, payload: Dict[str, Any], enabled: bool) -> Dict[str, Any]:
        if not enabled:
            return {"name": name, "enabled": False, "status": "skipped", "reason": "disabled in config.py"}
        if payload.get("available"):
            return {"name": name, "enabled": True, "status": "ok"}
        return {
            "name": name,
            "enabled": True,
            "status": "warning",
            "reason": payload.get("reason", "unavailable"),
        }

    def _ai_detail(self, ai_result: AIOrchestrationResult) -> str:
        parts = []
        labels = {
            "structured_extraction": "Qwen/Ollama",
            "deterministic_schema": "local rules (fallback)",
        }
        for key, label in labels.items():
            provider = ai_result.providers.get(key, {})
            status = provider.get("status", "unknown")
            parts.append(f"{label} {status}")
        return "; ".join(parts)

    def _extract_best_ocr_isolated(
        self,
        preprocessing_result: PreprocessingResult,
        doc_info: DocumentInfo,
        scripts: Sequence[str] | None = None,
        layout_blocks: Sequence[Dict[str, object]] | None = None,
    ) -> OCRCandidateResult:
        with tempfile.TemporaryDirectory(prefix="docai_ocr_") as temp_dir:
            temp_root = Path(temp_dir)
            candidates = self._write_ocr_candidate_images(preprocessing_result, temp_root)
            result, error = self._run_ocr_worker(candidates, doc_info, temp_root, scripts=scripts, layout_blocks=layout_blocks)
            if result is not None:
                return result

            last_error = error
            for candidate in candidates:
                result, error = self._run_ocr_worker([candidate], doc_info, temp_root, scripts=scripts, layout_blocks=layout_blocks)
                if result is not None:
                    result.candidate_index = int(candidate["index"])
                    result.candidate_name = str(candidate["name"])
                    return result
                last_error = error

        raise RuntimeError(f"OCR worker failed for all candidates: {last_error}")

    def _write_ocr_candidate_images(
        self,
        preprocessing_result: PreprocessingResult,
        temp_root: Path,
    ) -> List[Dict[str, Any]]:
        payloads: List[Dict[str, Any]] = []
        for index, candidate in enumerate(preprocessing_result.candidates):
            path = temp_root / f"ocr_candidate_{index}.png"
            if not cv2.imwrite(str(path), candidate.image):
                continue
            payloads.append({"index": index, "name": candidate.name, "path": str(path)})
        if not payloads:
            raise RuntimeError("No OCR candidate images could be written")
        return payloads

    def _run_ocr_worker(
        self,
        candidates: Sequence[Dict[str, Any]],
        doc_info: DocumentInfo,
        temp_root: Path,
        scripts: Sequence[str] | None = None,
        layout_blocks: Sequence[Dict[str, object]] | None = None,
    ) -> tuple[OCRCandidateResult | None, str]:
        input_path = temp_root / f"ocr_input_{time.perf_counter_ns()}.json"
        output_path = temp_root / f"ocr_output_{time.perf_counter_ns()}.json"
        payload = {
            "doc_info": asdict(doc_info),
            "candidates": [
                {"name": str(item["name"]), "path": str(item["path"])}
                for item in candidates
            ],
            "scripts": list(scripts or []),
            "layout_blocks": list(layout_blocks or []),
        }
        input_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        command = [sys.executable, "-m", "app.ocr_pipeline.core.ocr_worker", str(input_path), str(output_path)]
        try:
            completed = subprocess.run(
                command,
                cwd=str(Path(__file__).resolve().parents[3]),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=max(30, int(self.config.ocr.worker_timeout_sec)),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return None, f"timeout after {self.config.ocr.worker_timeout_sec}s"

        if completed.returncode != 0 or not output_path.exists():
            return None, f"worker exited with code {completed.returncode}"
        try:
            data = json.loads(output_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return None, f"invalid worker output: {exc}"
        if not isinstance(data, dict):
            return None, "worker output was not a JSON object"
        if data.get("error"):
            return None, str(data["error"])
        try:
            return self._ocr_selection_from_worker(data), ""
        except Exception as exc:
            return None, f"could not decode worker OCR result: {exc}"

    def _ocr_selection_from_worker(self, data: Dict[str, Any]) -> OCRCandidateResult:
        lines: List[OCRLine] = []
        for index, item in enumerate(data.get("lines", [])):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox", [])
            polygon = item.get("polygon", [])
            if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
                continue
            if not isinstance(polygon, (list, tuple)) or len(polygon) < 4:
                x1, y1, x2, y2 = [int(float(value)) for value in bbox[:4]]
                polygon = [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]
            else:
                polygon = [(float(point[0]), float(point[1])) for point in polygon[:4] if isinstance(point, (list, tuple)) and len(point) >= 2]
            if len(polygon) < 4:
                continue
            line = OCRLine(
                text=str(item.get("text", "")),
                polygon=polygon,
                bbox=tuple(int(float(value)) for value in bbox[:4]),  # type: ignore[arg-type]
                confidence=float(item.get("confidence", 0.0) or 0.0),
                source_index=int(item.get("source_index", index) or index),
                language=str(item.get("language", "unknown")),
                low_confidence=bool(item.get("low_confidence", False)),
            )
            if line.text.strip():
                lines.append(line)
        return OCRCandidateResult(
            candidate_name=str(data.get("candidate_name", "candidate_0")),
            candidate_index=int(data.get("candidate_index", 0) or 0),
            blocks=[],
            lines=lines,
            score=float(data.get("score", 0.0) or 0.0),
            avg_confidence=float(data.get("avg_confidence", 0.0) or 0.0),
            raw_line_count=int(data.get("raw_line_count", len(lines)) or len(lines)),
            char_count=int(data.get("char_count", sum(len(line.text) for line in lines)) or 0),
            engine_version=str(data.get("engine_version", "unknown")),
            languages_used=list(data.get("languages_used", [])),
            fallback_languages_skipped=list(data.get("fallback_languages_skipped", [])),
        )

    def _vector_text_ocr_selection(self, vector_lines: Sequence[Dict[str, Any]]) -> OCRCandidateResult:
        lines: List[OCRLine] = []
        for index, item in enumerate(vector_lines):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text", "")).strip()
            bbox_raw = item.get("bbox", [])
            if not text or not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) < 4:
                continue
            x1, y1, x2, y2 = [int(float(value)) for value in bbox_raw[:4]]
            if x2 <= x1 or y2 <= y1:
                continue
            line = OCRLine(
                text=text,
                polygon=[(x1, y1), (x2, y1), (x2, y2), (x1, y2)],
                bbox=(x1, y1, x2, y2),
                confidence=float(item.get("confidence", 1.0) or 1.0),
                source_index=index,
            )
            line.language = "vector"
            line.low_confidence = False
            lines.append(line)

        char_count = sum(len(line.text) for line in lines)
        avg_conf = float(np.mean([line.confidence for line in lines])) if lines else 0.0
        return OCRCandidateResult(
            candidate_name="svg_vector_text",
            candidate_index=0,
            blocks=[],
            lines=sorted(lines, key=lambda item: (item.y, item.x)),
            score=1.0 if lines else 0.0,
            avg_confidence=avg_conf,
            raw_line_count=len(lines),
            char_count=char_count,
            engine_version="svg_text_extractor",
            languages_used=["svg-vector"],
        )

    def _build_report_html(self, report: Dict[str, Any]) -> str:
        steps = report.get("steps", [])
        tools = report.get("tools", {})
        outputs = report.get("outputs", {})
        warnings = report.get("warnings", [])
        document = report.get("document", {})
        step_rows = "".join(
            "<tr>"
            f"<td>{escape(str(step.get('name', '')))}</td>"
            f"<td><span class=\"status {escape(str(step.get('status', '')).lower())}\">{escape(str(step.get('status', '')))}</span></td>"
            f"<td>{escape(str(step.get('detail', '')))}</td>"
            f"<td>{int(step.get('elapsed_ms', 0)) / 1000:.2f}s</td>"
            "</tr>"
            for step in steps
        )
        tool_rows = "".join(
            "<tr>"
            f"<td>{escape(str(value.get('name') or key))}</td>"
            f"<td><span class=\"status {escape(str(value.get('status', '')).lower())}\">{escape(str(value.get('status', 'unknown')).upper())}</span></td>"
            f"<td>{escape(self._tool_detail(value))}</td>"
            "</tr>"
            for key, value in tools.items()
            if isinstance(value, dict)
        )
        output_links = "".join(
            f'<li><a href="{escape(Path(path).name)}">{escape(label.replace("_", " ").title())}</a></li>'
            for label, path in outputs.items()
        )
        warning_items = "".join(f"<li>{escape(str(item))}</li>" for item in warnings) or "<li>None</li>"
        return TemplateLoader.load("pipeline_report.html").format(
            status=escape(str(report.get("status", ""))),
            source_file=escape(str(report.get("source_file", ""))),
            output_dir=escape(str(report.get("output_dir", ""))),
            document_type=escape(str(document.get("type", "unknown"))),
            language=escape(str(document.get("language", "unknown"))),
            confidence=f"{float(document.get('confidence', 0.0)):.2%}",
            time=f"{int(report.get('processing_time_ms', 0)) / 1000:.2f}s",
            step_rows=step_rows,
            tool_rows=tool_rows,
            output_links=output_links,
            warning_items=warning_items,
        )

    def _tool_detail(self, value: Dict[str, Any]) -> str:
        pieces = []
        for key in ("model", "engine_version", "selected_candidate", "reason", "lines", "avg_confidence"):
            if key in value and value[key] not in (None, "", []):
                pieces.append(f"{key}: {value[key]}")
        if value.get("languages_used"):
            pieces.append("languages: " + ",".join(str(item) for item in value["languages_used"]))
        if value.get("fallback_languages_skipped"):
            pieces.append("fallback skipped: " + ",".join(str(item) for item in value["fallback_languages_skipped"]))
        return "; ".join(pieces)

    def _empty_layout_html(self, errors: Sequence[str]) -> str:
        items = "".join(f"<li>{escape(str(error))}</li>" for error in errors)
        return f"""<!doctype html>
<html lang="en">
<head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Layout unavailable</title></head>
<body style="font-family:Arial,sans-serif;margin:24px;color:#172033">
<h1>Layout Viewer Unavailable</h1>
<p>The run failed before a document layout could be generated.</p>
<ul>{items}</ul>
</body>
</html>"""

    def _update_latest_outputs(self, output_files: Dict[str, str]) -> None:
        latest_dir = ensure_directory(Path(self.config.output_dir) / "latest")
        for path_text in output_files.values():
            path = Path(path_text)
            if not path.exists() or not path.is_file():
                continue
            shutil.copy2(path, latest_dir / path.name)

    def _create_output_dir(self, source: Path) -> Path:
        base = ensure_directory(Path(self.config.output_dir) / "runs")
        stem = self._slugify(source.stem or "document")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate = base / f"{stem}_{timestamp}"
        suffix = 2
        while candidate.exists():
            candidate = base / f"{stem}_{timestamp}_{suffix}"
            suffix += 1
        return ensure_directory(candidate)

    def _slugify(self, value: str) -> str:
        clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._-")
        return clean.lower() or "document"

    def _as_bgr(self, image: np.ndarray) -> np.ndarray:
        if image.ndim == 3:
            return image
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

