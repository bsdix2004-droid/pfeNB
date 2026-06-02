"""
app/workers/tasks.py - Celery background tasks
This file contains the actual work that Celery executes in the background
when the user clicks Next in the upload interface

This file is the worker that:
-Updates the job status at each step
-Runs the AI pipeline (not implemented yet)
-Handles errors and retries
-Calculates the processing duration
"""

import uuid
import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from sqlalchemy import delete, select

from app.config import get_settings
from app.db.session import get_sync_db
from app.engine.pipeline import run_document_pipeline
from app.models.document import Document
from app.models.job import ExtractionJob
from app.models.result import ExtractedField
from app.utils.logging import configure_logging, get_logger, log_stage
from app.workers.celery_app import celery_app

settings = get_settings()
configure_logging(settings.LOG_LEVEL)
logger = get_logger(None).bind(stage="celery")


def _get_s3_client():
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
    )


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return _to_plain(asdict(value))
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_plain(item) for item in value]
    return value


def _field_value_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(_to_plain(value), ensure_ascii=False)


def _data_type(value: Any) -> str:
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int) and not isinstance(value, bool):
        return "integer"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "string"


def _field_category(field_name: str) -> str | None:
    prefix = field_name.split(".", 1)[0].split("[", 1)[0].strip()
    return prefix or None


def _bbox_parts(bbox: Any) -> tuple[float | None, float | None, float | None, float | None]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) < 4:
        return None, None, None, None
    try:
        x1, y1, x2, y2 = [float(value) for value in bbox[:4]]
    except (TypeError, ValueError):
        return None, None, None, None
    return x1, y1, max(0.0, x2 - x1), max(0.0, y2 - y1)


def _flatten(data: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(data, dict):
        output: dict[str, Any] = {}
        for key, value in data.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            output.update(_flatten(value, path))
        return output
    if isinstance(data, list):
        output: dict[str, Any] = {}
        for index, value in enumerate(data):
            output.update(_flatten(value, f"{prefix}[{index}]"))
        return output
    return {prefix: data}


def _extracted_field_payloads(final_export: Any) -> list[dict[str, Any]]:
    evidence = {
        str(getattr(item, "field_path", "")): item
        for item in (getattr(final_export, "evidence", []) or [])
        if getattr(item, "field_path", None)
    }
    payloads: list[dict[str, Any]] = []
    fields = list(getattr(final_export, "fields", []) or [])

    if not fields:
        for field_name, value in _flatten(getattr(final_export, "data", {}) or {}).items():
            if value in (None, "", [], {}):
                continue
            ev = evidence.get(field_name)
            payloads.append(
                {
                    "field_name": field_name,
                    "field_label": field_name.replace("_", " ").replace(".", " > ").title(),
                    "value": value,
                    "confidence": getattr(ev, "confidence", None) if ev else None,
                    "bbox": getattr(ev, "bbox", []) if ev else [],
                    "page_number": getattr(ev, "page", None) if ev else None,
                }
            )
        return payloads

    for field in fields:
        field_name = str(getattr(field, "id", "")).strip()
        if not field_name:
            continue
        ev = evidence.get(field_name)
        bbox = getattr(field, "bbox", []) or (getattr(ev, "bbox", []) if ev else [])
        payloads.append(
            {
                "field_name": field_name,
                "field_label": getattr(field, "label", None),
                "value": getattr(field, "value", None),
                "confidence": getattr(field, "confidence", None),
                "bbox": bbox,
                "page_number": getattr(ev, "page", None) if ev else None,
            }
        )
    return payloads


def _classification_confidence(document_source: Any) -> float | None:
    metadata = getattr(document_source, "metadata", {}) or {}
    classification = metadata.get("classification", {}) if isinstance(metadata, dict) else {}
    try:
        return float(classification.get("confidence"))
    except (TypeError, ValueError):
        return None


def _persist_pipeline_result(db, job: ExtractionJob, document: Document, result: Any) -> int:
    final_export = result.final_export
    db.execute(delete(ExtractedField).where(ExtractedField.job_id == job.id))

    confidences: list[float] = []
    field_count = 0
    for payload in _extracted_field_payloads(final_export):
        value = payload["value"]
        raw_value = _field_value_text(value)
        bbox_x, bbox_y, bbox_w, bbox_h = _bbox_parts(payload.get("bbox"))
        confidence = payload.get("confidence")
        try:
            confidence = float(confidence) if confidence is not None else None
        except (TypeError, ValueError):
            confidence = None
        if confidence is not None:
            confidences.append(confidence)

        db.add(
            ExtractedField(
                id=uuid.uuid4(),
                document_id=document.id,
                job_id=job.id,
                field_name=payload["field_name"],
                field_label=payload.get("field_label"),
                field_category=_field_category(payload["field_name"]),
                raw_value=raw_value,
                ocr_value=raw_value,
                normalized_value=raw_value,
                data_type=_data_type(value),
                page_number=payload.get("page_number"),
                bbox_x=bbox_x,
                bbox_y=bbox_y,
                bbox_w=bbox_w,
                bbox_h=bbox_h,
                confidence=confidence,
                is_validated=False,
                is_skipped=False,
            )
        )
        field_count += 1

    document_source = result.document
    detected_type = getattr(document_source, "document_type", None) or getattr(final_export, "document_type", None)
    if detected_type:
        document.doc_type = str(detected_type).lower()
    document.pages = max(1, int(getattr(document_source, "page_count", None) or document.pages or 1))
    document.confidence_score = (
        sum(confidences) / len(confidences)
        if confidences
        else _classification_confidence(document_source)
    )
    document.status = "done"
    job.ocr_engine = "smart_document_pipeline"
    job.ai_model = "qwen/deterministic"
    return field_count

#_____ Main task
@celery_app.task(
    bind=True,
    
    max_retries=3, #If the pipeline fails, Celery automatically retries 3 times. After 3 failed attempts, the status is set to "failed".
    default_retry_delay=5, #Wait 5 seconds before retrying a failed task to avoid overwhelming the system with rapid retries.
    name="process_document",
)
def process_document(self, job_id: str, document_id: str, trace_id: str | None = None):
    """ 
    Main pipeline task
    Triggered when FastAPI calls: process_document.delay(job_id, document_id)
    """
    
    # convert string IDs to UUID
    job_id = uuid.UUID(job_id)
    document_id = uuid.UUID(document_id)
    
    with get_sync_db() as db:
        try:
            #___1: Retrieve the job from the DB
            job = db.execute(
                select(ExtractionJob).where(ExtractionJob.id == job_id)
            ).scalar_one_or_none()
            
            if job is None:
                logger.error("job_not_found", job_id=str(job_id))
                return

            trace_id = trace_id or str(job.trace_id)
            
            #Record the start of proceesing
            job.started_at = datetime.now(timezone.utc)
            job.status = "ocr_running"
            db.commit()
            
            # __2: Download the document from MinIO.
            with log_stage(trace_id, "celery.download_from_minio"):
                document = db.execute(select(Document).where(Document.id == document_id)).scalar_one_or_none()
                if document is None:
                    raise RuntimeError(f"Document {document_id} not found")
                document.status = "processing"
                db.commit()
                s3 = _get_s3_client()
                try:
                    resp = s3.get_object(Bucket=settings.S3_BUCKET_UPLOADS, Key=document.minio_path)
                    document_bytes = resp["Body"].read()
                except (BotoCoreError, ClientError) as exc:
                    raise RuntimeError(f"MinIO download failed: {exc}") from exc
            
            #__3: Run the pipeline
            job.status = "ai_running"
            db.commit()
            with log_stage(trace_id, "celery.run_pipeline"):
                result = run_document_pipeline(
                    document_bytes=document_bytes,
                    filename=document.original_filename,
                    trace_id=trace_id,
                )

            #__4: save the results
            field_count = _persist_pipeline_result(db, job, document, result)
            
            #__5: update job status to completed
            job.status = "done"
            # → frontend voit "✅ Traitement terminé !"
            job.completed_at = datetime.now(timezone.utc)

            # Calculer la durée du traitement en millisecondes
            job.duration_ms = int(
                (job.completed_at - job.started_at).total_seconds() * 1000
            )
            db.commit()

            get_logger(trace_id).bind(stage="celery").info(
                "job_completed",
                job_id=str(job_id),
                document_id=str(document_id),
                duration_ms=job.duration_ms,
                pipeline_status=result.status,
                field_count=field_count,
            )
        
        except Exception as e:
            # Best-effort fetch trace_id from the job row if possible.
            trace_id = None
            try:
                trace_id = str(job.trace_id)  # type: ignore[name-defined]
            except Exception:
                trace_id = None
            get_logger(trace_id).bind(stage="celery").error(
                "job_failed",
                job_id=str(job_id),
                document_id=str(document_id),
                error=str(e),
            )
            
            job.retry_count += 1
            
            if self.request.retries < self.max_retries:
                # The maximum number of retries has not been reached yet
                # → set the status back to "queued" and retry after 5s
                job.status = "queued"  
                db.commit()
                raise self.retry(exc=e, countdown=5)  
            
            else:
                # The maximum number of retries has been reached
                job.status = "failed"
                job.error_message = str(e)
                try:
                    document.status = "failed"  # type: ignore[name-defined]
                except Exception:
                    pass
                db.commit()
            
            
            
