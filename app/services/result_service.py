"""
app/services/result_service.py - Extracted fields and results business logic
"""
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from fastapi import status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.job import ExtractionJob
from app.models.result import ExtractedField, Result
from app.models.user import User
from app.models.document import Document
from app.utils.logging import get_logger

logger = get_logger(None).bind(stage="result_service")

_PATH_TOKEN_RE = re.compile(r"[^.\[\]]+|\[\d+\]")


def _parse_path(path: str) -> list[str | int]:
    """Parse a flat field path like 'a.b[0].c' into ['a', 'b', 0, 'c']."""
    tokens: list[str | int] = []
    for token in _PATH_TOKEN_RE.findall(path):
        if token.startswith("[") and token.endswith("]"):
            tokens.append(int(token[1:-1]))
        else:
            tokens.append(token)
    return tokens


def _set_nested(container: Any, path: str, value: Any) -> None:
    """Insert value at the nested path inside container (dicts/auto-created lists)."""
    segments = _parse_path(path)
    if not segments:
        return
    current: Any = container
    for index, segment in enumerate(segments):
        is_last = index == len(segments) - 1
        next_segment = segments[index + 1] if not is_last else None
        if isinstance(segment, int):
            if not isinstance(current, list):
                return
            while len(current) <= segment:
                current.append(None)
            if is_last:
                current[segment] = value
            else:
                if current[segment] is None:
                    current[segment] = [] if isinstance(next_segment, int) else {}
                current = current[segment]
        else:
            if not isinstance(current, dict):
                return
            if is_last:
                current[segment] = value
            else:
                if segment not in current or current[segment] is None:
                    current[segment] = [] if isinstance(next_segment, int) else {}
                current = current[segment]


def _unflatten(flat: dict[str, Any]) -> dict[str, Any]:
    """Rebuild a nested dict from flat field_name paths."""
    result: dict[str, Any] = {}
    for path, value in flat.items():
        if value in (None, "", [], {}):
            continue
        _set_nested(result, path, value)
    return _prune_empty(result)


def _prune_empty(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _prune_empty(item) for key, item in value.items() if item not in (None, "", [], {})}
    if isinstance(value, list):
        return [_prune_empty(item) for item in value if item not in (None, "", [], {})]
    return value


def _coerce_value(text: str | None, data_type: str | None = None) -> Any:
    """Best-effort restore of the original Python type from its stored text form."""
    if text is None:
        return None
    if data_type == "string":
        return text
    if data_type == "boolean":
        if isinstance(text, bool):
            return text
        return text.strip().lower() in ("true", "1", "yes")
    if data_type in ("integer", "number"):
        try:
            if data_type == "integer":
                return int(text)
            return float(text)
        except (TypeError, ValueError):
            return text
    if data_type in ("array", "object"):
        try:
            return json.loads(text)
        except (TypeError, ValueError):
            return text
    try:
        return json.loads(text)
    except (TypeError, ValueError):
        return text


class ResultError(Exception):
    """Domain-level result error — converted to HTTP response in the router."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

#_____ ResultService _______
class ResultService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def _get_job(
        self, 
        job_id: uuid.UUID, 
        current_user: User
    ) -> ExtractionJob:
        """Helper method to get a job and check permissions."""
        result = await self.db.execute(
            select(ExtractionJob).where(ExtractionJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        if job is None:
            raise ResultError("Job not found", status.HTTP_404_NOT_FOUND)
        if job.triggered_by != current_user.id:
            raise ResultError("You are not allowed to access this job", status.HTTP_403_FORBIDDEN)
        return job
    
    async def get_field(
        self,
        field_id: uuid.UUID,
        current_user: User,
    ) -> ExtractedField:
        """verify that the field exists and belongs to the user"""
        result = await self.db.execute(
            select(ExtractedField).where(ExtractedField.id == field_id)
        )
        field = result.scalar_one_or_none()
        if field is None:
            raise ResultError("Field not found", status.HTTP_404_NOT_FOUND)
        
        # Check that the user has access to the job this field belongs to
        await self._get_job(field.job_id, current_user)
        
        return field
    
    #_____ get all extracted fields _______

    async def get_extracted_fields(
        self,
        job_id: uuid.UUID,
        current_user: User,
    ) -> list[ExtractedField]:
        """
        Retrieve all extracted fields for a job
        used in Interface 1 (Extracted Fields + JSON Preview)
        and Interface 2 (Verification Editor)
        """
        job = await self._get_job(job_id, current_user)  
        
        #verify that the job is completed 
        if job.status != "done":
            raise ResultError(f"Job is not done yet, current status: {job.status}", status.HTTP_400_BAD_REQUEST)

        result = await self.db.execute(
            select(ExtractedField).where(ExtractedField.job_id == job_id).order_by(ExtractedField.created_at.asc())
        )
        return result.scalars().all()
    
    #_____ validate/correct a field _______
    async def validate_field(
        self,
        field_id: uuid.UUID,
        normalized_value: str,
        current_user: User,
    ) -> ExtractedField:
        """
        Triggered when the user clicks Correct 
        Updates the field with the corrected value
        """
        field = await self.get_field(field_id, current_user)
        
        # Update the field with the corrected value and set is_validated to True
        field.normalized_value = normalized_value
        field.is_validated = True
        field.is_skipped = False
        field.validated_by = current_user.id
        field.validated_at = datetime.now(timezone.utc)
        
        await self.db.flush()
        
        logger.info("field_validated", field_id=str(field_id), user_email=current_user.email)
        
        return field
    
    #_____ skip a field _______
    async def skip_field(
        self,
        field_id: uuid.UUID,
        current_user: User,
    ) -> ExtractedField:
        """
        Triggered when the user clicks Skip 
        """
        field = await self.get_field(field_id, current_user)
        
        # Mark the field as skipped
        field.is_validated = False
        field.is_skipped = True
        
        await self.db.flush()
        logger.info("field_skipped", field_id=str(field_id), user_email=current_user.email)
        
        return field
    
    #_____ approve _______
    async def approve(
        self,
        job_id: uuid.UUID,
        current_user: User,
    ) -> None:
        """
        Triggered when the user clicks Approve, confirms that the user has completed verification
        No more changes allowed to the fields after this action
        """
        await self._get_job(job_id, current_user)
        logger.info("job_approved", job_id=str(job_id), user_email=current_user.email)
        
    #_____ export results as JSON _______
    async def export_results(
        self,
        job_id: uuid.UUID,
        current_user: User
    ) -> str:
        """
        Export the results of a job as a clean nested JSON file.
        Used in the Export interface. The output contains ONLY the extracted
        field values (nested), with empty values pruned and original types
        restored when possible. No document_type, no warnings, no evidence.
        """
        job = await self._get_job(job_id, current_user)

        result = await self.db.execute(
            select(ExtractedField).where(ExtractedField.job_id == job_id)
        )
        fields = result.scalars().all()

        flat: dict[str, Any] = {}
        for field in fields:
            if field.is_validated and field.normalized_value is not None:
                value = field.normalized_value
            else:
                value = field.raw_value
            flat[field.field_name] = _coerce_value(value, field.data_type)

        nested = _unflatten(flat)

        confidence_values = [
            field.confidence for field in fields if field.confidence is not None
        ]
        confidence_score = (
            sum(confidence_values) / len(confidence_values)
            if confidence_values
            else None
        )

        doc_result = await self.db.execute(
            select(Document).where(Document.id == job.document_id)
        )
        document = doc_result.scalar_one_or_none()
        if document:
            document.confidence_score = confidence_score
            document.status = "done"
            await self.db.flush()

        exported_data_str = json.dumps(nested, ensure_ascii=False, indent=2)
        result_obj = Result(
            document_id=job.document_id,
            job_id=job_id,
            exported_data=exported_data_str,
            exported_by=current_user.id,
        )
        self.db.add(result_obj)
        await self.db.flush()

        logger.info(
            "results_exported",
            job_id=str(job_id),
            user_email=current_user.email,
            confidence_score=confidence_score,
            field_count=len(flat),
        )
        return exported_data_str

