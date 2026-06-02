"""
app/api/v1/results.py - Extracted fields and results endpoints

GET   /results/{job_id}                    - get all extracted fields for a job
PATCH /results/fields/{field_id}/validate  - validate/correct a field (Correct button)
PATCH /results/fields/{field_id}/skip      - skip a field (Skip button)
POST  /results/{job_id}/approve            - approve all fields (Approve button)
POST  /results/{job_id}/export             - export final JSON (Export button)
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, get_db
from app.schemas.auth import MessageResponse
from app.schemas.result import (
    ExtractedFieldResponse,
    FieldValidateRequest,
)
from app.services.result_service import ResultError, ResultService
from app.utils.logging import get_logger

router = APIRouter()
logger = get_logger(None).bind(stage="api")

def _result_error_to_http(e: ResultError) -> HTTPException:
    """Convert a domain-level ResultError to an HTTPException for API responses."""
    return HTTPException(status_code=e.status_code, detail=e.message)

#_____ get extracted fields _______
@router.get(
    "/{job_id}",
    response_model=list[ExtractedFieldResponse],
    summary="Get all extracted fields for a job",
)
async def get_extracted_fields(
    job_id: uuid.UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Get all extracted fields for a job, used in Interface 1 (Extracted Fields + JSON Preview)
    and Interface 2 (Verification Editor)
    """
    try:
        service = ResultService(db)
        fields = await service.get_extracted_fields(job_id=job_id, current_user=current_user)
        return [ExtractedFieldResponse.model_validate(field) for field in fields]
    except ResultError as e:
        raise _result_error_to_http(e)
    
#_____ validate/correct a field _______
@router.patch(
    "/fields/{field_id}/validate",
    response_model=ExtractedFieldResponse,
    summary="Validate/correct a field (Correct button)",
)
async def validate_field(
    field_id: uuid.UUID,
    data: FieldValidateRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Triggered when the user clicks Correct 
    Updates the field with the corrected value
    """
    try:
        service = ResultService(db)
        field = await service.validate_field(field_id=field_id, normalized_value=data.normalized_value, current_user=current_user)
        await db.commit()
        return ExtractedFieldResponse.model_validate(field)
    except ResultError as e:
        await db.rollback()
        raise _result_error_to_http(e)
    
#_____ skip a field _______
@router.patch(
    "/fields/{field_id}/skip",
    response_model=ExtractedFieldResponse,
    summary="Skip a field (Skip button)",
)
async def skip_field(
    field_id: uuid.UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Triggered when the user clicks Skip 
    """
    try:
        service = ResultService(db)
        field = await service.skip_field(field_id=field_id, current_user=current_user)
        await db.commit()
        return ExtractedFieldResponse.model_validate(field)
    except ResultError as e:
        await db.rollback()
        raise _result_error_to_http(e)

#_____ approve _______
@router.post(
    "/{job_id}/approve",
    response_model=MessageResponse,
    summary="Approve all fields (Approve button)",
)
async def approve(
    job_id: uuid.UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Triggered when the user clicks Approve, confirms that the user has completed
    the verification. The frontend redirects to the Export interface
    
    """
    try:
        service = ResultService(db)
        await service.approve(job_id=job_id, current_user=current_user)
        await db.commit()
        return MessageResponse(message="Job approved successfully, you can now export the results")
    except ResultError as e:
        await db.rollback()
        raise _result_error_to_http(e)

#______ export _______
@router.post(
    "/{job_id}/export",
    summary="Export final JSON (Export button)",
)
async def export_results(
    job_id: uuid.UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Triggered when the user clicks Export, returns the final JSON file for download
    """
    try:
        service = ResultService(db)
        exported_data_str = await service.export_results(job_id=job_id, current_user=current_user)
        await db.commit()
        return Response(content=exported_data_str, media_type="application/json", headers={"Content-Disposition": f"attachment; filename=results_{job_id}.json"})
    except ResultError as e:
        await db.rollback()
        raise _result_error_to_http(e)

    
    
    
