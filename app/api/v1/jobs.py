""" 
app/api/v1/jobs.py - Extraction job endpoints

POST /jobs/                        - create a job (user clicks Next)
GET  /jobs/{job_id}                - get job details
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, get_db
from app.schemas.job import JobCreateRequest, JobResponse
from app.services.job_service import JobError, JobService
from app.utils.logging import get_logger

router = APIRouter()
logger = get_logger(None).bind(stage="api")

def _job_error_to_http(e: JobError) -> HTTPException:
    """Convert a domain-level JobError to an HTTPException for API responses."""
    return HTTPException(status_code=e.status_code, detail=e.message)

#______ create job _______
@router.post(
    "/",
    response_model=JobResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create an extraction job for (user clicks Next)",
)
async def create_job(
    data: JobCreateRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    
    """
    Triggered when the user clicks Next in the upload interface
    """
    if current_user.role != "user":
       raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Only users can create jobs")
   
    try:
        trace_id = uuid.uuid4()
        service = JobService(db)
        job = await service.create_job(
            document_id=data.document_id,
            current_user=current_user,
            trace_id=trace_id,
        )
        await db.commit()
        # Trigger background processing immediately after job creation.
        from app.workers.task import process_document

        process_document.delay(str(job.id), str(data.document_id), str(trace_id))

        get_logger(trace_id).bind(stage="api").info(
            "job_enqueued",
            job_id=str(job.id),
            document_id=str(data.document_id),
        )
        return JobResponse.model_validate(job)
    except JobError as e:
        await db.rollback()
        raise _job_error_to_http(e)
    
#_____ get job _____
@router.get(
    "/{job_id}",
    response_model=JobResponse,
    summary="Get job details ",
)
async def get_job(
    job_id: uuid.UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """ 
    Retrieve all job details
    Used after the pipeline is completed (status = "done")
    """ 
    try:
        service = JobService(db)
        job = await service.get_job(
            job_id=job_id,
            current_user=current_user,
        )
        return JobResponse.model_validate(job)
    except JobError as e:
        raise _job_error_to_http(e)
    

    




