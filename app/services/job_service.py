""" 
app/services/job_service.py - Extraction job business logic
Handles job creation, status tracking and retrieval
"""

import uuid

from fastapi import status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.document import Document
from app.models.job import ExtractionJob
from app.models.user import User
from app.utils.logging import get_logger

logger = get_logger(None).bind(stage="job_service")

class JobError(Exception):
    """Domain-level job error — converted to HTTP response in the router."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

class JobService:
    def __init__(self, db: AsyncSession):
        self.db = db
        
    #______create job ________
    async def create_job(
        self,
        document_id: uuid.UUID,
        current_user: User,
        trace_id: uuid.UUID,
    ) -> ExtractionJob:
        """ 
        Create an ExtractionJob in the database when the user clicks Next (upload)
        1. Check that the document exists
        2. Check that the document belongs to the user
        3. Check that there is no existing running job for this document
        4. Create the job with status "queued"
        """
            
        #____ 1 _______
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        if document is None:
            raise JobError("Document not found", status.HTTP_404_NOT_FOUND)
        
        #___ 2 _______
        if document.user_id != current_user.id:
            raise JobError(
                "You are not allowed to process this document",
                status.HTTP_403_FORBIDDEN
            )
        
        #___ 3 _______
        existing_job = await self.db.execute(
            select(ExtractionJob).where(
                ExtractionJob.document_id == document_id,
                ExtractionJob.status.in_(["queued", "ocr_running", "ai_running"])
            )
        )
        if existing_job.scalar_one_or_none():
            raise JobError(
                "A job is already running for this document",
                status.HTTP_409_CONFLICT
            )
        
        #___ 4 _______
        job = ExtractionJob(
            id=uuid.uuid4(),
            document_id=document_id,
            trace_id=trace_id,
            triggered_by=current_user.id,
            status="queued",
            retry_count=0,
        )
        self.db.add(job)
        await self.db.flush()  
        
        get_logger(trace_id).bind(stage="job_service").info(
            "job_created",
            document_id=str(document_id),
            user_email=current_user.email,
            job_id=str(job.id),
        )
        return job
    
    #______get job ________
    async def get_job(
        self,
        job_id: uuid.UUID,
        current_user: User,
    ) -> ExtractionJob:
        """
        Retrieve a job by its id
        Verify that the job belongs to the user
        """
        result = await self.db.execute(
            select(ExtractionJob).where(ExtractionJob.id == job_id)
        )
        job = result.scalar_one_or_none()
        
        if job is None:
            raise JobError("Job not found", status.HTTP_404_NOT_FOUND)
        if job.triggered_by != current_user.id:
            raise JobError(
                "You are not allowed to access this job",
                status.HTTP_403_FORBIDDEN
            )
        return job
    
    
