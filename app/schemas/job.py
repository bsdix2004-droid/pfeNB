"""
app/schemas/job.py - Pydantic schemas for extraction jobs

JobCreateRequest  → what the frontend sends when the user clicks Next (upload)
JobResponse       → what the API returns after job creation
"""

import uuid
from datetime import datetime
from pydantic import BaseModel

#__________ JobCreateRequest ____________________
class JobCreateRequest(BaseModel):
    """ 
    what the frontend sends when the user clicks Next (upload interface)
    the frontend only spacifies whoch document it wantw to process
    """
    document_id: uuid.UUID
    #the ID of the uploaded document — the pipeline will process this document
   
#__________ JobResponse ____________________
class JobResponse(BaseModel):
    """" 
    what the API returns after job creation
    The frontend recieves the job_id so it can can poll the status
    """ 
    id: uuid.UUID
    document_id: uuid.UUID
    trace_id: uuid.UUID
    triggered_by: uuid.UUID
    status: str
    ocr_engine: str | None
    ai_model: str | None
    started_at: datetime | None
    completed_at: datetime | None
    duration_ms: int | None
    error_message: str | None
    retry_count: int
    created_at: datetime
    
    model_config = {"from_attributes": True}
    





