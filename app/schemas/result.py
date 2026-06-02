"""
app/schemas/result.py - Pydantic schemas for extracted fields and results

ExtractedFieldResponse → what the API returns for each extracted field
FieldValidateRequest   → what the frontend sends when the user corrects a field
FieldSkipRequest       → what the frontend sends when the user skips a field
ResultResponse         → what the API returns after export
"""
import uuid
from datetime import datetime
from pydantic import BaseModel

#_______ExtractedFieldResponse ______________
class ExtractedFieldResponse(BaseModel):
    """ 
    what the API returns for each extracted field by the AI pipeline
    Used in Interface 1 (Extracted Fields + JSON Preview)
    and Interface 2 (Verification Editor)
    """
    id: uuid.UUID
    document_id: uuid.UUID
    job_id: uuid.UUID
    field_name: str
    field_label: str | None
    field_category: str | None
    ocr_value: str | None
    raw_value: str | None
    normalized_value: str | None
    data_type: str | None
    page_number: int | None
    
    bbox_x: float | None
    bbox_y: float | None
    bbox_w: float | None
    bbox_h: float | None
    
    confidence: float | None
    is_validated: bool
    is_skipped: bool
    validated_by: uuid.UUID | None
    validated_at: datetime | None
    created_at: datetime
    
    model_config = {"from_attributes": True}
    
#_____FieldValidateRequest __________ 
class FieldValidateRequest(BaseModel):
    """ 
    what the frontend sends when the user corrects a field (Correct button)
    in Interface 2 (Verification Editor)
    """
    normalized_value: str 

#_____ FieldSkipRequest __________
class FieldSkipRequest(BaseModel):
    """ 
    what the frontend sends when the user skips a field (skip button)
    in Interface 2 (Verification Editor), the field will be marked is_skipped = True
    and exported with raw_value in the final JSON
    """
    field_id: uuid.UUID

#_____ ResultResponse __________
class ResultResponse(BaseModel):
    """ 
    what the API returns after the user clicks Export
    Contains the final assembled JSON with all validated values
    """    
    id: uuid.UUID
    document_id: uuid.UUID
    job_id: uuid.UUID
    exported_data: str
    exported_by: uuid.UUID
    exported_at: datetime
    
    model_config = {"from_attributes": True}
    
    
