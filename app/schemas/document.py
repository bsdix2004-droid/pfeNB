"""
app/schemas/document.py – Pydantic schemas for document upload
"""
from __future__ import annotations
import uuid
from datetime import datetime

from pydantic import BaseModel

class DocumentResponse(BaseModel):
    """What the API returns after a successful upload"""
    message: str
    id: uuid.UUID
    filename: str
    original_filename: str
    file_type: str
    file_size: int
    status: str # uploaded / processing / done
    minio_path: str
    uploaded_at : datetime
    
    model_config = {"from_attributes": True}
    
    
    