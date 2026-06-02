"""
app/api/v1/documents.py – Document endpoints

POST /documents/upload    Upload a document (user only)
GET /api/v1/documents/owned → returns all of the user's documents
GET /api/v1/documents/ → view all documents of all users
GET /api/v1/documents/{id} → view a specific document 
DELETE /api/v1/documents/{id} → delete document
"""
from typing import Annotated
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.deps import CurrentUser, get_db
from app.schemas.document import DocumentResponse
from app.services.document_service import DocumentError, DocumentService
from app.utils.logging import get_logger

router = APIRouter()
logger = get_logger(None).bind(stage="api")

def _document_error_to_http(e: DocumentError) -> HTTPException:
    """Convert a DocumentError into an HTTPException"""
    return HTTPException(status_code=e.status_code, detail=e.message)

#_______ Upload __________
@router.post(
    "/upload",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a document (user ONLY)",
)
async def upload(
    file: Annotated[UploadFile, File(description="PDF, PNG, JPEG, JPG ou TIFF — max 5MB")],
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Upload a document to MinIO and save metadata in DB.
    - Allowed extensions : .pdf, .png, .jpeg, .jpg, .tiff
    - Max size : 5 MB
    - Role required : user
    """
    
    #_______ role verification ________ 
    if current_user.role != "user":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only users can upload documents",
        )
    
    #_________ Service call ________
    try:
        #create the service with the DB session
        service = DocumentService(db)
        
        document = await service.upload_document(
            file=file,
            current_user=current_user,
        )
        return DocumentResponse(
            message="Document uploaded successfully",
            id=document.id,
            filename=document.filename,
            original_filename=document.original_filename,
            file_type=document.file_type,
            file_size=document.file_size,
            status=document.status,
            minio_path=document.minio_path,
            uploaded_at=document.uploaded_at,
        )
    except DocumentError as e:
        raise _document_error_to_http(e)
    
#________ GET my documents (User) _____________
@router.get(
    "/owned",
    response_model=list[DocumentResponse],
    summary="Get all my documents (user ONLY)",
)
async def get_my_documents(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        service = DocumentService(db)
        documents = await service.get_my_documents(current_user)
        return[
            DocumentResponse(
                message="",
                id=doc.id,
                filename=doc.filename,
                original_filename=doc.original_filename,
                file_type=doc.file_type,
                file_size=doc.file_size,
                status=doc.status,
                minio_path=doc.minio_path,
                uploaded_at=doc.uploaded_at,
            )
            for doc in documents    
        ]  
    except DocumentError as e:
        raise _document_error_to_http(e)
    
#________ GET all documents (admin) ________
@router.get(
    "/",
    response_model=list[DocumentResponse],
    summary="Get all documents (admin ONLY)",
)
async def get_all_documents(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    
    try:
        service = DocumentService(db)
        documents = await service.get_all_documents(current_user)
        return [
            DocumentResponse(
                message="",
                id=doc.id,
                filename=doc.filename,
                original_filename=doc.original_filename,
                file_type=doc.file_type,
                file_size=doc.file_size,
                status=doc.status,
                minio_path=doc.minio_path,
                uploaded_at=doc.uploaded_at,
            )
            for doc in documents
        ]
    except DocumentError as e:
        raise _document_error_to_http(e)
    
#_______ GET document by ID (admin) ____________
@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Get a specific document (admin ONLY)"
) 
async def get_document_by_id(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        service = DocumentService(db)
        doc = await service.get_document_by_id(document_id, current_user)
        return DocumentResponse(
            message="",
            id=doc.id,
            filename=doc.filename,
            original_filename=doc.original_filename,
            file_type=doc.file_type,
            file_size=doc.file_size,
            status=doc.status,
            minio_path=doc.minio_path,
            uploaded_at=doc.uploaded_at,
        )
    except DocumentError as e:
        raise _document_error_to_http(e)     

#________ DELETE document (admin) _________
@router.delete(
    "/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a document (admin ONLY)",
)
async def delete_document(
    document_id: uuid.UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    try:
        service = DocumentService(db)
        await service.delete_document(document_id, current_user)
    except DocumentError as e:
        raise _document_error_to_http(e)
