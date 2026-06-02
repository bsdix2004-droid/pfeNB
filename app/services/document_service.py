"""
app/services/document_service.py – Upload logic : validation + MinIO storage + DB save
"""
from __future__ import annotations
import uuid, fitz
from datetime import datetime, timezone
import boto3 # = the sender that uploads files to MinIO
from botocore.exceptions import BotoCoreError, ClientError
from fastapi import UploadFile, status
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import func, select

from app.config import get_settings
from app.constants.document_constants  import ALLOWED_EXTENSIONS, MAX_FILE_SIZE, MAGIC_BYTES
from app.models.document import Document
from app.models.user import User
from app.utils.logging import get_logger

settings = get_settings()
logger = get_logger(None).bind(stage="document_service")


class DocumentError(Exception):
    """Domain-level document error — converted to HTTP response in the router."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)

#________ MinIO client ____________
def get_s3_client():
    """create a boto3 client connected to MinIO"""
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        region_name=settings.S3_REGION,
    )
#_________ Validation _______________
def validate_file(file: UploadFile, content: bytes) -> str:
    """ 
    Validate the uploaded file:
    1 Allowed extension
    2 Non-empty file
    3 Actual file content matches the extension (magic bytes)
    4 Maximum size limit
    return the extension if valid, otherwise raise a DocumentError  
    """
    original_name = file.filename or ""
    ext = "." + original_name.rsplit(".", 1)[-1].lower() if "." in original_name else ""

    if ext not in ALLOWED_EXTENSIONS:
        raise DocumentError(
            f"Unauthorized extension. Allowed extensions: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    if len(content) == 0:
        raise DocumentError("File is empty")
    
    file_header = content[:8]
    valid_signatures = MAGIC_BYTES.get(ext, [])
    is_valid_content = any(
        file_header.startswith(signature)
        for signature in valid_signatures
    )
    
    if not is_valid_content:
        raise DocumentError(
            f"The file content does not match the '{ext}' extension. "
            f"The file appears to be corrupted or incorrectly renamed."
        )
        
    if len(content) > MAX_FILE_SIZE:
        raise DocumentError("File too large. Maximum size: 5 MB")

    return ext

def _count_pages(content: bytes, ext: str) -> int:
    """
    Count the nbr of pages in the document:
    PDF -> use PyMuPDF to count pages
    Images -> always 1 page
    """
    if ext == ".pdf":
        try:
            doc = fitz.open(stream=content, filetype="pdf")
            return doc.page_count
        except Exception as e:
            logger.warning("count_pages_failed", error=str(e))
            return 1  # fallback to 1 page if counting fails
    return 1  # for images

class DocumentService:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def upload_document(
        self,
        file: UploadFile,
        current_user: User,
    ) -> Document:
        """ 
        1 Read the file
        2 Validate the extension and size
        3 Upload it to MinIO
        4 Save the metadata in the database
        5 Return the created Document
        """
        #______1 read file content ______________
        content = await file.read()
        
        #_________2 validate ________________
        ext = validate_file(file, content)
        
        #__________3 Generate a unique MinIO filename _________
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        unique_filename = f"{uuid.uuid4().hex}_{timestamp}{ext}"
        
        date_folder = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        minio_path = f"uploads/{date_folder}/{unique_filename}"
        
        #________4 Upload to MinIO ____________
        try:
            s3 = get_s3_client()
            s3.put_object(
                Bucket=settings.S3_BUCKET_UPLOADS, #name of MinIO bucket
                Key=minio_path,
                Body=content,
                ContentType=file.content_type or "application/octet-stream",
            )
        except (BotoCoreError, ClientError) as e:
            raise DocumentError(
                f"Error during storage upload: {str(e)}",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )

        #________ 5 Count pages _____________
        pages = _count_pages(content, ext)
        
        #________ 6 Calculate doc_nmbr _________
        #Count the number of documents already uploaded by this user
        result = await self.db.execute(
            select(func.count(Document.id)).where(Document.user_id == current_user.id)
        )
        doc_count = result.scalar() or 0
        doc_number = doc_count + 1  # Increment the count for the new document
        #________ 7 Save in DataBase __________
        document = Document(
            user_id=current_user.id,
            filename=unique_filename,
            original_filename=file.filename or unique_filename,
            file_type=ext.lstrip("."),
            file_size=len(content),
            minio_path=minio_path,
            status="uploaded",
            pages=pages,
            doc_number=doc_number,
        )

        self.db.add(document)
        await self.db.commit()
        await self.db.refresh(document)

        logger.info("document_uploaded", user_email=current_user.email, document_id=str(document.id))

        return document
    
    async def get_my_documents(self, current_user: User) -> list[Document]:
        """Return all documents uploaded by the current user"""
        if current_user.role != "user":
            raise DocumentError("Only users can access this endpoint", status_code=403)
        result = await self.db.execute(
            select(Document).where(Document.user_id == current_user.id)
            .where(Document.is_deleted == False)
        )
        return result.scalars().all()
    
    async def get_all_documents(self, current_user: User) ->list[Document]:
        """Admin  return all documents from all users"""
        if current_user.role != "admin":
            raise DocumentError("Only admins can access this endpoint", status_code=403)
        
        result = await self.db.execute(select(Document).where(Document.is_deleted == False))
        return result.scalars().all()
    
    async def get_document_by_id(self, document_id: uuid.UUID, current_user: User) -> Document:
        """Admin — return a specific document by ID"""
        if current_user.role != "admin":
            raise DocumentError("Only admins can access this endpoint", status_code=403) 
        result = await self.db.execute(
            select(Document).where(Document.id == document_id)
        )
        document = result.scalar_one_or_none()
        if not document:
            raise DocumentError("Document not found", status_code=404)
        return document
    
    async def delete_document(self, document_id: uuid.UUID, current_user: User) -> None: 
        """Admin — delete a document from DB and MinIO."""
        document = await self.get_document_by_id(document_id, current_user)
        
        #verify if document is already deleted 
        if document.is_deleted:
            raise DocumentError("Document is already deleted", 400)
        
        # Delete from MinIO
        try: 
            s3 = get_s3_client()
            s3.delete_object(
                Bucket=settings.S3_BUCKET_UPLOADS,
                Key=document.minio_path,
            )
        except (BotoCoreError, ClientError) as e:
            raise DocumentError(
                f"Error during storage deletion: {str(e)}",
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        # Delete from DB
        document.is_deleted = True
        document.deleted_at = datetime.now(timezone.utc)
        document.deleted_by = current_user.id
        #The document remains in the db but is marked as deleted
        await self.db.commit()
        logger.info("document_deleted", document_id=str(document_id), admin_email=current_user.email)
