"""
app/services/admin_service.py – Admin business logic
"""
from datetime import datetime, timezone
import uuid

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.security import hash_password
from app.schemas.auth import AdminCreateUserRequest, UpdateUserRequest

from app.models.user import User
from app.services.email import (
    send_account_approved_email,
    send_account_rejected_email,
)
from app.models.document import Document
from app.schemas.dashboard import DashboardStats
from app.utils.logging import get_logger

logger = get_logger(None).bind(stage="admin_service")

class AdminError(Exception):
    """Domain-level admin error — converted to HTTP response in the router."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)
        
class AdminService:
    def __init__(self, db: AsyncSession):
        self.db = db
    #_________ List pending users __________
    async def get_pending_users(self) -> list[User]:
        """Return all users waiting for admin approval"""
        result = await self.db.execute(
            select(User).where(
                User.account_enabled == False,
                User.is_deleted == False
            )
        )
        return result.scalars().all()
    
    #__________ Enable user ______________
    async def enable_user(self, user_id: uuid.UUID) -> User:
        """Approuve a user account and notify them by eamil"""
        
        #find the user in DB
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        # Check if user exists
        if user is None:
            raise AdminError("User not found", 404)
        # Check if already enabled
        if user.account_enabled:
            raise AdminError("Account is already enabled", 400)
        # Approve the account 
        user.account_enabled = True
        await self.db.flush()
        
        # Notify the user by email
        await send_account_approved_email(
            user_email=user.email,
            user_full_name=user.full_name,
        )

        logger.info("account_approved", user_email=user.email)
        return user
    #______ disable user __________________
    async def disable_user(self, user_id: uuid.UUID) -> User:
        """Disable a user account and notify them by email"""
      
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        if user is None:
            raise AdminError("User not found", 404)

        # Check if already disabled
        if not user.account_enabled:
            raise AdminError("Account is already disabled", 400)
        
        # Disable the account
        user.account_enabled = False
        await self.db.flush()

        await send_account_rejected_email(
            user_email=user.email,
            user_full_name=user.full_name,
        )

        logger.info("account_disabled", user_email=user.email)
        return user
    #________ Get all users __________
    
    async def get_all_users(self) -> list[User]:
        """Return all active users and admins (not deleted)"""
        result = await self.db.execute(select(User).where(User.is_deleted == False))
        return result.scalars().all()
    
    #_____ Create user _______________
    async def create_user(self, data: AdminCreateUserRequest) -> User:
        """Creates a user directly activated by the admin"""
        
        #check if the email already exist
        existing = await self.db.execute(
            select(User).where(User.email == data.email.lower())
        )
        if existing.scalar_one_or_none():
           raise AdminError("Email already registred", 409)
        
        #create user 
        user = User(
            id=uuid.uuid4(),
            email=data.email.lower(),
            hashed_password=hash_password(data.password),
            full_name=data.full_name.strip(),
            role="user",                  # always user
            office_address=data.office_address,
            phone_nbr=data.phone_nbr,
            birth_date=data.birth_date,
            is_active=True,
            is_verified=True,             # directly verified
            account_enabled=True,         # directly enabled
        ) 
        self.db.add(user)
        await self.db.flush()
        
        logger.info("user_created_by_admin", user_email=user.email)
        return user
    #_______ Delete user _____________
    async def delete_user(self, user_id: uuid.UUID) -> User:
        """
        Soft delete of a user (not an admin), marked as deleted without being removed from the DB
        """
        
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        #check if the user exists
        if user is None:
            raise AdminError("User not found", 404)
        
        # Check that it is not an admin
        if user.role == "admin":   
            raise AdminError("Cannot delete an admin account", 403)
        
        #verify if user is already deleted
        if user.is_deleted:
            raise AdminError("User is already deleted", 400)
        
        user.is_deleted = True
        user.deleted_at = datetime.now(timezone.utc)
        
        await self.db.flush()
        
        logger.info("user_deleted_by_admin", user_email=user.email)
        return user
    #__________ update user ____________
    async def update_user(self, user_id: uuid.UUID, data: UpdateUserRequest) -> User:
        """Updates a user's information(not an admin)"""
        result = await self.db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()

        # Check if the user exists
        if user is None:
            raise AdminError("User not found", 404)
        # Check that it is not an admin
        if user.role == "admin":
            raise AdminError("Cannot modify an admin account", 403)
        
        # Update only provided fields (not None)
        if data.full_name is not None:
            user.full_name = data.full_name
        if data.email is not None:
            # Check that the new email is not already taken
            existing = await self.db.execute(
                select(User).where(User.email == data.email.lower())
            )
            if existing.scalar_one_or_none():
                raise AdminError("Email already registered", 409)
            user.email = data.email.lower()
        if data.office_address is not None:
            user.office_address = data.office_address
        if data.phone_nbr is not None:
            user.phone_nbr = data.phone_nbr
        if data.is_active is not None:
            user.is_active = data.is_active
        if data.account_enabled is not None:
            user.account_enabled = data.account_enabled
            
        await self.db.flush()
        
        logger.info("user_updated_by_admin", user_email=user.email)
        return user
    
    #__________ Get dashboard stats ____________
    async def get_dashboard_stats(self) -> DashboardStats:
        """
        Calculate and return dashboard statistics from the entire database
        """
        #______ documents processed _________
        docs_result = await self.db.execute(
            select(func.count(Document.id)).where(
                Document.is_deleted == False
            )
        )
        documents_processed = docs_result.scalar() or 0
        
        #______ extraction success rate _________
        processed_docs_result = await self.db.execute(
            select(func.count(Document.id)).where(
                Document.is_deleted == False,
                Document.confidence_score .isnot(None)
            )
        )
        processed_docs_count = processed_docs_result.scalar() or 0
        
        if processed_docs_count > 0:
            #Count documents with confidence_score >= 0.50
            success_docs_result = await self.db.execute(
                select(func.count(Document.id)).where(
                    Document.is_deleted == False,
                    Document.confidence_score >= 0.50,
                    Document.confidence_score.isnot(None)
                )
            )
            success_docs_count = success_docs_result.scalar() or 0

            # Calculate percentage
            extraction_success_rate = round(
                (success_docs_count / processed_docs_count) * 100, 2
            )
        else:
            extraction_success_rate = 0.0
        
        #______ low confidence documents _________
        #Number of documents with confidence_score < 0.50
        low_conf_result = await self.db.execute(
            select(func.count(Document.id)).where(
                Document.is_deleted == False,
                Document.confidence_score < 0.50,
                Document.confidence_score.isnot(None)
            )
        )
        low_confidence_documents = low_conf_result.scalar() or 0
        
        #______ most processed document type _________
        #Most frequent doc_type detected by the pipeline
        doc_type_result = await self.db.execute(
        select(Document.doc_type, func.count(Document.id).label("count"))
            .where(
                Document.is_deleted == False,    
                Document.doc_type.isnot(None)    # only documents with detected type
            ).group_by(Document.doc_type)
            # group by doc_type → count per type
            .order_by(func.count(Document.id).desc())
            # order by count descending → most frequent first
            .limit(1)
            # take only the first → most frequent
        )
        doc_type_row = doc_type_result.first()
        most_processed_doc_type = doc_type_row[0] if doc_type_row else None
        # doc_type_row[0] → the doc_type value
        
        logger.info("dashboard_stats_calculated")
        
        return DashboardStats(
            documents_processed=documents_processed,
            extraction_success_rate=extraction_success_rate,
            low_confidence_documents=low_confidence_documents,
            most_processed_doc_type=most_processed_doc_type
        )
        
        
