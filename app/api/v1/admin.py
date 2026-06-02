"""
app/api/v1/admin.py - Admin endpoints
Only users with role "admin" can access these endpoints

GET    /admin/users            - list all users and admins
GET  /admin/users/pending      - list users waiting for approval
POST   /admin/users            - create a user directly activated
POST /admin/users/{id}/enable  - approve a user account
POST /admin/users/{id}/disable - disable a user account
DELETE /admin/users/{id}       - permanently delete a user
PATCH  /admin/users/{id}       - update a user's information
GET    /admin/dashboard        - get dashboard statistics
"""

import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, get_db
from app.schemas.auth import (MessageResponse,
                            AdminCreateUserRequest,
                            UpdateUserRequest,
                            UserListResponse,
                            RegisterResponse,
)                           
from app.services.admin_service import AdminError, AdminService
from app.schemas.dashboard import DashboardStats
from app.utils.logging import get_logger

router = APIRouter()
logger = get_logger(None).bind(stage="api")

def _admin_error_to_http(e: AdminError) -> HTTPException:
    """Convert AdminError to HTTPException — same pattern as auth.py."""
    return HTTPException(status_code=e.status_code, detail=e.message)
def _require_admin(current_user: CurrentUser) -> None:
    """Check that the current has admin role"""
    if current_user.role != "admin":
        raise AdminError("Admin access required", 403)

#______ Get all users _________________   
@router.get(
    "/users",
    response_model=list[UserListResponse],
    summary="List all users and admins",
)
async def get_all_users(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Returns all users and admins - admin only"""
    try:
        _require_admin(current_user)
        service = AdminService(db)
        users = await service.get_all_users()
    except AdminError as e:
        raise _admin_error_to_http(e)

    return [UserListResponse.model_validate(user) for user in users]

#______ list pending users _____________
@router.get(
    "/users/pending",
    summary="List users waiting for admin approval",
)
async def list_pending_users(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Returns all users with account_enabled = False"""
    try:
        _require_admin(current_user)
        service = AdminService(db)
        users = await service.get_pending_users()
    except AdminError as e:
        raise _admin_error_to_http(e)

    return [
        {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "role": user.role,
            "created_at": user.created_at,
            "office_address": user.office_address,
            "phone_nbr": user.phone_nbr,
        }
        for user in users
    ]

#__________ Create user ________________
@router.post(
    "/users",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a user account (admin only)",
)
async def create_user(
    data: AdminCreateUserRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Creates a user directly activated — admin only"""
   
    try:
        _require_admin(current_user)
        service = AdminService(db)
        user = await service.create_user(data)
        await db.commit()
    except AdminError as e:
        await db.rollback()
        raise _admin_error_to_http(e)

    return RegisterResponse.model_validate(user)

#__________ Enable user _____________
@router.post(
    "/users/{user_id}/enable",
    response_model=MessageResponse,
    summary="Approve a user account",
)
async def enable_user(
    user_id: uuid.UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Set account_enabled = True and notify the user by email"""
    try:
        _require_admin(current_user)
        service = AdminService(db)
        user = await service.enable_user(user_id)
        await db.commit()
    except AdminError as e:
        await db.rollback()
        raise _admin_error_to_http(e)

    return MessageResponse(message=f"Account {user.email} has been approved successfully")
#______ Disable user _______________
@router.post(
    "/users/{user_id}/disable",
    response_model=MessageResponse,
    summary="Disable a user account",
)
async def disable_user(
    user_id: uuid.UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Set account_enabled = False and notify the user by email."""
    try:
        _require_admin(current_user)
        service = AdminService(db)
        user = await service.disable_user(user_id)
        await db.commit()
    except AdminError as e:
        await db.rollback()
        raise _admin_error_to_http(e)

    return MessageResponse(message=f"Account {user.email} has been disabled successfully")

#________ Delete user _______________
@router.delete(
    "/users/{user_id}",
    response_model=MessageResponse,
    summary="Delete a user account (admin only)",
)
async def delete_user(
    user_id: uuid.UUID,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """"Permanently delete a user - admin only"""
    try:
        _require_admin(current_user)
        service = AdminService(db)
        user = await service.delete_user(user_id)
        await db.commit()
    except AdminError as e:
        await db.rollback()
        raise _admin_error_to_http(e)

    return MessageResponse(message=f"User {user.email} deleted successfully")
    
#________ Update user __________
@router.patch(
    "/users/{user_id}",
    response_model=UserListResponse,
    summary="Update a user account (admin only)",
)
async def update_user(
    user_id: uuid.UUID,
    data: UpdateUserRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """update a user's information - admin only """
    try:
        _require_admin(current_user)
        service = AdminService(db)
        user = await service.update_user(user_id, data)
        await db.commit()
    except AdminError as e:
        await db.rollback()
        raise _admin_error_to_http(e)

    return UserListResponse.model_validate(user)

#________ Get dashboard stats __________
@router.get(
    "/dashboard",
    response_model=DashboardStats,
    summary="Get statistics for the admin dashboard(admin only)",
)
async def get_dashboard_stats(
   current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """ 
    Returns dashboard statistics calculated from the entire database :
    - documents_processed    → total number of uploaded documents
    - extraction_success_rate → % of documents with confidence >= 50%
    - low_confidence_documents → number of documents with confidence < 50%
    - most_processed_doc_type → most frequent document type
    """ 
    try:
        _require_admin(current_user)
        service = AdminService(db)
        stats = await service.get_dashboard_stats()
    except AdminError as e:
        raise _admin_error_to_http(e)
    
    return stats
