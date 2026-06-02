"""
app/api/v1/auth.py – Auth endpoints

POST /auth/register       Create account
POST /auth/token          Login → access + refresh token
POST /auth/refresh        Rotate access token using refresh token
POST /auth/logout         Revoke refresh token
POST /auth/logout-all     Revoke all refresh tokens (all devices)
PATCH /auth/update-profile      Update current user profile (user and admin)
GET  /auth/me             Get current user profile
POST /auth/forgot-password          Send password reset email
POST /auth/reset-password           Reset password using token
GET  /auth/reset-password-form      Display reset password HTML form
GET  /auth/verify-email             Verify admin email address
"""
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import CurrentUser, get_db
from app.schemas.auth import (
    UpdateProfileRequest,
    ForgotPasswordRequest,
    LoginRequest,
    LogoutRequest,
    MessageResponse,
    RefreshRequest,
    RefreshResponse,
    RegisterRequest,
    RegisterResponse,
    ResetPasswordRequest,
    TokenResponse,
    UserInfo,
)
from app.services.auth_service import AuthError, AuthService
from app.utils.logging import get_logger
from app.models.email_verification_token import EmailVerificationToken
from app.core.security import hash_token
from app.config import get_settings
from sqlalchemy import select
from datetime import datetime, timezone

settings = get_settings()

router = APIRouter()
logger = get_logger(None).bind(stage="api")


def _get_client_info(request: Request) -> tuple[str | None, str | None]:
    """Extract User-Agent and real IP (handles reverse proxy headers)."""
    user_agent = request.headers.get("user-agent")
    ip = (
        request.headers.get("x-forwarded-for", "").split(",")[0].strip()
        or request.headers.get("x-real-ip")
        or (request.client.host if request.client else None)
    )
    return user_agent, ip


def _auth_error_to_http(e: AuthError) -> HTTPException:
    return HTTPException(status_code=e.status_code, detail=e.message)


# ── Register ──────────────────────────────────────────────────────────────────

@router.post(
    "/register",
    response_model=RegisterResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new user account",
)
async def register(
    data: RegisterRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Register a new user. Returns the created user profile.
    Password rules: 8+ chars, uppercase, lowercase, digit, special char.
    """
    try:
        service = AuthService(db)
        user = await service.register(data, request)
        await db.commit()
        await db.refresh(user)
        return RegisterResponse.model_validate(user)
    except AuthError as e:
        await db.rollback()
        raise _auth_error_to_http(e)


# ── Login / Token ─────────────────────────────────────────────────────────────

@router.post(
    "/token",
    response_model=TokenResponse,
    summary="Login — returns access + refresh token",
)
async def login(
    data: LoginRequest,
    request: Request,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Authenticate with email + password.
    Returns a short-lived access token (JWT) and a long-lived refresh token.

    - Access token: put in `Authorization: Bearer <token>` header.
    - Refresh token: store securely (httpOnly cookie recommended for web apps).
    - After `MAX_FAILED_ATTEMPTS` (5) wrong passwords the account is locked for 15 min.
    """
    user_agent, ip = _get_client_info(request)

    try:
        service = AuthService(db)
        access_token, refresh_token, expires_at = await service.login(
            login=data.login,
            password=data.password,
            user_agent=user_agent,
            ip_address=ip,
        )
        await db.commit()
    except AuthError as e:
        await db.rollback()
        raise _auth_error_to_http(e)

    # We need the user object for the response — re-fetch via service
    from app.core.security import decode_access_token
    from app.models.user import User
    from sqlalchemy import select
    import uuid

    payload = decode_access_token(access_token)
    result = await db.execute(select(User).where(User.id == uuid.UUID(payload["sub"])))
    user = result.scalar_one()

    from app.config import get_settings
    settings = get_settings()

    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
        user=UserInfo.model_validate(user),
    )


# ── Refresh ───────────────────────────────────────────────────────────────────

@router.post(
    "/refresh",
    response_model=RefreshResponse,
    summary="Get a new access token using a refresh token",
)
async def refresh(
    data: RefreshRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Exchange a valid (non-expired, non-revoked) refresh token for a new access token.
    The refresh token itself is NOT rotated here — use /logout + /token to rotate it.
    """
    try:
        service = AuthService(db)
        access_token, expires_at = await service.refresh(data.refresh_token)
    except AuthError as e:
        raise _auth_error_to_http(e)

    from app.config import get_settings
    settings = get_settings()

    return RefreshResponse(
        access_token=access_token,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ── Logout ────────────────────────────────────────────────────────────────────

@router.post(
    "/logout",
    response_model=MessageResponse,
    summary="Revoke a refresh token (logout current device)",
)
async def logout(
    data: LogoutRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Revoke the provided refresh token. Idempotent — safe to call multiple times."""
    service = AuthService(db)
    await service.logout(data.refresh_token)
    await db.commit()
    return MessageResponse(message="Logged out successfully")


@router.post(
    "/logout-all",
    response_model=MessageResponse,
    summary="Revoke all refresh tokens (logout all devices)",
)
async def logout_all(
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """Requires a valid access token. Revokes ALL refresh tokens for this account."""
    service = AuthService(db)
    await service.logout_all(current_user.id)
    await db.commit()
    return MessageResponse(message="Logged out from all devices")

#_____ Update profile ___________________
@router.patch(
    "/update-profile",
    response_model=UserInfo,
    summary="Update current user profile (user and admin)",
)
async def update_profile(
    data: UpdateProfileRequest,
    current_user: CurrentUser,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Update profile for user and admin :
    - email → optional
    - phone_nbr → optional
    - new_password → requires current_password
    """
    try:
        service = AuthService(db)
        user = await service.update_profile(
            user=current_user,
            data=data,
        )
        await db.commit()
        return UserInfo.model_validate(user)
    except AuthError as e:
        await db.rollback()
        raise _auth_error_to_http(e)

# ── Me ────────────────────────────────────────────────────────────────────────

@router.get(
    "/me",
    response_model=UserInfo,
    summary="Get current user profile",
)
async def me(current_user: CurrentUser):
    """Returns the profile of the currently authenticated user."""
    return UserInfo.model_validate(current_user)

# _________ Forgot PWD _______________
@router.post(
    "/forgot-password",
    response_model=MessageResponse,
    summary="Send a password reset email",
)
async def forgot_password(
    data: ForgotPasswordRequest,
    request: Request, #reset URL
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Send a password reset email to the user.
    Always returns success to avoid email enumeration attacks.
    """
    try:
        service = AuthService(db)
        await service.forgot_password(email=data.email, request=request)
        await db.commit()
    except AuthError as e:
        await db.rollback()
        raise _auth_error_to_http(e)

    return MessageResponse(message="If this email exists, a reset link has been sent.")

#____ Reset PWD ___________________
@router.post(
    "/reset-password",
    response_model=MessageResponse,
    summary="Reset password using token",
)
async def reset_password(
    data: ResetPasswordRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Reset the user password using the token received by email.
    Token is valid for 24 hours and can only be used once.
    """
    try:
        service = AuthService(db)
        await service.reset_password(
            token=data.token,
            new_password=data.new_password,
        )
        await db.commit()
    except AuthError as e:
        await db.rollback()
        raise _auth_error_to_http(e)

    return MessageResponse(message="Password reset successfully. Please log in again.")

#______ Reset pwd Form ____________
templates = Jinja2Templates(directory="app/templates")
@router.get(
    "/reset-password-form",
    response_class=HTMLResponse,
    summary="Display the reset password form",
)
async def reset_password_form(request: Request, token: str):
    """
    Display the HTML form for resetting the password.
    The token is passed as a query parameter from the email link.
    """
    return templates.TemplateResponse(
        "reset_password.html",
        {"request": request, "token": token}
    )
#______ verify email _____________
@router.get(
    "/verify-email",
    response_class=HTMLResponse,
    summary="Verify admin email address",
)
async def verify_email(
    request: Request,
    token: str,
    db: Annotated[AsyncSession, Depends(get_db)],
):
    """
    Verify the admin email address using the token received by email.
    Token is valid for 24 hours and can only be used once.
    """
    from app.models.user import User
    
    #find the token in the database
    token_hash = hash_token(token)
    result = await db.execute(
        select(EmailVerificationToken).where(
            EmailVerificationToken.token_hash == token_hash
        )
    )
    stored = result.scalar_one_or_none()
    # Check if token exists
    if stored is None:
        return templates.TemplateResponse(
            "verify_email.html",
            {
                "request": request,
                "success": False,
                "error_message": "Invalid verification token.",
                "frontend_url": settings.FRONTEND_URL,
            }
        )
    # Check if token already used
    if stored.is_used:
        return templates.TemplateResponse(
            "verify_email.html",
            {
                "request": request,
                "success": False,
                "error_message": "This verification link has already been used.",
                "frontend_url": settings.FRONTEND_URL,
            }
        )
    # Check if token expired
    if stored.expires_at < datetime.now(timezone.utc):
        return templates.TemplateResponse(
            "verify_email.html",
            {
                "request": request,
                "success": False,
                "error_message": "This verification link has expired.",
                "frontend_url": settings.FRONTEND_URL,
            }
        )
    #find the user   
    user_result = await db.execute(
        select(User).where(User.id == stored.user_id)
    )
    user = user_result.scalar_one_or_none()

    if user is None:
        return templates.TemplateResponse(
            "verify_email.html",
            {
                "request": request,
                "success": False,
                "error_message": "User not found.",
                "frontend_url": settings.FRONTEND_URL,
            }
        )  
    # Mark email as verified
    user.is_verified = True
    stored.is_used = True
    await db.flush()
    await db.commit()

    logger.info("Email verified successfully for: %s", user.email)

    return templates.TemplateResponse(
        "verify_email.html",
        {
            "request": request,
            "success": True,
            "error_message": None,
            "frontend_url": settings.FRONTEND_URL,
        }
    )    
    
