"""
app/services/auth_service.py – Auth business logic (no HTTP concerns here)
"""
# this is the authentication "brain": it contains all business logic — no HTTP, no direct DB handling

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession # = non-blocking database session

from app.config import get_settings
from app.core.security import (
    create_access_token,
    generate_refresh_token,
    hash_password,
    hash_token,
    verify_password,
) #import security functions defined in "security.py"
from app.models.refresh_token import RefreshToken
from app.models.reset_token import PasswordResetToken
from app.models.email_verification_token import EmailVerificationToken
from app.models.user import User
from app.schemas.auth import RegisterRequest, UpdateProfileRequest
from app.services.email import (
    send_reset_password_email,
    send_admin_new_user_notification,
    send_email_verification,
)
                                
from fastapi import Request
#import database models and the registration schema
from app.utils.logging import get_logger

settings = get_settings()
logger = get_logger(None).bind(stage="auth_service")

# Lock account after this many failed attempts
#after 5 wrong passwords → account is locked for 15 minutes
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_MINUTES = 15


class AuthError(Exception):
    """Domain-level auth error — converted to HTTP response in the router."""
    def __init__(self, message: str, status_code: int = 400):
        self.message = message
        self.status_code = status_code
        super().__init__(message)


class AuthService:

    def __init__(self, db: AsyncSession):
        self.db = db

    # ── Register ──────────────────────────────────────────────────────────────

    async def register(self, data: RegisterRequest, request: Request) -> User:
        # Check email uniqueness
        existing = await self.db.execute(
            select(User).where(User.email == data.email.lower())
        )
        if existing.scalar_one_or_none():
            raise AuthError("Email already registered", 409)
        is_admin = data.role.value == "admin"
        user = User(
            id=uuid.uuid4(),
            email=data.email.lower(),
            hashed_password=hash_password(data.password),
            full_name=data.full_name.strip(),
            role=data.role.value, #"admin" or "user" from RoleEnum
            office_address =data.office_address,
            phone_nbr=data.phone_nbr,
            birth_date=data.birth_date,
            is_active=True,
            is_verified=False,   # Will be set to True after email verification
            account_enabled=is_admin,  # True for admin, False for user
        )
        self.db.add(user)
        await self.db.flush()   # get the ID without committing

        if is_admin:
            #generate email verification token for admin
            raw_token = generate_refresh_token()
            token_hash = hash_token(raw_token)
            expires_at = datetime.now(timezone.utc) + timedelta(hours=24)
            
            verification_token = EmailVerificationToken(
                id=uuid.uuid4(),
                user_id=user.id,
                token_hash=token_hash,
                expires_at=expires_at,
            )
            self.db.add(verification_token)
            await self.db.flush()
            
            #build verification URL 
            base_url = str(request.base_url).rstrip("/")
            verify_url = f"{base_url}/api/v1/auth/verify-email?token={raw_token}"
            #send verification email to admin
            await send_email_verification(
                email_to=user.email,
                verify_url=verify_url,
            )
            logger.info("verification_email_sent", user_email=user.email) 
        else: 
            #notify admin about new user registration
            await send_admin_new_user_notification(
                admin_email=settings.SMTP_USER,
                user_email=user.email,
                user_full_name=user.full_name,
            )
            logger.info("user_registered", user_email=user.email)
        return user

    # ── Login ─────────────────────────────────────────────────────────────────

    async def login(
        self,
        login: str,
        password: str,
        user_agent: str | None = None,
        ip_address: str | None = None,
    ) -> tuple[str, str, datetime]:
        """
        Returns (access_token, refresh_token, access_token_expires_at).
        Raises AuthError on failure.
        """
        user = await self._get_user_by_login(login)

        if user is None:
            # Constant-time fake verify to prevent user enumeration via timing
            hash_password("fake-constant-time-check")
            raise AuthError("Invalid credentials", 401)
        
        if user.is_deleted:
            raise AuthError("This account has been deleted", 403)

        if not user.is_active:
            raise AuthError("Account is disabled", 403)
        if user.role == "admin" and not user.is_verified:
            raise AuthError("Please verify your email before logging in", 403)
        if user.role == "user" and not user.account_enabled:
            raise AuthError("Account is pending admin approval",)
        if user.is_locked:
            raise AuthError(
                f"Account locked. Try again after {user.locked_until.strftime('%H:%M UTC')}",
                403,
            )

        if not verify_password(password, user.hashed_password):
            await self._record_failed_attempt(user)
            raise AuthError("Invalid credentials", 401)

        # Successful login — reset failure counter
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login_at = datetime.now(timezone.utc)

        access_token, expires_at = create_access_token(
            subject=str(user.id),
            role=user.role,
        )
        refresh_token = await self._create_refresh_token(
            user=user,
            user_agent=user_agent,
            ip_address=ip_address,
        )

        await self.db.flush()
        return access_token, refresh_token, expires_at

    # ── Refresh ───────────────────────────────────────────────────────────────

    async def refresh(self, raw_token: str) -> tuple[str, datetime]:
        """
        Validate a refresh token and issue a new access token.
        Returns (new_access_token, expires_at).
        """
        token_hash = hash_token(raw_token)

        result = await self.db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        stored = result.scalar_one_or_none()

        if stored is None or stored.is_revoked:
            raise AuthError("Invalid or revoked refresh token", 401)

        if stored.expires_at < datetime.now(timezone.utc):
            raise AuthError("Refresh token expired", 401)

        # Load the user
        user_result = await self.db.execute(
            select(User).where(User.id == stored.user_id)
        )
        user = user_result.scalar_one_or_none()

        if user is None or not user.is_active:
            raise AuthError("User not found or disabled", 401)

        if user.is_locked:
            raise AuthError("Account is locked", 403)

        access_token, expires_at = create_access_token(
            subject=str(user.id),
            role=user.role,
        )
        await self.db.flush()
        return access_token, expires_at

    # ── Logout ────────────────────────────────────────────────────────────────

    async def logout(self, raw_token: str) -> None:
        """Revoke a single refresh token."""
        token_hash = hash_token(raw_token)

        result = await self.db.execute(
            select(RefreshToken).where(RefreshToken.token_hash == token_hash)
        )
        stored = result.scalar_one_or_none()

        if stored and not stored.is_revoked:
            stored.is_revoked = True
            await self.db.flush()

    async def logout_all(self, user_id: uuid.UUID) -> None:
        """Revoke ALL refresh tokens for a user (sign out all devices)."""
        result = await self.db.execute(
            select(RefreshToken).where(
                RefreshToken.user_id == user_id,
                RefreshToken.is_revoked == False,  # noqa: E712
            )
        )
        for token in result.scalars().all():
            token.is_revoked = True
        await self.db.flush()
    
    #_____ Forgot PWD _____________________
    async def forgot_password(self, email: str, request: Request) -> None:
        """Generate a reset token and send it by email."""
        # Find user by email
        result = await self.db.execute(
            select(User).where(User.email == email.lower())
        )
        user = result.scalar_one_or_none()

        #always return success to avoid email enumeration
        if user is None:
            return

        #generate a raw token and hash it for storage
        raw_token = generate_refresh_token()
        token_hash = hash_token(raw_token)

        #save the token in the database
        expires_at = datetime.now(timezone.utc) + timedelta(
            hours=settings.PASSWORD_RESET_TOKEN_EXPIRE_HOURS
        )
        reset_token = PasswordResetToken(
            id=uuid.uuid4(),
            user_id=user.id,
            token_hash=token_hash,
            expires_at=expires_at,
        )
        self.db.add(reset_token)
        await self.db.flush()

        #build the reset URL
        base_url = str(request.base_url).rstrip("/")
        reset_url = f"{base_url}/api/v1/auth/reset-password-form?token={raw_token}"

        #send the email
        await send_reset_password_email(email_to=user.email, reset_url=reset_url)
        logger.info("password_reset_email_sent", user_email=user.email)
        
    # ── Reset Password ────────────────────────────────────────────────────────

    async def reset_password(self, token: str, new_password: str) -> None:
        """Verify the reset token and update the password."""

        token_hash = hash_token(token)

        #find the token in the database
        result = await self.db.execute(
            select(PasswordResetToken).where(
                PasswordResetToken.token_hash == token_hash
            )
        )
        stored = result.scalar_one_or_none()

        #check if token exists
        if stored is None:
            raise AuthError("Invalid reset token", 400)

        #check if token is already used
        if stored.is_used:
            raise AuthError("Reset token has already been used", 400)

        #check if token is expired
        if stored.expires_at < datetime.now(timezone.utc):
            raise AuthError("Reset token has expired", 400)

        #find the user
        user_result = await self.db.execute(
            select(User).where(User.id == stored.user_id)
        )
        user = user_result.scalar_one_or_none()

        if user is None or not user.is_active:
            raise AuthError("User not found or disabled", 400)

        #update the password
        user.hashed_password = hash_password(new_password)

        #mark token as used
        stored.is_used = True

        #revoke all refresh tokens — force re-login on all devices
        await self.logout_all(user.id)
        await self.db.flush()

        logger.info("password_reset_success", user_email=user.email)   
    
    # ── Private helpers ───────────────────────────────────────────────────────

    async def _get_user_by_login(self, login: str) -> User | None:
        """Find user by email """
        login = login.lower().strip()
        result = await self.db.execute(
            select(User).where(
                (User.email == login)
            )
        )
        return result.scalar_one_or_none()

    async def _record_failed_attempt(self, user: User) -> None:
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= MAX_FAILED_ATTEMPTS:
            user.locked_until = datetime.now(timezone.utc) + timedelta(
                minutes=LOCKOUT_MINUTES
            )
            logger.warning("account_locked", max_failed_attempts=MAX_FAILED_ATTEMPTS, user_email=user.email)
        await self.db.flush()

    async def _create_refresh_token(
        self,
        user: User,
        user_agent: str | None,
        ip_address: str | None,
    ) -> str:
        raw_token = generate_refresh_token()
        expires_at = datetime.now(timezone.utc) + timedelta(
            days=settings.REFRESH_TOKEN_EXPIRE_DAYS
        )
        stored = RefreshToken(
            id=uuid.uuid4(),
            user_id=user.id,
            token_hash=hash_token(raw_token),
            expires_at=expires_at,
            user_agent=user_agent,
            ip_address=ip_address,
        )
        self.db.add(stored)
        return raw_token
    
    async def update_profile(
        self, 
        user:User,
        data: UpdateProfileRequest,
    ) -> User:
        """
        Update user or admin profile
        - email, phone_nbr → optional
        - new_password → requires current_password
        """
        #update email
        if data.email is not None:
        # verify if the new email is not already taken by another user
            existing = await self.db.execute(
                select(User).where(User.email == data.email.lower())
            )
            if existing.scalar_one_or_none():
                raise AuthError("Email already registered", 409)
            user.email = data.email.lower()
        #update phone number
        if data.phone_nbr is not None:
            user.phone_nbr = data.phone_nbr
        
        #change password 
        if data.new_password is not None:
            #verify the old password 
            if not verify_password(data.current_password, user.hashed_password):
                raise AuthError("Current password is incorrect", 400)
            #update password
            user.hashed_password = hash_password(data.new_password)
            #revoke all refresh tokens — force re-login on all devices
            await self.logout_all(user.id)
        
        await self.db.flush()
        logger.info("profile_updated", user_email=user.email)
        return user
    
    
    
    
