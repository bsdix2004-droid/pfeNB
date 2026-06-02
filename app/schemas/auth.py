"""
app/schemas/auth.py – Request / Response Pydantic schemas for auth endpoints
"""
import uuid
from datetime import datetime , date
from enum import Enum
from pydantic import BaseModel, EmailStr, Field, field_validator, model_validator

class RoleEnum(str, Enum):
    ADMIN = "admin"
    USER = "user"
# ── Register ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=3, max_length=150)
    birth_date: date | None = Field(default= None) #optional, date format: year-month-day
    office_address: str = Field(min_length=1, max_length=255)#office address is required
    phone_nbr: str | None = Field(default=None, pattern=r"^\+?[0-9\s\-]{7,20}$")
    # OPTIONAL — accepts formats: +216 12 345 678 / 0021612345678
    # pattern = validates that it is a valid phone number (digits, spaces, dash, +)
    role : RoleEnum = Field(default=RoleEnum.USER)  
      
    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        errors = []
        if not any(c.isupper() for c in v):
            errors.append("at least one uppercase letter")
        if not any(c.islower() for c in v):
            errors.append("at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            errors.append("at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v):
            errors.append("at least one special character")
        if errors:
            raise ValueError(f"Password must contain: {', '.join(errors)}")
        return v

    @field_validator("full_name")
    @classmethod
    def full_name_valid(cls, v:str) -> str:
        if not all(c.isalpha() or c.isspace() for c in v):
            raise ValueError("Full name must contain only letters and spaces")
        return v.strip() #.strip() removes leading and trailing spaces

class RegisterResponse(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    is_verified: bool
    account_enabled: bool
    created_at: datetime
    office_address: str
    phone_nbr: str | None
    birth_date: date | None
    
    model_config = {"from_attributes": True}


# ── Login ─────────────────────────────────────────────────────────────────────

class LoginRequest(BaseModel):
    """Accepts email + password."""
    login: str = Field(description="Email address")
    password: str = Field(min_length=1, max_length=128)


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int       # seconds until access token expires
    user: "UserInfo"


class UserInfo(BaseModel):
    id: uuid.UUID
    email: str
    full_name: str | None
    role: str

    model_config = {"from_attributes": True}


TokenResponse.model_rebuild()


# ── Refresh ───────────────────────────────────────────────────────────────────

class RefreshRequest(BaseModel):
    refresh_token: str


class RefreshResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int


# ── Logout ────────────────────────────────────────────────────────────────────

class LogoutRequest(BaseModel):
    refresh_token: str


class MessageResponse(BaseModel):
    message: str


# ── Update profile ───────────────────────────────────────────────────────────

class UpdateProfileRequest(BaseModel):
    """Schema for updating profile (user and admin)"""
    email: EmailStr | None = None
    phone_nbr: str | None = Field(default=None, pattern=r"^\+?[0-9\s\-]{7,20}$")
    
    # update pwd is optional
    current_password: str | None = Field(default=None, min_length=8, max_length=128)
    new_password: str | None = Field(default=None, min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str | None) -> str | None:
        if v is None:
            return v
        errors = []
        if not any(c.isupper() for c in v):
            errors.append("at least one uppercase letter")
        if not any(c.islower() for c in v):
            errors.append("at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            errors.append("at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v):
            errors.append("at least one special character")
        if errors:
            raise ValueError(f"Password must contain: {', '.join(errors)}")
        return v

    @model_validator(mode="after")
    def check_passwords(self) -> "UpdateProfileRequest":
        if self.new_password and not self.current_password:
            raise ValueError("current_password is required to change password")
        if self.current_password and not self.new_password:
            raise ValueError("new_password is required when current_password is provided")
        return self
    
#___ pwd Reset ____________________
class ForgotPasswordRequest(BaseModel):
    email: EmailStr
    
class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str = Field(min_length=8, max_length=128)
    confirm_password: str
    
    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v:str) -> str:
        errors = []
        if not any(c.isupper() for c in v):
            errors.append("at least one uppercase letter")
        if not any(c.islower() for c in v):
            errors.append("at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            errors.append("at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v):
            errors.append("at least one special character")
        if errors:
            raise ValueError(f"Password must contain: {', '.join(errors)}")
        return v
    
    @field_validator("confirm_password")
    @classmethod
    def passwords_match(cls, v: str, info) -> str:
        if "new_password" in info.data and v != info.data["new_password"]:
            raise ValueError("Passwords do not match")
        return v
    
#________ Admin schemas ______________
class AdminCreateUserRequest(BaseModel):
    """Schema for creating a user by the admin"""
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=3, max_length=150)
    office_address: str = Field(min_length=1, max_length=255)
    phone_nbr: str | None = Field(default=None, pattern=r"^\+?[0-9\s\-]{7,20}$")
    birth_date: date | None = Field(default=None)
    
    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        errors = []
        if not any(c.isupper() for c in v):
            errors.append("at least one uppercase letter")
        if not any(c.islower() for c in v):
            errors.append("at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            errors.append("at least one digit")
        if not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in v):
            errors.append("at least one special character")
        if errors:
            raise ValueError(f"Password must contain: {', '.join(errors)}")
        return v
    
    @field_validator("full_name")
    @classmethod
    def full_name_valid(cls, v: str) -> str:
        if not all(c.isalpha() or c.isspace() for c in v):
            raise ValueError("Full name must contain only letters and spaces")
        return v.strip()
    
class UpdateUserRequest(BaseModel):
    """Schema for updating a user by the admin"""
    full_name: str | None = Field(default=None, min_length=3, max_length=150)
    email: EmailStr | None = None
    office_address: str | None = Field(default=None, min_length=1, max_length=255)
    phone_nbr: str | None = Field(default=None, pattern=r"^\+?[0-9\s\-]{7,20}$")
    is_active: bool | None = None
    account_enabled: bool | None = None

    @field_validator("full_name")
    @classmethod
    def full_name_valid(cls, v: str | None) -> str | None:
        if v is not None and not all(c.isalpha() or c.isspace() for c in v):
            raise ValueError("Full name must contain only letters and spaces")
        return v.strip() if v else v
    
class UserListResponse(BaseModel):
    """Schema for the list of users returned to the admin"""
    id: uuid.UUID
    email: str
    full_name: str
    role: str
    is_active: bool
    is_verified: bool
    account_enabled: bool
    office_address: str
    phone_nbr: str | None
    created_at: datetime

    model_config = {"from_attributes": True} 

   