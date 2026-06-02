"""Pydantic schema for CV extraction results."""
from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class ExperienceEntry(BaseModel):
    title: Optional[str] = None
    organization: Optional[str] = None
    date_range: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)


class EducationEntry(BaseModel):
    degree: Optional[str] = None
    institution: Optional[str] = None
    date_range: Optional[str] = None


class CVSchema(BaseModel):
    document_type: str = "cv"
    full_name: Optional[str] = None
    headline: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    summary: Optional[str] = None
    skills: List[str] = Field(default_factory=list)
    experience: List[ExperienceEntry] = Field(default_factory=list)
    education: List[EducationEntry] = Field(default_factory=list)
    languages: List[str] = Field(default_factory=list)

