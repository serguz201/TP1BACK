import uuid
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field


class UserCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=150)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: str = Field(..., pattern=r"^(admin|operativo|analista)$")


class UserUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=150)
    role: Optional[str] = Field(None, pattern=r"^(admin|operativo|analista)$")


class UserResponse(BaseModel):
    id: uuid.UUID
    name: str
    email: str
    role: str
    status: str
    created_at: datetime

    class Config:
        from_attributes = True


class UserListResponse(BaseModel):
    items: list[UserResponse]
    total: int
