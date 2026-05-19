from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

from pydantic import BaseModel, EmailStr, Field, validator

class UserCreate(BaseModel):
    first_name: str = Field(..., min_length=2, max_length=64)
    last_name: str = Field(..., min_length=2, max_length=64)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: int = Field(..., ge=1, le=5)

    @validator("first_name", "last_name")
    def name_cannot_be_blank(cls, v: str):
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class UserUpdate(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[int] = None
    is_active: Optional[bool] = None

class UserOut(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: EmailStr
    role: int
    is_active: bool
    created_ts: datetime
    last_login_ts: Optional[datetime]

    class Config:
        from_attributes = True
