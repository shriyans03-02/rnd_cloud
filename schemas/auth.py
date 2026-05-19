from pydantic import BaseModel, EmailStr , Field , field_validator
from app.db.models.user import User
from datetime import datetime, timedelta, timezone
import jwt
from app.core.config import settings


JWT_SECRET = settings.JWT_SECRET
JWT_ALGORITHM = "HS256"
JWT_EXPIRE_HOURS = 24
JWT_EXPIRE_MINUTES = 15 


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"

    @classmethod
    def from_user(cls, user: User) -> "LoginResponse":
        payload = {
            "id": user.id,
            "name": f"{user.first_name} {user.last_name}".strip(),
            "email": user.email,
            "role": user.role,
            "exp": datetime.now(timezone.utc) + timedelta(minutes=JWT_EXPIRE_MINUTES),  # changed
        }
        token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
        return cls(access_token=token)
    
class RegisterRequest(BaseModel):
    first_name: str = Field(..., min_length=2, max_length=64)
    last_name: str = Field(..., min_length=2, max_length=64)
    email: EmailStr
    password: str = Field(..., min_length=8)
    role: int = Field(..., ge=1, le=5)

    @field_validator("first_name", "last_name")
    @classmethod
    def name_cannot_be_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class RegisterResponse(BaseModel):
    id: int
    first_name: str
    last_name: str
    email: EmailStr
    role: int

    class Config:
        from_attributes = True    