from pydantic import BaseModel, Field, validator, computed_field
from typing import Optional


class CameraCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)
    ip_address: str = Field(..., min_length=7, max_length=16)
    site_location_id: int 

    @validator("name")
    def name_cannot_be_blank(cls, v: str):
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @validator("ip_address")
    def ip_not_blank(cls, v: str):
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


class CameraUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=64)
    ip_address: Optional[str] = Field(None, min_length=7, max_length=16)
    site_location_id: Optional[int] = None 
    is_active: Optional[bool] = None

    @validator("name")
    def name_cannot_be_blank(cls, v: Optional[str]):
        if v is not None and not v.strip():
            raise ValueError("must not be empty")
        return v.strip() if v else v

    @validator("ip_address")
    def ip_not_blank(cls, v: Optional[str]):
        if v is not None and not v.strip():
            raise ValueError("must not be empty")
        return v.strip() if v else v
 
class CameraOut(BaseModel):
    id: int
    name: str
    ip_address: str  
    is_active: bool
    site_location: str | None
    site_location_id: int

    class Config:
        orm_mode = True
