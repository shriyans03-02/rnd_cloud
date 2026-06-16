from typing import Optional

from pydantic import BaseModel, Field, validator


class NVRBrandTemplateOut(BaseModel):
    id: Optional[int] = None
    brand: str
    name: Optional[str] = None
    label: Optional[str] = None
    playback_rtsp_template: str
    playback_time_format: Optional[str] = None


class NVRCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)
    ip_address: str = Field(..., min_length=3, max_length=64)
    port: int = Field(554, ge=1, le=65535)
    brand_id: Optional[int] = None
    brand: str = Field("generic", min_length=2, max_length=32)
    username: str = Field(..., max_length=64)
    password: str = Field(..., max_length=128)
    playback_rtsp_template: Optional[str] = Field(None, max_length=1024)
    stream_key: Optional[str] = Field(None, max_length=128)

    @validator("name", "ip_address", "brand")
    def required_text_cannot_be_blank(cls, v: str):
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @validator("username", "password")
    def required_credentials_cannot_be_blank(cls, v: str):
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @validator("playback_rtsp_template", "stream_key")
    def optional_text_trim(cls, v: Optional[str]):
        if v is None:
            return None
        value = v.strip()
        return value or None


class NVRUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=64)
    ip_address: Optional[str] = Field(None, min_length=3, max_length=64)
    port: Optional[int] = Field(None, ge=1, le=65535)
    brand_id: Optional[int] = None
    brand: Optional[str] = Field(None, min_length=2, max_length=32)
    username: Optional[str] = Field(None, max_length=64)
    password: Optional[str] = Field(None, max_length=128)
    playback_rtsp_template: Optional[str] = Field(None, max_length=1024)
    stream_key: Optional[str] = Field(None, max_length=128)
    is_active: Optional[bool] = None

    @validator("name", "ip_address", "brand")
    def optional_required_text_cannot_be_blank(cls, v: Optional[str]):
        if v is not None and not v.strip():
            raise ValueError("must not be empty")
        return v.strip() if v else v

    @validator("username", "password", "playback_rtsp_template", "stream_key")
    def optional_text_trim(cls, v: Optional[str]):
        if v is None:
            return None
        value = v.strip()
        return value or None


class NVROut(BaseModel):
    id: int
    name: str
    ip_address: str
    port: int
    brand_id: Optional[int] = None
    brand: str
    brand_label: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    playback_rtsp_template: Optional[str] = None
    playback_time_format: Optional[str] = None
    stream_key: Optional[str] = None
    is_active: bool

    class Config:
        from_attributes = True
