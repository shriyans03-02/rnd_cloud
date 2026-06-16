from typing import Optional

from pydantic import BaseModel, Field, validator

from app.core.constants import SUPPORTED_DEVICE_TYPES, SUPPORTED_PLAYBACK_TIME_FORMATS


class PlaybackTimeFormatOption(BaseModel):
    value: str
    label: str
    example: str


class DeviceBrandBase(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)
    label: str = Field(..., min_length=2, max_length=120)
    device_type: str = Field(..., min_length=3, max_length=16)
    live_rtsp_template: Optional[str] = Field(None, max_length=1024)
    playback_rtsp_template: Optional[str] = Field(None, max_length=1024)
    playback_time_format: Optional[str] = Field(None, max_length=64)
    is_active: bool = True

    @validator("name", "label", "device_type")
    def required_text_cannot_be_blank(cls, v: str):
        if not v or not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @validator("device_type")
    def validate_device_type(cls, v: str):
        value = v.strip().lower()
        if value not in SUPPORTED_DEVICE_TYPES:
            raise ValueError("device_type must be camera or nvr")
        return value

    @validator("name")
    def normalize_name(cls, v: str):
        return v.strip().lower().replace(" ", "_")

    @validator("live_rtsp_template", "playback_rtsp_template", "playback_time_format")
    def optional_text_trim(cls, v: Optional[str]):
        if v is None:
            return None
        value = v.strip()
        return value or None

    @validator("playback_time_format")
    def validate_playback_time_format(cls, v: Optional[str]):
        if v is None:
            return None
        value = v.strip().lower()
        if value not in SUPPORTED_PLAYBACK_TIME_FORMATS:
            raise ValueError(
                "playback_time_format must be one of: "
                + ", ".join(SUPPORTED_PLAYBACK_TIME_FORMATS)
            )
        return value


class DeviceBrandCreate(DeviceBrandBase):
    pass


class DeviceBrandUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=64)
    label: Optional[str] = Field(None, min_length=2, max_length=120)
    device_type: Optional[str] = Field(None, min_length=3, max_length=16)
    live_rtsp_template: Optional[str] = Field(None, max_length=1024)
    playback_rtsp_template: Optional[str] = Field(None, max_length=1024)
    playback_time_format: Optional[str] = Field(None, max_length=64)
    is_active: Optional[bool] = None

    @validator("name")
    def normalize_name(cls, v: Optional[str]):
        if v is None:
            return None
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip().lower().replace(" ", "_")

    @validator("label")
    def optional_label_trim(cls, v: Optional[str]):
        if v is None:
            return None
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()

    @validator("device_type")
    def validate_device_type(cls, v: Optional[str]):
        if v is None:
            return None
        value = v.strip().lower()
        if value not in SUPPORTED_DEVICE_TYPES:
            raise ValueError("device_type must be camera or nvr")
        return value

    @validator("live_rtsp_template", "playback_rtsp_template", "playback_time_format")
    def optional_text_trim(cls, v: Optional[str]):
        if v is None:
            return None
        value = v.strip()
        return value or None

    @validator("playback_time_format")
    def validate_playback_time_format(cls, v: Optional[str]):
        if v is None:
            return None
        value = v.strip().lower()
        if value not in SUPPORTED_PLAYBACK_TIME_FORMATS:
            raise ValueError(
                "playback_time_format must be one of: "
                + ", ".join(SUPPORTED_PLAYBACK_TIME_FORMATS)
            )
        return value


class DeviceBrandOut(BaseModel):
    id: int
    name: str
    label: str
    device_type: str
    live_rtsp_template: Optional[str] = None
    playback_rtsp_template: Optional[str] = None
    playback_time_format: Optional[str] = None
    is_active: bool

    class Config:
        from_attributes = True
