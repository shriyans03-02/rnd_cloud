from pydantic import BaseModel, Field, conint
from typing import Optional, List, Tuple 

class StartRequest(BaseModel):
    member_id: conint(gt=0) = Field(..., description="Member/User ID to capture for")
    camera_ids: Optional[List[int]] = Field(None, description="Which cameras to use. If null -> all configured")
    show_viewer: bool = Field(True, description="Open OpenCV viewer window")
    clear_existing: bool = Field(False, description="Clear existing galleries before starting")


class StartResponse(BaseModel):
    status: str
    message: str = ""
    member_id: Optional[int] = None
    member_name: Optional[str] = None
    num_cams: int = 0
    camera_ids: List[int] = []
    configured_camera_ids: Optional[List[int]] = None


class StopResponse(BaseModel):
    status: str
    message: str = ""
    member_id: Optional[int] = None
    camera_ids: List[int] = []


class ExtractRequest(BaseModel):
    member_id: conint(gt=0)
    camera_ids: Optional[List[int]] = None
    sync: bool = False