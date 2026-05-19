from datetime import datetime
from typing import Optional

from fastapi import APIRouter,HTTPException, Depends, Query

from app.schemas.common import MessageResponse
from app.services.report_service import NormalizedReportService
from app.core.dependencies import get_current_user
from app.db.models.user import User
from app.core.constants import MOVEMENT_TYPES
from app.db.session import get_db
from app.db.models.site_location import SiteLocation 
from app.db.models.user import User
from app.db.models import Camera
from sqlalchemy.orm import Session


router = APIRouter()


@router.get("/movement_types", response_model=MessageResponse[dict])
def get_movement_types(current_user: User = Depends(get_current_user)):
    return {
        "message": "Movement types fetched successfully",
        "data": MOVEMENT_TYPES,
    }


@router.get("/site_locations", response_model=MessageResponse[list])
def list_site_locations(
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    data = NormalizedReportService.list_site_locations(search=search)
    return {"message": "Site locations fetched successfully", "data": data}


@router.get("/active/names", response_model=MessageResponse[list])
def list_active_member_names(
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
):
    members = NormalizedReportService.list_active_member_names(
        search=search, limit=limit
    )
    return {
        "message": "Active member names fetched successfully",
        "data": members,
    }


@router.get("/member_presence_report", response_model=MessageResponse[dict])
def get_normalized_report(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    start_ts: Optional[datetime] = Query(None),
    end_ts: Optional[datetime] = Query(None),
    site_location_id: Optional[int] = Query(None),
    member_id: Optional[int] = Query(None),
    movement_type: Optional[int] = Query(None),
    search: Optional[str] = Query(None),
    min_match: Optional[int] = Query(None),
    current_user: User = Depends(get_current_user),
):
    result = NormalizedReportService.get_report(
        page, page_size, start_ts, end_ts,
        site_location_id, member_id, movement_type, search, min_match,
    )
    return {"message": "Report fetched successfully", "data": result}




@router.get("/camera_display/{camera_id}", response_model=MessageResponse[dict])
def get_camera_by_id(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    camera = (
        db.query(Camera)
        .join(Camera.site_location_rel)
        .join(SiteLocation.site_hierarchy)
        .filter(Camera.id == camera_id, Camera.is_active == True)
        .first()
    )

    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    return {
        "message": "Camera fetched successfully",
        "data": {
            "id": camera.id,
            "name": camera.name,
            "ip_address": camera.ip_address,
            "site_location": camera.site_location,
        },
    }