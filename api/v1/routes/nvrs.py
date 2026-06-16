from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.models.user import User
from app.db.session import get_db
from app.schemas.common import MessageResponse
from app.schemas.nvr import NVRBrandTemplateOut, NVRCreate, NVROut, NVRUpdate
from app.services.nvr_service import NVRService

router = APIRouter()


@router.get("/brands", response_model=MessageResponse[list[NVRBrandTemplateOut]])
def list_nvr_brand_templates(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return {
        "message": "NVR brand templates fetched successfully",
        "data": NVRService.brand_templates(db),
    }


@router.get("/all", response_model=MessageResponse[list[dict]])
def list_all_active_nvrs(
    search: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    nvrs = NVRService.list_all_active_nvrs(db, search=search)
    data = [
        {
            "id": nvr.id,
            "name": f"{nvr.name} ({nvr.ip_address}:{nvr.port})",
            "raw_name": nvr.name,
            "ip_address": nvr.ip_address,
            "port": nvr.port,
            "brand": nvr.brand,
            "brand_id": nvr.brand_id,
            "brand_label": nvr.brand_label,
            "playback_time_format": nvr.playback_time_format,
        }
        for nvr in nvrs
    ]
    return {"message": "Active NVRs fetched successfully", "data": data}


@router.post("", response_model=MessageResponse[NVROut])
def create_nvr(
    payload: NVRCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    nvr = NVRService.create_nvr(db, payload, actor_id=current_user.id)
    return {"message": "NVR created successfully", "data": nvr}


@router.get("/{nvr_id}", response_model=MessageResponse[NVROut])
def get_nvr(
    nvr_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    nvr = NVRService.get_nvr(db, nvr_id)
    return {"message": "NVR fetched successfully", "data": nvr}


@router.get("")
def list_nvrs(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    nvrs, total = NVRService.list_nvrs(db, search, page, page_size)
    data = [
        {
            "id": nvr.id,
            "name": nvr.name,
            "ip_address": nvr.ip_address,
            "port": nvr.port,
            "brand_id": nvr.brand_id,
            "brand": nvr.brand,
            "brand_label": nvr.brand_label,
            "username": nvr.username,
            "password": nvr.password,
            "playback_rtsp_template": (
                nvr.brand_rel.playback_rtsp_template
                if nvr.brand_rel and nvr.brand_rel.playback_rtsp_template
                else nvr.playback_rtsp_template
            ),
            "playback_time_format": nvr.playback_time_format,
            "stream_key": nvr.stream_key,
            "is_active": nvr.is_active,
        }
        for nvr in nvrs
    ]
    return {"message": "NVRs fetched successfully", "data": data, "total": total}


@router.put("/{nvr_id}", response_model=MessageResponse[NVROut])
def update_nvr(
    nvr_id: int,
    payload: NVRUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    nvr = NVRService.update_nvr(db, nvr_id, payload, actor_id=current_user.id)
    return {"message": "NVR updated successfully", "data": nvr}


@router.delete("/{nvr_id}", response_model=MessageResponse[None])
def delete_nvr(
    nvr_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    success = NVRService.delete_nvr(db, nvr_id, actor_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="NVR not found")
    return {"message": "NVR deleted successfully"}
