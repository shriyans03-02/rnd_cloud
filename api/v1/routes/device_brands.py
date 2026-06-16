from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from app.core.dependencies import get_current_user
from app.db.models.user import User
from app.db.session import get_db
from app.schemas.common import MessageResponse
from app.schemas.device_brand import (
    DeviceBrandCreate,
    DeviceBrandOut,
    DeviceBrandUpdate,
    PlaybackTimeFormatOption,
)
from app.services.device_brand_service import DeviceBrandService

router = APIRouter()


@router.get("/playback-time-formats", response_model=MessageResponse[list[PlaybackTimeFormatOption]])
def playback_time_format_options(current_user: User = Depends(get_current_user)):
    return {
        "message": "Playback time formats fetched successfully",
        "data": DeviceBrandService.playback_time_format_options(),
    }


@router.post("/seed-defaults", response_model=MessageResponse[dict])
def seed_default_device_brands(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    created = DeviceBrandService.seed_defaults(db)
    return {"message": "Default device brands seeded successfully", "data": {"created": created}}


@router.get("/all", response_model=MessageResponse[list[DeviceBrandOut]])
def list_all_active_device_brands(
    device_type: Optional[str] = Query(None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows = DeviceBrandService.list_all_active_brands(db, device_type=device_type)
    return {"message": "Active device brands fetched successfully", "data": rows}


@router.post("", response_model=MessageResponse[DeviceBrandOut])
def create_device_brand(
    payload: DeviceBrandCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    brand = DeviceBrandService.create_brand(db, payload)
    return {"message": "Device brand created successfully", "data": brand}


@router.get("/{brand_id}", response_model=MessageResponse[DeviceBrandOut])
def get_device_brand(
    brand_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    brand = DeviceBrandService.get_brand(db, brand_id)
    return {"message": "Device brand fetched successfully", "data": brand}


@router.get("")
def list_device_brands(
    device_type: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    rows, total = DeviceBrandService.list_brands(
        db,
        device_type=device_type,
        search=search,
        page=page,
        page_size=page_size,
    )
    return {"message": "Device brands fetched successfully", "data": rows, "total": total}


@router.put("/{brand_id}", response_model=MessageResponse[DeviceBrandOut])
def update_device_brand(
    brand_id: int,
    payload: DeviceBrandUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    brand = DeviceBrandService.update_brand(db, brand_id, payload)
    return {"message": "Device brand updated successfully", "data": brand}


@router.delete("/{brand_id}", response_model=MessageResponse[None])
def delete_device_brand(
    brand_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    success = DeviceBrandService.delete_brand(db, brand_id)
    if not success:
        raise HTTPException(status_code=404, detail="Device brand not found")
    return {"message": "Device brand deleted successfully"}
