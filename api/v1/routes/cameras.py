import io
import openpyxl
from app.schemas.common import BulkImportResponse
from fastapi import APIRouter, HTTPException, Query, Depends , UploadFile, File
from typing import Optional
from sqlalchemy.orm import Session, joinedload
from app.repositories.site_hierarchy_repo import SiteHierarchyRepository
from app.schemas.camera import (
    CameraCreate,
    CameraUpdate,
    CameraOut,
)
from app.schemas.common import MessageResponse
from app.services.camera_service import CameraService
from app.db.session import get_db
from app.db.models import Camera, SiteHierarchy 
from app.db.models.site_location import SiteLocation 
from app.core.dependencies import get_current_user
from app.db.models.user import User
from sqlalchemy import or_

router = APIRouter()

@router.get("/search", response_model=MessageResponse[list[dict]])
def search_cameras(
    search: str | None = Query(None, description="Search by camera or site location"),
    limit: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = (
        db.query(Camera)
        .join(Camera.site_location_rel)
        .join(SiteLocation.site_hierarchy)
        .filter(Camera.is_active == True)
    )

    if search:
        like = f"%{search}%"
        query = query.filter(
            or_(
                Camera.name.ilike(like),
                Camera.ip_address.ilike(like),
                SiteHierarchy.name.ilike(like),
            )
        )

    cameras = query.order_by(Camera.name).limit(limit).all()

    data = [
        {
            "id": cam.id,
            "name": cam.name,
            "ip_address": cam.ip_address,
            "site_location": cam.site_location,  # uses computed property
        }
        for cam in cameras
    ]

    return {
        "message": "Camera search results fetched successfully",
        "data": data,
    }

# FETCH all active cameras
@router.get("/allcameras", response_model=MessageResponse[list[dict]])
def list_active_cameras(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cameras = CameraService.list_active_cameras(db)
    data = [{"id": cam.id, "name": cam.name} for cam in cameras]
    return {"message": "Active cameras fetched successfully", "data": data}


# FETCH site locations 
@router.get("/load_site_locations", response_model=MessageResponse[list[dict]])
def list_site_locations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    search: Optional[str] = Query(None),
):
    fully_active_hierarchy_ids = SiteHierarchyRepository.get_fully_active_hierarchy_ids(db)

    if not fully_active_hierarchy_ids:
        return {"message": "Site locations fetched successfully", "data": []}

    query = (
        db.query(SiteLocation)
        .join(SiteLocation.site_hierarchy)
        .options(joinedload(SiteLocation.site_hierarchy))
        .filter(
            SiteLocation.is_active.is_(True),
            SiteHierarchy.id.in_(fully_active_hierarchy_ids),
        )
    )

    if search:                                   # ← add this block
        query = query.filter(
            SiteHierarchy.name.ilike(f"%{search}%")
        )

    locations = query.all()

    data = [
        {
            "id": loc.id,
            "name": loc.site_hierarchy.name if loc.site_hierarchy else None,
        }
        for loc in locations
    ]
    return {"message": "Site locations fetched successfully", "data": data}


@router.post("/bulk_import", response_model=MessageResponse[BulkImportResponse])
async def bulk_import_cameras(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_user),
):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only .xlsx or .xls files are allowed")

    contents = await file.read()

    try:
        wb = openpyxl.load_workbook(filename=io.BytesIO(contents), read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(min_row=2, values_only=True))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {str(e)}")

    if not rows:
        raise HTTPException(status_code=400, detail="No data rows found in the uploaded file.")

    result = CameraService.bulk_import_cameras_from_rows(rows,actor_id=current_user.id)

    return {
        "message": result["message"],
        "data": BulkImportResponse(**result),
    }

# CREATE camera
@router.post("", response_model=MessageResponse[CameraOut])
def create_camera(
    payload: CameraCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    camera = CameraService.create_camera(db, payload, actor_id=current_user.id)
    camera = (
        db.query(Camera)
        .options(joinedload(Camera.site_location_rel))
        .filter(Camera.id == camera.id)
        .first()
    )
    return {"message": "Camera created successfully", "data": camera}

# GET camera by ID
@router.get("/{camera_id}", response_model=MessageResponse[CameraOut])
def get_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    camera = (
        db.query(Camera)
        .options(joinedload(Camera.site_location_rel))
        .filter(Camera.id == camera_id)
        .first()
    )

    if not camera:
        raise HTTPException(status_code=404, detail="Camera not found")

    return {"message": "Camera fetched successfully", "data": camera}


# LIST cameras
@router.get("", response_model=None)
def list_cameras(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    cameras, total = CameraService.list_cameras(db, search, page, page_size)
    return {
        "message": "Cameras fetched successfully",
        "data": cameras,
        "total": total,
    }


@router.put("/{camera_id}", response_model=MessageResponse[CameraOut])
def update_camera(
    camera_id: int,
    payload: CameraUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    camera = CameraService.update_camera(db, camera_id, payload, actor_id=current_user.id)
    return {"message": "Camera updated successfully", "data": camera}



# DELETE camera
@router.delete("/{camera_id}", response_model=MessageResponse[None])
def delete_camera(
    camera_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    success = CameraService.delete_camera(db, camera_id, actor_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")
    return {"message": "Camera deleted successfully"}