from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
from sqlalchemy.orm import Session

from app.schemas.site_location import (
    SiteLocationCreate,
    SiteLocationUpdate,
    SiteLocationOut,
    SiteLocationTreeOut
)
from app.schemas.common import MessageResponse
from app.services.site_location_service import SiteLocationService
from app.services.site_location_access_service import SiteLocationAccessService
from app.db.session import get_db
from app.core.dependencies import get_current_user
from app.db.models.user import User

router = APIRouter()


@router.get(
    "/list_unlinked_site_locations_by_access_group",
    response_model=MessageResponse[List[SiteLocationOut]],
)
def list_unlinked_site_locations_by_access_group(
    access_group_id: Optional[int] = Query(None),
    search: Optional[str] = Query(None),              # ← add
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    site_locations = SiteLocationAccessService.list_unlinked_site_locations_by_access_group(
        db, access_group_id, search=search             # ← pass through
    )
    return {
        "message": "Site locations fetched successfully",
        "data": site_locations,
        "total": len(site_locations),
    }

# ---------------- TREE ----------------
@router.get("/tree", response_model=MessageResponse[List[SiteLocationTreeOut]])
def get_tree(
    db: Session = Depends(get_db),
    search: str | None = None,
    current_user: User = Depends(get_current_user),
):
    tree = SiteLocationService.get_tree(db, search=search)

    return {
        "message": "Tree fetched successfully",
        "data": tree,
    }


# ---------------- ALL ----------------
@router.get("/all", response_model=MessageResponse[List[SiteLocationOut]])
def list_all_site_locations(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    locations, total = SiteLocationService.list_all_site_locations(db)

    return {
        "message": "Site locations fetched successfully",
        "data": locations,
        "total": total,
    }


# ---------------- FULL TREE ----------------
@router.get("/full_tree/{site_location_id}", response_model=MessageResponse[list])
def get_full_tree_by_site_location(
    site_location_id: int,
    current_user: User = Depends(get_current_user),
):
    tree = SiteLocationService.get_full_tree_from_site_location(site_location_id)

    if not tree:
        raise HTTPException(
            status_code=404,
            detail="No active site locations found for this hierarchy",
        )

    return {
        "message": "Site hierarchy + locations tree fetched successfully",
        "data": tree,
    }


# ---------------- LIST (paginated) ----------------
@router.get("", response_model=MessageResponse[List[SiteLocationOut]])
def list_site_locations(
    search: Optional[str] = None,
    site_hierarchy_id: Optional[int] = None,
    parent_site_location_id: Optional[int] = None,
    is_public: Optional[bool] = None,
    is_active: Optional[bool] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
):
    locations, total = SiteLocationService.list_site_locations(
        search=search,
        site_hierarchy_id=site_hierarchy_id,
        parent_site_location_id=parent_site_location_id,
        is_public=is_public,
        is_active=is_active,
        page=page,
        page_size=page_size,
    )

    return {
        "message": "Site locations fetched successfully",
        "data": locations,
        "total": total,
    }


# ---------------- CREATE ----------------
@router.post("", response_model=MessageResponse[SiteLocationOut])
def create_site_location(
    payload: SiteLocationCreate,
    current_user: User = Depends(get_current_user),
):
    location = SiteLocationService.create_site_location(payload)

    return {
        "message": "Site location created successfully",
        "data": location,
    }


# ---------------- GET BY ID ----------------
@router.get("/{site_location_id}", response_model=MessageResponse[SiteLocationOut])
def get_site_location(
    site_location_id: int,
    current_user: User = Depends(get_current_user),
):
    location = SiteLocationService.get_site_location(site_location_id)

    if not location:
        raise HTTPException(status_code=404, detail="Site location not found")

    return {
        "message": "Site location fetched successfully",
        "data": location,
    }


# ---------------- UPDATE ----------------
@router.put("/{site_location_id}", response_model=MessageResponse[SiteLocationOut])
def update_site_location(
    site_location_id: int,
    payload: SiteLocationUpdate,
    current_user: User = Depends(get_current_user),
):
    location = SiteLocationService.update_site_location(site_location_id, payload)

    return {
        "message": "Site location updated successfully",
        "data": location,
    }


# ---------------- DELETE ----------------
@router.delete("/{site_location_id}", response_model=MessageResponse[None])
def delete_site_location(
    site_location_id: int,
    current_user: User = Depends(get_current_user),
):
    success = SiteLocationService.delete_site_location(site_location_id)

    if not success:
        raise HTTPException(status_code=404, detail="Site location not found")

    return {
        "message": "Site location deleted permanently",
    }