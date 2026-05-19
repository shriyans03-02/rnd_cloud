from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import exists, and_

from app.schemas.site_hierarchy import (
    SiteHierarchyCreate,
    SiteHierarchyUpdate,
    SiteHierarchyOut,
    SiteHierarchyNode,
)
from app.schemas.common import MessageResponse
from app.db.session import get_db
from app.db.models.site_hierarchy import SiteHierarchy
from app.db.models.site_location import SiteLocation
from app.db.models import Camera
from app.services.site_hierarchy_service import SiteHierarchyService
from app.repositories.site_hierarchy_repo import SiteHierarchyRepository
from app.core.dependencies import get_current_user
from app.db.models.user import User

router = APIRouter()


# ── TREE ──────────────────────────────────────────────────────────────────────

@router.get("/tree", response_model=List[SiteHierarchyNode])
def get_site_hierarchy_tree(
    search: Optional[str] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    return SiteHierarchyService.get_tree(db, search=search)


# ── LIST ──────────────────────────────────────────────────────────────────────

@router.get("", response_model=MessageResponse[List[SiteHierarchyOut]])
def list_site_hierarchies(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=1000),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    used_by_camera = exists().where(
        and_(
            Camera.site_location_id == SiteLocation.id,
            SiteLocation.site_hierarchy_id == SiteHierarchy.id,
        )
    )
    query = db.query(SiteHierarchy).filter(~used_by_camera)
    if search:
        query = query.filter(SiteHierarchy.name.ilike(f"%{search}%"))

    total = query.count()
    sites = query.offset(page * page_size).limit(page_size).all()
    for site in sites:
        site.children = []

    return {
        "message": "Site hierarchies fetched successfully",
        "data": [SiteHierarchyOut.from_orm(s) for s in sites],
        "total": total,
    }


@router.get("/active_hierarchy", response_model=MessageResponse[List[SiteHierarchyOut]])
def list_active_site_hierarchies(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
    search: Optional[str] = Query(None),  # ← add this
):
    used_by_camera = exists().where(
        and_(
            Camera.site_location_id == SiteLocation.id,
            SiteLocation.site_hierarchy_id == SiteHierarchy.id,
        )
    )
    sites = (
        db.query(SiteHierarchy)
        .filter(~used_by_camera, SiteHierarchy.is_active == True)
        .all()
    )
    all_sites_map: dict[int, SiteHierarchy] = {
        s.id: s for s in db.query(SiteHierarchy).all()
    }

    def all_ancestors_active(site: SiteHierarchy) -> bool:
        current = site
        while current.parent_site_hierarchy_id is not None:
            parent = all_sites_map.get(current.parent_site_hierarchy_id)
            if parent is None or not parent.is_active:
                return False
            current = parent
        return True

    valid_sites = [s for s in sites if all_ancestors_active(s)]
    for site in valid_sites:
        site.children = []

    # ← apply search filter after hierarchy validation
    if search:
        valid_sites = [s for s in valid_sites if search.lower() in s.name.lower()]

    return {
        "message": "Active site hierarchies fetched successfully",
        "data": [SiteHierarchyOut.from_orm(s) for s in valid_sites],
        "total": len(valid_sites),
    }


# ── CREATE ────────────────────────────────────────────────────────────────────

@router.post("", response_model=MessageResponse[SiteHierarchyOut])
def create_site_hierarchy(
    payload: SiteHierarchyCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    site = SiteHierarchyService.create_site_hierarchy(db, payload, actor_id=current_user.id)
    site.children = []
    return {
        "message": "Site hierarchy created successfully",
        "data": SiteHierarchyOut.from_orm(site),
    }


# ── READ (single) ─────────────────────────────────────────────────────────────

@router.get("/{site_hierarchy_id}", response_model=MessageResponse[SiteHierarchyOut])
def get_site_hierarchy(
    site_hierarchy_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from sqlalchemy.orm import selectinload
    site = (
        db.query(SiteHierarchy)
        .options(selectinload(SiteHierarchy.children))
        .filter(SiteHierarchy.id == site_hierarchy_id)
        .first()
    )
    if not site:
        raise HTTPException(status_code=404, detail="Site hierarchy not found")
    site.children = []
    return {
        "message": "Site hierarchy fetched successfully",
        "data": SiteHierarchyOut.from_orm(site),
    }


# ── UPDATE ────────────────────────────────────────────────────────────────────

@router.put("/{site_hierarchy_id}", response_model=MessageResponse[SiteHierarchyOut])
def update_site_hierarchy(
    site_hierarchy_id: int,
    payload: SiteHierarchyUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    site = SiteHierarchyService.update_site_hierarchy(
        db, site_hierarchy_id, payload, actor_id=current_user.id
    )
    site.children = []
    return {
        "message": "Site hierarchy updated successfully",
        "data": SiteHierarchyOut.from_orm(site),
    }


# ── DELETE ────────────────────────────────────────────────────────────────────

@router.delete("/{site_hierarchy_id}", response_model=MessageResponse[None])
def delete_site_hierarchy(
    site_hierarchy_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    SiteHierarchyService.delete_site_hierarchy(db, site_hierarchy_id, actor_id=current_user.id)
    return {"message": "Site hierarchy and all its children deleted successfully"}


# ── FULL TREE (read-only deep view) ──────────────────────────────────────────

@router.get("/full_tree/{site_hierarchy_id}", response_model=MessageResponse[List[dict]])
def get_full_hierarchy_tree(
    site_hierarchy_id: int,
    current_user: User = Depends(get_current_user),
):
    tree = SiteHierarchyService.get_full_hierarchy_tree(site_hierarchy_id)
    return {
        "message": "Site hierarchy + locations tree fetched successfully",
        "data": tree,
    }