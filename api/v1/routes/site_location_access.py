from fastapi import APIRouter, Query, HTTPException, Depends
from typing import Optional

from app.schemas.site_location_access import SiteLocationAccessCreate, SiteLocationAccessBulkCreate
from app.schemas.common import MessageResponse
from app.services.site_location_access_service import SiteLocationAccessService
from app.core.dependencies import get_current_user
from app.db.models.user import User

router = APIRouter()

# -------- CREATE SINGLE --------
@router.post("", response_model=MessageResponse[None])
def create_site_location_access(
    payload: SiteLocationAccessCreate,
    current_user: User = Depends(get_current_user),
):
    SiteLocationAccessService.create_site_location_access(payload)
    return {"message": "Access assigned successfully"}


# -------- BULK ASSIGN --------
@router.post("/bulk", response_model=MessageResponse[dict])
def bulk_assign_site_location_access(
    payload: SiteLocationAccessBulkCreate,
    current_user: User = Depends(get_current_user),
):
    created = SiteLocationAccessService.bulk_assign_access(
        payload, actor_id=current_user.id
    )
    return {
        "message": f"{created['created_count']} access group(s) assigned successfully",
        "data": created,
    }



# -------- LIST --------
@router.get("", response_model=MessageResponse[list])
def list_site_location_access(
    search: Optional[str] = Query(None),
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1),
    current_user: User = Depends(get_current_user),
):
    data, total = SiteLocationAccessService.list_site_location_access(search, page, page_size)
    return {"message": "Access fetched successfully", "data": data, "total": total}


# -------- DELETE ALL ACCESS GROUPS FOR SITE LOCATION --------
@router.delete("/all", response_model=MessageResponse)
def delete_all_access_groups(
    site_location_id: int = Query(..., description="ID of the site location"),
    current_user: User = Depends(get_current_user),
):
    SiteLocationAccessService.delete_all_for_site_location(site_location_id)
    return {"message": "All access groups removed"}


# -------- DELETE SINGLE ACCESS GROUP --------
@router.delete("/single", response_model=MessageResponse)
def delete_single_access_group(
    site_location_id: int = Query(...),
    access_group_id: int = Query(...),
    current_user: User = Depends(get_current_user),
):
    SiteLocationAccessService.delete_single_access(
        site_location_id, access_group_id, actor_id=current_user.id
    )
    return {"message": "Access group removed successfully"}