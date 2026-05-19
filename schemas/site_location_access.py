from pydantic import BaseModel, Field
from typing import Optional
from typing import List

# -------------------------
# SINGLE CREATE
# -------------------------
class SiteLocationAccessCreate(BaseModel):
    site_location_id: int
    access_group_id: int


# -------------------------
# BULK CREATE
# -------------------------
class SiteLocationAccessBulkCreate(BaseModel):
    access_group_id: int
    site_location_ids: List[int]

# -------------------------
# UPDATE
# -------------------------
class SiteLocationAccessUpdate(BaseModel):
    site_location_id: Optional[int] = None
    access_group_id: Optional[int] = None
    is_active: Optional[bool] = None


# -------------------------
# PARENT (MINIMAL)
# -------------------------
class SiteLocationParentOut(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


class AccessGroupParentOut(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


# -------------------------
# OUTPUT
# -------------------------
class SiteLocationAccessOut(BaseModel):
    id: int
    site_location_id: int
    access_group_id: int
    is_active: bool

    # RELATED OBJECTS
    site_location: Optional[SiteLocationParentOut] = None
    access_group: Optional[AccessGroupParentOut] = None

    class Config:
        from_attributes = True
