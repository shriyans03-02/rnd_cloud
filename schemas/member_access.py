from pydantic import BaseModel, Field
from typing import Optional
from typing import List

# -------------------------
# SINGLE CREATE
# -------------------------
class MemberAccessCreate(BaseModel):
    member_id: int
    access_group_id: int


# -------------------------
# BULK CREATE
# -------------------------
class MemberAccessBulkCreate(BaseModel):
    access_group_id: int
    member_ids: List[int]

# -------------------------
# UPDATE
# -------------------------
class MemberAccessUpdate(BaseModel):
    member_id: Optional[int] = None
    access_group_id: Optional[int] = None
    is_active: Optional[bool] = None


# -------------------------
# PARENT (MINIMAL)
# -------------------------
class MemberParentOut(BaseModel):
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
class MemberAccessOut(BaseModel):
    id: int
    member_id: int
    access_group_id: int
    is_active: bool

    # RELATED OBJECTS
    member_access: Optional[MemberParentOut] = None
    access_group: Optional[AccessGroupParentOut] = None

    class Config:
        from_attributes = True
