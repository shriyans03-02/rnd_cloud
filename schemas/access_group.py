from pydantic import BaseModel, Field, validator
from typing import Optional, List


class AccessGroupNode(BaseModel):
    id: int
    name: str
    parent_access_group_id: Optional[int] = None
    is_active: bool
    children: List["AccessGroupNode"] = []

    class Config:
        orm_mode = True
        
AccessGroupNode.update_forward_refs()

# -------------------------
# CREATE
# -------------------------
class AccessGroupCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=64)
    parent_access_group_id: Optional[int] = None

    @validator("name")
    def name_cannot_be_blank(cls, v: str):
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


# -------------------------
# UPDATE
# -------------------------
class AccessGroupUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=64)
    parent_access_group_id: Optional[int] = None
    is_active: Optional[bool] = None

    @validator("name")
    def name_cannot_be_blank(cls, v: Optional[str]):
        if v is not None and not v.strip():
            raise ValueError("must not be empty")
        return v.strip() if v else v


# -------------------------
# PARENT (MINIMAL)
# -------------------------
class AccessGroupParentOut(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


# -------------------------
# OUTPUT
# -------------------------
class AccessGroupOut(BaseModel):
    id: int
    name: str
    parent_access_group_id: Optional[int]
    is_active: bool

    # ✅ Include parent for hierarchy
    parent: Optional[AccessGroupParentOut] = None

    class Config:
        from_attributes = True
