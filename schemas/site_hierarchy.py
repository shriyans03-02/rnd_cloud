from pydantic import BaseModel, Field, validator
from typing import Optional, List
from app.schemas.base import TrimmedModel 
# -------------------------
# CREATE
# -------------------------
class SiteHierarchyCreate(TrimmedModel):
    name: str
    parent_site_hierarchy_id: Optional[int] = None
    is_active: bool = True
    is_public: Optional[bool] = False
    is_protected: Optional[bool] = False


# -------------------------
# UPDATE
# -------------------------
class SiteHierarchyUpdate(TrimmedModel):
    name: Optional[str]
    parent_site_hierarchy_id: Optional[int]
    is_active: Optional[bool]
    is_public: Optional[bool] = False
    is_protected: Optional[bool] = False


# -------------------------
# OUTPUT
# -------------------------
class SiteHierarchyNode(BaseModel):
    id: int
    name: str
    parent_site_hierarchy_id: Optional[int]
    is_active: bool
    is_deletable: bool = True 
    children: List["SiteHierarchyNode"] = []
    is_locked: bool = False
    is_public: Optional[bool] = None
    is_protected: Optional[bool] = None
    is_leaf: bool = False
    model_config = {"from_attributes": True}

SiteHierarchyNode.update_forward_refs()


class SiteHierarchyOut(SiteHierarchyNode):
    pass
