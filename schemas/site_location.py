from pydantic import BaseModel, Field, validator
from typing import Optional


# -------------------------
# CREATE
# -------------------------
class SiteLocationCreate(BaseModel):
    name: str = Field(..., min_length=2, max_length=32)
    site_hierarchy_id: int 
    is_public: bool = False
    is_active: bool = True

    @validator("name")
    def name_cannot_be_blank(cls, v: str):
        if not v.strip():
            raise ValueError("must not be empty")
        return v.strip()


# -------------------------
# UPDATE
# -------------------------
class SiteLocationUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=2, max_length=32)
    site_hierarchy_id: Optional[int] = None 
    is_public: Optional[bool] = None
    is_active: Optional[bool] = None

    @validator("name")
    def name_cannot_be_blank(cls, v: Optional[str]):
        if v is not None and not v.strip():
            raise ValueError("must not be empty")
        return v.strip() if v else v


# -------------------------
# PARENT (MINIMAL)
# -------------------------
class SiteLocationParentOut(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True

class SiteHierarchyParentOut(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True
# -------------------------
# OUTPUT
# -------------------------
class SiteLocationOut(BaseModel):
    id: int
    name: str
    site_hierarchy_id: int 
    is_public: bool
    is_active: bool

    # Add these fields
    parent: Optional[SiteLocationParentOut] = None
    site_hierarchy: Optional[SiteHierarchyParentOut] = None

    class Config:
        from_attributes = True

class SiteLocationTreeOut(BaseModel):
    id: int
    name: str
    site_hierarchy_id: int
    site_hierarchy_name: Optional[str] 
    is_public: bool
    is_active: bool
    children: list["SiteLocationTreeOut"] = []

    class Config:
        from_attributes = True


SiteLocationTreeOut.model_rebuild()