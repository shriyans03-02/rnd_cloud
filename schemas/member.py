from pydantic import BaseModel, Field, validator, conint
from typing import Optional, List
from pydantic_core import PydanticCustomError

# -------------------------
# CREATE
# -------------------------
class MemberCreate(BaseModel):
    member_number: str = Field(..., max_length=16)
    first_name: str = Field(..., max_length=64)
    last_name: str | None = None
    department_id: Optional[int] = None 
    is_active: bool = True

    @validator("member_number")
    def validate_member_number(cls, v: str):
        if not v.strip():
            raise ValueError("Member number is required")
        return v.strip()

    @validator("first_name")
    def validate_first_name(cls, v: str):
        if not v.strip():
            raise ValueError("First name is required")
        return v.strip()

    @validator("last_name")
    def validate_last_name(cls, v: str | None):
        if v is None or v == "":
            return None  # allow blank

        if not v.strip():
            raise PydanticCustomError(
                "last_name_blank",
                "Last name cannot be only spaces",
            )

        return v.strip() 

    @validator("department_id")
    def validate_department(cls, v: int):
        if not v:
            raise ValueError("Department is required")
        return v 

# -------------------------
# UPDATE
# -------------------------
class MemberUpdate(BaseModel):
    member_number: Optional[str] = Field(None, max_length=16)
    first_name: Optional[str] = Field(None, max_length=64)
    last_name: Optional[str] = Field(None, max_length=64)
    department_id: Optional[int] = None
    is_active: Optional[bool] = None

    @validator("member_number", "first_name", "last_name")
    def field_cannot_be_blank(cls, v: Optional[str]):
        if v is None:
            return None
        v = v.strip()
        if v == "":
            return None  # allow clearing the value
        return v



# -------------------------
# DEPARTMENT (MINIMAL)
# -------------------------
class DepartmentOutMinimal(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


# -------------------------
# ACCESS GROUP (MINIMAL)
# -------------------------
class AccessGroupOutMinimal(BaseModel):
    id: int
    name: str

    class Config:
        from_attributes = True


# -------------------------
# OUTPUT
# -------------------------
class MemberOut(BaseModel):
    id: int
    member_number: str
    first_name: str
    last_name: str | None = None
    is_active: bool

    department_id: Optional[int]

    # relationships
    department: Optional[DepartmentOutMinimal] = None
    access_groups: List[AccessGroupOutMinimal] = []

    class Config:
        from_attributes = True 


