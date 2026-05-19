from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List

from app.schemas.department import DepartmentCreate, DepartmentUpdate, DepartmentOut
from app.services.department_service import DepartmentService
from app.schemas.common import MessageResponse
from app.core.dependencies import get_current_user
from app.db.models.user import User

router = APIRouter()


@router.get("/all", response_model=List[DepartmentOut])
def list_all_departments(
    current_user: User = Depends(get_current_user),
    name: str | None = Query(default=None),
):
    return DepartmentService.list_all_departments(name=name)


@router.post("", response_model=MessageResponse[DepartmentOut])
def create_department(payload: DepartmentCreate, current_user: User = Depends(get_current_user)):
    department = DepartmentService.create_department(payload, actor_id=current_user.id)
    return {"message": "Department created successfully", "data": department}


@router.get("/{department_id}", response_model=MessageResponse[DepartmentOut])
def get_department(department_id: int, current_user: User = Depends(get_current_user)):
    department = DepartmentService.get_department(department_id)
    if not department:
        raise HTTPException(status_code=404, detail="Department not found")
    return {"message": "Department fetched successfully", "data": department}


@router.get("")
def list_departments(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_user),
):
    departments, total = DepartmentService.list_departments(search, page, page_size)
    return {"message": "Departments fetched successfully", "data": departments, "total": total}


@router.put("/{department_id}", response_model=MessageResponse[DepartmentOut])
def update_department(department_id: int, payload: DepartmentUpdate, current_user: User = Depends(get_current_user)):
    department = DepartmentService.update_department(department_id, payload, actor_id=current_user.id)
    return {"message": "Department updated successfully", "data": department}


@router.delete("/{department_id}", response_model=MessageResponse[None])
def delete_department(department_id: int, current_user: User = Depends(get_current_user)):
    success = DepartmentService.delete_department(department_id, actor_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Department not found")
    return {"message": "Department deleted permanently"}