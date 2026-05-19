from fastapi import APIRouter, HTTPException, Depends
from typing import List, Optional

from app.schemas.user import UserCreate, UserUpdate, UserOut
from app.services.user_service import UserService
from app.schemas.common import MessageResponse
from fastapi import Query
from app.core.constants import USER_ROLES
from app.core.dependencies import get_current_user
from app.db.models.user import User

router = APIRouter()


# GET user roles
@router.get("/roles", response_model=MessageResponse[dict])
def get_user_roles(current_user: User = Depends(get_current_user)):
    return {
        "message": "Roles fetched successfully",
        "data": USER_ROLES
    }

# CREATE user
@router.post("", response_model=MessageResponse[UserOut])
def create_user(payload: UserCreate, current_user: User = Depends(get_current_user)):
    user = UserService.create_user(payload, actor_id=current_user.id)
    return {"message": "User created successfully", "data": user}

# GET user by ID
@router.get("/{user_id}", response_model=MessageResponse[UserOut])
def get_user(user_id: int, current_user: User = Depends(get_current_user)):
    user = UserService.get_user(user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return {
        "message": "User fetched successfully",
        "data": user
    }

# LIST users
@router.get("")
def list_users(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_user),
):
    users, total = UserService.list_users(search, page, page_size)
    return {
        "message": "Users fetched successfully",
        "data": users,
        "total": total,
    }

# UPDATE user
@router.put("/{user_id}", response_model=MessageResponse[UserOut])
def update_user(user_id: int, payload: UserUpdate, current_user: User = Depends(get_current_user)):
    user = UserService.update_user(user_id, payload, actor_id=current_user.id)
    return {"message": "User updated successfully", "data": user}

# HARD DELETE
@router.delete("/{user_id}", response_model=MessageResponse[None])
def delete_user(user_id: int, current_user: User = Depends(get_current_user)):
    success = UserService.delete_user(user_id, actor_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="User not found")
    return {"message": "User deleted permanently"}