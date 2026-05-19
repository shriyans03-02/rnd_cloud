from fastapi import APIRouter, HTTPException, Query, Depends
from typing import Optional, List, Dict, Any
from sqlalchemy.orm import Session

from app.schemas.access_group import (
    AccessGroupCreate,
    AccessGroupUpdate,
    AccessGroupOut,
    AccessGroupNode
)
from app.schemas.member import MemberOut
from app.services.access_group_service import AccessGroupService
from app.schemas.common import MessageResponse
from app.db.session import get_db
from app.db.models import AccessGroup
from app.core.dependencies import get_current_user
from app.db.models.user import User

router = APIRouter()


# ---------- STATIC ROUTES FIRST ----------

@router.get("/tree", response_model=List[AccessGroupNode])
def get_access_group_tree(
    search: Optional[str] = None,          # ← added
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """
    Return nested access group tree.
    If search is provided, returns full trees of roots that contain matching nodes.
    """
    all_groups: List[AccessGroup] = db.query(AccessGroup).all()
    tree = build_tree_from_flat(all_groups)

    if not search:
        return tree

    # Find root trees whose subtree contains the search term
    search_lower = search.strip().lower()
    node_map: Dict[int, AccessGroup] = {g.id: g for g in all_groups}

    def get_root_id(group: AccessGroup) -> int:
        current = group
        while current.parent_access_group_id is not None:
            parent = node_map.get(current.parent_access_group_id)
            if parent is None:
                break
            current = parent
        return current.id

    matched_root_ids = {
        get_root_id(g)
        for g in all_groups
        if search_lower in g.name.lower()
    }

    return [node for node in tree if node["id"] in matched_root_ids]


@router.get(
    "/list_unlinked_members_by_access_groups",
    response_model=MessageResponse[List[MemberOut]],
)
def list_unlinked_members_by_access_groups(
    access_group_id: Optional[int] = Query(None),
    search: Optional[str] = Query(None),          # ← add
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    members = AccessGroupService.list_unlinked_members_by_access_groups(
        db, access_group_id, search=search          # ← pass through
    )
    return {
        "message": "Members fetched successfully",
        "data": members,
        "total": len(members),
    }


@router.get("/all", response_model=MessageResponse[List[AccessGroupOut]])
def get_all_access_groups(
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    groups, total = AccessGroupService.list_access_groups(
        search=None,
        page=0,
        page_size=1000000
    )
    return {
        "message": "Access groups fetched successfully",
        "data": groups,
        "total": len(groups),
    }


@router.get("/active_hierarchy", response_model=MessageResponse[List[AccessGroupOut]])
def get_active_hierarchy_access_groups(
    search: str | None = Query(default=None),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    groups, total = AccessGroupService.list_access_groups_active_hierarchy(search=search)
    return {
        "message": "Access groups fetched successfully",
        "data": groups,
        "total": total,
    }


# ---------- LIST (search + pagination) ----------

@router.get("", response_model=MessageResponse[List[AccessGroupOut]])
def list_access_groups(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),
    page_size: int = Query(10, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
):
    groups, total = AccessGroupService.list_access_groups(search, page, page_size)
    return {
        "message": "Access groups fetched successfully",
        "data": groups,
        "total": total,
    }


# ---------- CREATE ----------

@router.post("", response_model=MessageResponse[AccessGroupOut])
def create_access_group(
    payload: AccessGroupCreate,
    current_user: User = Depends(get_current_user),
):
    group = AccessGroupService.create_access_group(payload, actor_id=current_user.id)
    return {"message": "Access group created successfully", "data": group}


# ---------- DYNAMIC ROUTES LAST ----------

@router.get("/{access_group_id}", response_model=MessageResponse[AccessGroupOut])
def get_access_group(
    access_group_id: int,
    current_user: User = Depends(get_current_user),
):
    group = AccessGroupService.get_access_group(access_group_id)
    if not group:
        raise HTTPException(status_code=404, detail="Access group not found")
    return {
        "message": "Access group fetched successfully",
        "data": group,
    }


@router.put("/{access_group_id}", response_model=MessageResponse[AccessGroupOut])
def update_access_group(
    access_group_id: int,
    payload: AccessGroupUpdate,
    current_user: User = Depends(get_current_user),
):
    group = AccessGroupService.update_access_group(access_group_id, payload, actor_id=current_user.id)
    return {"message": "Access group updated successfully", "data": group}


@router.delete("/{access_group_id}", response_model=MessageResponse[None])
def delete_access_group(
    access_group_id: int,
    current_user: User = Depends(get_current_user),
):
    success = AccessGroupService.delete_access_group(access_group_id, actor_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Access group not found")
    return {"message": "Access group deleted permanently"}


def build_tree_from_flat(flat: List[AccessGroup]) -> List[Dict[str, Any]]:
    node_map: Dict[int, Dict[str, Any]] = {}
    roots: List[Dict[str, Any]] = []

    for r in flat:
        node_map[r.id] = {
            "id": r.id,
            "name": r.name,
            "parent_access_group_id": r.parent_access_group_id,
            "is_active": bool(r.is_active),
            "children": [],
        }

    for r in flat:
        node = node_map[r.id]
        parent_id = r.parent_access_group_id
        if parent_id is not None and parent_id in node_map:
            node_map[parent_id]["children"].append(node)
        else:
            roots.append(node)

    return roots