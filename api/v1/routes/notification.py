from fastapi import APIRouter, Query, Depends
from pydantic import BaseModel
from typing import Optional
from datetime import datetime

from app.services.notification_service import NotificationService
from app.core.dependencies import get_current_user
from app.db.models.user import User
from app.core.constants import NOTIFICATION_STATUS, NOTIFICATION_TYPE

router = APIRouter()

_status_desc = ", ".join(f"{k}={v}" for k, v in NOTIFICATION_STATUS.items())
_type_desc   = ", ".join(f"{k}={v}" for k, v in NOTIFICATION_TYPE.items())


# ── Request bodies ─────────────────────────────────────────────────────────────
class IdsPayload(BaseModel):
    ids: list[int]


# ── Routes ─────────────────────────────────────────────────────────────────────
@router.get("")
def list_notifications(
    status:            Optional[int]      = Query(None, description=f"Filter by status ({_status_desc})"),
    type:              Optional[int]      = Query(None, description=f"Filter by type ({_type_desc})"),
    camera_id:         Optional[int]      = Query(None),
    member_id:         Optional[int]      = Query(None),
    site_hierarchy_id: Optional[int]      = Query(None),
    date_from:         Optional[datetime] = Query(None, description="Filter by created_ts >= this datetime (ISO 8601)"),
    date_to:           Optional[datetime] = Query(None, description="Filter by created_ts <= this datetime (ISO 8601)"),
    page:              int                = Query(0, ge=0),
    page_size:         int                = Query(10, ge=1, le=500),
    current_user: User = Depends(get_current_user),
):
    notifications, total = NotificationService.list_notifications(
        status=status,
        type=type,
        camera_id=camera_id,
        member_id=member_id,
        site_hierarchy_id=site_hierarchy_id,
        date_from=date_from,
        date_to=date_to,
        page=page,
        page_size=page_size,
    )
    return {"message": "Notifications fetched successfully", "data": notifications, "total": total}


@router.get("/meta/statuses")
def get_statuses(current_user: User = Depends(get_current_user)):
    return {"data": [{"id": k, "label": v} for k, v in NOTIFICATION_STATUS.items()]}


@router.get("/meta/types")
def get_types(current_user: User = Depends(get_current_user)):
    return {"data": [{"id": k, "label": v} for k, v in NOTIFICATION_TYPE.items()]}


@router.get("/{notification_id}")
def get_notification(
    notification_id: int,
    current_user: User = Depends(get_current_user),
):
    notification = NotificationService.get_notification(notification_id)
    return {"message": "Notification fetched successfully", "data": notification}


@router.patch("/{notification_id}/read")
def mark_as_read(
    notification_id: int,
    current_user: User = Depends(get_current_user),
):
    notification = NotificationService.mark_as_read(notification_id)
    return {"message": "Notification marked as read", "data": notification}


# ── NEW: mark all notifications as read ───────────────────────────────────────
@router.patch("/bulk/read-all")
def mark_all_read(
    current_user: User = Depends(get_current_user),
):
    count = NotificationService.mark_all_read()
    return {"message": f"{count} notification(s) marked as read", "updated_count": count}


# ── NEW: mark selected notifications as read ──────────────────────────────────
@router.patch("/bulk/read-selected")
def mark_selected_read(
    payload: IdsPayload,
    current_user: User = Depends(get_current_user),
):
    count = NotificationService.mark_selected_read(payload.ids)
    return {"message": f"{count} notification(s) marked as read", "updated_count": count}


# ── NEW: delete selected notifications by IDs ─────────────────────────────────
@router.delete("/bulk/selected")
def delete_selected(
    payload: IdsPayload,
    current_user: User = Depends(get_current_user),
):
    count = NotificationService.delete_selected(payload.ids)
    return {"message": f"{count} notification(s) deleted successfully", "deleted_count": count}


# ── Existing bulk delete by age ────────────────────────────────────────────────
@router.delete("/bulk")
def bulk_delete_notifications(
    older_than_days: Optional[int] = Query(
        None,
        description="Delete notifications older than N days. Omit to delete all.",
    ),
    current_user: User = Depends(get_current_user),
):
    deleted_count = NotificationService.bulk_delete(older_than_days=older_than_days)
    return {
        "message": f"{deleted_count} notification(s) deleted successfully",
        "deleted_count": deleted_count,
    }