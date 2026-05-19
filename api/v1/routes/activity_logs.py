from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional

from app.db.session import get_db
from app.services.activity_log_service import ActivityLogService
from app.core.dependencies import get_current_user
from app.db.models.user import User
from app.core.constants import TARGET_TYPE

router = APIRouter()


@router.get("")
def list_activity_logs(
    page: int = Query(0, ge=0),
    page_size: int = Query(20, ge=1, le=100),
    date_from: Optional[datetime] = Query(None, description="Filter from this datetime (ISO 8601)"),
    date_to: Optional[datetime] = Query(None, description="Filter to this datetime (ISO 8601)"),
    action: Optional[str] = Query(None, description="Filter by action: create | update | delete"),
    entity: Optional[str] = Query(None, description="Filter by entity: user | camera | department etc."),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    result = ActivityLogService.list_activity_logs(
        db=db,
        page=page,
        page_size=page_size,
        date_from=date_from,
        date_to=date_to,
        action=action,
        entity=entity,
    )
    return {"message": "Activity logs fetched successfully", **result}

@router.get("/logs_entity")
def list_log_entities(
    current_user: User = Depends(get_current_user),
):
    """Return the list of entity types derived from TARGET_TYPE constants."""
    entities = [v["entity"] for v in TARGET_TYPE.values()]
    return {"message": "Entities fetched successfully", "entities": entities}    

@router.delete("/bulk")
def bulk_delete_activity_logs(
    older_than_days: Optional[int] = Query(
        None,
        description="Delete logs older than N days. Omit to delete all."
    ),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    deleted_count = ActivityLogService.bulk_delete(
        db=db, older_than_days=older_than_days
    )
    return {
        "message": f"{deleted_count} activity log(s) deleted successfully",
        "deleted_count": deleted_count,
    }