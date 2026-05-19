from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, Query, HTTPException

from app.schemas.common import MessageResponse
from app.services.member_tracing_service import MemberTracingService
from app.core.dependencies import get_current_user
from app.db.models.user import User

router = APIRouter()


@router.get("/", response_model=MessageResponse[dict])
def get_member_tracing(
    member_id: int = Query(..., description="ID of the member to trace"),
    date: datetime = Query(
        ...,
        description="The date to trace (ISO 8601). Time component is ignored "
                    "unless entry_ts / exit_ts are omitted.",
    ),
    entry_ts: Optional[datetime] = Query(
        None,
        description="Explicit window start datetime. Defaults to midnight of `date`.",
    ),
    exit_ts: Optional[datetime] = Query(
        None,
        description="Explicit window end datetime. Defaults to end-of-day of `date`.",
    ),
    current_user: User = Depends(get_current_user),
):
    """
    Trace a member's movement for a given date/time window.

    Returns:
    - **member**    – full member details (name, department, …)
    - **locations** – each unique location visited, with per-visit breakdown
                      (entry/exit times, duration, visit number)
    - **timeline**  – flat chronological list of every camera event
    - **summary**   – aggregate stats (first seen, last seen, total time, …)
    """
    if entry_ts and exit_ts and entry_ts >= exit_ts:
        raise HTTPException(
            status_code=422,
            detail="`entry_ts` must be earlier than `exit_ts`.",
        )

    result = MemberTracingService.get_member_trace(
        member_id=member_id,
        date=date,
        entry_ts=entry_ts,
        exit_ts=exit_ts,
    )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    return {"message": "Member trace fetched successfully", "data": result}


@router.get("/active/names", response_model=MessageResponse[list])
def list_active_member_names(
    search: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_user),
):
    members = MemberTracingService.list_active_member_names(search=search, limit=limit)
    return {"message": "Active member names fetched successfully", "data": members}


@router.get("/active/site_locations", response_model=MessageResponse[list])
def list_site_locations(
    search: Optional[str] = Query(None),
    current_user: User = Depends(get_current_user),
):
    data = MemberTracingService.list_site_locations(search=search)
    return {"message": "Site locations fetched successfully", "data": data}

@router.get("/location/cameras", response_model=MessageResponse[list])
def get_cameras_by_location(
    site_location_id: int = Query(..., description="Site location ID to fetch cameras for"),
    current_user: User = Depends(get_current_user),
):
    data = MemberTracingService.get_cameras_by_location(site_location_id=site_location_id)
    return {"message": "Cameras fetched successfully", "data": data}