import io
import openpyxl
from fastapi import APIRouter, HTTPException, Query, UploadFile, File, Depends
from typing import Optional, List

from app.schemas.member import MemberCreate, MemberUpdate, MemberOut
from app.services.member_service import MemberService
from app.schemas.common import MessageResponse, BulkImportResponse
from app.core.dependencies import get_current_user
from app.db.models.user import User 



router = APIRouter()


# LIST all site locations (for dropdowns, no pagination)
@router.get("/all")
def list_all_members(
    search: str | None = None,
    page: int = 0,
    page_size: int = 10,
    current_user: User = Depends(get_current_user),
):
    members, total = MemberService.list_members(
        search=search,
        page=page,
        page_size=page_size,
    )

    return {
        "data": members,
        "total": total,
    }
# -------------------------
# LIST members (search + pagination)
# -------------------------
@router.get("")
def list_members(
    search: Optional[str] = None,
    page: int = Query(0, ge=0),     
    page_size: int = Query(10, ge=1, le=1000),
    current_user: User = Depends(get_current_user),
):
    members, total = MemberService.list_members(
        search, page, page_size
    )
    return {
        "message": "Members fetched successfully",
        "data": members,
        "total": total,
    }


# -------------------------
# CREATE member
# -------------------------
@router.post("", response_model=MessageResponse[MemberOut])
def create_member(
    payload: MemberCreate,
    current_user: User = Depends(get_current_user),
):
    member = MemberService.create_member(payload, actor_id=current_user.id)
    return {"message": "Member created successfully", "data": member}

# -------------------------
# BULK IMPORT
# -------------------------

@router.post("/bulk_import", response_model=MessageResponse[BulkImportResponse])
async def bulk_import_members(file: UploadFile = File(...),current_user: User = Depends(get_current_user),):
    if not file.filename.endswith((".xlsx", ".xls")):
        raise HTTPException(status_code=400, detail="Only .xlsx or .xls files are allowed")

    contents = await file.read()

    try:
        wb = openpyxl.load_workbook(filename=io.BytesIO(contents), read_only=True, data_only=True)
        ws = wb.active
        rows = list(ws.iter_rows(min_row=2, values_only=True))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse Excel file: {str(e)}")

    if not rows:
        raise HTTPException(status_code=400, detail="No data rows found in the uploaded file.")

    result = MemberService.bulk_import_members_from_rows(rows,actor_id=current_user.id)

    return {
        "message": result["message"],
        "data": BulkImportResponse(**result),
    }


# -------------------------
# GET member by ID
# -------------------------
@router.get("/{member_id}", response_model=MessageResponse[MemberOut])
def get_member(member_id: int,current_user: User = Depends(get_current_user),):
    member = MemberService.get_member(member_id)
    if not member:
        raise HTTPException(status_code=404, detail="Member not found")

    return {
        "message": "Member fetched successfully",
        "data": member,
    }
 
# -------------------------
# UPDATE member
# -------------------------
@router.put("/{member_id}", response_model=MessageResponse[MemberOut])
def update_member(
    member_id: int,
    payload: MemberUpdate,
    current_user: User = Depends(get_current_user),
):
    member = MemberService.update_member(member_id, payload, actor_id=current_user.id)
    return {"message": "Member updated successfully", "data": member}


# -------------------------
# HARD DELETE (permanent)
# -------------------------
@router.delete("/{member_id}", response_model=MessageResponse[None])
def delete_member(
    member_id: int,
    current_user: User = Depends(get_current_user),
):
    success = MemberService.delete_member(member_id, actor_id=current_user.id)
    if not success:
        raise HTTPException(status_code=404, detail="Member not found")
    return {"message": "Member deleted permanently"}

