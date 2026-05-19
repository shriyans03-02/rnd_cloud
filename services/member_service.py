from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import select, func, text

from app.db.session import SessionLocal
from app.repositories.member_repo import MemberRepository
from app.schemas.member import MemberUpdate
from app.db.models.member import Member
from app.services.activity_log_service import ActivityLogService
from app.schemas.activity_log import ActivityDetail
from app.core.activity_helper import (
    snapshot,
    build_create_changes,
    build_update_changes,
    build_delete_changes,
)
from typing import Any
from app.core.constants import TARGET_TYPE


MEMBER_TARGET_TYPE = 6
MEMBER_ENTITY = TARGET_TYPE[MEMBER_TARGET_TYPE]["entity"]
MEMBER_EXCLUDE = {"id", "created_ts"}


def _get_department_name(db: Session, department_id: int) -> str | None:
    try:
        row = db.execute(
            text("SELECT name FROM departments WHERE id = :id"),
            {"id": department_id},
        ).mappings().first()
        return row["name"] if row else None
    except Exception:
        return None


def _resolve_member_changes(
    db: Session,
    changes: dict[str, list[Any]],
) -> dict[str, list[Any]]:
    """Resolve department_id → name at write time."""
    resolved = {}
    for field, (old, new) in changes.items():
        if field == "department_id":
            old_name = _get_department_name(db, old) if old is not None else None
            new_name = _get_department_name(db, new) if new is not None else None
            # only include if at least one side resolved
            if old_name is not None or new_name is not None:
                resolved["department"] = [old_name, new_name]
        else:
            resolved[field] = [old, new]
    return resolved


class MemberService:

    @staticmethod
    def create_member(payload, actor_id: int) -> Member:
        db = SessionLocal()
        try:
            if MemberRepository.exists_with_member_number(db, payload.member_number):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Member number already exists"
                )

            member = MemberRepository.create(db, payload)

            detail = ActivityDetail(
                action="create",
                entity=MEMBER_ENTITY,
                changes=_resolve_member_changes(
                    db,
                    build_create_changes(member, exclude=MEMBER_EXCLUDE),
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": f"{member.first_name} {member.last_name or ''}".strip(),
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=MEMBER_TARGET_TYPE,
                target_id=member.id,
                detail=detail,
            )

            db.commit()
            return MemberRepository.get_with_relations(db, member.id)

        except HTTPException:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def update_member(member_id: int, payload: MemberUpdate, actor_id: int) -> Member:
        db = SessionLocal()
        try:
            member = MemberRepository.get_by_id(db, member_id)
            if not member:
                raise HTTPException(status_code=404, detail="Member not found")

            if (
                payload.member_number
                and MemberRepository.exists_with_member_number(
                    db, payload.member_number, exclude_id=member_id
                )
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Member number already exists"
                )

            before = snapshot(member)
            updated_member = MemberRepository.update(db, member, payload)

            detail = ActivityDetail(
                action="update",
                entity=MEMBER_ENTITY,
                changes=_resolve_member_changes(
                    db,
                    build_update_changes(before, updated_member, exclude=MEMBER_EXCLUDE),
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": f"{updated_member.first_name} {updated_member.last_name or ''}".strip(),
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=MEMBER_TARGET_TYPE,
                target_id=member_id,
                detail=detail,
            )

            db.commit()
            return MemberRepository.get_with_relations(db, member_id)

        except HTTPException:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def get_member(member_id: int) -> Member | None:
        db = SessionLocal()
        try:
            return MemberRepository.get_by_id(db, member_id)
        finally:
            db.close()

    @staticmethod
    def list_members(
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db = SessionLocal()
        try:
            return MemberRepository.list(db=db, search=search, page=page, page_size=page_size)
        finally:
            db.close()

    @staticmethod
    def delete_member(member_id: int, actor_id: int) -> bool:
        db = SessionLocal()
        try:
            member = MemberRepository.get_by_id(db, member_id)
            if not member:
                return False

            before = snapshot(member)
            MemberRepository.delete(db, member)

            detail = ActivityDetail(
                action="delete",
                entity=MEMBER_ENTITY,
                changes=_resolve_member_changes(
                    db,
                    build_delete_changes(before, exclude=MEMBER_EXCLUDE),
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": f"{before.get('first_name', '')} {before.get('last_name', '') or ''}".strip(),
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=MEMBER_TARGET_TYPE,
                target_id=member_id,
                detail=detail,
            )

            db.commit()
            return True

        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def list_all_site_locations(db: Session):
        stmt = select(Member).where(Member.is_active.is_(True))
        members = db.execute(stmt).scalars().all()
        total = db.execute(
            select(func.count(Member.id)).where(Member.is_active.is_(True))
        ).scalar()
        return members, total

    @staticmethod
    def bulk_import_members_from_rows(rows: list,actor_id: int = 0):
        db = SessionLocal()
        try:
            dept_name_to_id = MemberRepository.get_department_name_map(db)
            raw_numbers = [str(row[0]).strip() for row in rows if row and row[0]]
            existing_numbers = MemberRepository.get_existing_member_numbers(db, raw_numbers)

            members_to_create = []
            skipped = []

            for i, row in enumerate(rows, start=2):
                if not row or not row[0]:
                    continue

                member_number = str(row[0]).strip() if row[0] else None
                first_name    = str(row[1]).strip() if row[1] else None
                last_name     = str(row[2]).strip() if row[2] else None
                dept_raw      = str(row[3]).strip() if row[3] else None

                if not member_number or not first_name:
                    skipped.append({
                        "row": i,
                        "member_number": member_number or "N/A",
                        "reason": "Member Number or First Name is missing",
                    })
                    continue

                if member_number.lower() in existing_numbers:
                    skipped.append({
                        "row": i,
                        "member_number": member_number,
                        "reason": f"Member number '{member_number}' already exists",
                    })
                    continue

                department_id = None
                if dept_raw:
                    department_id = dept_name_to_id.get(dept_raw.lower())
                    if department_id is None:
                        skipped.append({
                            "row": i,
                            "member_number": member_number,
                            "reason": f"Department '{dept_raw}' does not exist",
                        })
                        continue

                members_to_create.append(Member(
                    member_number=member_number,
                    first_name=first_name,
                    last_name=last_name,
                    department_id=department_id,
                    is_active=True,
                ))
                existing_numbers.add(member_number.lower())

            added_count = 0
            if members_to_create:
                db.add_all(members_to_create)
                db.commit()  # bulk import keeps its own commit, no activity log
                added_count = len(members_to_create)

                    # ── Activity log for bulk import ──────────────────────────────
            if added_count > 0:
                detail = ActivityDetail(
                    action="bulk_import",
                    entity=MEMBER_ENTITY,
                    changes={},
                    meta={
                        "display_name": f"{added_count} {'member' if added_count == 1 else 'members'} added",
                    },
                )
                ActivityLogService.log(
                    db=db,
                    actor_id=actor_id,
                    target_type=MEMBER_TARGET_TYPE,
                    target_id=0,
                    detail=detail,
                )
                db.commit()    

            total_skipped = len(skipped)
            message = (
                f"{added_count} {'entry' if added_count == 1 else 'entries'} added successfully. "
                f"{total_skipped} {'entry' if total_skipped == 1 else 'entries'} skipped."
                if total_skipped
                else f"{added_count} {'entry' if added_count == 1 else 'entries'} added successfully."
            )

            return {
                "message": message,
                "added_count": added_count,
                "skipped_count": total_skipped,
                "skipped": skipped,
            }
        finally:
            db.close()       