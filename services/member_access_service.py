from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.db.session import SessionLocal, get_db
from app.repositories.member_access_repo import MemberAccessRepository
from app.schemas.member_access import MemberAccessCreate, MemberAccessBulkCreate
from app.db.models.member import member_access
from app.services.activity_log_service import ActivityLogService
from app.schemas.activity_log import ActivityDetail
from app.core.constants import TARGET_TYPE


MEMBER_ACCESS_TARGET_TYPE = 9
MEMBER_ACCESS_ENTITY = TARGET_TYPE[MEMBER_ACCESS_TARGET_TYPE]["entity"]


def _get_member_name(db: Session, member_id: int) -> str:
    try:
        row = db.execute(
            text("SELECT first_name, last_name FROM members WHERE id = :id"),
            {"id": member_id},
        ).mappings().first()
        if row:
            return f"{row['first_name']} {row['last_name'] or ''}".strip()
        return str(member_id)
    except Exception:
        return str(member_id)


def _get_access_group_name(db: Session, access_group_id: int) -> str:
    try:
        row = db.execute(
            text("SELECT name FROM access_groups WHERE id = :id"),
            {"id": access_group_id},
        ).mappings().first()
        return row["name"] if row else str(access_group_id)
    except Exception:
        return str(access_group_id)


class MemberAccessService:

    # -------- SINGLE CREATE --------
    @staticmethod
    def create_member_access(payload: MemberAccessCreate):
        db: Session = SessionLocal()
        try:
            if MemberAccessRepository.exists(
                db, payload.member_id, payload.access_group_id
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Access already exists",
                )

            MemberAccessRepository.create(
                db, payload.member_id, payload.access_group_id
            )

            return {
                "member_id": payload.member_id,
                "access_group_id": payload.access_group_id,
            }
        finally:
            db.close()

    # -------- BULK ASSIGN --------
    @staticmethod
    def bulk_assign_access(payload: MemberAccessBulkCreate, actor_id: int):
        db: Session = SessionLocal()
        try:
            # find already linked members
            existing = (
                db.query(member_access.c.member_id)
                .filter(member_access.c.access_group_id == payload.access_group_id)
                .filter(member_access.c.member_id.in_(payload.member_ids))
                .all()
            )
            existing_ids = {e[0] for e in existing}

            new_ids = [mid for mid in payload.member_ids if mid not in existing_ids]

            if not new_ids:
                return {
                    "access_group_id": payload.access_group_id,
                    "created_count": 0,
                }

            # insert new rows
            to_insert = [
                {"member_id": mid, "access_group_id": payload.access_group_id}
                for mid in new_ids
            ]
            db.execute(member_access.insert(), to_insert)

            # resolve access group name once
            access_group_name = _get_access_group_name(db, payload.access_group_id)

            # one activity log per inserted member
            for mid in new_ids:
                member_name = _get_member_name(db, mid)
                detail = ActivityDetail(
                    action="create",
                    entity=MEMBER_ACCESS_ENTITY,
                    changes={
                        "member":        [None, member_name],
                        "access_group":  [None, access_group_name],
                    },
                    meta={
                        "actor_id": actor_id,
                        "display_name": access_group_name,  # ← summary shows access group name
                    },
                )
                ActivityLogService.log(
                    db=db,
                    actor_id=actor_id,
                    target_type=MEMBER_ACCESS_TARGET_TYPE,
                    target_id=mid,
                    detail=detail,
                )

            db.commit()
            return {
                "access_group_id": payload.access_group_id,
                "created_count": len(new_ids),
            }

        except HTTPException:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # -------- LIST --------
    @staticmethod
    def list_member_access(search: str | None = None, page: int = 0, page_size: int = 10):
        db: Session = SessionLocal()
        try:
            data, total = MemberAccessRepository.list(db, search, page, page_size)
            return data, total
        finally:
            db.close()

    # -------- DELETE SINGLE ACCESS --------
    @staticmethod
    def delete_single_access(member_id: int, access_group_id: int, actor_id: int):
        """
        Delete a single access group from a member access
        """
        db: Session = SessionLocal()
        try:
            # resolve names BEFORE delete
            member_name = _get_member_name(db, member_id)
            access_group_name = _get_access_group_name(db, access_group_id)

            deleted = MemberAccessRepository.delete_single(db, member_id, access_group_id)

            if not deleted:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Access group not assigned to this member access"
                )

            detail = ActivityDetail(
                action="delete",
                entity=MEMBER_ACCESS_ENTITY,
                changes={
                    "member":       [member_name, None],
                    "access_group": [access_group_name, None],
                },
                meta={
                    "actor_id": actor_id,
                    "display_name": access_group_name,  # ← summary shows access group name
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=MEMBER_ACCESS_TARGET_TYPE,
                target_id=member_id,
                detail=detail,
            )

            db.commit()

        except HTTPException:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # -------- DELETE ALL ACCESS GROUPS FOR MEMBER ACCESS --------
    @staticmethod
    def delete_all_for_site_location(member_id: int):
        db: Session = SessionLocal()
        try:
            MemberAccessRepository.delete_all_for_site_location(db, member_id)
        finally:
            db.close()