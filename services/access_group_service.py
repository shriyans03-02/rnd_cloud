from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text,func

from app.db.session import SessionLocal
from app.repositories.access_group_repo import AccessGroupRepository
from app.schemas.access_group import AccessGroupUpdate
from app.services.activity_log_service import ActivityLogService
from app.schemas.activity_log import ActivityDetail
from app.core.activity_helper import (
    snapshot,
    build_create_changes,
    build_update_changes,
    build_delete_changes,
)
from app.db.models import AccessGroup, Member
from app.db.models.associations import member_access
from app.core.constants import TARGET_TYPE
from typing import Any

ACCESS_GROUP_TARGET_TYPE = 5
ACCESS_GROUP_ENTITY = TARGET_TYPE[ACCESS_GROUP_TARGET_TYPE]["entity"]
ACCESS_GROUP_EXCLUDE = {"id"}


def _get_access_group_name(db: Session, group_id: int) -> str | None:
    try:
        row = db.execute(
            text("SELECT name FROM access_groups WHERE id = :id"),
            {"id": group_id},
        ).mappings().first()
        return row["name"] if row else None
    except Exception:
        return None


def _resolve_access_group_changes(
    db: Session,
    changes: dict[str, list[Any]],
) -> dict[str, list[Any]]:
    """Resolve parent_access_group_id → name at write time."""
    resolved = {}
    for field, (old, new) in changes.items():
        if field == "parent_access_group_id":
            old_name = _get_access_group_name(db, old) if old is not None else None
            new_name = _get_access_group_name(db, new) if new is not None else None
            if old_name is not None or new_name is not None:
                resolved["parent_access_group"] = [old_name, new_name]
        else:
            resolved[field] = [old, new]
    return resolved


class AccessGroupService:

    @staticmethod
    def create_access_group(payload, actor_id: int) -> AccessGroup:
        db = SessionLocal()
        try:
            if AccessGroupRepository.exists_with_name(
                db, payload.name, payload.parent_access_group_id
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Access group name already exists under this parent"
                )

            group = AccessGroupRepository.create(db, payload)

            detail = ActivityDetail(
                action="create",
                entity=ACCESS_GROUP_ENTITY,
                changes=_resolve_access_group_changes(
                    db,
                    build_create_changes(group, exclude=ACCESS_GROUP_EXCLUDE),
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": group.name,
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=ACCESS_GROUP_TARGET_TYPE,
                target_id=group.id,
                detail=detail,
            )

            db.commit()
            return AccessGroupRepository.get_with_relations(db, group.id)

        except HTTPException:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def update_access_group(
        access_group_id: int, payload: AccessGroupUpdate, actor_id: int
    ) -> AccessGroup:
        db = SessionLocal()
        try:
            group = AccessGroupRepository.get_by_id(db, access_group_id)
            if not group:
                raise HTTPException(status_code=404, detail="Access group not found")

            before = snapshot(group)
            updated_group = AccessGroupRepository.update(db, group, payload)

            detail = ActivityDetail(
                action="update",
                entity=ACCESS_GROUP_ENTITY,
                changes=_resolve_access_group_changes(
                    db,
                    build_update_changes(before, updated_group, exclude=ACCESS_GROUP_EXCLUDE),
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": updated_group.name,
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=ACCESS_GROUP_TARGET_TYPE,
                target_id=access_group_id,
                detail=detail,
            )

            db.commit()
            return AccessGroupRepository.get_with_relations(db, access_group_id)

        except HTTPException:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def get_access_group(access_group_id: int) -> AccessGroup | None:
        db = SessionLocal()
        try:
            return AccessGroupRepository.get_by_id(db, access_group_id)
        finally:
            db.close()

    @staticmethod
    def list_access_groups(
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db = SessionLocal()
        try:
            return AccessGroupRepository.list(db, search, page, page_size)
        finally:
            db.close()

    @staticmethod
    def list_access_groups_active_hierarchy(search: str | None = None):
        db = SessionLocal()
        try:
            return AccessGroupRepository.list_active_hierarchy(db, search=search)
        finally:
            db.close()

    @staticmethod
    def delete_access_group(access_group_id: int, actor_id: int) -> bool:
        """
        Cascade-delete an access group node and ALL its descendants (leaf-first).
        Logs activity only for the root node being deleted.
        """
        db = SessionLocal()
        try:
            # ── 1. Build children map from all nodes ─────────────────────────
            raw_nodes = AccessGroupRepository.fetch_all_nodes_raw(db)
            children_map = AccessGroupRepository.build_children_map(raw_nodes)

            if access_group_id not in children_map:
                return False

            # ── 2. Snapshot root node for activity log before deletion ────────
            root_node = AccessGroupRepository.get_by_id(db, access_group_id)
            if not root_node:
                return False

            before = snapshot(root_node)

            # ── 3. Delete leaf-first (children before parents) ────────────────
            AccessGroupRepository.delete_subtree(db, access_group_id, children_map)

            # ── 4. Activity log for root only ─────────────────────────────────
            detail = ActivityDetail(
                action="delete",
                entity=ACCESS_GROUP_ENTITY,
                changes=_resolve_access_group_changes(
                    db,
                    build_delete_changes(before, exclude=ACCESS_GROUP_EXCLUDE),
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": before.get("name"),
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=ACCESS_GROUP_TARGET_TYPE,
                target_id=access_group_id,
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
    def list_unlinked_members_by_access_groups(
        db: Session,
        access_group_id: int | None = None,
        search: str | None = None,                    # ← add
    ):
        query = db.query(Member).order_by(Member.id)
        if access_group_id:
            linked_ids = (
                db.query(member_access.c.member_id)
                .filter(member_access.c.access_group_id == access_group_id)
                .subquery()
            )
            query = query.filter(
                ~Member.id.in_(linked_ids),
                Member.is_active.is_(True),
            )
        if search:
            search_lower = f"%{search.lower()}%"
            query = query.filter(
                func.lower(Member.first_name + " " + Member.last_name).like(search_lower)
            )
        return query.all()