from app.db.session import SessionLocal
from app.repositories.user_repo import UserRepository
from app.schemas.user import UserCreate, UserUpdate
from app.db.models.user import User
from app.core.security import hash_password
from app.services.activity_log_service import ActivityLogService
from app.schemas.activity_log import ActivityDetail
from app.core.activity_helper import snapshot, build_create_changes, build_update_changes, build_delete_changes
from app.core.constants import USER_ROLES , TARGET_TYPE
from fastapi import HTTPException, status
from typing import Any

USER_TARGET_TYPE = 1
USER_ENTITY = TARGET_TYPE[USER_TARGET_TYPE]["entity"]
USER_EXCLUDE = {"id", "created_ts", "last_login_ts", "password"}


def _resolve_user_changes(changes: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Resolve FK integers to human-readable values at write time."""
    resolved = {}
    for field, (old, new) in changes.items():
        if field == "role":
            resolved["role"] = [
                USER_ROLES.get(old, old) if old is not None else None,
                USER_ROLES.get(new, new) if new is not None else None,
            ]
        else:
            resolved[field] = [old, new]
    return resolved


class UserService:

    @staticmethod
    def create_user(payload: UserCreate, actor_id: int) -> User:
        db = SessionLocal()
        try:
            if UserRepository.get_by_email(db, payload.email):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already exists"
                )

            payload.password = hash_password(payload.password)
            user = UserRepository.create(db, payload)

            detail = ActivityDetail(
                action="create",
                entity=USER_ENTITY,
                changes=_resolve_user_changes(
                    build_create_changes(user, exclude=USER_EXCLUDE)
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": f"{user.first_name} {user.last_name}",
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=USER_TARGET_TYPE,
                target_id=user.id,
                detail=detail,
            )

            db.commit()
            db.refresh(user)
            return user
        except HTTPException:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def get_user(user_id: int) -> User | None:
        db = SessionLocal()
        try:
            return UserRepository.get_by_id(db, user_id)
        finally:
            db.close()

    @staticmethod
    def list_users(search: str | None = None, page: int = 0, page_size: int = 10):
        db = SessionLocal()
        try:
            return UserRepository.list(db, search, page, page_size)
        finally:
            db.close()

    @staticmethod
    def update_user(user_id: int, payload: UserUpdate, actor_id: int) -> User | None:
        db = SessionLocal()
        try:
            user = UserRepository.get_by_id(db, user_id)
            if not user:
                return None

            before = snapshot(user)
            updated_user = UserRepository.update(db, user, payload)

            detail = ActivityDetail(
                action="update",
                entity=USER_ENTITY,
                changes=_resolve_user_changes(
                    build_update_changes(before, updated_user, exclude=USER_EXCLUDE)
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": f"{updated_user.first_name} {updated_user.last_name}",
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=USER_TARGET_TYPE,
                target_id=user_id,
                detail=detail,
            )

            db.commit()
            db.refresh(updated_user)
            return updated_user
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def delete_user(user_id: int, actor_id: int) -> bool:
        db = SessionLocal()
        try:
            user = UserRepository.get_by_id(db, user_id)
            if not user:
                return False

            before = snapshot(user)
            UserRepository.delete(db, user)

            detail = ActivityDetail(
                action="delete",
                entity=USER_ENTITY,
                changes=_resolve_user_changes(
                    build_delete_changes(before, exclude=USER_EXCLUDE)
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": f"{before.get('first_name', '')} {before.get('last_name', '')}".strip(),
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=USER_TARGET_TYPE,
                target_id=user_id,
                detail=detail,
            )

            db.commit()
            return True
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()