from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status
from app.core.constants import TARGET_TYPE

from app.db.session import SessionLocal
from app.repositories.department_repo import DepartmentRepository
from app.schemas.department import DepartmentCreate, DepartmentUpdate
from app.db.models.department import Department
from app.services.activity_log_service import ActivityLogService
from app.schemas.activity_log import ActivityDetail
from app.core.activity_helper import (
    snapshot,
    build_create_changes,
    build_update_changes,
    build_delete_changes,
)

DEPARTMENT_TARGET_TYPE = 2
DEPARTMENT_ENTITY = TARGET_TYPE[DEPARTMENT_TARGET_TYPE]["entity"]
DEPARTMENT_EXCLUDE = {"id", "created_ts"}


class DepartmentService:

    @staticmethod
    def create_department(payload: DepartmentCreate, actor_id: int) -> Department:
        db = SessionLocal()
        try:
            department = DepartmentRepository.create(db, payload)

            detail = ActivityDetail(
                action="create",
                entity=DEPARTMENT_ENTITY,
                changes=build_create_changes(department, exclude=DEPARTMENT_EXCLUDE),
                meta={
                    "actor_id": actor_id,
                    "display_name": department.name,
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=DEPARTMENT_TARGET_TYPE,
                target_id=department.id,
                detail=detail,
            )

            db.commit()
            db.refresh(department)
            return department

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Department already exists"
            )
        finally:
            db.close()

    @staticmethod
    def get_department(department_id: int) -> Department | None:
        db = SessionLocal()
        try:
            return DepartmentRepository.get_by_id(db, department_id)
        finally:
            db.close()

    @staticmethod
    def list_departments(
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db = SessionLocal()
        try:
            return DepartmentRepository.list(db, search, page, page_size)
        finally:
            db.close()

    @staticmethod
    def update_department(
        department_id: int, payload: DepartmentUpdate, actor_id: int
    ) -> Department:
        db = SessionLocal()
        try:
            department = DepartmentRepository.get_by_id(db, department_id)
            if not department:
                raise HTTPException(status_code=404, detail="Department not found")

            before = snapshot(department)
            updated_department = DepartmentRepository.update(db, department, payload)

            detail = ActivityDetail(
                action="update",
                entity=DEPARTMENT_ENTITY,
                changes=build_update_changes(
                    before, updated_department, exclude=DEPARTMENT_EXCLUDE
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": updated_department.name,  # use updated name
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=DEPARTMENT_TARGET_TYPE,
                target_id=department_id,
                detail=detail,
            )

            db.commit()
            db.refresh(updated_department)
            return updated_department

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Department name already exists"
            )
        finally:
            db.close()

    @staticmethod
    def delete_department(department_id: int, actor_id: int) -> bool:
        db = SessionLocal()
        try:
            department = DepartmentRepository.get_by_id(db, department_id)
            if not department:
                return False

            before = snapshot(department)
            DepartmentRepository.delete(db, department)

            detail = ActivityDetail(
                action="delete",
                entity=DEPARTMENT_ENTITY,
                changes=build_delete_changes(before, exclude=DEPARTMENT_EXCLUDE),
                meta={
                    "actor_id": actor_id,
                    "display_name": before.get("name"),  # from snapshot before delete
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=DEPARTMENT_TARGET_TYPE,
                target_id=department_id,
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
    def list_all_departments(name: str | None = None):
        db = SessionLocal()
        try:
            return DepartmentRepository.list_all_active(db, name=name)
        finally:
            db.close()