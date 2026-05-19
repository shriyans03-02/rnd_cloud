from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from sqlalchemy import text,func
from app.db.session import SessionLocal
from app.repositories.site_location_access_repo import SiteLocationAccessRepository
from app.schemas.site_location_access import SiteLocationAccessCreate, SiteLocationAccessBulkCreate
from app.db.models.site_location import SiteLocation, site_location_access
from app.services.activity_log_service import ActivityLogService
from app.schemas.activity_log import ActivityDetail
from typing import Optional, List
from app.db.models import SiteHierarchy
from app.repositories.site_hierarchy_repo import SiteHierarchyRepository
from app.core.constants import TARGET_TYPE

SITE_LOCATION_ACCESS_TARGET_TYPE = 8
SITE_LOCATION_ACCESS_ENTITY = TARGET_TYPE[SITE_LOCATION_ACCESS_TARGET_TYPE]["entity"]


def _get_location_name(db: Session, site_location_id: int) -> str:
    try:
        row = db.execute(
            text("""
                SELECT sh.name
                FROM site_locations sl
                JOIN site_hierarchies sh ON sh.id = sl.site_hierarchy_id
                WHERE sl.id = :id
            """),
            {"id": site_location_id},
        ).mappings().first()
        return row["name"] if row else str(site_location_id)
    except Exception:
        return str(site_location_id)


def _get_access_group_name(db: Session, access_group_id: int) -> str:
    try:
        row = db.execute(
            text("SELECT name FROM access_groups WHERE id = :id"),
            {"id": access_group_id},
        ).mappings().first()
        return row["name"] if row else str(access_group_id)
    except Exception:
        return str(access_group_id)


class SiteLocationAccessService:

    @staticmethod
    def create_site_location_access(payload: SiteLocationAccessCreate):
        db: Session = SessionLocal()
        try:
            if SiteLocationAccessRepository.exists(
                db, payload.site_location_id, payload.access_group_id
            ):
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Access already exists",
                )
            SiteLocationAccessRepository.create(
                db, payload.site_location_id, payload.access_group_id
            )
            return {
                "site_location_id": payload.site_location_id,
                "access_group_id": payload.access_group_id,
            }
        finally:
            db.close()

    @staticmethod
    def bulk_assign_access(payload: SiteLocationAccessBulkCreate, actor_id: int):
        db: Session = SessionLocal()
        try:
            if not payload.site_location_ids:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="site_location_ids cannot be empty",
                )

            inserted_ids = SiteLocationAccessRepository.bulk_create_for_access_group(
                db,
                payload.access_group_id,
                payload.site_location_ids,
            )

            if not inserted_ids:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="All selected site locations are already assigned",
                )

            # resolve names once before looping
            access_group_name = _get_access_group_name(db, payload.access_group_id)

            for sl_id in inserted_ids:
                location_name = _get_location_name(db, sl_id)
                detail = ActivityDetail(
                    action="create",
                    entity=SITE_LOCATION_ACCESS_ENTITY,
                    changes={
                        "site_location": [None, location_name],
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
                    target_type=SITE_LOCATION_ACCESS_TARGET_TYPE,
                    target_id=sl_id,
                    detail=detail,
                )

            db.commit()
            return {
                "access_group_id": payload.access_group_id,
                "created_count": len(inserted_ids),
            }

        except HTTPException:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def delete_single_access(
        site_location_id: int,
        access_group_id: int,
        actor_id: int,
    ):
        db: Session = SessionLocal()
        try:
            # resolve names BEFORE delete
            location_name = _get_location_name(db, site_location_id)
            access_group_name = _get_access_group_name(db, access_group_id)

            deleted = SiteLocationAccessRepository.delete_single(
                db, site_location_id, access_group_id
            )

            if not deleted:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Access group not assigned to this site location",
                )

            detail = ActivityDetail(
                action="delete",
                entity=SITE_LOCATION_ACCESS_ENTITY,
                changes={
                    "site_location": [location_name, None],
                    "access_group":  [access_group_name, None],
                },
                meta={
                    "actor_id": actor_id,
                    "display_name": access_group_name,  # ← summary shows access group name
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=SITE_LOCATION_ACCESS_TARGET_TYPE,
                target_id=site_location_id,
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

    @staticmethod
    def list_unlinked_site_locations_by_access_group(
        db: Session,
        access_group_id: Optional[int] = None,
        search: Optional[str] = None,                     # ← add
    ) -> List[SiteLocation]:
        fully_active_hierarchy_ids = SiteHierarchyRepository.get_fully_active_hierarchy_ids(db)
        if not fully_active_hierarchy_ids:
            return []
        query = (
            db.query(SiteLocation)
            .join(SiteLocation.site_hierarchy)
            .filter(
                SiteLocation.is_active.is_(True),
                SiteHierarchy.id.in_(fully_active_hierarchy_ids),
            )
            .order_by(SiteLocation.id)
        )
        if access_group_id:
            linked_ids = (
                db.query(site_location_access.c.site_location_id)
                .filter(site_location_access.c.access_group_id == access_group_id)
                .subquery()
            )
            query = query.filter(~SiteLocation.id.in_(linked_ids))
        if search:
            query = query.filter(func.lower(SiteHierarchy.name).contains(search.lower()))  # ← add
        return query.all()

    @staticmethod
    def list_site_location_access(
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db: Session = SessionLocal()
        try:
            return SiteLocationAccessRepository.list(db, search, page, page_size)
        finally:
            db.close()

    @staticmethod
    def delete_all_for_site_location(site_location_id: int):
        db: Session = SessionLocal()
        try:
            SiteLocationAccessRepository.delete_all_for_site_location(db, site_location_id)
        finally:
            db.close()