from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy import text, select
from app.repositories.camera_repo import CameraRepository
from app.schemas.camera import CameraCreate, CameraUpdate
from app.db.models.camera import Camera
from app.repositories.site_hierarchy_repo import SiteHierarchyRepository
from app.services.activity_log_service import ActivityLogService
from app.schemas.activity_log import ActivityDetail
from app.core.activity_helper import (
    snapshot,
    build_create_changes,
    build_update_changes,
    build_delete_changes,
)
from app.db.models.site_location import SiteLocation
from app.db.models.site_hierarchy import SiteHierarchy
from app.core.constants import TARGET_TYPE
from typing import Any

CAMERA_TARGET_TYPE = 4
CAMERA_ENTITY = TARGET_TYPE[CAMERA_TARGET_TYPE]["entity"]
CAMERA_EXCLUDE = {"id"}


def _get_site_location_name(db: Session, site_location_id: int) -> str | None:
    """Resolve site_location_id → site_hierarchy name at write time."""
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
        return row["name"] if row else None
    except Exception:
        return None


def _resolve_camera_changes(db: Session, changes: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Resolve FK integers to human-readable values at write time."""
    resolved = {}
    for field, (old, new) in changes.items():
        if field == "site_location_id":
            old_name = _get_site_location_name(db, old) if old is not None else None
            new_name = _get_site_location_name(db, new) if new is not None else None
            # only include if at least one side resolved
            if old_name is not None or new_name is not None:
                resolved["site_location"] = [old_name, new_name]
            # if neither resolved (e.g. orphaned FK), skip the field
        else:
            resolved[field] = [old, new]
    return resolved


class CameraService:

    @staticmethod
    def list_active_cameras(db: Session):
        return db.query(Camera).filter(Camera.is_active == True).all()

    @staticmethod
    def create_camera(db: Session, payload: CameraCreate, actor_id: int) -> Camera:
        if payload.site_location_id is None:
            raise HTTPException(status_code=400, detail="site_location_id is required")
        try:
            camera = CameraRepository.create(db, payload)

            detail = ActivityDetail(
                action="create",
                entity=CAMERA_ENTITY,
                changes=_resolve_camera_changes(
                    db,
                    build_create_changes(camera, exclude=CAMERA_EXCLUDE),
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": camera.name,
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=CAMERA_TARGET_TYPE,
                target_id=camera.id,
                detail=detail,
            )

            db.commit()
            db.refresh(camera)
            return camera

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Camera with this name or IP already exists",
            )

    @staticmethod
    def get_camera(db: Session, camera_id: int) -> Camera:
        camera = CameraRepository.get_by_id(db, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")
        return camera

    @staticmethod
    def list_cameras(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        return CameraRepository.list(db, search, page, page_size)

    @staticmethod
    def update_camera(db: Session, camera_id: int, payload: CameraUpdate, actor_id: int) -> Camera:
        if payload.site_location_id is None:
            raise HTTPException(status_code=400, detail="site_location_id is required")

        camera = CameraRepository.get_by_id(db, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        try:
            before = snapshot(camera)
            updated_camera = CameraRepository.update(db, camera, payload)

            detail = ActivityDetail(
                action="update",
                entity=CAMERA_ENTITY,
                changes=_resolve_camera_changes(
                    db,
                    build_update_changes(before, updated_camera, exclude=CAMERA_EXCLUDE),
                ),
                meta={
                    "actor_id": actor_id,
                    "display_name": updated_camera.name,
                },
            )
            ActivityLogService.log(
                db=db,
                actor_id=actor_id,
                target_type=CAMERA_TARGET_TYPE,
                target_id=camera_id,
                detail=detail,
            )

            db.commit()
            db.refresh(updated_camera)
            return updated_camera

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Camera with this name or IP already exists",
            )

    @staticmethod
    def delete_camera(db: Session, camera_id: int, actor_id: int) -> bool:
        camera = CameraRepository.get_by_id(db, camera_id)
        if not camera:
            return False

        before = snapshot(camera)
        CameraRepository.delete(db, camera)

        detail = ActivityDetail(
            action="delete",
            entity=CAMERA_ENTITY,
            changes=_resolve_camera_changes(
                db,
                build_delete_changes(before, exclude=CAMERA_EXCLUDE),
            ),
            meta={
                "actor_id": actor_id,
                "display_name": before.get("name"),
            },
        )
        ActivityLogService.log(
            db=db,
            actor_id=actor_id,
            target_type=CAMERA_TARGET_TYPE,
            target_id=camera_id,
            detail=detail,
        )

        db.commit()
        return True

    @staticmethod
    def bulk_import_cameras_from_rows(rows: list,actor_id: int = 0):
        from app.db.session import SessionLocal

        db = SessionLocal()
        try:
            fully_active_hierarchy_ids = SiteHierarchyRepository.get_fully_active_hierarchy_ids(db)
            if fully_active_hierarchy_ids:
                location_rows = db.execute(
                    select(SiteLocation.id, SiteHierarchy.name)
                    .join(SiteHierarchy, SiteHierarchy.id == SiteLocation.site_hierarchy_id)
                    .where(
                        SiteLocation.is_active.is_(True),
                        SiteHierarchy.id.in_(fully_active_hierarchy_ids),
                    )
                ).all()
                site_location_name_to_id = {
                    row.name.strip().lower(): row.id for row in location_rows
                }
            else:
                site_location_name_to_id = {}

            raw_ips = [str(row[1]).strip() for row in rows if row and row[1]]
            existing_ips = CameraRepository.get_existing_ip_addresses(db, raw_ips)

            cameras_to_create = []
            skipped = []

            for i, row in enumerate(rows, start=2):
                if not row or not row[0]:
                    continue

                name = str(row[0]).strip() if row[0] else None
                ip_address = str(row[1]).strip() if row[1] else None
                site_loc_raw = str(row[2]).strip() if row[2] else None

                if not name or not ip_address or not site_loc_raw:
                    skipped.append({
                        "row": i,
                        "name": name or "N/A",
                        "reason": "Name, IP address, or Site Location is missing",
                    })
                    continue

                if ip_address in existing_ips:
                    skipped.append({
                        "row": i,
                        "name": name,
                        "reason": f"IP address '{ip_address}' already exists",
                    })
                    continue

                site_location_id = site_location_name_to_id.get(site_loc_raw.lower())
                if site_location_id is None:
                    skipped.append({
                        "row": i,
                        "name": name,
                        "reason": f"Site location '{site_loc_raw}' does not exist or is not active",
                    })
                    continue

                cameras_to_create.append(Camera(
                    name=name,
                    ip_address=ip_address,
                    site_location_id=site_location_id,
                    is_active=True,
                ))
                existing_ips.add(ip_address)

            added_count = 0
            if cameras_to_create:
                db.add_all(cameras_to_create)
                db.commit()
                added_count = len(cameras_to_create)

            # ── Activity log for bulk import ──────────────────────────────
            if added_count > 0:
                detail = ActivityDetail(
                    action="bulk_import",
                    entity=CAMERA_ENTITY,
                    changes={},
                    meta={
                        "display_name": f"{added_count} {'camera' if added_count == 1 else 'cameras'} added",
                    },
                )
                ActivityLogService.log(
                    db=db,
                    actor_id=actor_id,
                    target_type=CAMERA_TARGET_TYPE,
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