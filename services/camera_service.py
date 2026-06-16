from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from sqlalchemy import text, select
from app.repositories.camera_repo import CameraRepository
from app.repositories.device_brand_repo import DeviceBrandRepository
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
from app.db.models.nvr import NVR
from app.core.constants import CAMERA_RTSP_URL_TEMPLATES, TARGET_TYPE
from typing import Any, Optional

CAMERA_TARGET_TYPE = 4
CAMERA_ENTITY = TARGET_TYPE[CAMERA_TARGET_TYPE]["entity"]
CAMERA_EXCLUDE = {"id"}


def _brand_name(value: str | None) -> str:
    return (value or "generic").strip().lower()


def _fallback_camera_template(brand_name: str) -> str:
    return CAMERA_RTSP_URL_TEMPLATES.get(brand_name) or CAMERA_RTSP_URL_TEMPLATES["generic"]


def _resolve_camera_brand(db: Session, brand_id: Optional[int], brand_name: Optional[str]):
    if brand_id is not None:
        brand = DeviceBrandRepository.get_by_id(db, int(brand_id))
        if not brand or brand.device_type != "camera":
            raise HTTPException(status_code=400, detail="Selected camera brand does not exist")
        return brand

    name = _brand_name(brand_name)
    brand = DeviceBrandRepository.get_by_name_type(db, name, "camera")
    if brand:
        return brand
    return None


def _resolve_camera_rtsp_template(brand, brand_name: str, rtsp_url_template: str | None) -> str:
    template = (rtsp_url_template or "").strip()
    if template:
        return template
    if brand and brand.live_rtsp_template:
        return brand.live_rtsp_template
    return _fallback_camera_template(brand_name)


def _get_site_location_name(db: Session, site_location_id: int) -> str | None:
    """Resolve site_location_id -> site_hierarchy name at write time."""
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


def _get_nvr_name(db: Session, nvr_id: int) -> str | None:
    try:
        row = db.execute(select(NVR.name).where(NVR.id == nvr_id)).first()
        return row[0] if row else None
    except Exception:
        return None


def _resolve_camera_changes(db: Session, changes: dict[str, list[Any]]) -> dict[str, list[Any]]:
    """Resolve FK integers to human-readable values at write time."""
    resolved = {}
    for field, (old, new) in changes.items():
        if field == "site_location_id":
            old_name = _get_site_location_name(db, old) if old is not None else None
            new_name = _get_site_location_name(db, new) if new is not None else None
            if old_name is not None or new_name is not None:
                resolved["site_location"] = [old_name, new_name]
        elif field == "nvr_id":
            old_name = _get_nvr_name(db, old) if old is not None else None
            new_name = _get_nvr_name(db, new) if new is not None else None
            resolved["nvr"] = [old_name, new_name]
        else:
            resolved[field] = [old, new]
    return resolved


def _ensure_nvr_exists(db: Session, nvr_id: int | None) -> None:
    if nvr_id is None:
        return
    exists = db.execute(select(NVR.id).where(NVR.id == nvr_id)).first()
    if not exists:
        raise HTTPException(status_code=400, detail="Selected NVR does not exist")


class CameraService:

    @staticmethod
    def brand_templates(db: Session) -> list[dict[str, str]]:
        rows = DeviceBrandRepository.list_all_active(db, device_type="camera")
        if rows:
            return [
                {
                    "id": row.id,
                    "brand": row.name,
                    "name": row.name,
                    "label": row.label,
                    "rtsp_url_template": row.live_rtsp_template or "",
                }
                for row in rows
            ]
        return [
            {
                "id": None,
                "brand": brand,
                "name": brand,
                "label": brand.title(),
                "rtsp_url_template": CAMERA_RTSP_URL_TEMPLATES[brand],
            }
            for brand in sorted(CAMERA_RTSP_URL_TEMPLATES)
        ]

    @staticmethod
    def list_active_cameras(db: Session):
        return db.query(Camera).filter(Camera.is_active == True).all()

    @staticmethod
    def create_camera(db: Session, payload: CameraCreate, actor_id: int) -> Camera:
        if payload.site_location_id is None:
            raise HTTPException(status_code=400, detail="site_location_id is required")

        brand = _resolve_camera_brand(db, payload.brand_id, payload.brand)
        brand_name = brand.name if brand else _brand_name(payload.brand)
        rtsp_url_template = _resolve_camera_rtsp_template(
            brand,
            brand_name,
            payload.rtsp_url_template,
        )
        _ensure_nvr_exists(db, payload.nvr_id)

        try:
            camera = CameraRepository.create(
                db,
                payload,
                brand_name=brand_name,
                brand_id=brand.id if brand else None,
                rtsp_url_template=rtsp_url_template,
            )

            detail = ActivityDetail(
                action="create",
                entity=CAMERA_ENTITY,
                changes=_resolve_camera_changes(
                    db,
                    build_create_changes(camera, exclude=CAMERA_EXCLUDE),
                ),
                meta={"actor_id": actor_id, "display_name": camera.name},
            )
            ActivityLogService.log(db=db, actor_id=actor_id, target_type=CAMERA_TARGET_TYPE, target_id=camera.id, detail=detail)

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
    def list_cameras(db: Session, search: str | None = None, page: int = 0, page_size: int = 10):
        return CameraRepository.list(db, search, page, page_size)

    @staticmethod
    def update_camera(db: Session, camera_id: int, payload: CameraUpdate, actor_id: int) -> Camera:
        camera = CameraRepository.get_by_id(db, camera_id)
        if not camera:
            raise HTTPException(status_code=404, detail="Camera not found")

        data = payload.model_dump(exclude_unset=True)
        if "site_location_id" in data and data["site_location_id"] is None:
            raise HTTPException(status_code=400, detail="site_location_id is required")

        brand = None
        brand_name: Optional[str] = None
        brand_id: Optional[int] = None
        set_brand_id = False
        rtsp_url_template: Optional[str] = None
        set_rtsp_url_template = False

        if "brand_id" in data or "brand" in data:
            brand = _resolve_camera_brand(db, data.get("brand_id", camera.brand_id), data.get("brand", camera.brand))
            brand_name = brand.name if brand else _brand_name(data.get("brand", camera.brand))
            brand_id = brand.id if brand else None
            set_brand_id = True
            if "rtsp_url_template" not in data:
                rtsp_url_template = _resolve_camera_rtsp_template(brand, brand_name, None)
                set_rtsp_url_template = True

        if "rtsp_url_template" in data:
            brand_for_template = brand or (camera.brand_rel if getattr(camera, "brand_rel", None) else None)
            brand_name_for_template = brand_name or _brand_name(camera.brand)
            rtsp_url_template = _resolve_camera_rtsp_template(
                brand_for_template,
                brand_name_for_template,
                data.get("rtsp_url_template"),
            )
            set_rtsp_url_template = True

        if "nvr_id" in data:
            _ensure_nvr_exists(db, data.get("nvr_id"))

        try:
            before = snapshot(camera)
            updated_camera = CameraRepository.update(
                db,
                camera,
                payload,
                brand_name=brand_name,
                brand_id=brand_id,
                rtsp_url_template=rtsp_url_template,
                set_brand_id=set_brand_id,
                set_rtsp_url_template=set_rtsp_url_template,
            )

            detail = ActivityDetail(
                action="update",
                entity=CAMERA_ENTITY,
                changes=_resolve_camera_changes(
                    db,
                    build_update_changes(before, updated_camera, exclude=CAMERA_EXCLUDE),
                ),
                meta={"actor_id": actor_id, "display_name": updated_camera.name},
            )
            ActivityLogService.log(db=db, actor_id=actor_id, target_type=CAMERA_TARGET_TYPE, target_id=camera_id, detail=detail)

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
            changes=_resolve_camera_changes(db, build_delete_changes(before, exclude=CAMERA_EXCLUDE)),
            meta={"actor_id": actor_id, "display_name": before.get("name")},
        )
        ActivityLogService.log(db=db, actor_id=actor_id, target_type=CAMERA_TARGET_TYPE, target_id=camera_id, detail=detail)

        db.commit()
        return True

    @staticmethod
    def bulk_import_cameras_from_rows(rows: list, actor_id: int = 0):
        from app.db.session import SessionLocal

        db = SessionLocal()
        try:
            generic_brand = DeviceBrandRepository.get_by_name_type(db, "generic", "camera")
            generic_template = (
                generic_brand.live_rtsp_template if generic_brand else CAMERA_RTSP_URL_TEMPLATES["generic"]
            )
            generic_brand_id = generic_brand.id if generic_brand else None

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
                site_location_name_to_id = {row.name.strip().lower(): row.id for row in location_rows}
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
                    skipped.append({"row": i, "name": name or "N/A", "reason": "Name, IP address, or Site Location is missing"})
                    continue

                if ip_address in existing_ips:
                    skipped.append({"row": i, "name": name, "reason": f"IP address '{ip_address}' already exists"})
                    continue

                site_location_id = site_location_name_to_id.get(site_loc_raw.lower())
                if site_location_id is None:
                    skipped.append({"row": i, "name": name, "reason": f"Site location '{site_loc_raw}' does not exist or is not active"})
                    continue

                cameras_to_create.append(Camera(
                    name=name,
                    ip_address=ip_address,
                    brand="generic",
                    brand_id=generic_brand_id,
                    rtsp_url_template=generic_template,
                    rtsp_port=554,
                    rtsp_channel=1,
                    rtsp_subtype="0",
                    site_location_id=site_location_id,
                    is_active=True,
                ))
                existing_ips.add(ip_address)

            added_count = 0
            if cameras_to_create:
                db.add_all(cameras_to_create)
                db.commit()
                added_count = len(cameras_to_create)

            if added_count > 0:
                detail = ActivityDetail(
                    action="bulk_import",
                    entity=CAMERA_ENTITY,
                    changes={},
                    meta={"display_name": f"{added_count} {'camera' if added_count == 1 else 'cameras'} added"},
                )
                ActivityLogService.log(db=db, actor_id=actor_id, target_type=CAMERA_TARGET_TYPE, target_id=0, detail=detail)
                db.commit()

            total_skipped = len(skipped)
            message = (
                f"{added_count} {'entry' if added_count == 1 else 'entries'} added successfully. "
                f"{total_skipped} {'entry' if total_skipped == 1 else 'entries'} skipped."
                if total_skipped
                else f"{added_count} {'entry' if added_count == 1 else 'entries'} added successfully."
            )

            return {"message": message, "added_count": added_count, "skipped_count": total_skipped, "skipped": skipped}

        finally:
            db.close()
