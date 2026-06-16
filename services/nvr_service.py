from typing import Optional

from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.activity_helper import (
    build_create_changes,
    build_delete_changes,
    build_update_changes,
    snapshot,
)
from app.core.constants import NVR_PLAYBACK_RTSP_TEMPLATES, TARGET_TYPE
from app.db.models.nvr import NVR
from app.repositories.device_brand_repo import DeviceBrandRepository
from app.repositories.nvr_repo import NVRRepository
from app.schemas.activity_log import ActivityDetail
from app.schemas.nvr import NVRCreate, NVRUpdate
from app.services.activity_log_service import ActivityLogService

NVR_TARGET_TYPE = 10
NVR_ENTITY = TARGET_TYPE[NVR_TARGET_TYPE]["entity"]
NVR_EXCLUDE = {"id"}


def _brand_name(value: str | None) -> str:
    return (value or "generic").strip().lower()


def _fallback_nvr_template(brand_name: str) -> str:
    return NVR_PLAYBACK_RTSP_TEMPLATES.get(brand_name) or NVR_PLAYBACK_RTSP_TEMPLATES["generic"]


def _resolve_nvr_brand(db: Session, brand_id: Optional[int], brand_name: Optional[str]):
    if brand_id is not None:
        brand = DeviceBrandRepository.get_by_id(db, int(brand_id))
        if not brand or brand.device_type != "nvr":
            raise HTTPException(status_code=400, detail="Selected NVR brand does not exist")
        return brand

    name = _brand_name(brand_name)
    brand = DeviceBrandRepository.get_by_name_type(db, name, "nvr")
    if brand:
        return brand
    return None


def _resolve_playback_rtsp_template(brand, brand_name: str, playback_rtsp_template: str | None) -> str:
    template = (playback_rtsp_template or "").strip()
    if template:
        return template
    if brand and brand.playback_rtsp_template:
        return brand.playback_rtsp_template
    return _fallback_nvr_template(brand_name)


class NVRService:

    @staticmethod
    def brand_templates(db: Session) -> list[dict[str, str]]:
        rows = DeviceBrandRepository.list_all_active(db, device_type="nvr")
        if rows:
            return [
                {
                    "id": row.id,
                    "brand": row.name,
                    "name": row.name,
                    "label": row.label,
                    "playback_rtsp_template": row.playback_rtsp_template or "",
                    "playback_time_format": row.playback_time_format,
                }
                for row in rows
            ]
        return [
            {
                "id": None,
                "brand": brand,
                "name": brand,
                "label": brand.title(),
                "playback_rtsp_template": NVR_PLAYBACK_RTSP_TEMPLATES[brand],
                "playback_time_format": "cpplus_local",
            }
            for brand in sorted(NVR_PLAYBACK_RTSP_TEMPLATES)
        ]

    @staticmethod
    def list_all_active_nvrs(db: Session, search: str | None = None) -> list[NVR]:
        return NVRRepository.list_all_active(db, search=search)

    @staticmethod
    def create_nvr(db: Session, payload: NVRCreate, actor_id: int) -> NVR:
        brand = _resolve_nvr_brand(db, payload.brand_id, payload.brand)
        brand_name = brand.name if brand else _brand_name(payload.brand)
        playback_rtsp_template = _resolve_playback_rtsp_template(
            brand,
            brand_name,
            payload.playback_rtsp_template,
        )

        try:
            nvr = NVRRepository.create(
                db,
                payload,
                brand_name=brand_name,
                brand_id=brand.id if brand else None,
                playback_rtsp_template=playback_rtsp_template,
            )

            detail = ActivityDetail(
                action="create",
                entity=NVR_ENTITY,
                changes=build_create_changes(nvr, exclude=NVR_EXCLUDE),
                meta={"actor_id": actor_id, "display_name": nvr.name},
            )
            ActivityLogService.log(db=db, actor_id=actor_id, target_type=NVR_TARGET_TYPE, target_id=nvr.id, detail=detail)

            db.commit()
            db.refresh(nvr)
            return nvr

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="NVR with this IP address and port already exists",
            )

    @staticmethod
    def get_nvr(db: Session, nvr_id: int) -> NVR:
        nvr = NVRRepository.get_by_id(db, nvr_id)
        if not nvr:
            raise HTTPException(status_code=404, detail="NVR not found")
        return nvr

    @staticmethod
    def list_nvrs(db: Session, search: str | None = None, page: int = 0, page_size: int = 10):
        return NVRRepository.list(db, search, page, page_size)

    @staticmethod
    def update_nvr(db: Session, nvr_id: int, payload: NVRUpdate, actor_id: int) -> NVR:
        nvr = NVRRepository.get_by_id(db, nvr_id)
        if not nvr:
            raise HTTPException(status_code=404, detail="NVR not found")

        data = payload.model_dump(exclude_unset=True)
        brand = None
        brand_name: Optional[str] = None
        brand_id: Optional[int] = None
        set_brand_id = False
        playback_rtsp_template: Optional[str] = None
        set_playback_rtsp_template = False

        if "brand_id" in data or "brand" in data:
            brand = _resolve_nvr_brand(db, data.get("brand_id", nvr.brand_id), data.get("brand", nvr.brand))
            brand_name = brand.name if brand else _brand_name(data.get("brand", nvr.brand))
            brand_id = brand.id if brand else None
            set_brand_id = True
            if "playback_rtsp_template" not in data:
                playback_rtsp_template = _resolve_playback_rtsp_template(brand, brand_name, None)
                set_playback_rtsp_template = True

        if "playback_rtsp_template" in data:
            brand_for_template = brand or (nvr.brand_rel if getattr(nvr, "brand_rel", None) else None)
            brand_name_for_template = brand_name or _brand_name(nvr.brand)
            playback_rtsp_template = _resolve_playback_rtsp_template(
                brand_for_template,
                brand_name_for_template,
                data.get("playback_rtsp_template"),
            )
            set_playback_rtsp_template = True

        try:
            before = snapshot(nvr)
            updated_nvr = NVRRepository.update(
                db,
                nvr,
                payload,
                brand_name=brand_name,
                brand_id=brand_id,
                playback_rtsp_template=playback_rtsp_template,
                set_brand_id=set_brand_id,
                set_playback_rtsp_template=set_playback_rtsp_template,
            )

            detail = ActivityDetail(
                action="update",
                entity=NVR_ENTITY,
                changes=build_update_changes(before, updated_nvr, exclude=NVR_EXCLUDE),
                meta={"actor_id": actor_id, "display_name": updated_nvr.name},
            )
            ActivityLogService.log(db=db, actor_id=actor_id, target_type=NVR_TARGET_TYPE, target_id=nvr_id, detail=detail)

            db.commit()
            db.refresh(updated_nvr)
            return updated_nvr

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="NVR with this IP address and port already exists",
            )

    @staticmethod
    def delete_nvr(db: Session, nvr_id: int, actor_id: int) -> bool:
        nvr = NVRRepository.get_by_id(db, nvr_id)
        if not nvr:
            return False

        before = snapshot(nvr)
        NVRRepository.delete(db, nvr)

        detail = ActivityDetail(
            action="delete",
            entity=NVR_ENTITY,
            changes=build_delete_changes(before, exclude=NVR_EXCLUDE),
            meta={"actor_id": actor_id, "display_name": before.get("name")},
        )
        ActivityLogService.log(db=db, actor_id=actor_id, target_type=NVR_TARGET_TYPE, target_id=nvr_id, detail=detail)

        db.commit()
        return True
