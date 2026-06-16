from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.core.constants import (
    DEFAULT_DEVICE_BRANDS,
    PLAYBACK_TIME_FORMAT_LABELS,
    SUPPORTED_DEVICE_TYPES,
)
from app.db.models.device_brand import DeviceBrand
from app.repositories.device_brand_repo import DeviceBrandRepository
from app.schemas.device_brand import DeviceBrandCreate, DeviceBrandUpdate


PLAYBACK_TIME_FORMAT_EXAMPLES = {
    "hikvision_utc": "20260325T000000Z",
    "hikvision_local": "20260325T000000",
    "cpplus_local": "2026_03_25_00_00_00",
    "iso_local": "2026-03-25T00:00:00",
}


def _normalize_device_type(device_type: str | None) -> str | None:
    if device_type is None:
        return None
    value = str(device_type).strip().lower()
    if value not in SUPPORTED_DEVICE_TYPES:
        raise HTTPException(status_code=400, detail="device_type must be camera or nvr")
    return value


def _validate_templates(payload: DeviceBrandCreate | DeviceBrandUpdate, existing: DeviceBrand | None = None) -> None:
    device_type = getattr(payload, "device_type", None) or (existing.device_type if existing else None)
    device_type = _normalize_device_type(device_type)

    live_rtsp_template = getattr(payload, "live_rtsp_template", None)
    playback_rtsp_template = getattr(payload, "playback_rtsp_template", None)
    playback_time_format = getattr(payload, "playback_time_format", None)

    if existing is not None:
        if live_rtsp_template is None and "live_rtsp_template" not in payload.model_fields_set:
            live_rtsp_template = existing.live_rtsp_template
        if playback_rtsp_template is None and "playback_rtsp_template" not in payload.model_fields_set:
            playback_rtsp_template = existing.playback_rtsp_template
        if playback_time_format is None and "playback_time_format" not in payload.model_fields_set:
            playback_time_format = existing.playback_time_format

    if device_type == "camera":
        if not live_rtsp_template:
            raise HTTPException(status_code=400, detail="Camera brands require live_rtsp_template")
        # Clear NVR-only fields when type is camera.
        payload.playback_rtsp_template = None
        payload.playback_time_format = None

    if device_type == "nvr":
        if not playback_rtsp_template:
            raise HTTPException(status_code=400, detail="NVR brands require playback_rtsp_template")
        if not playback_time_format:
            raise HTTPException(status_code=400, detail="NVR brands require playback_time_format")
        # Clear camera-only field when type is nvr.
        payload.live_rtsp_template = None


class DeviceBrandService:

    @staticmethod
    def playback_time_format_options() -> list[dict[str, str]]:
        return [
            {
                "value": value,
                "label": label,
                "example": PLAYBACK_TIME_FORMAT_EXAMPLES.get(value, ""),
            }
            for value, label in PLAYBACK_TIME_FORMAT_LABELS.items()
        ]

    @staticmethod
    def seed_defaults(db: Session) -> int:
        created = 0
        for item in DEFAULT_DEVICE_BRANDS:
            exists = DeviceBrandRepository.get_by_name_type(
                db,
                item["name"],
                item["device_type"],
            )
            if exists:
                continue
            brand = DeviceBrand(**item, is_active=True)
            db.add(brand)
            created += 1
        if created:
            db.commit()
        return created

    @staticmethod
    def list_all_active_brands(db: Session, device_type: str | None = None) -> list[DeviceBrand]:
        return DeviceBrandRepository.list_all_active(db, device_type=_normalize_device_type(device_type))

    @staticmethod
    def get_brand(db: Session, brand_id: int) -> DeviceBrand:
        brand = DeviceBrandRepository.get_by_id(db, brand_id)
        if not brand:
            raise HTTPException(status_code=404, detail="Device brand not found")
        return brand

    @staticmethod
    def list_brands(
        db: Session,
        device_type: str | None = None,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        return DeviceBrandRepository.list(
            db,
            device_type=_normalize_device_type(device_type),
            search=search,
            page=page,
            page_size=page_size,
        )

    @staticmethod
    def create_brand(db: Session, payload: DeviceBrandCreate) -> DeviceBrand:
        _validate_templates(payload)
        try:
            brand = DeviceBrandRepository.create(db, payload)
            db.commit()
            db.refresh(brand)
            return brand
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Brand with this name and device type already exists",
            )

    @staticmethod
    def update_brand(db: Session, brand_id: int, payload: DeviceBrandUpdate) -> DeviceBrand:
        brand = DeviceBrandRepository.get_by_id(db, brand_id)
        if not brand:
            raise HTTPException(status_code=404, detail="Device brand not found")
        _validate_templates(payload, existing=brand)
        try:
            updated = DeviceBrandRepository.update(db, brand, payload)
            db.commit()
            db.refresh(updated)
            return updated
        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Brand with this name and device type already exists",
            )

    @staticmethod
    def delete_brand(db: Session, brand_id: int) -> bool:
        brand = DeviceBrandRepository.get_by_id(db, brand_id)
        if not brand:
            return False
        DeviceBrandRepository.delete(db, brand)
        db.commit()
        return True
