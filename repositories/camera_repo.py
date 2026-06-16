from typing import List, Optional, Set
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, or_
from fastapi import HTTPException

from app.db.models import Camera, DeviceBrand, SiteLocation, SiteHierarchy, NVR
from app.schemas.camera import CameraCreate, CameraUpdate


class CameraRepository:

    @staticmethod
    def create(
        db: Session,
        payload: CameraCreate,
        brand_name: str,
        brand_id: Optional[int],
        rtsp_url_template: Optional[str],
    ) -> Camera:
        existing = db.execute(
            select(Camera.id).where(Camera.ip_address == payload.ip_address)
        ).first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Camera with this IP address already exists",
            )

        camera = Camera(
            name=payload.name,
            ip_address=payload.ip_address,
            brand=brand_name,
            brand_id=brand_id,
            rtsp_url_template=rtsp_url_template,
            rtsp_port=payload.rtsp_port,
            rtsp_channel=payload.rtsp_channel,
            rtsp_subtype=payload.rtsp_subtype,
            rtsp_username=payload.rtsp_username,
            rtsp_password=payload.rtsp_password,
            nvr_id=payload.nvr_id,
            nvr_channel=payload.nvr_channel,
            site_location_id=payload.site_location_id,
        )
        db.add(camera)
        db.flush()
        db.refresh(camera)
        return camera

    @staticmethod
    def get_by_id(db: Session, camera_id: int) -> Camera | None:
        return (
            db.query(Camera)
            .options(
                joinedload(Camera.site_location_rel)
                .joinedload(SiteLocation.site_hierarchy),
                joinedload(Camera.nvr_rel),
                joinedload(Camera.brand_rel),
            )
            .filter(Camera.id == camera_id)
            .first()
        )

    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[list, int]:
        all_hierarchies = {s.id: s for s in db.query(SiteHierarchy).all()}

        def all_hierarchy_ancestors_active(site_hierarchy_id: int) -> bool:
            current = all_hierarchies.get(site_hierarchy_id)
            if current is None or not current.is_active:
                return False
            while current.parent_site_hierarchy_id is not None:
                parent = all_hierarchies.get(current.parent_site_hierarchy_id)
                if parent is None or not parent.is_active:
                    return False
                current = parent
            return True

        stmt = (
            db.query(Camera)
            .options(
                joinedload(Camera.site_location_rel)
                .joinedload(SiteLocation.site_hierarchy),
                joinedload(Camera.nvr_rel),
                joinedload(Camera.brand_rel),
            )
        )

        if search:
            like = f"%{search}%"
            stmt = stmt.outerjoin(Camera.nvr_rel).outerjoin(Camera.brand_rel).filter(
                or_(
                    Camera.name.ilike(like),
                    Camera.ip_address.ilike(like),
                    Camera.brand.ilike(like),
                    DeviceBrand.label.ilike(like),
                    NVR.name.ilike(like),
                )
            )

        total = stmt.count()
        cameras = (
            stmt.order_by(Camera.is_active.desc(), Camera.name.asc())
            .offset(page * page_size)
            .limit(page_size)
            .all()
        )

        result = []
        for camera in cameras:
            site_location_active = (
                all_hierarchy_ancestors_active(camera.site_location_rel.site_hierarchy_id)
                if camera.site_location_rel
                else False
            )
            result.append({
                **{c.name: getattr(camera, c.name) for c in Camera.__table__.columns},
                "site_location": (
                    camera.site_location_rel.site_hierarchy.name
                    if camera.site_location_rel and camera.site_location_rel.site_hierarchy
                    else None
                ),
                "site_location_active": site_location_active,
                "nvr_name": camera.nvr_rel.name if camera.nvr_rel else None,
                "brand_label": camera.brand_rel.label if camera.brand_rel else camera.brand,
            })

        return result, total

    @staticmethod
    def update(
        db: Session,
        camera: Camera,
        payload: CameraUpdate,
        brand_name: Optional[str] = None,
        brand_id: Optional[int] = None,
        rtsp_url_template: Optional[str] = None,
        set_brand_id: bool = False,
        set_rtsp_url_template: bool = False,
    ) -> Camera:
        data = payload.model_dump(exclude_unset=True)

        if "ip_address" in data:
            existing = db.execute(
                select(Camera.id).where(
                    Camera.ip_address == data["ip_address"],
                    Camera.id != camera.id,
                )
            ).first()
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail="Camera with this IP address already exists",
                )

        data.pop("brand_id", None)
        data.pop("brand", None)
        data.pop("rtsp_url_template", None)

        for field, value in data.items():
            setattr(camera, field, value)

        if brand_name is not None:
            camera.brand = brand_name
        if set_brand_id:
            camera.brand_id = brand_id
        if set_rtsp_url_template:
            camera.rtsp_url_template = rtsp_url_template

        db.flush()
        db.refresh(camera)
        return camera

    @staticmethod
    def delete(db: Session, camera: Camera) -> None:
        db.delete(camera)
        db.flush()

    @staticmethod
    def get_existing_ip_addresses(db: Session, ip_addresses: List[str]) -> Set[str]:
        rows = db.execute(
            select(Camera.ip_address).where(Camera.ip_address.in_(ip_addresses))
        ).scalars().all()
        return set(rows)

    @staticmethod
    def bulk_create(db: Session, cameras: List) -> List:
        db.add_all(cameras)
        db.flush()
        for camera in cameras:
            db.refresh(camera)
        return cameras
