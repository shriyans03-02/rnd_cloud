from typing import List, Set
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, or_
from app.db.models import Camera, SiteLocation, SiteHierarchy
from app.schemas.camera import CameraCreate, CameraUpdate
from fastapi import HTTPException


class CameraRepository:

    @staticmethod
    def create(db: Session, payload: CameraCreate) -> Camera:
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
            site_location_id=payload.site_location_id,
        )
        db.add(camera)
        db.flush()       # ← service commits
        db.refresh(camera)
        return camera

    @staticmethod
    def get_by_id(db: Session, camera_id: int) -> Camera | None:
        return (
            db.query(Camera)
            .options(
                joinedload(Camera.site_location_rel)
                .joinedload(SiteLocation.site_hierarchy)
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
                .joinedload(SiteLocation.site_hierarchy)
            )
        )

        if search:
            stmt = stmt.filter(
                or_(
                    Camera.name.ilike(f"%{search}%"),
                    Camera.ip_address.ilike(f"%{search}%"),
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
            })

        return result, total

    @staticmethod
    def update(db: Session, camera: Camera, payload: CameraUpdate) -> Camera:
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

        for field, value in data.items():
            setattr(camera, field, value)

        db.flush()       # ← service commits
        db.refresh(camera)
        return camera

    @staticmethod
    def delete(db: Session, camera: Camera) -> None:
        db.delete(camera)
        db.flush()       # ← service commits

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