from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session, joinedload

from app.db.models.device_brand import DeviceBrand
from app.db.models.nvr import NVR
from app.schemas.nvr import NVRCreate, NVRUpdate


class NVRRepository:

    @staticmethod
    def create(
        db: Session,
        payload: NVRCreate,
        brand_name: str,
        brand_id: Optional[int],
        playback_rtsp_template: Optional[str] = None,
    ) -> NVR:
        existing = db.execute(
            select(NVR.id).where(
                NVR.ip_address == payload.ip_address,
                NVR.port == payload.port,
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="NVR with this IP address and port already exists",
            )

        nvr = NVR(
            name=payload.name,
            ip_address=payload.ip_address,
            port=payload.port,
            brand=brand_name,
            brand_id=brand_id,
            username=payload.username,
            password=payload.password,
            playback_rtsp_template=playback_rtsp_template,
            stream_key=payload.stream_key,
            is_active=True,
        )
        db.add(nvr)
        db.flush()
        db.refresh(nvr)
        return nvr

    @staticmethod
    def get_by_id(db: Session, nvr_id: int) -> Optional[NVR]:
        return (
            db.query(NVR)
            .options(joinedload(NVR.brand_rel))
            .filter(NVR.id == nvr_id)
            .first()
        )

    @staticmethod
    def list(
        db: Session,
        search: Optional[str] = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[List[NVR], int]:
        stmt = select(NVR).options(joinedload(NVR.brand_rel))

        if search:
            search_term = f"%{search.lower()}%"
            stmt = stmt.outerjoin(NVR.brand_rel).where(
                or_(
                    func.lower(NVR.name).like(search_term),
                    func.lower(NVR.ip_address).like(search_term),
                    func.lower(NVR.brand).like(search_term),
                    func.lower(DeviceBrand.label).like(search_term),
                )
            )

        total = db.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar() or 0

        stmt = (
            stmt.order_by(NVR.is_active.desc(), func.lower(NVR.name).asc())
            .offset(page * page_size)
            .limit(page_size)
        )
        nvrs = db.execute(stmt).scalars().unique().all()
        return nvrs, total

    @staticmethod
    def list_all_active(db: Session, search: Optional[str] = None) -> List[NVR]:
        stmt = select(NVR).options(joinedload(NVR.brand_rel)).where(NVR.is_active.is_(True))

        if search:
            search_term = f"%{search.lower()}%"
            stmt = stmt.outerjoin(NVR.brand_rel).where(
                or_(
                    func.lower(NVR.name).like(search_term),
                    func.lower(NVR.ip_address).like(search_term),
                    func.lower(NVR.brand).like(search_term),
                    func.lower(DeviceBrand.label).like(search_term),
                )
            )

        stmt = stmt.order_by(func.lower(NVR.name).asc())
        return db.execute(stmt).scalars().unique().all()

    @staticmethod
    def update(
        db: Session,
        nvr: NVR,
        payload: NVRUpdate,
        brand_name: Optional[str] = None,
        brand_id: Optional[int] = None,
        playback_rtsp_template: Optional[str] = None,
        set_brand_id: bool = False,
        set_playback_rtsp_template: bool = False,
    ) -> NVR:
        data = payload.model_dump(exclude_unset=True)

        new_ip = data.get("ip_address", nvr.ip_address)
        new_port = data.get("port", nvr.port)
        if "ip_address" in data or "port" in data:
            existing = db.execute(
                select(NVR.id).where(
                    NVR.ip_address == new_ip,
                    NVR.port == new_port,
                    NVR.id != nvr.id,
                )
            ).first()
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail="NVR with this IP address and port already exists",
                )

        data.pop("brand_id", None)
        data.pop("brand", None)
        data.pop("playback_rtsp_template", None)

        for field, value in data.items():
            setattr(nvr, field, value)

        if brand_name is not None:
            nvr.brand = brand_name
        if set_brand_id:
            nvr.brand_id = brand_id
        if set_playback_rtsp_template:
            nvr.playback_rtsp_template = playback_rtsp_template

        db.flush()
        db.refresh(nvr)
        return nvr

    @staticmethod
    def delete(db: Session, nvr: NVR) -> None:
        db.delete(nvr)
        db.flush()
