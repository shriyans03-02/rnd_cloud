from typing import List, Optional

from fastapi import HTTPException
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.db.models.device_brand import DeviceBrand
from app.schemas.device_brand import DeviceBrandCreate, DeviceBrandUpdate


class DeviceBrandRepository:

    @staticmethod
    def create(db: Session, payload: DeviceBrandCreate) -> DeviceBrand:
        existing = db.execute(
            select(DeviceBrand.id).where(
                func.lower(DeviceBrand.name) == payload.name.lower(),
                DeviceBrand.device_type == payload.device_type,
            )
        ).first()
        if existing:
            raise HTTPException(
                status_code=400,
                detail="Brand with this name and device type already exists",
            )

        brand = DeviceBrand(**payload.model_dump())
        db.add(brand)
        db.flush()
        db.refresh(brand)
        return brand

    @staticmethod
    def get_by_id(db: Session, brand_id: int) -> Optional[DeviceBrand]:
        return db.get(DeviceBrand, brand_id)

    @staticmethod
    def get_by_name_type(db: Session, name: str, device_type: str) -> Optional[DeviceBrand]:
        if not name or not device_type:
            return None
        stmt = select(DeviceBrand).where(
            func.lower(DeviceBrand.name) == str(name).strip().lower(),
            DeviceBrand.device_type == str(device_type).strip().lower(),
        )
        return db.execute(stmt).scalars().first()

    @staticmethod
    def list(
        db: Session,
        device_type: Optional[str] = None,
        search: Optional[str] = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[List[DeviceBrand], int]:
        stmt = select(DeviceBrand)

        if device_type:
            stmt = stmt.where(DeviceBrand.device_type == device_type.strip().lower())

        if search:
            search_term = f"%{search.lower()}%"
            stmt = stmt.where(
                or_(
                    func.lower(DeviceBrand.name).like(search_term),
                    func.lower(DeviceBrand.label).like(search_term),
                    func.lower(DeviceBrand.device_type).like(search_term),
                )
            )

        total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar() or 0
        rows = db.execute(
            stmt.order_by(
                DeviceBrand.is_active.desc(),
                DeviceBrand.device_type.asc(),
                func.lower(DeviceBrand.label).asc(),
            )
            .offset(page * page_size)
            .limit(page_size)
        ).scalars().all()
        return rows, total

    @staticmethod
    def list_all_active(db: Session, device_type: Optional[str] = None) -> List[DeviceBrand]:
        stmt = select(DeviceBrand).where(DeviceBrand.is_active.is_(True))
        if device_type:
            stmt = stmt.where(DeviceBrand.device_type == device_type.strip().lower())
        return db.execute(
            stmt.order_by(DeviceBrand.device_type.asc(), func.lower(DeviceBrand.label).asc())
        ).scalars().all()

    @staticmethod
    def update(db: Session, brand: DeviceBrand, payload: DeviceBrandUpdate) -> DeviceBrand:
        data = payload.model_dump(exclude_unset=True)

        new_name = data.get("name", brand.name)
        new_device_type = data.get("device_type", brand.device_type)
        if "name" in data or "device_type" in data:
            existing = db.execute(
                select(DeviceBrand.id).where(
                    func.lower(DeviceBrand.name) == str(new_name).lower(),
                    DeviceBrand.device_type == new_device_type,
                    DeviceBrand.id != brand.id,
                )
            ).first()
            if existing:
                raise HTTPException(
                    status_code=400,
                    detail="Brand with this name and device type already exists",
                )

        for field, value in data.items():
            setattr(brand, field, value)

        db.flush()
        db.refresh(brand)
        return brand

    @staticmethod
    def delete(db: Session, brand: DeviceBrand) -> None:
        db.delete(brand)
        db.flush()
