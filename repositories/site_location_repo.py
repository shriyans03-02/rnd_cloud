from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, func
from typing import Optional, Tuple, List

from app.db.models.site_location import SiteLocation
from app.schemas.site_location import (
    SiteLocationCreate,
    SiteLocationUpdate,
)
from sqlalchemy.orm import Session, selectinload
from typing import List

from app.db.models.site_location import SiteLocation


class SiteLocationRepository:

    @staticmethod
    def get_all(db: Session) -> List[SiteLocation]:
        return (
            db.query(SiteLocation)
            .options(
                selectinload(SiteLocation.parent),
                selectinload(SiteLocation.site_hierarchy),
            )
            .all()
        )

    # -------------------- CREATE --------------------
    @staticmethod
    def create(db: Session, payload: SiteLocationCreate) -> SiteLocation:
        location = SiteLocation(
            name=payload.name,
            site_hierarchy_id=payload.site_hierarchy_id,
            parent_site_location_id=payload.parent_site_location_id,
            is_public=payload.is_public,
        )
        db.add(location)
        db.commit()

        #CRITICAL: re-fetch with relationships eagerly loaded
        return (
            db.query(SiteLocation)
            .options(
                joinedload(SiteLocation.parent),
                joinedload(SiteLocation.site_hierarchy),
            )
            .filter(SiteLocation.id == location.id)
            .one()
        )

    # -------------------- GET BY ID --------------------
    @staticmethod
    def get_by_id(db: Session, location_id: int) -> Optional[SiteLocation]:
        return (
            db.query(SiteLocation)
            .options(
                joinedload(SiteLocation.parent),
                joinedload(SiteLocation.site_hierarchy),
            )
            .filter(SiteLocation.id == location_id)
            .first()
        )

    # -------------------- UPDATE --------------------
    @staticmethod
    def update(
        db: Session,
        location: SiteLocation,
        payload: SiteLocationUpdate,
    ) -> SiteLocation:

        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(location, field, value)

        db.commit()

        # CRITICAL: re-fetch with eager loading
        return (
            db.query(SiteLocation)
            .options(
                joinedload(SiteLocation.parent),
                joinedload(SiteLocation.site_hierarchy),
            )
            .filter(SiteLocation.id == location.id)
            .one()
        )

    # -------------------- DELETE (HARD) --------------------
    @staticmethod
    def delete(db: Session, location: SiteLocation) -> None:
        db.delete(location)
        db.commit()

    # -------------------- DUPLICATE CHECK --------------------
    @staticmethod
    def exists_with_name(
        db: Session,
        name: str,
        site_hierarchy_id: int,
        parent_site_location_id: Optional[int],
        exclude_id: Optional[int] = None,
    ) -> bool:
        stmt = select(SiteLocation).where(
            func.lower(SiteLocation.name) == name.lower(),
            SiteLocation.site_hierarchy_id == site_hierarchy_id,
            SiteLocation.parent_site_location_id == parent_site_location_id,
        )

        if exclude_id:
            stmt = stmt.where(SiteLocation.id != exclude_id)

        return db.execute(stmt).scalars().first() is not None

    # -------------------- LIST --------------------
    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        site_hierarchy_id: int | None = None,
        parent_site_location_id: int | None = None,
        is_public: bool | None = None,
        is_active: bool | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        stmt = select(SiteLocation).options(
            joinedload(SiteLocation.parent),
            joinedload(SiteLocation.site_hierarchy),  # ✅ Add this
        )

        if search:
            stmt = stmt.where(func.lower(SiteLocation.name).like(f"%{search.lower()}%"))

        if site_hierarchy_id:
            stmt = stmt.where(SiteLocation.site_hierarchy_id == site_hierarchy_id)

        if parent_site_location_id:
            stmt = stmt.where(SiteLocation.parent_site_location_id == parent_site_location_id)

        if is_public is not None:
            stmt = stmt.where(SiteLocation.is_public == is_public)

        if is_active is not None:
            stmt = stmt.where(SiteLocation.is_active == is_active)

        total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar()
        stmt = stmt.offset(page * page_size).limit(page_size)
        rows = db.execute(stmt).scalars().all()
        return rows, total
