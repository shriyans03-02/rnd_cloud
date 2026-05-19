from typing import Optional, Tuple, List
from datetime import datetime

from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, func, and_, or_, text

from app.db.models import NormalizedData, Member, Camera
from app.db.models.site_hierarchy import SiteHierarchy
from app.db.models.site_location import SiteLocation


class NormalizedReportRepository:

    @staticmethod
    def _get_fully_active_hierarchy_ids(db: Session) -> set:
        cte = text("""
            WITH RECURSIVE hierarchy_check AS (
                SELECT id, is_active
                FROM site_hierarchies
                WHERE parent_site_hierarchy_id IS NULL

                UNION ALL

                SELECT sh.id, sh.is_active
                FROM site_hierarchies sh
                INNER JOIN hierarchy_check hc ON sh.parent_site_hierarchy_id = hc.id
                WHERE hc.is_active = TRUE
            )
            SELECT id FROM hierarchy_check WHERE is_active = TRUE
        """)
        return {row[0] for row in db.execute(cte).fetchall()}

    @staticmethod
    def list_site_locations(db: Session, search: Optional[str] = None) -> List[dict]:
        fully_active_ids = NormalizedReportRepository._get_fully_active_hierarchy_ids(db)
        if not fully_active_ids:
            return []

        stmt = (
            select(SiteLocation)
            .join(SiteHierarchy, SiteLocation.site_hierarchy_id == SiteHierarchy.id)
            .where(
                SiteLocation.is_active.is_(True),
                SiteHierarchy.id.in_(fully_active_ids),
            )
        )

        if search:
            s = search.strip().lower()
            stmt = stmt.where(func.lower(SiteHierarchy.name).like(f"%{s}%"))

        stmt = stmt.order_by(func.lower(SiteHierarchy.name).asc())
        stmt = stmt.options(selectinload(SiteLocation.site_hierarchy))

        locations = db.execute(stmt).scalars().all()
        return [{"id": loc.id, "name": loc.name} for loc in locations]

    @staticmethod
    def list_all_active(
        db: Session, search: Optional[str] = None, limit: int = 50
    ) -> List[Member]:
        stmt = select(Member).where(Member.is_active.is_(True))

        if search:
            s = search.strip().lower()
            stmt = stmt.where(
                or_(
                    func.lower(Member.first_name).like(f"%{s}%"),
                    func.lower(func.coalesce(Member.last_name, "")).like(f"%{s}%"),
                    func.lower(
                        func.concat(
                            Member.first_name, " ", func.coalesce(Member.last_name, "")
                        )
                    ).like(f"%{s}%"),
                )
            )

        stmt = stmt.order_by(
            func.lower(Member.first_name).asc(),
            func.lower(func.coalesce(Member.last_name, "")).asc(),
        ).limit(limit)

        return db.execute(stmt).scalars().all()

    @staticmethod
    def fetch_report(
        db: Session,
        page: int,
        page_size: int,
        start_ts: Optional[datetime],
        end_ts: Optional[datetime],
        site_location_id: Optional[int],
        member_id: Optional[int],
        movement_type: Optional[int],
        search: Optional[str],
        min_match: Optional[int],
    ) -> Tuple[List[NormalizedData], int]:

        filters = []

        if site_location_id is not None:
            filters.append(Camera.site_location_id == site_location_id)
        if member_id is not None:
            filters.append(NormalizedData.member_id == member_id)
        if movement_type is not None:
            filters.append(NormalizedData.movement_type == movement_type)
        if start_ts is not None:
            filters.append(NormalizedData.entry_ts >= start_ts)          # ← updated
        if end_ts is not None:
            filters.append(NormalizedData.entry_ts <= end_ts)            # ← updated
        if min_match is not None:
            filters.append(NormalizedData.average_match_value >= min_match)
        if search:
            filters.append(
                or_(
                    NormalizedData.guest_temp_id.ilike(f"%{search}%"),
                    Member.first_name.ilike(f"%{search}%"),
                    Member.last_name.ilike(f"%{search}%"),
                )
            )

        count_stmt = (
            select(func.count())
            .select_from(NormalizedData)
            .join(Camera, NormalizedData.camera_id == Camera.id)
            .join(Member, isouter=True)
        )
        if filters:
            count_stmt = count_stmt.where(and_(*filters))
        total = db.execute(count_stmt).scalar() or 0

        offset = (page - 1) * page_size
        query = (
            select(NormalizedData)
            .options(
                selectinload(NormalizedData.member),
                selectinload(NormalizedData.camera).selectinload(
                    Camera.site_location_rel
                ).selectinload(SiteLocation.site_hierarchy),
            )
            .join(Camera, NormalizedData.camera_id == Camera.id)
            .join(Member, isouter=True)
        )
        if filters:
            query = query.where(and_(*filters))

        query = (
            query.order_by(NormalizedData.entry_ts.desc())               # ← updated
            .offset(offset)
            .limit(page_size)
        )

        rows = db.execute(query).scalars().all()
        return rows, total