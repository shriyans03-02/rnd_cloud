from typing import Optional, List, Set
from datetime import datetime, timedelta

from sqlalchemy.orm import Session, selectinload
from sqlalchemy import select, and_, func, or_, text, exists

from app.db.models import NormalizedData, Member, Camera
from app.db.models.access_group import AccessGroup

from typing import Optional, List
from datetime import datetime

from sqlalchemy.orm import Session, selectinload
from app.db.models.site_hierarchy import SiteHierarchy
from app.db.models.site_location import SiteLocation
from app.db.models.member_embedding import MemberEmbedding



class MemberTracingRepository:

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
    def get_member_details(db: Session, member_id: int) -> Optional[Member]:
        stmt = (
            select(Member)
            .options(
                selectinload(Member.department),
                # Load access groups → their site_locations
                selectinload(Member.access_groups).selectinload(
                    AccessGroup.site_locations
                ),
            )
            .where(Member.id == member_id)
        )
        return db.execute(stmt).scalars().first()

    @staticmethod
    def get_allowed_site_location_ids(member: Member) -> Set[int]:
        """
        Returns the set of site_location_ids the member is authorized to access,
        derived from all their access groups and those groups' site_locations.
        """
        allowed: Set[int] = set()
        for group in (member.access_groups or []):
            for site_loc in (group.site_locations or []):
                allowed.add(site_loc.id)
        return allowed

    @staticmethod
    def get_member_events(
        db: Session,
        member_id: int,
        date: datetime,
        entry_ts: Optional[datetime],
        exit_ts: Optional[datetime],
    ) -> List[NormalizedData]:
        """
        Fetch all NormalizedData rows for the member within the given window.
          - date      : calendar date used as fallback day boundary (midnight–midnight)
          - entry_ts  : explicit window start (overrides date start-of-day)
          - exit_ts   : explicit window end   (overrides date end-of-day)
        """
        day_start = datetime(date.year, date.month, date.day, 0, 0, 0,
                             tzinfo=date.tzinfo)
        day_end   = day_start + timedelta(days=1)

        window_start = entry_ts if entry_ts is not None else day_start
        window_end   = exit_ts  if exit_ts  is not None else day_end

        filters = [
            NormalizedData.member_id == member_id,
            NormalizedData.entry_ts  >= window_start,
            NormalizedData.entry_ts  <= window_end,
        ]

        stmt = (
            select(NormalizedData)
            .options(
                selectinload(NormalizedData.camera).selectinload(
                    Camera.site_location_rel
                ),
            )
            .join(Camera, NormalizedData.camera_id == Camera.id)
            .where(and_(*filters))
            .order_by(NormalizedData.entry_ts.asc())
        )

        results = db.execute(stmt).scalars().all()
        print(f"[TRACING] member={member_id} window={window_start} → {window_end} | rows={len(results)}")
        for r in results:
            print(f"  entry_ts={r.entry_ts}  tzinfo={r.entry_ts.tzinfo}")
        return results
    


    @staticmethod
    def list_all_active(
        db: Session, search: Optional[str] = None, limit: int = 50
    ) -> List[Member]:
        stmt = (
            select(Member)
            .where(Member.is_active.is_(True))
            .where(
                exists(
                    select(1).where(
                        and_(
                            MemberEmbedding.member_id == Member.id,
                            MemberEmbedding.body_embedding.isnot(None),
                            MemberEmbedding.face_embedding.isnot(None),
                            MemberEmbedding.back_body_embedding.isnot(None),
                            MemberEmbedding.body_embeddings_raw.isnot(None),
                            MemberEmbedding.face_embeddings_raw.isnot(None),
                            MemberEmbedding.back_body_embeddings_raw.isnot(None),
                        )
                    )
                )
            )
        )

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
    def list_site_locations(db: Session, search: Optional[str] = None) -> List[dict]:
        fully_active_ids = MemberTracingRepository._get_fully_active_hierarchy_ids(db)
        if not fully_active_ids:
            return []

        stmt = (
            select(SiteLocation)
            .join(SiteHierarchy, SiteLocation.site_hierarchy_id == SiteHierarchy.id)
            .where(
                SiteLocation.is_active.is_(True),
                SiteHierarchy.id.in_(fully_active_ids),
                # ✅ NEW: only locations that have at least one active camera
                select(Camera.id)
                .where(
                    Camera.site_location_id == SiteLocation.id,
                    Camera.is_active.is_(True),
                )
                .correlate(SiteLocation)
                .exists()
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
    def get_cameras_by_site_location(db: Session, site_location_id: int) -> List[dict]:
        stmt = (
            select(Camera)
            .where(
                Camera.site_location_id == site_location_id,
                Camera.is_active.is_(True),
            )
            .order_by(Camera.name.asc())
        )
        cameras = db.execute(stmt).scalars().all()
        return [{"id": c.id, "name": c.name} for c in cameras]