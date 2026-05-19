from typing import Dict, List, Optional, Set

from sqlalchemy import and_, exists, func, or_, select, text
from sqlalchemy.orm import Session, selectinload

from app.db.models import Camera, Member, NormalizedData
from app.db.models.member_embedding import MemberEmbedding
from app.db.models.site_hierarchy import SiteHierarchy
from app.db.models.site_location import SiteLocation


class TrackingConsoleRepository:

    # ─────────────────────────────────────────────────────────────────────────
    #  Internal helpers
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _get_fully_active_hierarchy_ids(db: Session) -> Set[int]:
        """
        Recursively walks the site_hierarchies tree and returns the IDs of
        every node whose entire ancestor chain (up to the root) is active.
        """
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
    def _active_member_with_embeddings_filter():
        """
        Returns the SQLAlchemy EXISTS subquery that checks all 6 embedding
        fields are populated — used wherever member validation is needed.
        """
        return exists(
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

    # ─────────────────────────────────────────────────────────────────────────
    #  Members
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def list_members(
        db: Session,
        search: Optional[str] = None,
        limit: int = 50,
    ) -> List[Member]:
        """
        Returns active members that have all 6 embedding fields populated.
        """
        stmt = (
            select(Member)
            .where(
                Member.is_active.is_(True),
                TrackingConsoleRepository._active_member_with_embeddings_filter(),
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

    # ─────────────────────────────────────────────────────────────────────────
    #  Site locations
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def list_site_locations(
        db: Session,
        search: Optional[str] = None,
    ) -> List[Dict]:
        """
        Returns active site locations that:
          - belong to a fully-active hierarchy (recursive ancestor check)
          - have at least one active camera assigned
        """
        fully_active_ids = TrackingConsoleRepository._get_fully_active_hierarchy_ids(db)
        if not fully_active_ids:
            return []

        stmt = (
            select(SiteLocation)
            .join(SiteHierarchy, SiteLocation.site_hierarchy_id == SiteHierarchy.id)
            .where(
                SiteLocation.is_active.is_(True),
                SiteHierarchy.id.in_(fully_active_ids),
                select(Camera.id)
                .where(
                    Camera.site_location_id == SiteLocation.id,
                    Camera.is_active.is_(True),
                )
                .correlate(SiteLocation)
                .exists(),
            )
            .options(selectinload(SiteLocation.site_hierarchy))
        )

        if search:
            s = search.strip().lower()
            stmt = stmt.where(func.lower(SiteHierarchy.name).like(f"%{s}%"))

        stmt = stmt.order_by(func.lower(SiteHierarchy.name).asc())
        locations = db.execute(stmt).scalars().all()
        return [{"id": loc.id, "name": loc.name} for loc in locations]

    # ─────────────────────────────────────────────────────────────────────────
    #  Resolve: cameras for members
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_live_cameras_for_members(
        db: Session,
        member_ids: List[int],
    ) -> List[Dict]:
        """
        1. Filters member_ids to those that are active and have complete embeddings.
        2. Queries NormalizedData for rows where:
             - member_id in the validated list
             - movement_type == 1  (entered camera frame)
             - exit_ts IS NULL     (still live)
        Returns a deduplicated list of cameras.
        """
        # ── Validate member_ids ───────────────────────────────────────────────
        valid_stmt = (
            select(Member.id)
            .where(
                Member.id.in_(member_ids),
                Member.is_active.is_(True),
                TrackingConsoleRepository._active_member_with_embeddings_filter(),
            )
        )
        valid_ids = [row[0] for row in db.execute(valid_stmt).fetchall()]

        if not valid_ids:
            return []

        # ── Resolve live cameras ──────────────────────────────────────────────
        stmt = (
            select(NormalizedData)
            .options(selectinload(NormalizedData.camera))
            .where(
                and_(
                    NormalizedData.member_id.in_(valid_ids),
                    NormalizedData.movement_type == 1,
                    NormalizedData.exit_ts.is_(None),
                )
            )
        )
        rows = db.execute(stmt).scalars().all()

        seen: Set[int] = set()
        cameras: List[Dict] = []
        for row in rows:
            if row.camera_id not in seen and row.camera is not None:
                seen.add(row.camera_id)
                cameras.append(
                    {
                        "id": row.camera.id,
                        "name": row.camera.name,
                        "location": row.camera.site_location,  # computed property → str | None
                    }
                )

        return cameras

    # ─────────────────────────────────────────────────────────────────────────
    #  Resolve: cameras for locations
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def get_cameras_for_locations(
        db: Session,
        site_location_ids: List[int],
    ) -> List[Dict]:
        """
        1. Filters site_location_ids to those that pass the fully-active-hierarchy
           + active-camera check.
        2. Fetches all active cameras for the validated location IDs.
        Returns a deduplicated, name-sorted list of cameras.
        """
        # ── Validate site_location_ids ────────────────────────────────────────
        fully_active_ids = TrackingConsoleRepository._get_fully_active_hierarchy_ids(db)
        if not fully_active_ids:
            return []

        valid_loc_stmt = (
            select(SiteLocation.id)
            .join(SiteHierarchy, SiteLocation.site_hierarchy_id == SiteHierarchy.id)
            .where(
                SiteLocation.id.in_(site_location_ids),
                SiteLocation.is_active.is_(True),
                SiteHierarchy.id.in_(fully_active_ids),
                select(Camera.id)
                .where(
                    Camera.site_location_id == SiteLocation.id,
                    Camera.is_active.is_(True),
                )
                .correlate(SiteLocation)
                .exists(),
            )
        )
        valid_location_ids = [row[0] for row in db.execute(valid_loc_stmt).fetchall()]

        if not valid_location_ids:
            return []

        # ── Fetch cameras ─────────────────────────────────────────────────────
        stmt = (
            select(Camera)
            .where(
                and_(
                    Camera.site_location_id.in_(valid_location_ids),
                    Camera.is_active.is_(True),
                )
            )
            .order_by(Camera.name.asc())
        )
        cam_rows = db.execute(stmt).scalars().all()

        seen: Set[int] = set()
        cameras: List[Dict] = []
        for c in cam_rows:
            if c.id not in seen:
                seen.add(c.id)
                cameras.append(
                    {
                        "id": c.id,
                        "name": c.name,
                        "location": c.site_location,  # computed property → str | None
                    }
                )

        return cameras