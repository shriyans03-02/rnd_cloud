from sqlalchemy.orm import Session, joinedload, selectinload
from sqlalchemy import select, func, text
from typing import List, Optional, Set, Dict, Tuple
from fastapi import HTTPException, status
from sqlalchemy.exc import IntegrityError

from app.db.models.site_hierarchy import SiteHierarchy
from app.db.models.site_location import SiteLocation
from app.schemas.site_hierarchy import SiteHierarchyCreate, SiteHierarchyUpdate


class SiteHierarchyRepository:

    # ── DUPLICATION CHECK ─────────────────────────────────────────────────────

    @staticmethod
    def exists_with_name(
        db: Session,
        name: str,
        exclude_id: int | None = None,
    ) -> bool:
        stmt = select(SiteHierarchy).where(
            func.lower(SiteHierarchy.name) == name.lower(),
        )
        if exclude_id:
            stmt = stmt.where(SiteHierarchy.id != exclude_id)
        return db.execute(stmt).scalars().first() is not None

    # ── CREATE ────────────────────────────────────────────────────────────────

    @staticmethod
    def create(db: Session, payload: SiteHierarchyCreate) -> SiteHierarchy:
        site = SiteHierarchy(
            name=payload.name,
            parent_site_hierarchy_id=payload.parent_site_hierarchy_id,
        )
        db.add(site)

        try:
            db.flush()

            # If parent exists → remove it from SiteLocation (it's no longer a leaf)
            if payload.parent_site_hierarchy_id:
                db.query(SiteLocation).filter(
                    SiteLocation.site_hierarchy_id == payload.parent_site_hierarchy_id
                ).delete(synchronize_session=False)

            # New node is a leaf → create its SiteLocation
            location = SiteLocation(
                site_hierarchy_id=site.id,
                is_active=True,
                is_protected=bool(getattr(payload, "is_protected", False)),
                is_public=bool(getattr(payload, "is_public", False)),
            )
            db.add(location)
            db.flush()

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Site name already exists",
            )

        return (
            db.query(SiteHierarchy)
            .options(joinedload(SiteHierarchy.parent))
            .filter(SiteHierarchy.id == site.id)
            .one()
        )

    # ── READ ──────────────────────────────────────────────────────────────────

    @staticmethod
    def get_by_id(db: Session, site_id: int) -> SiteHierarchy | None:
        return db.get(SiteHierarchy, site_id)

    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[list[SiteHierarchy], int]:
        stmt = (
            select(SiteHierarchy)
            .options(joinedload(SiteHierarchy.parent))
            .where(SiteHierarchy.is_active.is_(True))
        )
        if search:
            stmt = stmt.where(
                func.lower(SiteHierarchy.name).like(f"%{search.lower()}%")
            )
        total = db.execute(select(func.count()).select_from(stmt.subquery())).scalar()
        stmt = stmt.order_by(func.lower(SiteHierarchy.name).asc())
        stmt = stmt.offset(page * page_size).limit(page_size)
        sites = db.execute(stmt).scalars().all()
        return sites, total

    @staticmethod
    def list_all(db: Session, search: str = None) -> List[SiteHierarchy]:
        stmt = select(SiteHierarchy).where(SiteHierarchy.is_active.is_(True))
        if search:
            stmt = stmt.where(
                func.lower(SiteHierarchy.name).like(f"%{search.lower()}%")
            )
        return db.execute(stmt).scalars().all()

    @staticmethod
    def list_all_active(db: Session) -> List[SiteHierarchy]:
        stmt = (
            select(SiteHierarchy)
            .where(SiteHierarchy.is_active.is_(True))
            .order_by(SiteHierarchy.name.asc())
        )
        return db.execute(stmt).scalars().all()

    @staticmethod
    def list_hierarchies_with_locations(
        db: Session,
        include_inactive: bool = False,
        site_hierarchy_id: Optional[int] = None,
    ) -> List[SiteHierarchy]:
        stmt = select(SiteHierarchy).options(
            selectinload(SiteHierarchy.site_locations).selectinload(SiteLocation.cameras)
        )
        if not include_inactive:
            stmt = stmt.where(SiteHierarchy.is_active.is_(True))
        if site_hierarchy_id is not None:
            stmt = stmt.where(SiteHierarchy.id == site_hierarchy_id)
        stmt = stmt.order_by(SiteHierarchy.id)
        return db.execute(stmt).scalars().all()

    @staticmethod
    def get_fully_active_hierarchy_ids(db: Session) -> set:
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

    # ── TREE DATA FOR LOCK/DELETABLE COMPUTATION ──────────────────────────────

    @staticmethod
    def fetch_all_nodes_raw(db: Session) -> List[Tuple[int, Optional[int]]]:
        """Returns list of (id, parent_site_hierarchy_id) for ALL nodes."""
        return db.query(SiteHierarchy.id, SiteHierarchy.parent_site_hierarchy_id).all()

    @staticmethod
    def fetch_direct_locations_map(db: Session) -> Tuple[Dict[int, List[int]], Dict[int, Dict]]:
        loc_rows = db.query(
            SiteLocation.id,
            SiteLocation.site_hierarchy_id,
            SiteLocation.is_public,
            SiteLocation.is_protected,
            SiteLocation.is_active,  # ← add this
        ).all()

        direct_locations_map: Dict[int, List[int]] = {}
        location_meta_map: Dict[int, Dict] = {}

        for loc_id, hierarchy_id, is_public, is_protected, is_active in loc_rows:  # ← unpack
            if hierarchy_id is None:
                continue
            direct_locations_map.setdefault(hierarchy_id, []).append(loc_id)
            location_meta_map[hierarchy_id] = {
                "is_public": is_public,
                "is_protected": is_protected,
                "is_leaf": is_active,  # ← active SiteLocation = leaf node
            }

        return direct_locations_map, location_meta_map

    @staticmethod
    def fetch_used_location_ids(db: Session) -> Set[int]:
        """Returns IDs of SiteLocations currently assigned to a Camera."""
        from app.db.models import Camera
        used_rows = (
            db.query(Camera.site_location_id)
            .filter(Camera.site_location_id.isnot(None))
            .distinct()
            .all()
        )
        return {r[0] for r in used_rows if r[0] is not None}

    # ── UPDATE ────────────────────────────────────────────────────────────────

    @staticmethod
    def sync_leaf_state(db: Session, hierarchy_id: int):
        """Ensure a node's SiteLocation reflects whether it is a leaf or not."""
        has_children = (
            db.query(SiteHierarchy.id)
            .filter(SiteHierarchy.parent_site_hierarchy_id == hierarchy_id)
            .first()
        ) is not None

        location = (
            db.query(SiteLocation)
            .filter(SiteLocation.site_hierarchy_id == hierarchy_id)
            .first()
        )

        if has_children:
            if location and location.is_active:
                location.is_active = False
        else:
            if not location:
                db.add(SiteLocation(
                    name="Auto",
                    site_hierarchy_id=hierarchy_id,
                    is_active=True,
                ))
            elif not location.is_active:
                location.is_active = True

    @staticmethod
    def update(
        db: Session,
        site: SiteHierarchy,
        payload: SiteHierarchyUpdate,
    ) -> SiteHierarchy:
        if SiteHierarchyRepository.exists_with_name(db, payload.name, exclude_id=site.id):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Site name already exists",
            )

        old_parent_id = site.parent_site_hierarchy_id
        data = payload.model_dump(exclude_unset=True)

        for field in ["name", "parent_site_hierarchy_id", "is_active"]:
            if field in data:
                setattr(site, field, data[field])

        db.flush()

        if "is_public" in data or "is_protected" in data:
            for loc in db.query(SiteLocation).filter(
                SiteLocation.site_hierarchy_id == site.id
            ).all():
                if "is_public" in data:
                    loc.is_public = data["is_public"]
                if "is_protected" in data:
                    loc.is_protected = data["is_protected"]

        SiteHierarchyRepository.sync_leaf_state(db, site.id)

        if old_parent_id and old_parent_id != site.parent_site_hierarchy_id:
            SiteHierarchyRepository.sync_leaf_state(db, old_parent_id)

        if site.parent_site_hierarchy_id:
            SiteHierarchyRepository.sync_leaf_state(db, site.parent_site_hierarchy_id)

        db.flush()
        db.refresh(site)

        return (
            db.query(SiteHierarchy)
            .options(joinedload(SiteHierarchy.parent))
            .filter(SiteHierarchy.id == site.id)
            .one()
        )

    # ── DELETE ────────────────────────────────────────────────────────────────

    @staticmethod
    def build_children_map(
        raw_nodes: List[Tuple[int, Optional[int]]]
    ) -> Dict[int, List[int]]:
        """Build {parent_id -> [child_id, ...]} from raw node tuples."""
        children_map: Dict[int, List[int]] = {}
        for nid, pid in raw_nodes:
            children_map.setdefault(nid, [])
            if pid is not None:
                children_map.setdefault(pid, []).append(nid)
        return children_map

    @staticmethod
    def collect_descendant_ids(
        node_id: int,
        children_map: Dict[int, List[int]],
    ) -> Set[int]:
        """Recursively collect node_id + all descendant IDs."""
        ids: Set[int] = {node_id}
        for child_id in children_map.get(node_id, []):
            ids |= SiteHierarchyRepository.collect_descendant_ids(child_id, children_map)
        return ids

    @staticmethod
    def delete_subtree(db: Session, node_id: int, children_map: Dict[int, List[int]]):
        """Recursively delete children before parent (leaf-first) to respect FK constraints."""
        for child_id in children_map.get(node_id, []):
            SiteHierarchyRepository.delete_subtree(db, child_id, children_map)

        # Get parent_id before deleting
        node = db.get(SiteHierarchy, node_id)
        parent_id = node.parent_site_hierarchy_id if node else None

        # Delete SiteLocation first
        db.query(SiteLocation).filter(
            SiteLocation.site_hierarchy_id == node_id
        ).delete(synchronize_session="fetch")

        # Delete the node
        if node:
            db.delete(node)

        db.flush()

        # After deletion, check if parent has no remaining children → restore it as leaf
        if parent_id:
            remaining_children = (
                db.query(SiteHierarchy.id)
                .filter(SiteHierarchy.parent_site_hierarchy_id == parent_id)
                .first()
            )
            if not remaining_children:
                existing_location = (
                    db.query(SiteLocation)
                    .filter(SiteLocation.site_hierarchy_id == parent_id)
                    .first()
                )
                if not existing_location:
                    db.add(SiteLocation(
                        site_hierarchy_id=parent_id,
                        is_active=True,
                        is_public=False,
                        is_protected=False,
                    ))
                    db.flush()