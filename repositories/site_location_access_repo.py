from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, insert, delete, func, or_
from app.db.models.associations import site_location_access
from app.db.models import SiteLocation, AccessGroup, SiteHierarchy
from typing import List, Set


class SiteLocationAccessRepository:

    @staticmethod
    def create(db: Session, site_location_id: int, access_group_id: int):
        db.execute(
            insert(site_location_access).values(
                site_location_id=site_location_id,
                access_group_id=access_group_id,
            )
        )
        db.commit()

    @staticmethod
    def exists(db: Session, site_location_id: int, access_group_id: int) -> bool:
        stmt = select(site_location_access).where(
            site_location_access.c.site_location_id == site_location_id,
            site_location_access.c.access_group_id == access_group_id,
        )
        return db.execute(stmt).first() is not None

    @staticmethod
    def bulk_create_for_access_group(
        db: Session,
        access_group_id: int,
        site_location_ids: List[int],
    ) -> List[int]:
        """Returns list of actually inserted site_location_ids."""
        if not site_location_ids:
            return []

        existing_stmt = select(site_location_access.c.site_location_id).where(
            site_location_access.c.access_group_id == access_group_id,
            site_location_access.c.site_location_id.in_(site_location_ids),
        )
        existing_ids: Set[int] = {row[0] for row in db.execute(existing_stmt).all()}

        new_ids = [sl_id for sl_id in site_location_ids if sl_id not in existing_ids]
        if not new_ids:
            return []

        db.execute(
            insert(site_location_access),
            [{"site_location_id": sl_id, "access_group_id": access_group_id} for sl_id in new_ids],
        )
        db.flush()  # ← service commits
        return new_ids

    @staticmethod
    def delete_single(db: Session, site_location_id: int, access_group_id: int) -> bool:
        """Returns True if a row was deleted."""
        result = db.execute(
            delete(site_location_access).where(
                site_location_access.c.site_location_id == site_location_id,
                site_location_access.c.access_group_id == access_group_id,
            )
        )
        db.flush()  # ← service commits
        return result.rowcount > 0

    @staticmethod
    def delete(db: Session, site_location_id: int, access_group_id: int):
        db.execute(
            delete(site_location_access).where(
                site_location_access.c.site_location_id == site_location_id,
                site_location_access.c.access_group_id == access_group_id,
            )
        )
        db.commit()

    @staticmethod
    def delete_all_for_site_location(db: Session, site_location_id: int):
        db.execute(
            delete(site_location_access).where(
                site_location_access.c.site_location_id == site_location_id,
            )
        )
        db.commit()

    @staticmethod
    def list(
        db: Session,
        search: str | None,
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

        all_access_groups = {ag.id: ag for ag in db.query(AccessGroup).all()}

        def all_access_group_ancestors_active(access_group_id: int) -> bool:
            current = all_access_groups.get(access_group_id)
            while current and current.parent_access_group_id is not None:
                parent = all_access_groups.get(current.parent_access_group_id)
                if parent is None or not parent.is_active:
                    return False
                current = parent
            return True

        query = (
            db.query(SiteLocation)
            .join(SiteLocation.site_hierarchy)
            .join(SiteLocation.access_groups)
            .options(
                joinedload(SiteLocation.access_groups),
                joinedload(SiteLocation.site_hierarchy),
            )
            .filter(SiteLocation.is_active.is_(True))
        )

        if search:
            query = query.filter(
                or_(
                    SiteHierarchy.name.ilike(f"%{search}%"),
                    AccessGroup.name.ilike(f"%{search}%"),
                )
            )

        query = query.distinct()
        total = query.count()
        locations = query.offset(page * page_size).limit(page_size).all()

        access_group_map: dict[int, dict] = {}
        for loc in locations:
            if not all_hierarchy_ancestors_active(loc.site_hierarchy_id):
                continue
            for ag in loc.access_groups:
                if not ag.is_active or not all_access_group_ancestors_active(ag.id):
                    continue
                if search:
                    hierarchy_matches = search.lower() in (loc.site_hierarchy.name or "").lower()
                    group_matches = search.lower() in ag.name.lower()
                    if not hierarchy_matches and not group_matches:
                        continue
                if ag.id not in access_group_map:
                    access_group_map[ag.id] = {
                        "access_group_id": ag.id,
                        "access_group_name": ag.name,
                        "site_locations": [],
                    }
                access_group_map[ag.id]["site_locations"].append({
                    "site_location_id": loc.id,
                    "site_location_name": loc.site_hierarchy.name if loc.site_hierarchy else None,
                })

        return list(access_group_map.values()), total