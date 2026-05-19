from sqlalchemy.exc import IntegrityError
from fastapi import HTTPException, status

from app.db.session import SessionLocal
from app.repositories.site_location_repo import SiteLocationRepository
from app.schemas.site_location import (
    SiteLocationCreate,
    SiteLocationUpdate,
)
from app.db.models.site_location import SiteLocation
from app.db.models.site_hierarchy import SiteHierarchy
from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select,func


class SiteLocationService:

    @staticmethod
    def get_tree(db: Session, search: str | None = None):
        locations = (
            db.query(SiteLocation)
            .options(joinedload(SiteLocation.site_hierarchy))
            .filter(SiteLocation.is_active == True)
            .all()
        )

        if search:
            locations = [
                loc for loc in locations
                if search.lower() in loc.name.lower()
            ]

        lookup = {
            loc.id: {
                "id": loc.id,
                "name": loc.name,
                "site_hierarchy_id": loc.site_hierarchy_id,
                "site_hierarchy_name": loc.site_hierarchy.name if loc.site_hierarchy else None,
                "parent_site_location_id": loc.parent_site_location_id,
                "is_public": loc.is_public,
                "is_active": loc.is_active,
                "children": []
            }
            for loc in locations
        }

        root = []

        for loc in lookup.values():
            parent_id = loc["parent_site_location_id"]

            if parent_id and parent_id in lookup:
                lookup[parent_id]["children"].append(loc)
            else:
                root.append(loc)

        return root
    
    # CREATE
    @staticmethod
    def create_site_location(payload: SiteLocationCreate) -> SiteLocation:
        db = SessionLocal()
        try:
            return SiteLocationRepository.create(db, payload)

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Site location already exists"
            )

        finally:
            db.close()

    # GET by ID
    @staticmethod
    def get_site_location(site_location_id: int) -> SiteLocation | None:
        db = SessionLocal()
        try:
            return SiteLocationRepository.get_by_id(db, site_location_id)
        finally:
            db.close()

    # LIST (search + pagination)
    @staticmethod
    def list_site_locations(
        search: str | None = None,
        site_hierarchy_id: int | None = None,
        parent_site_location_id: int | None = None,
        is_public: bool | None = None,
        is_active: bool | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db = SessionLocal()
        try:
            return SiteLocationRepository.list(
                db=db,
                search=search,
                site_hierarchy_id=site_hierarchy_id,
                parent_site_location_id=parent_site_location_id,
                is_public=is_public,
                is_active=is_active,
                page=page,
                page_size=page_size,
            )
        finally:
            db.close() 

    # UPDATE
    @staticmethod
    def update_site_location(
        site_location_id: int,
        payload: SiteLocationUpdate,
    ) -> SiteLocation:
        db = SessionLocal()
        try:
            site_location = SiteLocationRepository.get_by_id(db, site_location_id)
            if not site_location:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Site location not found"
                )

            return SiteLocationRepository.update(db, site_location, payload)

        except IntegrityError:
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Site location name already exists"
            )

        finally:
            db.close()

    # HARD DELETE
    @staticmethod
    def delete_site_location(site_location_id: int) -> bool:
        db = SessionLocal()
        try:
            site_location = SiteLocationRepository.get_by_id(db, site_location_id)
            if not site_location:
                return False

            SiteLocationRepository.delete(db, site_location)
            return True
        finally:
            db.close()

    @staticmethod
    def list_all_site_locations(db: Session):
        # fetch active site locations
        stmt = select(SiteLocation).where(SiteLocation.is_active.is_(True))
        locations = db.execute(stmt).scalars().all()

        # total count
        total = db.execute(
            select(func.count(SiteLocation.id))
            .where(SiteLocation.is_active.is_(True))
        ).scalar()

        return locations, total
    
    @staticmethod
    def get_full_tree_from_site_location(site_location_id: int):
        db = SessionLocal()
        try:
            # 1️⃣ Fetch the site location
            site_location = db.query(SiteLocation).filter(
                SiteLocation.id == site_location_id,
                SiteLocation.is_active.is_(True)
            ).first()
            if not site_location:
                return []

            # 2️⃣ Fetch the hierarchy
            hierarchy = db.query(SiteHierarchy).filter(
                SiteHierarchy.id == site_location.site_hierarchy_id,
                SiteHierarchy.is_active.is_(True)
            ).first()
            if not hierarchy:
                return []

            # 3️⃣ Fetch all active site locations under this hierarchy
            locations = db.query(SiteLocation).options(
                joinedload(SiteLocation.cameras)  # preload cameras
            ).filter(
                SiteLocation.site_hierarchy_id == hierarchy.id,
                SiteLocation.is_active.is_(True)
            ).all()

            # 4️⃣ Build the tree
            return SiteLocationService.build_tree(locations)

        finally:
            db.close()

    @staticmethod
    def build_tree(nodes: list):
        """
        Build nested tree from flat site_locations list
        Each node contains its cameras
        """
        def serialize(node):
            return {
                "id": node.id,
                "name": node.site_hierarchy.name if node.site_hierarchy else None,
                "site_hierarchy": {
                    "id": node.site_hierarchy.id,
                    "name": node.site_hierarchy.name,
                } if node.site_hierarchy else None, 
                "is_public": node.is_public,
                "is_active": node.is_active,
                "cameras": [
                    {
                        "id": cam.id,
                        "name": cam.name,
                        "ip_address": cam.ip_address, 
                        "is_active": cam.is_active,
                    } for cam in getattr(node, "cameras", [])
                ],
                "children": [],
            }

        node_map = {node.id: serialize(node) for node in nodes}
        roots = []

        for node in node_map.values():
            pid = node["parent_site_location_id"]
            if pid and pid in node_map:
                node_map[pid]["children"].append(node)
            else:
                roots.append(node)

        return roots