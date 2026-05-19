from fastapi import HTTPException, status
from sqlalchemy.orm import Session
from typing import List, Optional, Set, Dict

from app.db.session import SessionLocal
from app.repositories.site_hierarchy_repo import SiteHierarchyRepository
from app.schemas.site_hierarchy import (
    SiteHierarchyCreate,
    SiteHierarchyUpdate,
    SiteHierarchyNode,
)
from app.db.models.site_hierarchy import SiteHierarchy
from app.db.models.site_location import SiteLocation
from app.db.models import Camera
from app.services.activity_log_service import ActivityLogService
from app.schemas.activity_log import ActivityDetail
from app.core.activity_helper import snapshot, build_create_changes, build_update_changes, build_delete_changes
from app.core.constants import TARGET_TYPE
from sqlalchemy import text

SITE_HIERARCHY_TARGET_TYPE = 3
SITE_HIERARCHY_ENTITY = TARGET_TYPE[SITE_HIERARCHY_TARGET_TYPE]["entity"]
SITE_HIERARCHY_EXCLUDE = {"id"}


# ── ACTIVITY LOG HELPERS ──────────────────────────────────────────────────────

def _get_hierarchy_name(db: Session, hierarchy_id: int) -> str | None:
    try:
        row = db.execute(
            text("SELECT name FROM site_hierarchies WHERE id = :id"),
            {"id": hierarchy_id},
        ).mappings().first()
        return row["name"] if row else None
    except Exception:
        return None


def _resolve_site_hierarchy_changes(
    db: Session,
    changes: dict,
) -> dict:
    """Resolve parent_site_hierarchy_id → human-readable name."""
    resolved = {}
    for field, (old, new) in changes.items():
        if field == "parent_site_hierarchy_id":
            old_name = _get_hierarchy_name(db, old) if old is not None else None
            new_name = _get_hierarchy_name(db, new) if new is not None else None
            if old_name is not None or new_name is not None:
                resolved["parent_site_hierarchy"] = [old_name, new_name]
        else:
            resolved[field] = [old, new]
    return resolved


# ── TREE BUILDER ──────────────────────────────────────────────────────────────

class SiteHierarchyTreeBuilder:
    """
    Encapsulates the logic for converting ORM nodes into a nested Pydantic tree
    with computed is_locked and is_deletable flags.
    """

    def __init__(
        self,
        nodes: List[SiteHierarchy],
        direct_locations_map: Dict[int, List[int]],
        used_location_ids: Set[int],
        location_meta_map: Dict[int, Dict],
    ):
        self.nodes = nodes
        self.direct_locations_map = direct_locations_map
        self.used_location_ids = used_location_ids
        self.location_meta_map = location_meta_map

        # Build internal children map for descendant traversal
        self.children_map: Dict[int, List[int]] = {node.id: [] for node in nodes}
        self.node_map: Dict[int, SiteHierarchy] = {node.id: node for node in nodes}
        self.roots: List[SiteHierarchy] = []
        self._build_structure()

    def _build_structure(self):
        for node in self.nodes:
            node.children = []

        for node in self.nodes:
            if node.parent_site_hierarchy_id:
                parent = self.node_map.get(node.parent_site_hierarchy_id)
                if parent:
                    parent.children.append(node)
                    self.children_map[node.parent_site_hierarchy_id].append(node.id)
            else:
                self.roots.append(node)

    def _node_is_locked(self, node: SiteHierarchy) -> bool:
        """True when THIS node itself has a camera-assigned location."""
        direct_loc_ids = self.direct_locations_map.get(node.id, [])
        return any(loc_id in self.used_location_ids for loc_id in direct_loc_ids)

    def _node_is_deletable(self, node_id: int) -> bool:
        """False when self OR any descendant has a camera-assigned location."""
        all_ids = SiteHierarchyRepository.collect_descendant_ids(node_id, self.children_map)
        for nid in all_ids:
            loc_ids = self.direct_locations_map.get(nid, [])
            if any(loc_id in self.used_location_ids for loc_id in loc_ids):
                return False
        return True

    def _to_pydantic(self, node: SiteHierarchy) -> SiteHierarchyNode:
        pyd = SiteHierarchyNode.from_orm(node)

        pyd.children = [self._to_pydantic(child) for child in getattr(node, "children", [])]
        pyd.is_locked = self._node_is_locked(node)
        pyd.is_deletable = self._node_is_deletable(node.id)

        # Leaf-level public/protected meta
        direct_loc_ids = self.direct_locations_map.get(node.id, [])
        if direct_loc_ids:
            loc_meta = self.location_meta_map.get(node.id)
            if loc_meta:
                pyd.is_public = loc_meta.get("is_public", False)
                pyd.is_protected = loc_meta.get("is_protected", False)
                pyd.is_leaf = loc_meta.get("is_leaf", False)

        return pyd

    def build(self) -> List[SiteHierarchyNode]:
        return [self._to_pydantic(root) for root in self.roots]


# ── SERVICE ───────────────────────────────────────────────────────────────────

class SiteHierarchyService:

    # ── TREE ──────────────────────────────────────────────────────────────────

    @staticmethod
    def get_tree(db: Session, search: Optional[str] = None) -> List[SiteHierarchyNode]:
        """
        Load all nodes and build a nested tree with is_locked and is_deletable flags.

        If `search` is provided:
          - Find all nodes whose name matches (case-insensitive)
          - Walk up to find the root of each matched node
          - Return the full subtrees of those roots (not just the matched nodes)
        """
        from sqlalchemy.orm import selectinload

        nodes = db.query(SiteHierarchy).options(
            selectinload(SiteHierarchy.children)
        ).all()

        direct_locations_map, location_meta_map = SiteHierarchyRepository.fetch_direct_locations_map(db)
        used_location_ids = SiteHierarchyRepository.fetch_used_location_ids(db)

        builder = SiteHierarchyTreeBuilder(
            nodes=nodes,
            direct_locations_map=direct_locations_map,
            used_location_ids=used_location_ids,
            location_meta_map=location_meta_map,
        )
        full_tree = builder.build()

        if not search:
            return full_tree

        # ── Search: find root IDs whose subtree contains the search term ──────
        search_lower = search.strip().lower()
        node_map: Dict[int, SiteHierarchy] = {n.id: n for n in nodes}

        def get_root_id(node: SiteHierarchy) -> int:
            """Walk up parent chain to find the root node's id."""
            current = node
            while current.parent_site_hierarchy_id is not None:
                parent = node_map.get(current.parent_site_hierarchy_id)
                if parent is None:
                    break
                current = parent
            return current.id

        # Collect root IDs for all matching nodes
        matched_root_ids: Set[int] = set()
        for node in nodes:
            if search_lower in node.name.lower():
                matched_root_ids.add(get_root_id(node))

        # Return only the full trees whose root matched
        return [tree_node for tree_node in full_tree if tree_node.id in matched_root_ids]

    # ── CREATE ────────────────────────────────────────────────────────────────

    @staticmethod
    def create_site_hierarchy(
        db: Session,
        payload: SiteHierarchyCreate,
        actor_id: int,
    ) -> SiteHierarchy:
        if SiteHierarchyRepository.exists_with_name(db, payload.name):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Site name already exists",
            )

        site = SiteHierarchyRepository.create(db, payload)

        detail = ActivityDetail(
            action="create",
            entity=SITE_HIERARCHY_ENTITY,
            changes=_resolve_site_hierarchy_changes(
                db,
                build_create_changes(site, exclude=SITE_HIERARCHY_EXCLUDE),
            ),
            meta={"actor_id": actor_id, "display_name": site.name},
        )
        ActivityLogService.log(
            db=db,
            actor_id=actor_id,
            target_type=SITE_HIERARCHY_TARGET_TYPE,
            target_id=site.id,
            detail=detail,
        )

        db.commit()
        db.refresh(site)
        return site

    # ── UPDATE ────────────────────────────────────────────────────────────────

    @staticmethod
    def update_site_hierarchy(
        db: Session,
        site_hierarchy_id: int,
        payload: SiteHierarchyUpdate,
        actor_id: int,
    ) -> SiteHierarchy:
        site = SiteHierarchyRepository.get_by_id(db, site_hierarchy_id)
        if not site:
            raise HTTPException(status_code=404, detail="Site not found")

        before = snapshot(site)
        updated_site = SiteHierarchyRepository.update(db, site, payload)

        detail = ActivityDetail(
            action="update",
            entity=SITE_HIERARCHY_ENTITY,
            changes=_resolve_site_hierarchy_changes(
                db,
                build_update_changes(before, updated_site, exclude=SITE_HIERARCHY_EXCLUDE),
            ),
            meta={"actor_id": actor_id, "display_name": updated_site.name},
        )
        ActivityLogService.log(
            db=db,
            actor_id=actor_id,
            target_type=SITE_HIERARCHY_TARGET_TYPE,
            target_id=site_hierarchy_id,
            detail=detail,
        )

        db.commit()
        db.refresh(updated_site)
        return updated_site

    # ── DELETE ────────────────────────────────────────────────────────────────

    @staticmethod
    def delete_site_hierarchy(
        db: Session,
        site_hierarchy_id: int,
        actor_id: int,
    ) -> None:
        """
        Cascade-delete a site hierarchy node and ALL its descendants.
        Raises 404 if not found, 400 if any node in the subtree is camera-assigned.
        Logs activity only for the root node being deleted.
        """
        # ── 1. Build children map from all nodes ─────────────────────────────
        raw_nodes = SiteHierarchyRepository.fetch_all_nodes_raw(db)
        children_map = SiteHierarchyRepository.build_children_map(raw_nodes)

        if site_hierarchy_id not in children_map:
            raise HTTPException(status_code=404, detail="Site hierarchy not found")

        # ── 2. Collect root + all descendants ────────────────────────────────
        ids_to_delete = SiteHierarchyRepository.collect_descendant_ids(
            site_hierarchy_id, children_map
        )

        # ── 3. Guard: ensure no location in the subtree is camera-assigned ───
        location_ids = [
            row[0]
            for row in db.query(SiteLocation.id)
            .filter(SiteLocation.site_hierarchy_id.in_(ids_to_delete))
            .all()
        ]
        if location_ids:
            in_use = db.query(Camera.id).filter(
                Camera.site_location_id.in_(location_ids)
            ).first()
            if in_use:
                raise HTTPException(
                    status_code=400,
                    detail="Cannot delete. One or more sites in this hierarchy are used by a camera.",
                )

        # ── 4. Snapshot root node for activity log before deletion ────────────
        root_node = SiteHierarchyRepository.get_by_id(db, site_hierarchy_id)
        before = snapshot(root_node)

        # ── 5. Delete leaf-first (children before parents) ────────────────────
        SiteHierarchyRepository.delete_subtree(db, site_hierarchy_id, children_map)

        # ── 6. Activity log for root only ─────────────────────────────────────
        detail = ActivityDetail(
            action="delete",
            entity=SITE_HIERARCHY_ENTITY,
            changes=_resolve_site_hierarchy_changes(
                db,
                build_delete_changes(before, exclude=SITE_HIERARCHY_EXCLUDE),
            ),
            meta={"actor_id": actor_id, "display_name": before.get("name")},
        )
        ActivityLogService.log(
            db=db,
            actor_id=actor_id,
            target_type=SITE_HIERARCHY_TARGET_TYPE,
            target_id=site_hierarchy_id,
            detail=detail,
        )

        db.commit()

    # ── LEGACY / STANDALONE HELPERS (unchanged public API) ────────────────────

    @staticmethod
    def get_site_hierarchy(site_hierarchy_id: int) -> SiteHierarchy | None:
        db = SessionLocal()
        try:
            return SiteHierarchyRepository.get_by_id(db, site_hierarchy_id)
        finally:
            db.close()

    @staticmethod
    def list_site_hierarchies(
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ):
        db = SessionLocal()
        try:
            return SiteHierarchyRepository.list(db, search, page, page_size)
        finally:
            db.close()

    @staticmethod
    def get_tree_legacy(search: str = None):
        db = SessionLocal()
        try:
            nodes = SiteHierarchyRepository.list_all(db, search)
            return SiteHierarchyService._build_flat_tree(nodes)
        finally:
            db.close()

    @staticmethod
    def _build_flat_tree(nodes: list):
        node_map = {node.id: dict(node.__dict__, children=[]) for node in nodes}
        roots = []
        for node in node_map.values():
            pid = node.get("parent_site_hierarchy_id")
            if pid and pid in node_map:
                node_map[pid]["children"].append(node)
            else:
                roots.append(node)
        return roots

    @staticmethod
    def _build_site_location_tree(locations: List[SiteLocation]) -> List[dict]:
        def serialize_loc(loc: SiteLocation) -> dict:
            return {
                "id": loc.id,
                "name": loc.name,
                "site_hierarchy_id": loc.site_hierarchy_id,
                "parent_site_location_id": loc.parent_site_location_id,
                "is_public": loc.is_public,
                "is_active": loc.is_active,
                "cameras": [
                    {
                        "id": cam.id,
                        "name": cam.name,
                        "ip_address": cam.ip_address,
                        "is_active": cam.is_active,
                    }
                    for cam in getattr(loc, "cameras", []) or []
                ],
                "children": [],
            }

        node_map = {loc.id: serialize_loc(loc) for loc in locations}
        roots = []
        for node in node_map.values():
            pid = node["parent_site_location_id"]
            if pid and pid in node_map:
                node_map[pid]["children"].append(node)
            else:
                roots.append(node)
        return roots

    @staticmethod
    def _serialize_hierarchy_node(h: SiteHierarchy, loc_tree: List[dict]) -> dict:
        return {
            "id": h.id,
            "name": h.name,
            "parent_site_hierarchy_id": h.parent_site_hierarchy_id,
            "is_active": h.is_active,
            "site_locations": loc_tree,
            "children": [],
        }

    @staticmethod
    def build_hierarchy_tree(hierarchies: List[SiteHierarchy]) -> List[dict]:
        node_map = {}
        for h in hierarchies:
            locs = getattr(h, "site_locations", []) or []
            loc_tree = SiteHierarchyService._build_site_location_tree(locs)
            node_map[h.id] = SiteHierarchyService._serialize_hierarchy_node(h, loc_tree)

        roots = []
        for node in node_map.values():
            pid = node["parent_site_hierarchy_id"]
            if pid and pid in node_map:
                node_map[pid]["children"].append(node)
            else:
                roots.append(node)
        return roots

    @staticmethod
    def get_full_hierarchy_tree(
        site_hierarchy_id: Optional[int] = None,
        include_inactive: bool = False,
    ) -> List[dict]:
        db = SessionLocal()
        try:
            hierarchies = SiteHierarchyRepository.list_hierarchies_with_locations(
                db, include_inactive=include_inactive, site_hierarchy_id=site_hierarchy_id
            )
            return SiteHierarchyService.build_hierarchy_tree(hierarchies)
        finally:
            db.close()