from sqlalchemy.orm import Session, joinedload, aliased
from sqlalchemy import select, func
from typing import List, Dict, Optional, Set, Tuple

from app.db.models.access_group import AccessGroup
from app.schemas.access_group import AccessGroupCreate, AccessGroupUpdate


class AccessGroupRepository:

    @staticmethod
    def create(db: Session, payload: AccessGroupCreate) -> AccessGroup:
        group = AccessGroup(
            name=payload.name,
            parent_access_group_id=payload.parent_access_group_id,
        )
        db.add(group)
        db.flush()
        return group

    @staticmethod
    def get_by_id(db: Session, group_id: int) -> AccessGroup | None:
        return db.get(AccessGroup, group_id)

    @staticmethod
    def get_with_relations(db: Session, group_id: int) -> AccessGroup | None:
        return (
            db.query(AccessGroup)
            .options(joinedload(AccessGroup.parent))
            .filter(AccessGroup.id == group_id)
            .one_or_none()
        )

    @staticmethod
    def exists_with_name(
        db: Session,
        name: str,
        parent_id: int | None,
        exclude_id: int | None = None,
    ) -> bool:
        stmt = select(AccessGroup).where(
            func.lower(AccessGroup.name) == name.lower(),
            AccessGroup.parent_access_group_id == parent_id,
        )
        if exclude_id:
            stmt = stmt.where(AccessGroup.id != exclude_id)
        return db.execute(stmt).scalars().first() is not None

    @staticmethod
    def update(db: Session, group: AccessGroup, payload: AccessGroupUpdate) -> AccessGroup:
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(group, field, value)
        db.flush()
        return group

    @staticmethod
    def list_active_hierarchy(db: Session, search: str | None = None) -> tuple[list[AccessGroup], int]:
        anchor = (
            select(AccessGroup.id)
            .where(AccessGroup.parent_access_group_id.is_(None))
            .where(AccessGroup.is_active == True)
            .cte(name="active_hierarchy", recursive=True)
        )
        child = aliased(AccessGroup)
        recursive_part = (
            select(child.id)
            .join(anchor, child.parent_access_group_id == anchor.c.id)
            .where(child.is_active == True)
        )
        active_hierarchy = anchor.union_all(recursive_part)
        stmt = (
            select(AccessGroup)
            .options(joinedload(AccessGroup.parent))
            .where(AccessGroup.id.in_(select(active_hierarchy.c.id)))
        )
        if search:
            stmt = stmt.where(func.lower(AccessGroup.name).contains(search.lower()))

        stmt = stmt.order_by(func.lower(AccessGroup.name).asc())
        groups = db.execute(stmt).scalars().all()
        return groups, len(groups)

    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[list[AccessGroup], int]:
        stmt = (
            select(AccessGroup)
            .options(joinedload(AccessGroup.parent))
        )
        if search:
            search_term = f"%{search.lower()}%"
            stmt = stmt.where(func.lower(AccessGroup.name).like(search_term))

        total = db.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar()

        stmt = stmt.order_by(
            AccessGroup.is_active.desc(),
            func.lower(AccessGroup.name).asc()
        )
        stmt = stmt.offset(page * page_size).limit(page_size)
        groups = db.execute(stmt).scalars().all()
        return groups, total

    @staticmethod
    def list_all(db: Session):
        return (
            db.query(AccessGroup)
            .filter(AccessGroup.is_active.is_(True))
            .order_by(AccessGroup.name)
            .all()
        )

    # ── DELETE SUBTREE HELPERS ────────────────────────────────────────────────

    @staticmethod
    def fetch_all_nodes_raw(db: Session) -> List[Tuple[int, Optional[int]]]:
        """Returns list of (id, parent_access_group_id) for ALL nodes."""
        return db.query(AccessGroup.id, AccessGroup.parent_access_group_id).all()

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
            ids |= AccessGroupRepository.collect_descendant_ids(child_id, children_map)
        return ids

    @staticmethod
    def delete_subtree(db: Session, node_id: int, children_map: Dict[int, List[int]]):
        """Recursively delete children before parent (leaf-first) to respect FK constraints."""
        for child_id in children_map.get(node_id, []):
            AccessGroupRepository.delete_subtree(db, child_id, children_map)

        node = db.get(AccessGroup, node_id)
        if node:
            db.delete(node)
            db.flush()

    @staticmethod
    def delete(db: Session, group: AccessGroup) -> None:
        db.delete(group)
        db.flush()