from sqlalchemy.orm import Session
from sqlalchemy import select, insert, delete, or_
from app.db.models.associations import member_access
from sqlalchemy.orm import joinedload
from app.db.models import Member, AccessGroup


class MemberAccessRepository:

    # -------- CREATE (single) --------
    @staticmethod
    def create(db: Session, member_id: int, access_group_id: int):
        db.execute(
            insert(member_access).values(
                member_id=member_id,
                access_group_id=access_group_id,
            )
        )
        db.commit()

    # -------- EXISTS --------
    @staticmethod
    def exists(db: Session, member_id: int, access_group_id: int) -> bool:
        stmt = select(member_access).where(
            member_access.c.member_id == member_id,
            member_access.c.access_group_id == access_group_id,
        )
        return db.execute(stmt).first() is not None

    # -------- BULK CREATE --------
    @staticmethod
    def bulk_create(
        db: Session,
        member_id: int,
        access_group_ids: list[int],
    ) -> list[int]:
        """Returns list of actually inserted access_group_ids."""
        existing_stmt = select(member_access.c.access_group_id).where(
            member_access.c.member_id == member_id,
            member_access.c.access_group_id.in_(access_group_ids),
        )
        existing_ids = {row[0] for row in db.execute(existing_stmt).all()}

        new_ids = [ag_id for ag_id in access_group_ids if ag_id not in existing_ids]

        if not new_ids:
            return []

        db.execute(
            insert(member_access),
            [{"member_id": member_id, "access_group_id": ag_id} for ag_id in new_ids],
        )
        db.flush()  # ← service commits
        return new_ids

    # -------- LIST --------
    @staticmethod
    def list(db, search: str | None, page: int = 0, page_size: int = 10):

        def all_access_group_ancestors_active(access_group: AccessGroup) -> bool:
            current = access_group.parent
            while current is not None:
                if not current.is_active:
                    return False
                current = current.parent
            return True

        query = (
            db.query(Member)
            .options(joinedload(Member.access_groups))
        )

        if search:
            query = query.filter(
                or_(
                    (Member.first_name + " " + Member.last_name).ilike(f"%{search}%"),
                    Member.access_groups.any(AccessGroup.name.ilike(f"%{search}%"))
                )
            )

        total = query.count()
        members = query.offset(page * page_size).limit(page_size).all()

        access_group_map: dict[int, dict] = {}

        for member in members:
            full_name = " ".join(
                part for part in [member.first_name, member.last_name] if part
            )

            member_name_matches = not search or search.lower() in full_name.lower()

            for ag in member.access_groups:
                if not ag.is_active or not all_access_group_ancestors_active(ag):
                    continue

                group_name_matches = not search or search.lower() in ag.name.lower()

                if not member_name_matches and not group_name_matches:
                    continue

                if ag.id not in access_group_map:
                    access_group_map[ag.id] = {
                        "access_group_id": ag.id,
                        "access_group_name": ag.name,
                        "members": [],
                    }

                access_group_map[ag.id]["members"].append({
                    "member_id": member.id,
                    "member_name": full_name,
                })

        result = list(access_group_map.values())
        return result, total

    # -------- DELETE (single) --------
    @staticmethod
    def delete_single(db: Session, member_id: int, access_group_id: int) -> bool:
        """Returns True if a row was deleted."""
        result = db.execute(
            delete(member_access).where(
                member_access.c.member_id == member_id,
                member_access.c.access_group_id == access_group_id,
            )
        )
        db.flush()  # ← service commits
        return result.rowcount > 0

    # -------- DELETE (original — kept for non-logged usage) --------
    @staticmethod
    def delete(db: Session, member_id: int, access_group_id: int):
        db.execute(
            delete(member_access).where(
                member_access.c.member_id == member_id,
                member_access.c.access_group_id == access_group_id,
            )
        )
        db.commit()