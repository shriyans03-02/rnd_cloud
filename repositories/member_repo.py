from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, func, exists, and_, or_
from typing import List

from app.db.models.member import Member
from app.db.models.member_embedding import MemberEmbedding
from app.schemas.member import MemberCreate, MemberUpdate


class MemberRepository:

    @staticmethod
    def create(db: Session, payload: MemberCreate) -> Member:
        member = Member(
            member_number=payload.member_number,
            first_name=payload.first_name,
            last_name=payload.last_name,
            department_id=payload.department_id,
            is_active=payload.is_active,
        )
        db.add(member)
        db.flush()  # get ID, service commits
        return member  # ← plain object, no re-fetch here

    @staticmethod
    def get_by_id(db: Session, member_id: int) -> Member | None:
        return (
            db.query(Member)
            .options(
                joinedload(Member.department),
                joinedload(Member.access_groups),
            )
            .filter(Member.id == member_id)
            .one_or_none()
        )

    @staticmethod
    def exists_with_member_number(
        db: Session,
        member_number: str,
        exclude_id: int | None = None,
    ) -> bool:
        stmt = select(Member).where(
            func.lower(Member.member_number) == member_number.lower()
        )
        if exclude_id:
            stmt = stmt.where(Member.id != exclude_id)
        return db.execute(stmt).scalars().first() is not None

    @staticmethod
    def update(db: Session, member: Member, payload: MemberUpdate) -> Member:
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(member, field, value)
        db.flush() 
        return member
    
    @staticmethod
    def get_with_relations(db: Session, member_id: int) -> Member | None:
        """Fetch member with all relationships eagerly loaded."""
        return (
            db.query(Member)
            .options(
                joinedload(Member.department),
                joinedload(Member.access_groups),
            )
            .filter(Member.id == member_id)
            .one_or_none()
        ) 

    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[list, int]:
        has_embeddings_expr = exists(
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
        ).label("has_embeddings")

        stmt = (
            select(Member, has_embeddings_expr)
            .options(
                joinedload(Member.department),
                joinedload(Member.access_groups),
            )
        )

        if search:
            search_term = f"%{search.lower()}%"
            stmt = stmt.where(
                func.lower(Member.first_name).like(search_term)
                | func.lower(Member.last_name).like(search_term)
                | func.lower(Member.member_number).like(search_term)
            )

        total = db.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar()

        stmt = (
            stmt.order_by(
                Member.is_active.desc(),
                func.lower(Member.first_name).asc(),
            )
            .offset(page * page_size)
            .limit(page_size)
        )

        rows = db.execute(stmt).unique().all()
        members = []
        for member, has_embeddings in rows:
            member.has_embeddings = has_embeddings
            members.append(member)
        return members, total


    @staticmethod
    def delete(db: Session, member: Member) -> None:
        db.delete(member)
        db.flush()  # ← service commits

    @staticmethod
    def bulk_create(db: Session, members: List[Member]) -> List[Member]:
        db.add_all(members)
        db.flush()
        for member in members:
            db.refresh(member)
        return members

    @staticmethod
    def get_existing_member_numbers(db: Session, member_numbers: List[str]) -> set[str]:
        rows = db.execute(
            select(Member.member_number).where(
                func.lower(Member.member_number).in_(
                    [mn.lower() for mn in member_numbers]
                )
            )
        ).scalars().all()
        return {mn.lower() for mn in rows}

    @staticmethod
    def get_department_name_map(db: Session) -> dict[str, int]:
        from app.db.models.department import Department
        rows = db.execute(
            select(Department.id, Department.name)
            .where(Department.is_active == True)
        ).all()
        return {row.name.strip().lower(): row.id for row in rows}