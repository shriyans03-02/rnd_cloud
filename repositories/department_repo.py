from sqlalchemy.orm import Session
from sqlalchemy import select, func

from app.db.models.department import Department
from app.schemas.department import DepartmentCreate, DepartmentUpdate
from typing import List


class DepartmentRepository:

    @staticmethod
    def create(db: Session, payload: DepartmentCreate) -> Department:
        department = Department(name=payload.name)
        db.add(department)
        db.flush()
        db.refresh(department)
        return department

    @staticmethod
    def get_by_id(db: Session, department_id: int) -> Department | None:
        return db.get(Department, department_id)

    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[list[Department], int]:
        stmt = select(Department)

        if search:
            search_term = f"%{search.lower()}%"
            stmt = stmt.where(func.lower(Department.name).like(search_term))

        total = db.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar()

        stmt = stmt.order_by(
            Department.is_active.desc(),
            func.lower(Department.name).asc()
        )

        stmt = stmt.offset(page * page_size).limit(page_size)
        departments = db.execute(stmt).scalars().all()
        return departments, total

    @staticmethod
    def update(db: Session, department: Department, payload: DepartmentUpdate) -> Department:
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(department, field, value)
        db.flush()
        db.refresh(department)
        return department

    @staticmethod
    def delete(db: Session, department: Department) -> None:
        db.delete(department)
        db.flush()

    @staticmethod
    def list_all_active(db: Session, name: str | None = None) -> List[Department]:
        stmt = (
            select(Department)
            .where(Department.is_active.is_(True))
        )

        if name:
            stmt = stmt.where(Department.name.ilike(f"%{name}%"))

        stmt = stmt.order_by(func.lower(Department.name).asc())
        return db.execute(stmt).scalars().all()