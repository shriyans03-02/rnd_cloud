from sqlalchemy.orm import Session
from sqlalchemy import select, or_, func

from app.db.models.user import User
from app.schemas.user import UserCreate, UserUpdate
from datetime import datetime, timezone

class UserRepository:

    # Create User
    @staticmethod
    def create(db: Session, payload: UserCreate) -> User:
        user = User(
            first_name=payload.first_name,
            last_name=payload.last_name,
            email=payload.email,
            password=payload.password,
            role=payload.role,
        )
        db.add(user)
        db.flush()
        db.refresh(user)
        return user

    # Update last login timestamp
    @staticmethod
    def update_last_login(db: Session, user: User) -> None:
        user.last_login_ts = func.now()
        db.commit()

    # Get user by ID
    @staticmethod
    def get_by_id(db: Session, user_id: int) -> User | None:
        return db.get(User, user_id)

    # Get user by email
    @staticmethod
    def get_by_email(db: Session, email: str) -> User | None:
        stmt = select(User).where(User.email == email)
        return db.execute(stmt).scalars().first()

    @staticmethod
    def list(
        db: Session,
        search: str | None = None,
        page: int = 0,
        page_size: int = 10,
    ) -> tuple[list[User], int]:

        stmt = select(User)

        # search
        if search:
            search_term = f"%{search.lower()}%"
            full_name = func.lower(User.first_name + " " + User.last_name)
            stmt = stmt.where(
                or_(
                    func.lower(User.first_name).like(search_term),
                    func.lower(User.last_name).like(search_term),
                    func.lower(User.email).like(search_term),
                    full_name.like(search_term),
                )
            )

        # total count (before pagination)
        total = db.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar()

        # ORDER:
        #   1) active users first
        #   2) then by first name
        #   3) then by last name
        stmt = stmt.order_by(
            User.is_active.desc(),
            func.lower(User.first_name).asc(),
            func.lower(User.last_name).asc(),
        )

        # pagination
        stmt = stmt.offset(page * page_size).limit(page_size)

        users = db.execute(stmt).scalars().all()
        return users, total


    # Update user
    @staticmethod
    def update(
        db: Session,
        user: User,
        payload: UserUpdate,
    ) -> User:
        for field, value in payload.model_dump(exclude_unset=True).items():
            setattr(user, field, value)

        db.flush()
        db.refresh(user)
        return user

    # Hard delete (PERMANENT DELETE)
    @staticmethod
    def delete(db: Session, user: User) -> None:
        db.delete(user)
        db.flush() 
    
    # Update last_login
    @staticmethod
    def update_last_login(db: Session, user: User) -> None:
        user.last_login_ts = datetime.now(timezone.utc)
        db.commit()

