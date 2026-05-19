from sqlalchemy.orm import Session
from app.repositories.user_repo import UserRepository
from app.schemas.auth import RegisterRequest
from app.core.security import verify_password
from app.db.models.user import User
from app.core.security import hash_password, verify_password

class AuthService:

    @staticmethod
    def login(db: Session, email: str, password: str) -> User:
        user = UserRepository.get_by_email(db, email)

        if not user:
            raise ValueError("Invalid email or password")

        if not user.is_active:
            raise ValueError("User is inactive")

        if not verify_password(password, user.password):
            raise ValueError("Invalid email or password")

        UserRepository.update_last_login(db, user)

        return user
    
    @staticmethod
    def register(db: Session, payload: RegisterRequest) -> User:
        if UserRepository.get_by_email(db, payload.email):
            raise ValueError("Email already exists")

        user = User(
            first_name=payload.first_name,
            last_name=payload.last_name,
            email=payload.email,
            password=hash_password(payload.password),
            role=payload.role
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        return user