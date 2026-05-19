from sqlalchemy import (
    Column,
    Integer,
    String,
    Boolean,
    DateTime,
    func,
)
from app.db.base import Base



class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)

    first_name = Column(String(64), nullable=False)
    last_name = Column(String(64), nullable=False)

    email = Column(String(128), nullable=False, unique=True, index=True)

    password = Column(String(256), nullable=False)

    role = Column(Integer, nullable=False, server_default="1")

    created_ts = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    last_login_ts = Column(
        DateTime(timezone=True),
        nullable=True,
    )

    is_active = Column(
        Boolean,
        nullable=False,
        server_default="true",
    )



    