from sqlalchemy import Column, Integer, String, Boolean
from app.db.base import Base


class Department(Base):
    __tablename__ = "departments"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(100), nullable=False, unique=True)

    is_active = Column(
        Boolean,
        nullable=False,
        server_default="true"
    )
