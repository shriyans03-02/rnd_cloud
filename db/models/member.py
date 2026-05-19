from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.db.models.associations import member_access
from app.db.base import Base


class Member(Base):
    __tablename__ = "members"

    id = Column(Integer, primary_key=True, index=True)

    member_number = Column(String(16), nullable=False, unique=True)

    first_name = Column(String(64), nullable=False)
    last_name = Column(String(64), nullable=True)

    department_id = Column(
        Integer,
        ForeignKey("departments.id", ondelete="SET NULL"),
        nullable=True
    )

    is_active = Column(
        Boolean,
        nullable=False,
        server_default="true"
    )

    access_groups = relationship(
        "AccessGroup",
        secondary=member_access,
        backref="members"
    )

    department = relationship("Department", backref="members")
