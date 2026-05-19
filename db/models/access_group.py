from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base


class AccessGroup(Base):
    __tablename__ = "access_groups"

    id = Column(Integer, primary_key=True, index=True)

    name = Column(String(64), nullable=False)

    parent_access_group_id = Column(
        Integer,
        ForeignKey("access_groups.id", ondelete="SET NULL"),
        nullable=True
    )

    is_active = Column(
        Boolean,
        nullable=False,
        server_default="true"
    )

    parent = relationship(
        "AccessGroup",
        remote_side=[id],
        backref="children",
        lazy="selectin",
    )
