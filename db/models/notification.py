from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    String,
    ForeignKey,
    DateTime,
    JSON,
    Index,
)
from sqlalchemy.orm import relationship ,backref
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.db.base import Base


class Notification(Base):
    __tablename__ = "notifications"

    id = Column(BigInteger, primary_key=True, index=True)

    # 1 = unauthorized member, 2 = unauthorized guest, etc.
    type = Column(Integer, nullable=False)

    camera_id = Column(
        Integer,
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False
    )

    member_id = Column(
        Integer,
        ForeignKey("members.id", ondelete="SET NULL"),
        nullable=True
    )

    guest_temp_id = Column(String(64), nullable=True)

    average_guest_data_vector = Column(
        Vector(512),
        nullable=True
    ) 
    
    # 1 = Open, 2 = Under Action, 3 = Resolved
    status = Column(
        Integer,
        nullable=False,
        server_default="1"
    )

    details = Column(JSON, nullable=True)

    created_ts = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    closed_ts = Column(
        DateTime(timezone=True),
        nullable=True
    )

    # 🔽 NEW FIELDS 🔽
    closed_by = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )

    closed_notes = Column(
        String(512),
        nullable=True
    )

    # relationships
    camera = relationship(
    "Camera",
    backref=backref("notifications", passive_deletes=True)
)
    member = relationship("Member", backref="notifications")
    closed_by_user = relationship("User", foreign_keys=[closed_by])

    # indexes
    __table_args__ = (
        Index("ix_notification_type_status", "type", "status"),
        Index("ix_notification_member_created", "member_id", "created_ts"),
        Index("ix_notification_camera_created", "camera_id", "created_ts"),
    )
