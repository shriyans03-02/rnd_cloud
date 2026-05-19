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
from sqlalchemy.orm import relationship , backref
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
from app.db.base import Base


class NormalizedData(Base):
    __tablename__ = "normalized_data"

    id = Column(BigInteger, primary_key=True, index=True)

    member_id = Column(
        Integer,
        ForeignKey("members.id", ondelete="SET NULL"),
        nullable=True
    )

    guest_temp_id = Column(
        String(64),
        nullable=True
    )
 
    average_guest_data_vector = Column(
        Vector(512),
        nullable=True
    ) 
    camera_id = Column(
        Integer,
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False
    )

    # 1 = Enter Camera Frame, 2 = Exit Camera Frame
    movement_type = Column(
        Integer,
        nullable=False
    )

    entry_ts = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    exit_ts = Column(
        DateTime(timezone=True),
        nullable=True
    )

    average_match_value = Column(
        Integer,
        nullable=True
    )

    # relationships (optional but useful)
    member = relationship("Member", backref="normalized_events")
    camera = relationship(
    "Camera",
    backref=backref("normalized_events", passive_deletes=True)
)

    __table_args__ = (
        Index("ix_norm_member_entry_ts", "member_id", "entry_ts"),
        Index("ix_norm_guest_entry_ts", "guest_temp_id", "entry_ts"),
    )