from sqlalchemy import (
    Column,
    Integer,
    BigInteger,
    ForeignKey,
    DateTime,
    JSON,
    Index,
)
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.db.base import Base


class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(BigInteger, primary_key=True, index=True)

    actor_id = Column(
        Integer,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True
    )

    # 1=user, 2=department, 3=site_hierarchy, 4=site_location,
    # 5=camera, 6=access_group, 7=member, 8=notification
    target_type = Column(Integer, nullable=False)

    target_id = Column(BigInteger, nullable=False)

    details = Column(JSON, nullable=True)

    activity_ts = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False
    )

    actor = relationship("User", backref="activity_logs")

    __table_args__ = (
        Index(
            "ix_activity_log_target",
            "target_type",
            "target_id"
        ),
    )
