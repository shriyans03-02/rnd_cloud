from sqlalchemy import (
    Boolean,
    BigInteger,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    JSON,
    String,
)
from sqlalchemy.orm import relationship, backref
from sqlalchemy.sql import func

from app.db.base import Base


class Detection(Base):
    """
    Lightweight per-person metadata produced by the live AI pipeline.

    One row is written only for each known/recognized member in a processed frame.
    The frame image/video is not stored here; only the metadata needed for
    canvas overlays, audit search, and later playback is stored. Unknown tracks are intentionally not inserted.
    """

    __tablename__ = "detection"

    id = Column(BigInteger, primary_key=True, index=True)

    camera_id = Column(
        Integer,
        ForeignKey("cameras.id", ondelete="CASCADE"),
        nullable=False,
    )

    member_id = Column(
        Integer,
        ForeignKey("members.id", ondelete="SET NULL"),
        nullable=True,
    )

    # Timestamp of the captured source frame, as Unix seconds with decimals.
    # This is the sync key used by the delayed MJPEG stream + canvas overlay.
    timestamp = Column(Float, nullable=False)

    # Sequential processed-frame number for this camera worker.
    frame_seq = Column(BigInteger, nullable=True)

    track_id = Column(Integer, nullable=False)
    raw_track_id = Column(Integer, nullable=True)

    # [x1, y1, x2, y2] in the processed frame coordinate space.
    bbox = Column(JSON, nullable=False)

    person_name = Column(String(128), nullable=True)
    is_known = Column(Boolean, nullable=False, server_default="false")

    face_conf = Column(Float, nullable=True)
    det_conf = Column(Float, nullable=True)

    frame_width = Column(Integer, nullable=True)
    frame_height = Column(Integer, nullable=True)
    processing_latency_ms = Column(Float, nullable=True)

    created_ts = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    camera = relationship(
        "Camera",
        backref=backref("detections", passive_deletes=True),
    )
    member = relationship("Member", backref="detections")

    __table_args__ = (
        Index("ix_detection_camera_timestamp", "camera_id", "timestamp"),
        Index("ix_detection_camera_track_timestamp", "camera_id", "track_id", "timestamp"),
        Index("ix_detection_member_timestamp", "member_id", "timestamp"),
        Index("ix_detection_person_timestamp", "person_name", "timestamp"),
    )
