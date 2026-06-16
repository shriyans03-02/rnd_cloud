from sqlalchemy import Boolean, Column, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base import Base


class DeviceBrand(Base):
    __tablename__ = "device_brands"
    __table_args__ = (
        UniqueConstraint("name", "device_type", name="uq_device_brands_name_type"),
    )

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), nullable=False)
    label = Column(String(120), nullable=False)
    device_type = Column(String(16), nullable=False)  # camera | nvr
    live_rtsp_template = Column(String(1024), nullable=True)
    playback_rtsp_template = Column(String(1024), nullable=True)
    playback_time_format = Column(String(64), nullable=True)
    is_active = Column(Boolean, nullable=False, server_default="true")

    cameras = relationship(
        "Camera",
        back_populates="brand_rel",
        lazy="selectin",
        passive_deletes=True,
    )
    nvrs = relationship(
        "NVR",
        back_populates="brand_rel",
        lazy="selectin",
        passive_deletes=True,
    )
