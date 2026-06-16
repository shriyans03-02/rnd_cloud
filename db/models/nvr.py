from sqlalchemy import Boolean, Column, ForeignKey, Integer, String, UniqueConstraint
from sqlalchemy.orm import relationship

from app.db.base import Base


class NVR(Base):
    __tablename__ = "nvrs"
    __table_args__ = (
        UniqueConstraint("ip_address", "port", name="uq_nvrs_ip_address_port"),
    )

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), nullable=False)
    ip_address = Column(String(64), nullable=False)
    port = Column(Integer, nullable=False, server_default="554")
    brand = Column(String(32), nullable=False, server_default="generic")
    brand_id = Column(
        Integer,
        ForeignKey("device_brands.id", ondelete="SET NULL"),
        nullable=True,
    )
    username = Column(String(64), nullable=True)
    password = Column(String(128), nullable=True)
    playback_rtsp_template = Column(String(1024), nullable=True)
    stream_key = Column(String(128), nullable=True)
    is_active = Column(Boolean, nullable=False, server_default="true")

    cameras = relationship(
        "Camera",
        back_populates="nvr_rel",
        lazy="selectin",
        passive_deletes=True,
    )

    brand_rel = relationship(
        "DeviceBrand",
        back_populates="nvrs",
        lazy="selectin",
    )

    @property
    def brand_label(self) -> str | None:
        if self.brand_rel:
            return self.brand_rel.label
        return self.brand

    @property
    def playback_time_format(self) -> str | None:
        if self.brand_rel:
            return self.brand_rel.playback_time_format
        return None
