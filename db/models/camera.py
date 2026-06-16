from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), nullable=False)
    ip_address = Column(String(64), nullable=False)
    brand = Column(String(32), nullable=False, server_default="generic")
    brand_id = Column(
        Integer,
        ForeignKey("device_brands.id", ondelete="SET NULL"),
        nullable=True,
    )
    rtsp_url_template = Column(String(1024), nullable=True)
    rtsp_port = Column(Integer, nullable=True, server_default="554")
    rtsp_channel = Column(Integer, nullable=True, server_default="1")
    rtsp_subtype = Column(String(32), nullable=True, server_default="0")
    rtsp_username = Column(String(64), nullable=True)
    rtsp_password = Column(String(128), nullable=True)
    nvr_id = Column(
        Integer,
        ForeignKey("nvrs.id", ondelete="SET NULL"),
        nullable=True,
    )
    nvr_channel = Column(Integer, nullable=True)

    site_location_id = Column(
        Integer,
        ForeignKey("site_locations.id", ondelete="CASCADE"),
        nullable=False,
    )

    is_active = Column(
        Boolean,
        nullable=False,
        server_default="true",
    )

    site_location_rel = relationship(
        "SiteLocation",
        back_populates="cameras",
    )

    nvr_rel = relationship(
        "NVR",
        back_populates="cameras",
    )

    brand_rel = relationship(
        "DeviceBrand",
        back_populates="cameras",
        lazy="selectin",
    )

    @property
    def site_location(self) -> str | None:
        if self.site_location_rel and self.site_location_rel.site_hierarchy:
            return self.site_location_rel.site_hierarchy.name
        return None

    @property
    def nvr_name(self) -> str | None:
        return self.nvr_rel.name if self.nvr_rel else None

    @property
    def brand_label(self) -> str | None:
        if self.brand_rel:
            return self.brand_rel.label
        return self.brand
