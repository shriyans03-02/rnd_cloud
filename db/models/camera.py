from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base


class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(64), nullable=False)
    ip_address = Column(String(16), nullable=False)

    site_location_id = Column(
        Integer,
        ForeignKey("site_locations.id", ondelete="CASCADE"),
        nullable=False
    ) 

    is_active = Column(
        Boolean,
        nullable=False,
        server_default="true"
    )

    # rename relationship
    site_location_rel = relationship(
        "SiteLocation",
        back_populates="cameras"
    )

    # computed string property
    @property
    def site_location(self) -> str | None:
        if self.site_location_rel and self.site_location_rel.site_hierarchy:
            return self.site_location_rel.site_hierarchy.name
        return None
