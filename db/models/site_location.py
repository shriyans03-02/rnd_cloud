from sqlalchemy import Column, Integer, String, Boolean, ForeignKey
from sqlalchemy.orm import relationship
from app.db.base import Base
from app.db.models.associations import site_location_access


class SiteLocation(Base):
    __tablename__ = "site_locations"

    id = Column(Integer, primary_key=True, index=True) 
    site_hierarchy_id = Column(
        Integer,
        ForeignKey("site_hierarchies.id"),
        nullable=False
    )
    is_public = Column(Boolean, default=False, nullable=False)
    is_protected = Column(Boolean, default=False, nullable=False)
    is_active = Column(Boolean, default=True, nullable=False)

    # link back to hierarchy
    site_hierarchy = relationship(
        "SiteHierarchy",
        back_populates="site_locations"
    ) 

    # many-to-many access
    access_groups = relationship(
        "AccessGroup",
        secondary=site_location_access,
        backref="site_locations"
    )

    # optional but recommended (for cameras)
    cameras = relationship(
        "Camera",
        back_populates="site_location_rel",
        lazy="selectin"
    )

    @property
    def name(self) -> str | None:
        return self.site_hierarchy.name if self.site_hierarchy else None
