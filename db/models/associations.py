from sqlalchemy import Table, Column, Integer, ForeignKey
from app.db.base import Base


site_location_access = Table(
    "site_location_access",
    Base.metadata,
    Column(
        "site_location_id",
        Integer,
        ForeignKey("site_locations.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "access_group_id",
        Integer,
        ForeignKey("access_groups.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)


member_access = Table(
    "member_access",
    Base.metadata,
    Column(
        "member_id",
        Integer,
        ForeignKey("members.id", ondelete="CASCADE"),
        primary_key=True,
    ),
    Column(
        "access_group_id",
        Integer,
        ForeignKey("access_groups.id", ondelete="CASCADE"),
        primary_key=True,
    ),
)
