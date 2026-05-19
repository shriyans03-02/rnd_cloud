from sqlalchemy.orm import Session, joinedload
from sqlalchemy import select, func, and_, delete, update
from typing import Optional, List 
from datetime import datetime, timedelta

from app.db.models.notification import Notification
from app.db.models.camera import Camera
from app.db.models.site_location import SiteLocation
from app.db.models.site_hierarchy import SiteHierarchy
from app.db.models.member import Member


class NotificationRepository:

    @staticmethod
    def get_by_id(db: Session, notification_id: int) -> Notification | None:
        return (
            db.query(Notification)
            .options(
                joinedload(Notification.camera).joinedload(Camera.site_location_rel).joinedload(SiteLocation.site_hierarchy),
                joinedload(Notification.member).joinedload(Member.department),
            )
            .filter(Notification.id == notification_id)
            .one_or_none()
        )

    @staticmethod
    def list(
        db:                Session,
        status:            Optional[int]      = None,
        type:              Optional[int]      = None,
        camera_id:         Optional[int]      = None,
        member_id:         Optional[int]      = None,
        site_hierarchy_id: Optional[int]      = None,
        date_from:         Optional[datetime] = None,
        date_to:           Optional[datetime] = None,
        page:              int                = 0,
        page_size:         int                = 10,
    ) -> tuple[list, int]:
        stmt = (
            select(Notification)
            .options(
                joinedload(Notification.camera).joinedload(Camera.site_location_rel).joinedload(SiteLocation.site_hierarchy),
                joinedload(Notification.member).joinedload(Member.department),
            )
        )

        if site_hierarchy_id is not None:
            stmt = (
                stmt
                .join(Camera,        Camera.id        == Notification.camera_id)
                .join(SiteLocation,  SiteLocation.id  == Camera.site_location_id)
                .join(SiteHierarchy, SiteHierarchy.id == SiteLocation.site_hierarchy_id)
            )

        filters = []
        if status is not None:
            filters.append(Notification.status == status)
        if type is not None:
            filters.append(Notification.type == type)
        if camera_id is not None:
            filters.append(Notification.camera_id == camera_id)
        if member_id is not None:
            filters.append(Notification.member_id == member_id)
        if site_hierarchy_id is not None:
            filters.append(SiteHierarchy.id == site_hierarchy_id)
        if date_from is not None:
            filters.append(Notification.created_ts >= date_from)
        if date_to is not None:
            filters.append(Notification.created_ts <= date_to)

        if filters:
            stmt = stmt.where(and_(*filters))

        total = db.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar()

        stmt = (
            stmt
            .order_by(
                (Notification.status == 1).desc(),  # unread first
                Notification.created_ts.desc(),
            )
            .offset(page * page_size)
            .limit(page_size)
        )

        rows = db.execute(stmt).unique().scalars().all()
        return rows, total

    @staticmethod
    def update_fields(db: Session, notification: Notification, fields: dict) -> Notification:
        for field, value in fields.items():
            setattr(notification, field, value)
        db.flush()
        return notification

    @staticmethod
    def bulk_delete(db: Session, older_than_days: Optional[int] = None) -> int:
        stmt = delete(Notification)
        if older_than_days is not None:
            cutoff = datetime.utcnow() - timedelta(days=older_than_days)
            stmt = stmt.where(Notification.created_ts < cutoff)
        result = db.execute(stmt)
        return result.rowcount

    # ── NEW: delete selected notifications by IDs ──────────────────────────
    @staticmethod
    def delete_by_ids(db: Session, ids: List[int]) -> int:
        if not ids:
            return 0
        stmt = delete(Notification).where(Notification.id.in_(ids))
        result = db.execute(stmt)
        return result.rowcount

    # ── NEW: mark selected notifications as read by IDs ────────────────────
    @staticmethod
    def mark_read_by_ids(db: Session, ids: List[int]) -> int:
        if not ids:
            return 0
        stmt = (
            update(Notification)
            .where(Notification.id.in_(ids))
            .where(Notification.status != 2)          # skip already-read
            .values(status=2)
        )
        result = db.execute(stmt)
        return result.rowcount

    # ── NEW: mark ALL unread notifications as read ─────────────────────────
    @staticmethod
    def mark_all_read(db: Session) -> int:
        stmt = (
            update(Notification)
            .where(Notification.status == 1)
            .values(status=2)
        )
        result = db.execute(stmt)
        return result.rowcount