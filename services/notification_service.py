from fastapi import HTTPException
from typing import Optional
from datetime import datetime
from app.db.session import SessionLocal
from app.db.models.notification import Notification
from app.repositories.notification_repo import NotificationRepository
from app.core.constants import NOTIFICATION_STATUS, NOTIFICATION_TYPE


class NotificationService:

    @staticmethod
    def get_notification(notification_id: int) -> Notification:
        db = SessionLocal()
        try:
            notification = NotificationRepository.get_by_id(db, notification_id)
            if not notification:
                raise HTTPException(status_code=404, detail="Notification not found")
            return notification
        finally:
            db.close()

    @staticmethod
    def list_notifications(
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
        if status is not None and status not in NOTIFICATION_STATUS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid status. Must be one of: {list(NOTIFICATION_STATUS.keys())}",
            )
        if type is not None and type not in NOTIFICATION_TYPE:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid type. Must be one of: {list(NOTIFICATION_TYPE.keys())}",
            )
        db = SessionLocal()
        try:
            return NotificationRepository.list(
                db=db,
                status=status,
                type=type,
                camera_id=camera_id,
                member_id=member_id,
                site_hierarchy_id=site_hierarchy_id,
                date_from=date_from,
                date_to=date_to,
                page=page,
                page_size=page_size,
            )
        finally:
            db.close()

    @staticmethod
    def mark_as_read(notification_id: int) -> Notification:
        db = SessionLocal()
        try:
            notification = NotificationRepository.get_by_id(db, notification_id)
            if not notification:
                raise HTTPException(status_code=404, detail="Notification not found")
            NotificationRepository.update_fields(db, notification, {"status": 2})
            db.commit()
            return NotificationRepository.get_by_id(db, notification_id)
        except HTTPException:
            db.rollback()
            raise
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def bulk_delete(older_than_days: Optional[int] = None) -> int:
        db = SessionLocal()
        try:
            count = NotificationRepository.bulk_delete(db, older_than_days=older_than_days)
            db.commit()
            return count
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ── NEW: delete specific notifications by IDs ──────────────────────────
    @staticmethod
    def delete_selected(ids: list[int]) -> int:
        if not ids:
            raise HTTPException(status_code=400, detail="No notification IDs provided")
        db = SessionLocal()
        try:
            count = NotificationRepository.delete_by_ids(db, ids)
            db.commit()
            return count
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ── NEW: mark specific notifications as read by IDs ────────────────────
    @staticmethod
    def mark_selected_read(ids: list[int]) -> int:
        if not ids:
            raise HTTPException(status_code=400, detail="No notification IDs provided")
        db = SessionLocal()
        try:
            count = NotificationRepository.mark_read_by_ids(db, ids)
            db.commit()
            return count
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()

    # ── NEW: mark ALL notifications as read ────────────────────────────────
    @staticmethod
    def mark_all_read() -> int:
        db = SessionLocal()
        try:
            count = NotificationRepository.mark_all_read(db)
            db.commit()
            return count
        except Exception:
            db.rollback()
            raise
        finally:
            db.close()