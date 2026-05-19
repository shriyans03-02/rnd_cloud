from sqlalchemy.orm import Session
from sqlalchemy import select, func, cast, String
from sqlalchemy.types import JSON
from app.db.models.activity_log import ActivityLog
from app.db.models.user import User
from datetime import datetime ,timedelta
from sqlalchemy import delete as sql_delete



class ActivityLogRepository:

    @staticmethod
    def create(
        db: Session,
        actor_id: int,
        target_type: int,
        target_id: int,
        details: dict | None = None,
    ) -> ActivityLog:
        log = ActivityLog(
            actor_id=actor_id,
            target_type=target_type,
            target_id=target_id,
            details=details,
        )
        db.add(log)
        db.flush()
        return log

    @staticmethod
    def list(
        db: Session,
        page: int = 0,
        page_size: int = 20,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        action: str | None = None,
        entity: str | None = None,
    ) -> tuple[list, int]:

        stmt = (
            select(
                ActivityLog,
                (User.first_name + " " + User.last_name).label("actor_name"),
            )
            .outerjoin(User, User.id == ActivityLog.actor_id)
            .order_by(ActivityLog.activity_ts.desc())
        )

        if date_from:
            stmt = stmt.where(ActivityLog.activity_ts >= date_from)
        if date_to:
            stmt = stmt.where(ActivityLog.activity_ts <= date_to)
        if action:
            # DB-agnostic JSON field extraction
            stmt = stmt.where(
                cast(ActivityLog.details["action"], String) == f'"{action}"'
            )
        if entity:
            stmt = stmt.where(
                cast(ActivityLog.details["entity"], String) == f'"{entity}"'
            )

        total = db.execute(
            select(func.count()).select_from(stmt.subquery())
        ).scalar()

        stmt = stmt.offset(page * page_size).limit(page_size)
        rows = db.execute(stmt).all()

        return rows, total
    
    @staticmethod
    def bulk_delete(db: Session, older_than_days: int | None = None) -> int:
        stmt = sql_delete(ActivityLog)
        if older_than_days is not None:
            cutoff = datetime.utcnow() - timedelta(days=older_than_days)
            stmt = stmt.where(ActivityLog.activity_ts < cutoff)
        result = db.execute(stmt)
        return result.rowcount