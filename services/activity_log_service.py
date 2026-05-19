from sqlalchemy.orm import Session
from typing import Any
from datetime import datetime
from app.repositories.activity_log_repo import ActivityLogRepository
from app.schemas.activity_log import ActivityDetail, ActivityLogOut, ActivityChangeItem


class ActivityLogService:

    @staticmethod
    def log(
        db: Session,
        actor_id: int,
        target_type: int,
        target_id: int,
        detail: ActivityDetail,
    ) -> None:
        ActivityLogRepository.create(
            db=db,
            actor_id=actor_id,
            target_type=target_type,
            target_id=target_id,
            details=detail.to_json(),
        )

    @staticmethod
    def _build_summary(action: str, entity: str, name: str | None) -> str:
        entity_label = entity.replace("_", " ").title()
        name_label = f" — {name}" if name else ""
        if action == "create":
            return f"New {entity_label} added{name_label}"
        elif action == "update":
            return f"{entity_label} updated{name_label}"
        elif action == "delete":
            return f"{entity_label} deleted{name_label}"
        elif action == "login":                          # ← add this
            return f"{name or entity_label} logged in"
        elif action == "bulk_import":
            return f"{name}" if name else f"{entity_label} bulk imported"
        elif action == "logout":
            return f"{name or entity_label} logged out"
        return f"{entity_label} {action}{name_label}"

    @staticmethod
    def list_activity_logs(
        db: Session,
        page: int = 0,
        page_size: int = 20,
        date_from: datetime | None = None,
        date_to: datetime | None = None,
        action: str | None = None,
        entity: str | None = None,
    ) -> dict:
        rows, total = ActivityLogRepository.list(
            db=db,
            page=page,
            page_size=page_size,
            date_from=date_from,
            date_to=date_to,
            action=action,
            entity=entity,
        )

        data = []
        for log, actor_name in rows:
            details = log.details or {}
            action_val = details.get("action", "")
            entity_val = details.get("entity", "")
            changes_raw = details.get("changes", {})
            meta = details.get("meta", {})

            # display_name stored at write time — zero DB lookups
            target_name = meta.get("display_name")

            data.append(ActivityLogOut(
                id=log.id,
                actor_name=actor_name or "Unknown",
                action=action_val,
                entity=entity_val,
                summary=ActivityLogService._build_summary(action_val, entity_val, target_name),
                changes=[
                    ActivityChangeItem(field=field, old=values[0], new=values[1])
                    for field, values in changes_raw.items()
                ],
                activity_ts=log.activity_ts,
            ))

        return {
            "data": data,
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    
    @staticmethod
    def bulk_delete(db: Session, older_than_days: int | None = None) -> int:
        try:
            count = ActivityLogRepository.bulk_delete(db, older_than_days=older_than_days)
            db.commit()
            return count
        except Exception:
            db.rollback()
            raise