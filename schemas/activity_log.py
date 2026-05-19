from pydantic import BaseModel
from typing import Any, Optional
from datetime import datetime


class ActivityDetail(BaseModel):
    action: str
    entity: str
    changes: dict[str, list[Any]]
    meta: dict[str, Any] | None = None

    def to_json(self) -> dict:
        return self.model_dump()


class ActivityChangeItem(BaseModel):
    field: str
    old: Any
    new: Any


class ActivityLogOut(BaseModel):
    id: int
    actor_name: str
    action: str
    entity: str
    summary: str
    changes: list[ActivityChangeItem]
    activity_ts: datetime

    class Config:
        from_attributes = True