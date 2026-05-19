# app/services/ws_service.py

import json
from datetime import datetime
from typing import Optional

from app.core.redis import get_redis

DETECTION_CHANNEL    = "detections"
NOTIFICATION_CHANNEL = "notifications"   # ← new


def _serialize(obj):
    if isinstance(obj, datetime):
        return obj.isoformat()
    raise TypeError(f"Type {type(obj)} not serializable")


async def publish_detection(
    member_id:           Optional[int],
    member_name:         Optional[str],
    guest_temp_id:       Optional[str],
    camera_id:           int,
    camera_name:         str,
    location:            str,
    movement_type:       int,
    entry_ts:            Optional[datetime],
    exit_ts:             Optional[datetime],
    average_match_value: Optional[float],
) -> None:
    redis = get_redis()
    payload = json.dumps({
        "event":               "new_detection",
        "member":              member_name,
        "guest_temp_id":       guest_temp_id,
        "camera_id":           camera_id,
        "camera":              camera_name,
        "location":            location,
        "movement_type":       movement_type,
        "entry_ts":            entry_ts.isoformat() if entry_ts else None,
        "exit_ts":             exit_ts.isoformat()  if exit_ts  else None,
        "average_match_value": average_match_value,
    })
    await redis.publish(DETECTION_CHANNEL, payload)


async def publish_notification(
    notification_id: int,
    type:            int,          # 1 = UNAUTHORIZED_MEMBER, 2 = UNKNOWN_PERSON
    camera_id:       int,
    camera_name:     str,
    location:        str,
    member_id:       Optional[int]   = None,
    member_name:     Optional[str]   = None,
    member_number:   Optional[str]   = None,
    department:      Optional[str]   = None,
    created_ts:      Optional[datetime] = None,
) -> None:
    """
    Publish a new notification event to the Redis notifications channel.
    The WebSocket listener broadcasts this to all connected clients.
    """
    redis = get_redis()
    payload = json.dumps({
        "event":           "new_notification",
        "id":              notification_id,
        "type":            type,
        "camera_id":       camera_id,
        "camera_name":     camera_name,
        "location":        location,
        "member_id":       member_id,
        "member_name":     member_name,
        "member_number":   member_number,
        "department":      department,
        "status":          1,   # always unread when first published
        "created_ts":      (created_ts or datetime.utcnow()).isoformat(),
    }, default=_serialize)
    await redis.publish(NOTIFICATION_CHANNEL, payload)