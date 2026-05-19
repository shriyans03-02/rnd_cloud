import json
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query
import redis.asyncio as aioredis

from app.core.config import settings
from app.core import redis_listener

router = APIRouter()


async def _verify_ws_ticket(ticket: str) -> bool:
    """
    Look up ticket in Redis. If found, delete it immediately (one-use).
    """
    r = aioredis.from_url(settings.REDIS_URL, decode_responses=True)
    try:
        key = f"ws_ticket:{ticket}"
        user_id = await r.get(key)
        if user_id is None:
            return False
        await r.delete(key)   # one-use: consumed on first connect
        return True
    finally:
        await r.aclose()


@router.websocket("/ws/detections")
async def websocket_detections(
    websocket: WebSocket,
    ticket: str = Query(...),   # /ws/detections?ticket=<ticket>
):
    if not await _verify_ws_ticket(ticket):
        await websocket.close(code=4001)
        return

    await websocket.accept()
    redis_listener.register(websocket)

    await websocket.send_text(json.dumps({
        "event": "connected",
        "message": "Live detection feed active"
    }))

    try:
        while True:
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text(json.dumps({"event": "pong"}))
    except WebSocketDisconnect:
        pass
    finally:
        redis_listener.unregister(websocket)