# app/core/redis_listener.py

import asyncio
import json
from typing import Set

from fastapi import WebSocket

from app.core.redis import get_redis
from app.services.ws_service import DETECTION_CHANNEL, NOTIFICATION_CHANNEL

_connections: Set[WebSocket] = set()


def register(ws: WebSocket)   -> None: _connections.add(ws)
def unregister(ws: WebSocket) -> None: _connections.discard(ws)
def get_connection_count()    -> int:  return len(_connections)


async def broadcast(message: str) -> None:
    dead = set()
    for ws in _connections:
        try:
            await ws.send_text(message)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _connections.discard(ws)


async def start_redis_listener() -> None:
    print(f"[Redis Listener] Subscribing to: {DETECTION_CHANNEL}, {NOTIFICATION_CHANNEL}")

    while True:
        try:
            redis  = get_redis()
            pubsub = redis.pubsub()
            await pubsub.subscribe(DETECTION_CHANNEL, NOTIFICATION_CHANNEL)  # ← both
            print(f"[Redis Listener] Subscribed.")

            async for message in pubsub.listen():
                if message["type"] != "message":
                    continue
                data = message.get("data", "")
                if data:
                    await broadcast(data)

        except asyncio.CancelledError:
            print("[Redis Listener] Shutting down.")
            break
        except Exception as e:
            print(f"[Redis Listener] Error: {e} — reconnecting in 2s")
            await asyncio.sleep(2)