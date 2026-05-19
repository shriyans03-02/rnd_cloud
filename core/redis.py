import redis.asyncio as aioredis
from app.core.config import settings

# Single shared async Redis client
# Reused by publisher (YOLO) and subscriber (WebSocket listener)
redis_client: aioredis.Redis = None


async def init_redis():
    global redis_client
    redis_client = aioredis.from_url(
        settings.REDIS_URL,          # e.g. redis://localhost:6379
        encoding="utf-8",
        decode_responses=True,
        max_connections=20,
    )
    # Verify connection on startup
    await redis_client.ping()
    print("[Redis] Connected successfully")


async def close_redis():
    global redis_client
    if redis_client:
        await redis_client.aclose()
        print("[Redis] Connection closed")


def get_redis() -> aioredis.Redis:
    return redis_client