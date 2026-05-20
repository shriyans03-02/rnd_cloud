from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI

from app.core import redis_listener
from app.core.pipeline_args import build_pipeline_argv
from app.core.redis import close_redis, init_redis

try:
    from app.services.tracking.pipeline import parse_args
except Exception:
    from app.services.tracking.pipeline_tracing import parse_args

from app.services.tracking.service import DetectionService, PlaybackTracingService
from app.services.mediamtx_autoconfig import maybe_generate_mediamtx_config


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Starts Redis, redis listener, and the tracking pipeline when Uvicorn starts.
    Stops everything gracefully on shutdown.
    """
    svc: Optional[DetectionService] = None
    playback_svc: Optional[PlaybackTracingService] = None
    redis_task: Optional[asyncio.Task] = None

    try:
        # ---- Redis ----
        await init_redis()
        redis_task = asyncio.create_task(redis_listener.start_redis_listener())
        print("[Lifespan] Redis initialized")

        # ---- MediaMTX dynamic camera paths ----
        # Writes /root/mediamtx.yml from the active cameras table. MediaMTX hot-reloads the file.
        try:
            maybe_generate_mediamtx_config()
        except Exception as e:
            print(f"[MEDIAMTX-AUTO] failed: {e}")

        # ---- Pipeline ----
        argv = build_pipeline_argv()
        args = parse_args(argv)

        svc = DetectionService(args)
        svc.start()                          # ← your Code 2 called .start() explicitly

        playback_svc = PlaybackTracingService(args)

        # ---- Expose to routers via app.state ----
        app.state.detection_service         = svc
        app.state.playback_tracing_service  = playback_svc
        app.state.pipeline_argv             = argv
        app.state.pipeline_args             = args

        print("[Lifespan] Pipeline started")
        yield

    finally:
        # ---- Cancel Redis listener ----
        if redis_task is not None:
            redis_task.cancel()
            try:
                await redis_task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                print(f"[Lifespan] Redis listener stop failed: {e}")

        # ---- Stop playback tracing ----
        if playback_svc is not None:
            try:
                playback_svc.stop_all()
            except Exception as e:
                print(f"[Lifespan] Stop playback tracing failed: {e}")

        # ---- Stop live detection ----
        if svc is not None:
            try:
                svc.stop()
            except Exception as e:
                print(f"[Lifespan] Stop detection service failed: {e}")

        # ---- Close Redis ----
        await close_redis()

        # ---- Clear app state ----
        app.state.detection_service        = None
        app.state.playback_tracing_service = None
        app.state.pipeline_argv            = None
        app.state.pipeline_args            = None

        print("[Lifespan] Shutdown complete")

# from __future__ import annotations

# from contextlib import asynccontextmanager
# from typing import Optional

# from fastapi import FastAPI

# from app.core.pipeline_args import build_pipeline_argv
# from app.services.tracking.pipeline import parse_args
# from app.services.tracking.service import DetectionService


# @asynccontextmanager
# async def lifespan(app: FastAPI):
#     """
#     Starts the tracking pipeline when Uvicorn starts, stops on shutdown.
#     """

#     svc: Optional[DetectionService] = None

#     try:
#         argv = build_pipeline_argv()
#         args = parse_args(argv)

#         svc = DetectionService(args)
#         svc.start()

#         # Expose service to routers
#         app.state.detection_service = svc

#         # Debug info
#         app.state.pipeline_argv = argv
#         app.state.pipeline_args = args

#         yield

#     finally:
#         # ---- Graceful shutdown ----
#         if svc is not None:
#             try:
#                 svc.stop()
#             except Exception as e:
#                 print(f"[lifespan] stop failed: {e}")

#         app.state.detection_service = None
#         app.state.pipeline_argv = None
#         app.state.pipeline_args = None