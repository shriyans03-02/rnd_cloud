from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
# Do not force CUDA_LAUNCH_BLOCKING in production.  That flag is for debugging
# and serializes CUDA kernels.  Set PIPELINE_CUDA_DEBUG_SYNC=1 only when chasing
# a device-side CUDA fault.
if str(os.environ.get("PIPELINE_CUDA_DEBUG_SYNC", "0")).strip().lower() in {"1", "true", "yes", "on"}:
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
os.environ.setdefault("TORCH_CUDAGRAPH_ENABLE", "0")

class NormalizeDuplicateV1Middleware:
    """Normalize accidental duplicate API prefixes from the frontend.

    Some UI builds configure the API base URL with /v1 and also call
    endpoints that start with /v1, producing requests like
    /v1/v1/tracking/webrtc/19.  The real backend route is
    /v1/tracking/webrtc/19, so this middleware rewrites duplicate
    prefixes before routing.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope.get("type") in {"http", "websocket"}:
            path = str(scope.get("path") or "")
            new_path = path
            while new_path == "/v1/v1" or new_path.startswith("/v1/v1/"):
                new_path = "/v1" + new_path[len("/v1/v1"):]
            while new_path == "/api/v1/v1" or new_path.startswith("/api/v1/v1/"):
                new_path = "/api/v1" + new_path[len("/api/v1/v1"):]
            if new_path != path:
                scope = dict(scope)
                scope["path"] = new_path
                scope["raw_path"] = new_path.encode("utf-8")
        await self.app(scope, receive, send)

from app.api.v1.router import v1_router
from app.core.lifespan import lifespan
from app.api.v1.routes.ws_route import router as ws_router       # ← new

app = FastAPI(lifespan=lifespan)

app.add_middleware(NormalizeDuplicateV1Middleware)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://cbt-reid-admin-ui.onrender.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)
app.include_router(ws_router)                             # ← new