from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import os
# Force synchronous execution to prevent thread race conditions on the GPU
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
# Disable implicit torch backend graph profiling
os.environ["TORCH_CUDAGRAPH_ENABLE"] = "0"

import torch
# Disable PyTorch's internal benchmarking allocator which breaks multi-threading
torch.backends.cudnn.benchmark = False

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
    allow_origins=["http://100.111.17.5:5173", "https://cbt-reid-admin-ui.onrender.com","http://164.52.214.233:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(v1_router)
app.include_router(ws_router)                             # ← new