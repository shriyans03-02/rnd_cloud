from __future__ import annotations
from fastapi import Request, HTTPException
from app.services.tracking.service import DetectionService


def get_detection_service(request: Request) -> DetectionService:
    svc = getattr(request.app.state, "detection_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Detection service not ready")
    return svc
