from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from app.schemas.common import MessageResponse
from app.services.tracking_console_service import TrackingConsoleService
from app.core.config import settings
from app.db.session import SessionLocal
from app.services.tracking.mjpeg import mjpeg_generator, mjpeg_generator_multi
from app.services.tracking.service import PlaybackTracingService

router = APIRouter()


def _get_playback_service(request: Request) -> Optional[PlaybackTracingService]:
    svc = getattr(request.app.state, "playback_tracing_service", None)
    if svc is None:
        pipeline_args = getattr(request.app.state, "pipeline_args", None)
        if pipeline_args is None:
            return None
        svc = PlaybackTracingService(pipeline_args)
        request.app.state.playback_tracing_service = svc
    return svc


# ─────────────────────────────────────────────────────────────────────────────
#  Existing endpoints (keep as-is)
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/status")
def status(request: Request):
    svc = getattr(request.app.state, "detection_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Detection service not started")
    return svc.status()


def _public_webrtc_base(request: Request) -> str:
    configured = str(getattr(settings, "MEDIAMTX_WEBRTC_PUBLIC_BASE", "") or "").strip().rstrip("/")
    if configured:
        return configured
    host = request.url.hostname or "localhost"
    # MediaMTX WebRTC uses its own HTTP listener, normally 8889.
    return f"http://{host}:8889"


def _parse_cam_ids(raw: Optional[str]) -> Optional[List[int]]:
    if raw is None or not str(raw).strip():
        return None
    try:
        ids = [int(x.strip()) for x in str(raw).split(",") if x.strip()]
    except Exception:
        raise HTTPException(status_code=422, detail="cam_ids must be an integer or comma-separated integers")
    return ids or None


@router.get("/cameras")
def cameras(request: Request):
    svc = getattr(request.app.state, "detection_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Detection service not started")
    data = svc.list_cameras()
    # Refresh public WebRTC URLs using the current request host/config.
    try:
        streams = {int(x.get("camera_id")): x for x in svc.get_webrtc_streams(public_base=_public_webrtc_base(request))}
        for cam in data:
            cam_id = int(cam.get("camera_id") or cam.get("id"))
            if cam_id in streams:
                cam.update(streams[cam_id])
    except Exception:
        pass
    return data


@router.get("/webrtc")
def webrtc_streams(
    request: Request,
    cam_ids: Optional[str] = Query(None, description="Optional camera ID or comma-separated camera IDs"),
):
    """Return processed WebRTC/WHEP URLs for live annotated camera streams."""
    svc = getattr(request.app.state, "detection_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Detection service not started")
    ids = _parse_cam_ids(cam_ids)
    return {
        "message": "Processed WebRTC streams",
        "data": svc.get_webrtc_streams(cam_ids=ids, public_base=_public_webrtc_base(request)),
    }


@router.get("/webrtc/debug")
def webrtc_debug(request: Request):
    """Return live WebRTC publisher diagnostics for troubleshooting UI/MediaMTX."""
    svc = getattr(request.app.state, "detection_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Detection service not started")
    status = svc.status()
    return {
        "message": "Processed WebRTC debug status",
        "data": {
            "mediamtx_webrtc_public_base": _public_webrtc_base(request),
            "mediamtx_rtsp": str(getattr(settings, "MEDIAMTX_RTSP", "") or ""),
            "tracking_webrtc_enabled": bool(getattr(settings, "TRACKING_WEBRTC_ENABLED", True)),
            "active_camera_ids": status.get("camera_ids", []),
            "webrtc_publishers": status.get("webrtc_publishers", []),
        },
    }




@router.get("/webrtc-debug")
def webrtc_debug_alias(request: Request):
    return webrtc_debug(request)


@router.get("/webrtc/status")
def webrtc_status_alias(request: Request):
    return webrtc_debug(request)

@router.get("/webrtc/{camera_id:int}")
def webrtc_stream(camera_id: int, request: Request):
    """Return the processed WebRTC/WHEP URL for one live annotated camera.

    This intentionally returns HTTP 200 even when the camera is not part of the
    currently running live pipeline.  The response contains active_in_pipeline
    and publisher_ready so the UI can decide whether to render the iframe or a
    friendly "stream not active" message instead of showing a hard 404.
    """
    svc = getattr(request.app.state, "detection_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Detection service not started")
    item = svc.get_webrtc_stream(int(camera_id), public_base=_public_webrtc_base(request))
    if not item:
        raise HTTPException(status_code=404, detail=f"Camera {int(camera_id)} could not be resolved")

    active = bool(item.get("active_in_pipeline"))
    ready = bool(item.get("publisher_ready"))
    if ready:
        message = "Processed WebRTC stream is ready"
    elif active:
        message = "Processed WebRTC stream exists but publisher is not ready yet"
    else:
        message = "Camera is not active in the current live pipeline"
    return {"message": message, "data": item}


@router.post("/webrtc/restart")
def restart_webrtc_publishers(request: Request):
    """Restart FFmpeg publishers after MediaMTX/RTSP publishing failures."""
    svc = getattr(request.app.state, "detection_service", None)
    if svc is None:
        raise HTTPException(status_code=503, detail="Detection service not started")
    svc.restart_processed_publishers()
    return {
        "message": "Processed WebRTC publishers restarted",
        "data": svc.get_webrtc_streams(public_base=_public_webrtc_base(request)),
    }


@router.get("/mjpeg")
def mjpeg(
    request: Request,
    cam_ids: Optional[str] = Query(None, description="Live camera ID, or comma-separated camera IDs for processed-only grid"),
    session_id: Optional[str] = Query(None, description="Annotated playback session ID"),
    stream_name: Optional[str] = Query(None, description="Annotated playback stream name"),
    max_fps: int = 15,
    jpeg_quality: int = 85,
    raw_fallback_after_ms: int = Query(0, ge=0, le=5000, description="Deprecated/ignored: stream is processed-only"),
    out_w: int = Query(1280, ge=320, le=3840),
    out_h: int = Query(720, ge=240, le=2160),
    grid_mode: str = Query("cover"),
    grid_rows: int = Query(0, ge=0, le=8),
    grid_cols: int = Query(0, ge=0, le=8),
):
    """Processed-only MJPEG endpoint.

    Live MJPEG intentionally streams only processed/annotated frames.  It does
    not use raw capture fallback, because switching between newer raw frames and
    older processed frames causes the visible forward/backward jitter.
    """
    buf = None

    if str(session_id or "").strip() or str(stream_name or "").strip():
        playback_svc = _get_playback_service(request)
        if playback_svc is None:
            raise HTTPException(status_code=503, detail="Playback tracing service not started")

        resolved_session_id = str(session_id or "").strip()
        if (not resolved_session_id) and str(stream_name or "").strip():
            resolved_session_id = str(
                playback_svc.get_session_id_by_stream_name(str(stream_name).strip()) or ""
            ).strip()
        if not resolved_session_id:
            raise HTTPException(status_code=404, detail="Playback session not found")

        buf = playback_svc.get_session_buffer(resolved_session_id)
        if buf is None:
            raise HTTPException(status_code=404, detail="Playback buffer not available")

        generator = mjpeg_generator(buf, max_fps=max_fps, jpeg_quality=jpeg_quality)
    else:
        if cam_ids is None or not str(cam_ids).strip():
            raise HTTPException(
                status_code=422,
                detail="Provide cam_ids for live MJPEG or session_id/stream_name for playback MJPEG.",
            )
        live_svc = getattr(request.app.state, "detection_service", None)
        if live_svc is None:
            raise HTTPException(status_code=503, detail="Detection service not started")

        try:
            ids = [int(x.strip()) for x in str(cam_ids).split(",") if x.strip()]
        except Exception:
            raise HTTPException(status_code=422, detail="cam_ids must be an integer or comma-separated integers")
        if not ids:
            raise HTTPException(status_code=422, detail="cam_ids is empty")

        bufs = []
        missing = []
        for cid in ids:
            b = live_svc.get_camera_buffer(int(cid))
            if b is None:
                missing.append(int(cid))
            else:
                bufs.append(b)
        if missing:
            raise HTTPException(status_code=404, detail=f"Camera buffer(s) not available: {missing}")

        if len(bufs) == 1:
            generator = mjpeg_generator(bufs[0], max_fps=max_fps, jpeg_quality=jpeg_quality)
        else:
            generator = mjpeg_generator_multi(
                bufs,
                max_fps=max_fps,
                jpeg_quality=jpeg_quality,
                out_w=out_w,
                out_h=out_h,
                grid_mode=grid_mode,
                grid_rows=grid_rows,
                grid_cols=grid_cols,
            )

    return StreamingResponse(
        generator,
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate, max-age=0",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "close",
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Console: member search
#  GET /tracking/console/members?search=john&limit=50
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/console/members", response_model=MessageResponse[list])
def console_list_members(
    search: Optional[str] = Query(None, description="Search by name"),
    limit: int = Query(50, ge=1, le=200),
):
    """
    Returns active members (with complete embeddings) for the Camera Console
    member dropdown. Uses the same condition as /tracing/active/names.
    """
    data = TrackingConsoleService.list_members(search=search, limit=limit)
    return {"message": "Members fetched successfully", "data": data}


# ─────────────────────────────────────────────────────────────────────────────
#  Console: site location search
#  GET /tracking/console/site_locations?search=hall
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/console/site_locations", response_model=MessageResponse[list])
def console_list_site_locations(
    search: Optional[str] = Query(None, description="Search by location name"),
):
    """
    Returns active site locations (fully active hierarchy + at least one active
    camera) for the Camera Console location dropdown.
    Uses the same condition as /tracing/active/site_locations.
    """
    data = TrackingConsoleService.list_site_locations(search=search)
    return {"message": "Site locations fetched successfully", "data": data}


# ─────────────────────────────────────────────────────────────────────────────
#  Console: resolve cameras for selected members
#  GET /tracking/console/resolve/member?member_ids=1&member_ids=2
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/console/resolve/member", response_model=MessageResponse[list])
def resolve_cameras_for_members(
    member_ids: List[int] = Query(..., description="One or more member IDs"),
):
    """
    Resolves which cameras the given members are currently live on.
    Only considers members that pass the same active+embedding filter as
    /tracing/active/names.

    A member is considered live when:
      - movement_type == 1  (entered camera frame)
      - exit_ts IS NULL     (has not exited yet)
    """
    data = TrackingConsoleService.resolve_cameras_for_members(member_ids=member_ids)
    return {"message": "Live cameras resolved for members", "data": data}


# ─────────────────────────────────────────────────────────────────────────────
#  Console: resolve cameras for selected locations
#  GET /tracking/console/resolve/location?site_location_ids=12&site_location_ids=45
# ─────────────────────────────────────────────────────────────────────────────

@router.get("/console/resolve/location", response_model=MessageResponse[list])
def resolve_cameras_for_locations(
    site_location_ids: List[int] = Query(..., description="One or more site location IDs"),
):
    """
    Resolves all active cameras assigned to the given site locations.
    Only considers locations that pass the same fully-active-hierarchy +
    active-camera filter as /tracing/active/site_locations.
    """
    data = TrackingConsoleService.resolve_cameras_for_locations(
        site_location_ids=site_location_ids
    )
    return {"message": "Cameras resolved for locations", "data": data}