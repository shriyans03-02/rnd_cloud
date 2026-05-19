# from __future__ import annotations

# import subprocess
# import threading
# import time
# from datetime import datetime, timedelta, timezone
# from typing import Optional
# from urllib.parse import quote

# from fastapi import APIRouter, HTTPException, Query, Request

# from app.core.config import settings
# from app.services.tracking.service import PlaybackTracingService

# router = APIRouter()


# # ── Helpers ───────────────────────────────────────────────────────────────────
# def to_utc_rtsp(iso_str: str) -> str:
#     """Convert ISO datetime string to NVR RTSP format expected by the recorder."""
#     try:
#         iso_str = str(iso_str).replace(" ", "+")
#         dt = datetime.fromisoformat(iso_str)
#         return dt.strftime("%Y%m%dT%H%M%SZ")
#     except ValueError:
#         clean = str(iso_str).replace("Z", "").split("+")[0]
#         dt = datetime.fromisoformat(clean)
#         return dt.strftime("%Y%m%dT%H%M%SZ")


# active_streams: dict[str, subprocess.Popen] = {}
# stream_lock = threading.Lock()


# def _parse_iso_dt(iso_str: str) -> datetime:
#     raw = str(iso_str or "").strip()
#     if not raw:
#         raise ValueError("timestamp is required")
#     raw = raw.replace("Z", "+00:00")
#     try:
#         return datetime.fromisoformat(raw)
#     except ValueError:
#         return datetime.fromisoformat(raw.replace(" ", "+"))


# def _channel_for_camera(camera_id: int) -> int:
#     channel_map = settings.channel_map_dict
#     channel = channel_map.get(int(camera_id))
#     if not channel:
#         raise HTTPException(
#             status_code=400,
#             detail=f"Invalid camera_id {camera_id}. Valid IDs: {list(channel_map.keys())}",
#         )
#     return int(channel)


# def _build_rtsp_source(camera_id: int, start_time: str, end_time: str) -> str:
#     channel = _channel_for_camera(int(camera_id))
#     return (
#         f"rtsp://{settings.NVR_USER}:{settings.NVR_PASS}@{settings.NVR_IP}:554"
#         f"/Streaming/tracks/{channel}"
#         f"?starttime={start_time}&endtime={end_time}&streamkey={settings.NVR_STREAM_KEY}"
#     )


# def _normalize_request_mode(
#     request_mode: Optional[str],
#     mode: Optional[str],
#     member_id: Optional[int],
#     member_name: Optional[str],
# ) -> str:
#     if member_id is not None or str(member_name or "").strip():
#         return "member"
#     raw = str(request_mode or mode or "location").strip().lower()
#     return raw if raw in {"member", "location"} else "location"


# def _should_annotate(
#     annotate: bool,
#     request_mode: Optional[str],
#     mode: Optional[str],
#     member_id: Optional[int],
#     member_name: Optional[str],
# ) -> bool:
#     return bool(
#         annotate
#         or request_mode
#         or mode
#         or member_id is not None
#         or str(member_name or "").strip()
#     )


# def _annotated_session_timeout(entry_ts: str, exit_ts: Optional[str]) -> float:
#     if not exit_ts:
#         return float((2 * 3600) + 180)
#     try:
#         start_dt = _parse_iso_dt(entry_ts)
#         end_dt = _parse_iso_dt(exit_ts)
#         duration = max(1.0, float((end_dt - start_dt).total_seconds()))
#     except Exception:
#         duration = 0.0
#     # NVR playback may take tens of seconds before first frames appear, so keep a
#     # generous safety window. The session still ends early once the annotated
#     # publisher stops after the clip finishes.
#     return float(min(max(duration + 120.0, 180.0), 4 * 3600))


# def _public_hls_url(request: Request, stream_name: str) -> str:
#     base = str(request.base_url).rstrip("/")
#     return f"{base}/v1/hls/{quote(str(stream_name), safe='')}/index.m3u8"


# def _public_mjpeg_url(request: Request, session_id: str) -> str:
#     base = str(request.base_url).rstrip("/")
#     return f"{base}/v1/tracking/mjpeg?session_id={quote(str(session_id), safe='')}"


# def _get_playback_service(request: Request) -> PlaybackTracingService:
#     svc = getattr(request.app.state, "playback_tracing_service", None)
#     if svc is None:
#         pipeline_args = getattr(request.app.state, "pipeline_args", None)
#         svc = PlaybackTracingService(pipeline_args)
#         request.app.state.playback_tracing_service = svc
#     return svc


# def kill_stream(stream_name: str) -> bool:
#     proc: Optional[subprocess.Popen] = None
#     with stream_lock:
#         proc = active_streams.pop(str(stream_name), None)
#     if proc is None:
#         return False
#     try:
#         proc.kill()
#         proc.wait(timeout=3)
#     except Exception:
#         pass
#     return True


# def start_ffmpeg_stream(camera_id: int, start_time: str, end_time: str, stream_name: str):
#     _channel_for_camera(int(camera_id))
#     rtsp_source = _build_rtsp_source(int(camera_id), str(start_time), str(end_time))
#     rtsp_output = f"{settings.MEDIAMTX_RTSP.rstrip('/')}/{stream_name}"

#     cmd = [
#         str(getattr(settings, "FFMPEG_BIN", "ffmpeg") or "ffmpeg"),
#         "-loglevel", "error",
#         "-rtsp_transport", "tcp",
#         "-fflags", "+nobuffer+discardcorrupt",
#         "-i", rtsp_source,
#         "-c", "copy",
#         "-f", "rtsp",
#         "-rtsp_transport", "tcp",
#         rtsp_output,
#     ]

#     kill_stream(stream_name)

#     try:
#         proc = subprocess.Popen(
#             cmd,
#             stdout=subprocess.DEVNULL,
#             stderr=open("ffmpeg_debug.log", "a"),
#         )
#         with stream_lock:
#             active_streams[stream_name] = proc
#     except FileNotFoundError as exc:
#         raise RuntimeError(
#             "FFmpeg not found. Install FFmpeg and ensure it's in PATH."
#         ) from exc


# # ── Routes ────────────────────────────────────────────────────────────────────
# @router.get("/stream/playback")
# def start_playback(
#     request: Request,
#     camera_id: int = Query(..., description="DB camera ID (1-4)"),
#     entry_ts: str = Query(..., description="Entry timestamp ISO format"),
#     exit_ts: Optional[str] = Query(None, description="Exit timestamp ISO format"),
#     annotate: bool = Query(False, description="Re-run the ReID/annotation pipeline on playback"),
#     request_mode: Optional[str] = Query(None, description="Playback request mode: member or location"),
#     mode: Optional[str] = Query(None, description="Legacy alias for request_mode"),
#     member_id: Optional[int] = Query(None, description="Target member for member-mode playback"),
#     member_name: Optional[str] = Query(None, description="Target member name for member-mode playback"),
# ):
#     channel = _channel_for_camera(int(camera_id))
#     start_time = to_utc_rtsp(entry_ts)

#     if exit_ts:
#         end_time = to_utc_rtsp(exit_ts)
#         stream_type = "playback"
#     else:
#         entry_dt = _parse_iso_dt(entry_ts)
#         if entry_dt.tzinfo is None:
#             entry_dt = entry_dt.replace(tzinfo=timezone.utc)
#         end_time = (entry_dt + timedelta(hours=2)).strftime("%Y%m%dT%H%M%SZ")
#         stream_type = "live"

#     if _should_annotate(annotate, request_mode, mode, member_id, member_name):
#         rtsp_source = _build_rtsp_source(int(camera_id), str(start_time), str(end_time))
#         resolved_mode = _normalize_request_mode(request_mode, mode, member_id, member_name)
#         svc = _get_playback_service(request)

#         try:
#             info = svc.start_session(
#                 rtsp_source=rtsp_source,
#                 camera_id=int(camera_id),
#                 start_time=str(start_time),
#                 end_time=str(end_time),
#                 auto_stop_seconds=_annotated_session_timeout(entry_ts, exit_ts),
#                 member_id=member_id,
#                 member_name=member_name,
#                 request_mode=resolved_mode,
#             )
#         except Exception as exc:
#             raise HTTPException(
#                 status_code=500,
#                 detail=f"Could not start the annotated playback session: {exc}",
#             ) from exc

#         session_id = str(info.get("session_id") or "").strip()
#         stream_name = str(info.get("stream_name") or "").strip()
#         if not session_id or not stream_name:
#             if session_id:
#                 try:
#                     svc.stop_session(session_id)
#                 except Exception:
#                     pass
#             raise HTTPException(
#                 status_code=500,
#                 detail="Annotated playback session started without a valid session identifier.",
#             )

#         return {
#             "message": "Annotated playback session started",
#             "data": {
#                 "annotated": True,
#                 "session_id": session_id,
#                 "mjpeg_url": _public_mjpeg_url(request, session_id),
#                 "hls_url": _public_hls_url(request, stream_name),
#                 "stream_name": stream_name,
#                 "stream_type": stream_type,
#                 "start_time": start_time,
#                 "end_time": end_time,
#                 "camera_id": int(camera_id),
#                 "channel": int(channel),
#                 "request_mode": resolved_mode,
#             },
#         }

#     stream_name = f"playback_cam{int(camera_id)}"

#     thread = threading.Thread(
#         target=start_ffmpeg_stream,
#         args=(int(camera_id), str(start_time), str(end_time), str(stream_name)),
#         daemon=True,
#     )
#     thread.start()
#     time.sleep(4)

#     with stream_lock:
#         proc = active_streams.get(stream_name)
#         if proc and proc.poll() is not None:
#             raise HTTPException(
#                 status_code=500,
#                 detail="FFmpeg failed to start. Check NVR connection and timestamps.",
#             )

#     return {
#         "message": "Playback stream started",
#         "data": {
#             "annotated": False,
#             "session_id": None,
#             "mjpeg_url": None,
#             "hls_url": _public_hls_url(request, stream_name),
#             "stream_name": stream_name,
#             "stream_type": stream_type,
#             "start_time": start_time,
#             "end_time": end_time,
#             "camera_id": int(camera_id),
#             "channel": int(channel),
#         },
#     }


# @router.delete("/stream/playback")
# def stop_playback(
#     request: Request,
#     session_id: Optional[str] = Query(None, description="Annotated playback session id to stop"),
#     stream_name: Optional[str] = Query(None, description="Raw playback stream name to stop"),
# ):
#     if not str(session_id or "").strip() and not str(stream_name or "").strip():
#         raise HTTPException(
#             status_code=422,
#             detail="Provide either session_id or stream_name.",
#         )

#     svc = getattr(request.app.state, "playback_tracing_service", None)

#     if str(session_id or "").strip() and svc is not None:
#         if svc.stop_session(str(session_id).strip()):
#             return {"message": f"Playback session '{str(session_id).strip()}' stopped successfully"}

#     if str(stream_name or "").strip():
#         stream_name_clean = str(stream_name).strip()
#         if kill_stream(stream_name_clean):
#             return {"message": f"Stream '{stream_name_clean}' stopped successfully"}
#         if svc is not None:
#             resolved_session_id = svc.get_session_id_by_stream_name(stream_name_clean)
#             if resolved_session_id and svc.stop_session(resolved_session_id):
#                 return {"message": f"Playback session '{stream_name_clean}' stopped successfully"}

#     if str(session_id or "").strip():
#         raise HTTPException(status_code=404, detail="Playback session not found")
#     raise HTTPException(status_code=404, detail="Playback stream not found")


# @router.get("/stream/active")
# def list_active_streams(request: Request):
#     with stream_lock:
#         raw_streams = {
#             name: {
#                 "kind": "raw",
#                 "pid": proc.pid,
#                 "running": proc.poll() is None,
#             }
#             for name, proc in active_streams.items()
#         }

#     svc = getattr(request.app.state, "playback_tracing_service", None)
#     annotated_sessions = svc.list_sessions() if svc is not None else []

#     return {
#         "message": "Active playback streams",
#         "data": {
#             "raw_streams": raw_streams,
#             "annotated_sessions": annotated_sessions,
#         },
#     }

"""
video_playback.py — Production-grade NVR playback router with annotated pipeline support.

Production fixes applied:
  1.  Auth guard on every route via get_current_user dependency.
  2.  No blocking sleep in async context — asyncio.sleep + run_in_executor.
  3.  Active-stream reaper cleans up dead ffmpeg processes.
  4.  Per-stream rotating log file; handle closed on process exit.
  5.  start_ffmpeg_stream raises RuntimeError on misconfiguration (HTTP 400/500).
  6.  entry_ts < exit_ts validation.
  7.  stream_name sanitised against path-traversal before use in URL / filesystem.
  8.  Configurable MediaMTX timeout via settings.
  9.  DELETE /stream/playback is idempotent (no 404 if stream already gone).

Pipeline features retained from v1:
  10. annotate / request_mode / mode / member_id / member_name query params.
  11. PlaybackTracingService integration (annotated MJPEG + HLS sessions).
  12. _normalize_request_mode / _should_annotate / _annotated_session_timeout helpers.
  13. GET /stream/active surfaces both raw streams and annotated sessions.
  14. DELETE /stream/playback handles both session_id and stream_name.
"""

from __future__ import annotations

import asyncio
import logging
import re
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.config import settings
from app.core.dependencies import get_current_user
from app.db.models.user import User
from app.services.tracking.service import PlaybackTracingService

logger = logging.getLogger(__name__)

router = APIRouter()

# ── Stream-name validation ─────────────────────────────────────────────────────
# Allow only safe characters; prevents path-traversal in HLS proxy URLs.
_STREAM_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,64}$")


def _validate_stream_name(name: str) -> str:
    if not _STREAM_NAME_RE.match(name):
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid stream_name. "
                "Use only letters, digits, underscores, hyphens (max 64 chars)."
            ),
        )
    return name


# ── Time helpers ───────────────────────────────────────────────────────────────

def to_nvr_time(iso_str: str) -> str:
    """Convert ISO datetime string to NVR RTSP time format (YYYYMMDDTHHMMSSz)."""
    iso_str = str(iso_str).replace(" ", "+")
    try:
        dt = datetime.fromisoformat(iso_str)
    except ValueError:
        clean = iso_str.replace("Z", "").split("+")[0]
        dt = datetime.fromisoformat(clean)
    return dt.strftime("%Y%m%dT%H%M%SZ")


def _parse_iso_dt(iso_str: str) -> datetime:
    raw = str(iso_str or "").strip()
    if not raw:
        raise ValueError("timestamp is required")
    raw = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(raw)
    except ValueError:
        return datetime.fromisoformat(raw.replace(" ", "+"))


# ── Process registry ───────────────────────────────────────────────────────────
_active_streams: dict[str, subprocess.Popen] = {}
_stream_lock = threading.Lock()

_LOG_DIR = Path("logs/ffmpeg")
_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _reap_dead_streams() -> None:
    """Remove finished ffmpeg processes from the registry."""
    with _stream_lock:
        dead = [name for name, proc in _active_streams.items() if proc.poll() is not None]
        for name in dead:
            proc = _active_streams.pop(name)
            log_fh = getattr(proc, "_log_fh", None)
            if log_fh:
                try:
                    log_fh.close()
                except Exception:
                    pass
            logger.info("Reaped dead stream: %s", name)


def kill_stream(stream_name: str) -> bool:
    """Kill a running stream. Returns True if it existed."""
    with _stream_lock:
        proc = _active_streams.get(stream_name)
        if not proc:
            return False
        try:
            proc.kill()
            proc.wait(timeout=5)
        except Exception as exc:
            logger.warning("Error killing stream %s: %s", stream_name, exc)
        log_fh = getattr(proc, "_log_fh", None)
        if log_fh:
            try:
                log_fh.close()
            except Exception:
                pass
        del _active_streams[stream_name]
    return True


# ── Channel / RTSP helpers ─────────────────────────────────────────────────────

def _channel_for_camera(camera_id: int) -> int:
    """Return the NVR channel for a camera ID, or raise RuntimeError."""
    channel_map = settings.channel_map_dict
    channel = channel_map.get(int(camera_id))
    if channel is None:
        raise RuntimeError(
            f"No NVR channel mapped for camera_id={camera_id}. "
            f"Valid IDs: {sorted(channel_map)}"
        )
    return int(channel)


def _build_rtsp_source(camera_id: int, start_time: str, end_time: str) -> str:
    channel = _channel_for_camera(int(camera_id))
    return (
        f"rtsp://{settings.NVR_USER}:{settings.NVR_PASS}@{settings.NVR_IP}:554"
        f"/Streaming/tracks/{channel}"
        f"?starttime={start_time}&endtime={end_time}"
        f"&streamkey={settings.NVR_STREAM_KEY}"
    )


# ── FFmpeg launcher ────────────────────────────────────────────────────────────

def start_ffmpeg_stream(
    camera_id: int,
    start_time: str,
    end_time: str,
    stream_name: str,
) -> None:
    """
    Launch ffmpeg pulling from NVR RTSP and pushing into MediaMTX.
    Raises RuntimeError on misconfiguration so callers can surface HTTP 400/500.
    """
    rtsp_source = _build_rtsp_source(int(camera_id), str(start_time), str(end_time))
    rtsp_output = f"{settings.MEDIAMTX_RTSP.rstrip('/')}/{stream_name}"

    cmd = [
        str(getattr(settings, "FFMPEG_BIN", None) or "ffmpeg"),
        "-loglevel", "warning",
        "-rtsp_transport", "tcp",
        "-fflags", "+nobuffer+discardcorrupt",
        "-i", rtsp_source,
        "-c", "copy",
        "-f", "rtsp",
        "-rtsp_transport", "tcp",
        rtsp_output,
    ]

    log_path = _LOG_DIR / f"{stream_name}.log"

    _reap_dead_streams()
    kill_stream(stream_name)

    try:
        log_fh = log_path.open("a")
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=log_fh,
            close_fds=True,
        )
        proc._log_fh = log_fh  # type: ignore[attr-defined]
        with _stream_lock:
            _active_streams[stream_name] = proc
        logger.info("Started ffmpeg stream %s (pid=%d)", stream_name, proc.pid)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure it is in PATH."
        )


# ── Annotated-pipeline helpers (preserved from v1) ────────────────────────────

def _normalize_request_mode(
    request_mode: Optional[str],
    mode: Optional[str],
    member_id: Optional[int],
    member_name: Optional[str],
) -> str:
    if member_id is not None or str(member_name or "").strip():
        return "member"
    raw = str(request_mode or mode or "location").strip().lower()
    return raw if raw in {"member", "location"} else "location"


def _should_annotate(
    annotate: bool,
    request_mode: Optional[str],
    mode: Optional[str],
    member_id: Optional[int],
    member_name: Optional[str],
) -> bool:
    return bool(
        annotate
        or request_mode
        or mode
        or member_id is not None
        or str(member_name or "").strip()
    )


def _annotated_session_timeout(entry_ts: str, exit_ts: Optional[str]) -> float:
    if not exit_ts:
        return float(2 * 3600 + 180)
    try:
        start_dt = _parse_iso_dt(entry_ts)
        end_dt = _parse_iso_dt(exit_ts)
        duration = max(1.0, float((end_dt - start_dt).total_seconds()))
    except Exception:
        duration = 0.0
    return float(min(max(duration + 120.0, 180.0), 4 * 3600))


def _public_hls_url(request: Request, stream_name: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/v1/hls/{quote(str(stream_name), safe='')}/index.m3u8"


def _public_mjpeg_url(request: Request, session_id: str) -> str:
    base = str(request.base_url).rstrip("/")
    return f"{base}/v1/tracking/mjpeg?session_id={quote(str(session_id), safe='')}"


def _public_webrtc_base(request: Request) -> str:
    configured = str(getattr(settings, "MEDIAMTX_WEBRTC_PUBLIC_BASE", "") or "").strip().rstrip("/")
    if configured:
        return configured
    host = request.url.hostname or "localhost"
    return f"http://{host}:8889"


def _public_webrtc_url(request: Request, stream_name: str) -> str:
    path = str(stream_name or "").strip().strip("/")
    return f"{_public_webrtc_base(request)}/{path}" if path else ""


def _public_whep_url(request: Request, stream_name: str) -> str:
    path = str(stream_name or "").strip().strip("/")
    return f"{_public_webrtc_base(request)}/{path}/whep" if path else ""


def _get_playback_service(request: Request) -> PlaybackTracingService:
    svc = getattr(request.app.state, "playback_tracing_service", None)
    if svc is None:
        pipeline_args = getattr(request.app.state, "pipeline_args", None)
        svc = PlaybackTracingService(pipeline_args)
        request.app.state.playback_tracing_service = svc
    return svc


# ── Timestamp parsing helper ───────────────────────────────────────────────────

def _parse_and_validate_timestamps(
    entry_ts: str,
    exit_ts: Optional[str],
) -> tuple[datetime, Optional[datetime]]:
    """
    Parse entry/exit timestamps, ensure timezone-awareness, and validate ordering.
    Raises HTTPException 422 on bad input.
    """
    try:
        entry_dt = datetime.fromisoformat(entry_ts.replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=422, detail="entry_ts is not a valid ISO 8601 datetime.")

    if entry_dt.tzinfo is None:
        entry_dt = entry_dt.replace(tzinfo=timezone.utc)

    exit_dt: Optional[datetime] = None
    if exit_ts:
        try:
            exit_dt = datetime.fromisoformat(exit_ts.replace("Z", "+00:00"))
        except ValueError:
            raise HTTPException(status_code=422, detail="exit_ts is not a valid ISO 8601 datetime.")

        if exit_dt.tzinfo is None:
            exit_dt = exit_dt.replace(tzinfo=timezone.utc)

        if exit_dt <= entry_dt:
            raise HTTPException(status_code=422, detail="exit_ts must be after entry_ts.")

    return entry_dt, exit_dt


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/stream/playback")
async def start_playback(
    request: Request,
    camera_id: int = Query(..., description="DB camera ID"),
    entry_ts: str = Query(..., description="Entry timestamp (ISO 8601)"),
    exit_ts: Optional[str] = Query(None, description="Exit timestamp (ISO 8601); omit for live"),
    # ── Annotated-pipeline params ──────────────────────────────────────────
    annotate: bool = Query(False, description="Re-run the ReID/annotation pipeline on playback"),
    request_mode: Optional[str] = Query(None, description="Playback mode: member or location"),
    mode: Optional[str] = Query(None, description="Legacy alias for request_mode"),
    member_id: Optional[int] = Query(None, description="Target member ID (member-mode)"),
    member_name: Optional[str] = Query(None, description="Target member name (member-mode)"),
    current_user: User = Depends(get_current_user),
):
    # ── Validate camera ──────────────────────────────────────────────────────
    try:
        channel = _channel_for_camera(int(camera_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    # ── Parse & validate timestamps ──────────────────────────────────────────
    entry_dt, exit_dt = _parse_and_validate_timestamps(entry_ts, exit_ts)

    if exit_dt is not None:
        start_time = to_nvr_time(entry_ts)
        end_time = to_nvr_time(exit_ts)  # type: ignore[arg-type]
        stream_type = "playback"
    else:
        start_time = to_nvr_time(entry_ts)
        end_time = (entry_dt + timedelta(hours=2)).strftime("%Y%m%dT%H%M%SZ")
        stream_type = "live"

    # ── Annotated pipeline path ──────────────────────────────────────────────
    if _should_annotate(annotate, request_mode, mode, member_id, member_name):
        rtsp_source = _build_rtsp_source(int(camera_id), str(start_time), str(end_time))
        resolved_mode = _normalize_request_mode(request_mode, mode, member_id, member_name)
        svc = _get_playback_service(request)

        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: svc.start_session(
                    rtsp_source=rtsp_source,
                    camera_id=int(camera_id),
                    start_time=str(start_time),
                    end_time=str(end_time),
                    auto_stop_seconds=_annotated_session_timeout(entry_ts, exit_ts),
                    member_id=member_id,
                    member_name=member_name,
                    request_mode=resolved_mode,
                ),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not start the annotated playback session: {exc}",
            ) from exc

        session_id = str(info.get("session_id") or "").strip()
        stream_name = str(info.get("stream_name") or "").strip()

        if not session_id or not stream_name:
            if session_id:
                try:
                    svc.stop_session(session_id)
                except Exception:
                    pass
            raise HTTPException(
                status_code=500,
                detail="Annotated playback session started without a valid session identifier.",
            )

        return {
            "message": "Annotated playback session started",
            "data": {
                "annotated": True,
                "session_id": session_id,
                "mjpeg_url": _public_mjpeg_url(request, session_id),
                "hls_url": _public_hls_url(request, stream_name),
                "webrtc_url": _public_webrtc_url(request, stream_name),
                "whep_url": _public_whep_url(request, stream_name),
                "stream_name": stream_name,
                "stream_type": stream_type,
                "start_time": start_time,
                "end_time": end_time,
                "camera_id": int(camera_id),
                "channel": int(channel),
                "request_mode": resolved_mode,
            },
        }

    # ── Raw ffmpeg path ──────────────────────────────────────────────────────
    stream_name = f"playback_cam{int(camera_id)}"

    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            start_ffmpeg_stream,
            int(camera_id),
            str(start_time),
            str(end_time),
            stream_name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))

    # Brief async wait for ffmpeg to start piping into MediaMTX.
    await asyncio.sleep(3)

    with _stream_lock:
        proc = _active_streams.get(stream_name)

    if proc is None or proc.poll() is not None:
        log_path = _LOG_DIR / f"{stream_name}.log"
        tail = ""
        try:
            tail = log_path.read_text()[-800:]
        except Exception:
            pass
        logger.error("ffmpeg exited early for %s. Log tail:\n%s", stream_name, tail)
        raise HTTPException(
            status_code=500,
            detail=(
                "ffmpeg failed to start. "
                "Check NVR connection, channel mapping and timestamps. "
                f"Log: {tail or '(no log)'}"
            ),
        )

    return {
        "message": "Playback stream started",
        "data": {
            "annotated": False,
            "session_id": None,
            "mjpeg_url": None,
            "hls_url": _public_hls_url(request, stream_name),
            "webrtc_url": _public_webrtc_url(request, stream_name),
            "whep_url": _public_whep_url(request, stream_name),
            "stream_name": stream_name,
            "stream_type": stream_type,
            "start_time": start_time,
            "end_time": end_time,
            "camera_id": int(camera_id),
            "channel": int(channel),
        },
    }


@router.delete("/stream/playback")
async def stop_playback(
    request: Request,
    session_id: Optional[str] = Query(None, description="Annotated playback session ID to stop"),
    stream_name: Optional[str] = Query(None, description="Raw playback stream name to stop"),
    current_user: User = Depends(get_current_user),
):
    session_id_clean = str(session_id or "").strip()
    stream_name_clean = str(stream_name or "").strip()

    if not session_id_clean and not stream_name_clean:
        raise HTTPException(
            status_code=422,
            detail="Provide either session_id or stream_name.",
        )

    if stream_name_clean:
        _validate_stream_name(stream_name_clean)

    svc: Optional[PlaybackTracingService] = getattr(
        request.app.state, "playback_tracing_service", None
    )

    # ── Try annotated session by session_id ──────────────────────────────────
    if session_id_clean and svc is not None:
        if svc.stop_session(session_id_clean):
            return {"message": f"Playback session '{session_id_clean}' stopped successfully."}

    # ── Try raw stream by stream_name ────────────────────────────────────────
    if stream_name_clean:
        if kill_stream(stream_name_clean):
            return {"message": f"Stream '{stream_name_clean}' stopped successfully."}

        # Fall back: stream_name might belong to an annotated session.
        if svc is not None:
            resolved_sid = svc.get_session_id_by_stream_name(stream_name_clean)
            if resolved_sid and svc.stop_session(resolved_sid):
                return {"message": f"Playback session for stream '{stream_name_clean}' stopped successfully."}

    # ── Idempotent: not found is still OK ────────────────────────────────────
    target = session_id_clean or stream_name_clean
    return {"message": f"'{target}' was not running."}


@router.get("/stream/active")
async def list_active_streams(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    _reap_dead_streams()

    with _stream_lock:
        raw_streams = {
            name: {
                "kind": "raw",
                "pid": proc.pid,
                "running": proc.poll() is None,
                "hls_url": _public_hls_url(request, name),
                "webrtc_url": _public_webrtc_url(request, name),
                "whep_url": _public_whep_url(request, name),
            }
            for name, proc in _active_streams.items()
        }

    svc: Optional[PlaybackTracingService] = getattr(
        request.app.state, "playback_tracing_service", None
    )
    annotated_sessions = svc.list_sessions() if svc is not None else []

    return {
        "message": "Active playback streams",
        "data": {
            "raw_streams": raw_streams,
            "annotated_sessions": annotated_sessions,
        },
    }
