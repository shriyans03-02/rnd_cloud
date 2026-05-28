"""
CP Plus NVR playback router with annotated tracing/WebRTC support.

Flow used by the tracing UI:
  UI camera_id -> CP Plus NVR channel=<camera_id> playback RTSP
  -> pipeline_tracing.py draws boxes/names
  -> FFmpeg publishes processed frames to MediaMTX playback_trace_* path
  -> MediaMTX exposes the processed path over WebRTC/HLS.
"""

from __future__ import annotations

import asyncio
import logging
import re
import shlex
import subprocess
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import quote
from zoneinfo import ZoneInfo

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from app.core.config import settings
from app.core.dependencies import get_current_user
from app.db.models.user import User
from app.services.tracking.service import PlaybackTracingService

logger = logging.getLogger(__name__)

router = APIRouter()

_STREAM_NAME_RE = re.compile(r"^[a-zA-Z0-9_\-]{1,96}$")
_CPPLUS_TIME_FORMAT = "%Y_%m_%d_%H_%M_%S"


def _nvr_time_format() -> str:
    fmt = _settings_str("NVR_PLAYBACK_TIME_FORMAT", _CPPLUS_TIME_FORMAT).strip()
    return fmt or _CPPLUS_TIME_FORMAT


def _validate_stream_name(name: str) -> str:
    cleaned = str(name or "").strip()
    if not _STREAM_NAME_RE.match(cleaned):
        raise HTTPException(
            status_code=400,
            detail="Invalid stream_name. Use only letters, digits, underscores, and hyphens.",
        )
    return cleaned


def _settings_str(name: str, default: str = "") -> str:
    try:
        value = getattr(settings, name)
    except Exception:
        value = default
    if value is None:
        return str(default)
    return str(value)


def _settings_bool(name: str, default: bool) -> bool:
    raw = _settings_str(name, "")
    if raw == "":
        return bool(default)
    return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}


def _settings_int(name: str, default: int) -> int:
    try:
        return int(_settings_str(name, str(default)))
    except Exception:
        return int(default)


def _settings_float(name: str, default: float) -> float:
    try:
        return float(_settings_str(name, str(default)))
    except Exception:
        return float(default)


def _split_ffmpeg_flags(value: str) -> list[str]:
    try:
        return shlex.split(str(value or ""))
    except Exception:
        return []


# ── CP Plus timestamp helpers ─────────────────────────────────────────────────

def _nvr_tz() -> Optional[ZoneInfo]:
    tz_name = _settings_str("NVR_PLAYBACK_TIMEZONE", "Asia/Kolkata").strip()
    if not tz_name:
        return None
    try:
        return ZoneInfo(tz_name)
    except Exception:
        logger.warning("Invalid NVR_PLAYBACK_TIMEZONE=%s; using timestamp as supplied", tz_name)
        return None


def _normalise_iso_text(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("timestamp is required")

    # Browser/query-string variants seen in the app:
    #   2026-04-24T11:30:00
    #   2026-04-24T06:00:00.000Z
    #   2026-04-24T11:30:00+05:30
    #   2026-04-24T11:30:00 05:30  (when '+' was decoded as space)
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"

    if "T" not in raw and " " in raw:
        raw = raw.replace(" ", "T", 1)

    if "T" in raw and re.search(r"\s[+\-]?\d{2}:?\d{2}$", raw):
        left, right = raw.rsplit(" ", 1)
        if right and right[0] not in "+-":
            right = "+" + right
        raw = left + right

    return raw


def _parse_iso_dt(value: str) -> datetime:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("timestamp is required")

    # Accept CP Plus formatted values too, useful for direct testing/curl.
    for fmt in (_nvr_time_format(), _CPPLUS_TIME_FORMAT, "%Y%m%dT%H%M%SZ"):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass

    return datetime.fromisoformat(_normalise_iso_text(raw))


def _dt_for_compare(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc)
    tz = _nvr_tz()
    if tz is not None:
        return dt.replace(tzinfo=tz).astimezone(timezone.utc)
    return dt.replace(tzinfo=timezone.utc)


def _format_cpplus_time(dt: datetime) -> str:
    tz = _nvr_tz()
    if dt.tzinfo is not None and tz is not None:
        dt = dt.astimezone(tz)
    return dt.strftime(_nvr_time_format())


def to_nvr_time(value: str) -> str:
    """Return CP Plus playback time: YYYY_MM_DD_HH_MM_SS."""
    return _format_cpplus_time(_parse_iso_dt(value))


def _parse_and_validate_timestamps(
    entry_ts: str,
    exit_ts: Optional[str],
) -> tuple[datetime, Optional[datetime]]:
    try:
        entry_dt = _parse_iso_dt(entry_ts)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"entry_ts is not valid: {exc}") from exc

    exit_dt: Optional[datetime] = None
    if exit_ts:
        try:
            exit_dt = _parse_iso_dt(exit_ts)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"exit_ts is not valid: {exc}") from exc

        if _dt_for_compare(exit_dt) <= _dt_for_compare(entry_dt):
            raise HTTPException(status_code=422, detail="exit_ts must be after entry_ts.")

    return entry_dt, exit_dt


# ── Process registry for raw playback ─────────────────────────────────────────

_active_streams: dict[str, subprocess.Popen] = {}
_stream_lock = threading.Lock()

_LOG_DIR = Path("logs/ffmpeg")
_LOG_DIR.mkdir(parents=True, exist_ok=True)


def _reap_dead_streams() -> None:
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
            logger.info("Reaped dead playback stream: %s", name)


def kill_stream(stream_name: str) -> bool:
    stream_name = _validate_stream_name(stream_name)
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


# ── CP Plus channel / RTSP helpers ────────────────────────────────────────────

def _quote_userinfo(value: str) -> str:
    # safe="%" lets existing encoded .env values such as Admin%40123 pass through
    # while still encoding raw values such as Admin@123 correctly.
    return quote(str(value or ""), safe="%")


def _channel_for_camera(camera_id: int) -> int:
    cam_id = int(camera_id)
    if cam_id <= 0:
        raise RuntimeError("camera_id must be a positive integer.")

    mode = _settings_str("NVR_PLAYBACK_CHANNEL_SOURCE", "camera_id").strip().lower()
    if mode in {"camera_id", "camera", "direct", "id", ""}:
        return cam_id

    channel_map = settings.channel_map_dict
    channel = channel_map.get(cam_id)
    if channel is None:
        raise RuntimeError(
            f"No NVR channel mapped for camera_id={cam_id}. "
            f"Either set NVR_PLAYBACK_CHANNEL_SOURCE=camera_id for CP Plus direct channels "
            f"or add camera_id:channel to CHANNEL_MAP. Valid mapped IDs: {sorted(channel_map)}"
        )
    return int(channel)


def _build_rtsp_source(camera_id: int, start_time: str, end_time: str) -> str:
    channel = _channel_for_camera(int(camera_id))
    path = _settings_str("NVR_PLAYBACK_PATH", "/cam/playback").strip() or "/cam/playback"
    if not path.startswith("/"):
        path = "/" + path

    values = {
        "username": _quote_userinfo(_settings_str("NVR_USER", "admin")),
        "password": _quote_userinfo(_settings_str("NVR_PASS", "")),
        "ip": _settings_str("NVR_IP", "").strip(),
        "port": _settings_str("NVR_PORT", "554").strip() or "554",
        "path": path,
        "channel": int(channel),
        "camera_id": int(camera_id),
        "starttime": str(start_time),
        "endtime": str(end_time),
    }

    if not values["ip"]:
        raise RuntimeError("NVR_IP is required for CP Plus playback.")

    template = _settings_str(
        "NVR_PLAYBACK_URL_TEMPLATE",
        "rtsp://{username}:{password}@{ip}:{port}/cam/playback?channel={channel}&starttime={starttime}&endtime={endtime}",
    ).strip()
    if not template:
        template = "rtsp://{username}:{password}@{ip}:{port}{path}?channel={channel}&starttime={starttime}&endtime={endtime}"

    try:
        return template.format(**values)
    except Exception as exc:
        raise RuntimeError(f"Invalid NVR_PLAYBACK_URL_TEMPLATE: {exc}") from exc


# ── Raw FFmpeg launcher ───────────────────────────────────────────────────────

def start_ffmpeg_stream(
    camera_id: int,
    start_time: str,
    end_time: str,
    stream_name: str,
) -> None:
    stream_name = _validate_stream_name(stream_name)
    rtsp_source = _build_rtsp_source(int(camera_id), str(start_time), str(end_time))
    rtsp_output = f"{settings.MEDIAMTX_RTSP.rstrip('/')}/{stream_name}"

    ffmpeg_bin = str(getattr(settings, "FFMPEG_BIN", None) or "ffmpeg")

    if _settings_bool("PLAYBACK_CLEAN_RESTREAM_ENABLED", True):
        # Raw/non-annotated playback also benefits from the same buffered H265
        # decode -> H264 restream path. The annotated path uses playback_clean_*
        # first and then publishes playback_trace_*; this raw path publishes the
        # cleaned H264 directly to playback_cam<ID>.
        decoder = _settings_str("PLAYBACK_CLEAN_RESTREAM_DECODER", "hevc").strip()
        encoder = _settings_str("PLAYBACK_CLEAN_RESTREAM_ENCODER", "libx264").strip() or "libx264"
        fps = max(1.0, _settings_float("PLAYBACK_CLEAN_RESTREAM_FPS", 8.0))
        width = max(0, _settings_int("PLAYBACK_CLEAN_RESTREAM_WIDTH", 1280))
        height = max(0, _settings_int("PLAYBACK_CLEAN_RESTREAM_HEIGHT", 720))
        bitrate = _settings_str("PLAYBACK_CLEAN_RESTREAM_BITRATE", "8000k") or "4000k"
        bufsize = _settings_str("PLAYBACK_CLEAN_RESTREAM_BUFSIZE", "16000k") or "8000k"
        preset = _settings_str("PLAYBACK_CLEAN_RESTREAM_PRESET", "veryfast") or "veryfast"
        gop = max(1, _settings_int("PLAYBACK_CLEAN_RESTREAM_GOP", 16))
        all_i = _settings_bool("PLAYBACK_CLEAN_RESTREAM_ALL_I", True)
        vf_parts = [f"fps={fps:g}"]
        if width > 0 and height > 0:
            width -= width % 2
            height -= height % 2
            vf_parts.append(f"scale={width}:{height}:flags=bicubic")

        cmd = [ffmpeg_bin, "-hide_banner", "-loglevel", "warning", "-rtsp_transport", "tcp"]
        cmd += _split_ffmpeg_flags(_settings_str(
            "PLAYBACK_CLEAN_RESTREAM_FFMPEG_FLAGS",
            "-fflags +genpts+discardcorrupt -err_detect ignore_err -analyzeduration 10000000 -probesize 10000000 -max_delay 5000000",
        ))
        if decoder and decoder.lower() not in {"auto", "none", "default"}:
            if decoder.lower() in {"hevc_cuvid", "h264_cuvid"}:
                cmd += ["-hwaccel", "cuda", "-c:v", decoder]
            else:
                cmd += ["-c:v", decoder]
        cmd += ["-i", rtsp_source, "-map", "0:v:0", "-an", "-vf", ",".join(vf_parts), "-c:v", encoder]
        if encoder.lower() in {"libx264", "h264"}:
            cmd += [
                "-preset", preset,
                "-tune", "zerolatency",
                "-pix_fmt", "yuv420p",
                "-b:v", bitrate,
                "-maxrate", bitrate,
                "-bufsize", bufsize,
                "-sc_threshold", "0",
                "-bf", "0",
            ]
            if all_i:
                cmd += ["-g", "1", "-keyint_min", "1", "-x264-params", "keyint=1:min-keyint=1:scenecut=0"]
            else:
                cmd += ["-g", str(gop), "-keyint_min", str(gop)]
        cmd += ["-f", "rtsp", "-rtsp_transport", "tcp", rtsp_output]
    else:
        cmd = [
            ffmpeg_bin,
            "-loglevel", "warning",
            "-rtsp_transport", "tcp",
            "-fflags", "+genpts+discardcorrupt",
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
        logger.info("Started CP Plus playback ffmpeg stream %s (pid=%d)", stream_name, proc.pid)
    except FileNotFoundError as exc:
        raise RuntimeError("ffmpeg not found. Install ffmpeg and ensure it is in PATH.") from exc


# ── Annotated-pipeline helpers ────────────────────────────────────────────────

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
        start_dt = _dt_for_compare(_parse_iso_dt(entry_ts))
        end_dt = _dt_for_compare(_parse_iso_dt(exit_ts))
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
    configured = _settings_str("MEDIAMTX_WEBRTC_PUBLIC_BASE", "").strip().rstrip("/")
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


def _response_payload(
    *,
    request: Request,
    annotated: bool,
    session_id: Optional[str],
    stream_name: str,
    stream_type: str,
    start_time: str,
    end_time: str,
    camera_id: int,
    channel: int,
    request_mode: Optional[str] = None,
) -> dict:
    data = {
        "annotated": bool(annotated),
        "session_id": session_id,
        "mjpeg_url": _public_mjpeg_url(request, session_id) if session_id else None,
        "hls_url": _public_hls_url(request, stream_name),
        "webrtc_url": _public_webrtc_url(request, stream_name),
        "whep_url": _public_whep_url(request, stream_name),
        "stream_name": stream_name,
        "stream_type": stream_type,
        "start_time": start_time,
        "end_time": end_time,
        "camera_id": int(camera_id),
        "channel": int(channel),
        "nvr_vendor": "cpplus",
    }
    if request_mode:
        data["request_mode"] = request_mode
    return data


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/stream/playback")
async def start_playback(
    request: Request,
    camera_id: int = Query(..., description="DB camera ID; CP Plus channel uses this same value by default"),
    entry_ts: str = Query(..., description="Entry timestamp"),
    exit_ts: Optional[str] = Query(None, description="Exit timestamp; omit for a 2-hour window"),
    annotate: bool = Query(False, description="Run pipeline_tracing.py and publish processed frames"),
    request_mode: Optional[str] = Query(None, description="Playback mode: member or location"),
    mode: Optional[str] = Query(None, description="Legacy alias for request_mode"),
    member_id: Optional[int] = Query(None, description="Target member ID for member mode"),
    member_name: Optional[str] = Query(None, description="Target member name for member mode"),
    current_user: User = Depends(get_current_user),
):
    try:
        channel = _channel_for_camera(int(camera_id))
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    entry_dt, exit_dt = _parse_and_validate_timestamps(entry_ts, exit_ts)
    start_time = _format_cpplus_time(entry_dt)

    if exit_dt is not None:
        end_time = _format_cpplus_time(exit_dt)
        stream_type = "playback"
    else:
        end_time = _format_cpplus_time(entry_dt + timedelta(hours=2))
        stream_type = "live"

    if _should_annotate(annotate, request_mode, mode, member_id, member_name):
        try:
            rtsp_source = _build_rtsp_source(int(camera_id), start_time, end_time)
        except RuntimeError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        resolved_mode = _normalize_request_mode(request_mode, mode, member_id, member_name)
        svc = _get_playback_service(request)

        try:
            info = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: svc.start_session(
                    rtsp_source=rtsp_source,
                    camera_id=int(camera_id),
                    start_time=start_time,
                    end_time=end_time,
                    auto_stop_seconds=_annotated_session_timeout(entry_ts, exit_ts),
                    member_id=member_id,
                    member_name=member_name,
                    request_mode=resolved_mode,
                ),
            )
        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Could not start the annotated CP Plus playback session: {exc}",
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
                detail="Annotated playback session started without a valid session/stream identifier.",
            )

        return {
            "message": "Annotated CP Plus playback session started",
            "data": _response_payload(
                request=request,
                annotated=True,
                session_id=session_id,
                stream_name=stream_name,
                stream_type=stream_type,
                start_time=start_time,
                end_time=end_time,
                camera_id=int(camera_id),
                channel=int(channel),
                request_mode=resolved_mode,
            ),
        }

    stream_name = f"playback_cam{int(camera_id)}"

    try:
        await asyncio.get_event_loop().run_in_executor(
            None,
            start_ffmpeg_stream,
            int(camera_id),
            start_time,
            end_time,
            stream_name,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
        logger.error("CP Plus ffmpeg exited early for %s. Log tail:\n%s", stream_name, tail)
        raise HTTPException(
            status_code=500,
            detail=(
                "ffmpeg failed to start CP Plus playback. "
                "Check NVR_IP, camera/channel ID, timestamps, and CP Plus playback permissions. "
                f"Log: {tail or '(no log)'}"
            ),
        )

    return {
        "message": "CP Plus playback stream started",
        "data": _response_payload(
            request=request,
            annotated=False,
            session_id=None,
            stream_name=stream_name,
            stream_type=stream_type,
            start_time=start_time,
            end_time=end_time,
            camera_id=int(camera_id),
            channel=int(channel),
        ),
    }


@router.delete("/stream/playback")
async def stop_playback(
    request: Request,
    session_id: Optional[str] = Query(None, description="Annotated playback session ID to stop"),
    stream_name: Optional[str] = Query(None, description="Raw/processed stream name to stop"),
    current_user: User = Depends(get_current_user),
):
    session_id_clean = str(session_id or "").strip()
    stream_name_clean = str(stream_name or "").strip()

    if not session_id_clean and not stream_name_clean:
        raise HTTPException(status_code=422, detail="Provide either session_id or stream_name.")

    if stream_name_clean:
        _validate_stream_name(stream_name_clean)

    svc: Optional[PlaybackTracingService] = getattr(request.app.state, "playback_tracing_service", None)

    if session_id_clean and svc is not None:
        if svc.stop_session(session_id_clean):
            return {"message": f"Playback session '{session_id_clean}' stopped successfully."}

    if stream_name_clean:
        if kill_stream(stream_name_clean):
            return {"message": f"Stream '{stream_name_clean}' stopped successfully."}
        if svc is not None:
            resolved_sid = svc.get_session_id_by_stream_name(stream_name_clean)
            if resolved_sid and svc.stop_session(resolved_sid):
                return {"message": f"Playback session for stream '{stream_name_clean}' stopped successfully."}

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

    svc: Optional[PlaybackTracingService] = getattr(request.app.state, "playback_tracing_service", None)
    annotated_sessions = svc.list_sessions() if svc is not None else []

    return {
        "message": "Active playback streams",
        "data": {
            "raw_streams": raw_streams,
            "annotated_sessions": annotated_sessions,
        },
    }
