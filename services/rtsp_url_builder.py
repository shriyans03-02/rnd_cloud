from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional
from urllib.parse import quote, unquote
from zoneinfo import ZoneInfo

from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.core.constants import (
    CAMERA_RTSP_URL_TEMPLATES,
    NVR_PLAYBACK_RTSP_TEMPLATES,
    NVR_PLAYBACK_TIME_FORMATS,
)


class RTSPTemplateError(ValueError):
    """Raised when a DB RTSP template cannot be used to build a URL."""


@dataclass(frozen=True)
class CameraLiveSourceInfo:
    camera_id: int
    name: str
    ip_address: str
    url: str
    source_type: str
    nvr_id: Optional[int] = None


@dataclass(frozen=True)
class PlaybackSourceInfo:
    camera_id: int
    channel: int
    nvr_id: Optional[int]
    url: str
    start_time: str
    end_time: str
    playback_time_format: str


def _text(value: Any, default: str = "") -> str:
    if value is None:
        return default
    value_s = str(value).strip()
    return value_s if value_s else default


def _int(value: Any, default: int) -> int:
    try:
        if value is None or str(value).strip() == "":
            return int(default)
        return int(value)
    except Exception:
        return int(default)


def _is_rtsp_url(value: str) -> bool:
    return str(value or "").strip().lower().startswith(("rtsp://", "rtsps://"))


def _encode_userinfo(value: Any) -> str:
    """Encode RTSP username/password while accepting already-encoded values."""
    raw = _text(value)
    if not raw:
        return ""
    return quote(unquote(raw), safe="")


def _hikvision_suffix_from_subtype(subtype: Any) -> str:
    subtype_s = _text(subtype, "0").lower()
    if subtype_s in {"main", "mainstream", "primary"}:
        return "01"
    if subtype_s in {"sub", "substream", "secondary"}:
        return "02"
    try:
        return f"{int(subtype_s) + 1:02d}"
    except Exception:
        return "01"


def _build_channel_stream(channel: Any, subtype: Any, explicit_stream: Any = "") -> str:
    explicit = _text(explicit_stream)
    if explicit:
        return explicit
    channel_s = _text(channel, "1")
    if channel_s.isdigit() and int(channel_s) >= 100:
        return channel_s
    return f"{channel_s}{_hikvision_suffix_from_subtype(subtype)}"


def _camera_id(camera: Any) -> int:
    cid = _int(getattr(camera, "id", None), 0)
    if cid <= 0:
        raise RTSPTemplateError("Camera id is missing or invalid.")
    return cid


def _camera_name(camera: Any) -> str:
    cid = _int(getattr(camera, "id", None), 0)
    return _text(getattr(camera, "name", None), f"Camera {cid}" if cid else "Camera")


def _get_nvr(camera: Any) -> Any | None:
    try:
        return getattr(camera, "nvr_rel", None)
    except Exception:
        return None


def _camera_template(camera: Any) -> str:
    brand_rel = getattr(camera, "brand_rel", None)
    brand_template = _text(getattr(brand_rel, "live_rtsp_template", None)) if brand_rel else ""
    if brand_template:
        return brand_template

    template = _text(getattr(camera, "rtsp_url_template", None))
    if template:
        return template

    brand = _text(getattr(camera, "brand", None), "generic").lower()
    return CAMERA_RTSP_URL_TEMPLATES.get(brand) or CAMERA_RTSP_URL_TEMPLATES["generic"]


def _nvr_playback_template(nvr: Any) -> str:
    brand_rel = getattr(nvr, "brand_rel", None)
    brand_template = _text(getattr(brand_rel, "playback_rtsp_template", None)) if brand_rel else ""
    if brand_template:
        return brand_template

    template = _text(getattr(nvr, "playback_rtsp_template", None))
    if template:
        return template

    brand = _text(getattr(nvr, "brand", None), "generic").lower()
    return NVR_PLAYBACK_RTSP_TEMPLATES.get(brand) or NVR_PLAYBACK_RTSP_TEMPLATES["generic"]


def _nvr_playback_time_format(nvr: Any) -> str:
    brand_rel = getattr(nvr, "brand_rel", None)
    fmt = _text(getattr(brand_rel, "playback_time_format", None)) if brand_rel else ""
    if fmt:
        return fmt.lower()
    fmt = _text(getattr(nvr, "playback_time_format", None))
    if fmt:
        return fmt.lower()
    brand = _text(getattr(nvr, "brand", None), "generic").lower()
    return (NVR_PLAYBACK_TIME_FORMATS.get(brand) or "cpplus_local").lower()


def _playback_timezone() -> ZoneInfo:
    tz_name = _text(getattr(settings, "NVR_PLAYBACK_TIMEZONE", None), "Asia/Kolkata")
    try:
        return ZoneInfo(tz_name)
    except Exception:
        return ZoneInfo("Asia/Kolkata")


def _normalize_iso_text(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        raise ValueError("timestamp is required")
    if raw.endswith("Z"):
        raw = raw[:-1] + "+00:00"
    if "T" not in raw and " " in raw:
        raw = raw.replace(" ", "T", 1)
    return raw


def _coerce_datetime(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    raw = _text(value)
    if not raw:
        raise RTSPTemplateError("Playback timestamp is required.")

    for fmt in (
        "%Y%m%dT%H%M%SZ",
        "%Y%m%dT%H%M%S",
        "%Y_%m_%d_%H_%M_%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
    ):
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            pass

    try:
        return datetime.fromisoformat(_normalize_iso_text(raw))
    except Exception as exc:
        raise RTSPTemplateError(f"Playback timestamp is not valid: {value}") from exc


def format_playback_time_for_nvr(value: Any, nvr: Any) -> str:
    fmt = _nvr_playback_time_format(nvr)
    dt = _coerce_datetime(value)
    local_tz = _playback_timezone()

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=local_tz)

    if fmt == "hikvision_utc":
        return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if fmt == "hikvision_local":
        return dt.astimezone(local_tz).strftime("%Y%m%dT%H%M%S")
    if fmt == "cpplus_local":
        return dt.astimezone(local_tz).strftime("%Y_%m_%d_%H_%M_%S")
    if fmt == "iso_local":
        return dt.astimezone(local_tz).strftime("%Y-%m-%dT%H:%M:%S")

    # Safe fallback for unknown values created before validation existed.
    return dt.astimezone(local_tz).strftime("%Y_%m_%d_%H_%M_%S")


def _direct_camera_values(camera: Any) -> Dict[str, Any]:
    cid = _camera_id(camera)
    name = _camera_name(camera)
    ip_address = _text(getattr(camera, "ip_address", None))
    rtsp_port = _int(getattr(camera, "rtsp_port", None), 554)
    rtsp_channel = _int(getattr(camera, "rtsp_channel", None), 1)
    rtsp_subtype = _text(getattr(camera, "rtsp_subtype", None), "0")
    channel_stream = _build_channel_stream(rtsp_channel, rtsp_subtype)
    username = _encode_userinfo(getattr(camera, "rtsp_username", None))
    password = _encode_userinfo(getattr(camera, "rtsp_password", None))
    brand = _text(getattr(camera, "brand", None), "generic").lower()
    nvr_channel = getattr(camera, "nvr_channel", None)

    return {
        "scheme": "rtsp",
        "camera_id": cid,
        "id": cid,
        "camera_name": name,
        "name": name,
        "brand": brand,
        "ip_address": ip_address,
        "ip": ip_address,
        "host": ip_address,
        "rtsp_port": rtsp_port,
        "port": rtsp_port,
        "rtsp_username": username,
        "rtsp_user": username,
        "username": username,
        "user": username,
        "rtsp_password": password,
        "password": password,
        "rtsp_channel": rtsp_channel,
        "channel": rtsp_channel,
        "nvr_channel": nvr_channel if nvr_channel is not None else "",
        "rtsp_subtype": rtsp_subtype,
        "subtype": rtsp_subtype,
        "stream": channel_stream,
        "channel_stream": channel_stream,
        "stream_key": "",
        "nvr_id": getattr(camera, "nvr_id", None),
    }


def _nvr_camera_values(camera: Any, start_time: Any = "", end_time: Any = "") -> Dict[str, Any]:
    """Return template values for NVR playback only."""
    cid = _camera_id(camera)
    name = _camera_name(camera)
    nvr = _get_nvr(camera)
    if nvr is None:
        raise RTSPTemplateError(
            f"Camera {cid} is not linked to an NVR. Set cameras.nvr_id before using NVR playback."
        )

    channel = _int(getattr(camera, "nvr_channel", None), cid)
    nvr_stream_key = _text(getattr(nvr, "stream_key", None))
    rtsp_subtype = _text(getattr(camera, "rtsp_subtype", None), "0")
    channel_stream = _build_channel_stream(channel, rtsp_subtype)

    nvr_ip = _text(getattr(nvr, "ip_address", None))
    nvr_port = _int(getattr(nvr, "port", None), 554)
    nvr_username = _encode_userinfo(getattr(nvr, "username", None))
    nvr_password = _encode_userinfo(getattr(nvr, "password", None))
    nvr_brand = _text(getattr(nvr, "brand", None), "generic").lower()

    return {
        "scheme": "rtsp",
        "camera_id": cid,
        "id": cid,
        "camera_name": name,
        "name": name,
        "camera_ip_address": _text(getattr(camera, "ip_address", None)),
        "camera_ip": _text(getattr(camera, "ip_address", None)),
        "camera_brand": _text(getattr(camera, "brand", None), "generic").lower(),
        "nvr_id": getattr(nvr, "id", None),
        "nvr_name": _text(getattr(nvr, "name", None)),
        "brand": nvr_brand,
        "nvr_brand": nvr_brand,
        "ip_address": nvr_ip,
        "ip": nvr_ip,
        "host": nvr_ip,
        "nvr_ip_address": nvr_ip,
        "nvr_ip": nvr_ip,
        "port": nvr_port,
        "rtsp_port": nvr_port,
        "nvr_port": nvr_port,
        "username": nvr_username,
        "user": nvr_username,
        "rtsp_username": nvr_username,
        "rtsp_user": nvr_username,
        "nvr_username": nvr_username,
        "password": nvr_password,
        "rtsp_password": nvr_password,
        "nvr_password": nvr_password,
        "nvr_channel": channel,
        "channel": channel,
        "rtsp_channel": channel,
        "stream_key": nvr_stream_key,
        "rtsp_subtype": rtsp_subtype,
        "subtype": rtsp_subtype,
        "stream": channel_stream,
        "channel_stream": channel_stream,
        "starttime": _text(start_time),
        "endtime": _text(end_time),
        "start_time": _text(start_time),
        "end_time": _text(end_time),
    }


def _format_db_template(template: str, values: Dict[str, Any], label: str) -> str:
    template = _text(template)
    if not template:
        raise RTSPTemplateError(f"{label} is empty. Store the RTSP template in the database first.")
    try:
        url = template.format(**values)
    except KeyError as exc:
        missing = str(exc.args[0]) if exc.args else "unknown"
        raise RTSPTemplateError(f"{label} uses unknown placeholder {{{missing}}}.") from exc
    except Exception as exc:
        raise RTSPTemplateError(f"{label} could not be formatted: {exc}") from exc
    url = _text(url)
    if not url:
        raise RTSPTemplateError(f"{label} produced an empty RTSP URL.")
    return url


def build_camera_live_rtsp_url(camera: Any) -> str:
    """Build a live RTSP URL from the camera row only.

    Live streaming never uses NVR connection details. NVR fields are used only
    by build_nvr_playback_rtsp_url().
    """
    ip_address = _text(getattr(camera, "ip_address", None))
    if _is_rtsp_url(ip_address):
        return ip_address

    values = _direct_camera_values(camera)
    return _format_db_template(_camera_template(camera), values, "camera live RTSP template")


def build_nvr_playback_rtsp_url(camera: Any, start_time: Any, end_time: Any) -> PlaybackSourceInfo:
    """Build an NVR playback RTSP URL from cameraId + time range.

    Timestamps are formatted using the linked NVR brand's
    device_brands.playback_time_format.
    """
    cid = _camera_id(camera)
    nvr = _get_nvr(camera)
    if nvr is None:
        raise RTSPTemplateError(
            f"Camera {cid} is not linked to an NVR. Set cameras.nvr_id and cameras.nvr_channel."
        )

    template = _nvr_playback_template(nvr)
    if not template:
        raise RTSPTemplateError(
            "NVR playback template is missing. Add playback_rtsp_template in the linked NVR brand."
        )

    fmt = _nvr_playback_time_format(nvr)
    formatted_start = format_playback_time_for_nvr(start_time, nvr)
    formatted_end = format_playback_time_for_nvr(end_time, nvr)

    values = _nvr_camera_values(camera, start_time=formatted_start, end_time=formatted_end)
    url = _format_db_template(template, values, "NVR playback RTSP template")
    return PlaybackSourceInfo(
        camera_id=cid,
        channel=int(values["nvr_channel"]),
        nvr_id=getattr(nvr, "id", None),
        url=url,
        start_time=formatted_start,
        end_time=formatted_end,
        playback_time_format=fmt,
    )


# Override the helper above with explicit model imports. This keeps imports local
# and avoids circular import issues during Alembic model discovery.
def get_camera_with_nvr(db: Session, camera_id: int) -> Any | None:
    from app.db.models.camera import Camera
    from app.db.models.nvr import NVR

    return (
        db.query(Camera)
        .options(
            joinedload(Camera.nvr_rel).joinedload(NVR.brand_rel),
            joinedload(Camera.brand_rel),
        )
        .filter(Camera.id == int(camera_id))
        .first()
    )


def build_playback_source_from_db(db: Session, camera_id: int, start_time: Any, end_time: Any) -> PlaybackSourceInfo:
    camera = get_camera_with_nvr(db, int(camera_id))
    if camera is None:
        raise RTSPTemplateError(f"Camera {camera_id} not found.")
    return build_nvr_playback_rtsp_url(camera, start_time=start_time, end_time=end_time)


def load_active_camera_live_sources(
    db: Session,
    camera_ids: Optional[Iterable[int]] = None,
    only_active: bool = True,
) -> list[CameraLiveSourceInfo]:
    from app.db.models.camera import Camera
    stmt = db.query(Camera).options(joinedload(Camera.brand_rel))
    if camera_ids:
        stmt = stmt.filter(Camera.id.in_([int(x) for x in camera_ids]))
    if only_active:
        stmt = stmt.filter(Camera.is_active.is_(True))

    out: list[CameraLiveSourceInfo] = []
    for cam in stmt.all():
        url = build_camera_live_rtsp_url(cam)
        cid = _camera_id(cam)
        nvr_id = getattr(cam, "nvr_id", None)
        out.append(
            CameraLiveSourceInfo(
                camera_id=cid,
                name=_camera_name(cam),
                ip_address=_text(getattr(cam, "ip_address", None)),
                url=url,
                source_type="camera",
                nvr_id=nvr_id,
            )
        )
    out.sort(key=lambda item: int(item.camera_id))
    return out
