from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import quote, unquote

from sqlalchemy import Boolean, Column, Integer, String, create_engine, select
from sqlalchemy.orm import declarative_base, sessionmaker

from app.core.config import settings
from app.db.models.camera import Camera
from app.services.rtsp_url_builder import RTSPTemplateError, build_camera_live_rtsp_url


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    if val is None:
        try:
            val = getattr(settings, name)
        except Exception:
            val = None
    return str(val if val is not None else default)


def _bool_env(name: str, default: bool = False) -> bool:
    val = _env(name, "1" if default else "0").strip().lower()
    return val in {"1", "true", "yes", "on", "y"}


def _quote_yaml(value: str) -> str:
    # Single-quote YAML scalar, escaping single quote by doubling it.
    return "'" + str(value).replace("'", "''") + "'"


def _encode_url_component(value: str) -> str:
    """Accept either raw or already URL-encoded credentials."""
    return quote(unquote(str(value or "")), safe="")


def _parse_channel_map() -> Dict[int, str]:
    result: Dict[int, str] = {}
    raw = _env("CHANNEL_MAP", "")
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair or ":" not in pair:
            continue
        cam_id, channel = pair.split(":", 1)
        try:
            result[int(cam_id.strip())] = channel.strip()
        except ValueError:
            continue
    return result


def _resolve_rtsp_channel(camera_id: int | None = None) -> str:
    source = _env("RTSP_CHANNEL_SOURCE", "env").strip().lower()
    if source in {"camera_id", "id", "camera"} and camera_id is not None:
        return str(camera_id)
    if source in {"channel_map", "map", "mapped"} and camera_id is not None:
        mapped = _parse_channel_map().get(int(camera_id))
        if mapped:
            return str(mapped)
    return _env("RTSP_CHANNEL", "1").strip() or "1"


def _hikvision_suffix_from_subtype(subtype: str) -> str:
    subtype_s = str(subtype or "0").strip().lower()
    if subtype_s in {"main", "mainstream", "primary"}:
        return "01"
    if subtype_s in {"sub", "substream", "secondary"}:
        return "02"
    try:
        # Existing config uses 0 for main stream and 1 for sub stream.
        return f"{int(subtype_s) + 1:02d}"
    except Exception:
        return "01"


def _build_channel_stream(channel: str, subtype: str, explicit_stream: str = "") -> str:
    stream = str(explicit_stream or "").strip()
    if stream:
        return stream

    channel_s = str(channel or "1").strip() or "1"
    # Allow CHANNEL_MAP/RTSP_CHANNEL to contain a complete Hikvision stream id like 101 or 1201.
    if channel_s.isdigit() and int(channel_s) >= 100:
        return channel_s

    return f"{channel_s}{_hikvision_suffix_from_subtype(subtype)}"


def build_camera_rtsp_url(ip: str, camera_id: int | None = None, camera_name: str = "") -> str:
    """Compatibility wrapper.

    Live RTSP URLs are now generated from the full Camera DB row by
    build_camera_live_rtsp_url(). This wrapper only returns full RTSP URLs that
    are already stored in cameras.ip_address; it no longer builds from env
    templates.
    """
    ip = str(ip or "").strip()
    if ip.lower().startswith(("rtsp://", "rtsps://")):
        return ip
    return ""


def load_active_cameras_from_db() -> List[Dict[str, Any]]:
    db_url = str(_env("DATABASE_URL", "")).strip()
    if not db_url:
        raise RuntimeError("DATABASE_URL is empty; cannot generate MediaMTX camera paths")

    engine = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    rows: List[Dict[str, Any]] = []
    try:
        with Session() as session:
            cameras = (
                session.query(Camera)
                .filter(Camera.is_active.is_(True))
                .order_by(Camera.id.asc())
                .all()
            )
            for cam in cameras:
                try:
                    cam_id = int(cam.id)
                except Exception:
                    continue
                ip = str(getattr(cam, "ip_address", "") or "").strip()
                if not ip:
                    continue
                try:
                    source_url = build_camera_live_rtsp_url(cam)
                except RTSPTemplateError as exc:
                    print(f"[MEDIAMTX-AUTO] skipping camera {cam_id}: {exc}")
                    continue
                rows.append(
                    {
                        "id": cam_id,
                        "name": str(getattr(cam, "name", None) or f"Camera {cam_id}"),
                        "ip_address": ip,
                        "rtsp_source": source_url,
                        "source_type": "camera",
                        "nvr_id": getattr(cam, "nvr_id", None),
                    }
                )
    finally:
        try:
            engine.dispose()
        except Exception:
            pass
    rows.sort(key=lambda x: int(x["id"]))
    return rows


def build_mediamtx_config(cameras: List[Dict[str, Any]]) -> str:
    public_ip = _env("MEDIAMTX_PUBLIC_IP", "").strip() or _env("PUBLIC_IP", "").strip() or "164.52.214.233"
    write_queue = _env("MEDIAMTX_WRITE_QUEUE_SIZE", "256")
    source_on_demand = _bool_env("MEDIAMTX_SOURCE_ON_DEMAND", False)
    on_demand_str = "yes" if source_on_demand else "no"
    rtsp_transport = _env("MEDIAMTX_CAMERA_RTSP_TRANSPORT", "tcp") or "tcp"

    # IMPORTANT: sourceOnDemand must NOT be placed in pathDefaults because
    # pathDefaults also applies to publisher paths such as tracked/cam<ID>, ai/cam<ID>,
    # playback_cam<ID>. MediaMTX rejects sourceOnDemand on source: publisher paths.
    lines: List[str] = []
    lines.append("################################################")
    lines.append("# AUTO-GENERATED BY FastAPI from cameras table")
    lines.append("# Do not hard-code live/cam paths here; edit DB cameras instead.")
    lines.append("################################################")
    lines.append("logLevel: info")
    lines.append("logDestinations: [stdout, file]")
    lines.append("logFile: mediamtx.log")
    lines.append("readTimeout: 30s")
    lines.append("writeTimeout: 30s")
    lines.append(f"writeQueueSize: {write_queue}")
    lines.append("")
    lines.append("authMethod: internal")
    lines.append("authInternalUsers:")
    lines.append("  - user: any")
    lines.append("    pass:")
    lines.append("    ips: []")
    lines.append("    permissions:")
    lines.append("      - action: publish")
    lines.append("      - action: read")
    lines.append("      - action: playback")
    lines.append("  - user: any")
    lines.append("    pass:")
    lines.append("    ips: ['127.0.0.1', '::1']")
    lines.append("    permissions:")
    lines.append("      - action: api")
    lines.append("      - action: metrics")
    lines.append("      - action: pprof")
    lines.append("")
    lines.append("api: yes")
    lines.append("apiAddress: :9997")
    lines.append("playback: yes")
    lines.append("")
    lines.append("rtsp: yes")
    lines.append("rtspAddress: :8554")
    lines.append("protocols: [tcp]")
    lines.append("encryption: 'no'")
    lines.append("rtspAuthMethods: [basic]")
    lines.append("")
    lines.append("hls: yes")
    lines.append("hlsAddress: :8888")
    lines.append("hlsAllowOrigin: '*'")
    lines.append("hlsVariant: fmp4")
    lines.append("hlsSegmentCount: 4")
    lines.append("hlsSegmentDuration: 1s")
    lines.append("")
    lines.append("webrtc: yes")
    lines.append("webrtcAddress: :8889")
    lines.append("webrtcEncryption: no")
    lines.append("webrtcLocalUDPAddress: :8189")
    lines.append("webrtcLocalTCPAddress: :8189")
    lines.append(f"webrtcAdditionalHosts: [{_quote_yaml(public_ip)}]")
    lines.append("webrtcAllowOrigin: '*'")
    lines.append("")
    lines.append("rtmp: no")
    lines.append("srt: no")
    lines.append("")
    lines.append("pathDefaults:")
    lines.append("  maxReaders: 0")
    lines.append(f"  rtspTransport: {rtsp_transport}")
    lines.append("")
    lines.append("paths:")

    for cam in cameras:
        cam_id = int(cam["id"])
        name = str(cam.get("name") or f"Camera {cam_id}")
        ip = str(cam.get("ip_address") or "")
        url = str(cam.get("rtsp_source") or "").strip()
        if not url:
            url = build_camera_rtsp_url(ip, camera_id=cam_id, camera_name=name)
        if not url:
            print(f"[MEDIAMTX-AUTO] skipping live/cam{cam_id}: no RTSP source URL built from DB")
            continue
        lines.append(f"  live/cam{cam_id}:")
        lines.append(f"    # {name} | {ip} | source={cam.get('source_type', 'direct')}")
        lines.append(f"    source: {_quote_yaml(url)}")
        lines.append(f"    rtspTransport: {rtsp_transport}")
        if source_on_demand:
            lines.append(f"    sourceOnDemand: {on_demand_str}")
            lines.append("    sourceOnDemandStartTimeout: 15s")
            lines.append("    sourceOnDemandCloseAfter: 5s")
        lines.append("")

    lines.append("  '~^ai/cam[0-9]+$':")
    lines.append("    source: publisher")
    lines.append("  '~^tracked/cam[0-9]+$':")
    lines.append("    source: publisher")
    lines.append("  '~^playback_cam[0-9]+$':")
    lines.append("    source: publisher")
    lines.append("  '~^playback_clean_.*$':")
    lines.append("    source: publisher")
    lines.append("  '~^playback_trace_.*$':")
    lines.append("    source: publisher")
    lines.append("  all_others:")
    lines.append("    source: publisher")
    lines.append("")
    return "\n".join(lines)

def maybe_generate_mediamtx_config() -> Tuple[bool, str]:
    if not _bool_env("MEDIAMTX_AUTOCONFIG", True):
        return False, "MEDIAMTX_AUTOCONFIG disabled"
    path = Path(_env("MEDIAMTX_CONFIG_PATH", "/root/mediamtx.yml")).expanduser()
    keep_existing_empty = _bool_env("MEDIAMTX_KEEP_EXISTING_ON_EMPTY_DB", True)
    try:
        cams = load_active_cameras_from_db()
    except Exception as exc:
        if keep_existing_empty and path.exists():
            msg = f"[MEDIAMTX-AUTO] DB camera load failed ({type(exc).__name__}: {exc}); keeping existing {path}"
            print(msg)
            return False, msg
        raise
    if not cams and keep_existing_empty and path.exists():
        msg = f"[MEDIAMTX-AUTO] no DB cameras found; keeping existing {path}"
        print(msg)
        return False, msg
    data = build_mediamtx_config(cams)
    path.parent.mkdir(parents=True, exist_ok=True)
    old = path.read_text(encoding="utf-8") if path.exists() else ""
    if old == data:
        msg = f"[MEDIAMTX-AUTO] config already up to date: {path} ({len(cams)} cameras)"
        print(msg)
        return False, msg
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(data, encoding="utf-8")
    os.replace(str(tmp), str(path))
    msg = f"[MEDIAMTX-AUTO] wrote {path} with {len(cams)} DB camera paths live/cam<ID>"
    print(msg)
    for cam in cams[:20]:
        print(f"[MEDIAMTX-AUTO] live/cam{int(cam['id'])} <- {cam.get('ip_address')} ({cam.get('name')})")
    if len(cams) > 20:
        print(f"[MEDIAMTX-AUTO] ... {len(cams) - 20} more cameras")
    return True, msg
