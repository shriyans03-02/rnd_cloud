from __future__ import annotations

import os
import shutil
import socket
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.parse import urlparse

from app.core.config import settings


def _env(name: str, default: str = "") -> str:
    val = os.environ.get(name)
    if val is None:
        try:
            val = getattr(settings, name)
        except Exception:
            val = None
    return str(val if val is not None else default)


def _bool_env(name: str, default: bool = False) -> bool:
    return _env(name, "1" if default else "0").strip().lower() in {"1", "true", "yes", "on", "y"}


def _parse_host_port_from_rtsp(url: str) -> Tuple[str, int]:
    try:
        u = urlparse(str(url))
        return (u.hostname or "127.0.0.1", int(u.port or 8554))
    except Exception:
        return "127.0.0.1", 8554


def wait_for_tcp(host: str, port: int, timeout_s: float = 8.0) -> bool:
    deadline = time.time() + max(0.0, float(timeout_s or 0.0))
    while time.time() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.15)
    return False


class ManagedMediaMTX:
    def __init__(self, proc: Optional[subprocess.Popen] = None):
        self.proc = proc

    @property
    def started_by_us(self) -> bool:
        return self.proc is not None

    def stop(self) -> None:
        proc = self.proc
        self.proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
        except Exception:
            pass


def maybe_start_mediamtx() -> ManagedMediaMTX:
    """Start MediaMTX only when enabled and no RTSP listener is already up."""
    rtsp_base = _env("MEDIAMTX_RTSP", "rtsp://127.0.0.1:8554")
    host, port = _parse_host_port_from_rtsp(rtsp_base)
    wait_s = float(_env("MEDIAMTX_START_WAIT_SECONDS", "8") or 8)

    if wait_for_tcp(host, port, timeout_s=0.5):
        print(f"[MEDIAMTX] already running on {host}:{port}")
        return ManagedMediaMTX(None)

    if not _bool_env("MEDIAMTX_AUTOSTART", True):
        print(f"[MEDIAMTX] not running on {host}:{port}; MEDIAMTX_AUTOSTART=False")
        return ManagedMediaMTX(None)

    config_path = Path(_env("MEDIAMTX_CONFIG_PATH", "mediamtx.yml")).expanduser()
    bin_cfg = _env("MEDIAMTX_BIN", "./mediamtx").strip() or "./mediamtx"
    candidates = []
    if bin_cfg:
        candidates.append(bin_cfg)
    candidates += ["./mediamtx", "mediamtx", "/usr/local/bin/mediamtx", "/usr/bin/mediamtx"]

    bin_path = None
    cwd = str(config_path.parent if config_path.parent.exists() else Path.cwd())
    for cand in candidates:
        if cand.startswith("./"):
            p = (Path(cwd) / cand[2:]).resolve()
            if p.exists() and os.access(str(p), os.X_OK):
                bin_path = str(p)
                break
        else:
            found = shutil.which(cand)
            if found:
                bin_path = found
                break
            p = Path(cand)
            if p.exists() and os.access(str(p), os.X_OK):
                bin_path = str(p)
                break
    if not bin_path:
        print(f"[MEDIAMTX] binary not found; set MEDIAMTX_BIN. Tried: {candidates}")
        return ManagedMediaMTX(None)

    print(f"[MEDIAMTX] starting: {bin_path} {config_path}")
    proc = subprocess.Popen([bin_path, str(config_path)], cwd=cwd)
    if wait_for_tcp(host, port, timeout_s=wait_s):
        print(f"[MEDIAMTX] ready on {host}:{port}")
    else:
        print(f"[MEDIAMTX] WARNING: not reachable on {host}:{port} after {wait_s:.1f}s")
    return ManagedMediaMTX(proc)
