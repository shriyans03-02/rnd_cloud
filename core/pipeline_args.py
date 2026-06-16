from __future__ import annotations

import os
import shlex
from typing import List, Sequence

from app.core.config import settings


def _truthy(value: object) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on", "y"}


def _has_option(argv: Sequence[str], option: str) -> bool:
    prefix = option + "="
    return any(str(arg) == option or str(arg).startswith(prefix) for arg in argv)


def build_pipeline_argv() -> List[str]:
    """
    Returns argv list for parse_args().

    Priority:
      1. real process environment variables PIPELINE_ARGS / pipeline_args
      2. .env value loaded by pydantic settings as settings.pipeline_args

    When NVDEC_RESTREAM_ENABLED=True, live AI should read the cleaned H264
    MediaMTX ai/cam<ID> paths, not the original H265 camera/NVR URLs.  Older
    .env files often forgot --db-camera-source-mode mediamtx-ai-id; this helper
    inserts it safely only when the user did not provide a source mode already.
    """
    s = (
        os.environ.get("PIPELINE_ARGS")
        or os.environ.get("pipeline_args")
        or getattr(settings, "pipeline_args", "")
        or ""
    )
    s = str(s).strip()
    argv = shlex.split(s) if s else []

    if _has_option(argv, "--use-db") and not _has_option(argv, "--db-url"):
        db_url = str(os.environ.get("DATABASE_URL") or getattr(settings, "DATABASE_URL", "") or "").strip()
        if db_url:
            argv += ["--db-url", db_url]

    restream_enabled = _truthy(os.environ.get("NVDEC_RESTREAM_ENABLED", getattr(settings, "NVDEC_RESTREAM_ENABLED", False)))
    if restream_enabled and _has_option(argv, "--auto-db-cameras"):
        if not _has_option(argv, "--db-camera-source-mode"):
            argv += ["--db-camera-source-mode", "mediamtx-ai-id"]
        if not _has_option(argv, "--rtsp-transport"):
            argv += ["--rtsp-transport", "tcp"]
    return argv
