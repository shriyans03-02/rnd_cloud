from __future__ import annotations

import os
import shlex
from typing import List

from app.core.config import settings


def build_pipeline_argv() -> List[str]:
    """
    Returns argv list for parse_args().

    Priority:
      1. real process environment variables PIPELINE_ARGS / pipeline_args
      2. .env value loaded by pydantic settings as settings.pipeline_args
    """
    s = (
        os.environ.get("PIPELINE_ARGS")
        or os.environ.get("pipeline_args")
        or getattr(settings, "pipeline_args", "")
        or ""
    )
    s = str(s).strip()
    return shlex.split(s) if s else []
