from __future__ import annotations

import os
import argparse
from typing import Any, Dict, List, Optional, Union

from app.services.tracking.pipeline import TrackingRunner, parse_pipeline_args, RenderedFrame


class DetectionService:
    """
    FastAPI service wrapper.

    Accepts either:
      - pipeline_args as string (env-style)
      - pipeline_args as argparse.Namespace (already parsed)

    Cameras are started dynamically when MJPEG is requested,
    BUT TrackingRunner also auto-starts all DB cameras at startup
    when in DB mode and --src has no numeric IDs.
    """

    def __init__(self, pipeline_args: Union[str, argparse.Namespace, None] = None):
        self._runner: TrackingRunner | None = None

        if isinstance(pipeline_args, argparse.Namespace):
            self._args: argparse.Namespace | None = pipeline_args
            self._pipeline_args_str: str = ""
        else:
            self._args = None
            s = pipeline_args
            if s is None:
                s = os.environ.get("PIPELINE_ARGS") or os.environ.get("pipeline_args") or ""
            self._pipeline_args_str = str(s).strip()

    def start(self) -> None:
        if self._runner is not None:
            return

        args = self._args if self._args is not None else parse_pipeline_args(self._pipeline_args_str)

        # DB url fallback
        if not getattr(args, "db_url", ""):
            args.db_url = os.environ.get("DATABASE_URL", "") or ""

        self._runner = TrackingRunner(args)
        self._runner.start()

    def stop(self) -> None:
        if self._runner is None:
            return
        try:
            self._runner.stop()
        finally:
            self._runner = None

    def get_camera_buffer(self, cam_id: int) -> Optional[RenderedFrame]:
        """
        Called by the router.
        Auto-starts camera from DB if not already running.
        """
        if self._runner is None:
            self.start()
        assert self._runner is not None
        return self._runner.get_camera_buffer(int(cam_id))

    def list_cameras(self) -> List[Dict[str, Any]]:
        """
        Returns DB cameras (active) with running status.
        """
        if self._runner is None:
            self.start()
        assert self._runner is not None
        return self._runner.list_db_cameras(active_only=True)

    def status(self) -> Dict[str, Any]:
        if self._runner is None:
            self.start()
        assert self._runner is not None
        return self._runner.status()

    def write_report_snapshot(self, path: str | None = None) -> str:
        if self._runner is None:
            self.start()
        assert self._runner is not None
        return self._runner.write_report_snapshot(path=path)
