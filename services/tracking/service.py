from __future__ import annotations

import argparse
import copy
import os
import shlex
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np

from app.core.config import settings
from app.services.tracking.rtsp_publisher import ProcessedFrameRtspPublisher

# Regular live pipeline (continuous multi-camera tracking). This remains
# separate from playback tracing.
try:
    from app.services.tracking.pipeline import (
        RenderedFrame as LiveRenderedFrame,
        TrackingRunner as LiveTrackingRunner,
        parse_pipeline_args as live_parse_pipeline_args,
    )
except Exception:
    from app.services.tracking.pipeline_tracing import (
        RenderedFrame as LiveRenderedFrame,
        TrackingRunner as LiveTrackingRunner,
        parse_pipeline_args as live_parse_pipeline_args,
    )

# Playback tracing pipeline (on-demand, started only when playback is requested).
from app.services.tracking.pipeline_tracing import (
    RenderedFrame,
    TrackingRunner,
    parse_args,
    parse_pipeline_args,
)


class DetectionService:
    """
    FastAPI service wrapper for the regular long-running tracking pipeline.

    Accepts either:
      - pipeline_args as string (env-style)
      - pipeline_args as argparse.Namespace (already parsed)
    """

    def __init__(self, pipeline_args: Union[str, argparse.Namespace, None] = None):
        self._runner: LiveTrackingRunner | None = None
        self._publishers: Dict[int, ProcessedFrameRtspPublisher] = {}
        self._publishers_lock = threading.Lock()

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

        args = self._args if self._args is not None else live_parse_pipeline_args(self._pipeline_args_str)

        if not getattr(args, "db_url", ""):
            args.db_url = os.environ.get("DATABASE_URL", "") or ""

        self._runner = LiveTrackingRunner(args)
        try:
            self._runner.start()
            if bool(getattr(settings, "TRACKING_WEBRTC_AUTOSTART", False)):
                self._start_processed_publishers(args)
            else:
                try:
                    cam_count = len(self._runner.status().get("camera_ids") or [])
                except Exception:
                    cam_count = 0
                print(f"[INIT] Processed WebRTC publishers: lazy/on-demand (active cameras={cam_count}). First /v1/tracking/webrtc/<camera_id> request starts that camera publisher.")
        except Exception:
            try:
                self.stop_processed_publishers()
            except Exception:
                pass
            try:
                self._runner.stop()
            except Exception:
                pass
            self._runner = None
            raise

    @staticmethod
    def _live_stream_name(camera_id: int) -> str:
        prefix = str(getattr(settings, "TRACKING_WEBRTC_PATH_PREFIX", "tracked/cam") or "tracked/cam").strip().strip("/")
        if not prefix:
            prefix = "tracked/cam"
        if "{camera_id}" in prefix or "{cam_id}" in prefix:
            return prefix.format(camera_id=int(camera_id), cam_id=int(camera_id)).strip("/")
        if prefix.endswith("cam"):
            return f"{prefix}{int(camera_id)}"
        return f"{prefix}/cam{int(camera_id)}"

    @staticmethod
    def _webrtc_urls(stream_name: str, public_base: Optional[str] = None) -> Dict[str, str]:
        base = str(public_base or getattr(settings, "MEDIAMTX_WEBRTC_PUBLIC_BASE", "") or getattr(settings, "MEDIAMTX_WEBRTC_INTERNAL", "") or "http://localhost:8889").rstrip("/")
        path = str(stream_name or "").strip().strip("/")
        if not path:
            return {"webrtc_url": "", "whep_url": "", "player_url": ""}
        return {
            "webrtc_url": f"{base}/{path}",
            "whep_url": f"{base}/{path}/whep",
            "player_url": f"{base}/{path}",
        }

    def _make_processed_publisher(self, cam_id: int) -> Optional[ProcessedFrameRtspPublisher]:
        if not bool(getattr(settings, "TRACKING_WEBRTC_ENABLED", True)):
            return None
        if self._runner is None:
            return None
        mediamtx_rtsp = str(getattr(settings, "MEDIAMTX_RTSP", "") or "").strip()
        if not mediamtx_rtsp:
            print("[WARN] TRACKING_WEBRTC_ENABLED is true but MEDIAMTX_RTSP is empty; processed WebRTC publisher not started")
            return None
        try:
            buf = self._runner.get_camera_buffer(int(cam_id))
        except Exception:
            buf = None
        try:
            raw_buf = self._runner.get_camera_raw_buffer(int(cam_id))
        except Exception:
            raw_buf = None
        if buf is None:
            print(f"[WARN] No processed frame buffer for camera_id={cam_id}; WebRTC publisher skipped")
            return None
        stream_name = self._live_stream_name(int(cam_id))
        return ProcessedFrameRtspPublisher(
            buffer=buf,
            raw_buffer=raw_buf,
            stream_name=stream_name,
            ffmpeg_bin=str(getattr(settings, "FFMPEG_BIN", "ffmpeg") or "ffmpeg"),
            mediamtx_rtsp_base=mediamtx_rtsp,
            mode=str(getattr(settings, "TRACKING_WEBRTC_MODE", "hybrid") or "hybrid"),
            fps=float(getattr(settings, "TRACKING_WEBRTC_FPS", 15.0) or 15.0),
            width=int(getattr(settings, "TRACKING_WEBRTC_WIDTH", 852) or 0),
            height=int(getattr(settings, "TRACKING_WEBRTC_HEIGHT", 480) or 0),
            codec=str(getattr(settings, "TRACKING_WEBRTC_CODEC", "auto") or "auto"),
            bitrate=str(getattr(settings, "TRACKING_WEBRTC_BITRATE", "1800k") or "1800k"),
            bufsize=str(getattr(settings, "TRACKING_WEBRTC_BUFSIZE", "360k") or "360k"),
            x264_preset=str(getattr(settings, "TRACKING_WEBRTC_X264_PRESET", "ultrafast") or "ultrafast"),
            overlay_max_age_ms=int(getattr(settings, "TRACKING_WEBRTC_OVERLAY_MAX_AGE_MS", 1500) or 1500),
            gop=int(getattr(settings, "TRACKING_WEBRTC_GOP", 20) or 20),
            draw_stats=bool(getattr(settings, "TRACKING_WEBRTC_DRAW_STATS", True)),
            log_dir=str(getattr(settings, "TRACKING_WEBRTC_LOG_DIR", "logs/ffmpeg_webrtc") or "logs/ffmpeg_webrtc"),
        )

    def ensure_processed_publisher(self, cam_id: int) -> None:
        cam_id = int(cam_id)
        if self._runner is None:
            self.start()
        with self._publishers_lock:
            existing = self._publishers.get(cam_id)
            if existing is not None:
                st = existing.status()
                if bool(st.get("alive")):
                    return
                try:
                    existing.stop()
                except Exception:
                    pass
                self._publishers.pop(cam_id, None)
            max_active = int(getattr(settings, "TRACKING_WEBRTC_MAX_ACTIVE_PUBLISHERS", 0) or 0)
            if max_active > 0 and len(self._publishers) >= max_active:
                # Stop the oldest inserted publisher.  Dict preserves insertion order.
                old_cam_id, old_pub = next(iter(self._publishers.items()))
                self._publishers.pop(old_cam_id, None)
                try:
                    old_pub.stop()
                except Exception:
                    pass
                print(f"[WEBRTC] max active publishers={max_active}; stopped camera_id={old_cam_id} before starting camera_id={cam_id}")
        publisher = self._make_processed_publisher(cam_id)
        if publisher is None:
            return
        try:
            publisher.start()
            with self._publishers_lock:
                old = self._publishers.get(cam_id)
                self._publishers[cam_id] = publisher
            if old is not None:
                try:
                    old.stop()
                except Exception:
                    pass
            print(f"[WEBRTC] on-demand publisher started camera_id={cam_id} mode={publisher.mode} -> {publisher.rtsp_output}")
        except Exception as exc:
            print(f"[WARN] Could not start on-demand WebRTC publisher camera_id={cam_id}: {exc}")

    def _start_processed_publishers(self, args: argparse.Namespace) -> None:
        if not bool(getattr(settings, "TRACKING_WEBRTC_ENABLED", True)):
            return
        if self._runner is None:
            return

        mediamtx_rtsp = str(getattr(settings, "MEDIAMTX_RTSP", "") or "").strip()
        if not mediamtx_rtsp:
            print("[WARN] TRACKING_WEBRTC_ENABLED is true but MEDIAMTX_RTSP is empty; processed WebRTC publishers not started")
            return

        try:
            status = self._runner.status()
            cam_ids = [int(x) for x in (status.get("camera_ids") or [])]
        except Exception:
            cam_ids = []
        if not cam_ids:
            cam_ids = [int(x) for x in (getattr(args, "camera_ids", []) or [])]

        for cam_id in sorted(set(cam_ids)):
            try:
                buf = self._runner.get_camera_buffer(int(cam_id))
            except Exception:
                buf = None
            try:
                raw_buf = self._runner.get_camera_raw_buffer(int(cam_id))
            except Exception:
                raw_buf = None
            if buf is None:
                print(f"[WARN] No processed frame buffer for camera_id={cam_id}; WebRTC publisher skipped")
                continue

            stream_name = self._live_stream_name(int(cam_id))
            publisher = ProcessedFrameRtspPublisher(
                buffer=buf,
                raw_buffer=raw_buf,
                stream_name=stream_name,
                ffmpeg_bin=str(getattr(settings, "FFMPEG_BIN", "ffmpeg") or "ffmpeg"),
                mediamtx_rtsp_base=mediamtx_rtsp,
                mode=str(getattr(settings, "TRACKING_WEBRTC_MODE", "hybrid") or "hybrid"),
                fps=float(getattr(settings, "TRACKING_WEBRTC_FPS", 15.0) or 15.0),
                width=int(getattr(settings, "TRACKING_WEBRTC_WIDTH", 1280) or 0),
                height=int(getattr(settings, "TRACKING_WEBRTC_HEIGHT", 720) or 0),
                codec=str(getattr(settings, "TRACKING_WEBRTC_CODEC", "auto") or "auto"),
                bitrate=str(getattr(settings, "TRACKING_WEBRTC_BITRATE", "3500k") or "3500k"),
                bufsize=str(getattr(settings, "TRACKING_WEBRTC_BUFSIZE", "700k") or "700k"),
                x264_preset=str(getattr(settings, "TRACKING_WEBRTC_X264_PRESET", "superfast") or "superfast"),
                overlay_max_age_ms=int(getattr(settings, "TRACKING_WEBRTC_OVERLAY_MAX_AGE_MS", 1500) or 1500),
                gop=int(getattr(settings, "TRACKING_WEBRTC_GOP", 15) or 15),
                draw_stats=bool(getattr(settings, "TRACKING_WEBRTC_DRAW_STATS", True)),
                log_dir=str(getattr(settings, "TRACKING_WEBRTC_LOG_DIR", "logs/ffmpeg_webrtc") or "logs/ffmpeg_webrtc"),
            )
            try:
                publisher.start()
                with self._publishers_lock:
                    old = self._publishers.get(int(cam_id))
                    self._publishers[int(cam_id)] = publisher
                if old is not None:
                    old.stop()
                print(f"[INIT] Processed WebRTC publisher camera_id={cam_id} mode={publisher.mode} -> {publisher.rtsp_output}")
            except Exception as exc:
                print(f"[WARN] Could not start processed WebRTC publisher camera_id={cam_id}: {exc}")

    def restart_processed_publishers(self) -> None:
        if self._runner is None:
            self.start()
            return
        self.stop_processed_publishers()
        args = self._args if self._args is not None else live_parse_pipeline_args(self._pipeline_args_str)
        self._start_processed_publishers(args)

    def stop_processed_publishers(self) -> None:
        with self._publishers_lock:
            pubs = list(self._publishers.values())
            self._publishers.clear()
        for pub in pubs:
            try:
                pub.stop()
            except Exception:
                pass

    def stop(self) -> None:
        self.stop_processed_publishers()
        if self._runner is None:
            return
        try:
            self._runner.stop()
        finally:
            self._runner = None

    def get_camera_buffer(self, cam_id: int) -> Optional[LiveRenderedFrame]:
        if self._runner is None:
            self.start()
        assert self._runner is not None
        return self._runner.get_camera_buffer(int(cam_id))

    def get_camera_raw_buffer(self, cam_id: int) -> Optional[LiveRenderedFrame]:
        if self._runner is None:
            self.start()
        assert self._runner is not None
        return self._runner.get_camera_raw_buffer(int(cam_id))

    def list_cameras(self) -> List[Dict[str, Any]]:
        if self._runner is None:
            self.start()
        assert self._runner is not None
        out = self._runner.list_db_cameras(active_only=True)
        streams = {int(x.get("camera_id")): x for x in self.get_webrtc_streams()}
        for cam in out:
            try:
                cam_id = int(cam.get("camera_id") or cam.get("id"))
                if cam_id in streams:
                    cam.update(streams[cam_id])
            except Exception:
                pass
        return out

    def get_webrtc_streams(
        self,
        cam_ids: Optional[List[int]] = None,
        public_base: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        if self._runner is None:
            self.start()
        assert self._runner is not None

        try:
            active_ids = {int(x) for x in (self._runner.status().get("camera_ids") or [])}
        except Exception:
            active_ids = set()

        if cam_ids is None:
            cam_ids = sorted(active_ids)

        out: List[Dict[str, Any]] = []
        with self._publishers_lock:
            publishers = dict(self._publishers)
        for cam_id in sorted(set(int(x) for x in (cam_ids or []))):
            stream_name = self._live_stream_name(int(cam_id))
            pub = publishers.get(int(cam_id))
            pub_status = pub.status() if pub is not None else None
            publisher_ready = bool(pub_status and pub_status.get("path_ready", pub_status.get("ready")))
            processed_frames_ready = bool(pub_status and pub_status.get("processed_ready"))
            raw_motion_ready = bool(pub_status and pub_status.get("raw_motion_ready"))
            overlay_ready = bool(pub_status and pub_status.get("overlay_ready"))
            publisher_alive = bool(pub_status and pub_status.get("alive"))
            active = int(cam_id) in active_ids
            if overlay_ready:
                status_reason = "hybrid_raw_video_with_ai_overlay_ready"
            elif raw_motion_ready:
                status_reason = "hybrid_raw_video_ready_waiting_for_ai_boxes"
            elif processed_frames_ready:
                status_reason = "processed_frames_ready"
            elif publisher_ready:
                status_reason = "publisher_path_ready_waiting_for_frames"
            elif active and pub is not None:
                status_reason = "publisher_waiting_for_ffmpeg_or_first_bootstrap_frame"
            elif active:
                status_reason = "active_but_publisher_missing"
            else:
                status_reason = "camera_not_in_current_pipeline_args"
            info: Dict[str, Any] = {
                "camera_id": int(cam_id),
                "stream_name": stream_name,
                "rtsp_publish_url": f"{str(getattr(settings, 'MEDIAMTX_RTSP', '')).rstrip('/')}/{stream_name}",
                "webrtc_enabled": bool(getattr(settings, "TRACKING_WEBRTC_ENABLED", True)),
                **self._webrtc_urls(stream_name, public_base=public_base),
                "active_in_pipeline": bool(active),
                "publisher_ready": bool(publisher_ready),
                "processed_frames_ready": bool(processed_frames_ready),
                "raw_motion_ready": bool(raw_motion_ready),
                "overlay_ready": bool(overlay_ready),
                "publisher_alive": bool(publisher_alive),
                "available": bool(active and publisher_ready),
                "status_reason": status_reason,
                "publisher": pub_status,
            }
            out.append(info)
        return out

    def get_webrtc_stream(self, cam_id: int, public_base: Optional[str] = None) -> Dict[str, Any]:
        # Start the FFmpeg/MediaMTX publisher lazily for exactly the camera the UI opened.
        # This avoids running 12+ CPU encoders at backend startup.
        try:
            self.ensure_processed_publisher(int(cam_id))
        except Exception as exc:
            print(f"[WEBRTC] ensure publisher failed camera_id={int(cam_id)}: {exc}")
        streams = self.get_webrtc_streams(cam_ids=[int(cam_id)], public_base=public_base)
        return streams[0] if streams else {}

    def status(self) -> Dict[str, Any]:
        if self._runner is None:
            self.start()
        assert self._runner is not None
        st = self._runner.status()
        st["webrtc_publishers"] = self.get_webrtc_streams()
        return st

    def write_report_snapshot(self, path: str | None = None) -> str:
        if self._runner is None:
            self.start()
        assert self._runner is not None
        return self._runner.write_report_snapshot(path=path)


class AnnotatedRtspPublisher:
    """
    Publish annotated frames from RenderedFrame into MediaMTX as RTSP.

    MediaMTX can then expose that RTSP path as HLS, so the frontend can keep
    using hls_url for detected playback.
    """

    def __init__(
        self,
        *,
        buffer: RenderedFrame,
        runner: TrackingRunner,
        stream_name: str,
        ffmpeg_bin: str,
        mediamtx_rtsp_base: str,
        fps: float = 20.0,
        codec: str = "libx264",
        debug_log_path: str = "ffmpeg_annotated_debug.log",
    ):
        self.buffer = buffer
        self.runner = runner
        self.stream_name = str(stream_name)
        self.ffmpeg_bin = str(ffmpeg_bin or "ffmpeg")
        self.fps = float(max(1.0, float(fps or 20.0)))
        self.codec = str(codec or "libx264")
        self.debug_log_path = str(debug_log_path or "ffmpeg_annotated_debug.log")
        self.rtsp_output = f"{str(mediamtx_rtsp_base).rstrip('/')}/{self.stream_name}"

        self._stop_evt = threading.Event()
        self._ready_evt = threading.Event()
        self._done_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_handle = None
        self._error: str = ""
        self._size: Optional[tuple[int, int]] = None
        self._started_writes: int = 0
        self._real_frames_written: int = 0
        self._placeholder_frames_written: int = 0
        self._held_frames_written: int = 0
        self._last_seq: int = -1
        self._last_good_frame: Optional[np.ndarray] = None
        self._hold_last_frame: bool = self._env_bool("PLAYBACK_HOLD_LAST_FRAME", True)
        self._placeholder_before_first_only: bool = self._env_bool(
            "PLAYBACK_PLACEHOLDER_BEFORE_FIRST_FRAME_ONLY", True
        )
        self._bootstrap_placeholder: bool = self._env_bool("PLAYBACK_BOOTSTRAP_PLACEHOLDER", True)

    @staticmethod
    def _even_size(w: int, h: int) -> tuple[int, int]:
        ww = max(2, int(w) - (int(w) % 2))
        hh = max(2, int(h) - (int(h) % 2))
        return ww, hh

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(str(name))
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}

    def _build_cmd(self, w: int, h: int) -> list[str]:
        fps_txt = f"{self.fps:.3f}"
        try:
            keyint = int(os.environ.get("PLAYBACK_WEBRTC_GOP", "") or 0)
        except Exception:
            keyint = 0
        if keyint <= 0:
            keyint = max(1, int(round(self.fps * 2.0)))

        preset = str(os.environ.get("PLAYBACK_WEBRTC_PRESET", "ultrafast") or "ultrafast").strip()
        bitrate = str(os.environ.get("PLAYBACK_WEBRTC_BITRATE", "5000k") or "5000k").strip()
        bufsize = str(os.environ.get("PLAYBACK_WEBRTC_BUFSIZE", "10000k") or "10000k").strip()
        codec = str(self.codec or "libx264").strip() or "libx264"
        codec_lower = codec.lower()

        cmd = [
            self.ffmpeg_bin,
            "-loglevel", "error",
            "-fflags", "+genpts",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{int(w)}x{int(h)}",
            "-r", fps_txt,
            "-i", "-",
            "-an",
            "-c:v", codec,
        ]

        if codec_lower in {"h264_nvenc", "hevc_nvenc"}:
            # NVENC uses the GPU video encoder block and removes the CPU x264
            # bottleneck from playback_trace_* publishing. Keep it optional;
            # libx264 remains the fallback for machines without NVENC.
            nv_preset = preset
            if nv_preset in {"ultrafast", "superfast", "veryfast", "faster", "fast", "medium", "slow"}:
                nv_preset = "p2" if nv_preset in {"ultrafast", "superfast"} else "p3"
            cmd += [
                "-preset", nv_preset,
                "-tune", "ll",
                "-pix_fmt", "yuv420p",
                "-b:v", bitrate,
                "-maxrate", bitrate,
                "-bufsize", bufsize,
                "-g", str(keyint),
                "-bf", "0",
            ]
        else:
            cmd += [
                "-preset", preset,
                "-tune", "zerolatency",
                "-profile:v", "baseline",
                "-level:v", "3.1",
                "-pix_fmt", "yuv420p",
                "-b:v", bitrate,
                "-maxrate", bitrate,
                "-bufsize", bufsize,
                "-g", str(keyint),
                "-keyint_min", str(keyint),
                "-sc_threshold", "0",
                "-bf", "0",
            ]

        cmd += [
            "-muxdelay", "0",
            "-muxpreload", "0",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_output,
        ]
        return cmd

    def _start_proc(self, w: int, h: int) -> None:
        os.makedirs(os.path.dirname(self.debug_log_path) or ".", exist_ok=True)
        self._stderr_handle = open(self.debug_log_path, "a", buffering=1)
        cmd = self._build_cmd(int(w), int(h))
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_handle,
                bufsize=0,
            )
        except Exception:
            try:
                if self._stderr_handle is not None:
                    self._stderr_handle.close()
            except Exception:
                pass
            self._stderr_handle = None
            raise

    def _close_proc(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                if proc.stdin is not None:
                    proc.stdin.close()
            except Exception:
                pass
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._stderr_handle is not None:
            try:
                self._stderr_handle.close()
            except Exception:
                pass
            self._stderr_handle = None

    def _runner_running(self) -> bool:
        try:
            st = self.runner.status()
            return bool(st.get("running", False))
        except Exception:
            return True

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("Empty frame")
        fh, fw = frame.shape[:2]
        if fh <= 1 or fw <= 1:
            raise ValueError("Invalid frame size")
        out_w, out_h = self._even_size(fw, fh)
        if self._size is None:
            self._size = (out_w, out_h)
        else:
            out_w, out_h = self._size
        if frame.shape[1] != out_w or frame.shape[0] != out_h:
            frame = cv2.resize(frame, (int(out_w), int(out_h)), interpolation=cv2.INTER_LINEAR)
        return np.ascontiguousarray(frame)

    def _make_placeholder_frame(self) -> np.ndarray:
        if self._size is None:
            w = int(getattr(settings, "TRACKING_WEBRTC_WIDTH", 852) or 852)
            h = int(getattr(settings, "TRACKING_WEBRTC_HEIGHT", 480) or 480)
            self._size = self._even_size(w, h)
        w, h = self._size
        frame = np.zeros((int(h), int(w), 3), dtype=np.uint8)
        cv2.putText(
            frame,
            "Waiting for CP Plus playback frames...",
            (24, max(40, int(h // 2))),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (220, 220, 220),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            self.stream_name,
            (24, max(72, int(h // 2) + 34)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (160, 160, 160),
            1,
            cv2.LINE_AA,
        )
        return np.ascontiguousarray(frame)

    def _ensure_proc_for_frame(self, frame_bgr: np.ndarray) -> bool:
        if self._proc is None:
            try:
                self._start_proc(frame_bgr.shape[1], frame_bgr.shape[0])
            except Exception as exc:
                self._error = f"ffmpeg start failed: {exc}"
                return False
        if self._proc is None or self._proc.stdin is None:
            self._error = "ffmpeg stdin not available"
            return False
        if self._proc.poll() is not None:
            self._error = f"ffmpeg exited early with code {self._proc.returncode}"
            return False
        return True

    def _write_frame_to_proc(
        self,
        frame_bgr: np.ndarray,
        *,
        source: str = "real",
        mark_ready: bool = True,
    ) -> bool:
        if not self._ensure_proc_for_frame(frame_bgr):
            return False
        assert self._proc is not None and self._proc.stdin is not None
        try:
            self._proc.stdin.write(frame_bgr.tobytes())
        except (BrokenPipeError, OSError) as exc:
            self._error = f"ffmpeg pipe failed: {exc}"
            return False

        self._started_writes += 1
        src = str(source or "real").strip().lower()
        if src == "placeholder":
            self._placeholder_frames_written += 1
        elif src == "held":
            self._held_frames_written += 1
        else:
            self._real_frames_written += 1

        if mark_ready:
            self._ready_evt.set()
        return True

    def _loop(self) -> None:
        idle_loops = 0
        max_idle_loops = max(20, int(self.fps * 10.0))
        wait_timeout = min(0.5, max(0.02, 1.0 / max(1.0, float(self.fps))))
        self.buffer.add_client()
        try:
            # Start the MediaMTX publisher path immediately, but mark the
            # session ready only after the first real processed frame. The
            # placeholder is allowed before the decoder/model produces output;
            # after that, gaps repeat the last good frame instead of flashing
            # back to black. This fixes CP Plus playback flicker when CPU
            # annotation is slower than the WebRTC output cadence.
            if self._bootstrap_placeholder:
                try:
                    placeholder = self._prepare_frame(self._make_placeholder_frame())
                    self._write_frame_to_proc(placeholder, source="placeholder", mark_ready=False)
                except Exception:
                    pass

            while not self._stop_evt.is_set():
                try:
                    frame, _ts, _meta, seq = self.buffer.wait_for_seq(self._last_seq, timeout=wait_timeout)
                except Exception as exc:
                    self._error = f"buffer wait failed: {exc}"
                    break

                if self._stop_evt.is_set():
                    break

                if frame is None or int(seq) <= int(self._last_seq):
                    idle_loops += 1

                    # Do not publish a black waiting frame after real playback
                    # has already begun. Repeating the previous real frame keeps
                    # WebRTC's RTP cadence stable and avoids blink/flicker.
                    if self._proc is not None and self._proc.stdin is not None and self._proc.poll() is None:
                        try:
                            if self._last_good_frame is not None and self._hold_last_frame:
                                held = np.ascontiguousarray(self._last_good_frame)
                                if not self._write_frame_to_proc(held, source="held", mark_ready=True):
                                    break
                            elif (not self._placeholder_before_first_only) or self._real_frames_written <= 0:
                                placeholder = self._prepare_frame(self._make_placeholder_frame())
                                if not self._write_frame_to_proc(placeholder, source="placeholder", mark_ready=False):
                                    break
                        except Exception:
                            pass

                    if idle_loops >= max_idle_loops:
                        # Do not tear the session down just because annotated
                        # playback has a long decode gap. NVR playback often
                        # starts mid-GOP, so OpenCV can sit on decoder errors
                        # until the next clean keyframe arrives. The session is
                        # cleaned by DELETE or auto-stop.
                        if self._started_writes <= 0 and (not self._runner_running()):
                            break
                        idle_loops = max_idle_loops
                    continue

                idle_loops = 0
                self._last_seq = int(seq)

                try:
                    frame_bgr = self._prepare_frame(frame)
                except Exception as exc:
                    self._error = f"frame prepare failed: {exc}"
                    break

                self._last_good_frame = np.ascontiguousarray(frame_bgr.copy())
                if not self._write_frame_to_proc(frame_bgr, source="real", mark_ready=True):
                    break
        finally:
            try:
                self.buffer.remove_client()
            except Exception:
                pass
            self._close_proc()
            self._done_evt.set()

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def wait_until_ready(self, timeout: float = 8.0) -> bool:
        deadline = time.monotonic() + float(max(0.1, float(timeout or 0.0)))
        while time.monotonic() < deadline:
            if self._ready_evt.wait(timeout=0.1):
                return True
            if self._done_evt.is_set():
                break
        return bool(self._ready_evt.is_set())

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            if self._thread is not None:
                self._thread.join(timeout=3.0)
        except Exception:
            pass
        self._close_proc()

    def is_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def error(self) -> str:
        return str(self._error or "")

    def status(self) -> Dict[str, Any]:
        return {
            "stream_name": str(self.stream_name),
            "rtsp_output": str(self.rtsp_output),
            "ready": bool(self._ready_evt.is_set()),
            "alive": bool(self.is_alive()),
            "frames_published": int(self._started_writes),
            "real_frames_published": int(self._real_frames_written),
            "held_frames_published": int(self._held_frames_written),
            "placeholder_frames_published": int(self._placeholder_frames_written),
            "holding_last_frame": bool(self._last_good_frame is not None and self._hold_last_frame),
            "error": str(self._error or ""),
        }


def _redact_rtsp_url(url: str) -> str:
    text = str(url or "")
    try:
        if "://" not in text or "@" not in text:
            return text
        scheme, rest = text.split("://", 1)
        userinfo, tail = rest.split("@", 1)
        if ":" in userinfo:
            user = userinfo.split(":", 1)[0]
            return f"{scheme}://{user}:***@{tail}"
        return f"{scheme}://***@{tail}"
    except Exception:
        return text


def _setting_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(str(name))
    if raw is not None:
        return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}
    try:
        return bool(getattr(settings, str(name), default))
    except Exception:
        return bool(default)


def _setting_str(name: str, default: str = "") -> str:
    raw = os.environ.get(str(name))
    if raw is not None:
        return str(raw)
    try:
        value = getattr(settings, str(name), default)
    except Exception:
        value = default
    return str(default if value is None else value)


def _setting_int(name: str, default: int) -> int:
    raw = os.environ.get(str(name))
    if raw is None:
        try:
            raw = getattr(settings, str(name), default)
        except Exception:
            raw = default
    try:
        return int(raw)
    except Exception:
        return int(default)


def _setting_float(name: str, default: float) -> float:
    raw = os.environ.get(str(name))
    if raw is None:
        try:
            raw = getattr(settings, str(name), default)
        except Exception:
            raw = default
    try:
        return float(raw)
    except Exception:
        return float(default)


def _split_ffmpeg_flags(value: str) -> list[str]:
    try:
        return shlex.split(str(value or ""))
    except Exception:
        # Do not let a malformed optional flags env break playback completely.
        return []


class PlaybackCleanRestream:
    """
    CP Plus playback clean-up stage.

    The NVR records H265/HEVC. VLC can display it because it has a tolerant,
    buffered decoder. OpenCV/AI reading the NVR RTSP directly can drop HEVC
    reference frames and then produces artifacts such as:

        Could not find ref with POC ...
        Error constructing the frame RPS

    This class starts FFmpeg as a dedicated buffered decoder/transcoder:

        CP Plus H265 playback RTSP
          -> FFmpeg software HEVC decode
          -> stable local H264 all-I RTSP
          -> MediaMTX playback_clean_*
          -> pipeline_tracing/OpenCV

    The clean RTSP path is local, so all-I H264 is acceptable and avoids long
    dependency chains before the AI pipeline receives frames.
    """

    def __init__(
        self,
        *,
        raw_rtsp_source: str,
        stream_name: str,
        ffmpeg_bin: str,
        mediamtx_rtsp_base: str,
    ):
        self.raw_rtsp_source = str(raw_rtsp_source or "")
        self.stream_name = str(stream_name or "").strip().strip("/")
        self.ffmpeg_bin = str(ffmpeg_bin or "ffmpeg")
        self.mediamtx_rtsp_base = str(mediamtx_rtsp_base or "").rstrip("/")
        self.rtsp_output = f"{self.mediamtx_rtsp_base}/{self.stream_name}"
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_handle = None
        self._started_at: float = 0.0
        self._error: str = ""
        log_dir = _setting_str("PLAYBACK_CLEAN_RESTREAM_LOG_DIR", "logs/playback_clean_restream") or "logs/playback_clean_restream"
        os.makedirs(log_dir, exist_ok=True)
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in self.stream_name)
        self.log_path = os.path.join(log_dir, f"{safe_name or 'playback_clean'}.log")

    @staticmethod
    def enabled() -> bool:
        return _setting_bool("PLAYBACK_CLEAN_RESTREAM_ENABLED", True)

    @staticmethod
    def make_stream_name(camera_id: int, request_mode: str, token: Optional[str] = None) -> str:
        prefix = _setting_str("PLAYBACK_CLEAN_RESTREAM_PATH_PREFIX", "playback_clean_") or "playback_clean_"
        prefix = prefix.strip().strip("/")
        if not prefix:
            prefix = "playback_clean_"
        if not prefix.endswith("_") and not prefix.endswith("/"):
            prefix += "_"
        suffix = str(token or uuid.uuid4().hex[:12])
        mode = str(request_mode or "location").strip().lower() or "location"
        return f"{prefix}{mode}_cam{int(camera_id)}_{suffix}".strip("/")

    def _build_cmd(self) -> list[str]:
        rtsp_transport = _setting_str("PLAYBACK_CLEAN_RESTREAM_RTSP_TRANSPORT", "tcp") or "tcp"
        decoder = _setting_str("PLAYBACK_CLEAN_RESTREAM_DECODER", "hevc").strip()
        encoder = _setting_str("PLAYBACK_CLEAN_RESTREAM_ENCODER", "libx264").strip() or "libx264"
        width = max(0, _setting_int("PLAYBACK_CLEAN_RESTREAM_WIDTH", 1280))
        height = max(0, _setting_int("PLAYBACK_CLEAN_RESTREAM_HEIGHT", 720))
        fps = max(1.0, _setting_float("PLAYBACK_CLEAN_RESTREAM_FPS", 8.0))
        bitrate = _setting_str("PLAYBACK_CLEAN_RESTREAM_BITRATE", "8000k") or "4000k"
        bufsize = _setting_str("PLAYBACK_CLEAN_RESTREAM_BUFSIZE", "16000k") or "8000k"
        gop = max(1, _setting_int("PLAYBACK_CLEAN_RESTREAM_GOP", 16))
        all_i = _setting_bool("PLAYBACK_CLEAN_RESTREAM_ALL_I", True)
        preset = _setting_str("PLAYBACK_CLEAN_RESTREAM_PRESET", "veryfast") or "veryfast"

        cmd: list[str] = [self.ffmpeg_bin, "-hide_banner", "-loglevel", "warning"]
        cmd += ["-rtsp_transport", rtsp_transport]
        cmd += _split_ffmpeg_flags(
            _setting_str(
                "PLAYBACK_CLEAN_RESTREAM_FFMPEG_FLAGS",
                "-fflags +genpts+discardcorrupt -err_detect ignore_err -analyzeduration 10000000 -probesize 10000000 -max_delay 5000000",
            )
        )

        # Force the software decoder by default. Do not use hevc_cuvid for CP Plus
        # playback unless explicitly configured; the GPU decoder tends to expose
        # the broken reference frames as large visual block corruption.
        if decoder and decoder.lower() not in {"auto", "none", "default"}:
            if decoder.lower() in {"hevc_cuvid", "h264_cuvid"}:
                cmd += ["-hwaccel", "cuda", "-c:v", decoder]
            else:
                cmd += ["-c:v", decoder]

        cmd += ["-i", self.raw_rtsp_source, "-map", "0:v:0", "-an"]

        vf_parts: list[str] = [f"fps={fps:g}"]
        if width > 0 and height > 0:
            # Ensure even dimensions for yuv420p/H264.
            width -= width % 2
            height -= height % 2
            vf_parts.append(f"scale={width}:{height}:flags=bicubic")
        cmd += ["-vf", ",".join(vf_parts)]

        cmd += ["-c:v", encoder]
        enc_lower = encoder.lower()
        if enc_lower in {"libx264", "h264"}:
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
        elif enc_lower in {"h264_nvenc", "hevc_nvenc"}:
            cmd += [
                "-preset", "p1" if preset == "ultrafast" else preset,
                "-tune", "ll",
                "-pix_fmt", "yuv420p",
                "-b:v", bitrate,
                "-maxrate", bitrate,
                "-bufsize", bufsize,
                "-g", "1" if all_i else str(gop),
                "-bf", "0",
            ]

        cmd += [
            "-muxdelay", "0",
            "-muxpreload", "0",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_output,
        ]
        return cmd

    def _tail_log(self, max_chars: int = 4000) -> str:
        try:
            with open(self.log_path, "r", encoding="utf-8", errors="replace") as fh:
                return fh.read()[-max_chars:]
        except Exception:
            return ""

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        if not self.raw_rtsp_source:
            raise RuntimeError("raw CP Plus playback RTSP source is empty")
        if not self.mediamtx_rtsp_base:
            raise RuntimeError("MEDIAMTX_RTSP is required for playback clean restream")

        self._stderr_handle = open(self.log_path, "a", buffering=1)
        cmd = self._build_cmd()
        # Log a redacted command for debugging without exposing the NVR password.
        try:
            redacted = [(_redact_rtsp_url(x) if str(x).startswith("rtsp://") else x) for x in cmd]
            self._stderr_handle.write("\n[playback-clean] command: " + " ".join(shlex.quote(str(x)) for x in redacted) + "\n")
        except Exception:
            pass
        try:
            self._proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_handle,
                close_fds=True,
            )
            self._started_at = time.time()
        except Exception as exc:
            self._error = f"could not start playback clean restream: {exc}"
            try:
                if self._stderr_handle is not None:
                    self._stderr_handle.close()
            except Exception:
                pass
            self._stderr_handle = None
            raise

    def wait_until_ready(self, timeout: Optional[float] = None) -> bool:
        # MediaMTX does not need an explicit API call here: once FFmpeg publishes
        # to playback_clean_*, the AI runner can open that local RTSP path. We
        # just give FFmpeg a short warmup and fail early if the process exits.
        if timeout is None:
            timeout = _setting_float("PLAYBACK_CLEAN_RESTREAM_WARMUP_SECONDS", 2.0)
        deadline = time.monotonic() + max(0.0, float(timeout or 0.0))
        while time.monotonic() < deadline:
            if self._proc is not None and self._proc.poll() is not None:
                self._error = f"playback clean restream exited early with code {self._proc.returncode}: {self._tail_log(1200)}"
                return False
            time.sleep(0.1)
        return bool(self._proc is not None and self._proc.poll() is None)

    def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass
            try:
                proc.wait(timeout=3.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        if self._stderr_handle is not None:
            try:
                self._stderr_handle.close()
            except Exception:
                pass
            self._stderr_handle = None

    def is_alive(self) -> bool:
        return bool(self._proc is not None and self._proc.poll() is None)

    def error(self) -> str:
        if self._error:
            return str(self._error)
        if self._proc is not None and self._proc.poll() is not None:
            return f"playback clean restream exited with code {self._proc.returncode}: {self._tail_log(1200)}"
        return ""

    def status(self) -> Dict[str, Any]:
        return {
            "stream_name": self.stream_name,
            "rtsp_output": self.rtsp_output,
            "source": _redact_rtsp_url(self.raw_rtsp_source),
            "pid": int(self._proc.pid) if self._proc is not None and self._proc.pid else None,
            "alive": self.is_alive(),
            "started_at": float(self._started_at or 0.0),
            "log_path": self.log_path,
            "error": self.error(),
        }


@dataclass
class PlaybackTraceSession:
    session_id: str
    stream_name: str
    camera_id: int
    request_mode: str
    member_id: Optional[int]
    member_name: str
    # raw_rtsp_source is the original CP Plus H265 playback URL from the NVR.
    # rtsp_source is the stream actually consumed by pipeline_tracing; when the
    # clean restream is enabled this becomes rtsp://127.0.0.1:8554/playback_clean_*.
    raw_rtsp_source: str
    rtsp_source: str
    start_time: str
    end_time: str
    started_at: float
    auto_stop_at: float
    runner: TrackingRunner
    buffer: Optional[RenderedFrame]
    publisher: Optional[AnnotatedRtspPublisher]
    clean_restream: Optional[PlaybackCleanRestream]


class PlaybackTracingService:
    """
    On-demand playback tracing manager.

    Each playback request gets its own one-source TrackingRunner so the frames
    coming from the NVR playback RTSP URL go through pipeline_tracing.py and
    are then republished as annotated RTSP/HLS for the existing UI video
    player.
    """

    def __init__(self, pipeline_args: Union[str, argparse.Namespace, None] = None):
        self._lock = threading.Lock()
        self._sessions: Dict[str, PlaybackTraceSession] = {}

        if isinstance(pipeline_args, argparse.Namespace):
            self._args: argparse.Namespace | None = pipeline_args
            self._pipeline_args_str: str = ""
        else:
            self._args = None
            s = pipeline_args
            if s is None:
                s = os.environ.get("PIPELINE_ARGS") or os.environ.get("pipeline_args") or ""
            self._pipeline_args_str = str(s).strip()

    def _base_args(self) -> argparse.Namespace:
        if isinstance(self._args, argparse.Namespace):
            return copy.deepcopy(self._args)

        raw = str(self._pipeline_args_str or "").strip()
        if raw:
            try:
                return parse_pipeline_args(raw)
            except BaseException:
                pass

        return parse_args(["--src", "rtsp://127.0.0.1/dummy"])

    @staticmethod
    def _clean_member_name(member_name: Optional[str]) -> str:
        return str(member_name or "").strip()

    @staticmethod
    def _normalize_mode(
        request_mode: str,
        member_id: Optional[int],
        member_name: str,
    ) -> str:
        mode = str(request_mode or "location").strip().lower()
        if member_id is not None or member_name:
            return "member"
        if mode not in {"member", "location"}:
            return "location"
        return mode

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        raw = os.environ.get(str(name))
        if raw is None:
            return bool(default)
        return str(raw).strip().lower() in {"1", "true", "yes", "on", "y"}

    @staticmethod
    def _parse_face_det_size(value: Any, default: tuple[int, int] = (640, 640)) -> list[int]:
        try:
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return [int(value[0]), int(value[1])]
            text = str(value or "").replace(",", " ").strip()
            parts = [p for p in text.split() if p]
            if len(parts) >= 2:
                return [max(64, int(parts[0])), max(64, int(parts[1]))]
        except Exception:
            pass
        return [int(default[0]), int(default[1])]

    def _apply_playback_stability_overrides(self, args: argparse.Namespace) -> None:
        """
        Playback tracing runs on demand while the continuous live camera pipeline
        may already be using CUDA/ONNX/cuDNN.  Starting another CUDA YOLO +
        InsightFace + DeepSORT stack in the same process can trigger errors like:

            CUDA error: operation not permitted when stream is capturing
            cuDNN error: CUDNN_STATUS_EXECUTION_FAILED
            Segmentation fault

        Use CPU-safe playback defaults unless explicitly overridden.  The live
        pipeline can remain GPU/batched; only finite playback tracing is moved
        to the safer profile.
        """
        device = str(getattr(settings, "PLAYBACK_DEVICE", "cpu") or os.environ.get("PLAYBACK_DEVICE", "cpu") or "cpu").strip()
        if not device:
            device = "cpu"
        args.device = device

        yolo_weights = str(getattr(settings, "PLAYBACK_YOLO_WEIGHTS", "") or os.environ.get("PLAYBACK_YOLO_WEIGHTS", "") or "").strip()
        if yolo_weights:
            args.yolo_weights = yolo_weights
        try:
            args.yolo_imgsz = int(getattr(settings, "PLAYBACK_YOLO_IMGSZ", 640) or os.environ.get("PLAYBACK_YOLO_IMGSZ", 640) or 640)
        except Exception:
            args.yolo_imgsz = 640
        try:
            args.conf = float(getattr(settings, "PLAYBACK_CONF", 0.30) or os.environ.get("PLAYBACK_CONF", 0.30) or 0.30)
        except Exception:
            args.conf = 0.30
        try:
            args.iou = float(getattr(settings, "PLAYBACK_IOU", 0.45) or os.environ.get("PLAYBACK_IOU", 0.45) or 0.45)
        except Exception:
            args.iou = 0.45

        use_cuda = str(device).lower().startswith("cuda")
        half_default = bool(getattr(settings, "PLAYBACK_HALF", False))
        args.half = bool(use_cuda and self._env_bool("PLAYBACK_HALF", half_default))
        args.cudnn_benchmark = bool(use_cuda and self._env_bool("PLAYBACK_CUDNN_BENCHMARK", bool(getattr(settings, "PLAYBACK_CUDNN_BENCHMARK", False))))

        # Avoid GPU embedding trackers by default.  For playback, IOU/ByteTrack is
        # enough to draw boxes; face recognition performs identity matching.
        tracker_backend = str(getattr(settings, "PLAYBACK_TRACKER_BACKEND", "iou") or os.environ.get("PLAYBACK_TRACKER_BACKEND", "iou") or "iou").strip().lower()
        for attr in ("no_deepsort", "use_deepsort", "use_strongsort", "use_bytetrack"):
            if hasattr(args, attr):
                setattr(args, attr, False)
        if tracker_backend == "deepsort" and hasattr(args, "use_deepsort"):
            args.use_deepsort = True
        elif tracker_backend == "strongsort" and hasattr(args, "use_strongsort"):
            args.use_strongsort = True
        elif tracker_backend == "bytetrack" and hasattr(args, "use_bytetrack"):
            args.use_bytetrack = True
        elif hasattr(args, "no_deepsort"):
            args.no_deepsort = True

        use_face_default = bool(getattr(settings, "PLAYBACK_USE_FACE", True))
        args.use_face = self._env_bool("PLAYBACK_USE_FACE", use_face_default)
        args.face_provider = str(getattr(settings, "PLAYBACK_FACE_PROVIDER", "cpu") or os.environ.get("PLAYBACK_FACE_PROVIDER", "cpu") or "cpu").strip().lower()
        if args.face_provider not in {"auto", "cuda", "cpu"}:
            args.face_provider = "cpu"
        args.face_det_size = self._parse_face_det_size(getattr(settings, "PLAYBACK_FACE_DET_SIZE", "640 640"), default=(640, 640))
        try:
            args.face_every_n = max(1, int(getattr(settings, "PLAYBACK_FACE_EVERY_N", 5) or os.environ.get("PLAYBACK_FACE_EVERY_N", 5) or 5))
        except Exception:
            args.face_every_n = 5

        try:
            # PLAYBACK_VIDEO_FPS controls how fast annotated frames are written
            # into MediaMTX. If it is not set, honor PLAYBACK_WEBRTC_FPS from
            # the env so the publisher does not run at the old 20 FPS default.
            video_fps_raw = (
                os.environ.get("PLAYBACK_VIDEO_FPS")
                or os.environ.get("PLAYBACK_WEBRTC_FPS")
                or getattr(settings, "PLAYBACK_VIDEO_FPS", 10.0)
                or 10.0
            )
            args.video_fps = float(video_fps_raw)
        except Exception:
            args.video_fps = 10.0
        try:
            q_default = 24 if self._env_bool("PLAYBACK_REALTIME_MODE", bool(getattr(settings, "PLAYBACK_REALTIME_MODE", True))) else 256
            args.queue_size = max(1, int(os.environ.get("PLAYBACK_QUEUE_SIZE") or getattr(settings, "PLAYBACK_QUEUE_SIZE", q_default) or q_default))
        except Exception:
            args.queue_size = 24
        try:
            realtime_mode = self._env_bool("PLAYBACK_REALTIME_MODE", bool(getattr(settings, "PLAYBACK_REALTIME_MODE", True)))
            keep_all = self._env_bool("PLAYBACK_KEEP_ALL_FRAMES", bool(getattr(settings, "PLAYBACK_KEEP_ALL_FRAMES", False)))
            default_age = 0 if keep_all or (not realtime_mode) else 1200
            args.max_queue_age_ms = max(0, int(os.environ.get("PLAYBACK_MAX_QUEUE_AGE_MS") or getattr(settings, "PLAYBACK_MAX_QUEUE_AGE_MS", default_age) or default_age))
        except Exception:
            args.max_queue_age_ms = 1200
        try:
            args.max_drain_per_cycle = max(1, int(os.environ.get("PLAYBACK_MAX_DRAIN_PER_CYCLE") or getattr(settings, "PLAYBACK_MAX_DRAIN_PER_CYCLE", 64) or 64))
        except Exception:
            args.max_drain_per_cycle = 64
        try:
            args.stream_freeze_seconds = max(30.0, float(getattr(settings, "PLAYBACK_STREAM_FREEZE_SECONDS", 300.0) or os.environ.get("PLAYBACK_STREAM_FREEZE_SECONDS", 300.0) or 300.0))
        except Exception:
            args.stream_freeze_seconds = 300.0
        try:
            args.stream_open_timeout_ms = int(getattr(settings, "PLAYBACK_STREAM_OPEN_TIMEOUT_MS", 8000) or os.environ.get("PLAYBACK_STREAM_OPEN_TIMEOUT_MS", 8000) or 8000)
        except Exception:
            args.stream_open_timeout_ms = 8000
        try:
            args.stream_read_timeout_ms = int(getattr(settings, "PLAYBACK_STREAM_READ_TIMEOUT_MS", 8000) or os.environ.get("PLAYBACK_STREAM_READ_TIMEOUT_MS", 8000) or 8000)
        except Exception:
            args.stream_read_timeout_ms = 8000

    def _build_session_args(
        self,
        *,
        rtsp_source: str,
        camera_id: int,
        request_mode: str,
        member_id: Optional[int],
        member_name: str,
    ) -> argparse.Namespace:
        args = self._base_args()
        self._apply_playback_stability_overrides(args)
        args.src = [str(rtsp_source)]
        args.camera_ids = [int(camera_id)]
        args.use_db = True
        if not getattr(args, "db_url", ""):
            args.db_url = os.environ.get("DATABASE_URL", "") or ""
        args.use_face = bool(getattr(args, "use_face", True))
        args.save_csv = False
        args.csv = ""
        args.no_save_video = True
        args.show = False
        args.overlay_fps = False
        args.write_normalized_data = False
        args.update_db_embeddings = False
        args.gallery_request_mode = str(request_mode or "location")
        args.gallery_member_ids = [int(member_id)] if member_id is not None else []
        args.gallery_member_names = [member_name] if member_name else []

        # Playback requests now read from a clean local H264 restream.  In realtime
        # playback mode we are allowed to skip stale *decoded* frames so the UI
        # does not run in slow motion.  Set PLAYBACK_KEEP_ALL_FRAMES=True only for
        # offline analysis where completeness is more important than realtime speed.
        try:
            realtime_mode = self._env_bool("PLAYBACK_REALTIME_MODE", bool(getattr(settings, "PLAYBACK_REALTIME_MODE", True)))
            keep_all = self._env_bool("PLAYBACK_KEEP_ALL_FRAMES", bool(getattr(settings, "PLAYBACK_KEEP_ALL_FRAMES", False)))
            if keep_all or (not realtime_mode):
                args.max_queue_age_ms = max(0, int(os.environ.get("PLAYBACK_MAX_QUEUE_AGE_MS") or getattr(settings, "PLAYBACK_MAX_QUEUE_AGE_MS", 0) or 0))
                args.queue_size = max(64, int(os.environ.get("PLAYBACK_QUEUE_SIZE") or getattr(args, "queue_size", 256) or 256))
            else:
                args.max_queue_age_ms = max(1, int(os.environ.get("PLAYBACK_MAX_QUEUE_AGE_MS") or getattr(settings, "PLAYBACK_MAX_QUEUE_AGE_MS", 1200) or 1200))
                args.queue_size = max(1, int(os.environ.get("PLAYBACK_QUEUE_SIZE") or getattr(args, "queue_size", 24) or 24))
        except Exception:
            pass
        try:
            # Playback clips are finite RTSP sessions from the NVR. Keep the
            # freeze window large enough that we do not reconnect and loop the
            # clip after it naturally reaches the end. The session still stops
            # quickly once the annotated publisher sees no new frames.
            args.stream_freeze_seconds = max(300.0, float(getattr(args, "stream_freeze_seconds", 2.0) or 0.0))
        except Exception:
            pass
        try:
            args.draw_only_matched = False
        except Exception:
            pass
        try:
            args.report_use_drawn_only = False
        except Exception:
            pass

        # Member playback should show only the target person when face matching
        # is enabled. If playback is running in CPU/no-face stabilisation mode,
        # keep Unknown boxes visible instead of returning a clean video with all
        # boxes hidden. This avoids perceived information loss while still
        # preventing the CUDA crash path.
        if str(request_mode) == "member":
            args.hide_unknown = bool(getattr(args, "use_face", False))
            args.face_confirm_hits = 1
            args.face_switch_confirm_hits = 1
            args.camera_name_switch_hits = 1
        else:
            args.hide_unknown = False

        return args

    @staticmethod
    def _make_session_id(camera_id: int, request_mode: str) -> str:
        return f"trace_{request_mode}_cam{int(camera_id)}_{uuid.uuid4().hex[:12]}"

    @staticmethod
    def _make_stream_name(camera_id: int, request_mode: str) -> str:
        return f"playback_trace_{request_mode}_cam{int(camera_id)}_{uuid.uuid4().hex[:12]}"

    def _matching_session_ids(
        self,
        *,
        camera_id: int,
        request_mode: str,
        member_id: Optional[int],
        member_name: str,
    ) -> List[str]:
        ids: List[str] = []
        with self._lock:
            for session_id, session in self._sessions.items():
                if int(session.camera_id) != int(camera_id):
                    continue
                if str(session.request_mode) != str(request_mode):
                    continue
                if int(session.member_id or -1) != int(member_id or -1):
                    continue
                if str(session.member_name or "") != str(member_name or ""):
                    continue
                ids.append(str(session_id))
        return ids

    def _watch_session(self, session_id: str) -> None:
        while self.is_session_active(session_id):
            with self._lock:
                session = self._sessions.get(str(session_id))
            if session is None:
                return

            # Keep the playback session alive for MJPEG even if the optional
            # RTSP/HLS publisher exits. The frontend consumes MJPEG directly for
            # annotated playback, so stopping the whole session when the
            # publisher dies freezes the visible clip on its last frame.
            # Cleanup is handled by the explicit DELETE route and the session
            # auto-stop timer.
            runner_running = True
            try:
                runner_running = bool(session.runner.status().get("running", False))
            except Exception:
                runner_running = True

            if not runner_running:
                self.stop_session(session_id)
                return

            time.sleep(0.5)

    def _auto_stop_after(self, session_id: str, delay_seconds: float) -> None:
        delay = float(max(0.0, delay_seconds))
        if delay <= 0.0:
            return
        end_at = time.monotonic() + delay
        while time.monotonic() < end_at:
            remaining = end_at - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(1.0, remaining))
            if not self.is_session_active(session_id):
                return
        self.stop_session(session_id)

    def start_session(
        self,
        *,
        rtsp_source: str,
        camera_id: int,
        start_time: str,
        end_time: str,
        auto_stop_seconds: float,
        member_id: Optional[int] = None,
        member_name: Optional[str] = None,
        request_mode: str = "location",
    ) -> Dict[str, Any]:
        member_name_clean = self._clean_member_name(member_name)
        mode = self._normalize_mode(request_mode, member_id, member_name_clean)

        if not str(getattr(settings, "MEDIAMTX_RTSP", "") or "").strip():
            raise RuntimeError("MEDIAMTX_RTSP is required for annotated HLS playback.")

        # Avoid duplicated tracing sessions for the same exact request.
        for old_session_id in self._matching_session_ids(
            camera_id=int(camera_id),
            request_mode=str(mode),
            member_id=member_id,
            member_name=member_name_clean,
        ):
            self.stop_session(old_session_id)

        # Keep playback tracing bounded.  A second playback request should replace
        # the existing one instead of starting another model stack and competing
        # with the live pipeline for CPU/GPU resources.
        try:
            max_sessions = int(getattr(settings, "PLAYBACK_MAX_ACTIVE_SESSIONS", 1) or os.environ.get("PLAYBACK_MAX_ACTIVE_SESSIONS", 1) or 1)
        except Exception:
            max_sessions = 1
        if max_sessions > 0:
            with self._lock:
                active_session_ids = list(self._sessions.keys())
            overflow = len(active_session_ids) - max_sessions + 1
            if overflow > 0:
                for old_session_id in active_session_ids[:overflow]:
                    self.stop_session(old_session_id)

        raw_rtsp_source = str(rtsp_source)
        pipeline_rtsp_source = raw_rtsp_source
        clean_restream: Optional[PlaybackCleanRestream] = None

        # Generate the identifiers before starting FFmpeg so the clean path and
        # processed trace path can be correlated in logs/status.
        session_id = self._make_session_id(int(camera_id), mode)
        stream_name = self._make_stream_name(int(camera_id), mode)

        # IMPORTANT CP Plus playback flow:
        # raw H265 NVR playback -> FFmpeg buffered decode -> H264 all-I local RTSP
        # playback_clean_* -> OpenCV/AI -> playback_trace_* -> WebRTC.
        if PlaybackCleanRestream.enabled():
            clean_stream_name = PlaybackCleanRestream.make_stream_name(int(camera_id), mode)
            clean_restream = PlaybackCleanRestream(
                raw_rtsp_source=raw_rtsp_source,
                stream_name=clean_stream_name,
                ffmpeg_bin=str(getattr(settings, "FFMPEG_BIN", "ffmpeg") or "ffmpeg"),
                mediamtx_rtsp_base=str(getattr(settings, "MEDIAMTX_RTSP", "") or ""),
            )
            try:
                clean_restream.start()
                if not clean_restream.wait_until_ready():
                    err = clean_restream.error() or "clean restream did not stay alive"
                    clean_restream.stop()
                    raise RuntimeError(err)
                pipeline_rtsp_source = clean_restream.rtsp_output
                print(
                    f"[PLAYBACK-CLEAN] camera_id={int(camera_id)} "
                    f"raw={_redact_rtsp_url(raw_rtsp_source)} -> clean={pipeline_rtsp_source}"
                )
            except Exception:
                try:
                    clean_restream.stop()
                except Exception:
                    pass
                raise

        args = self._build_session_args(
            rtsp_source=str(pipeline_rtsp_source),
            camera_id=int(camera_id),
            request_mode=mode,
            member_id=member_id,
            member_name=member_name_clean,
        )
        if not str(getattr(args, "db_url", "") or "").strip():
            if clean_restream is not None:
                clean_restream.stop()
            raise RuntimeError("DATABASE_URL / --db-url is required for playback tracing sessions.")

        runner = TrackingRunner(args)
        try:
            runner.start()
        except Exception:
            try:
                runner.stop()
            except Exception:
                pass
            if clean_restream is not None:
                try:
                    clean_restream.stop()
                except Exception:
                    pass
            raise

        buf = runner.get_camera_buffer(int(camera_id))
        if buf is None:
            try:
                runner.stop()
            except Exception:
                pass
            if clean_restream is not None:
                try:
                    clean_restream.stop()
                except Exception:
                    pass
            raise RuntimeError(f"Camera buffer not available for camera_id={int(camera_id)}")

        publisher = AnnotatedRtspPublisher(
            buffer=buf,
            runner=runner,
            stream_name=stream_name,
            ffmpeg_bin=str(getattr(settings, "FFMPEG_BIN", "ffmpeg") or "ffmpeg"),
            mediamtx_rtsp_base=str(getattr(settings, "MEDIAMTX_RTSP", "") or ""),
            fps=float(getattr(args, "video_fps", 20.0) or 20.0),
            codec=_setting_str("PLAYBACK_WEBRTC_CODEC", "libx264") or "libx264",
        )
        try:
            publisher.start()
        except Exception:
            try:
                runner.stop()
            except Exception:
                pass
            if clean_restream is not None:
                try:
                    clean_restream.stop()
                except Exception:
                    pass
            raise

        # Do not block waiting for the first frame. NVR playback can take 30-50s.
        now = time.time()
        auto_stop_at = now + float(max(0.0, auto_stop_seconds)) if auto_stop_seconds > 0 else 0.0
        session = PlaybackTraceSession(
            session_id=session_id,
            stream_name=stream_name,
            camera_id=int(camera_id),
            request_mode=str(mode),
            member_id=int(member_id) if member_id is not None else None,
            member_name=str(member_name_clean),
            raw_rtsp_source=raw_rtsp_source,
            rtsp_source=str(pipeline_rtsp_source),
            start_time=str(start_time),
            end_time=str(end_time),
            started_at=float(now),
            auto_stop_at=float(auto_stop_at),
            runner=runner,
            buffer=buf,
            publisher=publisher,
            clean_restream=clean_restream,
        )
        with self._lock:
            self._sessions[session_id] = session

        watch_thr = threading.Thread(target=self._watch_session, args=(session_id,), daemon=True)
        watch_thr.start()

        if auto_stop_seconds > 0:
            thr = threading.Thread(
                target=self._auto_stop_after,
                args=(session_id, float(auto_stop_seconds)),
                daemon=True,
            )
            thr.start()

        return self.get_session_info(session_id) or {"session_id": session_id}

    def stop_session(self, session_id: str) -> bool:
        session: Optional[PlaybackTraceSession] = None
        with self._lock:
            session = self._sessions.pop(str(session_id), None)
        if session is None:
            return False
        try:
            if session.publisher is not None:
                session.publisher.stop()
        except Exception:
            pass
        try:
            session.runner.stop()
        except Exception:
            pass
        try:
            if session.clean_restream is not None:
                session.clean_restream.stop()
        except Exception:
            pass
        return True

    def stop_all(self) -> None:
        with self._lock:
            session_ids = list(self._sessions.keys())
        for session_id in session_ids:
            self.stop_session(session_id)

    def is_session_active(self, session_id: str) -> bool:
        with self._lock:
            return str(session_id) in self._sessions

    def get_session_id_by_stream_name(self, stream_name: str) -> Optional[str]:
        want = str(stream_name or "").strip()
        if not want:
            return None
        with self._lock:
            for session_id, session in self._sessions.items():
                if str(session.stream_name) == want:
                    return str(session_id)
        return None

    def get_session_buffer(self, session_id: str) -> Optional[RenderedFrame]:
        with self._lock:
            session = self._sessions.get(str(session_id))
            return session.buffer if session is not None else None

    def get_session_info(self, session_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            session = self._sessions.get(str(session_id))
            if session is None:
                return None
            return {
                "session_id": str(session.session_id),
                "stream_name": str(session.stream_name),
                "camera_id": int(session.camera_id),
                "request_mode": str(session.request_mode),
                "member_id": int(session.member_id) if session.member_id is not None else None,
                "member_name": str(session.member_name or ""),
                "start_time": str(session.start_time),
                "end_time": str(session.end_time),
                "started_at": float(session.started_at),
                "auto_stop_at": float(session.auto_stop_at),
                "raw_source": _redact_rtsp_url(str(session.raw_rtsp_source)),
                "pipeline_source": str(session.rtsp_source),
                "clean_restream_enabled": bool(session.clean_restream is not None),
                "clean_restream": session.clean_restream.status() if session.clean_restream is not None else None,
                "hls_url": (
                    f"{str(getattr(settings, 'HLS_BASE_URL', '')).rstrip('/')}/{str(session.stream_name)}/index.m3u8"
                    if session.publisher is not None else ""
                ),
                "webrtc_url": (
                    f"{str(getattr(settings, 'MEDIAMTX_WEBRTC_PUBLIC_BASE', '') or getattr(settings, 'MEDIAMTX_WEBRTC_INTERNAL', '') or 'http://localhost:8889').rstrip('/')}/{str(session.stream_name).strip('/')}"
                    if session.publisher is not None else ""
                ),
                "whep_url": (
                    f"{str(getattr(settings, 'MEDIAMTX_WEBRTC_PUBLIC_BASE', '') or getattr(settings, 'MEDIAMTX_WEBRTC_INTERNAL', '') or 'http://localhost:8889').rstrip('/')}/{str(session.stream_name).strip('/')}/whep"
                    if session.publisher is not None else ""
                ),
                "status": session.runner.status(),
                "publisher": session.publisher.status() if session.publisher is not None else None,
            }

    def list_sessions(self) -> List[Dict[str, Any]]:
        with self._lock:
            session_ids = list(self._sessions.keys())
        out: List[Dict[str, Any]] = []
        for session_id in session_ids:
            info = self.get_session_info(session_id)
            if info is not None:
                out.append(info)
        out.sort(key=lambda x: (float(x.get("started_at", 0.0)), str(x.get("session_id", ""))))
        return out
