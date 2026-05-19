from __future__ import annotations

import argparse
import copy
import os
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
            self._start_processed_publishers(args)
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
        self._last_seq: int = -1

    @staticmethod
    def _even_size(w: int, h: int) -> tuple[int, int]:
        ww = max(2, int(w) - (int(w) % 2))
        hh = max(2, int(h) - (int(h) % 2))
        return ww, hh

    def _build_cmd(self, w: int, h: int) -> list[str]:
        fps_txt = f"{self.fps:.3f}"
        keyint = max(1, int(round(self.fps)))
        return [
            self.ffmpeg_bin,
            "-loglevel", "error",
            "-fflags", "+genpts",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{int(w)}x{int(h)}",
            "-r", fps_txt,
            "-i", "-",
            "-an",
            "-c:v", self.codec,
            "-preset", "ultrafast",
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-level:v", "3.1",
            "-pix_fmt", "yuv420p",
            "-g", str(keyint),
            "-keyint_min", str(keyint),
            "-sc_threshold", "0",
            "-bf", "0",
            "-muxdelay", "0",
            "-muxpreload", "0",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_output,
        ]

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

    def _loop(self) -> None:
        idle_loops = 0
        self.buffer.add_client()
        try:
            while not self._stop_evt.is_set():
                try:
                    frame, _ts, _meta, seq = self.buffer.wait_for_seq(self._last_seq, timeout=0.5)
                except Exception as exc:
                    self._error = f"buffer wait failed: {exc}"
                    break

                if self._stop_evt.is_set():
                    break

                if frame is None or int(seq) <= int(self._last_seq):
                    idle_loops += 1
                    if idle_loops >= 20:
                        # Do not tear the session down just because annotated
                        # playback has a long decode gap. NVR playback often
                        # starts mid-GOP, so OpenCV can sit on decoder errors
                        # until the next clean keyframe arrives. Breaking here
                        # freezes MJPEG playback on the first annotated frame.
                        #
                        # The session is cleaned up by the explicit DELETE route
                        # or the session auto-stop timer. If the underlying
                        # runner has already stopped before we ever published a
                        # frame, we can still exit early.
                        if self._started_writes <= 0 and (not self._runner_running()):
                            break
                        idle_loops = 20
                    continue

                idle_loops = 0
                self._last_seq = int(seq)

                try:
                    frame_bgr = self._prepare_frame(frame)
                except Exception as exc:
                    self._error = f"frame prepare failed: {exc}"
                    break

                if self._proc is None:
                    try:
                        self._start_proc(frame_bgr.shape[1], frame_bgr.shape[0])
                    except Exception as exc:
                        self._error = f"ffmpeg start failed: {exc}"
                        break

                if self._proc is None or self._proc.stdin is None:
                    self._error = "ffmpeg stdin not available"
                    break

                if self._proc.poll() is not None:
                    self._error = f"ffmpeg exited early with code {self._proc.returncode}"
                    break

                try:
                    self._proc.stdin.write(frame_bgr.tobytes())
                except (BrokenPipeError, OSError) as exc:
                    self._error = f"ffmpeg pipe failed: {exc}"
                    break

                self._started_writes += 1
                if self._started_writes >= 1:
                    self._ready_evt.set()
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
            "error": str(self._error or ""),
        }


@dataclass
class PlaybackTraceSession:
    session_id: str
    stream_name: str
    camera_id: int
    request_mode: str
    member_id: Optional[int]
    member_name: str
    rtsp_source: str
    start_time: str
    end_time: str
    started_at: float
    auto_stop_at: float
    runner: TrackingRunner
    buffer: Optional[RenderedFrame]
    publisher: Optional[AnnotatedRtspPublisher]


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
        args.src = [str(rtsp_source)]
        args.camera_ids = [int(camera_id)]
        args.use_db = True
        if not getattr(args, "db_url", ""):
            args.db_url = os.environ.get("DATABASE_URL", "") or ""
        args.use_face = True
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

        # Playback requests should prioritize correctness over low-latency frame dropping.
        try:
            args.max_queue_age_ms = 0
        except Exception:
            pass
        try:
            args.queue_size = max(256, int(getattr(args, "queue_size", 128) or 128))
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

        # Member playback should show only the target person. Location playback
        # should still show all detected people, even when they are Unknown.
        if str(request_mode) == "member":
            args.hide_unknown = True
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

        args = self._build_session_args(
            rtsp_source=str(rtsp_source),
            camera_id=int(camera_id),
            request_mode=mode,
            member_id=member_id,
            member_name=member_name_clean,
        )
        if not str(getattr(args, "db_url", "") or "").strip():
            raise RuntimeError("DATABASE_URL / --db-url is required for playback tracing sessions.")

        runner = TrackingRunner(args)
        try:
            runner.start()
        except Exception:
            try:
                runner.stop()
            except Exception:
                pass
            raise

        buf = runner.get_camera_buffer(int(camera_id))
        if buf is None:
            try:
                runner.stop()
            except Exception:
                pass
            raise RuntimeError(f"Camera buffer not available for camera_id={int(camera_id)}")

        session_id = self._make_session_id(int(camera_id), mode)
        stream_name = self._make_stream_name(int(camera_id), mode)
        publisher = AnnotatedRtspPublisher(
            buffer=buf,
            runner=runner,
            stream_name=stream_name,
            ffmpeg_bin=str(getattr(settings, "FFMPEG_BIN", "ffmpeg") or "ffmpeg"),
            mediamtx_rtsp_base=str(getattr(settings, "MEDIAMTX_RTSP", "") or ""),
            fps=float(getattr(args, "video_fps", 20.0) or 20.0),
        )
        try:
            publisher.start()
        except Exception:
            try:
                runner.stop()
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
            rtsp_source=str(rtsp_source),
            start_time=str(start_time),
            end_time=str(end_time),
            started_at=float(now),
            auto_stop_at=float(auto_stop_at),
            runner=runner,
            buffer=buf,
            publisher=publisher,
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
