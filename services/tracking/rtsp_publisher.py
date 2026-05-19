from __future__ import annotations

import os
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np


class ProcessedFrameRtspPublisher:
    """
    Publishes a browser-compatible H264 RTSP stream into MediaMTX.

    Two modes are supported:

      processed
        Encode the latest processed frame from the AI pipeline. This is the old
        behaviour and is useful when the UI strictly needs frames exactly as the
        detector produced them. Motion will only update at the detector FPS.

      hybrid  (recommended for live UI)
        Encode the newest raw camera frame at the WebRTC FPS and draw the latest
        AI boxes/labels over it. This keeps the live view smooth and low-latency
        while detection can still run at 5-8 FPS. Old frames are never queued.
    """

    _encoder_cache: Dict[Tuple[str, str], bool] = {}
    _encoder_cache_lock = threading.Lock()

    def __init__(
        self,
        *,
        buffer: Any,
        stream_name: str,
        ffmpeg_bin: str,
        mediamtx_rtsp_base: str,
        raw_buffer: Any = None,
        mode: str = "hybrid",
        fps: float = 15.0,
        width: int = 1280,
        height: int = 720,
        codec: str = "auto",
        bitrate: str = "3500k",
        bufsize: str = "700k",
        x264_preset: str = "superfast",
        overlay_max_age_ms: int = 1500,
        gop: int = 15,
        draw_stats: bool = True,
        log_dir: str = "logs/ffmpeg_webrtc",
        restart_delay_seconds: float = 0.5,
    ) -> None:
        self.buffer = buffer
        self.raw_buffer = raw_buffer
        self.stream_name = str(stream_name).strip().strip("/")
        self.ffmpeg_bin = str(ffmpeg_bin or "ffmpeg")
        self.mediamtx_rtsp_base = str(mediamtx_rtsp_base or "").rstrip("/")
        self.rtsp_output = f"{self.mediamtx_rtsp_base}/{self.stream_name}"
        self.mode = str(mode or "hybrid").strip().lower()
        if self.mode not in {"hybrid", "processed"}:
            self.mode = "hybrid"
        self.fps = float(max(1.0, min(30.0, float(fps or 15.0))))
        self.width = int(width or 0)
        self.height = int(height or 0)
        self.codec = str(codec or "auto").strip().lower()
        self.bitrate = str(bitrate or "3500k").strip()
        self.bufsize = str(bufsize or "700k").strip()
        self.x264_preset = str(x264_preset or "superfast").strip()
        self.overlay_max_age_ms = int(max(0, int(overlay_max_age_ms or 0)))
        self.gop = int(max(1, int(gop or round(self.fps))))
        self.draw_stats = bool(draw_stats)
        self.log_dir = str(log_dir or "logs/ffmpeg_webrtc")
        self.restart_delay_seconds = float(max(0.05, float(restart_delay_seconds or 0.5)))

        self._stop_evt = threading.Event()
        self._ready_evt = threading.Event()
        self._done_evt = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._proc: Optional[subprocess.Popen] = None
        self._stderr_handle = None
        self._size: Optional[Tuple[int, int]] = None

        self._last_processed_seq: int = -1
        self._last_raw_seq: int = -1
        self._latest_processed_frame: Optional[np.ndarray] = None
        self._latest_raw_frame: Optional[np.ndarray] = None
        self._latest_meta: Dict[str, Any] = {}
        self._latest_processed_arrival_ts: float = 0.0
        self._latest_raw_arrival_ts: float = 0.0

        self._frames_published: int = 0
        self._real_frames_published: int = 0
        self._raw_frames_published: int = 0
        self._processed_frames_published: int = 0
        self._overlay_frames_published: int = 0
        self._placeholder_frames_published: int = 0
        self._frames_dropped: int = 0
        self._restarts: int = 0
        self._last_error: str = ""
        self._last_frame_ts: float = 0.0
        self._last_publish_ts: float = 0.0
        self._publish_fps_ema: float = 0.0
        self._last_cmd: list[str] = []
        self._active_codec: str = ""
        self._forced_codec: str = ""
        self._nvenc_failed: bool = False

    def _resolve_ffmpeg_bin(self) -> str:
        candidate = str(self.ffmpeg_bin or "ffmpeg").strip()
        if not candidate:
            candidate = "ffmpeg"
        if os.path.isabs(candidate) or os.sep in candidate or (os.altsep and os.altsep in candidate):
            if os.path.isfile(candidate):
                return candidate
            raise RuntimeError(f"FFMPEG_BIN does not exist: {candidate}")
        found = shutil.which(candidate)
        if found:
            return found
        raise RuntimeError(
            "FFmpeg executable not found. Set FFMPEG_BIN in .env to the full ffmpeg.exe path "
            "or add ffmpeg to PATH."
        )

    @staticmethod
    def _even_size(w: int, h: int) -> Tuple[int, int]:
        ww = max(2, int(w) - (int(w) % 2))
        hh = max(2, int(h) - (int(h) % 2))
        return ww, hh

    @staticmethod
    def _safe_log_name(stream_name: str) -> str:
        safe = str(stream_name or "stream").strip().replace("/", "_").replace("\\", "_")
        return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe) or "stream"

    @classmethod
    def _encoder_available(cls, ffmpeg_bin: str, encoder: str) -> bool:
        key = (str(ffmpeg_bin), str(encoder))
        with cls._encoder_cache_lock:
            if key in cls._encoder_cache:
                return bool(cls._encoder_cache[key])
        ok = False
        try:
            proc = subprocess.run(
                [ffmpeg_bin, "-hide_banner", "-encoders"],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=5,
                check=False,
            )
            ok = str(encoder) in str(proc.stdout or "")
        except Exception:
            ok = False
        with cls._encoder_cache_lock:
            cls._encoder_cache[key] = bool(ok)
        return bool(ok)

    def _select_codec(self) -> str:
        if self._forced_codec:
            return self._forced_codec
        codec = (self.codec or "auto").lower().strip()
        if codec in {"auto", "nvenc", "h264_nvenc"}:
            ffmpeg_bin = self._resolve_ffmpeg_bin()
            if self._encoder_available(ffmpeg_bin, "h264_nvenc") and not self._nvenc_failed:
                return "h264_nvenc"
            if codec in {"nvenc", "h264_nvenc"}:
                self._last_error = "h264_nvenc not available or failed; falling back to libx264"
            return "libx264"
        return codec

    def _codec_args(self) -> list[str]:
        codec = self._select_codec()
        self._active_codec = codec
        gop = str(self.gop)
        br = str(self.bitrate or "3500k")
        bs = str(self.bufsize or "700k")

        if codec == "h264_nvenc":
            # NVENC keeps CPU low and removes the encoder bottleneck for 4x 720p streams.
            # Options are intentionally conservative for broad FFmpeg compatibility.
            return [
                "-c:v", "h264_nvenc",
                "-preset", "p1",
                "-tune", "ull",
                "-rc", "cbr",
                "-b:v", br,
                "-maxrate", br,
                "-bufsize", bs,
                "-pix_fmt", "yuv420p",
                "-g", gop,
                "-bf", "0",
            ]

        if codec == "libx264":
            preset = self.x264_preset or "superfast"
            return [
                "-c:v", "libx264",
                "-preset", preset,
                "-tune", "zerolatency",
                "-threads", "2",
                "-profile:v", "baseline",
                "-level:v", "3.1",
                "-pix_fmt", "yuv420p",
                "-b:v", br,
                "-maxrate", br,
                "-bufsize", bs,
                "-g", gop,
                "-keyint_min", gop,
                "-sc_threshold", "0",
                "-bf", "0",
                "-x264-params", f"bframes=0:keyint={gop}:min-keyint={gop}:scenecut=0",
            ]

        # Explicit custom encoder. Keep this minimal so user-provided encoders still work.
        return [
            "-c:v", codec,
            "-pix_fmt", "yuv420p",
            "-b:v", br,
            "-maxrate", br,
            "-bufsize", bs,
            "-g", gop,
            "-bf", "0",
        ]

    def _build_cmd(self, w: int, h: int) -> list[str]:
        fps_txt = f"{self.fps:.3f}"
        return [
            self._resolve_ffmpeg_bin(),
            "-hide_banner",
            "-loglevel", "warning",
            "-fflags", "+genpts+nobuffer",
            "-flags", "low_delay",
            "-thread_queue_size", "1",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-s", f"{int(w)}x{int(h)}",
            "-framerate", fps_txt,
            "-i", "-",
            "-an",
            *self._codec_args(),
            "-muxdelay", "0",
            "-muxpreload", "0",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            self.rtsp_output,
        ]

    @staticmethod
    def _draw_centered_text(img: np.ndarray, lines: list[str]) -> None:
        h, w = img.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.8, w / 900.0))
        thickness = 1 if w < 900 else 2
        line_gap = int(26 * scale) + 8
        total_h = max(1, len(lines)) * line_gap
        y = max(30, (h - total_h) // 2)
        for line in lines:
            text = str(line or "")
            (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
            x = max(8, (w - tw) // 2)
            cv2.putText(img, text, (x + 1, y + th + 1), font, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
            cv2.putText(img, text, (x, y + th), font, scale, (230, 230, 230), thickness, cv2.LINE_AA)
            y += line_gap

    def _placeholder_frame(self, w: int, h: int) -> np.ndarray:
        w, h = self._even_size(int(w), int(h))
        img = np.zeros((int(h), int(w), 3), dtype=np.uint8)
        lines = ["Waiting for camera frames", str(self.stream_name), time.strftime("%Y-%m-%d %H:%M:%S")]
        if self._last_error:
            lines.append(str(self._last_error)[:90])
        self._draw_centered_text(img, lines)
        return img

    def _prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        if frame is None or getattr(frame, "size", 0) == 0:
            raise ValueError("empty frame")
        if len(frame.shape) != 3 or int(frame.shape[2]) != 3:
            raise ValueError(f"expected BGR frame with 3 channels, got shape={getattr(frame, 'shape', None)}")
        src_h, src_w = frame.shape[:2]
        if self.width > 0 and self.height > 0:
            out_w, out_h = self._even_size(self.width, self.height)
        else:
            out_w, out_h = self._even_size(src_w, src_h)
        if self._size is None:
            self._size = (out_w, out_h)
        else:
            out_w, out_h = self._size
        if int(src_w) != int(out_w) or int(src_h) != int(out_h):
            interp = cv2.INTER_AREA if (src_w > out_w or src_h > out_h) else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (int(out_w), int(out_h)), interpolation=interp)
        return np.ascontiguousarray(frame)

    def _start_proc(self, w: int, h: int) -> None:
        w, h = self._even_size(int(w), int(h))
        if self._size is None:
            self._size = (int(w), int(h))
        Path(self.log_dir).mkdir(parents=True, exist_ok=True)
        log_path = Path(self.log_dir) / f"{self._safe_log_name(self.stream_name)}.log"
        self._stderr_handle = open(log_path, "a", buffering=1)
        cmd = self._build_cmd(w, h)
        self._last_cmd = list(cmd)
        self._stderr_handle.write("\n[cmd] " + shlex.join(cmd) + "\n")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._stderr_handle,
            bufsize=0,
        )
        self._restarts += 1
        print(
            f"[WEBRTC] FFmpeg publisher started stream={self.stream_name} "
            f"mode={self.mode} codec={self._active_codec or self.codec} {w}x{h}@{self.fps:.1f} -> {self.rtsp_output}"
        )

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
                proc.wait(timeout=2.0)
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

    def _handle_ffmpeg_failure(self, reason: str) -> None:
        self._last_error = str(reason or "ffmpeg failed")
        if self._active_codec == "h264_nvenc":
            # Do not keep restarting forever if this FFmpeg build exposes NVENC but
            # cannot actually open it on this machine/driver.
            self._nvenc_failed = True
            self._forced_codec = "libx264"
            self._last_error += "; falling back to libx264"
        self._close_proc()
        time.sleep(self.restart_delay_seconds)

    def _write_frame_to_ffmpeg(self, frame_bgr: np.ndarray, *, source: str) -> bool:
        if frame_bgr is None or getattr(frame_bgr, "size", 0) == 0:
            return False
        h, w = frame_bgr.shape[:2]
        proc = self._proc
        if proc is not None and proc.poll() is not None:
            self._handle_ffmpeg_failure(f"ffmpeg exited with code {proc.returncode}")

        if self._proc is None:
            try:
                self._start_proc(int(w), int(h))
            except Exception as exc:
                self._last_error = f"ffmpeg start failed: {exc}"
                self._close_proc()
                time.sleep(self.restart_delay_seconds)
                return False

        if self._proc is None or self._proc.stdin is None:
            self._handle_ffmpeg_failure("ffmpeg stdin unavailable")
            return False

        try:
            self._proc.stdin.write(frame_bgr.tobytes())
            self._frames_published += 1
            if source == "placeholder":
                self._placeholder_frames_published += 1
            else:
                self._real_frames_published += 1
                if source == "raw":
                    self._raw_frames_published += 1
                elif source == "processed":
                    self._processed_frames_published += 1
                elif source == "overlay":
                    self._raw_frames_published += 1
                    self._overlay_frames_published += 1
            now_pub = time.time()
            if self._last_publish_ts > 0.0:
                dt_pub = max(1e-6, now_pub - self._last_publish_ts)
                inst_pub_fps = 1.0 / dt_pub
                alpha = 0.15
                self._publish_fps_ema = inst_pub_fps if self._publish_fps_ema <= 0.0 else ((1.0 - alpha) * self._publish_fps_ema + alpha * inst_pub_fps)
            self._last_publish_ts = now_pub
            self._ready_evt.set()
            return True
        except (BrokenPipeError, OSError) as exc:
            self._handle_ffmpeg_failure(f"ffmpeg pipe failed: {exc}")
            return False

    def _normalize_wait_result(self, ret: Any, previous_seq: int) -> Tuple[Optional[np.ndarray], float, Dict[str, Any], int]:
        frame: Optional[np.ndarray] = None
        ts = 0.0
        meta: Dict[str, Any] = {}
        seq = int(previous_seq)
        if isinstance(ret, (list, tuple)):
            if len(ret) >= 1 and isinstance(ret[0], np.ndarray):
                frame = ret[0]
            if len(ret) >= 2:
                try:
                    ts = float(ret[1] or 0.0)
                except Exception:
                    ts = 0.0
            if len(ret) >= 3 and isinstance(ret[2], dict):
                meta = dict(ret[2])
            if len(ret) >= 4:
                try:
                    seq = int(ret[3])
                except Exception:
                    seq = int(previous_seq)
        elif isinstance(ret, np.ndarray):
            frame = ret
            seq = int(previous_seq) + 1
            ts = time.time()
        return frame, ts, meta, int(seq)

    def _poll_buffer(self, buf: Any, last_seq: int, timeout: float) -> Tuple[Optional[np.ndarray], float, Dict[str, Any], int]:
        if buf is None:
            return None, 0.0, {}, int(last_seq)
        if hasattr(buf, "wait_for_seq"):
            ret = buf.wait_for_seq(int(last_seq), timeout=float(max(0.0, timeout)))
        elif hasattr(buf, "get"):
            ret = buf.get()
        else:
            raise RuntimeError("buffer does not expose wait_for_seq() or get()")
        return self._normalize_wait_result(ret, int(last_seq))

    @staticmethod
    def _parse_size(value: Any) -> Tuple[int, int]:
        try:
            if isinstance(value, dict):
                return int(value.get("width") or value.get("w") or 0), int(value.get("height") or value.get("h") or 0)
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                return int(value[0] or 0), int(value[1] or 0)
        except Exception:
            pass
        return 0, 0

    def _overlay_is_fresh(self) -> bool:
        if not self._latest_meta or self._latest_processed_arrival_ts <= 0:
            return False
        if self.overlay_max_age_ms <= 0:
            return True
        age_ms = (time.time() - float(self._latest_processed_arrival_ts)) * 1000.0
        return age_ms <= float(self.overlay_max_age_ms)

    @staticmethod
    def _first_present(mapping: Dict[str, Any], *keys: str) -> Any:
        for key in keys:
            try:
                value = mapping.get(key)
            except Exception:
                continue
            if value is not None:
                return value
        return None

    @staticmethod
    def _format_track_label(name: str, is_known: bool, tracker_id: Any = None, logical_id: Any = None) -> str:
        """Build the visible box label.

        `tracker_id` is the raw StrongSORT/ByteTrack track id.  `logical_id` is
        the app-side identity/reattach id.  The UI overlay should show the raw
        tracker id so it matches what StrongSORT/ByteTrack is currently using.
        """
        base = str(name or "").strip()
        if not base or (not bool(is_known) and base.lower() == "unknown"):
            base = "Unknown"

        tid = tracker_id
        try:
            if tid is None or int(tid) <= 0:
                tid = logical_id
        except Exception:
            tid = logical_id

        try:
            if tid is not None and int(tid) > 0:
                return f"{base} | ID {int(tid)}"
        except Exception:
            pass
        return base

    @classmethod
    def _overlay_item_to_box_label(cls, item: Any) -> Optional[Tuple[float, float, float, float, str, bool]]:
        """Normalize pipeline metadata into (x1, y1, x2, y2, label, is_known).

        Supported metadata shapes:
          - events tuple: (logical_tid, x1, y1, x2, y2, name, score, member_id, is_known, raw_tid)
          - visible_tracks dict: {bbox: [x1,y1,x2,y2], raw_track_id, track_id, label/name, is_known}
          - security_events dict: {bbox: [x1,y1,x2,y2], raw_tid, logical_tid, name, is_known}
          - generic dict: {x1/y1/x2/y2} or {left/top/right/bottom}
        """
        try:
            if isinstance(item, dict):
                bbox = item.get("bbox") or item.get("xyxy") or item.get("box")
                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                    x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
                else:
                    x1 = float(cls._first_present(item, "x1", "left", "xmin"))
                    y1 = float(cls._first_present(item, "y1", "top", "ymin"))
                    x2 = float(cls._first_present(item, "x2", "right", "xmax"))
                    y2 = float(cls._first_present(item, "y2", "bottom", "ymax"))

                raw_label = cls._first_present(item, "label", "name", "person_name")
                label = str(raw_label or "").strip()
                member_id = cls._first_present(item, "member_id", "person_id")
                explicit_known = cls._first_present(item, "is_known", "known")
                if explicit_known is not None:
                    is_known = bool(explicit_known)
                else:
                    try:
                        is_known = bool(label and label.lower() != "unknown") or int(member_id or -1) > 0
                    except Exception:
                        is_known = bool(label and label.lower() != "unknown")

                # Prefer the raw tracker id.  This is the id emitted by
                # StrongSORT/ByteTrack.  Fall back to the app logical id only if
                # raw id is missing.
                raw_tid = cls._first_present(item, "raw_track_id", "raw_tid", "tracker_id", "track_id_raw")
                logical_tid = cls._first_present(item, "track_id", "logical_tid", "tid")
                label = cls._format_track_label(label, bool(is_known), raw_tid, logical_tid)
                return x1, y1, x2, y2, label, bool(is_known)

            if isinstance(item, (list, tuple)):
                vals = list(item)
                if len(vals) >= 9:
                    # process_one_frame event tuple.  Newer metadata appends
                    # raw_tid at index 9; older metadata only has logical tid at
                    # index 0, so fall back safely.
                    logical_tid = vals[0]
                    raw_tid = vals[9] if len(vals) >= 10 else logical_tid
                    x1, y1, x2, y2 = float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])
                    name = str(vals[5] or "").strip() if len(vals) > 5 else ""
                    try:
                        is_known = bool(int(vals[8]))
                    except Exception:
                        is_known = bool(name and name.lower() != "unknown")
                    label = cls._format_track_label(name, bool(is_known), raw_tid, logical_tid)
                    return x1, y1, x2, y2, label, bool(is_known)
                if len(vals) >= 6:
                    # Alternate tuple: id + xyxy + label.
                    logical_tid = vals[0]
                    x1, y1, x2, y2 = float(vals[1]), float(vals[2]), float(vals[3]), float(vals[4])
                    name = str(vals[5] or "").strip()
                    is_known = bool(name and name.lower() != "unknown")
                    label = cls._format_track_label(name, is_known, logical_tid, logical_tid)
                    return x1, y1, x2, y2, label, is_known
                if len(vals) >= 4:
                    x1, y1, x2, y2 = float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])
                    return x1, y1, x2, y2, "Unknown", False
        except Exception:
            return None
        return None


    def _draw_stats_overlay(self, frame: np.ndarray, meta: Dict[str, Any], *, source: str) -> None:
        """Draw WebRTC/pipeline FPS on the encoded frame.

        The AI pipeline FPS overlay is only drawn on the already-processed frame
        when --overlay-fps is enabled.  In hybrid WebRTC mode we usually publish
        a fresh raw frame plus redrawn boxes, so that text is not present unless
        the publisher draws it again here.
        """
        if not self.draw_stats or frame is None or getattr(frame, "size", 0) == 0:
            return
        try:
            h, w = frame.shape[:2]
            pipe_fps = float((meta or {}).get("fps") or 0.0)
            shown = int((meta or {}).get("shown") or 0)
            tracks = int((meta or {}).get("tracks") or 0)
            lag_ms = 0.0
            cap_ts = float((meta or {}).get("capture_ts") or 0.0)
            if cap_ts > 0:
                lag_ms = max(0.0, (time.time() - cap_ts) * 1000.0)

            webrtc_fps = float(self._publish_fps_ema or self.fps)
            timings = (meta or {}).get("timings_ms") or {}
            try:
                yolo_ms = float(timings.get("yolo", 0.0) or 0.0)
                trk_ms = float(timings.get("tracker", 0.0) or 0.0)
                total_ms = float((meta or {}).get("process_total_ms", 0.0) or 0.0)
            except Exception:
                yolo_ms = trk_ms = total_ms = 0.0
            ss_every = int((meta or {}).get("strongsort_every_n", 1) or 1)
            ss_used = bool((meta or {}).get("strongsort_used", False))
            ss_txt = f"SS {('hit' if ss_used else 'skip')}/{ss_every}" if ss_every > 1 else ("SS" if ss_used else "trk")
            lines = [
                f"AI FPS {pipe_fps:.1f} | WebRTC FPS {webrtc_fps:.1f} | {source}",
                f"tracks {tracks} | shown {shown} | lag {lag_ms:.0f}ms",
                f"yolo {yolo_ms:.0f}ms | trk {trk_ms:.0f}ms | total {total_ms:.0f}ms | {ss_txt}",
            ]
            font = cv2.FONT_HERSHEY_SIMPLEX
            scale = max(0.45, min(0.7, w / 1300.0))
            thickness = 2 if w >= 700 else 1
            pad = 6
            x = 10
            y = 24
            for line in lines:
                text = str(line)
                (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
                cv2.rectangle(
                    frame,
                    (max(0, x - pad), max(0, y - th - pad)),
                    (min(w - 1, x + tw + pad), min(h - 1, y + pad)),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(frame, text, (x, y), font, scale, (255, 255, 255), thickness, cv2.LINE_AA)
                y += int(24 * scale) + 14
        except Exception:
            return

    def _draw_overlay(self, frame: np.ndarray, meta: Dict[str, Any]) -> bool:
        # Prefer the same filtered list used for the already-rendered frame.
        # When --hide-unknown is enabled, `events` / `visible_tracks` are empty
        # for hidden unknowns, so do not fall back to security_events in that case.
        if "events" in meta:
            overlay_items = meta.get("events") or []
        elif "visible_tracks" in meta:
            overlay_items = meta.get("visible_tracks") or []
        else:
            overlay_items = meta.get("security_events") or []

        if not isinstance(overlay_items, (list, tuple)) or not overlay_items:
            return False

        out_h, out_w = frame.shape[:2]
        src_w, src_h = self._parse_size(
            meta.get("processed_size")
            or meta.get("frame_size")
            or meta.get("source_size")
            or [meta.get("frame_width"), meta.get("frame_height")]
        )
        if src_w <= 0 or src_h <= 0:
            src_w, src_h = out_w, out_h
        sx = float(out_w) / float(max(1, src_w))
        sy = float(out_h) / float(max(1, src_h))
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = max(0.45, min(0.8, out_w / 1400.0))
        thickness = max(2, int(round(out_w / 700.0)))
        label_thickness = max(1, thickness - 1)
        drew = False

        for item in overlay_items:
            parsed = self._overlay_item_to_box_label(item)
            if parsed is None:
                continue
            x1, y1, x2, y2, label, is_known = parsed
            xx1 = int(max(0, min(out_w - 1, round(float(x1) * sx))))
            yy1 = int(max(0, min(out_h - 1, round(float(y1) * sy))))
            xx2 = int(max(0, min(out_w - 1, round(float(x2) * sx))))
            yy2 = int(max(0, min(out_h - 1, round(float(y2) * sy))))
            if xx2 <= xx1 or yy2 <= yy1:
                continue
            color = (0, 255, 0) if bool(is_known) else (0, 255, 255)
            label = str(label or "Unknown")
            cv2.rectangle(frame, (xx1, yy1), (xx2, yy2), color, thickness)
            (tw, th), _ = cv2.getTextSize(label, font, scale, label_thickness)
            y_text = max(th + 4, yy1 - 6)
            cv2.rectangle(
                frame,
                (xx1, max(0, y_text - th - 5)),
                (min(out_w - 1, xx1 + tw + 6), min(out_h - 1, y_text + 4)),
                (0, 0, 0),
                -1,
            )
            cv2.putText(frame, label, (xx1 + 3, y_text), font, scale, color, label_thickness, cv2.LINE_AA)
            drew = True
        return bool(drew)

    def _poll_processed(self, timeout: float = 0.0) -> None:
        try:
            frame, ts, meta, seq = self._poll_buffer(self.buffer, self._last_processed_seq, timeout=timeout)
        except Exception as exc:
            self._last_error = f"processed buffer read failed: {exc}"
            return
        if int(seq) <= int(self._last_processed_seq):
            return
        self._last_processed_seq = int(seq)
        meta = dict(meta or {})
        has_frame = bool(frame is not None and getattr(frame, "size", 0) != 0)
        if not has_frame and not meta:
            # Initial empty RenderedFrame state. Advance the sequence but do not
            # report processed_ready until the AI thread publishes a real frame.
            return
        self._last_frame_ts = float(ts or time.time())
        if has_frame:
            try:
                h, w = frame.shape[:2]
                meta.setdefault("processed_size", [int(w), int(h)])
                meta.setdefault("frame_size", [int(w), int(h)])
                self._latest_processed_frame = self._prepare_frame(frame)
            except Exception as exc:
                self._last_error = f"processed frame prepare failed: {exc}"
        self._latest_meta = meta
        self._latest_processed_arrival_ts = time.time()

    def _poll_raw(self, timeout: float = 0.0) -> None:
        if self.raw_buffer is None:
            return
        try:
            frame, ts, _meta, seq = self._poll_buffer(self.raw_buffer, self._last_raw_seq, timeout=timeout)
        except Exception as exc:
            self._last_error = f"raw buffer read failed: {exc}"
            return
        if int(seq) <= int(self._last_raw_seq):
            return
        self._last_raw_seq = int(seq)
        if frame is not None and getattr(frame, "size", 0) != 0:
            self._latest_raw_frame = frame
            self._latest_raw_arrival_ts = float(ts or time.time())

    def _make_output_frame(self) -> Tuple[np.ndarray, str]:
        bootstrap_w = int(self.width or 1280)
        bootstrap_h = int(self.height or 720)
        bootstrap_w, bootstrap_h = self._even_size(bootstrap_w, bootstrap_h)

        if self.mode == "hybrid" and self.raw_buffer is not None:
            self._poll_raw(timeout=0.001)
            if self._latest_raw_frame is not None:
                out = self._prepare_frame(self._latest_raw_frame).copy()
                if self._overlay_is_fresh() and self._draw_overlay(out, self._latest_meta):
                    self._draw_stats_overlay(out, self._latest_meta, source="overlay")
                    return out, "overlay"

                # Safety fallback: never let WebRTC look permanently like the
                # raw MediaMTX camera feed after detections have started.  If
                # overlay metadata is missing/stale but the latest processed
                # frame contains drawn boxes, publish that annotated frame.
                try:
                    has_drawn_boxes = int((self._latest_meta or {}).get("shown") or 0) > 0
                except Exception:
                    has_drawn_boxes = False
                if has_drawn_boxes and self._latest_processed_frame is not None:
                    out = self._latest_processed_frame.copy()
                    self._draw_stats_overlay(out, self._latest_meta, source="processed")
                    return out, "processed"

                self._draw_stats_overlay(out, self._latest_meta, source="raw")
                return out, "raw"

        if self._latest_processed_frame is not None:
            out = self._latest_processed_frame.copy()
            self._draw_stats_overlay(out, self._latest_meta, source="processed")
            return out, "processed"

        out = self._placeholder_frame(*(self._size or (bootstrap_w, bootstrap_h)))
        self._draw_stats_overlay(out, self._latest_meta, source="placeholder")
        return out, "placeholder"

    def _loop(self) -> None:
        min_dt = 1.0 / self.fps
        next_publish_at = 0.0

        for buf in (self.buffer, self.raw_buffer):
            if buf is not None and hasattr(buf, "add_client"):
                try:
                    buf.add_client()
                except Exception:
                    pass

        try:
            while not self._stop_evt.is_set():
                # Always consume the newest detection metadata. Never queue old detections.
                self._poll_processed(timeout=0.001)

                now = time.monotonic()
                if now < next_publish_at:
                    time.sleep(min(0.004, max(0.0, next_publish_at - now)))
                    continue
                next_publish_at = now + min_dt

                out, source = self._make_output_frame()
                ok = self._write_frame_to_ffmpeg(out, source=source)
                if not ok:
                    # Do not burn CPU when MediaMTX/FFmpeg is unavailable.
                    time.sleep(self.restart_delay_seconds)
        finally:
            for buf in (self.buffer, self.raw_buffer):
                if buf is not None and hasattr(buf, "remove_client"):
                    try:
                        buf.remove_client()
                    except Exception:
                        pass
            self._close_proc()
            self._done_evt.set()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        if not self.mediamtx_rtsp_base:
            raise RuntimeError("MEDIAMTX_RTSP is empty; cannot publish processed stream")
        # Validate FFmpeg synchronously so startup logs show the real problem.
        self._resolve_ffmpeg_bin()
        self._stop_evt.clear()
        self._done_evt.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name=f"rtsp-publisher-{self._safe_log_name(self.stream_name)}",
            daemon=True,
        )
        self._thread.start()

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

    def wait_until_ready(self, timeout: float = 8.0) -> bool:
        return bool(self._ready_evt.wait(timeout=max(0.1, float(timeout or 0.1))))

    def status(self) -> Dict[str, Any]:
        w, h = self._size or (self.width, self.height)
        return {
            "stream_name": self.stream_name,
            "rtsp_output": self.rtsp_output,
            "mode": self.mode,
            "fps": self.fps,
            "publish_fps": float(self._publish_fps_ema or 0.0),
            "draw_stats": bool(self.draw_stats),
            "size": {"width": int(w or 0), "height": int(h or 0)},
            "codec_requested": self.codec,
            "codec_active": self._active_codec or self._select_codec(),
            "bitrate": self.bitrate,
            "bufsize": self.bufsize,
            "gop": self.gop,
            "ready": bool(self._ready_evt.is_set()),
            "path_ready": bool(self._ready_evt.is_set()),
            "video_ready": bool(self._real_frames_published > 0),
            "raw_motion_ready": bool(self._raw_frames_published > 0),
            "processed_ready": bool(self._latest_processed_arrival_ts > 0),
            "overlay_ready": bool(self._overlay_frames_published > 0),
            "alive": self.is_alive(),
            "frames_published": int(self._frames_published),
            "real_frames_published": int(self._real_frames_published),
            "raw_frames_published": int(self._raw_frames_published),
            "processed_frames_published": int(self._processed_frames_published),
            "overlay_frames_published": int(self._overlay_frames_published),
            "placeholder_frames_published": int(self._placeholder_frames_published),
            "frames_dropped": int(self._frames_dropped),
            "ffmpeg_restarts": int(max(0, self._restarts - 1)),
            "last_error": self._last_error,
            "last_frame_ts": float(self._last_frame_ts),
            "last_processed_arrival_ts": float(self._latest_processed_arrival_ts),
            "last_raw_arrival_ts": float(self._latest_raw_arrival_ts),
            "last_publish_ts": float(self._last_publish_ts),
            "last_cmd": shlex.join(self._last_cmd) if self._last_cmd else "",
        }
