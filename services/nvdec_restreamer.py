from __future__ import annotations

import os
import shlex
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from app.core.config import settings
from app.services.mediamtx_autoconfig import load_active_cameras_from_db


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


def _int_env(name: str, default: int) -> int:
    try:
        return int(float(_env(name, str(default))))
    except Exception:
        return int(default)


def _float_env(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except Exception:
        return float(default)


def _csv_ints(value: str) -> List[int]:
    out: List[int] = []
    for part in str(value or "").replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            out.append(int(part))
        except Exception:
            pass
    return out


class NVDecRestreamManager:
    """Runs one FFmpeg process per DB camera to turn HEVC/H265 RTSP into clean AI RTSP.

    Flow:
      MediaMTX live/cam<ID>  -> ffmpeg hevc_cuvid/h264_cuvid -> MediaMTX ai/cam<ID>

    The Python/OpenCV pipeline then reads ai/cam<ID>, which is lower resolution and
    H264, reducing CPU decode pressure and avoiding unstable HEVC reference-frame errors.
    """

    def __init__(self, cameras: Sequence[Dict[str, Any]]):
        self.cameras = list(cameras)
        self.stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._procs: Dict[int, subprocess.Popen] = {}
        self._threads: List[threading.Thread] = []

        self.ffmpeg_bin = _env("FFMPEG_BIN", "ffmpeg") or "ffmpeg"
        self.rtsp_base = _env("MEDIAMTX_RTSP", "rtsp://127.0.0.1:8554").rstrip("/")
        self.input_prefix = _env("NVDEC_RESTREAM_INPUT_PREFIX", "live/cam").strip("/")
        self.output_prefix = _env("NVDEC_RESTREAM_OUTPUT_PREFIX", "ai/cam").strip("/")
        self.input_codec = _env("NVDEC_RESTREAM_INPUT_CODEC", "hevc").strip().lower()
        decoder = _env("NVDEC_RESTREAM_DECODER", "auto").strip().lower()
        # Supported values:
        #   auto      -> use NVIDIA CUVID decoder (old behaviour)
        #   software  -> let FFmpeg auto-select the CPU decoder
        #   cpu       -> let FFmpeg auto-select the CPU decoder
        #   h264/hevc -> force FFmpeg native CPU decoder
        #   h264_cuvid/hevc_cuvid -> force NVIDIA decoder
        # Do not pass "none" to FFmpeg; there is no decoder called none.
        if decoder in {"", "software", "cpu", "native", "ffmpeg"}:
            decoder = ""
        elif decoder == "none":
            decoder = ""
        elif decoder == "auto":
            decoder = "hevc_cuvid" if self.input_codec in {"hevc", "h265"} else "h264_cuvid"
        self.decoder = decoder
        self.use_hw_decode = bool(self.decoder and self.decoder.endswith("_cuvid"))
        self.encoder = _env("NVDEC_RESTREAM_ENCODER", "libx264").strip().lower() or "libx264"
        self.width = _int_env("NVDEC_RESTREAM_WIDTH", 852)
        self.height = _int_env("NVDEC_RESTREAM_HEIGHT", 480)
        self.fps = _int_env("NVDEC_RESTREAM_FPS", 20)
        self.bitrate = _env("NVDEC_RESTREAM_BITRATE", "1200k")
        self.bufsize = _env("NVDEC_RESTREAM_BUFSIZE", "2400k")
        self.gop = _int_env("NVDEC_RESTREAM_GOP", max(1, self.fps))
        self.preset = _env("NVDEC_RESTREAM_PRESET", "ultrafast")
        self.gpu_scale = _bool_env("NVDEC_RESTREAM_USE_GPU_SCALE", True)
        self.restart_seconds = _float_env("NVDEC_RESTREAM_RESTART_SECONDS", 2.0)
        self.log_dir = Path(_env("NVDEC_RESTREAM_LOG_DIR", "logs/nvdec_restream"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_env(cls) -> "NVDecRestreamManager":
        cams = load_active_cameras_from_db()
        ids = set(_csv_ints(_env("NVDEC_RESTREAM_CAMERA_IDS", "")))
        if ids:
            cams = [c for c in cams if int(c.get("id", 0)) in ids]
        max_cams = _int_env("NVDEC_RESTREAM_MAX_CAMERAS", 0)
        if max_cams > 0:
            cams = cams[:max_cams]
        return cls(cams)

    @staticmethod
    def enabled() -> bool:
        return _bool_env("NVDEC_RESTREAM_ENABLED", False)

    def _urls(self, cam_id: int) -> tuple[str, str]:
        return (
            f"{self.rtsp_base}/{self.input_prefix}{int(cam_id)}",
            f"{self.rtsp_base}/{self.output_prefix}{int(cam_id)}",
        )

    def _filter(self) -> str:
        # When using CPU/software decode, frames are already in system memory.
        # hwdownload is valid only after CUDA/CUVID hardware decode.
        if not self.use_hw_decode:
            return f"scale={self.width}:{self.height},fps={self.fps},format=yuv420p"
        if self.gpu_scale:
            return f"scale_cuda={self.width}:{self.height},hwdownload,format=nv12"
        return f"hwdownload,format=nv12,scale={self.width}:{self.height},fps={self.fps},format=yuv420p"

    def _encoder_args(self) -> List[str]:
        if self.encoder == "h264_nvenc":
            return [
                "-c:v", "h264_nvenc",
                "-preset", _env("NVDEC_RESTREAM_NVENC_PRESET", "p1"),
                "-tune", "ll",
                "-b:v", self.bitrate,
                "-maxrate", self.bitrate,
                "-bufsize", self.bufsize,
                "-g", str(self.gop),
                "-bf", "0",
            ]
        # A100 cloud instances often have NVDEC but no NVENC, so libx264 is the safe default.
        return [
            "-c:v", "libx264",
            "-preset", self.preset,
            "-tune", "zerolatency",
            "-profile:v", "baseline",
            "-pix_fmt", "yuv420p",
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-bufsize", self.bufsize,
            "-g", str(self.gop),
            "-keyint_min", str(self.gop),
            "-sc_threshold", "0",
        ]

    def _cmd(self, cam_id: int) -> List[str]:
        inp, out = self._urls(cam_id)

        cmd: List[str] = [
            self.ffmpeg_bin,
            "-hide_banner", "-nostdin", "-loglevel", _env("NVDEC_RESTREAM_LOGLEVEL", "warning"),
            "-rtsp_transport", "tcp",
            # Do not use nobuffer/low_delay here. Some Hikvision/H.264 streams need
            # enough probe data before FFmpeg can determine width/height and SPS/PPS.
            "-fflags", "+genpts+igndts+discardcorrupt",
            "-err_detect", "ignore_err",
            "-analyzeduration", _env("NVDEC_RESTREAM_ANALYZEDURATION", "15000000"),
            "-probesize", _env("NVDEC_RESTREAM_PROBESIZE", "15000000"),
        ]

        if self.use_hw_decode:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

        # Empty decoder means: let FFmpeg choose the software decoder automatically.
        if self.decoder:
            cmd += ["-c:v", self.decoder]

        cmd += [
            "-i", inp,
            "-an",
            "-vf", self._filter(),
            *self._encoder_args(),
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
            out,
        ]
        return cmd

    def _start_one(self, cam: Dict[str, Any]) -> Optional[subprocess.Popen]:
        cam_id = int(cam.get("id"))
        cmd = self._cmd(cam_id)
        log_path = self.log_dir / f"nvdec_cam{cam_id}.log"
        try:
            log_f = open(log_path, "ab", buffering=0)
            proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f)
            with self._lock:
                self._procs[cam_id] = proc
            inp, out = self._urls(cam_id)
            print(
                f"[NVDEC] started cam_id={cam_id} codec={self.input_codec}/{self.decoder} "
                f"encoder={self.encoder} {self.width}x{self.height}@{self.fps} {inp} -> {out} log={log_path}"
            )
            return proc
        except Exception as e:
            print(f"[NVDEC] failed to start cam_id={cam_id}: {e} cmd={shlex.join(cmd)}")
            return None

    def _watch_one(self, cam: Dict[str, Any]) -> None:
        cam_id = int(cam.get("id"))
        proc = self._start_one(cam)
        while not self.stop_evt.is_set():
            if proc is None:
                time.sleep(self.restart_seconds)
                proc = self._start_one(cam)
                continue
            rc = proc.poll()
            if rc is None:
                time.sleep(1.0)
                continue
            print(f"[NVDEC] cam_id={cam_id} exited rc={rc}; restarting in {self.restart_seconds:.1f}s")
            with self._lock:
                self._procs.pop(cam_id, None)
            time.sleep(self.restart_seconds)
            proc = self._start_one(cam)

    def start(self) -> None:
        if not self.cameras:
            print("[NVDEC] no DB cameras to restream")
            return
        print(f"[NVDEC] starting restreamers cameras={len(self.cameras)} encoder={self.encoder} decoder={self.decoder}")
        for cam in self.cameras:
            t = threading.Thread(target=self._watch_one, args=(dict(cam),), daemon=True)
            t.start()
            self._threads.append(t)
            # Stagger startup a bit so 12 cameras do not all probe at the same millisecond.
            time.sleep(0.15)
        warm = _float_env("NVDEC_RESTREAM_WARMUP_SECONDS", 5.0)
        if warm > 0:
            print(f"[NVDEC] warmup {warm:.1f}s before AI pipeline opens ai/cam<ID> sources")
            time.sleep(warm)

    def stop(self) -> None:
        self.stop_evt.set()
        with self._lock:
            procs = list(self._procs.items())
            self._procs.clear()
        for cam_id, proc in procs:
            try:
                if proc.poll() is None:
                    proc.terminate()
            except Exception:
                pass
        deadline = time.time() + 4.0
        for cam_id, proc in procs:
            try:
                if proc.poll() is None:
                    remaining = max(0.1, deadline - time.time())
                    proc.wait(timeout=remaining)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        print("[NVDEC] stopped")
