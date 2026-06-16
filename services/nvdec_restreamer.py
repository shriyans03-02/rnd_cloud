from __future__ import annotations

import os
import random
import shlex
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

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


def _split_flags(value: str) -> List[str]:
    value = str(value or "").strip()
    if not value:
        return []
    try:
        return shlex.split(value)
    except Exception:
        return []


def _redact_url(url: str) -> str:
    raw = str(url or "")
    try:
        p = urlsplit(raw)
        if not p.scheme or not p.netloc:
            return raw
        if p.username is None and p.password is None:
            return raw
        host = p.hostname or ""
        if ":" in host and not host.startswith("["):
            host = f"[{host}]"
        port = f":{p.port}" if p.port else ""
        user = p.username or "user"
        netloc = f"{user}:***@{host}{port}"
        return urlunsplit((p.scheme, netloc, p.path, p.query, p.fragment))
    except Exception:
        return raw.replace("://", "://***:***@", 1) if "://" in raw and "@" in raw else raw


def _redact_cmd(cmd: Sequence[str]) -> str:
    return shlex.join([_redact_url(x) if str(x).lower().startswith(("rtsp://", "rtsps://")) else str(x) for x in cmd])


def _safe_log_name(value: Any) -> str:
    safe = str(value or "stream").strip().replace("/", "_").replace("\\", "_")
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in safe) or "stream"


_ffmpeg_feature_cache: Dict[Tuple[str, str, str], bool] = {}
_ffmpeg_feature_lock = threading.Lock()


def _ffmpeg_has(ffmpeg_bin: str, list_name: str, needle: str) -> bool:
    key = (str(ffmpeg_bin or "ffmpeg"), str(list_name), str(needle))
    with _ffmpeg_feature_lock:
        if key in _ffmpeg_feature_cache:
            return bool(_ffmpeg_feature_cache[key])
    ok = False
    try:
        proc = subprocess.run(
            [str(ffmpeg_bin or "ffmpeg"), "-hide_banner", f"-{list_name}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=6,
            check=False,
        )
        ok = str(needle) in str(proc.stdout or "")
    except Exception:
        ok = False
    with _ffmpeg_feature_lock:
        _ffmpeg_feature_cache[key] = bool(ok)
    return bool(ok)


class NVDecRestreamManager:
    """Runs one FFmpeg process per DB camera to create a clean H264 AI stream.

    Default production flow in this patched build:

        DB camera RTSP URL  -> FFmpeg robust TCP HEVC/H264 decode -> MediaMTX ai/cam<ID>

    The old hop through MediaMTX live/cam<ID> can still be used by setting
    NVDEC_RESTREAM_INPUT_MODE=mediamtx, but direct mode is smoother on cloud
    deployments because it removes one RTSP reader/remuxer from the fragile H265
    path and gives FFmpeg the camera/NVR stream directly.
    """

    def __init__(self, cameras: Sequence[Dict[str, Any]]):
        self.cameras = list(cameras)
        self.stop_evt = threading.Event()
        self._lock = threading.Lock()
        self._procs: Dict[int, subprocess.Popen] = {}
        self._threads: List[threading.Thread] = []
        self._last_encoder_by_cam: Dict[int, str] = {}
        self._force_libx264 = False

        self.ffmpeg_bin = _env("FFMPEG_BIN", "ffmpeg") or "ffmpeg"
        # Resolve once for feature checks; Popen can still use the configured value.
        self.ffmpeg_probe_bin = shutil.which(self.ffmpeg_bin) or self.ffmpeg_bin
        self.rtsp_base = _env("MEDIAMTX_RTSP", "rtsp://127.0.0.1:8554").rstrip("/")
        self.input_mode = _env("NVDEC_RESTREAM_INPUT_MODE", "direct").strip().lower() or "direct"
        if self.input_mode not in {"direct", "mediamtx", "live"}:
            self.input_mode = "direct"
        self.input_prefix = _env("NVDEC_RESTREAM_INPUT_PREFIX", "live/cam").strip("/")
        self.output_prefix = _env("NVDEC_RESTREAM_OUTPUT_PREFIX", "ai/cam").strip("/")
        self.rtsp_transport = _env("NVDEC_RESTREAM_RTSP_TRANSPORT", _env("MEDIAMTX_CAMERA_RTSP_TRANSPORT", "tcp")).strip().lower() or "tcp"

        self.input_codec = _env("NVDEC_RESTREAM_INPUT_CODEC", "auto").strip().lower() or "auto"
        decoder = _env("NVDEC_RESTREAM_DECODER", "software").strip().lower()
        # Supported values:
        #   software/cpu/native/ffmpeg/empty -> let FFmpeg auto-select the CPU decoder
        #   auto with input_codec=auto        -> let FFmpeg auto-detect codec+decoder
        #   auto with input_codec=hevc/h264   -> old CUVID behaviour for explicit codec
        #   cuda/cuvid/nvdec                 -> CUVID decoder for explicit codec only
        #   h264/hevc                        -> force native CPU decoder
        #   h264_cuvid/hevc_cuvid            -> force NVIDIA decoder
        if decoder in {"", "software", "cpu", "native", "ffmpeg", "none"}:
            decoder = ""
        elif decoder == "auto":
            if self.input_codec in {"hevc", "h265"}:
                decoder = "hevc_cuvid"
            elif self.input_codec in {"h264", "avc"}:
                decoder = "h264_cuvid"
            else:
                decoder = ""
        elif decoder in {"cuda", "cuvid", "nvdec"}:
            if self.input_codec in {"hevc", "h265"}:
                decoder = "hevc_cuvid"
            elif self.input_codec in {"h264", "avc"}:
                decoder = "h264_cuvid"
            else:
                decoder = ""
        self.decoder = decoder
        self.use_hw_decode = bool(self.decoder and self.decoder.endswith("_cuvid"))

        self.encoder_requested = _env("NVDEC_RESTREAM_ENCODER", "libx264").strip().lower() or "libx264"
        self.width = _int_env("NVDEC_RESTREAM_WIDTH", 852)
        self.height = _int_env("NVDEC_RESTREAM_HEIGHT", 480)
        self.fps = _int_env("NVDEC_RESTREAM_FPS", 10)
        self.bitrate = _env("NVDEC_RESTREAM_BITRATE", "1400k")
        self.bufsize = _env("NVDEC_RESTREAM_BUFSIZE", "2800k")
        self.gop = _int_env("NVDEC_RESTREAM_GOP", max(1, self.fps))
        self.preset = _env("NVDEC_RESTREAM_PRESET", "ultrafast")
        self.gpu_scale = _bool_env("NVDEC_RESTREAM_USE_GPU_SCALE", False)

        self.thread_queue_size = _int_env("NVDEC_RESTREAM_THREAD_QUEUE_SIZE", 512)
        self.reorder_queue_size = _int_env("NVDEC_RESTREAM_REORDER_QUEUE_SIZE", 2048)
        self.max_delay = _env("NVDEC_RESTREAM_MAX_DELAY", "5000000")
        self.rtbufsize = _env("NVDEC_RESTREAM_RTBUF_SIZE", "64M")
        self.analyzeduration = _env("NVDEC_RESTREAM_ANALYZEDURATION", "20000000")
        self.probesize = _env("NVDEC_RESTREAM_PROBESIZE", "20000000")
        self.output_pkt_size = _int_env("NVDEC_RESTREAM_OUTPUT_PKT_SIZE", 1200)
        self.loglevel = _env("NVDEC_RESTREAM_LOGLEVEL", "warning")
        self.extra_input_flags = _split_flags(_env("NVDEC_RESTREAM_EXTRA_INPUT_FLAGS", ""))
        self.extra_output_flags = _split_flags(_env("NVDEC_RESTREAM_EXTRA_OUTPUT_FLAGS", ""))

        self.restart_seconds = _float_env("NVDEC_RESTREAM_RESTART_SECONDS", 2.0)
        self.restart_max_seconds = _float_env("NVDEC_RESTREAM_RESTART_MAX_SECONDS", 30.0)
        self.stable_seconds = _float_env("NVDEC_RESTREAM_STABLE_SECONDS", 20.0)
        self.start_stagger_seconds = _float_env("NVDEC_RESTREAM_START_STAGGER_SECONDS", 0.50)
        self.auto_fallback_libx264 = _bool_env("NVDEC_RESTREAM_AUTO_FALLBACK_LIBX264", True)
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

    def _input_url(self, cam: Dict[str, Any]) -> str:
        cam_id = int(cam.get("id"))
        if self.input_mode == "direct":
            for key in ("rtsp_source", "url", "source", "src"):
                value = str(cam.get(key) or "").strip()
                if value.lower().startswith(("rtsp://", "rtsps://")):
                    return value
            print(f"[NVDEC] cam_id={cam_id} direct RTSP URL missing; falling back to MediaMTX live path")
        return f"{self.rtsp_base}/{self.input_prefix}{cam_id}"

    def _output_url(self, cam_id: int) -> str:
        return f"{self.rtsp_base}/{self.output_prefix}{int(cam_id)}"

    def _urls(self, cam: Dict[str, Any]) -> Tuple[str, str]:
        cam_id = int(cam.get("id"))
        return self._input_url(cam), self._output_url(cam_id)

    def _filter(self) -> str:
        # When using CPU/software decode, frames are already in system memory.
        # hwdownload is valid only after CUDA/CUVID hardware decode.
        if not self.use_hw_decode:
            return f"scale={self.width}:{self.height}:flags=fast_bilinear,fps={self.fps},format=yuv420p"
        if self.gpu_scale:
            return f"scale_cuda={self.width}:{self.height}:format=nv12,hwdownload,format=nv12,fps={self.fps},format=yuv420p"
        return f"hwdownload,format=nv12,scale={self.width}:{self.height}:flags=fast_bilinear,fps={self.fps},format=yuv420p"

    def _active_encoder(self) -> str:
        requested = (self.encoder_requested or "libx264").lower().strip()
        if self._force_libx264:
            return "libx264"
        if requested in {"auto", "nvenc", "h264_nvenc"}:
            if _ffmpeg_has(self.ffmpeg_probe_bin, "encoders", "h264_nvenc"):
                return "h264_nvenc"
            return "libx264"
        return requested

    def _encoder_args(self, encoder: str) -> List[str]:
        encoder = str(encoder or "libx264").strip().lower()
        if encoder == "h264_nvenc":
            return [
                "-c:v", "h264_nvenc",
                "-preset", _env("NVDEC_RESTREAM_NVENC_PRESET", "p1"),
                "-tune", _env("NVDEC_RESTREAM_NVENC_TUNE", "ll"),
                "-rc", "cbr",
                "-b:v", self.bitrate,
                "-maxrate", self.bitrate,
                "-bufsize", self.bufsize,
                "-pix_fmt", "yuv420p",
                "-g", str(self.gop),
                "-bf", "0",
                "-forced-idr", "1",
            ]
        # libx264 remains the safest option on cloud images whose FFmpeg exposes
        # NVENC headers but cannot load the NVIDIA encoder at runtime.
        return [
            "-c:v", "libx264",
            "-preset", self.preset,
            "-tune", "zerolatency",
            "-threads", _env("NVDEC_RESTREAM_X264_THREADS", "1"),
            "-profile:v", "baseline",
            "-level:v", "3.1",
            "-pix_fmt", "yuv420p",
            "-b:v", self.bitrate,
            "-maxrate", self.bitrate,
            "-bufsize", self.bufsize,
            "-g", str(self.gop),
            "-keyint_min", str(self.gop),
            "-sc_threshold", "0",
            "-bf", "0",
            "-x264-params", f"bframes=0:keyint={self.gop}:min-keyint={self.gop}:scenecut=0",
        ]

    def _cmd(self, cam: Dict[str, Any]) -> List[str]:
        inp, out = self._urls(cam)
        encoder = self._active_encoder()
        self._last_encoder_by_cam[int(cam.get("id"))] = encoder

        cmd: List[str] = [
            self.ffmpeg_bin,
            "-hide_banner", "-nostdin", "-loglevel", self.loglevel,
            "-threads", "1",
        ]

        if str(inp).lower().startswith(("rtsp://", "rtsps://")):
            cmd += [
                "-rtsp_transport", self.rtsp_transport,
                "-thread_queue_size", str(self.thread_queue_size),
                "-use_wallclock_as_timestamps", "1",
                # Do not use nobuffer here. Cloud RTSP jitter needs probe and reorder room.
                "-fflags", "+genpts+igndts+discardcorrupt",
                "-flags2", "+showall",
                "-err_detect", "ignore_err",
                "-max_delay", str(self.max_delay),
                "-rtbufsize", str(self.rtbufsize),
                "-reorder_queue_size", str(self.reorder_queue_size),
                "-analyzeduration", str(self.analyzeduration),
                "-probesize", str(self.probesize),
            ]
            if self.rtsp_transport == "tcp":
                cmd += ["-rtsp_flags", "prefer_tcp"]
        else:
            cmd += ["-thread_queue_size", str(self.thread_queue_size)]

        cmd += list(self.extra_input_flags)

        if self.use_hw_decode:
            cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

        # Empty decoder means: let FFmpeg choose the decoder automatically.
        if self.decoder:
            cmd += ["-c:v", self.decoder]

        cmd += [
            "-i", inp,
            "-map", "0:v:0",
            "-an", "-sn", "-dn",
            "-vf", self._filter(),
            *self._encoder_args(encoder),
            "-max_muxing_queue_size", "1024",
            "-muxdelay", "0",
            "-muxpreload", "0",
            "-f", "rtsp",
            "-rtsp_transport", "tcp",
        ]
        if self.output_pkt_size > 0:
            cmd += ["-pkt_size", str(self.output_pkt_size)]
        cmd += list(self.extra_output_flags)
        cmd += [out]
        return cmd

    def _start_one(self, cam: Dict[str, Any]) -> Optional[subprocess.Popen]:
        cam_id = int(cam.get("id"))
        cmd = self._cmd(cam)
        log_path = self.log_dir / f"nvdec_cam{cam_id}.log"
        try:
            with open(log_path, "ab", buffering=0) as f:
                f.write(("\n[cmd] " + _redact_cmd(cmd) + "\n").encode("utf-8", errors="ignore"))
            log_f = open(log_path, "ab", buffering=0)
            try:
                proc = subprocess.Popen(cmd, stdout=log_f, stderr=log_f, close_fds=True)
            finally:
                try:
                    log_f.close()
                except Exception:
                    pass
            with self._lock:
                self._procs[cam_id] = proc
            inp, out = self._urls(cam)
            print(
                f"[NVDEC] started cam_id={cam_id} mode={self.input_mode} codec={self.input_codec}/{self.decoder or 'auto'} "
                f"encoder={self._last_encoder_by_cam.get(cam_id, self.encoder_requested)} "
                f"{self.width}x{self.height}@{self.fps} {_redact_url(inp)} -> {out} log={log_path}"
            )
            return proc
        except Exception as e:
            print(f"[NVDEC] failed to start cam_id={cam_id}: {e} cmd={_redact_cmd(cmd)}")
            return None

    def _restart_delay(self, failures: int) -> float:
        try:
            delay = float(self.restart_seconds) * (2.0 ** max(0, min(int(failures) - 1, 6)))
        except Exception:
            delay = float(self.restart_seconds)
        delay = min(float(self.restart_max_seconds), max(float(self.restart_seconds), delay))
        # Small jitter prevents all cameras from reconnecting on the same millisecond.
        return max(0.1, delay + (random.random() * 0.2 - 0.1) * delay)

    def _watch_one(self, cam: Dict[str, Any]) -> None:
        cam_id = int(cam.get("id"))
        failures = 0
        while not self.stop_evt.is_set():
            start_mono = time.monotonic()
            proc = self._start_one(cam)
            if proc is None:
                failures += 1
                self.stop_evt.wait(self._restart_delay(failures))
                continue

            while not self.stop_evt.is_set():
                rc = proc.poll()
                if rc is None:
                    time.sleep(1.0)
                    continue
                break

            rc = proc.poll()
            if rc is None:
                continue
            runtime = time.monotonic() - start_mono
            with self._lock:
                self._procs.pop(cam_id, None)

            encoder_used = self._last_encoder_by_cam.get(cam_id, self.encoder_requested)
            if (
                self.auto_fallback_libx264
                and encoder_used == "h264_nvenc"
                and runtime < 5.0
                and not self._force_libx264
            ):
                self._force_libx264 = True
                print(f"[NVDEC] cam_id={cam_id} h264_nvenc exited quickly; forcing libx264 fallback for restreamers")

            if runtime >= float(self.stable_seconds):
                failures = 0
            else:
                failures += 1
            delay = self._restart_delay(failures)
            print(f"[NVDEC] cam_id={cam_id} exited rc={rc} after {runtime:.1f}s; restarting in {delay:.1f}s")
            self.stop_evt.wait(delay)

    def start(self) -> None:
        if not self.cameras:
            print("[NVDEC] no DB cameras to restream")
            return
        print(
            f"[NVDEC] starting restreamers cameras={len(self.cameras)} mode={self.input_mode} "
            f"encoder={self.encoder_requested} decoder={self.decoder or 'auto'}"
        )
        for cam in self.cameras:
            t = threading.Thread(target=self._watch_one, args=(dict(cam),), daemon=True)
            t.start()
            self._threads.append(t)
            if self.start_stagger_seconds > 0:
                time.sleep(self.start_stagger_seconds)
        warm = _float_env("NVDEC_RESTREAM_WARMUP_SECONDS", 8.0)
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
