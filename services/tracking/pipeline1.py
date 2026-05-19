from __future__ import annotations

import argparse
import contextlib
import ctypes
import gzip
import math
import os
import shlex
import sys
from datetime import datetime
from collections import deque
import queue
import threading
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch

# --- Room analytics / Excel export (separate module) ---
# Uses room_presence_analytics.py (kept unchanged) to build spatio-temporal segments and export
# the Excel in the requested Summary format.
try:
    from room_presence_analytics import SpatioTemporalRoomTracker, RoomTopology
except Exception:
    SpatioTemporalRoomTracker = None  # type: ignore
    RoomTopology = None  # type: ignore

def parse_room_graph_str(graph_str: str, rooms: List[str]) -> Dict[str, List[str]]:
    """Parse a simple undirected room adjacency graph.

    Format: comma-separated edges like:
        c1-c2,c2-c5,c5-c4,c2-c3,c3-c4
    Also accepts ':' instead of '-'.

    If graph_str is empty, we build the default topology you described:

        - c1 <-> c2
        - c2 <-> c3
        - c2 <-> c5 (NO camera in c5)
        - c5 <-> c4
        - c3 <-> c4 (door is far from the camera)

    Notes:
        - The default edge order intentionally prefers paths through c5 (no camera)
          over c3<->c4 when multiple shortest paths exist.
    """
    rooms = [str(r) for r in (rooms or []) if str(r).strip()]
    if not graph_str:
        # Default topology (ordered): prefer c5 path over the far c3<->c4 doorway.
        graph_str = "c1-c2,c2-c5,c5-c4,c2-c3,c3-c4"

    edges: List[Tuple[str, str]] = []
    for tok in str(graph_str).split(","):
        tok = tok.strip()
        if not tok:
            continue
        if "-" in tok:
            a, b = tok.split("-", 1)
        elif ":" in tok:
            a, b = tok.split(":", 1)
        else:
            continue
        a, b = a.strip(), b.strip()
        if not a or not b:
            continue
        edges.append((a, b))

    # Build adjacency (preserve edge insertion order; avoids sorting bias)
    g: Dict[str, List[str]] = {}

    def _add(u: str, v: str) -> None:
        u = str(u)
        v = str(v)
        g.setdefault(u, [])
        if v not in g[u]:
            g[u].append(v)

    for a, b in edges:
        _add(a, b)
        _add(b, a)

    # Ensure all declared rooms exist in the graph (even isolated)
    for r in rooms:
        g.setdefault(str(r), [])

    return g


# --- YOLO (Ultralytics) ---
try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

# --- StrongSORT (BoxMOT) ---
try:
    from boxmot import StrongSort as BoxMOTStrongSort
except Exception:
    try:
        from boxmot.trackers.strongsort.strongsort import StrongSort as BoxMOTStrongSort
    except Exception:
        BoxMOTStrongSort = None  # type: ignore

# --- TorchReID ---
try:
    from torchreid.utils import FeatureExtractor as TorchreidExtractor
except Exception:
    TorchreidExtractor = None

# --- InsightFace ---
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHT_OK = True
except Exception:
    insightface = None
    FaceAnalysis = None
    INSIGHT_OK = False

# --- ONNX Runtime (for InsightFace providers check) ---
try:
    import onnxruntime as ort
except Exception:
    ort = None

# --- SQLAlchemy / pgvector ---
try:
    from sqlalchemy import (
        Column,
        Integer,
        BigInteger,
        String,
        Boolean,
        DateTime,
        LargeBinary,
        ForeignKey,
        create_engine,
        select,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    from sqlalchemy.dialects.postgresql import ARRAY
    from sqlalchemy import Float

except Exception as e:
    raise RuntimeError("SQLAlchemy is required for --use-db mode") from e

try:
    from pgvector.sqlalchemy import Vector
except Exception:
    Vector = None


EXPECTED_DIM = 512
_yolo_lock = threading.Lock()

reid_lock = threading.Lock()
back_proj_lock = threading.Lock()

# Fallback IoU tracker params (used only when StrongSORT is disabled/unavailable)
FALLBACK_IOU_THRESH = 0.30

# ===== NEW: back-side / no-face matching embedding =====
BACK_HEAD_CUT_RATIO_DEFAULT = 0.18
BACK_SHAPE_SEED_DEFAULT = 1337
BACK_REID_WEIGHT_DEFAULT = 0.65
BACK_SHAPE_WEIGHT_DEFAULT = 0.35

# HOG settings (must match extract_service)
BACK_HOG_WIN = (64, 128)     # (w, h)
BACK_HOG_BLOCK = (16, 16)
BACK_HOG_STRIDE = (8, 8)
BACK_HOG_CELL = (8, 8)
BACK_HOG_BINS = 9

_back_hog = None
_back_proj = None
_back_proj_in_dim = None


def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        return v
    return v / n


def l2_normalize_rows(m: np.ndarray) -> np.ndarray:
    m = np.asarray(m, dtype=np.float32)
    if m.ndim != 2:
        return m
    norms = np.linalg.norm(m, axis=1, keepdims=True)
    norms = np.where((norms == 0) | (~np.isfinite(norms)), 1.0, norms)
    return m / norms


def safe_iter_faces(obj):
    if obj is None:
        return []
    try:
        return list(obj)
    except TypeError:
        return [obj]


def extract_face_embedding(face):
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    return emb


def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    denom = a_area + b_area - inter
    return float(inter / denom) if denom > 0 else 0.0


def ioa_xyxy(inner: Tuple[float, float, float, float], outer: Tuple[float, float, float, float]) -> float:
    """
    Intersection-over-AREA(inner). Useful when 'inner' is a face box and 'outer' is a person box.
    If the face is fully inside the person box, IoA ~= 1.0 even though IoU would be tiny.
    """
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    inner_area = max(0.0, (ix2 - ix1)) * max(0.0, (iy2 - iy1))
    return float(inter / inner_area) if inner_area > 0 else 0.0


def _point_in_xyxy(px: float, py: float, box: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _cuda_ep_loadable() -> bool:
    if ort is None:
        return False
    try:
        if sys.platform.startswith("darwin"):
            return False
        capi_dir = os.path.join(os.path.dirname(ort.__file__), "capi")
        name = "onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so"
        lib_path = os.path.join(capi_dir, name)
        if not os.path.exists(lib_path):
            return False
        ctypes.CDLL(str(lib_path))
        return True
    except Exception:
        return False


# ===== back-body HOG projection embedding =====
def _init_back_shape_embedder():
    global _back_hog
    if _back_hog is None:
        try:
            _back_hog = cv2.HOGDescriptor(
                _winSize=BACK_HOG_WIN,
                _blockSize=BACK_HOG_BLOCK,
                _blockStride=BACK_HOG_STRIDE,
                _cellSize=BACK_HOG_CELL,
                _nbins=BACK_HOG_BINS,
            )
        except Exception:
            _back_hog = None


def _get_back_proj(in_dim: int, seed: int) -> Optional[np.ndarray]:
    global _back_proj, _back_proj_in_dim
    if in_dim <= 0:
        return None
    with back_proj_lock:
        if _back_proj is None or _back_proj_in_dim != int(in_dim):
            rng = np.random.RandomState(int(seed))
            mat = rng.normal(
                loc=0.0,
                scale=float(1.0 / max(1.0, np.sqrt(in_dim))),
                size=(EXPECTED_DIM, in_dim),
            ).astype(np.float32)
            _back_proj = mat
            _back_proj_in_dim = int(in_dim)
        return _back_proj


def back_shape_embed_hog(img_bgr: np.ndarray, seed: int) -> Optional[np.ndarray]:
    """
    Fast "silhouette-ish" embedding:
      - HOG on 64x128 grayscale
      - normalize hog
      - random projection to 512 (deterministic seed)
      - L2 normalize
    """
    if img_bgr is None or img_bgr.size == 0:
        return None
    _init_back_shape_embedder()
    if _back_hog is None:
        return None
    try:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, BACK_HOG_WIN, interpolation=cv2.INTER_AREA)
        hog = _back_hog.compute(g)
        if hog is None:
            return None
        hog = np.asarray(hog, dtype=np.float32).reshape(-1)
        if hog.size <= 0 or not np.isfinite(hog).all():
            return None
        hn = float(np.linalg.norm(hog))
        if hn > 0:
            hog = hog / hn
        proj = _get_back_proj(int(hog.size), seed=seed)
        if proj is None:
            return None
        emb = proj @ hog
        return l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
    except Exception:
        return None


def crop_back_body_no_face_from_tlbr(frame_bgr: np.ndarray, tlbr: Tuple[int, int, int, int], head_cut_ratio: float) -> np.ndarray:
    """
    Headless crop to guarantee "no face" in the crop. Must match extract_service.
    """
    x1, y1, x2, y2 = map(int, tlbr)
    H, W = frame_bgr.shape[:2]
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    h = y2 - y1
    y1b = int(y1 + h * float(max(0.0, min(0.45, head_cut_ratio))))
    if y1b >= y2 - 2:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    return frame_bgr[y1b:y2, x1:x2].copy()



def extract_clothing_descriptor(frame_bgr: np.ndarray, tlbr: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Lightweight clothing signature used ONLY to repair tracker ID switches around crossings.

    Design goals:
      - emphasize clothing colors/brightness (e.g. black dress vs white dress)
      - avoid head pixels as much as possible
      - cheap enough to run every frame on CPU

    Output is a small L2-normalized descriptor (NOT the 512-dim ReID embedding).
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return None

    try:
        x1, y1, x2, y2 = map(int, tlbr)
    except Exception:
        return None

    H, W = frame_bgr.shape[:2]
    x1 = max(0, min(W - 1, x1))
    y1 = max(0, min(H - 1, y1))
    x2 = max(0, min(W, x2))
    y2 = max(0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame_bgr[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None

    h, w = crop.shape[:2]
    if h < 24 or w < 12:
        return None

    # Trim likely head + outer borders so the signature focuses on clothing.
    top = int(round(h * 0.12))
    bottom = int(round(h * 0.92))
    left = int(round(w * 0.08))
    right = int(round(w * 0.92))
    if bottom - top < 16 or right - left < 8:
        top, bottom, left, right = 0, h, 0, w

    cloth = crop[top:bottom, left:right]
    if cloth is None or cloth.size == 0:
        return None

    parts = np.array_split(cloth, 2, axis=0)
    feats: List[np.ndarray] = []
    hsv_norm = np.asarray([180.0, 255.0, 255.0], dtype=np.float32)

    for part in parts:
        if part is None or part.size == 0:
            continue
        try:
            hsv = cv2.cvtColor(part, cv2.COLOR_BGR2HSV)
        except Exception:
            continue

        for ch, bins, rng in ((0, 12, [0, 180]), (1, 4, [0, 256]), (2, 4, [0, 256])):
            hist = cv2.calcHist([hsv], [ch], None, [bins], rng)
            if hist is None:
                continue
            hist = np.asarray(hist, dtype=np.float32).reshape(-1)
            s = float(hist.sum())
            if s > 0.0 and np.isfinite(s):
                hist = hist / s
            feats.append(hist)

        flat = hsv.reshape(-1, 3).astype(np.float32)
        if flat.size > 0:
            feats.append(np.mean(flat, axis=0) / hsv_norm)
            feats.append(np.std(flat, axis=0) / hsv_norm)

    if not feats:
        return None

    desc = np.concatenate(feats, axis=0).astype(np.float32)
    if desc.size <= 0 or not np.isfinite(desc).all():
        return None
    return l2_normalize(desc)


def _cosine_sim_anydim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        aa = np.asarray(a, dtype=np.float32).reshape(-1)
        bb = np.asarray(b, dtype=np.float32).reshape(-1)
        if aa.size <= 0 or bb.size <= 0 or aa.size != bb.size:
            return None
        if (not np.isfinite(aa).all()) or (not np.isfinite(bb).all()):
            return None
        aa = l2_normalize(aa)
        bb = l2_normalize(bb)
        return float(np.dot(aa, bb))
    except Exception:
        return None


def _box_center_score(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    """Soft spatial continuity score in [0, 1]."""
    try:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
    except Exception:
        return 0.0

    acx = 0.5 * (ax1 + ax2)
    acy = 0.5 * (ay1 + ay2)
    bcx = 0.5 * (bx1 + bx2)
    bcy = 0.5 * (by1 + by2)
    dist = float(np.hypot(acx - bcx, acy - bcy))

    aw = max(1.0, ax2 - ax1)
    ah = max(1.0, ay2 - ay1)
    bw = max(1.0, bx2 - bx1)
    bh = max(1.0, by2 - by1)
    norm = float(max(1.0, np.hypot(0.5 * (aw + bw), 0.5 * (ah + bh)) * 1.5))
    return float(max(0.0, min(1.0, 1.0 - (dist / norm))))


def _clone_state_entry(src: Optional[Dict]) -> Dict:
    """Clone per-track state so ID-switch repair never aliases two track IDs to the same dict."""
    if not isinstance(src, dict):
        return {}
    out: Dict = {}
    for k, v in src.items():
        if isinstance(v, np.ndarray):
            out[k] = np.array(v, copy=True)
        elif isinstance(v, dict):
            out[k] = dict(v)
        elif isinstance(v, list):
            out[k] = list(v)
        elif isinstance(v, tuple):
            out[k] = tuple(v)
        elif isinstance(v, deque):
            out[k] = deque(v, maxlen=v.maxlen)
        else:
            out[k] = v
    return out


class VideoStream:
    """
    Adaptive FIFO reader for RTSP/HTTP/file sources, with auto-reconnect.

    Behavior:
      - Normal case (pipeline keeps up): frames are processed in order (FIFO).
      - If processing falls behind: the internal buffer fills and we start *dropping the oldest frames*
        (skipping) so the pipeline can catch up and latency doesn't grow without bound.
      - If the stream stalls (network/pipeline freeze) or read() errors persist: we reopen the source
        in-process so you don't have to restart the script.
    """

    def __init__(
        self,
        src: str,
        rtsp_buffer: int = 2,
        queue_size: int = 64,
        max_queue_age_ms: int = 0,
        grab_skip: int = 0,
        stall_seconds: float = 8.0,
        reconnect_backoff_seconds: float = 2.0,
        open_timeout_ms: int = 5000,
        read_timeout_ms: int = 5000,
        max_reconnect_tries: int = 0,
    ):
        self.src = str(src)
        self._rtsp_buffer = max(1, int(rtsp_buffer))
        self._grab_skip = max(0, int(grab_skip))

        # Auto-reconnect knobs
        self._stall_seconds = float(max(0.0, float(stall_seconds)))
        self._reconnect_backoff_seconds = float(max(0.0, float(reconnect_backoff_seconds)))
        self._open_timeout_ms = max(0, int(open_timeout_ms))
        self._read_timeout_ms = max(0, int(read_timeout_ms))
        self._max_reconnect_tries = max(0, int(max_reconnect_tries))  # 0 => infinite

        qsz = int(queue_size)
        self._maxlen = None if qsz <= 0 else max(1, qsz)
        self._buf = deque(maxlen=self._maxlen)  # stores (timestamp, frame)

        self._cond = threading.Condition()
        self._stop = threading.Event()

        self._dropped_full = 0
        self._dropped_age = 0
        self._written = 0
        self._read = 0

        self._max_queue_age_s = 0.0 if int(max_queue_age_ms) <= 0 else float(max_queue_age_ms) / 1000.0

        # Reconnect state
        self._restart_requested = threading.Event()
        self._restart_count = 0
        self._last_ok_mono = time.monotonic()
        self._last_reconnect_mono = 0.0
        self._reconnect_tries = 0
        self._fail_streak = 0

        self._cap_lock = threading.Lock()
        self.cap = self._open_capture()
        self.ok = bool(self.cap is not None and self.cap.isOpened())
        if not self.ok:
            print(f"[WARN] cannot open source: {self.src}")

        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _open_capture(self) -> cv2.VideoCapture:
        """(Re)open cv2.VideoCapture with best-effort timeouts."""
        cap = cv2.VideoCapture()

        # For RTSP sources, set FFmpeg options (best-effort) unless the user already configured them.
        # This helps avoid hard freezes where read() blocks forever on network issues.
        src_l = self.src.lower()
        if (src_l.startswith("rtsp://") or src_l.startswith("rtsps://")) and ("OPENCV_FFMPEG_CAPTURE_OPTIONS" not in os.environ):
            # stimeout is in microseconds (FFmpeg). Keep it modest so reconnect can kick in.
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp|stimeout;5000000|max_delay;500000"

        # Best-effort timeouts (supported on some OpenCV builds/backends).
        # These MUST be set *before* open() to have a chance to take effect.
        if self._open_timeout_ms > 0 and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
            try:
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(self._open_timeout_ms))
            except Exception:
                pass
        if self._read_timeout_ms > 0 and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
            try:
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(self._read_timeout_ms))
            except Exception:
                pass

        try:
            cap.open(self.src, cv2.CAP_FFMPEG)
        except Exception:
            # If open() itself raises, keep cap closed.
            pass

        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self._rtsp_buffer))
        except Exception:
            pass

        return cap

    def request_restart(self) -> None:
        """Ask the capture thread to reopen the stream ASAP."""
        self._restart_requested.set()

    def _should_reconnect_now(self) -> bool:
        if self._reconnect_backoff_seconds <= 0.0:
            return True
        return (time.monotonic() - float(self._last_reconnect_mono)) >= float(self._reconnect_backoff_seconds)

    def _reopen(self, reason: str = "") -> None:
        if not self._should_reconnect_now():
            return

        self._last_reconnect_mono = time.monotonic()
        self._reconnect_tries += 1

        if self._max_reconnect_tries > 0 and self._reconnect_tries > self._max_reconnect_tries:
            # Give up for now; another request_restart() (or another stall event) will try again.
            return

        with self._cap_lock:
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
            self.cap = self._open_capture()
            self.ok = bool(self.cap is not None and self.cap.isOpened())

        with self._cond:
            self._buf.clear()
            self._cond.notify_all()

        if self.ok:
            self._restart_count += 1
            self._fail_streak = 0
            self._reconnect_tries = 0  # reset tries on success
            self._last_ok_mono = time.monotonic()
            if reason:
                print(f"[STREAM] Reconnected source: {self.src} (reason={reason})")
            else:
                print(f"[STREAM] Reconnected source: {self.src}")

    def _loop(self):
        while not self._stop.is_set():
            # Handle explicit restart requests
            if self._restart_requested.is_set():
                self._restart_requested.clear()
                self._reopen(reason="requested")

            # If not open, keep attempting to reconnect (backoff applies)
            if not bool(self.ok):
                self._reopen(reason="not_open")
                time.sleep(0.05)
                continue

            # If we're fully buffered, optionally fast-skip some frames without decoding (grab-only)
            # to reduce CPU overhead on high-res RTSP streams.
            if self._grab_skip > 0:
                try:
                    with self._cond:
                        full = (self._maxlen is not None) and (len(self._buf) >= self._maxlen)
                    if full:
                        with self._cap_lock:
                            for _ in range(self._grab_skip):
                                if self._stop.is_set():
                                    break
                                try:
                                    _ = self.cap.grab()
                                except Exception:
                                    break
                except Exception:
                    pass

            # Read one frame
            ok, frame = False, None
            try:
                with self._cap_lock:
                    ok, frame = self.cap.read()
            except Exception:
                ok, frame = False, None

            if not ok or frame is None:
                self._fail_streak += 1

                # Stall detection: if we haven't received a frame for N seconds, reconnect.
                if self._stall_seconds > 0.0 and (time.monotonic() - float(self._last_ok_mono)) >= float(self._stall_seconds):
                    self._reopen(reason="stall")
                # Or if we are continuously failing reads, reconnect sooner.
                elif self._fail_streak >= 60:
                    self._reopen(reason="read_fail")
                time.sleep(0.02)
                continue

            # Success
            self._fail_streak = 0
            self._last_ok_mono = time.monotonic()

            ts = time.time()
            with self._cond:
                # Drop-by-age (optional): keep latency bounded even before buffer is full.
                if self._max_queue_age_s > 0.0:
                    cutoff = ts - self._max_queue_age_s
                    while self._buf and self._buf[0][0] < cutoff:
                        self._buf.popleft()
                        self._dropped_age += 1

                # Drop-by-full: if bounded and full, discard oldest.
                if self._maxlen is not None and len(self._buf) >= self._maxlen:
                    try:
                        self._buf.popleft()
                        self._dropped_full += 1
                    except Exception:
                        pass

                self._buf.append((ts, frame))
                self._written += 1
                self._cond.notify()

    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray], Optional[float]]:
        """Pop the oldest frame (FIFO). Returns (ok, frame, timestamp)."""
        deadline = time.time() + float(max(0.0, timeout))
        with self._cond:
            while not self._stop.is_set():
                if self._buf:
                    ts, frame = self._buf.popleft()
                    self._read += 1
                    return True, frame, float(ts)

                remaining = deadline - time.time()
                if remaining <= 0:
                    break
                self._cond.wait(timeout=remaining)

        return False, None, None

    def get_stats(self) -> Dict[str, float]:
        with self._cond:
            qlen = float(len(self._buf))
            dropped_full = float(self._dropped_full)
            dropped_age = float(self._dropped_age)
            written = float(self._written)
            read = float(self._read)
        stall_s = float(max(0.0, time.monotonic() - float(self._last_ok_mono)))
        return {
            "ok": 1.0 if bool(self.ok) else 0.0,
            "stall_s": stall_s,
            "restarts": float(self._restart_count),
            "qlen": qlen,
            "dropped_full": dropped_full,
            "dropped_age": dropped_age,
            "dropped_total": dropped_full + dropped_age,
            "written": written,
            "read": read,
        }

    def release(self):
        self._stop.set()
        try:
            self.thread.join(timeout=1.0)
        except Exception:
            pass
        with self._cap_lock:
            try:
                if self.cap is not None:
                    self.cap.release()
            except Exception:
                pass
        with self._cond:
            self._buf.clear()

@dataclass
class PersonEntry:
    user_id: int
    name: str
    body_centroid: Optional[np.ndarray]
    face_centroid: Optional[np.ndarray]

    # NEW: back-body (no-face) centroid
    back_body_centroid: Optional[np.ndarray]

    body_bank: Optional[np.ndarray]  # shape [N,512], normalized
    face_bank: Optional[np.ndarray]  # shape [N,512], normalized

    # NEW: back-body raw bank
    back_body_bank: Optional[np.ndarray]  # shape [N,512], normalized


class TwoStageGallery:
    def __init__(self):
        self.people: List[PersonEntry] = []
        # modality -> (centroid_matrix [M,512], person_indices [M])
        self._centroid_cache: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}

    def labels(self) -> List[str]:
        return [p.name for p in self.people]

    def prepare(self) -> None:
        """Build fast centroid lookup matrices for each modality."""
        cache: Dict[str, Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = {}
        for modality in ("body", "face", "back_body"):
            mats: List[np.ndarray] = []
            idxs: List[int] = []
            for i, p in enumerate(self.people):
                if modality == "body":
                    c = p.body_centroid
                elif modality == "face":
                    c = p.face_centroid
                else:
                    c = p.back_body_centroid

                if c is None:
                    continue
                mats.append(np.asarray(c, dtype=np.float32).reshape(-1))
                idxs.append(int(i))

            if mats:
                cache[modality] = (l2_normalize_rows(np.stack(mats, axis=0)), np.asarray(idxs, dtype=np.int32))
            else:
                cache[modality] = (None, None)

        self._centroid_cache = cache

    def centroid_cache(self, modality: str) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        key = str(modality).lower().strip()
        if key in ("back", "back_body"):
            key = "back_body"
        return self._centroid_cache.get(key, (None, None))



def decode_bank_gzip_npy(raw: Optional[bytes]) -> Optional[np.ndarray]:
    if not raw:
        return None
    try:
        with gzip.GzipFile(fileobj=BytesIO(raw), mode="rb") as gz:
            data = gz.read()
        arr = np.load(BytesIO(data), allow_pickle=False)
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[1] != EXPECTED_DIM:
            return None
        if not np.isfinite(arr).all():
            return None
        return l2_normalize_rows(arr)
    except Exception:
        return None


def _as_vec512(x) -> Optional[np.ndarray]:
    if x is None:
        return None
    try:
        a = np.asarray(x, dtype=np.float32).reshape(-1)
        if a.size != EXPECTED_DIM:
            return None
        if not np.isfinite(a).all():
            return None
        return l2_normalize(a)
    except Exception:
        return None


def build_gallery_from_db(db_url: str, active_only: bool = True) -> TwoStageGallery:
    """Load the recognition gallery from the NEW DB schema.

    New schema (as provided):
      - members
      - member_embeddings  (multiple rows per member, typically per camera)

    This loader merges embeddings across cameras per member so matching works
    even if you don't pass any camera mapping to the script.

    For each member:
      - centroid per modality = mean(L2-normalized centroids across rows)  -> L2-normalized
      - bank per modality     = concatenation of all decoded raw banks (if present)
    """
    Base = declarative_base()

    class Member(Base):
        __tablename__ = "members"
        id = Column(Integer, primary_key=True)
        member_number = Column(String(16))
        first_name = Column(String(64))
        last_name = Column(String(64))
        is_active = Column(Boolean)

    class MemberEmbedding(Base):
        __tablename__ = "member_embeddings"
        id = Column(BigInteger, primary_key=True)

        member_id = Column(Integer, ForeignKey("members.id", ondelete="CASCADE"), nullable=False)
        # Keep camera_id as plain int (we don't need to define Camera here)
        camera_id = Column(Integer, nullable=False)

        if Vector is not None:
            body_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
            face_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
            back_body_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
        else:
            body_embedding = Column(ARRAY(Float), nullable=True)
            face_embedding = Column(ARRAY(Float), nullable=True)
            back_body_embedding = Column(ARRAY(Float), nullable=True)

        body_embeddings_raw = Column(LargeBinary, nullable=True)
        face_embeddings_raw = Column(LargeBinary, nullable=True)
        back_body_embeddings_raw = Column(LargeBinary, nullable=True)

        last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True)

    engine = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)

    gallery = TwoStageGallery()

    def _member_display_name(member_number: str, first: str, last: str) -> str:
        first = str(first or "").strip()
        last = str(last or "").strip()
        mn = str(member_number or "").strip()
        base = (first + " " + last).strip()
        if base:
            return base
        return mn or "member"

    # Aggregate across all embedding rows per member_id
    agg: Dict[int, Dict[str, object]] = {}

    with Session() as session:
        stmt = (
            select(
                Member.id.label("member_id"),
                Member.member_number,
                Member.first_name,
                Member.last_name,
                Member.is_active,
                MemberEmbedding.camera_id,
                MemberEmbedding.body_embedding,
                MemberEmbedding.face_embedding,
                MemberEmbedding.back_body_embedding,
                MemberEmbedding.body_embeddings_raw,
                MemberEmbedding.face_embeddings_raw,
                MemberEmbedding.back_body_embeddings_raw,
            )
            .select_from(Member)
            .join(MemberEmbedding, MemberEmbedding.member_id == Member.id)
        )

        if active_only:
            stmt = stmt.where(Member.is_active.is_(True))

        rows = session.execute(stmt).all()

        for r in rows:
            mid = int(r.member_id)

            ent = agg.get(mid)
            if ent is None:
                ent = {
                    "name": _member_display_name(r.member_number, r.first_name, r.last_name),
                    "body_vecs": [],
                    "face_vecs": [],
                    "back_vecs": [],
                    "body_banks": [],
                    "face_banks": [],
                    "back_banks": [],
                }
                agg[mid] = ent

            bv = _as_vec512(r.body_embedding)
            fv = _as_vec512(r.face_embedding)
            kv = _as_vec512(r.back_body_embedding)

            if bv is not None:
                ent["body_vecs"].append(bv)
            if fv is not None:
                ent["face_vecs"].append(fv)
            if kv is not None:
                ent["back_vecs"].append(kv)

            bb = decode_bank_gzip_npy(r.body_embeddings_raw)
            fb = decode_bank_gzip_npy(r.face_embeddings_raw)
            kb = decode_bank_gzip_npy(r.back_body_embeddings_raw)

            if bb is not None and bb.size > 0:
                ent["body_banks"].append(bb)
            if fb is not None and fb.size > 0:
                ent["face_banks"].append(fb)
            if kb is not None and kb.size > 0:
                ent["back_banks"].append(kb)

    def _merge_bank(banks: List[np.ndarray]) -> Optional[np.ndarray]:
        if not banks:
            return None
        try:
            return l2_normalize_rows(np.concatenate(banks, axis=0))
        except Exception:
            try:
                return l2_normalize_rows(np.vstack(banks))
            except Exception:
                return None

    def _centroid_from_vecs(vecs: List[np.ndarray]) -> Optional[np.ndarray]:
        if not vecs:
            return None
        try:
            m = np.stack([l2_normalize(np.asarray(v, dtype=np.float32).reshape(-1)) for v in vecs], axis=0)
            return l2_normalize(np.mean(m, axis=0))
        except Exception:
            return None

    for mid, ent in agg.items():
        body_bank = _merge_bank(ent.get("body_banks", []))
        face_bank = _merge_bank(ent.get("face_banks", []))
        back_bank = _merge_bank(ent.get("back_banks", []))

        body_cent = _centroid_from_vecs(ent.get("body_vecs", []))
        face_cent = _centroid_from_vecs(ent.get("face_vecs", []))
        back_cent = _centroid_from_vecs(ent.get("back_vecs", []))

        # Fallback: centroid from bank
        if body_cent is None and body_bank is not None and len(body_bank) > 0:
            body_cent = l2_normalize(np.mean(body_bank, axis=0))
        if face_cent is None and face_bank is not None and len(face_bank) > 0:
            face_cent = l2_normalize(np.mean(face_bank, axis=0))
        if back_cent is None and back_bank is not None and len(back_bank) > 0:
            back_cent = l2_normalize(np.mean(back_bank, axis=0))

        if body_cent is None and face_cent is None and back_cent is None:
            continue

        gallery.people.append(
            PersonEntry(
                user_id=int(mid),
                name=str(ent.get("name", "member")),
                body_centroid=body_cent,
                face_centroid=face_cent,
                back_body_centroid=back_cent,
                body_bank=body_bank,
                face_bank=face_bank,
                back_body_bank=back_bank,
            )
        )

    gallery.prepare()
    print(f"[DB] Loaded identities: {len(gallery.people)}")
    return gallery

def best_face_label_top2(emb: Optional[np.ndarray], gallery: TwoStageGallery) -> Tuple[Optional[str], float, float]:
    """Face identification ONLY (centroid-only, top-2 gap).

    Returns: (best_label, best_sim, second_sim)

    - Uses ONLY face centroids from the DB (no bank refinement, no multi-stage matching).
    - This mirrors the methodology in updated_frameid_faceonly_knownonly.py.
    """
    if emb is None or gallery is None or not getattr(gallery, "people", None):
        return None, 0.0, 0.0

    q = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
    if q.size != EXPECTED_DIM or not np.isfinite(q).all():
        return None, 0.0, 0.0

    centroids, idx_map = gallery.centroid_cache("face")
    if centroids is None or idx_map is None or len(idx_map) == 0:
        return None, 0.0, 0.0

    sims = centroids @ q  # [M]
    if sims.size == 0:
        return None, 0.0, 0.0

    if sims.size == 1:
        p = gallery.people[int(idx_map[0])]
        return str(p.name), float(sims[0]), 0.0

    # top2 without full sort
    idxs = np.argpartition(sims, -2)[-2:]
    i1, i2 = int(idxs[0]), int(idxs[1])
    if sims[i2] > sims[i1]:
        i1, i2 = i2, i1
    best_mi, second_mi = i1, i2

    best_person = gallery.people[int(idx_map[int(best_mi)])]
    return str(best_person.name), float(sims[best_mi]), float(sims[second_mi])


def _best_label_from_bank_or_centroid(
    emb: Optional[np.ndarray],
    people: List[PersonEntry],
    bank_attr: str,
    centroid_attr: str,
    topk: int = 3,
) -> Tuple[Optional[str], float, float]:
    """Support-only matcher (TopK-mean cosine over each person's bank).

    - If a person's bank is missing/empty, falls back to centroid cosine (if available).
    - Returns (best_label, best_score, second_score)
    """
    if emb is None or not people:
        return None, 0.0, 0.0

    q = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
    if q.size != EXPECTED_DIM or not np.isfinite(q).all():
        return None, 0.0, 0.0

    k_req = max(1, int(topk))
    scored: List[Tuple[str, float]] = []

    for p in people:
        score = None

        bank = getattr(p, bank_attr, None)
        if bank is not None:
            try:
                bank = np.asarray(bank, dtype=np.float32)
            except Exception:
                bank = None

        if bank is not None and bank.size > 0 and bank.ndim == 2 and bank.shape[1] == EXPECTED_DIM:
            try:
                sims = bank @ q
                if sims.ndim == 1 and sims.size > 0:
                    k = min(k_req, int(sims.size))
                    if k <= 1:
                        score = float(np.max(sims))
                    else:
                        top_vals = np.partition(sims, -k)[-k:]
                        score = float(np.mean(top_vals))
            except Exception:
                score = None

        if score is None:
            c = getattr(p, centroid_attr, None)
            if c is not None:
                try:
                    c = np.asarray(c, dtype=np.float32).reshape(-1)
                    if c.size == EXPECTED_DIM and np.isfinite(c).all():
                        score = float(np.dot(l2_normalize(c), q))
                except Exception:
                    score = None

        if score is None:
            continue

        scored.append((str(p.name), float(score)))

    if not scored:
        return None, 0.0, 0.0

    scored.sort(key=lambda x: x[1], reverse=True)
    best_label, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    return str(best_label), float(best_score), float(second_score)


def best_body_label_from_emb(emb: Optional[np.ndarray], gallery: TwoStageGallery, topk: int = 3) -> Tuple[Optional[str], float, float]:
    """Body matcher used ONLY as a continuity cue (never to create/switch identities)."""
    return _best_label_from_bank_or_centroid(
        emb=emb,
        people=gallery.people,
        bank_attr="body_bank",
        centroid_attr="body_centroid",
        topk=topk,
    )


def best_back_label_from_emb(emb: Optional[np.ndarray], gallery: TwoStageGallery, topk: int = 3) -> Tuple[Optional[str], float, float]:
    """Back-body matcher used ONLY as a continuity cue (never to create/switch identities)."""
    return _best_label_from_bank_or_centroid(
        emb=emb,
        people=gallery.people,
        bank_attr="back_body_bank",
        centroid_attr="back_body_centroid",
        topk=topk,
    )


def init_face_engine(

    use_face: bool,
    device: str,
    face_model: str,
    det_w: int,
    det_h: int,
    face_provider: str,
    face_detector_model: str,
    face_detector_split: bool,
) -> Optional["FaceAnalysis"]:
    if not use_face:
        return None
    if not INSIGHT_OK:
        print("[WARN] insightface not installed; face disabled.")
        return None

    is_cuda = ("cuda" in device.lower()) and torch.cuda.is_available()
    cuda_ok = _cuda_ep_loadable()

    providers = ["CPUExecutionProvider"]
    if face_provider == "cuda" and cuda_ok:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    elif face_provider == "auto" and is_cuda and cuda_ok:
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

    app = FaceAnalysis(name=face_model, providers=providers)
    detector_desc = f"pack:{face_model}"
    if bool(face_detector_split):
        det_model, det_query = _load_split_face_detector(str(face_detector_model or "").strip(), providers=providers)
        if det_model is not None:
            try:
                app.models["detection"] = det_model
            except Exception:
                pass
            try:
                app.det_model = det_model
            except Exception:
                pass
            detector_desc = str(det_query or face_detector_model or "split-detector")
            print(f"[INIT] Using split face detector={detector_desc} with recognizer pack={face_model}")
        elif str(face_detector_model or "").strip():
            print(
                f"[WARN] requested split face detector '{face_detector_model}' not found; "
                f"falling back to detector bundled with {face_model}."
            )

    ctx_id = 0 if providers[0].startswith("CUDA") else -1
    try:
        app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
    except TypeError:
        app.prepare(ctx_id=ctx_id)

    try:
        setattr(app, "split_detector_active", bool(face_detector_split and not str(detector_desc).startswith("pack:")))
        setattr(app, "detector_name", str(detector_desc))
        setattr(app, "recognizer_name", str(face_model))
    except Exception:
        pass

    print(f"[INIT] InsightFace ready recognizer={face_model} detector={detector_desc} providers={providers}")
    return app


def _resolve_existing_path(path_like: str) -> Path:
    p = Path(str(path_like or '').strip())
    if p.exists():
        return p
    base = Path(__file__).resolve().parent
    for cand in (base / p, base / 'weights' / p):
        if cand.exists():
            return cand
    return p


def _candidate_model_queries(path_like: str) -> List[str]:
    raw = str(path_like or "").strip()
    if not raw:
        return []

    out: List[str] = []
    seen = set()

    def _push(v: str) -> None:
        s = str(v or "").strip()
        if not s or s in seen:
            return
        seen.add(s)
        out.append(s)

    _push(raw)

    rp = _resolve_existing_path(raw)
    try:
        if rp.exists():
            _push(str(rp))
    except Exception:
        pass

    if not raw.lower().endswith(".onnx"):
        raw_onnx = f"{raw}.onnx"
        _push(raw_onnx)
        rp_onnx = _resolve_existing_path(raw_onnx)
        try:
            if rp_onnx.exists():
                _push(str(rp_onnx))
        except Exception:
            pass

    return out


def _load_split_face_detector(detector_model: str, providers: List[str]):
    """Load a standalone InsightFace detector model (e.g. RetinaFace) if available.

    The caller can pass either:
      - a model directory/name already available under ~/.insightface/models
      - a direct .onnx path
      - a relative path resolved from the script folder / weights folder
    """
    if not INSIGHT_OK or insightface is None:
        return None, None

    model_zoo = getattr(insightface, "model_zoo", None)
    if model_zoo is None or not hasattr(model_zoo, "get_model"):
        return None, None

    last_err = None
    for query in _candidate_model_queries(detector_model):
        try:
            model = model_zoo.get_model(query, providers=providers)
        except TypeError:
            try:
                model = model_zoo.get_model(query)
            except Exception as e:
                last_err = e
                continue
        except Exception as e:
            last_err = e
            continue

        if model is None:
            continue

        if hasattr(model, "detect") and hasattr(model, "prepare"):
            return model, query

    if last_err is not None:
        print(f"[WARN] Split face detector load failed for {detector_model}: {last_err}")
    return None, None


def create_strongsort_tracker(args):
    use_ss = bool(getattr(args, 'use_strongsort', False) or getattr(args, 'use_deepsort', False))
    if not use_ss:
        return None

    if BoxMOTStrongSort is None:
        print('[WARN] boxmot StrongSORT not available; using fallback IoU tracker.')
        return None

    gpu = torch.cuda.is_available() and ('cuda' in str(getattr(args, 'device', '')).lower())
    device_str = str(getattr(args, 'device', 'cuda:0')) if gpu else 'cpu'
    device = torch.device(device_str)
    half = bool(gpu and bool(getattr(args, 'half', False)))

    weights_arg = str(getattr(args, 'strongsort_reid_weights', '') or '').strip()
    if not weights_arg:
        weights_arg = 'osnet_x0_25_msmt17.pt'

    weights_path = _resolve_existing_path(weights_arg)
    if not weights_path.exists():
        print(f'[WARN] StrongSORT ReID weights not found: {weights_path}. Using fallback IoU tracker.')
        return None

    det_thresh = float(getattr(args, 'conf', 0.30) or 0.30)
    min_conf = max(0.01, det_thresh * 0.5)
    iou_threshold = 0.30

    primary_kwargs = dict(
        reid_weights=weights_path,
        device=device,
        half=half,
        det_thresh=det_thresh,
        max_age=int(getattr(args, 'max_age', 25) or 25),
        min_hits=int(getattr(args, 'n_init', 3) or 3),
        iou_threshold=iou_threshold,
        min_conf=min_conf,
        max_cos_dist=float(getattr(args, 'tracker_max_cosine', 0.4) or 0.4),
        nn_budget=int(getattr(args, 'nn_budget', 200) or 200),
        n_init=int(getattr(args, 'n_init', 3) or 3),
    )

    init_err = None
    for kwargs in (
        primary_kwargs,
        {
            **{k: v for k, v in primary_kwargs.items() if k != 'iou_threshold'},
            'max_iou_dist': 0.7,
        },
    ):
        try:
            tracker = BoxMOTStrongSort(**kwargs)
            print(f'[INIT] StrongSORT ready weights={weights_path} device={device_str}')
            return tracker
        except TypeError as e:
            init_err = e
            continue
        except Exception as e:
            print('[WARN] StrongSORT init failed, fallback to IoU tracker:', e)
            return None

    if init_err is not None:
        print('[WARN] StrongSORT init failed, fallback to IoU tracker:', init_err)
    return None


def init_models(args):
    gpu = torch.cuda.is_available() and ("cuda" in args.device.lower())
    if args.half and not gpu:
        print("[WARN] --half requested but CUDA not available; disabling FP16.")
        args.half = False

    yolo = None
    if YOLO is not None:
        weights = args.yolo_weights
        if not os.path.exists(weights):
            print(f"[INIT] {weights} not found -> fallback yolov8n.pt")
            weights = "yolov8n.pt"
        yolo = YOLO(weights)
        if gpu:
            try:
                yolo.to(args.device)
            except Exception:
                pass
        print("[INIT] YOLO ready")
    else:
        print("[WARN] ultralytics not installed; YOLO disabled")

    # TorchReID ensemble
    reid_extractors: List[TorchreidExtractor] = []
    if TorchreidExtractor is not None:
        dev = args.device if gpu else "cpu"
        model_list: List[str] = []
        if args.reid_models:
            # comma-separated string
            model_list = [m.strip() for m in str(args.reid_models).split(",") if m.strip()]
        if not model_list:
            model_list = [str(args.reid_model).strip()]
        for m in model_list:
            try:
                ext = TorchreidExtractor(model_name=m, device=dev)
                reid_extractors.append(ext)
                print(f"[INIT] TorchReID ready model={m} device={dev}")
            except Exception as e:
                print(f"[WARN] TorchReID init failed for {m}: {e}")
    else:
        print("[WARN] torchreid not installed; body reid disabled")

    face_app = init_face_engine(
        args.use_face,
        args.device,
        args.face_model,
        args.face_det_size[0],
        args.face_det_size[1],
        face_provider=args.face_provider,
        face_detector_model=args.face_detector_model,
        face_detector_split=bool(getattr(args, "face_detector_split", True)),
    )

    deep = None
    use_ss = bool(getattr(args, "use_strongsort", False) or getattr(args, "use_deepsort", False))
    if use_ss:
        if BoxMOTStrongSort is None:
            print("[WARN] boxmot StrongSORT not installed; tracking will fall back to IoU.")
        else:
            print("[INIT] StrongSORT selected (tracker will be initialized per source).")
    else:
        print("[INIT] Using fallback IoU tracker (stable IDs without StrongSORT).")

    return yolo, reid_extractors, face_app, deep


def parse_args(argv: Optional[List[str]] = None, allow_unknown: bool = False):
    ap = argparse.ArgumentParser(
        "Face-only identification (DB face centroid) + body/back continuity support + spatio-temporal room analytics"
    )

    # Video / UI
    ap.add_argument("--src", nargs="+", required=True, help="Video sources (RTSP/HTTP/file).")
    ap.add_argument(
        "--camera-ids",
        nargs="+",
        type=int,
        default=[],
        help="Optional external camera ids aligned with --src order. If omitted, defaults to 1..N (service-compatible).",
    )
    ap.add_argument("--show", action="store_true", help="Show window (press q to quit).")
    ap.add_argument("--rtsp-buffer", type=int, default=2)

    # Stream robustness / auto-reconnect (handles RTSP/network/pipeline freezes)
    ap.add_argument(
        "--stream-stall-seconds",
        type=float,
        default=8.0,
        help="If no frames are received for this many seconds, attempt to auto-restart the stream (0=disable).",
    )
    ap.add_argument(
        "--stream-reconnect-backoff",
        type=float,
        default=2.0,
        help="Minimum seconds between reconnect attempts per source.",
    )
    ap.add_argument(
        "--stream-open-timeout-ms",
        type=int,
        default=5000,
        help="Best-effort: OpenCV open timeout (ms) when supported by the backend (0=disable).",
    )
    ap.add_argument(
        "--stream-read-timeout-ms",
        type=int,
        default=5000,
        help="Best-effort: OpenCV read timeout (ms) when supported by the backend (0=disable).",
    )
    ap.add_argument(
        "--stream-reconnect-max-tries",
        type=int,
        default=0,
        help="Max reconnect tries per stall event (0=infinite).",
    )


    # Room mapping (order must match --src). If omitted, defaults to c1..cN.
    ap.add_argument(
        "--room-ids",
        nargs="+",
        default=[],
        help="Room IDs aligned with --src order (e.g., --room-ids c1 c2 c3 c4).",
    )

    ap.add_argument(
        "--room-graph",
        default="c1-c2,c2-c5,c5-c4,c2-c3,c3-c4",
        help="Room adjacency graph for spatio-temporal logic (comma-separated edges like c1-c3,c3-c4).",
    )
    ap.add_argument(
        "--room-entry-id",
        default="c3",
        help="Entry/corridor room id used for --excel-entry-impute-seconds (default: c3).",
    )
    ap.add_argument(
        "--excel-disable-path-impute",
        action="store_true",
        help="Disable intermediate-room imputation on non-adjacent transitions (LOGICAL).",
    )

    # Excel export (auto-written on exit; optional periodic export)
    ap.add_argument(
        "--excel-out",
        default="room_presence.xlsx",
        help="Output Excel path (default: room_presence.xlsx). Use 'off' to disable.",
    )
    ap.add_argument("--excel-with-date", action="store_true", help="Include date in Summary interval strings.")
    ap.add_argument("--excel-tz", default="Asia/Kolkata", help="Timezone used in Excel formatting.")
    ap.add_argument("--excel-raw-merge-gap", type=float, default=2.0, help="Seconds: merge adjacent detections for RAW segments.")
    ap.add_argument("--excel-fill-gap", type=float, default=90.0, help="Seconds: fill flicker gaps in SAME room for LOGICAL segments.")
    ap.add_argument(
        "--excel-adjacent-gap-policy",
        choices=["prev", "next", "split"],
        default="split",
        help="How to allocate time between adjacent-room detections (prev|next|split).",
    )
    ap.add_argument(
        "--excel-entry-impute-seconds",
        type=float,
        default=0.0,
        help="Optional: if >0, add an inferred entry segment in --room-entry-id (default: c3) when first seen elsewhere.",
    )
    ap.add_argument(
        "--excel-export-every-seconds",
        type=float,
        default=300.0,
        help="If >0, periodically write Excel while running (in addition to on-exit). Default is 300s (5 minutes).",
    )

    # FPS limiting (stability): keeps processing + display around a fixed FPS to avoid spikes/jitter.
    ap.add_argument(
        "--proc-fps",
        type=float,
        default=8.0,
        help="Limit per-source processing FPS (0=unlimited). Recommended 7-9 for stability.",
    )
    ap.add_argument(
        "--disp-fps",
        type=float,
        default=8.0,
        help="Limit display refresh FPS when --show (0=unlimited). Recommended 7-9 for stability.",
    )

    ap.add_argument(
        "--win-max-wh",
        type=int,
        nargs=2,
        default=[1280, 720],
        help="Max display window size W H when --show. Output will be scaled down to fit. 0 0 = no scaling.",
    )
    ap.add_argument(
        "--display-scale",
        type=float,
        default=0.0,
        help="Optional fixed display scale factor (e.g., 0.75). 0=auto fit to --win-max-wh.",
    )

    # Buffering / frame skipping behavior:
    # - If queue-size is large and processing can't keep up, you'll see high latency.
    # - If queue-size is bounded, we drop oldest frames when full (skip frames) to catch up.
    ap.add_argument(
        "--queue-size",
        type=int,
        default=64,
        help="Per-source buffer size in frames. When full, oldest frames are dropped (skip frames) to catch up. 0=unbounded (NOT recommended for RTSP).",
    )
    ap.add_argument(
        "--max-queue-age-ms",
        type=int,
        default=0,
        help="Optional: drop frames older than this age (ms) before processing to cap latency (0=disabled).",
    )
    ap.add_argument(
        "--grab-skip",
        type=int,
        default=0,
        help="If queue is full, do this many cap.grab() calls before cap.read() (skips decode of some frames). 0=disabled.",
    )

    ap.add_argument("--resize", type=int, nargs=2, default=[0, 0], help="Force resize W H (0 0 = keep).")

    # YOLO
    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--yolo-imgsz", type=int, default=640, help="YOLO inference imgsz (lower can be faster).")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--iou", type=float, default=0.40)
    ap.add_argument("--min-box-wh", type=int, default=40)

    # Tracking
    ap.add_argument("--use-strongsort", action="store_true", help="Enable StrongSORT tracking.")
    ap.add_argument(
        "--strongsort-reid-weights",
        default="osnet_x0_25_msmt17.pt",
        help="Path to StrongSORT ReID weights (.pt). Relative paths are also resolved from ./weights/.",
    )
    ap.add_argument("--use-deepsort", action="store_true", help="Deprecated alias for --use-strongsort.")
    ap.add_argument("--max-age", type=int, default=25)
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--nn-budget", type=int, default=200)
    ap.add_argument("--tracker-max-cosine", type=float, default=0.4)
    ap.add_argument("--tracker-nms-overlap", type=float, default=1.0)

    # Embedders
    ap.add_argument("--reid-model", default="osnet_x0_25", help="Single TorchReID model (legacy).")
    ap.add_argument(
        "--reid-models",
        default="osnet_x1_0,osnet_x0_25",
        help="Comma-separated TorchReID models to ensemble (must match DB extraction).",
    )

    # RTX-laptop VRAM protection: chunked ReID batches
    ap.add_argument(
        "--reid-batch-size",
        type=int,
        default=24,
        help="Max crops per TorchReID forward pass (chunked). Lower this if you hit CUDA OOM on 8GB GPUs.",
    )

    # Tracklet / identification cadence
    ap.add_argument(
        "--reid-every-n",
        type=int,
        default=2,
        help="Compute/update body & back embeddings every N frames per track while NOT yet identified (tracklet mode).",
    )
    ap.add_argument(
        "--reid-every-n-known",
        type=int,
        default=6,
        help="Compute/update body & back embeddings every N frames per track AFTER identification (saves GPU).",
    )
    ap.add_argument(
        "--tracklet-ema-beta",
        type=float,
        default=0.85,
        help="EMA smoothing beta for tracklet embeddings (0..1). Higher = more smoothing / slower to change.",
    )
    ap.add_argument(
        "--tracklet-min-samples",
        type=int,
        default=1,
        help="Minimum number of embeddings gathered for a modality before matching/voting (per track).",
    )
    ap.add_argument(
        "--tracklet-prune-after",
        type=int,
        default=60,
        help="Prune per-track feature/identity state if not seen for this many frames.",
    )

    # Face
    ap.add_argument("--use-face", action="store_true")
    ap.add_argument("--face-model", default="buffalo_l")
    ap.add_argument(
        "--face-detector-split",
        dest="face_detector_split",
        action="store_true",
        default=True,
        help="(default) Use a separate detector model for face detection while keeping --face-model for recognition.",
    )
    ap.add_argument(
        "--no-face-detector-split",
        dest="face_detector_split",
        action="store_false",
        help="Use the detector bundled inside --face-model instead of a separate detector.",
    )
    ap.add_argument(
        "--face-detector-model",
        default="retinaface_r50_v1",
        help="Standalone detector model/name/path used when --face-detector-split is enabled (for example a RetinaFace ONNX or model dir).",
    )
    ap.add_argument("--face-det-size", type=int, nargs=2, default=[640, 640])
    ap.add_argument("--face-provider", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--face-every-n", type=int, default=2)

    # NOTE:
    # We link face detections to person tracks using IoA(face, person) instead of IoU.
    ap.add_argument(
        "--face-iou-link",
        type=float,
        default=0.35,
        help="IoA(face, person) link threshold (despite name).",
    )

    # DB
    ap.add_argument("--use-db", action="store_true")
    ap.add_argument("--db-only", action="store_true", help="Only use DB gallery (ignore folder galleries).")
    ap.add_argument("--db-url", default="", help="SQLAlchemy URL (postgresql://...).")
    ap.add_argument("--db-refresh-seconds", type=float, default=1.0, help="Reload DB gallery every N seconds (0=off).")

    # Matching knobs (face-only ID; body/back support)
    ap.add_argument("--topn", type=int, default=8, help="LEGACY (ignored in face-only mode).")
    ap.add_argument("--body-topk", type=int, default=3, help="Bank refinement topK mean (body).")
    ap.add_argument("--face-topk", type=int, default=3, help="LEGACY (ignored in face-only mode).")
    ap.add_argument("--back-topk", type=int, default=3, help="Bank refinement topK mean (back-body).")
    ap.add_argument("--alpha", type=float, default=0.4, help="LEGACY (ignored in face-only mode).")

    ap.add_argument("--body-thresh", type=float, default=0.85)
    ap.add_argument("--body-gap", type=float, default=0.08, help="Gap size for full-confidence vote (body).")

    ap.add_argument("--face-thresh", type=float, default=0.45)
    ap.add_argument("--face-gap", type=float, default=0.05, help="Gap size for full-confidence vote (face).")

    # Back-body matching thresholds
    ap.add_argument("--back-thresh", type=float, default=0.80)
    ap.add_argument("--back-gap", type=float, default=0.08, help="Gap size for full-confidence vote (back-body).")

    # Back-body embedding settings (must match extraction)
    ap.add_argument("--back-head-cut", type=float, default=BACK_HEAD_CUT_RATIO_DEFAULT, help="Head cut ratio for no-face crop.")
    ap.add_argument("--back-shape-seed", type=int, default=BACK_SHAPE_SEED_DEFAULT, help="Random seed for shape projection.")
    ap.add_argument("--back-reid-weight", type=float, default=BACK_REID_WEIGHT_DEFAULT)
    ap.add_argument("--back-shape-weight", type=float, default=BACK_SHAPE_WEIGHT_DEFAULT)

    # Smoothing
    ap.add_argument("--name-decay", type=float, default=0.85)
    ap.add_argument("--name-min-score", type=float, default=0.40)
    ap.add_argument("--name-margin", type=float, default=0.15)
    ap.add_argument("--name-ttl", type=int, default=20)
    ap.add_argument("--name-face-weight", type=float, default=2.0)
    ap.add_argument("--name-body-weight", type=float, default=0.1)
    ap.add_argument("--name-back-weight", type=float, default=0.2)

    # NEW: extra stability / anti-"name jump" logic
    ap.add_argument(
        "--name-hold-seconds",
        type=float,
        default=1.0,
        help="Once a track is identified, keep showing that name for at least this many seconds (prevents flicker). 0=disable.",
    )
    ap.add_argument(
        "--name-reserve-seconds",
        type=float,
        default=1.0,
        help="Reserve a name for this many seconds after it was last seen so it cannot jump to another track right after collisions.",
    )
    ap.add_argument(
        "--name-transfer-iou",
        type=float,
        default=0.30,
        help="To allow a reserved name to transfer to a different track (ID switch), require IoU >= this with the last box of that name.",
    )
    ap.add_argument(
        "--name-transfer-sim",
        type=float,
        default=0.65,
        help="To allow a reserved name to transfer to a different track (ID switch), require embedding cosine similarity >= this (same modality).",
    )
    ap.add_argument(
        "--name-reserve-prune-seconds",
        type=float,
        default=10.0,
        help="Prune name reservations older than this many seconds (keeps memory bounded).",
    )

    # NEW: confirmed-identity persistence (surveillance-style)
    ap.add_argument(
        "--confirm-gap-min",
        type=float,
        default=0.85,
        help="Gap confidence (0..1) required to mark an identity as CONFIRMED (persist without TTL/decay until collision).",
    )
    ap.add_argument(
        "--persist-confirmed-until-collision",
        dest="persist_confirmed_until_collision",
        action="store_true",
        default=True,
        help="(default) Once confirmed by strong FACE evidence, keep identity alive until collision/ambiguity occurs.",
    )
    ap.add_argument(
        "--no-persist-confirmed-until-collision",
        dest="persist_confirmed_until_collision",
        action="store_false",
        help="Disable confirmed-until-collision persistence (revert to TTL/decay only).",
    )
    ap.add_argument(
        "--drop-name-on-collision",
        dest="drop_name_on_collision",
        action="store_true",
        default=True,
        help="(default) When person boxes collide/overlap, immediately hide names on colliding tracks to prevent name-bleed/steal.",
    )
    ap.add_argument(
        "--no-drop-name-on-collision",
        dest="drop_name_on_collision",
        action="store_false",
        help="Keep showing names during collisions (less safe, may allow name-bleed).",
    )

    # Optional: allow duplicates (debug)
    ap.add_argument("--allow-duplicate-names", action="store_true", help="Allow same name on multiple tracks (debug).")

    # Collision / ID-switch robustness (helps avoid name juggling during overlaps)
    ap.add_argument("--collision-iou", type=float, default=0.35, help="If IoU between two person tracks exceeds this, treat as collision/occlusion and freeze identity updates to avoid name juggling.")
    ap.add_argument("--collision-freeze-frames", type=int, default=18, help="Freeze identity updates for this many frames when a track is colliding/occluded.")
    ap.add_argument("--collision-embed-freeze-frames", type=int, default=10, help="When tracks overlap/collide, skip updating body/back/face embeddings for this many frames (reduces mixed-pixel contamination).")
    ap.add_argument("--idswitch-iou", type=float, default=0.30, help="IoU threshold to match current tracks to previous-track states to repair tracker ID switches.")
    ap.add_argument("--idswitch-max-age", type=int, default=3, help="Only consider previous track states seen within this many frames for ID-switch repair.")
    ap.add_argument(
        "--idswitch-cloth-sim",
        type=float,
        default=0.55,
        help="Minimum clothing cosine similarity required before a crossing/post-occlusion state transfer is allowed.",
    )
    ap.add_argument(
        "--idswitch-cloth-margin",
        type=float,
        default=0.08,
        help="Alternate clothing-aware mapping must beat keeping the same tracker ID by at least this much before we remap state.",
    )
    ap.add_argument("--disable-idswitch-fix", action="store_true", help="Disable ID-switch repair (debug).")

    # Collision-aware face identification (allowed during overlaps, but extremely strict to avoid "name stealing")
    ap.add_argument(
        "--keep-confirmed-name-on-collision",
        dest="keep_confirmed_name_on_collision",
        action="store_true",
        default=True,
        help="(default) If a track is CONFIRMED, keep showing its name during collisions and freeze switching (prevents flicker in crowds).",
    )
    ap.add_argument(
        "--no-keep-confirmed-name-on-collision",
        dest="keep_confirmed_name_on_collision",
        action="store_false",
        help="If set, even confirmed names are dropped on collision (most conservative).",
    )
    ap.add_argument(
        "--collision-face-enable",
        dest="collision_face_enable",
        action="store_true",
        default=True,
        help="(default) Allow face-based re-identification during collisions under extremely strict geometry+confidence rules.",
    )
    ap.add_argument(
        "--no-collision-face-enable",
        dest="collision_face_enable",
        action="store_false",
        help="Disable face-based re-identification during collisions (revert to freezing all modalities).",
    )
    ap.add_argument(
        "--collision-face-every-frame",
        dest="collision_face_every_frame",
        action="store_true",
        default=True,
        help="(default) When any tracked people are colliding/occluding, run face detection every frame until the collision ends.",
    )
    ap.add_argument(
        "--no-collision-face-every-frame",
        dest="collision_face_every_frame",
        action="store_false",
        help="Keep using --face-every-n even during collisions.",
    )
    ap.add_argument(
        "--crowd-face-only-labels",
        dest="crowd_face_only_labels",
        action="store_true",
        default=True,
        help="(default) During collisions/occlusions, if a face is visible, label that face directly without mapping it to a body track.",
    )
    ap.add_argument(
        "--no-crowd-face-only-labels",
        dest="crowd_face_only_labels",
        action="store_false",
        help="Keep forcing face->body track mapping even in crowded collision scenes.",
    )
    ap.add_argument(
        "--collision-face-ioa",
        type=float,
        default=0.85,
        help="During collision, require IoA(face, person_box) >= this to trust the face-track link.",
    )
    ap.add_argument(
        "--collision-face-ioa-gap",
        type=float,
        default=0.20,
        help="During collision, require IoA(face, assigned_box) - max_IoA(face, other_box) >= this (unambiguous face-to-track assignment).",
    )
    ap.add_argument(
        "--collision-face-thresh",
        type=float,
        default=0.60,
        help="During collision, require face match score >= this (stricter than --face-thresh).",
    )
    ap.add_argument(
        "--collision-face-gap-conf",
        type=float,
        default=0.95,
        help="During collision, require gap confidence (0..1) >= this (uses --face-gap).",
    )
    ap.add_argument(
        "--collision-face-confirm-frames",
        type=int,
        default=2,
        help="During collision, require this many consecutive strong face hits before assigning a NEW name to an unknown track.",
    )

    if allow_unknown:
        args, unknown = ap.parse_known_args(argv)
        if unknown:
            try:
                print("[WARN] Ignoring unsupported compatibility args:", " ".join(str(x) for x in unknown))
            except Exception:
                print("[WARN] Ignoring unsupported compatibility args.")
    else:
        args = ap.parse_args(argv)
    if bool(getattr(args, "use_deepsort", False)):
        args.use_strongsort = True
    if not getattr(args, "camera_ids", None):
        args.camera_ids = []
    return args



def update_track_identity(
    state: Dict[int, Dict],
    tid: int,
    candidates: List[Tuple[str, float, str]],
    decay: float,
    min_score: float,
    margin: float,
    ttl_reset: int,
    w_face: float,
    w_body: float,
    w_back: float,
    now_ts: float,
    hold_seconds: float,
) -> Tuple[str, float]:
    """
    candidates = [(label, vote_strength, "face"|"body"|"back"), ...]
    vote_strength should already incorporate per-frame confidence (e.g., gap scaling).

    Stability additions:
      - hold_seconds: once a name becomes stable on this track, keep it for at least this many
        seconds (even if subsequent frames are uncertain).
      - During the hold window, we do NOT switch to a different name (prevents "jumps" right after overlaps).

    Confirmed-mode additions (real-world surveillance behavior):
      - If a track has been CONFIRMED by strong FACE evidence,
        we keep its identity alive even when evidence is temporarily weak (e.g., back-only),
        UNTIL ambiguity occurs (collision/overlap resets confirmation outside this function).
      - This is *identity continuity*, not re-identification from back.
    """
    entry = state.setdefault(
        tid,
        {
            "scores": {},
            "last": "",
            "ttl": 0,
            "freeze": 0,
            "hold_until": 0.0,
            # confirmed identity persistence (cleared on collision)
            "confirmed": False,
            "confirmed_name": "",
            "confirmed_src": "",
        },
    )
    scores: Dict[str, float] = entry["scores"]

    cur_last = str(entry.get("last", "") or "")
    confirmed_active = bool(entry.get("confirmed", False)) and bool(cur_last) and (str(entry.get("confirmed_name", "")) == cur_last)

    # If a track is in a collision/occlusion state, we can "freeze" its identity
    # for a few frames to prevent label swapping / juggling.
    #
    # IMPORTANT: we do NOT auto-extend TTL here. That way, if you break confirmation on collision,
    # the name can drop quickly right after the freeze window unless re-confirmed.
    freeze = int(entry.get("freeze", 0) or 0)
    if freeze > 0 and cur_last:
        entry["freeze"] = max(0, freeze - 1)
        if float(hold_seconds) > 0.0:
            entry["hold_until"] = max(float(entry.get("hold_until", 0.0) or 0.0), float(now_ts) + float(hold_seconds))
        return cur_last, float(scores.get(cur_last, 0.0))

    # Decay old scores
    for k in list(scores.keys()):
        scores[k] *= float(decay)
        if scores[k] < 1e-6:
            del scores[k]

    # Apply new votes
    for label, vote, src in candidates:
        if not label:
            continue
        if src in ("face", "face_collision"):
            w = float(w_face)
        elif src == "back":
            w = float(w_back)
        else:
            w = float(w_body)
        scores[label] = scores.get(label, 0.0) + max(0.0, float(vote)) * w

    if scores:
        ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
        top_label, top_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else 0.0
    else:
        top_label, top_score, second_score = "", 0.0, 0.0

    hold_until = float(entry.get("hold_until", 0.0) or 0.0)
    in_hold = (float(hold_seconds) > 0.0) and bool(cur_last) and (float(now_ts) < hold_until)

    accepted = False
    if top_label and (top_score >= float(min_score)):
        if cur_last == top_label:
            accepted = True
        else:
            # Don't switch names while we're still inside the "hold" window.
            if not in_hold and (float(top_score) - float(second_score)) >= float(margin):
                accepted = True

    if accepted:
        entry["last"] = str(top_label)
        entry["ttl"] = int(ttl_reset)

        # If the stable name changes away from the confirmed identity, drop confirmation.
        if bool(entry.get("confirmed", False)) and str(entry.get("confirmed_name", "")) and str(top_label) != str(entry.get("confirmed_name", "")):
            entry["confirmed"] = False
            entry["confirmed_name"] = ""
            entry["confirmed_src"] = ""

        if float(hold_seconds) > 0.0:
            entry["hold_until"] = max(float(entry.get("hold_until", 0.0) or 0.0), float(now_ts) + float(hold_seconds))
    else:
        if confirmed_active:
            # Key behavior: keep identity alive indefinitely until ambiguity resets "confirmed".
            entry["last"] = cur_last
            # Do NOT decrement TTL here.
        elif in_hold:
            # Keep the current name until the hold expires (no flicker).
            entry["last"] = cur_last
        else:
            if int(entry.get("ttl", 0) or 0) > 0:
                entry["ttl"] = int(entry.get("ttl", 0) or 0) - 1
            else:
                entry["last"] = ""

    last = str(entry.get("last", "") or "")
    if not last:
        # No identity => no confirmation.
        entry["confirmed"] = False
        entry["confirmed_name"] = ""
        entry["confirmed_src"] = ""
    return last, float(scores.get(last, 0.0))

def _yolo_forward_safe(yolo, frame, args, gpu: bool):
    with _yolo_lock, torch.inference_mode():
        res = yolo(
            frame,
            imgsz=int(args.yolo_imgsz) if hasattr(args, 'yolo_imgsz') else 640,
            conf=args.conf,
            iou=args.iou,
            verbose=False,
            device=args.device,
            half=args.half,
        )
        return res


def _prep_reid_crop(crop_bgr: np.ndarray) -> np.ndarray:
    """
    Match extract_service preprocessing:
      resize to (128,256) then BGR->RGB
    """
    img = cv2.resize(crop_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
    return _to_rgb(img)


def extract_reid_embedding_ensemble(reid_extractors: List[TorchreidExtractor], crop_bgr: np.ndarray) -> Optional[np.ndarray]:
    if not reid_extractors or crop_bgr is None or crop_bgr.size == 0:
        return None
    try:
        h, w = crop_bgr.shape[:2]
        if h < 32 or w < 16:
            return None
        crop_rgb = _prep_reid_crop(crop_bgr)
        feats: List[np.ndarray] = []
        for ext in reid_extractors:
            try:
                with reid_lock, torch.inference_mode():
                    f = ext([crop_rgb])[0]
            except Exception:
                continue
            f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
            f = np.asarray(f, dtype=np.float32).reshape(-1)
            if f.size != EXPECTED_DIM or not np.isfinite(f).all():
                continue
            feats.append(l2_normalize(f))
        if not feats:
            return None
        fused = l2_normalize(np.mean(np.stack(feats, axis=0), axis=0))
        return fused
    except Exception:
        return None



def _features_to_numpy(feats) -> Optional[np.ndarray]:
    """Convert TorchReID FeatureExtractor outputs to np.ndarray [N,512] float32."""
    if feats is None:
        return None
    try:
        if isinstance(feats, torch.Tensor):
            arr = feats.detach().cpu().numpy()
        elif isinstance(feats, (list, tuple)):
            rows = []
            for f in feats:
                if f is None:
                    continue
                if isinstance(f, torch.Tensor):
                    rows.append(f.detach().cpu().numpy())
                else:
                    rows.append(np.asarray(f))
            if not rows:
                return None
            arr = np.asarray(rows)
        else:
            arr = np.asarray(feats)

        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        return arr
    except Exception:
        return None


def extract_reid_embeddings_batch_ensemble(
    reid_extractors: List[TorchreidExtractor],
    crops_bgr: List[Optional[np.ndarray]],
    use_half: bool = False,
    batch_size: int = 24,
) -> List[Optional[np.ndarray]]:
    """Batch version of extract_reid_embedding_ensemble (GPU-friendly).

    Returns a list aligned with crops_bgr where each element is either a 512-dim
    L2-normalized embedding or None.

    `batch_size` chunks the work to avoid CUDA OOM on laptop GPUs (e.g., 8GB VRAM).
    """
    if not reid_extractors:
        return [None for _ in crops_bgr]
    if not crops_bgr:
        return []

    bs = int(batch_size) if batch_size is not None else 24
    bs = max(1, bs)

    # Preprocess valid crops
    prepped: List[np.ndarray] = []
    valid_idx: List[int] = []
    for i, crop_bgr in enumerate(crops_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            continue
        try:
            h, w = crop_bgr.shape[:2]
        except Exception:
            continue
        if h < 32 or w < 16:
            continue
        prepped.append(_prep_reid_crop(crop_bgr))
        valid_idx.append(i)

    out: List[Optional[np.ndarray]] = [None for _ in crops_bgr]
    if not prepped:
        return out

    # Run each model on the batch (chunked)
    model_feats: List[np.ndarray] = []

    for ext in reid_extractors:
        chunks: List[np.ndarray] = []
        ok_model = True

        for s in range(0, len(prepped), bs):
            chunk = prepped[s : s + bs]
            amp_ctx = (
                torch.cuda.amp.autocast(enabled=bool(use_half)) if torch.cuda.is_available() else contextlib.nullcontext()
            )
            try:
                with reid_lock, torch.inference_mode(), amp_ctx:
                    feats = ext(chunk)
            except Exception:
                ok_model = False
                break

            arr = _features_to_numpy(feats)
            if arr is None:
                ok_model = False
                break
            if arr.ndim != 2 or arr.shape[0] != len(chunk) or arr.shape[1] != EXPECTED_DIM:
                ok_model = False
                break
            if not np.isfinite(arr).all():
                ok_model = False
                break

            chunks.append(arr.astype(np.float32, copy=False))

        if not ok_model or not chunks:
            continue

        arr_full = np.concatenate(chunks, axis=0)
        if arr_full.ndim != 2 or arr_full.shape[0] != len(prepped) or arr_full.shape[1] != EXPECTED_DIM:
            continue

        model_feats.append(l2_normalize_rows(arr_full))

    if not model_feats:
        return out

    # Fuse across models then L2-normalize per sample
    fused = l2_normalize_rows(np.mean(np.stack(model_feats, axis=0), axis=0))
    for j, i in enumerate(valid_idx):
        v = fused[j].reshape(-1)
        out[i] = v if v.size == EXPECTED_DIM else None
    return out


def extract_back_body_embeddings_batch(
    reid_extractors: List[TorchreidExtractor],
    back_crops_bgr: List[Optional[np.ndarray]],
    args,
    use_half: bool = False,
) -> List[Optional[np.ndarray]]:
    """Batch back-body embedding (ReID + shape-HOG) aligned with back_crops_bgr."""
    if not back_crops_bgr:
        return []

    reid_embs = (
        extract_reid_embeddings_batch_ensemble(reid_extractors, back_crops_bgr, use_half=use_half, batch_size=int(getattr(args, 'reid_batch_size', 24)))
        if reid_extractors
        else [None for _ in back_crops_bgr]
    )

    out: List[Optional[np.ndarray]] = []
    for crop_bgr, reid_emb in zip(back_crops_bgr, reid_embs):
        if crop_bgr is None or crop_bgr.size == 0:
            out.append(None)
            continue

        shape_emb = back_shape_embed_hog(crop_bgr, seed=int(args.back_shape_seed))
        parts: List[np.ndarray] = []
        if reid_emb is not None:
            parts.append(reid_emb * float(args.back_reid_weight))
        if shape_emb is not None:
            parts.append(shape_emb * float(args.back_shape_weight))
        if not parts:
            out.append(None)
            continue

        fused = l2_normalize(np.sum(np.stack(parts, axis=0), axis=0))
        if fused.size != EXPECTED_DIM or not np.isfinite(fused).all():
            out.append(None)
        else:
            out.append(fused)

    return out


def extract_back_body_embedding(
    reid_extractors: List[TorchreidExtractor],
    back_crop_bgr: np.ndarray,
    args,
) -> Optional[np.ndarray]:
    """
    back_body_embedding = L2( back_reid_weight * ReID + back_shape_weight * HOG(shape) )
    Must match extract_service.
    """
    if back_crop_bgr is None or back_crop_bgr.size == 0:
        return None
    reid_emb = extract_reid_embedding_ensemble(reid_extractors, back_crop_bgr) if reid_extractors else None
    shape_emb = back_shape_embed_hog(back_crop_bgr, seed=int(args.back_shape_seed))
    parts: List[np.ndarray] = []
    if reid_emb is not None:
        parts.append(reid_emb * float(args.back_reid_weight))
    if shape_emb is not None:
        parts.append(shape_emb * float(args.back_shape_weight))
    if not parts:
        return None
    fused = l2_normalize(np.sum(np.stack(parts, axis=0), axis=0))
    if fused.size != EXPECTED_DIM or not np.isfinite(fused).all():
        return None
    return fused


def detect_faces_with_embeddings(face_app, frame_bgr: np.ndarray) -> List[Dict]:
    out = []
    if face_app is None:
        return out
    try:
        faces = face_app.get(np.ascontiguousarray(frame_bgr))
        for f in safe_iter_faces(faces):
            bbox = getattr(f, "bbox", None)
            if bbox is None:
                continue
            b = np.asarray(bbox).reshape(-1)
            if b.size < 4:
                continue
            x1, y1, x2, y2 = map(float, b[:4])
            emb = extract_face_embedding(f)
            if emb is None:
                continue
            emb = np.asarray(emb, dtype=np.float32).reshape(-1)
            if emb.size != EXPECTED_DIM or not np.isfinite(emb).all():
                continue
            out.append({"bbox": (x1, y1, x2, y2), "emb": l2_normalize(emb)})
    except Exception:
        return []
    return out


def build_collision_face_only_labels(
    faces: List[Dict],
    track_infos: List[Dict],
    colliding_tids: set,
    gallery: TwoStageGallery,
    args,
) -> Tuple[List[Dict], set]:
    """During crowded collisions, label visible faces directly without body-track mapping.

    This is display-only logic: it does not rewrite track identity state. Once the collision ends,
    the normal tracker pipeline resumes.
    """
    if not faces or not track_infos or not colliding_tids or gallery is None or not getattr(gallery, "people", None):
        return [], set()

    out: List[Dict] = []
    suppress_tids: set = set()
    face_thresh = float(getattr(args, "face_thresh", 0.45) or 0.45)
    gap_min = float(getattr(args, "confirm_gap_min", 0.85) or 0.85)
    face_gap = float(getattr(args, "face_gap", 0.05) or 0.05)

    colliding_boxes: List[Tuple[int, Tuple[float, float, float, float]]] = []
    for info in track_infos:
        tid = int(info.get("tid", -1))
        if tid not in colliding_tids:
            continue
        try:
            x1, y1, x2, y2 = info.get("bbox", (0, 0, 0, 0))
            colliding_boxes.append((tid, (float(x1), float(y1), float(x2), float(y2))))
        except Exception:
            continue

    if not colliding_boxes:
        return [], set()

    for f in faces:
        fbox = f.get("bbox", None)
        if fbox is None:
            continue
        try:
            fx1, fy1, fx2, fy2 = map(float, fbox)
        except Exception:
            continue

        fcx = 0.5 * (fx1 + fx2)
        fcy = 0.5 * (fy1 + fy2)
        owners: List[int] = []
        best_link = 0.0
        for tid, box in colliding_boxes:
            link = float(ioa_xyxy((fx1, fy1, fx2, fy2), box))
            if _point_in_xyxy(fcx, fcy, box) or link >= 0.10:
                owners.append(int(tid))
                best_link = max(best_link, link)

        if not owners:
            continue

        label, score, second = best_face_label_top2(f.get("emb", None), gallery)
        if not label or float(score) < face_thresh:
            continue

        conf = _gap_conf(score, second, face_gap)
        if float(conf) < gap_min:
            continue

        out.append(
            {
                "bbox": (fx1, fy1, fx2, fy2),
                "label": str(label),
                "score": float(score),
                "conf": float(conf),
                "rank": float(score) * float(conf),
                "owner_tids": list(dict.fromkeys(int(t) for t in owners)),
                "face_link": float(best_link),
            }
        )

    if not out:
        return [], set()

    # Keep only the strongest instance per label to avoid duplicate face-only labels in the same crowd frame.
    best_by_label: Dict[str, Dict] = {}
    for row in out:
        lab = str(row.get("label", "") or "")
        if not lab:
            continue
        prev = best_by_label.get(lab)
        if prev is None or float(row.get("rank", 0.0) or 0.0) > float(prev.get("rank", 0.0) or 0.0):
            best_by_label[lab] = row

    final_rows = sorted(best_by_label.values(), key=lambda r: float(r.get("rank", 0.0) or 0.0), reverse=True)
    for row in final_rows:
        for tid in row.get("owner_tids", []):
            suppress_tids.add(int(tid))
    return final_rows, suppress_tids


def tlwh_to_tlbr(tlwh: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
    x, y, w, h = tlwh
    return (float(x), float(y), float(x + w), float(y + h))


@dataclass
class FallbackTrack:
    track_id: int
    _tlbr: Tuple[float, float, float, float]
    det_conf: float
    hits: int
    age: int
    n_init: int

    def is_confirmed(self) -> bool:
        return self.hits >= self.n_init

    def to_tlbr(self) -> Tuple[float, float, float, float]:
        return self._tlbr


@dataclass
class BoxMOTTrack:
    """Lightweight adapter for BoxMOT/StrongSORT outputs."""

    track_id: int
    _tlbr: Tuple[float, float, float, float]
    det_conf: float

    def is_confirmed(self) -> bool:
        return True

    def to_tlbr(self) -> Tuple[float, float, float, float]:
        return self._tlbr


class SimpleIoUTracker:
    """
    Minimal online tracker:
    - Associates detections to existing tracks by IoU (greedy)
    - Keeps stable IDs across frames
    """

    def __init__(self, max_age: int, n_init: int, iou_threshold: float = 0.30):
        self.max_age = int(max_age)
        self.n_init = int(n_init)
        self.iou_threshold = float(iou_threshold)
        self._next_id = 1
        self._tracks: List[FallbackTrack] = []

    def update(self, det_tlwh_conf: List[Tuple[float, float, float, float, float]]) -> List[FallbackTrack]:
        # age all tracks
        for tr in self._tracks:
            tr.age += 1

        if not det_tlwh_conf:
            self._tracks = [t for t in self._tracks if t.age <= self.max_age]
            return list(self._tracks)

        dets_tlbr = [tlwh_to_tlbr((x, y, w, h)) for x, y, w, h, _ in det_tlwh_conf]
        dets_conf = [float(cf) for *_, cf in det_tlwh_conf]

        # build IoU pairs
        pairs: List[Tuple[float, int, int]] = []
        for ti, tr in enumerate(self._tracks):
            for di, dbox in enumerate(dets_tlbr):
                pairs.append((iou_xyxy(tr._tlbr, dbox), ti, di))
        pairs.sort(key=lambda x: x[0], reverse=True)

        matched_tracks = set()
        matched_dets = set()

        for iou, ti, di in pairs:
            if iou < self.iou_threshold:
                break
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            tr = self._tracks[ti]
            tr._tlbr = dets_tlbr[di]
            tr.det_conf = dets_conf[di]
            tr.hits += 1
            tr.age = 0

        # create new tracks for unmatched detections
        for di in range(len(dets_tlbr)):
            if di in matched_dets:
                continue
            tr = FallbackTrack(
                track_id=self._next_id,
                _tlbr=dets_tlbr[di],
                det_conf=dets_conf[di],
                hits=1,
                age=0,
                n_init=self.n_init,
            )
            self._next_id += 1
            self._tracks.append(tr)

        # drop dead tracks
        self._tracks = [t for t in self._tracks if t.age <= self.max_age]
        return list(self._tracks)


def _drop_name_from_track(identity_state: Dict[int, Dict], tid: int, name: str) -> None:
    entry = identity_state.get(tid)
    if not entry:
        return
    scores = entry.get("scores", {})
    if name in scores:
        scores[name] *= 0.05
        if scores[name] < 1e-6:
            scores.pop(name, None)
    if entry.get("last") == name:
        entry["last"] = ""
        entry["ttl"] = 0


def enforce_unique_names(
    identity_state: Dict[int, Dict],
    track_rows: List[Dict],
) -> None:
    """
    If multiple tracks want to show the same stable name in the same frame,
    keep only one to avoid on-screen duplicates.

    Updated behavior (less "juggling" during collisions):
      - Prefer the track that already had this name in the previous frame (`prev_name == nm`)
      - Otherwise prefer the highest current score (instant_score, fallback stable_accum)
    """
    buckets: Dict[str, List[Dict]] = {}
    for r in track_rows:
        nm = r.get("stable_name", "")
        if nm:
            buckets.setdefault(nm, []).append(r)

    def _row_score(r: Dict) -> float:
        try:
            return float(r.get("instant_score", 0.0) or r.get("stable_accum", 0.0) or 0.0)
        except Exception:
            return 0.0

    for nm, rows in buckets.items():
        if len(rows) <= 1:
            continue

        sticky = [r for r in rows if str(r.get("prev_name", "")) == str(nm)]
        if sticky:
            sticky.sort(key=_row_score, reverse=True)
            winner = sticky[0]
        else:
            rows.sort(key=_row_score, reverse=True)
            winner = rows[0]

        for r in rows:
            if r is winner:
                continue
            r["stable_name"] = ""
            _drop_name_from_track(identity_state, int(r["tid"]), nm)




def _hard_drop_name_from_track(identity_state: Dict[int, Dict], tid: int, name: str) -> None:
    """Hard-remove a specific name from a track.

    Used when a *strong face* match proves that a name belongs to a different track.
    This is stricter than `_drop_name_from_track`:
      - removes the score entry for that name
      - clears `last`/`ttl` if this name was currently displayed
      - clears CONFIRMED state if it was tied to this name
    """
    entry = identity_state.get(tid)
    if not entry:
        return

    try:
        scores = entry.get("scores", {})
        if isinstance(scores, dict):
            scores.pop(str(name), None)
    except Exception:
        pass

    if str(entry.get("last", "") or "") == str(name):
        entry["last"] = ""
        entry["ttl"] = 0

    # Clear confirmation if it was for this identity
    try:
        if bool(entry.get("confirmed", False)) and str(entry.get("confirmed_name", "") or "") == str(name):
            entry["confirmed"] = False
            entry["confirmed_name"] = ""
            entry["confirmed_src"] = ""
    except Exception:
        pass


def apply_strong_face_reclaim(
    identity_state: Dict[int, Dict],
    track_rows: List[Dict],
    args,
    now_ts: float,
    name_registry: Optional[Dict[str, Dict]] = None,
) -> None:
    """Global "identity reclaim" using strong face evidence.

    Problem this solves (your example):
      - A stands in front of B (occlusion/overlap) -> mapping can temporarily drift.
      - When A moves away, A's face becomes clearly visible again.
      - If B was incorrectly holding A's name, we must **switch**:
          * assign A -> the track that has the strong face match
          * remove A from any other visible track immediately

    We only reclaim when:
      - the face match is strong (score >= --face-thresh AND gap_conf >= --confirm-gap-min)
      - the track is NOT currently colliding (no active overlap ambiguity)
      - geometry is clean enough:
          * normal: face_link >= --face-iou-link
          * post-collision cooldown: use the stricter collision face gates
    """
    if not track_rows:
        return

    # Respect debug mode
    if bool(getattr(args, "allow_duplicate_names", False)):
        return

    face_thresh = float(getattr(args, "face_thresh", 0.45) or 0.45)
    conf_min = float(getattr(args, "confirm_gap_min", 0.85) or 0.85)

    # Reclaim should never trigger during an *active* collision.
    normal_min_ioa = float(getattr(args, "face_iou_link", 0.35) or 0.35)

    # In the post-collision cooldown window (embed_freeze>0), require extra-strict geometry.
    cool_min_ioa = float(getattr(args, "collision_face_ioa", 0.85) or 0.85)
    cool_min_ioa_gap = float(getattr(args, "collision_face_ioa_gap", 0.20) or 0.20)

    ttl_reset = int(getattr(args, "name_ttl", 20) or 20)
    hold_seconds = float(getattr(args, "name_hold_seconds", 0.0) or 0.0)
    min_score = float(getattr(args, "name_min_score", 0.40) or 0.40)
    w_face = float(getattr(args, "name_face_weight", 1.2) or 1.2)

    # Visible tracks (this frame)
    tid_to_row_idxs: Dict[int, List[int]] = {}
    for i, r in enumerate(track_rows):
        try:
            tid = int(r.get("tid", -1))
        except Exception:
            continue
        if tid < 0:
            continue
        tid_to_row_idxs.setdefault(tid, []).append(i)
    visible_tids = set(tid_to_row_idxs.keys())
    if not visible_tids:
        return

    # Pick the best strong-face claimant per label
    best_claim: Dict[str, Tuple[int, float]] = {}  # label -> (row_idx, score)
    for i, r in enumerate(track_rows):
        try:
            tid = int(r.get("tid", -1))
        except Exception:
            continue
        if tid < 0:
            continue

        # Never reclaim while actively colliding this frame
        if bool(r.get("colliding", False)):
            continue

        label = str(r.get("face_label", "") or "")
        if not label:
            continue

        face_score = float(r.get("face_score", 0.0) or 0.0)
        face_conf = float(r.get("face_conf", 0.0) or 0.0)
        if (face_score < face_thresh) or (face_conf < conf_min):
            continue

        face_link = float(r.get("face_link", 0.0) or 0.0)
        face_other = float(r.get("face_othermax", 0.0) or 0.0)
        ef = int(r.get("embed_freeze", 0) or 0)

        # Geometry gate: if we're still in the cooldown window after a collision, require unambiguous face assignment.
        if ef > 0:
            if (face_link < cool_min_ioa) or ((face_link - face_other) < cool_min_ioa_gap):
                continue
        else:
            if face_link < normal_min_ioa:
                continue

        # Ranking score: prefer face_vote (already scaled by gap confidence), fallback to raw face_score.
        score = float(r.get("face_vote", 0.0) or 0.0)
        if score <= 0.0:
            score = float(face_score)

        prev = best_claim.get(label)
        if (prev is None) or (float(score) > float(prev[1])):
            best_claim[label] = (int(i), float(score))

    if not best_claim:
        return

    # Helper: clear a label from any row belonging to tid (display + state)
    def _clear_label_from_tid(_tid: int, _label: str) -> None:
        _hard_drop_name_from_track(identity_state, int(_tid), str(_label))
        for ridx in tid_to_row_idxs.get(int(_tid), []):
            if str(track_rows[ridx].get("stable_name", "") or "") == str(_label):
                track_rows[ridx]["stable_name"] = ""
                track_rows[ridx]["stable_accum"] = 0.0

    # Apply transfers
    for label, (claim_row_idx, _score) in best_claim.items():
        row_claim = track_rows[int(claim_row_idx)]
        try:
            tid_claim = int(row_claim.get("tid", -1))
        except Exception:
            continue
        if tid_claim < 0:
            continue

        # Identify any other visible "owners" of this label (by display, by identity_state, or by registry)
        owner_tids: set = set()

        for r in track_rows:
            try:
                t = int(r.get("tid", -1))
            except Exception:
                continue
            if t < 0 or t == tid_claim:
                continue
            if str(r.get("stable_name", "") or "") == str(label):
                owner_tids.add(t)

        for t in visible_tids:
            if t == tid_claim:
                continue
            ent = identity_state.get(int(t), {})
            if str(ent.get("last", "") or "") == str(label):
                owner_tids.add(int(t))

        if name_registry is not None:
            reg = name_registry.get(str(label))
            if reg is not None:
                try:
                    reg_tid = int(reg.get("tid", -1) or -1)
                except Exception:
                    reg_tid = -1
                if reg_tid in visible_tids and reg_tid != tid_claim:
                    owner_tids.add(int(reg_tid))

        # Remove from other owners
        for t in list(owner_tids):
            if int(t) == int(tid_claim):
                continue
            _clear_label_from_tid(int(t), str(label))

        # Assign to claimant (even if it previously had a wrong name)
        old_nm = str(row_claim.get("stable_name", "") or "")
        if old_nm and old_nm != str(label):
            _hard_drop_name_from_track(identity_state, int(tid_claim), str(old_nm))

        row_claim["stable_name"] = str(label)

        ent = identity_state.setdefault(
            int(tid_claim),
            {
                "scores": {},
                "last": "",
                "ttl": 0,
                "freeze": 0,
                "hold_until": 0.0,
                "confirmed": False,
                "confirmed_name": "",
                "confirmed_src": "",
            },
        )
        scores = ent.setdefault("scores", {})

        try:
            face_vote = float(row_claim.get("face_vote", 0.0) or 0.0)
        except Exception:
            face_vote = 0.0

        boost = max(float(min_score) * 2.0, float(face_vote) * float(w_face), float(_score))
        scores[str(label)] = max(float(scores.get(str(label), 0.0) or 0.0), float(boost))

        ent["last"] = str(label)
        ent["ttl"] = int(ttl_reset)

        if float(hold_seconds) > 0.0:
            ent["hold_until"] = max(float(ent.get("hold_until", 0.0) or 0.0), float(now_ts) + float(hold_seconds))

        # Mark confirmed (this came from strong face evidence)
        if bool(getattr(args, "persist_confirmed_until_collision", True)):
            ent["confirmed"] = True
            ent["confirmed_name"] = str(label)
            ent["confirmed_src"] = "face_reclaim"

        row_claim["stable_accum"] = float(scores.get(str(label), 0.0) or 0.0)
        row_claim["reclaimed_by_face"] = True



def _cosine_sim(a: Optional[np.ndarray], b: Optional[np.ndarray]) -> Optional[float]:
    if a is None or b is None:
        return None
    try:
        a = np.asarray(a, dtype=np.float32).reshape(-1)
        b = np.asarray(b, dtype=np.float32).reshape(-1)
        if a.size != EXPECTED_DIM or b.size != EXPECTED_DIM:
            return None
        if (not np.isfinite(a).all()) or (not np.isfinite(b).all()):
            return None
        # embeddings are L2-normalized -> dot == cosine similarity
        return float(np.dot(a, b))
    except Exception:
        return None


def _reg_emb_for_src(reg: Dict, src: str) -> Optional[np.ndarray]:
    if src == "face":
        return reg.get("face_emb", None)
    if src == "back":
        return reg.get("back_emb", None)
    return reg.get("body_emb", None)


def prune_name_registry(name_registry: Dict[str, Dict], now_ts: float, prune_seconds: float) -> None:
    if not name_registry:
        return
    try:
        prune_seconds = float(prune_seconds)
    except Exception:
        prune_seconds = 10.0
    if prune_seconds <= 0.0:
        return

    for nm in list(name_registry.keys()):
        try:
            last_ts = float(name_registry[nm].get("last_seen_ts", 0.0) or 0.0)
        except Exception:
            last_ts = 0.0
        if (float(now_ts) - last_ts) > prune_seconds:
            name_registry.pop(nm, None)


def filter_candidates_by_name_registry(
    candidates: List[Tuple[str, float, str]],
    tid: int,
    bbox_xyxy: Tuple[float, float, float, float],
    feat_state_for_tid: Dict,
    name_registry: Dict[str, Dict],
    now_ts: float,
    reserve_seconds: float,
    transfer_iou: float,
    transfer_sim: float,
) -> List[Tuple[str, float, str]]:
    """Prevent a name from "jumping" to another track right after a collision/overlap.

    If a name was seen on a *different* track within reserve_seconds, we block it unless BOTH:
      - IoU(current_box, last_box_of_name) >= transfer_iou, AND
      - cosine_sim(current_modality_emb, last_modality_emb_of_name) >= transfer_sim
    """
    if not candidates or not name_registry:
        return candidates

    out: List[Tuple[str, float, str]] = []
    rs = float(max(0.0, reserve_seconds))
    for label, vote, src in candidates:
        if not label:
            continue
        reg = name_registry.get(str(label))
        if not reg:
            out.append((label, vote, src))
            continue

        reg_tid = int(reg.get("tid", -1) or -1)
        if reg_tid == int(tid):
            out.append((label, vote, src))
            continue

        # If reservation expired -> allow
        last_ts = float(reg.get("last_seen_ts", 0.0) or 0.0)
        if rs <= 0.0 or (float(now_ts) - last_ts) > rs:
            out.append((label, vote, src))
            continue

        # Still reserved by someone else -> only allow transfer with strong evidence

        # Body/back cues are too weak to allow a reserved name to transfer/jump (face-only identification).
        # This prevents: known A collides with unknown B, then B steals 'A' while only body/back is visible.
        if str(src) in ("back", "body"):
            continue
        # Collision-face can confirm identity but must NEVER transfer a reserved name to a different track.
        if str(src) == "face_collision":
            continue
        reg_box = reg.get("bbox", None)
        if reg_box is None:
            continue

        try:
            iou = iou_xyxy(tuple(map(float, bbox_xyxy)), tuple(map(float, reg_box)))
        except Exception:
            iou = 0.0

        if src in ("face", "face_collision"):
            cur_emb = feat_state_for_tid.get("face_ema", None)
        elif src == "back":
            cur_emb = feat_state_for_tid.get("back_ema", None)
        else:
            cur_emb = feat_state_for_tid.get("body_ema", None)

        reg_emb = _reg_emb_for_src(reg, src)
        sim = _cosine_sim(cur_emb, reg_emb)

        if sim is not None and float(iou) >= float(transfer_iou) and float(sim) >= float(transfer_sim):
            out.append((label, vote, src))
        else:
            # block candidate (prevents "B becomes A" after overlap)
            continue

    return out


def update_name_registry_for_track(
    name_registry: Dict[str, Dict],
    stable_name: str,
    tid: int,
    bbox_xyxy: Tuple[int, int, int, int],
    feat_state_for_tid: Dict,
    now_ts: float,
) -> None:
    if not stable_name:
        return

    try:
        x1, y1, x2, y2 = bbox_xyxy
        box = (float(x1), float(y1), float(x2), float(y2))
    except Exception:
        box = None

    def _safe_emb(x) -> Optional[np.ndarray]:
        if x is None:
            return None
        try:
            a = np.asarray(x, dtype=np.float32).reshape(-1)
            if a.size != EXPECTED_DIM or (not np.isfinite(a).all()):
                return None
            return a
        except Exception:
            return None

    name_registry[str(stable_name)] = {
        "tid": int(tid),
        "last_seen_ts": float(now_ts),
        "bbox": box,
        "face_emb": _safe_emb(feat_state_for_tid.get("face_ema", None)),
        "body_emb": _safe_emb(feat_state_for_tid.get("body_ema", None)),
        "back_emb": _safe_emb(feat_state_for_tid.get("back_ema", None)),
    }
def repair_id_switches_by_iou(
    track_infos: List[Dict],
    feat_state: Dict[int, Dict],
    identity_state: Dict[int, Dict],
    frame_idx: int,
    iou_thresh: float,
    max_age: int,
    current_colliding_tids: Optional[set] = None,
    cloth_min_sim: float = 0.55,
    cloth_margin: float = 0.08,
) -> None:
    """
    Appearance-assisted ID-switch repair.

    Important fixes versus the older version:
      1) We no longer rely on IoU alone. During crossings, IoU-only matching can easily attach
         A's state to B's track.
      2) We clone per-track state before remapping. The older shallow assignment could alias two
         track IDs to the same dict and cause state bleed / "parasite" behavior.

    Current behavior:
      - never remap while tracks are actively colliding
      - when tracks separate, use clothing + IoU + center continuity to decide whether tracker IDs
        likely flipped
      - only transfer state if the alternate assignment clearly beats keeping the same tracker ID
    """
    try:
        if not track_infos or len(track_infos) < 2:
            return
        if not feat_state:
            return

        iou_thresh = float(iou_thresh)
        max_age = int(max_age)
        cloth_min_sim = float(cloth_min_sim)
        cloth_margin = float(cloth_margin)
        colliding_now = {int(t) for t in (current_colliding_tids or set())}

        cur_boxes: Dict[int, Tuple[float, float, float, float]] = {}
        cur_cloth: Dict[int, Optional[np.ndarray]] = {}
        for info in track_infos:
            tid = int(info.get("tid", -1))
            if tid < 0 or tid in colliding_now:
                continue
            x1, y1, x2, y2 = info.get("bbox", (0, 0, 0, 0))
            cur_boxes[tid] = (float(x1), float(y1), float(x2), float(y2))
            cur_cloth[tid] = info.get("cloth_desc", None)

        if len(cur_boxes) < 2:
            return

        prev_boxes: Dict[int, Tuple[float, float, float, float]] = {}
        prev_cloth: Dict[int, Optional[np.ndarray]] = {}
        for tid, st in feat_state.items():
            try:
                tid_i = int(tid)
            except Exception:
                continue
            lb = st.get("last_bbox", None)
            if lb is None:
                continue
            age = int(frame_idx) - int(st.get("last_seen", frame_idx))
            if age < 1 or age > max_age:
                continue
            x1, y1, x2, y2 = lb
            prev_boxes[tid_i] = (float(x1), float(y1), float(x2), float(y2))
            prev_cloth[tid_i] = st.get("cloth_ema", None)

        if len(prev_boxes) < 2:
            return

        def _pair_metrics(ptid: int, ctid: int) -> Tuple[float, Optional[float], float, float]:
            iou = float(iou_xyxy(prev_boxes[ptid], cur_boxes[ctid]))
            cloth = _cosine_sim_anydim(prev_cloth.get(ptid, None), cur_cloth.get(ctid, None))
            center = float(_box_center_score(prev_boxes[ptid], cur_boxes[ctid]))
            if cloth is None:
                score = (0.75 * iou) + (0.25 * center)
            else:
                score = (0.60 * max(0.0, cloth)) + (0.25 * iou) + (0.15 * center)
            return iou, cloth, center, float(score)

        # Baseline score for "keep the same tracker id".
        self_score: Dict[int, float] = {}
        for ctid in cur_boxes.keys():
            if ctid in prev_boxes:
                _, _, _, sc = _pair_metrics(ctid, ctid)
                self_score[int(ctid)] = float(sc)

        pairs: List[Tuple[float, float, float, float, int, int]] = []
        for ptid in prev_boxes.keys():
            for ctid in cur_boxes.keys():
                iou, cloth, center, score = _pair_metrics(ptid, ctid)
                cloth_ok = (cloth is not None) and (float(cloth) >= cloth_min_sim)
                if ptid == ctid or iou >= iou_thresh or cloth_ok:
                    pairs.append((score, iou, float(cloth) if cloth is not None else -1.0, center, int(ptid), int(ctid)))

        if not pairs:
            return

        pairs.sort(key=lambda x: (x[0], x[2], x[1], x[3]), reverse=True)

        used_prev = set()
        used_cur = set()
        mapping: Dict[int, int] = {}

        for score, iou, cloth, center, ptid, ctid in pairs:
            if ptid in used_prev or ctid in used_cur:
                continue

            if ptid != ctid:
                # Crossing/post-occlusion transfer is only allowed with strong clothing evidence,
                # and only if it clearly beats staying on the same tracker ID.
                if cloth < cloth_min_sim:
                    continue
                sc_self = self_score.get(int(ctid), None)
                if sc_self is not None and (float(score) - float(sc_self)) < cloth_margin:
                    continue
                # Also insist on at least some spatial continuity so a far-away new entrant can't steal state.
                if max(float(iou), float(center)) < max(0.10, iou_thresh * 0.5):
                    continue

            used_prev.add(int(ptid))
            used_cur.add(int(ctid))
            mapping[int(ctid)] = int(ptid)

        if not mapping:
            return

        # Only do work if at least one mismatch exists.
        if all(int(ctid) == int(ptid) for ctid, ptid in mapping.items()):
            return

        old_feat = {int(k): _clone_state_entry(v) for k, v in feat_state.items()}
        old_id = {int(k): _clone_state_entry(v) for k, v in identity_state.items()}

        for ctid, ptid in mapping.items():
            if ctid == ptid:
                continue
            if ptid in old_feat:
                new_feat = _clone_state_entry(old_feat[ptid])
                new_feat["last_bbox"] = cur_boxes.get(int(ctid), new_feat.get("last_bbox", None))
                new_feat["last_seen"] = int(frame_idx)
                feat_state[int(ctid)] = new_feat
            if ptid in old_id:
                identity_state[int(ctid)] = _clone_state_entry(old_id[ptid])
    except Exception:
        return





def _gap_conf(best: float, second: float, full_gap: float) -> float:
    """
    Convert a (best-second) gap into a [0..1] confidence scaler.
    - If full_gap is small/0, any positive gap becomes full confidence.
    - If gap <= 0 => 0 confidence.
    """
    gap = float(best) - float(second)
    if gap <= 0.0:
        return 0.0
    denom = max(1e-6, float(full_gap))
    return float(max(0.0, min(1.0, gap / denom)))


def process_one_frame(
    frame_idx: int,
    frame: np.ndarray,
    sid: int,
    yolo,
    reid_extractors: List[TorchreidExtractor],
    face_app,
    deep_tracker,
    fallback_tracker: Optional[SimpleIoUTracker],
    gallery: TwoStageGallery,
    args,
    identity_state: Dict[int, Dict],
    feat_state: Dict[int, Dict],
    name_registry: Optional[Dict[str, Dict]],
    cap_ts: Optional[float] = None,
    room_id: str = "",
    room_tracker: Optional["SpatioTemporalRoomTracker"] = None,
) -> np.ndarray:
    """Process a single frame.

    Identification strategy (updated):
      - Tracklet-level embeddings (EMA per track) for body/back/face.
      - Body/back ReID updates happen every N frames per track (configurable),
        and less frequently once a track already has a stable identity.
      - Face detection is still controlled by --face-every-n, but we also keep an EMA face embedding per track.
      - Face is the ONLY modality allowed to assign/switch an identity; body/back only SUPPORT continuity.
      - Only KNOWN identities are drawn (requested behavior).
    """

    def _ema_update(old: Optional[np.ndarray], new: Optional[np.ndarray], beta: float) -> Optional[np.ndarray]:
        if new is None:
            return old
        new = np.asarray(new, dtype=np.float32).reshape(-1)
        if new.size <= 0 or (not np.isfinite(new).all()):
            return old
        if old is None:
            return l2_normalize(new)
        old = np.asarray(old, dtype=np.float32).reshape(-1)
        if old.size != new.size or (not np.isfinite(old).all()):
            return l2_normalize(new)
        b = float(beta)
        b = max(0.0, min(0.999, b))
        return l2_normalize(old * b + new * (1.0 - b))
    def _assign_faces_greedy(
        track_infos: List[Dict],
        faces: List[Dict],
        link_thr: float,
        skip_tids: Optional[set] = None,
    ) -> None:
        """
        Greedy 1-to-1 face->track assignment to prevent multiple tracks from "sharing" the same face
        when person boxes overlap.

        For each track we store:
          - face_cand_emb: face embedding (this frame)
          - face_cand_bbox: face bbox
          - face_cand_link: IoA(face, assigned_track_box)
          - face_cand_othermax: max IoA(face, any OTHER track box)  (used as an ambiguity score)
        """
        skip_tids = {int(t) for t in (skip_tids or set())}
        for info in track_infos:
            info["face_cand_link"] = 0.0
            info["face_cand_othermax"] = 0.0
            info["face_cand_bbox"] = None
            info["face_cand_emb"] = None
            info["_face_cand_fi"] = None

        if not faces or not track_infos:
            return

        pairs: List[Tuple[float, int, int]] = []
        for ti, info in enumerate(track_infos):
            if int(info.get("tid", -1)) in skip_tids:
                continue
            x1, y1, x2, y2 = info["bbox"]
            tbox = (float(x1), float(y1), float(x2), float(y2))
            for fi, f in enumerate(faces):
                fbox = f.get("bbox", None)
                if fbox is None:
                    continue
                try:
                    fx1, fy1, fx2, fy2 = map(float, fbox)
                except Exception:
                    continue
                cx = 0.5 * (fx1 + fx2)
                cy = 0.5 * (fy1 + fy2)
                if not _point_in_xyxy(cx, cy, tbox):
                    continue
                link = float(ioa_xyxy((fx1, fy1, fx2, fy2), tbox))
                if link >= float(link_thr):
                    pairs.append((link, int(ti), int(fi)))

        if not pairs:
            # cleanup helper
            for info in track_infos:
                info.pop("_face_cand_fi", None)
            return

        pairs.sort(key=lambda x: x[0], reverse=True)
        used_t: set = set()
        used_f: set = set()
        for link, ti, fi in pairs:
            if ti in used_t or fi in used_f:
                continue
            used_t.add(ti)
            used_f.add(fi)
            info = track_infos[int(ti)]
            f = faces[int(fi)]
            info["face_cand_link"] = float(link)
            try:
                info["face_cand_bbox"] = tuple(map(float, f.get("bbox", ())))
            except Exception:
                info["face_cand_bbox"] = None
            info["face_cand_emb"] = f.get("emb", None)
            info["_face_cand_fi"] = int(fi)

        # For strict collision gating, estimate how ambiguous the assigned face is:
        # if the same face sits well inside other track boxes, treat it as ambiguous.
        for ti, info in enumerate(track_infos):
            if int(info.get("tid", -1)) in skip_tids:
                continue
            fi = info.get("_face_cand_fi", None)
            if fi is None:
                continue
            f = faces[int(fi)]
            fbox = f.get("bbox", None)
            if fbox is None:
                continue
            try:
                fx1, fy1, fx2, fy2 = map(float, fbox)
            except Exception:
                continue
            cx = 0.5 * (fx1 + fx2)
            cy = 0.5 * (fy1 + fy2)
            othermax = 0.0
            for tj, info2 in enumerate(track_infos):
                if tj == ti:
                    continue
                x1, y1, x2, y2 = info2["bbox"]
                tbox2 = (float(x1), float(y1), float(x2), float(y2))
                if not _point_in_xyxy(cx, cy, tbox2):
                    continue
                othermax = max(othermax, float(ioa_xyxy((fx1, fy1, fx2, fy2), tbox2)))
            info["face_cand_othermax"] = float(othermax)

        for info in track_infos:
            info.pop("_face_cand_fi", None)

    # For time-based stability logic (name hold / reservation)
    now_ts = time.monotonic()
    if name_registry is None:
        name_registry = {}
    prune_name_registry(
        name_registry,
        now_ts=now_ts,
        prune_seconds=float(getattr(args, "name_reserve_prune_seconds", 10.0) or 10.0),
    )

    H, W = frame.shape[:2]

    rw, rh = int(args.resize[0]), int(args.resize[1])
    if rw > 0 and rh > 0:
        frame = cv2.resize(frame, (rw, rh), interpolation=cv2.INTER_LINEAR)
        H, W = frame.shape[:2]

    # Make sure OpenCV views are safe for downstream models
    frame = np.ascontiguousarray(frame)
    out = frame.copy()

    gpu = torch.cuda.is_available() and ("cuda" in str(args.device).lower())
    use_half = bool(gpu and bool(getattr(args, "half", False)))

    # YOLO persons -> tlwh_conf
    tlwh_conf: List[Tuple[float, float, float, float, float]] = []
    if yolo is not None:
        try:
            res = _yolo_forward_safe(yolo, frame, args, gpu=gpu)
            boxes = res[0].boxes if (res and len(res)) else None
            if boxes is not None and len(boxes) > 0:
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
                cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
                keep = (cls == 0)  # 'person'
                xyxy, conf = xyxy[keep], conf[keep]
                for (x1, y1, x2, y2), c in zip(xyxy, conf):
                    x1f = float(max(0, min(W - 1, x1)))
                    y1f = float(max(0, min(H - 1, y1)))
                    x2f = float(max(0, min(W - 1, x2)))
                    y2f = float(max(0, min(H - 1, y2)))
                    ww = float(max(1.0, x2f - x1f))
                    hh = float(max(1.0, y2f - y1f))
                    if ww < args.min_box_wh or hh < args.min_box_wh:
                        continue
                    tlwh_conf.append((x1f, y1f, ww, hh, float(c)))
        except Exception as e:
            print(f"[SRC {sid}] YOLO error:", e)

    # Tracking
    tracks = []
    if deep_tracker is not None:
        if tlwh_conf:
            dets_ss = np.asarray(
                [[float(x), float(y), float(x + w), float(y + h), float(cf), 0.0] for x, y, w, h, cf in tlwh_conf],
                dtype=np.float32,
            )
        else:
            dets_ss = np.empty((0, 6), dtype=np.float32)

        try:
            mot_out = deep_tracker.update(dets_ss, frame)
            if mot_out is None or len(mot_out) == 0:
                tracks = []
            else:
                mot_arr = np.asarray(mot_out, dtype=np.float32)
                if mot_arr.ndim == 1:
                    mot_arr = mot_arr.reshape(1, -1)
                tracks = [
                    BoxMOTTrack(
                        track_id=int(r[4]),
                        _tlbr=(float(r[0]), float(r[1]), float(r[2]), float(r[3])),
                        det_conf=float(r[5]) if len(r) > 5 else 0.0,
                    )
                    for r in mot_arr
                    if len(r) >= 5
                ]
        except Exception as e:
            print(f"[SRC {sid}] StrongSORT update error:", e)
            tracks = []
    else:
        if fallback_tracker is None:
            fallback_tracker = SimpleIoUTracker(max_age=args.max_age, n_init=args.n_init, iou_threshold=FALLBACK_IOU_THRESH)
        tracks = fallback_tracker.update(list(tlwh_conf))

    # Collect confirmed tracks
    track_infos: List[Dict] = []
    for t in tracks:
        if hasattr(t, "is_confirmed") and callable(getattr(t, "is_confirmed")) and not t.is_confirmed():
            continue
        try:
            x1, y1, x2, y2 = map(float, t.to_tlbr())
            x1i, y1i = int(max(0, x1)), int(max(0, y1))
            x2i, y2i = int(min(W, x2)), int(min(H, y2))
            if x2i <= x1i or y2i <= y1i:
                continue
            tid = int(getattr(t, "track_id", -1))
            det_conf = float(getattr(t, "det_conf", 0.0) or 0.0)
        except Exception:
            continue

        track_infos.append(
            {
                "tid": tid,
                "bbox": (x1i, y1i, x2i, y2i),
                "det_conf": det_conf,
                "body_updated": False,
                "back_updated": False,
                "face_updated": False,
                "face_emb": None,
                "face_link": 0.0,
                "cloth_desc": extract_clothing_descriptor(frame, (x1i, y1i, x2i, y2i)),
            }
        )

    if not track_infos:
        return out

    # --- Collision detection (box overlap) to freeze identity updates ---
    colliding_tids = set()
    try:
        thr = float(getattr(args, "collision_iou", 0.35))
        if thr > 0.0 and len(track_infos) >= 2:
            for i in range(len(track_infos)):
                ti = int(track_infos[i]["tid"])
                bi = track_infos[i]["bbox"]
                b1 = (float(bi[0]), float(bi[1]), float(bi[2]), float(bi[3]))
                for j in range(i + 1, len(track_infos)):
                    tj = int(track_infos[j]["tid"])
                    bj = track_infos[j]["bbox"]
                    b2 = (float(bj[0]), float(bj[1]), float(bj[2]), float(bj[3]))
                    if iou_xyxy(b1, b2) >= thr:
                        colliding_tids.add(ti)
                        colliding_tids.add(tj)
    except Exception:
        colliding_tids = set()

    # Face detections: normal cadence, but force every frame while any active collision is ongoing.
    faces: List[Dict] = []
    face_tick = (frame_idx % max(1, args.face_every_n) == 0)
    force_face_every_frame = bool(getattr(args, "collision_face_every_frame", True)) and bool(colliding_tids)
    if args.use_face and face_app is not None and (force_face_every_frame or face_tick):
        faces = detect_faces_with_embeddings(face_app, frame)

    # Crowd/occlusion mode: if a visible face is present inside a collision region, label the FACE directly.
    # We deliberately do not map those faces back to the body tracks.
    crowd_face_rows: List[Dict] = []
    suppress_track_draw_tids: set = set()
    crowd_face_only_mode = bool(getattr(args, "crowd_face_only_labels", True)) and bool(colliding_tids)
    if crowd_face_only_mode and faces:
        crowd_face_rows, suppress_track_draw_tids = build_collision_face_only_labels(
            faces=faces,
            track_infos=track_infos,
            colliding_tids=colliding_tids,
            gallery=gallery,
            args=args,
        )

    # --- Face->track linking (greedy IoA assignment; avoids face "sharing" when boxes overlap) ---
    # In crowd-face-only mode we EXCLUDE colliding tracks from face->body assignment and just label
    # the faces themselves inside that collision region.
    if bool(getattr(args, "use_face", False)) and (face_app is not None) and faces:
        skip_face_track_tids = colliding_tids if crowd_face_only_mode else set()
        _assign_faces_greedy(
            track_infos,
            faces,
            link_thr=float(getattr(args, "face_iou_link", 0.35) or 0.35),
            skip_tids=skip_face_track_tids,
        )

    # --- Robustness: repair tracker ID switches after we know whether tracks are still colliding. ---
    # We never try to remap while boxes are actively overlapping. Instead, on the first clean frames
    # after separation, clothing + IoU continuity is used to recover the correct per-person state.
    if not bool(getattr(args, "disable_idswitch_fix", False)):
        repair_id_switches_by_iou(
            track_infos=track_infos,
            feat_state=feat_state,
            identity_state=identity_state,
            frame_idx=frame_idx,
            iou_thresh=float(getattr(args, "idswitch_iou", 0.30)),
            max_age=int(getattr(args, "idswitch_max_age", 3)),
            current_colliding_tids=colliding_tids,
            cloth_min_sim=float(getattr(args, "idswitch_cloth_sim", 0.55)),
            cloth_margin=float(getattr(args, "idswitch_cloth_margin", 0.08)),
        )

    # Collision introduces ambiguity.
    # Policy (crowd-safe):
    #   - Body/back embeddings stay frozen during collisions (handled below via embed_freeze).
    #   - If a track is CONFIRMED, keep its identity during collisions (prevents flicker in crowds),
    #     but freeze switching so it cannot jump.
    #   - Otherwise, optionally drop identity on collision (most conservative), or just freeze switching.
    if colliding_tids:
        freeze_frames = int(getattr(args, "collision_freeze_frames", 18))
        for tid in colliding_tids:
            entry = identity_state.setdefault(
                tid,
                {
                    "scores": {},
                    "last": "",
                    "ttl": 0,
                    "freeze": 0,
                    "hold_until": 0.0,
                    "confirmed": False,
                    "confirmed_name": "",
                    "confirmed_src": "",
                },
            )

            last_name = str(entry.get("last", "") or "")
            is_confirmed = bool(entry.get("confirmed", False)) and bool(last_name) and (
                str(entry.get("confirmed_name", "") or "") == last_name
            )

            if is_confirmed and bool(getattr(args, "keep_confirmed_name_on_collision", True)):
                # Keep showing the confirmed identity; just freeze switching for a bit while overlap persists.
                entry["freeze"] = max(int(entry.get("freeze", 0) or 0), freeze_frames)
                # Extend the name hold window slightly (optional stability).
                nhs = float(getattr(args, "name_hold_seconds", 0.0) or 0.0)
                if nhs > 0.0:
                    entry["hold_until"] = max(float(entry.get("hold_until", 0.0) or 0.0), float(now_ts) + nhs)
                continue

            # Non-confirmed: collision cancels any persistence guarantees.
            entry["confirmed"] = False
            entry["confirmed_name"] = ""
            entry["confirmed_src"] = ""

            if last_name:
                if bool(getattr(args, "drop_name_on_collision", True)):
                    # Hide identity immediately (most conservative).
                    entry["last"] = ""
                    entry["ttl"] = 0
                    entry["scores"] = {}
                    entry["freeze"] = 0
                else:
                    # Keep showing but freeze switching.
                    entry["freeze"] = max(int(entry.get("freeze", 0) or 0), freeze_frames)

    # Tracklet params
    beta = float(getattr(args, "tracklet_ema_beta", 0.85))
    min_samples = max(1, int(getattr(args, "tracklet_min_samples", 1)))
    reid_every_n = max(1, int(getattr(args, "reid_every_n", 2)))
    reid_every_n_known = max(1, int(getattr(args, "reid_every_n_known", 6)))

    # Init/update per-track state
    for info in track_infos:
        tid = int(info["tid"])
        st = feat_state.setdefault(
            tid,
            {
                "body_ema": None,
                "back_ema": None,
                "face_ema": None,
                "cloth_ema": None,
                "body_n": 0,
                "back_n": 0,
                "face_n": 0,
                "cloth_n": 0,
                "body_last": -10**9,
                "back_last": -10**9,
                "face_last": -10**9,
                "cloth_last": -10**9,
                "embed_freeze": 0,
                "last_bbox": None,
                "last_seen": frame_idx,
            },
        )
        st["last_seen"] = frame_idx
        st["last_bbox"] = info["bbox"]

        # Cheap clothing memory for crossing/post-occlusion repair.
        cloth_desc = info.get("cloth_desc", None)
        if cloth_desc is not None and tid not in colliding_tids:
            st["cloth_ema"] = _ema_update(st.get("cloth_ema"), cloth_desc, min(0.75, beta))
            st["cloth_n"] = int(st.get("cloth_n", 0)) + 1
            st["cloth_last"] = frame_idx

    
    # NEW: freeze embedding updates during overlaps so we don't "mix" pixels from 2 people
    # into the same ReID/face tracklet (this is one of the main causes of post-collision mislabels).
    emb_freeze_frames = int(getattr(args, "collision_embed_freeze_frames", 0) or 0)
    if emb_freeze_frames > 0 and colliding_tids:
        for _tid in colliding_tids:
            st = feat_state.get(int(_tid))
            if st is not None:
                st["embed_freeze"] = max(int(st.get("embed_freeze", 0) or 0), emb_freeze_frames)

    # Snapshot per-track freeze status for this frame + decrement the counter
    for info in track_infos:
        tid = int(info["tid"])
        st = feat_state.get(tid, {})
        ef = int(st.get("embed_freeze", 0) or 0)
        info["embed_freeze"] = ef
        if ef > 0:
            st["embed_freeze"] = ef - 1

# === Decide which tracks need BODY embedding update this frame ===
    body_update_idx: List[int] = []
    body_crops: List[Optional[np.ndarray]] = []
    for i, info in enumerate(track_infos):
        tid = int(info["tid"])
        st = feat_state.get(tid, {})
        prev_name = identity_state.get(tid, {}).get("last", "")
        interval = reid_every_n_known if prev_name else reid_every_n

        # If tracks are colliding (or just collided), crops can contain mixed pixels -> unstable embeddings.
        # We skip embedding updates for a short cooldown window to avoid post-collision mislabels.
        if int(info.get("embed_freeze", 0) or 0) > 0:
            continue

        if st.get("body_ema") is None or (frame_idx - int(st.get("body_last", -10**9)) >= interval):
            x1i, y1i, x2i, y2i = info["bbox"]
            body_update_idx.append(i)
            body_crops.append(frame[y1i:y2i, x1i:x2i])

    if body_update_idx and reid_extractors:
        new_body_embs = extract_reid_embeddings_batch_ensemble(
            reid_extractors,
            body_crops,
            use_half=use_half,
            batch_size=int(getattr(args, "reid_batch_size", 24)),
        )
        for j, i in enumerate(body_update_idx):
            tid = int(track_infos[i]["tid"])
            st = feat_state[tid]
            emb = new_body_embs[j] if j < len(new_body_embs) else None
            if emb is not None:
                st["body_ema"] = _ema_update(st.get("body_ema"), emb, beta=beta)
                st["body_n"] = int(st.get("body_n", 0)) + 1
                st["body_last"] = frame_idx
                track_infos[i]["body_updated"] = True

    # === Link faces to tracks (IoA) and update FACE EMA when available ===
    # Faces are linked to tracks globally (greedy) earlier; here we only decide whether to update EMA.
    for info in track_infos:
        tid = int(info["tid"])

        cand_link = float(info.get("face_cand_link", 0.0) or 0.0)
        cand_emb = info.get("face_cand_emb", None)

        info["face_link"] = float(cand_link)
        info["face_emb"] = cand_emb

        # NOTE: During collisions we do NOT update the face EMA (prevents wrong face->track mixing).
        if int(info.get("embed_freeze", 0) or 0) > 0:
            continue

        if cand_emb is not None and cand_link >= float(args.face_iou_link):
            info["face_updated"] = True

            st = feat_state[tid]
            st["face_ema"] = _ema_update(st.get("face_ema"), cand_emb, beta=beta)
            st["face_n"] = int(st.get("face_n", 0)) + 1
            st["face_last"] = frame_idx

    # Determine whether each track has a usable face tracklet
    for info in track_infos:
        tid = int(info["tid"])
        st = feat_state.get(tid, {})
        face_ok = (st.get("face_ema") is not None) and (int(st.get("face_n", 0)) >= min_samples)
        info["face_ok"] = bool(face_ok)

    # === Decide which tracks need BACK embedding update this frame (only if face isn't available) ===
    back_update_idx: List[int] = []
    back_crops: List[Optional[np.ndarray]] = []
    for i, info in enumerate(track_infos):
        if bool(info.get("face_ok")):
            continue

        tid = int(info["tid"])
        st = feat_state.get(tid, {})
        prev_name = identity_state.get(tid, {}).get("last", "")
        interval = reid_every_n_known if prev_name else reid_every_n

        # If tracks are colliding (or just collided), skip embedding updates to avoid mixed pixels.
        if int(info.get("embed_freeze", 0) or 0) > 0:
            continue

        if st.get("back_ema") is None or (frame_idx - int(st.get("back_last", -10**9)) >= interval):
            x1i, y1i, x2i, y2i = info["bbox"]
            back_crop = crop_back_body_no_face_from_tlbr(
                frame,
                (x1i, y1i, x2i, y2i),
                head_cut_ratio=float(args.back_head_cut),
            )
            back_update_idx.append(i)
            back_crops.append(back_crop)

    if back_update_idx:
        new_back_embs = extract_back_body_embeddings_batch(
            reid_extractors,
            back_crops,
            args=args,
            use_half=use_half,
        )
        for j, i in enumerate(back_update_idx):
            tid = int(track_infos[i]["tid"])
            st = feat_state[tid]
            emb = new_back_embs[j] if j < len(new_back_embs) else None
            if emb is not None:
                st["back_ema"] = _ema_update(st.get("back_ema"), emb, beta=beta)
                st["back_n"] = int(st.get("back_n", 0)) + 1
                st["back_last"] = frame_idx
                track_infos[i]["back_updated"] = True

    # === Match + identity smoothing ===
    # Names currently held by visible tracks (before updates).
    # Used to prevent collision-face from stealing a name from another on-screen track.
    active_name_to_tids: Dict[str, set] = {}
    for _info in track_infos:
        _tid = int(_info["tid"])
        _nm = identity_state.get(_tid, {}).get("last", "")
        if _nm:
            active_name_to_tids.setdefault(str(_nm), set()).add(_tid)

    track_rows: List[Dict] = []
    for info in track_infos:
        tid = int(info["tid"])
        x1i, y1i, x2i, y2i = info["bbox"]
        det_conf = float(info["det_conf"])

        # Snapshot the current stable name on this track (before we update it this frame).
        # POLICY: Face is the ONLY modality allowed to *identify* (assign/switch) an identity.
        # Body/back may only SUPPORT the already-selected face identity (continuity), never create/switch by themselves.
        prev_name_for_row = identity_state.get(tid, {}).get("last", "")

        freeze_active = False
        try:
            ent = identity_state.get(tid, {})
            freeze_active = (int(ent.get("freeze", 0) or 0) > 0) and bool(ent.get("last", ""))
        except Exception:
            freeze_active = False

        # When tracks are colliding (or just collided), we treat it as ambiguous.
        # We still allow FACE confirmation, but we avoid BODY/BACK re-id in this window.
        ambig_freeze = int(info.get("embed_freeze", 0) or 0) > 0

        st = feat_state.get(tid, {})
        face_ok = bool(info.get("face_ok"))
        body_ok = (st.get("body_ema") is not None) and (int(st.get("body_n", 0)) >= min_samples)
        back_ok = (st.get("back_ema") is not None) and (int(st.get("back_n", 0)) >= min_samples)

        # Only add votes when we actually updated that modality on this frame.
        # (Tracklet/EMA provides the stability; this reduces overcounting and saves compute.)
        candidates: List[Tuple[str, float, str]] = []
        chosen_face = False
        chosen_body = False
        chosen_back = False
        instant_score = 0.0

        face_label = None
        face_score = 0.0
        face_second = 0.0
        face_conf = 0.0
        face_vote = 0.0
        face_src = ""

        if (not freeze_active) and gallery.people:
            if not ambig_freeze:
                # Normal (non-collision) face vote: only when we updated the face EMA this frame.
                if bool(info.get("face_updated")) and face_ok:
                    face_label, face_score, face_second = best_face_label_top2(st.get("face_ema"), gallery)
                    if face_label is not None and float(face_score) >= float(args.face_thresh):
                        face_conf = _gap_conf(face_score, face_second, float(args.face_gap))
                        face_vote = float(face_score) * float(face_conf)
                        if face_vote > 0.0:
                            candidates.append((str(face_label), float(face_vote), "face"))
                            chosen_face = True
                            face_src = "face"
                            instant_score = max(instant_score, float(face_vote))

            else:
                # Collision-aware face identification:
                #   - body/back embeddings remain frozen in this window
                #   - face is allowed only if the face->track link is UNAMBIGUOUS and match is VERY strong
                if bool(getattr(args, "collision_face_enable", True)):
                    cand_emb = info.get("face_cand_emb", None)
                    cand_link = float(info.get("face_cand_link", 0.0) or 0.0)
                    cand_other = float(info.get("face_cand_othermax", 0.0) or 0.0)

                    if cand_emb is not None:
                        min_ioa = float(getattr(args, "collision_face_ioa", 0.85) or 0.85)
                        min_ioa_gap = float(getattr(args, "collision_face_ioa_gap", 0.20) or 0.20)

                        # Geometry gate: face must belong to ONLY this box (not inside neighbors)
                        if (cand_link >= min_ioa) and ((cand_link - cand_other) >= min_ioa_gap):
                            face_label, face_score, face_second = best_face_label_top2(cand_emb, gallery)

                            if face_label is not None and float(face_score) >= float(
                                getattr(args, "collision_face_thresh", 0.60) or 0.60
                            ):
                                face_conf = _gap_conf(face_score, face_second, float(args.face_gap))
                                if float(face_conf) >= float(getattr(args, "collision_face_gap_conf", 0.95) or 0.95):
                                    face_vote = float(face_score) * float(face_conf)
                                    if face_vote > 0.0:
                                        # Update collision-face streak (needs persistence before NEW acquisition)
                                        ent_cf = identity_state.setdefault(
                                            tid,
                                            {
                                                "scores": {},
                                                "last": "",
                                                "ttl": 0,
                                                "freeze": 0,
                                                "hold_until": 0.0,
                                                "confirmed": False,
                                                "confirmed_name": "",
                                                "confirmed_src": "",
                                            },
                                        )

                                        lab = str(face_label)
                                        streak_timeout = max(2, int(getattr(args, "face_every_n", 2) or 2) * 3)
                                        last_frame = int(ent_cf.get("coll_face_last_frame", -10**9) or -10**9)
                                        if (frame_idx - last_frame) > streak_timeout:
                                            ent_cf["coll_face_label"] = ""
                                            ent_cf["coll_face_streak"] = 0

                                        if lab == str(ent_cf.get("coll_face_label", "") or ""):
                                            ent_cf["coll_face_streak"] = int(ent_cf.get("coll_face_streak", 0) or 0) + 1
                                        else:
                                            ent_cf["coll_face_label"] = lab
                                            ent_cf["coll_face_streak"] = 1
                                        ent_cf["coll_face_last_frame"] = int(frame_idx)

                                        # Decide if this collision-face match is allowed to vote
                                        allow_vote = False

                                        if prev_name_for_row:
                                            # Never switch identities during collisions; only refresh SAME name.
                                            allow_vote = (lab == str(prev_name_for_row))
                                        else:
                                            # Block if another visible track already owns this name
                                            owners = active_name_to_tids.get(lab, set())
                                            if owners and (tid not in owners):
                                                allow_vote = False
                                            else:
                                                # Collision-face must NEVER move a name from one track_id to another.
                                                # Only allow if the name is either unseen in the registry, or already bound to THIS tid.
                                                reg = name_registry.get(lab) if name_registry else None
                                                if reg is not None:
                                                    reg_tid = int(reg.get("tid", -1) or -1)
                                                    allow_vote = (reg_tid == tid)
                                                else:
                                                    allow_vote = True

                                                if allow_vote:
                                                    req = int(getattr(args, "collision_face_confirm_frames", 2) or 2)
                                                    # If registry already associates this name with THIS tid, allow immediate reacquire.
                                                    if reg is not None and int(reg.get("tid", -1) or -1) == tid:
                                                        req = 1
                                                    if int(ent_cf.get("coll_face_streak", 0) or 0) < max(1, req):
                                                        allow_vote = False

                                        if allow_vote:
                                            candidates.append((lab, float(face_vote), "face_collision"))
                                            chosen_face = True
                                            face_src = "face_collision"
                                            instant_score = max(instant_score, float(face_vote))

                # If no recent strong collision-face, reset streak after a short timeout
                ent_cf = identity_state.get(tid, None)
                if ent_cf is not None:
                    streak_timeout = max(2, int(getattr(args, "face_every_n", 2) or 2) * 3)
                    last_frame = int(ent_cf.get("coll_face_last_frame", -10**9) or -10**9)
                    if (frame_idx - last_frame) > streak_timeout:
                        ent_cf["coll_face_label"] = ""
                        ent_cf["coll_face_streak"] = 0
        # Support target:
        #   - If we have a valid face vote this frame, body/back may support ONLY that face label.
        #   - Otherwise, body/back may support ONLY the previously-known name (continuity).
        #   - If there is no face vote and no previous name, body/back MUST NOT propose any identity.
        support_target = str(face_label) if (chosen_face and face_label) else (str(prev_name_for_row) if prev_name_for_row else "")

        back_label = None
        back_score = 0.0
        back_second = 0.0
        back_vote = 0.0
        if (not freeze_active) and (not ambig_freeze) and (not face_ok) and bool(info.get("back_updated")) and back_ok and gallery.people and support_target:
            back_label, back_score, back_second = best_back_label_from_emb(st.get("back_ema"), gallery, topk=args.back_topk)
            if back_label is not None and back_score >= float(args.back_thresh):
                conf = _gap_conf(back_score, back_second, float(args.back_gap))
                back_vote = float(back_score) * conf

        body_label = None
        body_score = 0.0
        body_second = 0.0
        body_conf = 0.0
        body_vote = 0.0
        if (not freeze_active) and (not ambig_freeze) and bool(info.get("body_updated")) and body_ok and gallery.people and support_target:
            # Body can ONLY support the already-selected identity (face label this frame OR previous name).
            # It MUST NOT introduce a new identity or switch to a different one.
            body_label, body_score, body_second = best_body_label_from_emb(st.get("body_ema"), gallery, topk=args.body_topk)
            if body_label is not None and body_score >= float(args.body_thresh) and (str(body_label) == str(support_target)):
                body_conf = _gap_conf(body_score, body_second, float(args.body_gap))
                body_vote = float(body_score) * float(body_conf)
                if body_vote > 0.0:
                    candidates.append((str(support_target), float(body_vote), "body"))
                    chosen_body = True
                    instant_score = max(instant_score, float(body_vote))

        # Anti-jump: if a name was just seen on another track, don't let it "hop"
        # unless this track is extremely likely to be the same person (IoU + embedding sim).
        candidates = filter_candidates_by_name_registry(
            candidates=candidates,
            tid=tid,
            bbox_xyxy=(float(x1i), float(y1i), float(x2i), float(y2i)),
            feat_state_for_tid=st,
            name_registry=name_registry,
            now_ts=now_ts,
            reserve_seconds=float(getattr(args, "name_reserve_seconds", 1.0) or 1.0),
            transfer_iou=float(getattr(args, "name_transfer_iou", 0.30) or 0.30),
            transfer_sim=float(getattr(args, "name_transfer_sim", 0.65) or 0.65),
        )

        # Back-only policy (support-only):
        # - Back embedding is WEAK -> it must NEVER create a new identity claim.
        # - Back may ONLY support the already-selected identity (face label this frame OR previous name).
        if back_label and (back_vote > 0.0) and support_target:
            if str(back_label) == str(support_target):
                candidates.append((str(support_target), float(back_vote), "back"))
                chosen_back = True
                instant_score = max(instant_score, float(back_vote))



        # Confirmed-mode guard:
        # If this track was confirmed by STRONG evidence recently, we treat back-only as
        # identity continuity (keep the same name) and do NOT allow weak cues (body/back)
        # to switch it to a different identity. Only a strong FACE match may override.
        confirmed_mode = False
        try:
            ent_prev = identity_state.get(tid, {})
            confirmed_mode = (
                bool(getattr(args, "persist_confirmed_until_collision", True))
                and bool(prev_name_for_row)
                and bool(ent_prev.get("confirmed", False))
                and (str(ent_prev.get("confirmed_name", "")) == str(prev_name_for_row))
            )
        except Exception:
            confirmed_mode = False

        if confirmed_mode and prev_name_for_row:
            gap_min = float(getattr(args, "confirm_gap_min", 0.85))
            allow_face_switch = (
                bool(face_label)
                and (str(face_label) != str(prev_name_for_row))
                and (float(face_score) >= float(args.face_thresh))
                and (float(face_conf) >= gap_min)
            )

            filtered: List[Tuple[str, float, str]] = []
            for (lbl, v, src) in candidates:
                if str(lbl) == str(prev_name_for_row):
                    filtered.append((str(lbl), float(v), str(src)))
                elif (str(src) == "face") and allow_face_switch and (str(lbl) == str(face_label)):
                    filtered.append((str(lbl), float(v), str(src)))
            candidates = filtered
        stable_name, stable_accum = update_track_identity(
            identity_state,
            tid,
            candidates,
            decay=args.name_decay,
            min_score=args.name_min_score,
            margin=args.name_margin,
            ttl_reset=args.name_ttl,
            w_face=args.name_face_weight,
            w_body=args.name_body_weight,
            w_back=args.name_back_weight,
            now_ts=now_ts,
            hold_seconds=float(getattr(args, "name_hold_seconds", 1.0) or 0.0),
        )



        # Confirmed-mode SET/REFRESH:
        # If we see strong FACE evidence for the current stable identity, mark this track as confirmed.
        # While confirmed, TTL/decay cannot erase the name unless ambiguity (collision) happens.
        if bool(getattr(args, "persist_confirmed_until_collision", True)):
            try:
                ent_now = identity_state.get(tid, {})
                if stable_name:
                    gap_min = float(getattr(args, "confirm_gap_min", 0.85))
                    # Face-only identification policy: ONLY a strong FACE match may mark a track as CONFIRMED.
                    strong_face = (
                        bool(face_label)
                        and (float(face_score) >= float(args.face_thresh))
                        and (float(face_conf) >= gap_min)
                        and (str(face_label) == str(stable_name))
                    )
                    if strong_face:
                        ent_now["confirmed"] = True
                        ent_now["confirmed_name"] = str(stable_name)
                        ent_now["confirmed_src"] = "face"
                    # If confirmation is stale for some other name, clear it.
                    if bool(ent_now.get("confirmed", False)) and str(ent_now.get("confirmed_name", "")) and (str(ent_now.get("confirmed_name", "")) != str(stable_name)):
                        ent_now["confirmed"] = False
                        ent_now["confirmed_name"] = ""
                        ent_now["confirmed_src"] = ""
                else:
                    if ent_now:
                        ent_now["confirmed"] = False
                        ent_now["confirmed_name"] = ""
                        ent_now["confirmed_src"] = ""
            except Exception:
                pass
        track_rows.append(
            {
                "tid": tid,
                "prev_name": prev_name_for_row,
                "bbox": (x1i, y1i, x2i, y2i),
                "det_conf": det_conf,
                "stable_name": stable_name,
                "stable_accum": float(stable_accum),
                "instant_score": float(instant_score),
                "chosen_face": chosen_face,
                "face_src": face_src,
                "chosen_body": chosen_body,
                "chosen_back": chosen_back,
                "face_score": float(face_score),
                "body_score": float(body_score),
                "back_score": float(back_score),
                "face_link": float(info.get("face_link", 0.0)),
                "face_othermax": float(info.get("face_cand_othermax", 0.0)),
                "embed_freeze": int(info.get("embed_freeze", 0) or 0),
                "colliding": bool(tid in colliding_tids),
                "face_label": str(face_label) if face_label else "",
                "face_conf": float(face_conf),
                "face_vote": float(face_vote),
                "reclaimed_by_face": False,
                "face_captured": info.get("face_emb") is not None,
            }
        )

    if not args.allow_duplicate_names:
        # NEW: strong-face reclaim.
        # If a face is strongly identified as a label, that label must belong to that track,
        # and must be removed from any other visible track immediately.
        apply_strong_face_reclaim(
            identity_state=identity_state,
            track_rows=track_rows,
            args=args,
            now_ts=now_ts,
            name_registry=name_registry,
        )

        # After reclaim, still enforce uniqueness as a final guardrail.
        enforce_unique_names(identity_state, track_rows)


    # Cache a per-track confidence number for display (name + confidence only; no bounding boxes).
    # We keep the last "good" confidence per track so it persists when the face is temporarily not visible.
    for r in track_rows:
        tid = int(r.get("tid", -1))
        if tid < 0:
            continue

        stable_name = str(r.get("stable_name", "") or "")
        if not stable_name:
            # Unknown: show the current best similarity as "confidence" (no persistence).
            try:
                identity_state.get(tid, {}).pop("last_conf", None)
            except Exception:
                pass

            conf = 0.0
            try:
                conf = max(
                    float(r.get("face_score", 0.0) or 0.0),
                    float(r.get("body_score", 0.0) or 0.0),
                    float(r.get("back_score", 0.0) or 0.0),
                )
            except Exception:
                conf = 0.0

            r["display_conf"] = float(conf)
            continue

        prev_conf = 0.0
        try:
            prev_conf = float(identity_state.get(tid, {}).get("last_conf", 0.0) or 0.0)
        except Exception:
            prev_conf = 0.0

        conf = prev_conf
        try:
            if bool(r.get("chosen_face")) and float(r.get("face_score", 0.0) or 0.0) > 0.0:
                conf = float(r.get("face_score", 0.0) or 0.0)
            elif bool(r.get("chosen_back")) and float(r.get("back_score", 0.0) or 0.0) > 0.0:
                conf = float(r.get("back_score", 0.0) or 0.0)
            elif bool(r.get("chosen_body")) and float(r.get("body_score", 0.0) or 0.0) > 0.0:
                conf = float(r.get("body_score", 0.0) or 0.0)
        except Exception:
            conf = prev_conf

        try:
            identity_state[tid]["last_conf"] = float(conf)
        except Exception:
            pass
        r["display_conf"] = float(conf)
    # Update per-name reservation with the FINAL stable names for this frame.
    # This is what prevents: A (known) + B (unknown) collide, A leaves, B suddenly becomes "A".
    for r in track_rows:
        nm = r.get("stable_name", "")
        if not nm:
            continue
        tid = int(r.get("tid", -1))
        if tid in suppress_track_draw_tids:
            continue
        st = feat_state.get(tid, {})
        update_name_registry_for_track(
            name_registry=name_registry,
            stable_name=str(nm),
            tid=tid,
            bbox_xyxy=r.get("bbox", (0, 0, 0, 0)),
            feat_state_for_tid=st,
            now_ts=now_ts,
        )

    # === Analytics ingest: whenever the green box (known ID) is shown ===
    # We use capture timestamp (cap_ts) so cross-camera ordering is consistent.
    if room_tracker is not None and cap_ts is not None and room_id:
        try:
            tsf = float(cap_ts)
            for r in track_rows:
                tid = int(r.get("tid", -1))
                if tid in suppress_track_draw_tids:
                    continue
                nm = r.get("stable_name", "")
                if nm:
                    room_tracker.ingest(str(nm), str(room_id), tsf)
            for fr in crowd_face_rows:
                nm = str(fr.get("label", "") or "")
                if nm:
                    room_tracker.ingest(str(nm), str(room_id), tsf)
        except Exception:
            pass


    # === Draw: show name + confidence only (NO bounding boxes) ===
    for r in track_rows:
        stable_name = str(r.get("stable_name", "") or "")
        disp_name = stable_name if stable_name else "Unknown"
        tid = int(r.get("tid", -1))
        if tid in suppress_track_draw_tids:
            continue

        x1i, y1i, x2i, y2i = r["bbox"]

        # Determine color based on recognition state
        if not stable_name:
            if r.get("face_captured", False):
                color = (0, 165, 255)  # Orange for unknown but face captured
            else:
                color = (0, 0, 255)  # Red for unknown
        else:
            # If confirmed/assigned this frame by ReID or face match
            if r.get("chosen_face") or r.get("chosen_body") or r.get("chosen_back") or r.get("reclaimed_by_face"):
                color = (0, 255, 0)  # Green
            else:
                color = (255, 0, 0)  # Blue (name held by tracking)

        conf = float(r.get("display_conf", 0.0) or 0.0)
        label_txt = f"{disp_name} #{tid} {conf:.2f}"

        cv2.putText(
            out,
            label_txt,
            (int(x1i), max(0, int(y1i) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )

    # In crowded collision regions, label the FACE itself and do not force a body-track match.
    for fr in crowd_face_rows:
        label_txt = str(fr.get("label", "") or "")
        if not label_txt:
            continue
        try:
            fx1, fy1, fx2, fy2 = fr.get("bbox", (0.0, 0.0, 0.0, 0.0))
        except Exception:
            continue
        cv2.putText(
            out,
            label_txt,
            (int(fx1), max(0, int(fy1) - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    return out


def processor_thread(
    sid: int,
    vs: VideoStream,
    yolo,
    reid_extractors: List[TorchreidExtractor],
    face_app,
    deep_tracker,
    gallery_ref: Dict[str, object],
    args,
    room_id: str,
    room_tracker: Optional["SpatioTemporalRoomTracker"],
    render_q: Optional["queue.Queue"],
    render_map: Dict[int, np.ndarray],
    render_lock: threading.Lock,
    stop_flag: threading.Event,
):
    frame_idx = 0
    identity_state: Dict[int, Dict] = {}
    feat_state: Dict[int, Dict] = {}
    name_registry: Dict[str, Dict] = {}

    if bool(getattr(args, "use_strongsort", False) or getattr(args, "use_deepsort", False)):
        deep_tracker = create_strongsort_tracker(args)
    else:
        deep_tracker = None

    fallback_tracker = None if deep_tracker is not None else SimpleIoUTracker(
        max_age=args.max_age,
        n_init=args.n_init,
        iou_threshold=FALLBACK_IOU_THRESH,
    )

    last_db_reload = 0.0

    # Per-source processing FPS (EMA)
    fps_ema = 0.0
    last_ts = time.time()

    # FPS pacing: keep per-source processing at a stable rate (prevents FPS spikes/jitter).
    proc_fps = float(getattr(args, "proc_fps", 0.0) or 0.0)
    proc_period = (1.0 / proc_fps) if proc_fps > 0.0 else 0.0
    next_proc_tick = time.monotonic()
    # Stream stall watchdog (per source)
    last_good_mono = time.monotonic()
    last_restart_req_mono = 0.0
    last_seen_restart_count = 0


    while not stop_flag.is_set():
        # If we can run faster than target FPS, sleep so output FPS stays stable.
        if proc_period > 0.0:
            now_m = time.monotonic()
            if now_m < next_proc_tick:
                time.sleep(next_proc_tick - now_m)
            else:
                # If we stalled, re-sync so we don't burst frames.
                if (now_m - next_proc_tick) > (2.0 * proc_period):
                    next_proc_tick = now_m
        ok, frame, cap_ts = vs.read(timeout=0.5)
        if not ok or frame is None:
            # If we stop receiving frames, request a restart so we don't need to relaunch the script.
            stall_s = float(max(0.0, time.monotonic() - float(last_good_mono)))
            stall_thr = float(getattr(args, "stream_stall_seconds", 8.0) or 0.0)
            backoff = float(getattr(args, "stream_reconnect_backoff", 2.0) or 0.0)
            if stall_thr > 0.0 and stall_s >= stall_thr:
                if (time.monotonic() - float(last_restart_req_mono)) >= max(0.25, backoff):
                    print(f"[SRC {sid}] Stream stalled for {stall_s:.1f}s -> requesting reconnect")
                    try:
                        vs.request_restart()
                    except Exception:
                        pass
                    last_restart_req_mono = time.monotonic()
            time.sleep(0.01)
            continue

        last_good_mono = time.monotonic()

        if proc_period > 0.0:
            # Drop buffered frames so we always process the most recent frame available at this tick.
            while True:
                ok2, frame2, cap_ts2 = vs.read(timeout=0.0)
                if not ok2 or frame2 is None:
                    break
                frame, cap_ts = frame2, cap_ts2


        # Detect capture reconnects and reset local states to avoid stale track IDs.
        try:
            _rc = int(vs.get_stats().get("restarts", 0) or 0)
        except Exception:
            _rc = int(last_seen_restart_count)

        if _rc != int(last_seen_restart_count):
            print(f"[SRC {sid}] Stream reconnected (count={_rc}). Resetting local states.")
            last_seen_restart_count = _rc
            identity_state.clear()
            feat_state.clear()
            name_registry.clear()
            if bool(getattr(args, "use_strongsort", False) or getattr(args, "use_deepsort", False)):
                deep_tracker = create_strongsort_tracker(args)
            else:
                deep_tracker = None

            if deep_tracker is None:
                fallback_tracker = SimpleIoUTracker(
                    max_age=args.max_age,
                    n_init=args.n_init,
                    iou_threshold=FALLBACK_IOU_THRESH,
                )
            else:
                fallback_tracker = None

        if args.use_db and args.db_refresh_seconds and args.db_refresh_seconds > 0:
            now = time.time()
            if now - last_db_reload >= float(args.db_refresh_seconds):
                try:
                    gallery_ref["gallery"] = build_gallery_from_db(args.db_url)
                    last_db_reload = now
                except Exception as e:
                    print("[DB] reload failed:", e)

        gallery: TwoStageGallery = gallery_ref["gallery"]

        out = process_one_frame(
            frame_idx=frame_idx,
            frame=frame,
            sid=sid,
            yolo=yolo,
            reid_extractors=reid_extractors,
            face_app=face_app,
            deep_tracker=deep_tracker,
            fallback_tracker=fallback_tracker,
            gallery=gallery,
            args=args,
            identity_state=identity_state,
            feat_state=feat_state,
            name_registry=name_registry,
            cap_ts=cap_ts,
            room_id=str(room_id),
            room_tracker=room_tracker,
        )
        frame_idx += 1

        # Update FPS EMA
        now = time.time()
        dt = now - last_ts
        last_ts = now
        if dt > 1e-6:
            inst = 1.0 / dt
            fps_ema = inst if fps_ema <= 0.0 else (0.9 * fps_ema + 0.1 * inst)

        # Stream / buffering stats
        stats = vs.get_stats()
        lag_ms = 0.0
        if cap_ts is not None:
            lag_ms = max(0.0, (time.time() - float(cap_ts)) * 1000.0)

        cv2.putText(
            out,
            f"SRC {sid} FPS: {fps_ema:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            out,
            f"Q:{int(stats['qlen'])}  drop:{int(stats['dropped_total'])}  lag:{lag_ms:.0f}ms",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        # Prune old per-track state (prevents unbounded growth)
        prune_after = int(getattr(args, "tracklet_prune_after", 0) or 0)
        if prune_after > 0 and (frame_idx % 30 == 0):
            stale = [
                tid
                for tid, st in feat_state.items()
                if (frame_idx - int(st.get("last_seen", frame_idx))) > prune_after
            ]
            for tid in stale:
                feat_state.pop(tid, None)
                identity_state.pop(tid, None)

        # If we are displaying, push every processed frame into the display queue.
        # The queue is bounded -> if display is slower, processing will block here (backpressure),
        # which guarantees: frames processed == frames displayed.
        if render_q is not None:
            while not stop_flag.is_set():
                try:
                    render_q.put((sid, out), timeout=0.1)
                    break
                except queue.Full:
                    continue
        else:
            with render_lock:
                render_map[sid] = out

        # Advance pacing clock (done after queue push so we account for any backpressure delays).
        if proc_period > 0.0:
            next_proc_tick += proc_period
            now_m = time.monotonic()
            # If we fell behind (e.g., slow frame), re-sync so we don't burst.
            if next_proc_tick < (now_m - 0.5 * proc_period):
                next_proc_tick = now_m + proc_period


def _legacy_local_main():
    args = parse_args()

    # CUDA / GPU speed knobs (safe no-ops on CPU)
    try:
        cv2.setUseOptimized(True)
    except Exception:
        pass

    if torch.cuda.is_available() and ("cuda" in str(args.device).lower()):
        try:
            torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if args.use_db:
        if not args.db_url:
            raise SystemExit("--use-db requires --db-url")
        gallery = build_gallery_from_db(args.db_url)
    else:
        raise SystemExit("This script version is DB-first. Use: --use-db --db-url ...")

    yolo, reid_extractors, face_app, deep_tracker = init_models(args)

    streams: List[Tuple[int, VideoStream]] = []
    for i, src in enumerate(args.src):
        vs = VideoStream(
            src,
            rtsp_buffer=args.rtsp_buffer,
            queue_size=args.queue_size,
            max_queue_age_ms=args.max_queue_age_ms,
            grab_skip=args.grab_skip,
            stall_seconds=args.stream_stall_seconds,
            reconnect_backoff_seconds=args.stream_reconnect_backoff,
            open_timeout_ms=args.stream_open_timeout_ms,
            read_timeout_ms=args.stream_read_timeout_ms,
            max_reconnect_tries=args.stream_reconnect_max_tries,
        )
        streams.append((i, vs))

    if not any(vs.ok for _, vs in streams):
        raise SystemExit("No sources opened. Check --src URL/path.")

    render_map: Dict[int, np.ndarray] = {}
    render_lock = threading.Lock()
    stop_flag = threading.Event()

    # --- Room analytics: map each source (sid) to a room id ---
    room_ids = [str(r) for r in (getattr(args, "room_ids", []) or [])]
    if room_ids and len(room_ids) != len(streams):
        print(f"[WARN] --room-ids count ({len(room_ids)}) != number of sources ({len(streams)}). Falling back to c1..cN.")
        room_ids = []
    if not room_ids:
        room_ids = [f"c{i+1}" for i in range(len(streams))]
    sid_to_room = {int(sid): str(room_ids[i]) for i, (sid, _) in enumerate(streams)}

    # --- Analytics tracker (thread-safe) ---
    excel_out_arg = str(getattr(args, "excel_out", "") or "").strip()
    excel_enabled = bool(excel_out_arg) and (excel_out_arg.lower() not in {"off", "none", "0", "false", "disable", "disabled"})

    room_tracker = None
    excel_path = ""

    def _resolve_excel_path(p: str) -> str:
        p = str(p or "").strip()
        if not p:
            p = "room_presence.xlsx"
        # If a directory (or ends with a separator), drop file into it
        if p.endswith(os.sep) or os.path.isdir(p):
            base_dir = p if os.path.isdir(p) else p.rstrip(os.sep)
            if not base_dir:
                base_dir = "."
            os.makedirs(base_dir, exist_ok=True)
            return os.path.join(base_dir, f"room_presence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")

        root, ext = os.path.splitext(p)
        if not ext:
            ext = ".xlsx"
        if os.path.exists(p):
            return f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
        return p

    if excel_enabled:
        if SpatioTemporalRoomTracker is None:
            print("[WARN] room_presence_analytics.py not available; Excel export disabled.")
        else:
            # Build room topology (spatio-temporal context) from --room-graph.
            # Default matches your topology:
            #   c1<->c2, c2<->c3, c2<->c5(no cam), c5<->c4, c3<->c4(far door)
            entry_room = str(getattr(args, "room_entry_id", "c3") or "c3")
            room_graph = parse_room_graph_str(
                str(getattr(args, "room_graph", "") or ""),
                sorted(set(room_ids + [entry_room])),
            )

            topology = None
            if RoomTopology is not None:
                try:
                    camera_rooms = sorted(set(room_ids))
                    all_rooms = set(room_graph.keys())
                    for u, nbs in room_graph.items():
                        for v in nbs:
                            all_rooms.add(v)
                    all_rooms.add(entry_room)

                    blind_rooms = sorted(all_rooms - set(camera_rooms))
                    low_vis_rooms = ["c3"] if "c3" in all_rooms else []

                    topology = RoomTopology(
                        graph=room_graph,
                        camera_rooms=camera_rooms,
                        blind_rooms=blind_rooms,
                        low_visibility_rooms=low_vis_rooms,
                        camera_intermediate_penalty=1.0,
                        low_vis_intermediate_penalty=0.2,
                        edge_cost=1.0,
                    )
                except Exception as e:
                    print("[WARN] Failed to build custom topology from --room-graph; using default topology:", e)
                    topology = None

            room_tracker = SpatioTemporalRoomTracker(
                topology=topology,
                tz=str(getattr(args, "excel_tz", "Asia/Kolkata")),
                raw_merge_gap_seconds=float(getattr(args, "excel_raw_merge_gap", 2.0) or 2.0),
                same_room_gap_fill_seconds=float(getattr(args, "excel_fill_gap", 90.0) or 90.0),
                adjacent_gap_policy=str(getattr(args, "excel_adjacent_gap_policy", "split") or "split"),
                entrance_room=entry_room,
                entry_impute_seconds=float(getattr(args, "excel_entry_impute_seconds", 0.0) or 0.0),
            )

            # Optionally disable intermediate-room imputation by forcing direct paths.
            if bool(getattr(args, "excel_disable_path_impute", False)):
                try:
                    room_tracker.topology.preferred_path = (
                        lambda start, end: [start] if start == end else [start, end]
                    )
                except Exception:
                    pass

            excel_path = _resolve_excel_path(excel_out_arg)
            print(f"[ANALYTICS] Room analytics enabled. Excel will be written to: {excel_path}")

            # Write once immediately so the file exists, then keep updating it.
            try:
                room_tracker.export_excel(excel_path, with_date=bool(getattr(args, "excel_with_date", False)))
                print(f"[ANALYTICS] Initial Excel written: {excel_path}")
            except Exception as e:
                print("[ANALYTICS] Initial export failed:", e)

            export_every = float(getattr(args, "excel_export_every_seconds", 0.0) or 0.0)
            if export_every > 0.0:

                def _export_loop():
                    while not stop_flag.is_set():
                        time.sleep(export_every)
                        if stop_flag.is_set():
                            break
                        try:
                            room_tracker.export_excel(excel_path, with_date=bool(getattr(args, "excel_with_date", False)))
                            print(f"[ANALYTICS] Periodic Excel update: {excel_path}")
                        except Exception as e:
                            print("[ANALYTICS] Periodic export failed:", e)

                threading.Thread(target=_export_loop, daemon=True).start()

# When --show is enabled, we push processed frames through a bounded queue so display consumes
    # exactly what processing produces (no duplicate display, no hidden drops).
    render_q: Optional["queue.Queue"] = None
    if args.show:
        # Small bounded queue -> backpressure (keeps proc==display). Increase if you want more buffering.
        render_q = queue.Queue(maxsize=max(2, int(len(streams) * 2)))


    gallery_ref = {"gallery": gallery}

    threads = []
    for sid, vs in streams:
        t = threading.Thread(
            target=processor_thread,
            args=(
                sid,
                vs,
                yolo,
                reid_extractors,
                face_app,
                deep_tracker,
                gallery_ref,
                args,
                sid_to_room.get(int(sid), f"c{int(sid)+1}"),
                room_tracker,
                render_q,
                render_map,
                render_lock,
                stop_flag,
            ),
            daemon=True,
        )
        t.start()
        threads.append(t)

    win_name = "Face-only ID (centroid) + body/back support"
    if args.show:
        try:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        except Exception:
            pass

    print("[Main] Running. Press 'q' to quit." if args.show else "[Main] Running (no --show). Ctrl+C to stop.")
    latest: Dict[int, np.ndarray] = {}
    disp_fps_ema = 0.0
    disp_last_ts = time.time()

    disp_fps = float(getattr(args, "disp_fps", 0.0) or 0.0)
    if disp_fps <= 0.0:
        disp_fps = float(getattr(args, "proc_fps", 0.0) or 0.0)
    disp_period = (1.0 / disp_fps) if disp_fps > 0.0 else 0.0
    next_disp_tick = time.monotonic()

    try:
        while True:
            if args.show:
                # Pull at most one item with a short timeout to keep the UI responsive.
                if render_q is not None:
                    timeout = 0.5
                    if disp_period > 0.0:
                        now_m = time.monotonic()
                        timeout = max(0.0, next_disp_tick - now_m)
                        timeout = min(0.05, timeout)
                    try:
                        sid_got, frame_got = render_q.get(timeout=timeout)
                        latest[int(sid_got)] = frame_got
                    except queue.Empty:
                        pass

                should_render = True
                if disp_period > 0.0:
                    should_render = (time.monotonic() >= next_disp_tick)

                if should_render:
                    # Drain any queued frames so we always render the newest available.
                    if render_q is not None:
                        while True:
                            try:
                                sid_got, frame_got = render_q.get_nowait()
                                latest[int(sid_got)] = frame_got
                            except queue.Empty:
                                break

                    frames = [latest.get(sid) for sid, _ in streams]
                    frames = [f for f in frames if f is not None]

                    if frames:
                        target_h = min(f.shape[0] for f in frames)
                        scaled = []
                        for f in frames:
                            h, w = f.shape[:2]
                            if h != target_h:
                                new_w = int(w * (target_h / h))
                                # Use high-quality INTER_AREA for downscaling
                                interp = cv2.INTER_AREA if target_h < h else cv2.INTER_LINEAR
                                f = cv2.resize(f, (new_w, target_h), interpolation=interp)
                            scaled.append(f)
                        
                        # Create grid layout: for N cameras, arrange in a roughly square grid
                        if len(scaled) > 1:
                            import math
                            num_cams = len(scaled)
                            cols = math.ceil(math.sqrt(num_cams))
                            rows = math.ceil(num_cams / cols)
                            
                            # Build grid row by row
                            grid_rows = []
                            max_row_width = 0
                            for r in range(rows):
                                row_frames = []
                                for c in range(cols):
                                    idx = r * cols + c
                                    if idx < len(scaled):
                                        row_frames.append(scaled[idx])
                                if row_frames:
                                    row_vis = np.concatenate(row_frames, axis=1)
                                    max_row_width = max(max_row_width, row_vis.shape[1])
                                    grid_rows.append(row_vis)
                            
                            # Pad all rows to the same width and concatenate
                            padded_rows = []
                            for row_vis in grid_rows:
                                if row_vis.shape[1] < max_row_width:
                                    pad_width = max_row_width - row_vis.shape[1]
                                    padding = np.zeros((row_vis.shape[0], pad_width, 3), dtype=row_vis.dtype)
                                    row_vis = np.concatenate([row_vis, padding], axis=1)
                                padded_rows.append(row_vis)
                            
                            vis = np.concatenate(padded_rows, axis=0) if padded_rows else scaled[0]
                        else:
                            vis = scaled[0]

                        # Display FPS (counts only renders, not queue reads)
                        now = time.time()
                        dt = now - disp_last_ts
                        disp_last_ts = now
                        if dt > 1e-6:
                            inst = 1.0 / dt
                            disp_fps_ema = inst if disp_fps_ema <= 0.0 else (0.9 * disp_fps_ema + 0.1 * inst)

                        if disp_period > 0.0 and disp_fps > 0.0:
                            txt = f"DISPLAY FPS (locked): {disp_fps_ema:.1f} / target {disp_fps:.1f}"
                        else:
                            txt = f"DISPLAY FPS: {disp_fps_ema:.1f}"
                        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        x = max(10, vis.shape[1] - tw - 10)
                        cv2.putText(vis, txt, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                        # Scale down the display so it fits on screen (processing uses full resolution).
                        disp_vis = vis
                        try:
                            ds = float(getattr(args, "display_scale", 0.0) or 0.0)
                        except Exception:
                            ds = 0.0
                        try:
                            mw, mh = getattr(args, "win_max_wh", [0, 0])
                            mw, mh = int(mw), int(mh)
                        except Exception:
                            mw, mh = 0, 0
                        
                        if ds > 0.0:
                            s = max(0.05, min(1.0, ds))
                        elif mw > 0 and mh > 0:
                            s = min(
                                float(mw) / max(1.0, disp_vis.shape[1]),
                                float(mh) / max(1.0, disp_vis.shape[0]),
                                1.0,
                            )
                        else:
                            s = 1.0
                        
                        if s < 0.999:
                            new_w = max(1, int(disp_vis.shape[1] * s))
                            new_h = max(1, int(disp_vis.shape[0] * s))
                            # Use INTER_AREA for downscaling, INTER_CUBIC for upscaling
                            interp = cv2.INTER_AREA if s < 1.0 else cv2.INTER_CUBIC
                            disp_vis = cv2.resize(disp_vis, (new_w, new_h), interpolation=interp)
                        cv2.imshow(win_name, disp_vis)

                    # Advance render tick
                    if disp_period > 0.0:
                        next_disp_tick += disp_period
                        now_m = time.monotonic()
                        if next_disp_tick < (now_m - 0.5 * disp_period):
                            next_disp_tick = now_m + disp_period

                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            else:
                time.sleep(0.25)
    except KeyboardInterrupt:
        pass
    finally:
        stop_flag.set()

        # Allow worker threads to flush their last few frames/analytics.
        try:
            for t in threads:
                try:
                    t.join(timeout=1.0)
                except Exception:
                    pass
        except Exception:
            pass

        # Write Excel one final time on exit.
        if room_tracker is not None and excel_path:
            try:
                room_tracker.export_excel(excel_path, with_date=bool(getattr(args, "excel_with_date", False)))
                print(f"[ANALYTICS] Final Excel written: {excel_path}")
            except Exception as e:
                print("[ANALYTICS] Final export failed:", e)

        for _, vs in streams:
            vs.release()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        print("Done.")



class RenderedFrame:
    """Thread-safe latest-frame store with optional cached JPEG encoding for MJPEG/web streaming."""

    def __init__(self):
        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._frm: Optional[np.ndarray] = None
        self._ts: float = 0.0
        self._meta: Dict[str, Any] = {}
        self._seq: int = 0
        self._jpeg: Optional[bytes] = None
        self._jpeg_ts: float = 0.0
        self._encode_lock = threading.Lock()
        self._clients: int = 0

    def add_client(self) -> None:
        with self._lock:
            self._clients += 1

    def remove_client(self) -> None:
        with self._lock:
            self._clients = max(0, self._clients - 1)

    def set(self, frame: np.ndarray, meta: Optional[Dict[str, Any]] = None) -> None:
        with self._cond:
            self._frm = frame
            self._ts = time.time()
            self._meta = dict(meta or {})
            self._seq += 1
            self._jpeg = None
            self._jpeg_ts = 0.0
            self._cond.notify_all()

    def get(self) -> Tuple[Optional[np.ndarray], float, Dict[str, Any]]:
        with self._lock:
            return self._frm, float(self._ts), dict(self._meta)

    def wait_for_seq(self, last_seq: int, timeout: float = 0.5) -> Tuple[Optional[np.ndarray], float, Dict[str, Any], int]:
        last_seq = int(last_seq)
        with self._cond:
            if self._seq <= last_seq:
                self._cond.wait(timeout=float(timeout))
            return self._frm, float(self._ts), dict(self._meta), int(self._seq)

    def wait_jpeg(self, last_ts: float, timeout: float = 0.5, jpeg_quality: int = 80) -> Tuple[Optional[bytes], float]:
        last_ts = float(last_ts)
        with self._cond:
            if self._ts <= last_ts:
                self._cond.wait(timeout=float(timeout))
            ts = float(self._ts)
            frame = self._frm
            clients = int(self._clients)
        if frame is None or ts <= 0.0:
            return None, 0.0
        if clients <= 0:
            return None, ts
        with self._lock:
            if self._jpeg is not None and float(self._jpeg_ts) == ts:
                return self._jpeg, ts
        with self._encode_lock:
            with self._lock:
                if self._jpeg is not None and float(self._jpeg_ts) == ts:
                    return self._jpeg, ts
                frame_ref = self._frm
                ts_ref = float(self._ts)
            if frame_ref is None or ts_ref <= 0.0:
                return None, 0.0
            q = int(max(30, min(95, int(jpeg_quality))))
            ok, enc = cv2.imencode('.jpg', frame_ref, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                return None, ts_ref
            jpg = enc.tobytes()
            with self._lock:
                if float(self._ts) == ts_ref:
                    self._jpeg = jpg
                    self._jpeg_ts = ts_ref
            return jpg, ts_ref


def parse_pipeline_args(pipeline_args: Optional[str]) -> argparse.Namespace:
    s = str(pipeline_args or '').strip()
    argv = shlex.split(s) if s else []
    return parse_args(argv=argv, allow_unknown=True)


def _resolve_excel_output_path(path_like: str) -> str:
    p = str(path_like or '').strip()
    if not p:
        p = 'room_presence.xlsx'
    if p.endswith(os.sep) or os.path.isdir(p):
        base_dir = p if os.path.isdir(p) else p.rstrip(os.sep)
        if not base_dir:
            base_dir = '.'
        os.makedirs(base_dir, exist_ok=True)
        return os.path.join(base_dir, f"room_presence_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
    root, ext = os.path.splitext(p)
    if not ext:
        ext = '.xlsx'
    if os.path.exists(p):
        return f"{root}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext}"
    return p


def _compose_display_grid(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    frames = [f for f in frames if f is not None]
    if not frames:
        return None
    target_h = min(f.shape[0] for f in frames)
    scaled: List[np.ndarray] = []
    for f in frames:
        h, w = f.shape[:2]
        if h != target_h:
            new_w = int(w * (target_h / max(1.0, float(h))))
            interp = cv2.INTER_AREA if target_h < h else cv2.INTER_LINEAR
            f = cv2.resize(f, (max(1, new_w), max(1, target_h)), interpolation=interp)
        scaled.append(f)
    if len(scaled) == 1:
        return scaled[0]
    cols = int(math.ceil(math.sqrt(len(scaled))))
    rows = int(math.ceil(len(scaled) / float(cols)))
    row_imgs: List[np.ndarray] = []
    max_row_w = 0
    for r in range(rows):
        start = r * cols
        row_frames = scaled[start:start + cols]
        if not row_frames:
            continue
        row_img = np.concatenate(row_frames, axis=1)
        row_imgs.append(row_img)
        max_row_w = max(max_row_w, int(row_img.shape[1]))
    if not row_imgs:
        return scaled[0]
    padded_rows: List[np.ndarray] = []
    for row_img in row_imgs:
        if row_img.shape[1] < max_row_w:
            pad_w = max_row_w - int(row_img.shape[1])
            pad = np.zeros((row_img.shape[0], pad_w, 3), dtype=row_img.dtype)
            row_img = np.concatenate([row_img, pad], axis=1)
        padded_rows.append(row_img)
    return np.concatenate(padded_rows, axis=0) if len(padded_rows) > 1 else padded_rows[0]


def processor_thread_service(
    sid: int,
    camera_id: int,
    room_id: str,
    vs: VideoStream,
    render_store: RenderedFrame,
    yolo,
    reid_extractors: List[TorchreidExtractor],
    face_app,
    gallery_ref: Dict[str, object],
    args,
    room_tracker: Optional['SpatioTemporalRoomTracker'],
    stop_evt: threading.Event,
):
    frame_idx = 0
    identity_state: Dict[int, Dict] = {}
    feat_state: Dict[int, Dict] = {}
    name_registry: Dict[str, Dict] = {}

    if bool(getattr(args, 'use_strongsort', False) or getattr(args, 'use_deepsort', False)):
        deep_tracker = create_strongsort_tracker(args)
    else:
        deep_tracker = None

    fallback_tracker = None if deep_tracker is not None else SimpleIoUTracker(
        max_age=args.max_age,
        n_init=args.n_init,
        iou_threshold=FALLBACK_IOU_THRESH,
    )

    last_db_reload = 0.0
    fps_ema = 0.0
    last_ts = time.time()
    proc_fps = float(getattr(args, 'proc_fps', 0.0) or 0.0)
    proc_period = (1.0 / proc_fps) if proc_fps > 0.0 else 0.0
    next_proc_tick = time.monotonic()
    last_good_mono = time.monotonic()
    last_restart_req_mono = 0.0
    last_seen_restart_count = 0

    while not stop_evt.is_set():
        if proc_period > 0.0:
            now_m = time.monotonic()
            if now_m < next_proc_tick:
                stop_evt.wait(max(0.0, next_proc_tick - now_m))
                if stop_evt.is_set():
                    break
            elif (now_m - next_proc_tick) > (2.0 * proc_period):
                next_proc_tick = now_m

        ok, frame, cap_ts = vs.read(timeout=0.5)
        if not ok or frame is None:
            stall_s = float(max(0.0, time.monotonic() - float(last_good_mono)))
            stall_thr = float(getattr(args, 'stream_stall_seconds', 8.0) or 0.0)
            backoff = float(getattr(args, 'stream_reconnect_backoff', 2.0) or 0.0)
            if stall_thr > 0.0 and stall_s >= stall_thr:
                if (time.monotonic() - float(last_restart_req_mono)) >= max(0.25, backoff):
                    print(f"[SRC {sid}] Stream stalled for {stall_s:.1f}s -> requesting reconnect")
                    try:
                        vs.request_restart()
                    except Exception:
                        pass
                    last_restart_req_mono = time.monotonic()
            stop_evt.wait(0.01)
            continue

        last_good_mono = time.monotonic()

        if proc_period > 0.0:
            while True:
                ok2, frame2, cap_ts2 = vs.read(timeout=0.0)
                if not ok2 or frame2 is None:
                    break
                frame, cap_ts = frame2, cap_ts2

        try:
            restart_count = int(vs.get_stats().get('restarts', 0) or 0)
        except Exception:
            restart_count = int(last_seen_restart_count)

        if restart_count != int(last_seen_restart_count):
            print(f"[SRC {sid}] Stream reconnected (count={restart_count}). Resetting local states.")
            last_seen_restart_count = restart_count
            identity_state.clear()
            feat_state.clear()
            name_registry.clear()
            if bool(getattr(args, 'use_strongsort', False) or getattr(args, 'use_deepsort', False)):
                deep_tracker = create_strongsort_tracker(args)
            else:
                deep_tracker = None
            if deep_tracker is None:
                fallback_tracker = SimpleIoUTracker(
                    max_age=args.max_age,
                    n_init=args.n_init,
                    iou_threshold=FALLBACK_IOU_THRESH,
                )
            else:
                fallback_tracker = None

        if args.use_db and args.db_refresh_seconds and args.db_refresh_seconds > 0:
            now = time.time()
            if now - last_db_reload >= float(args.db_refresh_seconds):
                try:
                    gallery_ref['gallery'] = build_gallery_from_db(args.db_url)
                    last_db_reload = now
                except Exception as e:
                    print('[DB] reload failed:', e)

        gallery: TwoStageGallery = gallery_ref['gallery']
        out = process_one_frame(
            frame_idx=frame_idx,
            frame=frame,
            sid=sid,
            yolo=yolo,
            reid_extractors=reid_extractors,
            face_app=face_app,
            deep_tracker=deep_tracker,
            fallback_tracker=fallback_tracker,
            gallery=gallery,
            args=args,
            identity_state=identity_state,
            feat_state=feat_state,
            name_registry=name_registry,
            cap_ts=cap_ts,
            room_id=str(room_id),
            room_tracker=room_tracker,
        )
        frame_idx += 1

        now = time.time()
        dt = now - last_ts
        last_ts = now
        if dt > 1e-6:
            inst = 1.0 / dt
            fps_ema = inst if fps_ema <= 0.0 else (0.9 * fps_ema + 0.1 * inst)

        stats = vs.get_stats()
        lag_ms = 0.0
        if cap_ts is not None:
            lag_ms = max(0.0, (time.time() - float(cap_ts)) * 1000.0)

        cv2.putText(
            out,
            f"SRC {sid} | CAM {camera_id} | FPS: {fps_ema:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            out,
            f"Q:{int(stats['qlen'])} drop:{int(stats['dropped_total'])} lag:{lag_ms:.0f}ms room:{room_id}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        prune_after = int(getattr(args, 'tracklet_prune_after', 0) or 0)
        if prune_after > 0 and (frame_idx % 30 == 0):
            stale = [
                tid
                for tid, st in feat_state.items()
                if (frame_idx - int(st.get('last_seen', frame_idx))) > prune_after
            ]
            for tid in stale:
                feat_state.pop(tid, None)
                identity_state.pop(tid, None)

        render_store.set(
            out,
            meta={
                'fps': float(fps_ema),
                'sid': int(sid),
                'camera_id': int(camera_id),
                'room_id': str(room_id),
                'lag_ms': float(lag_ms),
            },
        )

        if proc_period > 0.0:
            next_proc_tick += proc_period
            now_m = time.monotonic()
            if next_proc_tick < (now_m - 0.5 * proc_period):
                next_proc_tick = now_m + proc_period


class TrackingRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._stop_evt = threading.Event()
        self._threads: List[threading.Thread] = []
        self._streams: List[Dict[str, Any]] = []
        self._render_by_camera: Dict[int, RenderedFrame] = {}
        self._room_tracker: Optional['SpatioTemporalRoomTracker'] = None
        self._room_ids_by_camera: Dict[int, str] = {}
        self._excel_path: str = ''
        self._excel_thread: Optional[threading.Thread] = None
        self._started = False
        self._yolo = None
        self._reid_extractors: List[TorchreidExtractor] = []
        self._face_app = None
        self._gallery_ref: Dict[str, object] = {}

    def _init_excel_tracking(self, num_streams: int) -> None:
        args = self.args
        room_ids = [str(r) for r in (getattr(args, 'room_ids', []) or [])]
        if room_ids and len(room_ids) != num_streams:
            print(f"[WARN] --room-ids count ({len(room_ids)}) != number of sources ({num_streams}). Falling back to c1..cN.")
            room_ids = []
        if not room_ids:
            room_ids = [f'c{i+1}' for i in range(num_streams)]
        self._room_ids_by_camera = {}
        camera_ids = list(getattr(args, 'camera_ids', []) or [])
        if not camera_ids:
            camera_ids = list(range(1, num_streams + 1))
            args.camera_ids = list(camera_ids)
        for i in range(num_streams):
            cam_id = int(camera_ids[i])
            self._room_ids_by_camera[cam_id] = str(room_ids[i])

        excel_out_arg = str(getattr(args, 'excel_out', '') or '').strip()
        excel_enabled = bool(excel_out_arg) and (excel_out_arg.lower() not in {'off', 'none', '0', 'false', 'disable', 'disabled'})
        self._room_tracker = None
        self._excel_path = ''
        if not excel_enabled:
            return
        if SpatioTemporalRoomTracker is None:
            print('[WARN] room_presence_analytics.py not available; Excel export disabled.')
            return

        entry_room = str(getattr(args, 'room_entry_id', 'c3') or 'c3')
        room_graph = parse_room_graph_str(
            str(getattr(args, 'room_graph', '') or ''),
            sorted(set(room_ids + [entry_room])),
        )
        topology = None
        if RoomTopology is not None:
            try:
                camera_rooms = sorted(set(room_ids))
                all_rooms = set(room_graph.keys())
                for u, nbs in room_graph.items():
                    for v in nbs:
                        all_rooms.add(v)
                all_rooms.add(entry_room)
                blind_rooms = sorted(all_rooms - set(camera_rooms))
                low_vis_rooms = ['c3'] if 'c3' in all_rooms else []
                topology = RoomTopology(
                    graph=room_graph,
                    camera_rooms=camera_rooms,
                    blind_rooms=blind_rooms,
                    low_visibility_rooms=low_vis_rooms,
                    camera_intermediate_penalty=1.0,
                    low_vis_intermediate_penalty=0.2,
                    edge_cost=1.0,
                )
            except Exception as e:
                print('[WARN] Failed to build custom topology from --room-graph; using default topology:', e)
                topology = None

        self._room_tracker = SpatioTemporalRoomTracker(
            topology=topology,
            tz=str(getattr(args, 'excel_tz', 'Asia/Kolkata')),
            raw_merge_gap_seconds=float(getattr(args, 'excel_raw_merge_gap', 2.0) or 2.0),
            same_room_gap_fill_seconds=float(getattr(args, 'excel_fill_gap', 90.0) or 90.0),
            adjacent_gap_policy=str(getattr(args, 'excel_adjacent_gap_policy', 'split') or 'split'),
            entrance_room=entry_room,
            entry_impute_seconds=float(getattr(args, 'excel_entry_impute_seconds', 0.0) or 0.0),
        )
        if bool(getattr(args, 'excel_disable_path_impute', False)):
            try:
                self._room_tracker.topology.preferred_path = (
                    lambda src_room, dst_room: [src_room, dst_room] if src_room and dst_room and src_room != dst_room else [src_room]
                )
            except Exception:
                pass
        self._excel_path = _resolve_excel_output_path(excel_out_arg)
        try:
            room_dir = os.path.dirname(self._excel_path)
            if room_dir:
                os.makedirs(room_dir, exist_ok=True)
        except Exception:
            pass

    def _export_excel(self) -> None:
        if self._room_tracker is None or not self._excel_path:
            return
        try:
            self._room_tracker.export_excel(self._excel_path, with_date=bool(getattr(self.args, 'excel_with_date', False)))
        except Exception as e:
            print('[ANALYTICS] Excel export failed:', e)

    def _excel_export_loop(self) -> None:
        every = float(getattr(self.args, 'excel_export_every_seconds', 0.0) or 0.0)
        if every <= 0.0:
            return
        next_ts = time.time() + every
        while not self._stop_evt.wait(0.5):
            now = time.time()
            if now < next_ts:
                continue
            self._export_excel()
            if self._excel_path:
                print(f"[ANALYTICS] Periodic Excel update: {self._excel_path}")
            next_ts = now + every

    def start(self) -> None:
        if self._started:
            return
        if self._stop_evt.is_set():
            self._stop_evt = threading.Event()
        args = self.args
        if not bool(getattr(args, 'use_db', False)):
            raise RuntimeError('TrackingRunner requires --use-db and --db-url.')
        if not str(getattr(args, 'db_url', '') or '').strip():
            raise RuntimeError('TrackingRunner requires --db-url.')
        if not getattr(args, 'camera_ids', None):
            args.camera_ids = []
        if len(args.camera_ids) == 0:
            args.camera_ids = list(range(1, len(args.src) + 1))
        if len(args.camera_ids) != len(args.src):
            raise RuntimeError('--camera-ids must have the same length as --src')

        try:
            cv2.setUseOptimized(True)
        except Exception:
            pass
        if torch.cuda.is_available() and ('cuda' in str(args.device).lower()):
            try:
                torch.backends.cudnn.benchmark = True
            except Exception:
                pass
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision('high')
            except Exception:
                pass

        gallery = build_gallery_from_db(args.db_url)
        self._gallery_ref = {'gallery': gallery}
        yolo, reid_extractors, face_app, _deep = init_models(args)
        self._yolo = yolo
        self._reid_extractors = reid_extractors
        self._face_app = face_app

        self._init_excel_tracking(len(args.src))
        if self._room_tracker is not None and self._excel_path:
            try:
                self._export_excel()
                print(f"[ANALYTICS] Excel output: {self._excel_path}")
            except Exception as e:
                print('[ANALYTICS] Initial export failed:', e)
            if float(getattr(args, 'excel_export_every_seconds', 0.0) or 0.0) > 0.0:
                self._excel_thread = threading.Thread(target=self._excel_export_loop, daemon=True)
                self._excel_thread.start()

        self._streams = []
        self._render_by_camera = {}
        for sid, raw_src in enumerate(args.src):
            camera_id = int(args.camera_ids[sid])
            room_id = str(self._room_ids_by_camera.get(camera_id, f'c{sid + 1}'))
            vs = VideoStream(
                raw_src,
                rtsp_buffer=args.rtsp_buffer,
                queue_size=args.queue_size,
                max_queue_age_ms=args.max_queue_age_ms,
                grab_skip=args.grab_skip,
                stall_seconds=args.stream_stall_seconds,
                reconnect_backoff_seconds=args.stream_reconnect_backoff,
                open_timeout_ms=args.stream_open_timeout_ms,
                read_timeout_ms=args.stream_read_timeout_ms,
                max_reconnect_tries=args.stream_reconnect_max_tries,
            )
            buf = RenderedFrame()
            self._render_by_camera[camera_id] = buf
            self._streams.append({
                'sid': int(sid),
                'camera_id': int(camera_id),
                'room_id': str(room_id),
                'src': raw_src,
                'vs': vs,
                'buf': buf,
            })

        if not any(bool(getattr(s.get('vs'), 'ok', False)) for s in self._streams):
            for s in self._streams:
                try:
                    s['vs'].release()
                except Exception:
                    pass
            raise RuntimeError('No sources opened. Check --src URL/path.')

        self._threads = []
        for s in self._streams:
            t = threading.Thread(
                target=processor_thread_service,
                args=(
                    int(s['sid']),
                    int(s['camera_id']),
                    str(s['room_id']),
                    s['vs'],
                    s['buf'],
                    self._yolo,
                    self._reid_extractors,
                    self._face_app,
                    self._gallery_ref,
                    args,
                    self._room_tracker,
                    self._stop_evt,
                ),
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_evt.set()
        for s in self._streams:
            try:
                s['vs'].release()
            except Exception:
                pass
        for t in self._threads:
            try:
                t.join(timeout=1.0)
            except Exception:
                pass
        if self._excel_thread is not None:
            try:
                self._excel_thread.join(timeout=1.0)
            except Exception:
                pass
        try:
            self._export_excel()
            if self._excel_path:
                print(f"[ANALYTICS] Final Excel written: {self._excel_path}")
        except Exception as e:
            print('[ANALYTICS] Final export failed:', e)
        self._started = False

    def get_camera_buffer(self, cam_id: int) -> Optional[RenderedFrame]:
        return self._render_by_camera.get(int(cam_id))

    def list_db_cameras(self, active_only: bool = True) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in self._streams:
            vs = s.get('vs')
            out.append({
                'id': int(s.get('camera_id', -1)),
                'camera_id': int(s.get('camera_id', -1)),
                'room_id': str(s.get('room_id', '')),
                'src': str(s.get('src', '')),
                'running': bool(getattr(vs, 'ok', False)),
            })
        out.sort(key=lambda x: int(x.get('id', 0)))
        return out

    def status(self) -> Dict[str, Any]:
        cams = sorted(list(self._render_by_camera.keys()))
        return {
            'running': bool(self._started),
            'camera_ids': cams,
            'room_ids': {int(k): str(v) for k, v in self._room_ids_by_camera.items()},
            'num_cameras': len(cams),
            'use_db': bool(getattr(self.args, 'use_db', False)),
            'db_refresh_seconds': float(getattr(self.args, 'db_refresh_seconds', 0.0) or 0.0),
            'excel_enabled': bool(self._room_tracker is not None and self._excel_path),
            'excel_path': str(self._excel_path or ''),
            'show': bool(getattr(self.args, 'show', False)),
            'save_video': bool(getattr(self.args, 'save_video', False)),
            'video_dir': str(getattr(self.args, 'video_dir', '') or ''),
            'save_csv': bool(getattr(self.args, 'save_csv', False)),
            'csv_path': str(getattr(self.args, 'csv', '') or ''),
            'write_normalized_data': bool(getattr(self.args, 'write_normalized_data', False)),
        }

    def write_report_snapshot(self, path: Optional[str] = None) -> str:
        if self._room_tracker is None:
            return ''
        out_path = str(path or self._excel_path or '').strip()
        if not out_path:
            out_path = _resolve_excel_output_path('room_presence_snapshot.xlsx')
        try:
            self._room_tracker.export_excel(out_path, with_date=bool(getattr(self.args, 'excel_with_date', False)))
        except Exception:
            return ''
        return out_path



def main() -> None:
    args = parse_args()
    runner = TrackingRunner(args)
    try:
        runner.start()
        if not bool(getattr(args, 'show', False)):
            print('[Main] Running (service/local headless mode). Ctrl+C to stop.')
            while True:
                time.sleep(0.25)
        win_name = 'Face-only ID (centroid) + body/back support'
        try:
            cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        except Exception:
            pass
        print('[Main] Running. Press q to quit.')
        disp_fps = float(getattr(args, 'disp_fps', 0.0) or 0.0)
        if disp_fps <= 0.0:
            disp_fps = float(getattr(args, 'proc_fps', 0.0) or 0.0)
        disp_period = (1.0 / disp_fps) if disp_fps > 0.0 else 0.0
        next_disp_tick = time.monotonic()
        disp_fps_ema = 0.0
        disp_last_ts = time.time()
        while True:
            should_render = True
            if disp_period > 0.0:
                should_render = (time.monotonic() >= next_disp_tick)
            if should_render:
                cams = [int(s['camera_id']) for s in runner._streams]
                frames: List[np.ndarray] = []
                for cam_id in cams:
                    buf = runner.get_camera_buffer(cam_id)
                    frm = None
                    if buf is not None:
                        frm, _ts, _meta = buf.get()
                    if frm is not None:
                        frames.append(frm)
                vis = _compose_display_grid(frames)
                if vis is not None:
                    now = time.time()
                    dt = now - disp_last_ts
                    disp_last_ts = now
                    if dt > 1e-6:
                        inst = 1.0 / dt
                        disp_fps_ema = inst if disp_fps_ema <= 0.0 else (0.9 * disp_fps_ema + 0.1 * inst)
                    txt = f"DISPLAY FPS: {disp_fps_ema:.1f}" if disp_period <= 0.0 else f"DISPLAY FPS (locked): {disp_fps_ema:.1f} / target {disp_fps:.1f}"
                    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    x = max(10, vis.shape[1] - tw - 10)
                    cv2.putText(vis, txt, (x, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    disp_vis = vis
                    try:
                        ds = float(getattr(args, 'display_scale', 0.0) or 0.0)
                    except Exception:
                        ds = 0.0
                    try:
                        mw, mh = getattr(args, 'win_max_wh', [0, 0])
                        mw, mh = int(mw), int(mh)
                    except Exception:
                        mw, mh = 0, 0
                    if ds > 0.0:
                        scale = max(0.05, min(1.0, ds))
                    elif mw > 0 and mh > 0:
                        scale = min(
                            float(mw) / max(1.0, disp_vis.shape[1]),
                            float(mh) / max(1.0, disp_vis.shape[0]),
                            1.0,
                        )
                    else:
                        scale = 1.0
                    if scale < 0.999:
                        new_w = max(1, int(disp_vis.shape[1] * scale))
                        new_h = max(1, int(disp_vis.shape[0] * scale))
                        interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_CUBIC
                        disp_vis = cv2.resize(disp_vis, (new_w, new_h), interpolation=interp)
                    cv2.imshow(win_name, disp_vis)
                if disp_period > 0.0:
                    next_disp_tick += disp_period
                    now_m = time.monotonic()
                    if next_disp_tick < (now_m - 0.5 * disp_period):
                        next_disp_tick = now_m + disp_period
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            runner.stop()
        finally:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        print('Done.')


if __name__ == "__main__":
    try:
        print('[BOOT] updated_rtx5050_facelogic_excel_server_format.py starting...')
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        print('[FATAL] Unhandled exception:')
        traceback.print_exc()
        sys.exit(1)
