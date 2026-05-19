# -*- coding: utf-8 -*-
"""
Strict full-frame tracklet build (2026-03-24)

Requested strict rules included:
- full-frame InsightFace only (no YuNet / no head-crop path)
- a new track shows a known name only after that same track gets its own approved face
- ambiguous face-to-two-overlapping-person matches are dropped for that frame
- duplicate names are blocked; losing tracks are demoted back to Unknown
- visible UI keeps only person boxes and names (no face-score overlays)

Inherited behavior from the prior tracklet build:
- Unknown tracks are visible by default until a face confirms identity.
- Face-only identity persists on the same tracker much longer.
- Anti-swap continuity gate: if a named track suddenly jumps to a different box
  without a confirming face, the name is not reused on that box.
- Unknown / low-face tracks are drawn correctly instead of being always skipped.
- Service mode keeps segmented CSV and segmented video saving behavior.
"""

from __future__ import annotations

import argparse
import csv
import ctypes
import gzip
import math
import os
import random
import shlex
import sys
import threading
import time
import queue
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlsplit, urlunsplit, quote, unquote

import cv2
import numpy as np
import torch

try:
    from openpyxl import Workbook
    from openpyxl.utils import get_column_letter
    from openpyxl.styles import Alignment, Font
    OPENPYXL_OK = True
except Exception:
    Workbook = None
    get_column_letter = None
    Alignment = None
    Font = None
    OPENPYXL_OK = False

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:
    DeepSort = None

try:
    from boxmot import StrongSort as BoxStrongSort
    from boxmot import ByteTrack as BoxByteTrack
except Exception:
    BoxStrongSort = None
    BoxByteTrack = None

try:
    from torchreid.utils import FeatureExtractor as TorchreidExtractor
except Exception:
    TorchreidExtractor = None

try:
    from insightface.app import FaceAnalysis
    INSIGHT_OK = True
except Exception:
    FaceAnalysis = None
    INSIGHT_OK = False

try:
    import onnxruntime as ort
except Exception:
    ort = None

try:
    from sqlalchemy import Column, Integer, String, Boolean, LargeBinary, create_engine, select, BigInteger, DateTime
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
_reid_lock = threading.Lock()
_face_lock = threading.Lock()


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


def extract_face_det_score(face) -> float:
    try:
        s = getattr(face, "det_score", None)
        if s is None:
            s = getattr(face, "score", None)
        if s is None:
            return 1.0
        s = float(s)
        if not math.isfinite(s):
            return 0.0
        return s
    except Exception:
        return 1.0


def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _sanitize_rtsp_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.username or parts.password:
        user = quote(unquote(parts.username or ""), safe="")
        pwd = quote(unquote(parts.password or ""), safe="")
        host = parts.hostname or ""
        netloc = f"{user}:{pwd}@{host}"
        if parts.port:
            netloc += f":{parts.port}"
        return urlunsplit((parts.scheme, netloc, parts.path, parts.query, parts.fragment))
    return url


_VIDEO_FILE_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm", ".m4v",
    ".mpg", ".mpeg", ".ts", ".m2ts", ".mts", ".3gp", ".asf", ".vob",
}


def _is_probably_file_source(src: Any) -> bool:
    if src is None:
        return False
    if isinstance(src, (int, np.integer)):
        return False
    s = str(src).strip()
    if not s:
        return False
    low = s.lower()
    if low.isdigit():
        return False
    if low.startswith(("rtsp://", "rtmp://", "http://", "https://", "udp://", "tcp://")):
        return False
    try:
        p = Path(s)
        if p.exists() and p.is_file():
            return True
        return p.suffix.lower() in _VIDEO_FILE_EXTENSIONS
    except Exception:
        return False


def _get_screen_resolution(default: Tuple[int, int] = (1920, 1080)) -> Tuple[int, int]:
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        w = int(root.winfo_screenwidth())
        h = int(root.winfo_screenheight())
        root.destroy()
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass
    try:
        if os.name == "nt":
            user32 = ctypes.windll.user32
            try:
                user32.SetProcessDPIAware()
            except Exception:
                pass
            w = int(user32.GetSystemMetrics(0))
            h = int(user32.GetSystemMetrics(1))
            if w > 0 and h > 0:
                return w, h
    except Exception:
        pass
    return int(default[0]), int(default[1])


def _resize_to_cell_cover(img: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    scale = max(cell_w / float(iw), cell_h / float(ih))
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x1 = max(0, (new_w - cell_w) // 2)
    y1 = max(0, (new_h - cell_h) // 2)
    crop = resized[y1:y1 + cell_h, x1:x1 + cell_w]
    if crop.shape[0] != cell_h or crop.shape[1] != cell_w:
        crop = cv2.resize(crop, (cell_w, cell_h), interpolation=cv2.INTER_LINEAR)
    return crop


def _resize_to_cell_contain(img: np.ndarray, cell_w: int, cell_h: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    scale = min(cell_w / float(iw), cell_h / float(ih))
    new_w = max(1, int(round(iw * scale)))
    new_h = max(1, int(round(ih * scale)))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
    x1 = (cell_w - new_w) // 2
    y1 = (cell_h - new_h) // 2
    out[y1:y1 + new_h, x1:x1 + new_w] = resized
    return out


def _choose_grid_auto(n: int, screen_w: int, screen_h: int, frame_aspect: float) -> Tuple[int, int]:
    if n <= 1:
        return 1, 1
    best = None
    for rows in range(1, n + 1):
        cols = int(math.ceil(n / rows))
        cell_w = screen_w / float(cols)
        cell_h = screen_h / float(rows)
        cell_aspect = cell_w / max(1.0, cell_h)
        penalty = abs(math.log(max(1e-6, cell_aspect / max(1e-6, frame_aspect))))
        blanks = (rows * cols) - n
        score = penalty + 0.02 * float(blanks)
        cand = (score, rows, cols)
        if best is None or cand < best:
            best = cand
    assert best is not None
    return int(best[1]), int(best[2])


def make_grid_view(
    frames: List[np.ndarray],
    screen_w: int,
    screen_h: int,
    mode: str = "cover",
    grid_rows: int = 0,
    grid_cols: int = 0,
) -> np.ndarray:
    frames = [f for f in frames if f is not None]
    if not frames:
        return np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    n = len(frames)
    aspects = []
    for f in frames:
        h, w = f.shape[:2]
        if w > 0 and h > 0:
            aspects.append(w / float(h))
    frame_aspect = float(np.median(aspects)) if aspects else (16.0 / 9.0)
    if grid_rows > 0 and grid_cols > 0:
        rows, cols = int(grid_rows), int(grid_cols)
        if rows * cols < n:
            cols = int(math.ceil(n / rows))
    elif grid_rows > 0:
        rows = int(grid_rows)
        cols = int(math.ceil(n / rows))
    elif grid_cols > 0:
        cols = int(grid_cols)
        rows = int(math.ceil(n / cols))
    else:
        rows, cols = _choose_grid_auto(n, screen_w, screen_h, frame_aspect)
    rows = max(1, int(rows))
    cols = max(1, int(cols))
    cell_w = max(1, int(screen_w // cols))
    cell_h = max(1, int(screen_h // rows))
    resize_fn = _resize_to_cell_cover if str(mode).lower() == "cover" else _resize_to_cell_contain
    tiles: List[np.ndarray] = []
    for i in range(rows * cols):
        if i < n:
            tile = resize_fn(frames[i], cell_w, cell_h)
        else:
            tile = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
        tiles.append(tile)
    row_imgs = []
    idx = 0
    for _r in range(rows):
        row = np.concatenate(tiles[idx:idx + cols], axis=1)
        row_imgs.append(row)
        idx += cols
    vis = np.concatenate(row_imgs, axis=0)
    if vis.shape[1] != screen_w or vis.shape[0] != screen_h:
        vis = cv2.resize(vis, (screen_w, screen_h), interpolation=cv2.INTER_LINEAR)
    return vis


def _cuda_ep_loadable() -> bool:
    if ort is None:
        return False
    try:
        if sys.platform.startswith("darwin"):
            return False
        capi_dir = Path(ort.__file__).parent / "capi"
        name = "onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so"
        lib_path = capi_dir / name
        if not lib_path.exists():
            return False
        ctypes.CDLL(str(lib_path))
        return True
    except Exception:
        return False


def decode_bank_gzip_npy(raw: bytes | None) -> np.ndarray | None:
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


def encode_bank_gzip_npy(arr: np.ndarray | None) -> bytes | None:
    if arr is None:
        return None
    try:
        a = np.asarray(arr, dtype=np.float32)
        if a.ndim != 2 or a.shape[1] != EXPECTED_DIM:
            return None
        if not np.isfinite(a).all():
            return None
        buf = BytesIO()
        np.save(buf, a, allow_pickle=False)
        raw = buf.getvalue()
        out = BytesIO()
        with gzip.GzipFile(fileobj=out, mode="wb") as gz:
            gz.write(raw)
        return out.getvalue()
    except Exception:
        return None


def _as_vec512(x) -> np.ndarray | None:
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


def _bytes_per_embedding() -> int:
    return int(EXPECTED_DIM * 4)


def _rows_from_mb(mb: float) -> int:
    mb = float(mb or 0.0)
    if mb <= 0:
        return 0
    return max(1, int((mb * 1024.0 * 1024.0) // float(_bytes_per_embedding())))


class EmbeddingUpdateLogger:
    def __init__(self, path: str, warn_interval_s: float = 5.0):
        self.path = str(path)
        self._lock = threading.Lock()
        self._warn_interval_s = float(max(0.0, float(warn_interval_s or 0.0)))
        self._last_warn_mono = 0.0
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        except Exception as e:
            self._warn(f"[WARN] Could not create audit log folder for {self.path}: {e}")
        try:
            if self.path and (not os.path.exists(self.path)):
                with open(self.path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "timestamp", "member_id", "name", "camera_id", "tracks", "face_sim_max",
                        "body_added", "body_removed", "body_before", "body_after",
                        "face_added", "face_removed", "face_before", "face_after",
                    ])
        except Exception as e:
            self._warn(f"[WARN] Could not init embeddings audit CSV at {self.path}: {e}")

    def _warn(self, msg: str) -> None:
        if not msg:
            return
        if self._warn_interval_s <= 0:
            print(msg)
            return
        now = time.monotonic()
        if (now - float(self._last_warn_mono)) >= float(self._warn_interval_s):
            self._last_warn_mono = now
            print(msg)

    def log(self, ts: float, member_id: int, name: str, camera_id: int, tracks: str, face_sim_max: float,
            body_added: int, body_removed: int, body_before: int, body_after: int,
            face_added: int, face_removed: int, face_before: int, face_after: int) -> None:
        try:
            with self._lock:
                with open(self.path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M:%S"),
                        int(member_id), str(name), int(camera_id), str(tracks), f"{float(face_sim_max):.4f}",
                        int(body_added), int(body_removed), int(body_before), int(body_after),
                        int(face_added), int(face_removed), int(face_before), int(face_after),
                    ])
        except Exception as e:
            self._warn(f"[WARN] Embeddings audit CSV append failed ({self.path}): {e}")


class EmbeddingSampleLogger:
    def __init__(self, path: str, warn_interval_s: float = 5.0):
        self.path = str(path)
        self._lock = threading.Lock()
        self._warn_interval_s = float(max(0.0, float(warn_interval_s or 0.0)))
        self._last_warn_mono = 0.0
        try:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        except Exception as e:
            self._warn(f"[WARN] Could not create samples log folder for {self.path}: {e}")
        try:
            if self.path and (not os.path.exists(self.path)):
                with open(self.path, "w", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        "timestamp", "member_id", "name", "camera_id", "track_id", "face_sim",
                        "face_det_score", "has_body_emb", "has_face_emb", "action", "note",
                    ])
        except Exception as e:
            self._warn(f"[WARN] Could not init embeddings samples CSV at {self.path}: {e}")

    def _warn(self, msg: str) -> None:
        if not msg:
            return
        if self._warn_interval_s <= 0:
            print(msg)
            return
        now = time.monotonic()
        if (now - float(self._last_warn_mono)) < float(self._warn_interval_s):
            return
        self._last_warn_mono = now
        print(msg)

    def log(self, ts: float, member_id: int, name: str, camera_id: int, track_id: int, face_sim: float,
            face_det_score: float, has_body_emb: bool, has_face_emb: bool,
            action: str = "accepted", note: str = "") -> None:
        try:
            with self._lock:
                with open(self.path, "a", newline="", encoding="utf-8") as f:
                    w = csv.writer(f)
                    w.writerow([
                        float(ts), int(member_id), str(name), int(camera_id), int(track_id),
                        f"{float(face_sim):.4f}", f"{float(face_det_score):.4f}",
                        int(bool(has_body_emb)), int(bool(has_face_emb)), str(action), str(note),
                    ])
        except Exception as e:
            self._warn(f"[WARN] Embeddings samples CSV append failed ({self.path}): {e}")


@dataclass
class EmbeddingSample:
    member_id: int
    name: str
    camera_id: int
    track_id: int
    ts: float
    face_sim: float
    face_det_score: float = 1.0
    body_emb: Optional[np.ndarray] = None
    face_emb: Optional[np.ndarray] = None


class EmbeddingDBUpdater:
    def __init__(
        self,
        db_url: str,
        slot_mb: float,
        flush_seconds: float,
        min_sample_seconds: float,
        min_face_sim: float,
        min_face_det_score: float,
        log_csv_path: str,
        update_body: bool = True,
        update_face: bool = True,
        max_queue: int = 5000,
        reset_if_gap_days_ge: int = 2,
        reset_on_start: bool = False,
        samples_log_csv_path: str = "",
    ):
        self.db_url = str(db_url)
        self.slot_mb = float(slot_mb or 0.0)
        self.slot_rows = int(_rows_from_mb(self.slot_mb))
        self.total_rows = int(self.slot_rows * 2)
        self.flush_seconds = float(flush_seconds or 10.0)
        self.min_sample_seconds = float(min_sample_seconds or 0.5)
        self.min_face_sim = float(min_face_sim or 0.75)
        self.min_face_det_score = float(min_face_det_score or 0.75)
        self.update_body = bool(update_body)
        self.update_face = bool(update_face)
        self.reset_if_gap_days_ge = int(max(1, int(reset_if_gap_days_ge)))
        self._stop = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=max(1, int(max_queue)))
        self._body_buf: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
        self._face_buf: Dict[Tuple[int, int], List[np.ndarray]] = defaultdict(list)
        self._meta_buf: Dict[Tuple[int, int], List[EmbeddingSample]] = defaultdict(list)
        self._last_sample_ts: Dict[Tuple[int, int], float] = defaultdict(float)
        self._last_flush_ts: Dict[Tuple[int, int], float] = defaultdict(float)
        self._state_lock = threading.Lock()
        self._full_state: Dict[Tuple[int, int], Dict[str, Any]] = defaultdict(dict)
        self.logger = EmbeddingUpdateLogger(log_csv_path) if log_csv_path else None
        self.sample_logger = EmbeddingSampleLogger(samples_log_csv_path) if samples_log_csv_path else None

        Base = declarative_base()

        class MemberRow(Base):
            __tablename__ = "members"
            id = Column(Integer, primary_key=True)
            member_number = Column(String)
            first_name = Column(String)
            last_name = Column(String)
            is_active = Column(Boolean)

        class MemberEmbeddingRow(Base):
            __tablename__ = "member_embeddings"
            id = Column(BigInteger, primary_key=True)
            member_id = Column(Integer, nullable=False)
            camera_id = Column(Integer, nullable=False)
            if Vector is not None:
                body_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
                face_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
            else:
                body_embedding = Column(ARRAY(Float), nullable=True)
                face_embedding = Column(ARRAY(Float), nullable=True)
            body_embeddings_raw = Column(LargeBinary, nullable=True)
            face_embeddings_raw = Column(LargeBinary, nullable=True)
            last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True)

        self.MemberEmbeddingRow = MemberEmbeddingRow
        self.MemberRow = MemberRow
        self.engine = create_engine(self.db_url, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()
        if bool(reset_on_start):
            try:
                self._reset_all_to_empty()
            except Exception:
                pass

    @staticmethod
    def _utc_today() -> datetime.date:
        return datetime.now(timezone.utc).date()

    @staticmethod
    def _utc_date_of(dt: Optional[datetime]) -> Optional[datetime.date]:
        if dt is None:
            return None
        try:
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.astimezone(timezone.utc).date()
        except Exception:
            return None

    def _slot_idx_for_date(self, d: datetime.date) -> int:
        return int(d.toordinal() % 2)

    def _split_slots(self, bank: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if bank is None:
            bank = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        bank = np.asarray(bank, dtype=np.float32)
        if bank.ndim != 2 or bank.shape[1] != EXPECTED_DIM:
            bank = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        if self.total_rows > 0 and bank.shape[0] > self.total_rows:
            bank = bank[: self.total_rows]
        s = int(self.slot_rows)
        if s <= 0:
            return bank, np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        slot0 = bank[: min(s, bank.shape[0])]
        slot1 = bank[s: min(2 * s, bank.shape[0])] if bank.shape[0] > s else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        return slot0, slot1

    def _combine_slots(self, slot0: np.ndarray, slot1: np.ndarray) -> np.ndarray:
        slot0 = np.asarray(slot0, dtype=np.float32) if slot0 is not None else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        slot1 = np.asarray(slot1, dtype=np.float32) if slot1 is not None else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        if self.slot_rows > 0:
            slot0 = slot0[: self.slot_rows]
            slot1 = slot1[: self.slot_rows]
        out = np.concatenate([slot0, slot1], axis=0) if (slot0.size or slot1.size) else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        if self.total_rows > 0 and out.shape[0] > self.total_rows:
            out = out[: self.total_rows]
        return out

    def can_accept(self, member_id: int, camera_id: int) -> bool:
        mid = int(member_id)
        cid = int(camera_id)
        if mid <= 0 or cid <= 0:
            return False
        if self.slot_rows <= 0:
            return False
        today = self._utc_today()
        day_ord = int(today.toordinal())
        key = (mid, cid)
        with self._state_lock:
            st = self._full_state.get(key, {})
            st_day = int(st.get("day_ord", -1))
            if st_day != day_ord:
                return True
            return not bool(st.get("full", False))

    def enqueue(self, s: EmbeddingSample) -> None:
        if s is None:
            return
        try:
            self._q.put_nowait(s)
        except queue.Full:
            return

    def close(self) -> None:
        self._stop.set()
        try:
            self._thr.join(timeout=2.0)
        except Exception:
            pass
        try:
            self._flush_all()
        except Exception:
            pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            drained = 0
            while drained < 64:
                try:
                    s: EmbeddingSample = self._q.get_nowait()
                except queue.Empty:
                    break
                drained += 1
                self._handle_sample(s)
            try:
                self._flush_due()
            except Exception:
                pass
            self._stop.wait(0.05)
        try:
            self._flush_all()
        except Exception:
            pass

    def _handle_sample(self, s: EmbeddingSample) -> None:
        mid = int(s.member_id)
        cid = int(s.camera_id)
        if mid <= 0 or cid <= 0:
            return
        if float(s.face_sim) < float(self.min_face_sim):
            return
        if float(getattr(s, "face_det_score", 1.0) or 1.0) < float(self.min_face_det_score):
            return
        key = (mid, cid)
        now = float(time.time())
        last = float(self._last_sample_ts.get(key, 0.0))
        if (now - last) < float(self.min_sample_seconds):
            return
        self._last_sample_ts[key] = now
        body = _as_vec512(s.body_emb) if self.update_body else None
        face = _as_vec512(s.face_emb) if self.update_face else None
        if body is not None:
            self._body_buf[key].append(body.astype(np.float32))
        if face is not None:
            self._face_buf[key].append(face.astype(np.float32))
        if (body is not None) or (face is not None):
            self._meta_buf[key].append(s)
            if self.sample_logger is not None:
                try:
                    self.sample_logger.log(
                        ts=float(s.ts or now),
                        member_id=int(s.member_id),
                        name=str(s.name),
                        camera_id=int(s.camera_id),
                        track_id=int(s.track_id),
                        face_sim=float(s.face_sim),
                        face_det_score=float(s.face_det_score),
                        has_body_emb=bool(body is not None),
                        has_face_emb=bool(face is not None),
                        action="accepted",
                        note="",
                    )
                except Exception:
                    pass

    def _flush_due(self) -> None:
        if self.flush_seconds <= 0:
            return
        now = float(time.time())
        keys = set(self._body_buf.keys()) | set(self._face_buf.keys())
        for key in list(keys):
            has_any = (len(self._body_buf.get(key, [])) > 0) or (len(self._face_buf.get(key, [])) > 0)
            if not has_any:
                continue
            last_flush = float(self._last_flush_ts.get(key, 0.0))
            if (now - last_flush) >= float(self.flush_seconds):
                self._flush_key(key)

    def _flush_all(self) -> None:
        keys = set(self._body_buf.keys()) | set(self._face_buf.keys())
        for key in list(keys):
            try:
                self._flush_key(key)
            except Exception:
                pass

    def _flush_key(self, key: Tuple[int, int]) -> None:
        mid, cid = int(key[0]), int(key[1])
        body_list = self._body_buf.get(key, [])
        face_list = self._face_buf.get(key, [])
        if not body_list and not face_list:
            return
        now_dt = datetime.now(timezone.utc)
        today = now_dt.date()
        slot_idx = self._slot_idx_for_date(today)
        with self.Session() as session:
            stmt = select(self.MemberEmbeddingRow).where(
                (self.MemberEmbeddingRow.member_id == mid) &
                (self.MemberEmbeddingRow.camera_id == cid)
            )
            row = session.execute(stmt).scalars().first()
            if row is None:
                row = self.MemberEmbeddingRow(member_id=mid, camera_id=cid)
                session.add(row)
                session.flush()
            old_body = decode_bank_gzip_npy(getattr(row, "body_embeddings_raw", None)) if self.update_body else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
            old_face = decode_bank_gzip_npy(getattr(row, "face_embeddings_raw", None)) if self.update_face else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
            if old_body is None:
                old_body = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
            if old_face is None:
                old_face = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
            body_before = int(old_body.shape[0])
            face_before = int(old_face.shape[0])
            b0, b1 = self._split_slots(old_body)
            f0, f1 = self._split_slots(old_face)
            last_date = self._utc_date_of(getattr(row, "last_embedding_update_ts", None))
            removed_body = 0
            removed_face = 0
            if last_date is None:
                removed_body = int(b0.shape[0] + b1.shape[0])
                removed_face = int(f0.shape[0] + f1.shape[0])
                b0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                b1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                f0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                f1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
            else:
                try:
                    gap_days = int((today - last_date).days)
                except Exception:
                    gap_days = 0
                if gap_days >= self.reset_if_gap_days_ge:
                    removed_body = int(b0.shape[0] + b1.shape[0])
                    removed_face = int(f0.shape[0] + f1.shape[0])
                    b0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                    b1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                    f0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                    f1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                elif gap_days >= 1:
                    if slot_idx == 0:
                        removed_body = int(b0.shape[0])
                        removed_face = int(f0.shape[0])
                        b0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                        f0 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                    else:
                        removed_body = int(b1.shape[0])
                        removed_face = int(f1.shape[0])
                        b1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
                        f1 = np.zeros((0, EXPECTED_DIM), dtype=np.float32)
            if slot_idx == 0:
                b_active, b_other = b0, b1
                f_active, f_other = f0, f1
            else:
                b_active, b_other = b1, b0
                f_active, f_other = f1, f0
            body_added = 0
            face_added = 0
            if self.update_body and body_list:
                avail = max(0, self.slot_rows - int(b_active.shape[0]))
                take = min(len(body_list), avail)
                if take > 0:
                    add = l2_normalize_rows(np.stack(body_list[:take], axis=0))
                    b_active = np.concatenate([b_active, add], axis=0)
                    del body_list[:take]
                    body_added += int(take)
                if len(body_list) > 0 and int(b_other.shape[0]) == 0:
                    avail2 = max(0, self.slot_rows - int(b_other.shape[0]))
                    take2 = min(len(body_list), avail2)
                    if take2 > 0:
                        add2 = l2_normalize_rows(np.stack(body_list[:take2], axis=0))
                        b_other = np.concatenate([b_other, add2], axis=0)
                        del body_list[:take2]
                        body_added += int(take2)
                if body_list:
                    body_list.clear()
            if self.update_face and face_list:
                avail = max(0, self.slot_rows - int(f_active.shape[0]))
                take = min(len(face_list), avail)
                if take > 0:
                    add = l2_normalize_rows(np.stack(face_list[:take], axis=0))
                    f_active = np.concatenate([f_active, add], axis=0)
                    del face_list[:take]
                    face_added += int(take)
                if len(face_list) > 0 and int(f_other.shape[0]) == 0:
                    avail2 = max(0, self.slot_rows - int(f_other.shape[0]))
                    take2 = min(len(face_list), avail2)
                    if take2 > 0:
                        add2 = l2_normalize_rows(np.stack(face_list[:take2], axis=0))
                        f_other = np.concatenate([f_other, add2], axis=0)
                        del face_list[:take2]
                        face_added += int(take2)
                if face_list:
                    face_list.clear()
            if slot_idx == 0:
                b0, b1 = b_active, b_other
                f0, f1 = f_active, f_other
            else:
                b1, b0 = b_active, b_other
                f1, f0 = f_active, f_other
            upd_body = self._combine_slots(b0, b1)
            upd_face = self._combine_slots(f0, f1)
            body_after = int(upd_body.shape[0])
            face_after = int(upd_face.shape[0])
            changed = (body_added > 0 or face_added > 0 or removed_body > 0 or removed_face > 0)
            member_name = ""
            try:
                mrow = session.execute(
                    select(self.MemberRow.first_name, self.MemberRow.last_name, self.MemberRow.member_number).where(self.MemberRow.id == mid)
                ).first()
                if mrow:
                    first = str(mrow[0] or "").strip()
                    last = str(mrow[1] or "").strip()
                    member_name = (first + " " + last).strip() if last else first
                    if not member_name:
                        member_name = str(mrow[2] or "").strip()
            except Exception:
                member_name = ""
            if changed:
                try:
                    if self.update_body:
                        row.body_embeddings_raw = encode_bank_gzip_npy(l2_normalize_rows(upd_body))
                        if body_after > 0:
                            row.body_embedding = l2_normalize(np.mean(upd_body, axis=0)).tolist()
                    if self.update_face:
                        row.face_embeddings_raw = encode_bank_gzip_npy(l2_normalize_rows(upd_face))
                        if face_after > 0:
                            row.face_embedding = l2_normalize(np.mean(upd_face, axis=0)).tolist()
                    row.last_embedding_update_ts = now_dt
                    session.commit()
                except Exception as e:
                    session.rollback()
                    print(f"[DB-UPD] commit failed member_id={mid} camera_id={cid}: {e}")
                    return
            avail_body_now = 0
            avail_face_now = 0
            if self.slot_rows > 0:
                if self.update_body:
                    active_now = (b0 if slot_idx == 0 else b1)
                    avail_body_now = max(0, self.slot_rows - int(active_now.shape[0]))
                if self.update_face:
                    active_now = (f0 if slot_idx == 0 else f1)
                    avail_face_now = max(0, self.slot_rows - int(active_now.shape[0]))
            is_full = (not self.update_body or avail_body_now <= 0) and (not self.update_face or avail_face_now <= 0)
            with self._state_lock:
                self._full_state[key] = {"day_ord": int(today.toordinal()), "full": bool(is_full)}
            metas = self._meta_buf.get(key, [])
            track_ids = ",".join(str(int(m.track_id)) for m in metas[:20]) if metas else ""
            face_sim_max = max((float(m.face_sim) for m in metas), default=0.0)
            if metas:
                metas.clear()
            if self.logger is not None and (body_added or face_added or removed_body or removed_face):
                self.logger.log(
                    ts=float(time.time()), member_id=int(mid), name=str(member_name), camera_id=int(cid), tracks=track_ids,
                    face_sim_max=float(face_sim_max), body_added=int(body_added), body_removed=int(removed_body),
                    body_before=int(body_before), body_after=int(body_after), face_added=int(face_added),
                    face_removed=int(removed_face), face_before=int(face_before), face_after=int(face_after),
                )
                try:
                    print(
                        f"[DB-UPD] member_id={mid} cam={cid} | "
                        f"body +{body_added}/-{removed_body} ({body_before}->{body_after}) | "
                        f"face +{face_added}/-{removed_face} ({face_before}->{face_after})"
                    )
                except Exception:
                    pass
        self._last_flush_ts[key] = float(time.time())

    def _reset_all_to_empty(self):
        with self.Session() as session:
            rows = session.execute(select(self.MemberEmbeddingRow)).scalars().all()
            for row in rows:
                row.body_embeddings_raw = None
                row.face_embeddings_raw = None
                row.body_embedding = None
                row.face_embedding = None
                row.last_embedding_update_ts = datetime.now(timezone.utc)
            session.commit()


@dataclass
class PersonEntry:
    member_id: int
    name: str
    camera_id: int
    body_bank: np.ndarray | None
    body_centroid: np.ndarray | None


@dataclass
class FaceGallery:
    member_ids: list[int]
    names: list[str]
    mat: np.ndarray

    def is_empty(self) -> bool:
        return (not self.names) or (self.mat is None) or (self.mat.size == 0)


def best_face_top2(emb: np.ndarray | None, face_gallery: FaceGallery) -> tuple[int | None, str | None, float, float]:
    if emb is None or face_gallery is None or face_gallery.is_empty():
        return None, None, 0.0, 0.0
    q = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
    if q.size != EXPECTED_DIM or not np.isfinite(q).all():
        return None, None, 0.0, 0.0
    sims = face_gallery.mat @ q
    if sims.size == 0:
        return None, None, 0.0, 0.0
    if sims.size == 1:
        return int(face_gallery.member_ids[0]), str(face_gallery.names[0]), float(sims[0]), 0.0
    idxs = np.argpartition(sims, -2)[-2:]
    i1, i2 = int(idxs[0]), int(idxs[1])
    if sims[i2] > sims[i1]:
        i1, i2 = i2, i1
    best_idx, second_idx = i1, i2
    return int(face_gallery.member_ids[best_idx]), str(face_gallery.names[best_idx]), float(sims[best_idx]), float(sims[second_idx])


def _normalize_gallery_member_name(value: Any) -> str:
    txt = str(value or "").strip().lower()
    if not txt:
        return ""
    return " ".join(txt.split())


def _coerce_gallery_member_ids(values: Any) -> set[int]:
    out: set[int] = set()
    if values is None:
        return out
    if isinstance(values, (str, int, np.integer)):
        values = [values]
    for raw in values:
        if raw is None:
            continue
        if isinstance(raw, str):
            parts = [raw]
            if ("," in raw) or (";" in raw):
                parts = [p.strip() for p in raw.replace(";", ",").split(",")]
            for part in parts:
                if not part:
                    continue
                try:
                    val = int(str(part).strip())
                except Exception:
                    continue
                if val > 0:
                    out.add(int(val))
            continue
        try:
            val = int(raw)
        except Exception:
            continue
        if val > 0:
            out.add(int(val))
    return out


def _coerce_gallery_member_names(values: Any) -> set[str]:
    out: set[str] = set()
    if values is None:
        return out
    if isinstance(values, (str, bytes)):
        raw_items = [values]
    else:
        try:
            raw_items = list(values)
        except TypeError:
            raw_items = [values]
    for raw in raw_items:
        txt = str(raw or "").strip()
        if not txt:
            continue
        parts = [txt]
        if ("," in txt) or (";" in txt):
            parts = [p.strip() for p in txt.replace(";", ",").split(",")]
        for part in parts:
            key = _normalize_gallery_member_name(part)
            if key:
                out.add(key)
    return out


def build_galleries_from_db(
    db_url: str,
    active_only: bool = True,
    max_bank_per_entry: int = 0,
    request_mode: str = "location",
    requested_member_ids: Optional[List[int]] = None,
    requested_member_names: Optional[List[str]] = None,
) -> tuple[dict[int, list[PersonEntry]], FaceGallery, dict[str, int]]:
    Base = declarative_base()

    class MemberRow(Base):
        __tablename__ = "members"
        id = Column(Integer, primary_key=True)
        member_number = Column(String)
        first_name = Column(String)
        last_name = Column(String)
        is_active = Column(Boolean)

    class MemberEmbeddingRow(Base):
        __tablename__ = "member_embeddings"
        id = Column(BigInteger, primary_key=True)
        member_id = Column(Integer, nullable=False)
        camera_id = Column(Integer, nullable=False)
        if Vector is not None:
            body_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
            face_embedding = Column(Vector(EXPECTED_DIM), nullable=True)
        else:
            body_embedding = Column(ARRAY(Float), nullable=True)
            face_embedding = Column(ARRAY(Float), nullable=True)
        body_embeddings_raw = Column(LargeBinary, nullable=True)
        face_embeddings_raw = Column(LargeBinary, nullable=True)
        last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True)

    mode = str(request_mode or "location").strip().lower()
    filter_member_gallery = mode == "member"
    wanted_member_ids = _coerce_gallery_member_ids(requested_member_ids)
    wanted_member_names = _coerce_gallery_member_names(requested_member_names)

    engine = create_engine(db_url, pool_pre_ping=True)
    Session = sessionmaker(bind=engine)
    people_by_cam: dict[int, list[PersonEntry]] = defaultdict(list)
    face_vecs_by_member: dict[int, list[np.ndarray]] = defaultdict(list)
    member_id_to_name: dict[int, str] = {}
    name_to_member_id: dict[str, int] = {}

    try:
        with Session() as session:
            stmt = (
                select(
                    MemberEmbeddingRow.member_id,
                    MemberEmbeddingRow.camera_id,
                    MemberRow.member_number,
                    MemberRow.first_name,
                    MemberRow.last_name,
                    MemberRow.is_active,
                    MemberEmbeddingRow.body_embedding,
                    MemberEmbeddingRow.face_embedding,
                    MemberEmbeddingRow.body_embeddings_raw,
                    MemberEmbeddingRow.face_embeddings_raw,
                )
                .join(MemberRow, MemberRow.id == MemberEmbeddingRow.member_id)
            )
            rows = session.execute(stmt).all()
            for r in rows:
                mid = int(r.member_id)
                cam_id = int(r.camera_id)
                member_number = str(r.member_number or "").strip()
                first = str(r.first_name or "").strip()
                last = str(r.last_name or "").strip()
                name = (first + " " + last).strip() if last else first
                try:
                    is_active = bool(r.is_active) if (r.is_active is not None) else True
                except Exception:
                    is_active = True
                if active_only and (not is_active):
                    continue
                if not name:
                    name = member_number
                if not name:
                    name = f"member_{mid}"

                lookup_keys: set[str] = set()
                if name:
                    lookup_keys.add(_normalize_gallery_member_name(name))
                if member_number:
                    lookup_keys.add(_normalize_gallery_member_name(member_number))

                if filter_member_gallery:
                    matched = False
                    if mid in wanted_member_ids:
                        matched = True
                    elif wanted_member_names and any((k in wanted_member_names) for k in lookup_keys if k):
                        matched = True
                    if not matched:
                        continue

                if name and name not in name_to_member_id:
                    name_to_member_id[name] = mid
                member_id_to_name[mid] = name
                body_cent = _as_vec512(r.body_embedding)
                face_cent = _as_vec512(r.face_embedding)
                body_bank = decode_bank_gzip_npy(r.body_embeddings_raw)
                face_bank = decode_bank_gzip_npy(r.face_embeddings_raw)
                if max_bank_per_entry and max_bank_per_entry > 0:
                    m = int(max_bank_per_entry)
                    if body_bank is not None and body_bank.shape[0] > m:
                        body_bank = body_bank[:m]
                    if face_bank is not None and face_bank.shape[0] > m:
                        face_bank = face_bank[:m]
                if body_cent is None and body_bank is not None and body_bank.shape[0] > 0:
                    body_cent = l2_normalize(np.mean(body_bank, axis=0))
                if face_cent is None and face_bank is not None and face_bank.shape[0] > 0:
                    face_cent = l2_normalize(np.mean(face_bank, axis=0))
                if body_bank is None and body_cent is not None:
                    body_bank = body_cent.reshape(1, -1).astype(np.float32)
                if body_bank is not None and body_bank.ndim == 2 and body_bank.shape[1] == EXPECTED_DIM:
                    body_bank = l2_normalize_rows(body_bank.astype(np.float32))
                people_by_cam[cam_id].append(PersonEntry(mid, name, cam_id, body_bank, body_cent))
                if face_cent is not None:
                    face_vecs_by_member[mid].append(face_cent.astype(np.float32))
    finally:
        try:
            engine.dispose()
        except Exception:
            pass

    fg_member_ids: list[int] = []
    fg_names: list[str] = []
    fg_vecs: list[np.ndarray] = []
    for mid, vecs in face_vecs_by_member.items():
        if not vecs:
            continue
        mat = l2_normalize_rows(np.stack(vecs, axis=0))
        v = l2_normalize(np.mean(mat, axis=0))
        fg_member_ids.append(int(mid))
        fg_names.append(str(member_id_to_name.get(mid, f"member_{mid}")))
        fg_vecs.append(v.astype(np.float32))
    face_mat = l2_normalize_rows(np.stack(fg_vecs, axis=0)) if fg_vecs else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
    face_gallery = FaceGallery(fg_member_ids, fg_names, face_mat)
    total_body_entries = sum(len(v) for v in people_by_cam.values())
    if filter_member_gallery:
        filter_desc_parts: list[str] = []
        if wanted_member_ids:
            filter_desc_parts.append(f"ids={sorted(wanted_member_ids)}")
        if wanted_member_names:
            filter_desc_parts.append(f"names={sorted(wanted_member_names)}")
        filter_desc = ", ".join(filter_desc_parts) if filter_desc_parts else "none"
    else:
        filter_desc = "all"
    print(
        f"[DB] Loaded member_embeddings: body_entries={total_body_entries} | "
        f"face_identities={len(fg_names)} | mode={mode} | filter={filter_desc}"
    )
    return people_by_cam, face_gallery, name_to_member_id


class GalleryManager:
    def __init__(self, args):
        self._lock = threading.Lock()
        self._reload_lock = threading.Lock()
        self.people_by_cam: dict[int, list[PersonEntry]] = defaultdict(list)
        self.face_gallery: FaceGallery = FaceGallery([], [], np.zeros((0, EXPECTED_DIM), dtype=np.float32))
        self.name_to_member_id: dict[str, int] = {}
        self.last_load_ts: float = 0.0
        self.load(args)

    def load(self, args) -> None:
        active_only = not bool(getattr(args, "db_include_inactive", False))
        request_mode = str(getattr(args, "gallery_request_mode", "location") or "location")
        requested_member_ids = list(getattr(args, "gallery_member_ids", []) or [])
        requested_member_names = list(getattr(args, "gallery_member_names", []) or [])
        people_by_cam, fg, name_to_mid = build_galleries_from_db(
            args.db_url,
            active_only=active_only,
            max_bank_per_entry=int(getattr(args, "db_max_bank", 0) or 0),
            request_mode=request_mode,
            requested_member_ids=requested_member_ids,
            requested_member_names=requested_member_names,
        )
        with self._lock:
            self.people_by_cam = people_by_cam
            self.face_gallery = fg
            self.name_to_member_id = name_to_mid
            self.last_load_ts = time.time()

    def maybe_reload(self, args) -> None:
        period = float(getattr(args, "db_refresh_seconds", 0.0) or 0.0)
        if period <= 0:
            return
        now = time.time()
        if (now - self.last_load_ts) < period:
            return
        if not self._reload_lock.acquire(blocking=False):
            return
        try:
            if (time.time() - self.last_load_ts) < period:
                return
            try:
                self.load(args)
                print("[DB] Gallery reloaded")
            except Exception as e:
                print("[DB] reload failed:", e)
        finally:
            self._reload_lock.release()

    def snapshot(self) -> tuple[dict[int, list[PersonEntry]], FaceGallery, dict[str, int]]:
        with self._lock:
            return dict(self.people_by_cam), self.face_gallery, dict(self.name_to_member_id)


def iou_xyxy(a, b) -> float:
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


def ioa_xyxy(inner, outer) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_x1, inter_y1 = max(ix1, ox1), max(iy1, oy1)
    inter_x2, inter_y2 = min(ix2, ox2), min(iy2, oy2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    inner_area = max(0.0, (ix2 - ix1)) * max(0.0, (iy2 - iy1))
    return float(inter / inner_area) if inner_area > 0 else 0.0


def _point_in_xyxy(px: float, py: float, box) -> bool:
    x1, y1, x2, y2 = box
    return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


def _box_center_xyxy(box) -> Tuple[float, float]:
    x1, y1, x2, y2 = map(float, box)
    return 0.5 * (x1 + x2), 0.5 * (y1 + y2)


def same_person_continuation(prev_box, curr_box, min_iou: float = 0.05, max_center_shift: float = 0.60) -> bool:
    if prev_box is None or curr_box is None:
        return True
    if iou_xyxy(prev_box, curr_box) >= float(min_iou):
        return True
    px, py = _box_center_xyxy(prev_box)
    cx, cy = _box_center_xyxy(curr_box)
    pw = max(1.0, float(prev_box[2] - prev_box[0]))
    ph = max(1.0, float(prev_box[3] - prev_box[1]))
    dx = abs(cx - px) / pw
    dy = abs(cy - py) / ph
    return (dx <= float(max_center_shift)) and (dy <= float(max_center_shift))


def assign_faces_to_tracks_one_to_one(
    recognized_faces: List[Dict[str, Any]],
    raw_tracks: List[Dict[str, Any]],
    args,
) -> Dict[int, Dict[str, Any]]:
    if not recognized_faces or not raw_tracks:
        return {}
    candidates: List[Tuple[float, int, int]] = []
    face_valid_tracks: Dict[int, List[int]] = defaultdict(list)
    face_track_boxes: Dict[Tuple[int, int], Tuple[float, float, float, float]] = {}
    for fi, fm in enumerate(recognized_faces):
        fbox = fm.get("bbox", None)
        if not fbox:
            continue
        fx1, fy1, fx2, fy2 = map(float, fbox)
        fw = max(1.0, fx2 - fx1)
        fh = max(1.0, fy2 - fy1)
        f_area = max(1.0, fw * fh)
        fc_x = 0.5 * (fx1 + fx2)
        fc_y = 0.5 * (fy1 + fy2)
        for ti, tr in enumerate(raw_tracks):
            x1, y1, x2, y2 = tr["bbox"]
            t_xyxy = (float(x1), float(y1), float(x2), float(y2))
            p_area = float(max(1.0, (x2 - x1) * (y2 - y1)))
            if bool(getattr(args, "face_center_in_person", False)) and not _point_in_xyxy(fc_x, fc_y, t_xyxy):
                continue
            top_limit = float(y1) + float(getattr(args, "face_center_y_max_ratio", 0.70)) * float(max(1, (y2 - y1)))
            if fc_y > top_limit:
                continue
            link_mode = str(getattr(args, "face_link_mode", "ioa"))
            link = ioa_xyxy(fbox, t_xyxy) if link_mode == "ioa" else iou_xyxy(t_xyxy, fbox)
            if link < float(getattr(args, "face_iou_link", 0.35)):
                continue
            ratio = float(f_area / p_area)
            min_ratio = float(getattr(args, "min_face_area_ratio", 0.0) or 0.0)
            if min_ratio > 0.0 and ratio < min_ratio:
                continue
            sim = float(fm.get("sim", 0.0))
            person_w = max(1.0, float(x2 - x1))
            person_h = max(1.0, float(y2 - y1))
            pc_x = 0.5 * (float(x1) + float(x2))
            head_y = float(y1) + 0.18 * person_h
            dx = abs(fc_x - pc_x) / person_w
            dy = abs(fc_y - head_y) / person_h
            score = 1.75 * float(link) + 0.30 * sim + 0.20 * ratio - 0.35 * dx - 0.15 * dy
            candidates.append((score, fi, ti))
            face_valid_tracks[fi].append(ti)
            face_track_boxes[(fi, ti)] = t_xyxy
    if not candidates:
        return {}
    ambiguous_faces = set()
    for fi, tis in face_valid_tracks.items():
        if len(tis) <= 1:
            continue
        ambiguous = False
        for i in range(len(tis)):
            for j in range(i + 1, len(tis)):
                b1 = face_track_boxes.get((fi, tis[i]))
                b2 = face_track_boxes.get((fi, tis[j]))
                if b1 is None or b2 is None:
                    continue
                if iou_xyxy(b1, b2) > 0.0:
                    ambiguous = True
                    break
            if ambiguous:
                break
        if ambiguous:
            ambiguous_faces.add(fi)
    candidates = [c for c in candidates if c[1] not in ambiguous_faces]
    if not candidates:
        return {}
    candidates.sort(key=lambda x: x[0], reverse=True)
    used_faces = set()
    used_tracks = set()
    out: Dict[int, Dict[str, Any]] = {}
    for score, fi, ti in candidates:
        if fi in used_faces or ti in used_tracks:
            continue
        used_faces.add(fi)
        used_tracks.add(ti)
        tid = int(raw_tracks[ti]["tid"])
        out[tid] = recognized_faces[fi]
    return out


@dataclass
class TrackLike:
    track_id: int
    tlbr: Tuple[float, float, float, float]
    det_conf: Optional[float] = None
    last_detection: Optional[Any] = None
    time_since_update: int = 0
    _confirmed: bool = True

    def is_confirmed(self) -> bool:
        return bool(self._confirmed)

    def to_tlbr(self) -> Tuple[float, float, float, float]:
        return self.tlbr


def boxmot_results_to_tracks(res: Any) -> List[TrackLike]:
    if res is None:
        return []
    try:
        arr = np.asarray(res)
    except Exception:
        return []
    if arr.ndim != 2 or arr.shape[0] == 0:
        return []
    if arr.shape[1] < 5:
        return []
    out: List[TrackLike] = []
    for row in arr:
        try:
            x1, y1, x2, y2 = map(float, row[:4])
            tid = int(row[4])
            conf = float(row[5]) if arr.shape[1] > 5 else None
        except Exception:
            continue
        ld = None
        if conf is not None:
            ld = {"confidence": float(conf)}
        out.append(TrackLike(tid, (x1, y1, x2, y2), float(conf) if conf is not None else None, ld, 0, True))
    return out


def resolve_strongsort_reid_weights(args: argparse.Namespace) -> Optional[Path]:
    cand = str(getattr(args, 'strongsort_reid_weights', '') or '').strip()
    if cand and Path(cand).exists():
        return Path(cand)
    cand2 = str(getattr(args, 'reid_weights', '') or '').strip()
    if cand2 and Path(cand2).exists():
        return Path(cand2)
    for nm in (
        'osnet_x1_0_msmt17.pt',
        'osnet_x1_0_market1501.pt',
        'osnet_x0_25_msmt17.pt',
        'osnet_x0_25_market1501.pt',
    ):
        p = Path(nm)
        if p.exists():
            return p
    return None


class IOUTrack:
    def __init__(self, tlwh, tid):
        self.tlwh = np.array(tlwh, dtype=np.float32)
        self.tid = int(tid)
        self.miss = 0


class IOUTracker:
    def __init__(self, max_miss=5, iou_thresh=0.3):
        self.tracks: list[IOUTrack] = []
        self.next_id = 1
        self.max_miss = int(max_miss)
        self.iou_thresh = float(iou_thresh)

    def update(self, dets_tlwh_conf: np.ndarray):
        dets = np.asarray(dets_tlwh_conf, dtype=np.float32)
        if dets.ndim != 2 or dets.shape[1] < 4:
            dets = dets.reshape((0, 5)).astype(np.float32)
        assigned = set()
        for tr in self.tracks:
            tr.miss += 1
            t_x, t_y, t_w, t_h = tr.tlwh
            t_xyxy = np.array([t_x, t_y, t_x + t_w, t_y + t_h], dtype=np.float32)
            best_j, best_iou = -1, 0.0
            for j, d in enumerate(dets):
                if j in assigned:
                    continue
                x, y, w, h = d[:4]
                d_xyxy = np.array([x, y, x + w, y + h], dtype=np.float32)
                s = iou_xyxy(t_xyxy, d_xyxy)
                if s > best_iou:
                    best_iou, best_j = s, j
            if best_j >= 0 and best_iou >= self.iou_thresh:
                tr.tlwh = dets[best_j][:4]
                tr.miss = 0
                assigned.add(best_j)
        for j, d in enumerate(dets):
            if j in assigned:
                continue
            self.tracks.append(IOUTrack(d[:4], self.next_id))
            self.next_id += 1
        self.tracks = [t for t in self.tracks if t.miss <= self.max_miss]
        outs: List[TrackLike] = []
        for t in self.tracks:
            x, y, w, h = map(float, t.tlwh[:4])
            outs.append(TrackLike(int(t.tid), (x, y, x + w, y + h), None, None, 0, True))
        return outs


def make_identity_entry() -> dict:
    return {
        "scores": defaultdict(float),
        "last": "",
        "ttl": 0,
        "face_vis_ttl": 0,
        "last_face_label": "",
        "last_face_sim": 0.0,
        "assigned_name": "",
        "assigned_member_id": -1,
        "assigned_score": 0.0,
        "last_seen_frame": -1,
        "confirmed_face_label": "",
        "confirmed_face_member_id": -1,
        "confirmed_face_sim": 0.0,
        "has_approved_face": False,
        "pending_face_label": "",
        "pending_face_member_id": -1,
        "pending_face_count": 0,
        "pending_face_best_sim": 0.0,
        "locked": False,
        "lock_ttl": 0,
        "lock_label": "",
        "last_good_bbox": None,
        "continuity_break_frames": 0,
    }


def demote_identity_entry(entry: Optional[dict], clear_face: bool = True) -> None:
    if not isinstance(entry, dict):
        return
    scores = entry.get("scores", None)
    if isinstance(scores, dict):
        try:
            scores.clear()
        except Exception:
            entry["scores"] = defaultdict(float)
    else:
        entry["scores"] = defaultdict(float)
    entry["last"] = ""
    entry["ttl"] = 0
    entry["assigned_name"] = ""
    entry["assigned_member_id"] = -1
    entry["assigned_score"] = 0.0
    entry["locked"] = False
    entry["lock_ttl"] = 0
    entry["lock_label"] = ""
    entry["last_good_bbox"] = None
    entry["continuity_break_frames"] = 0
    if clear_face:
        entry["face_vis_ttl"] = 0
        entry["last_face_label"] = ""
        entry["last_face_sim"] = 0.0
        entry["confirmed_face_label"] = ""
        entry["confirmed_face_member_id"] = -1
        entry["confirmed_face_sim"] = 0.0
        entry["has_approved_face"] = False
        entry["pending_face_label"] = ""
        entry["pending_face_member_id"] = -1
        entry["pending_face_count"] = 0
        entry["pending_face_best_sim"] = 0.0


def is_force_face_override(entry: dict, label: str, sim: float, gap: float, det_score: float, args) -> bool:
    label = str(label or "").strip()
    if not label:
        return False
    current_label = str(entry.get("confirmed_face_label", "") or entry.get("assigned_name", "") or entry.get("last", "") or "")
    if (not current_label) or (label == current_label):
        return False
    sim = float(sim or 0.0)
    gap = float(gap or 0.0)
    det_score = float(det_score or 0.0)
    sim_thresh = float(getattr(args, "face_override_thresh", 0.0) or 0.0)
    if sim_thresh <= 0.0:
        sim_thresh = max(float(getattr(args, "face_strong_thresh", 0.65)), float(getattr(args, "face_thresh", 0.55)) + 0.10)
    gap_thresh = float(getattr(args, "face_override_gap", 0.0) or 0.0)
    if gap_thresh <= 0.0:
        gap_thresh = max(float(getattr(args, "face_gap", 0.05)), 0.03)
    det_thresh = float(getattr(args, "face_override_det_score", 0.0) or 0.0)
    if det_thresh <= 0.0:
        det_thresh = max(float(getattr(args, "embed_min_face_det_score", 0.50)), 0.50)
    return (sim >= sim_thresh) and (gap >= gap_thresh) and (det_score >= det_thresh)


def approve_face_candidate_for_track(entry: dict, label: str, member_id: int, sim: float, gap: float, det_score: float, args) -> tuple[str, int, float, bool, bool]:
    label = str(label or "").strip()
    if not label:
        pending_label = str(entry.get("pending_face_label", "") or "")
        if pending_label:
            left = max(0, int(entry.get("pending_face_count", 0)) - 1)
            if left <= 0:
                entry["pending_face_label"] = ""
                entry["pending_face_member_id"] = -1
                entry["pending_face_count"] = 0
                entry["pending_face_best_sim"] = 0.0
            else:
                entry["pending_face_count"] = left
        return "", -1, 0.0, False, False
    member_id = int(member_id) if member_id is not None else -1
    sim = float(sim or 0.0)
    gap = float(gap or 0.0)
    det_score = float(det_score or 0.0)
    current_label = str(entry.get("confirmed_face_label", "") or entry.get("assigned_name", "") or entry.get("last", "") or "")
    force_override = is_force_face_override(entry, label=label, sim=sim, gap=gap, det_score=det_score, args=args)
    lock_active = bool(entry.get("locked", False)) and int(entry.get("lock_ttl", 0)) > 0
    lock_label = str(entry.get("lock_label", "") or current_label or "")
    if lock_active and lock_label and label != lock_label:
        if force_override:
            entry["locked"] = False
            entry["lock_ttl"] = 0
            entry["lock_label"] = ""
        else:
            entry["pending_face_label"] = ""
            entry["pending_face_member_id"] = -1
            entry["pending_face_count"] = 0
            entry["pending_face_best_sim"] = 0.0
            return "", -1, 0.0, False, False
    if current_label and label == current_label:
        entry["confirmed_face_label"] = label
        if member_id > 0:
            entry["confirmed_face_member_id"] = int(member_id)
        entry["confirmed_face_sim"] = max(float(entry.get("confirmed_face_sim", 0.0)), sim)
        entry["has_approved_face"] = True
        entry["pending_face_label"] = ""
        entry["pending_face_member_id"] = -1
        entry["pending_face_count"] = 0
        entry["pending_face_best_sim"] = 0.0
        return label, int(entry.get("confirmed_face_member_id", member_id) or member_id or -1), sim, True, False
    pending_label = str(entry.get("pending_face_label", "") or "")
    if pending_label == label:
        entry["pending_face_count"] = int(entry.get("pending_face_count", 0)) + 1
        entry["pending_face_best_sim"] = max(float(entry.get("pending_face_best_sim", 0.0)), sim)
        if member_id > 0:
            entry["pending_face_member_id"] = int(member_id)
    else:
        entry["pending_face_label"] = label
        entry["pending_face_member_id"] = int(member_id)
        entry["pending_face_count"] = 1
        entry["pending_face_best_sim"] = sim
    strong_threshold = max(float(getattr(args, "face_strong_thresh", 0.65)), float(getattr(args, "face_thresh", 0.55)) + 0.10)
    strong_now = (sim >= strong_threshold) and (gap >= max(float(getattr(args, "face_gap", 0.05)), 0.03))
    if current_label:
        required_hits = max(2, int(getattr(args, "face_switch_confirm_hits", 3)))
        accept = bool(force_override) or (int(entry.get("pending_face_count", 0)) >= required_hits)
    else:
        required_hits = max(1, int(getattr(args, "face_confirm_hits", 2)))
        accept = strong_now or (int(entry.get("pending_face_count", 0)) >= required_hits)
    if not accept:
        return "", -1, 0.0, False, bool(force_override)
    approved_mid = int(entry.get("pending_face_member_id", member_id) or member_id or -1)
    approved_sim = max(sim, float(entry.get("pending_face_best_sim", sim)))
    entry["confirmed_face_label"] = label
    entry["confirmed_face_member_id"] = approved_mid
    entry["confirmed_face_sim"] = approved_sim
    entry["has_approved_face"] = True
    entry["pending_face_label"] = ""
    entry["pending_face_member_id"] = -1
    entry["pending_face_count"] = 0
    entry["pending_face_best_sim"] = 0.0
    return label, approved_mid, approved_sim, True, bool(force_override)


class CameraNameOwner:
    def __init__(self, args):
        hold_frames = int(getattr(args, "camera_name_hold_frames", 0) or 0)
        if hold_frames <= 0:
            hold_frames = max(
                8,
                int(getattr(args, "face_every_n", 1) or 1) * max(2, int(getattr(args, "face_switch_confirm_hits", 3))),
                int(getattr(args, "n_init", 3) or 3) + 2,
            )
        self.hold_frames = int(max(1, hold_frames))
        self.switch_margin = float(max(0.0, getattr(args, "camera_name_switch_margin", 0.05) or 0.0))
        self.switch_hits = int(max(2, getattr(args, "camera_name_switch_hits", 2) or 2))
        self._state: Dict[str, Dict[str, Any]] = {}

    def _cleanup(self, frame_idx: int) -> None:
        dead = []
        for name, st in self._state.items():
            last_frame = int(st.get("last_frame", -10**9))
            if (int(frame_idx) - last_frame) > int(self.hold_frames * 4):
                dead.append(name)
        for name in dead:
            self._state.pop(name, None)

    def allow(self, name: str, tid: int, score: float, frame_idx: int, force: bool = False) -> bool:
        name = str(name or "").strip()
        if not name:
            return False
        tid = int(tid)
        score = float(score or 0.0)
        frame_idx = int(frame_idx)
        force = bool(force)
        self._cleanup(frame_idx)
        st = self._state.get(name)
        if st is None:
            self._state[name] = {
                "owner_tid": tid, "owner_score": score, "last_frame": frame_idx,
                "pending_tid": -1, "pending_hits": 0, "pending_best_score": 0.0,
            }
            return True
        owner_tid = int(st.get("owner_tid", -1))
        owner_score = float(st.get("owner_score", 0.0))
        last_frame = int(st.get("last_frame", -10**9))
        if owner_tid == tid:
            st["owner_score"] = max(owner_score * 0.90, score)
            st["last_frame"] = frame_idx
            st["pending_tid"] = -1
            st["pending_hits"] = 0
            st["pending_best_score"] = 0.0
            return True
        if force:
            self._state[name] = {
                "owner_tid": tid, "owner_score": score, "last_frame": frame_idx,
                "pending_tid": -1, "pending_hits": 0, "pending_best_score": 0.0,
            }
            return True
        if (frame_idx - last_frame) > self.hold_frames:
            self._state[name] = {
                "owner_tid": tid, "owner_score": score, "last_frame": frame_idx,
                "pending_tid": -1, "pending_hits": 0, "pending_best_score": 0.0,
            }
            return True
        if int(st.get("pending_tid", -1)) == tid:
            st["pending_hits"] = int(st.get("pending_hits", 0)) + 1
            st["pending_best_score"] = max(float(st.get("pending_best_score", 0.0)), score)
        else:
            st["pending_tid"] = tid
            st["pending_hits"] = 1
            st["pending_best_score"] = score
        if float(st.get("pending_best_score", 0.0)) >= (owner_score + self.switch_margin) and int(st.get("pending_hits", 0)) >= self.switch_hits:
            st["owner_tid"] = tid
            st["owner_score"] = float(st.get("pending_best_score", score))
            st["last_frame"] = frame_idx
            st["pending_tid"] = -1
            st["pending_hits"] = 0
            st["pending_best_score"] = 0.0
            return True
        return False

    def owner_tid(self, name: str, frame_idx: int) -> Optional[int]:
        name = str(name or "").strip()
        if not name:
            return None
        self._cleanup(int(frame_idx))
        st = self._state.get(name)
        if not st:
            return None
        if (int(frame_idx) - int(st.get("last_frame", -10**9))) > self.hold_frames:
            return None
        return int(st.get("owner_tid", -1))


def update_track_identity(
    state: dict,
    tid: int,
    face_label: str,
    face_sim: float,
    body_label: str,
    body_sim: float,
    decay: float,
    min_score: float,
    margin: float,
    ttl_reset: int,
    w_face: float,
    w_body: float,
    lock_frames: int,
    lock_face_thresh: float,
) -> tuple[str, float, dict]:
    entry = state.setdefault(tid, make_identity_entry())
    scores = entry["scores"]
    for k in list(scores.keys()):
        scores[k] *= float(decay)
        if scores[k] < 1e-6:
            del scores[k]
    face_label = str(face_label or "").strip()
    face_sim = float(face_sim or 0.0)
    if face_label:
        scores[face_label] += max(0.0, face_sim) * float(w_face)
        for k in list(scores.keys()):
            if k == face_label:
                continue
            scores[k] *= 0.50
            if scores[k] < 1e-6:
                del scores[k]
        entry["last"] = face_label
        entry["ttl"] = int(max(0, int(ttl_reset)))
        if face_sim >= float(lock_face_thresh):
            entry["locked"] = True
            entry["lock_ttl"] = int(max(1, int(lock_frames)))
            entry["lock_label"] = face_label
        return entry["last"], float(max(scores.get(face_label, 0.0), face_sim)), entry
    if bool(entry.get("locked", False)):
        left = int(entry.get("lock_ttl", 0)) - 1
        entry["lock_ttl"] = max(0, left)
        lock_label = str(entry.get("lock_label", "") or entry.get("last", "") or entry.get("assigned_name", "") or "")
        if lock_label:
            entry["last"] = lock_label
            entry["ttl"] = int(max(int(entry.get("ttl", 0)), 1))
        if entry["lock_ttl"] <= 0:
            entry["locked"] = False
            entry["lock_label"] = ""
        return entry["last"], float(max(scores.get(entry["last"], 0.0), entry.get("confirmed_face_sim", 0.0))), entry
    if entry.get("last"):
        entry["ttl"] = max(0, int(entry.get("ttl", 0)) - 1)
        return str(entry.get("last", "")), float(max(scores.get(entry.get("last", ""), 0.0), entry.get("confirmed_face_sim", 0.0))), entry
    return "", 0.0, entry


def init_face_engine(use_face: bool, device: str, face_model: str, det_w: int, det_h: int, face_provider: str, ort_log: bool):
    if not use_face:
        return None
    if not INSIGHT_OK:
        print("[WARN] insightface not installed; face recognition disabled.")
        return None
    try:
        is_cuda = ("cuda" in device.lower()) and torch.cuda.is_available()
        cuda_ok = _cuda_ep_loadable()
        if ort is not None and ort_log:
            try:
                print(f"[INFO] ORT available providers: {ort.get_available_providers()}")
            except Exception:
                pass
        providers = ["CPUExecutionProvider"]
        if face_provider == "cuda":
            if cuda_ok:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
            else:
                print("[INFO] Requested CUDA EP, but not loadable. Using CPU.")
        elif face_provider == "auto":
            if is_cuda and cuda_ok:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        app = FaceAnalysis(name=face_model, providers=providers)
        ctx_id = 0 if providers[0].startswith("CUDA") else -1
        try:
            app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
        except TypeError:
            app.prepare(ctx_id=ctx_id)
        print(f"[INIT] InsightFace ready (model={face_model}, providers={providers}).")
        return app
    except Exception as e:
        print("[WARN] InsightFace init failed:", e)
        return None


def _yolo_forward_safe(yolo, frame, args):
    with _yolo_lock, torch.inference_mode():
        try:
            return yolo(frame, conf=args.conf, iou=args.iou, verbose=False, device=args.device, half=args.half,
                        imgsz=int(args.yolo_imgsz) if int(args.yolo_imgsz) > 0 else None)
        except TypeError:
            return yolo(frame, conf=args.conf, iou=args.iou, verbose=False, device=args.device, half=args.half)
        except Exception as e:
            if args.half:
                print("[YOLO] FP16 failed, retrying in FP32 once:", e)
                args.half = False
                return yolo(frame, conf=args.conf, iou=args.iou, verbose=False, device=args.device, half=False)
            raise


def extract_body_embeddings_batch(extractor, crops_bgr: List[np.ndarray], device_is_cuda: bool, use_half: bool) -> Optional[np.ndarray]:
    if extractor is None or not crops_bgr:
        return None
    crops_rgb: List[np.ndarray] = []
    for c in crops_bgr:
        if c is None or c.size == 0:
            crops_rgb.append(np.zeros((1, 1, 3), dtype=np.uint8))
            continue
        crops_rgb.append(_to_rgb(c))
    with _reid_lock, torch.inference_mode():
        if device_is_cuda and use_half:
            try:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    feats = extractor(crops_rgb)
            except Exception:
                feats = extractor(crops_rgb)
        else:
            feats = extractor(crops_rgb)
    try:
        if isinstance(feats, (list, tuple)):
            feats_arr = []
            for f in feats:
                f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
                feats_arr.append(np.asarray(f, dtype=np.float32).reshape(-1))
            mat = np.stack(feats_arr, axis=0)
        else:
            f = feats.detach().cpu().numpy() if hasattr(feats, "detach") else np.asarray(feats)
            mat = np.asarray(f, dtype=np.float32)
            if mat.ndim == 1:
                mat = mat.reshape(1, -1)
        if mat.ndim != 2 or mat.shape[1] != EXPECTED_DIM:
            return None
        if not np.isfinite(mat).all():
            return None
        return l2_normalize_rows(mat)
    except Exception:
        return None


def best_body_label_from_emb(emb: np.ndarray | None, people: list[PersonEntry], topk: int = 3) -> tuple[str | None, float, float]:
    if emb is None:
        return None, 0.0, 0.0
    q = l2_normalize(np.asarray(emb, dtype=np.float32).reshape(-1))
    if q.size != EXPECTED_DIM or not np.isfinite(q).all():
        return None, 0.0, 0.0
    k_req = max(1, int(topk))
    scored: list[tuple[str, float]] = []
    for p in people:
        bank = p.body_bank
        if bank is None or bank.size == 0:
            continue
        try:
            sims = bank @ q
        except Exception:
            continue
        if sims.ndim != 1 or sims.size == 0:
            continue
        k = min(k_req, sims.size)
        if k <= 1:
            score = float(np.max(sims))
        else:
            top_vals = np.partition(sims, -k)[-k:]
            score = float(np.mean(top_vals))
        scored.append((p.name, score))
    if not scored:
        return None, 0.0, 0.0
    scored.sort(key=lambda x: x[1], reverse=True)
    best_label, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    return best_label, float(best_score), float(second_score)


class AdaptiveQueueStream:
    def __init__(
        self,
        src: str,
        queue_size: int,
        rtsp_transport: str,
        use_opencv: bool = True,
        freeze_seconds: float = 2.0,
        open_timeout_ms: int = 1000,
        read_timeout_ms: int = 1000,
        reconnect_base_delay: float = 0.5,
        reconnect_max_delay: float = 3.0,
        reconnect_jitter: float = 0.10,
        reconnect_log_interval: float = 2.0,
    ):
        self.src = src
        self.use_opencv = bool(use_opencv)
        self.queue_size = max(1, int(queue_size))
        self.rtsp_transport = str(rtsp_transport or "tcp")
        self.is_file_source = bool(_is_probably_file_source(src))
        self.freeze_seconds = float(max(0.5, float(freeze_seconds or 2.0)))
        self.open_timeout_ms = int(max(0, int(open_timeout_ms or 0)))
        self.read_timeout_ms = int(max(0, int(read_timeout_ms or 0)))
        self.reconnect_base_delay = float(max(0.1, float(reconnect_base_delay or 0.5)))
        self.reconnect_max_delay = float(max(self.reconnect_base_delay, float(reconnect_max_delay or 3.0)))
        self.reconnect_jitter = float(max(0.0, min(0.50, float(reconnect_jitter or 0.0))))
        self.reconnect_log_interval = float(max(0.0, float(reconnect_log_interval or 0.0)))
        if isinstance(src, str) and src.lower().startswith("rtsp"):
            try:
                key = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
                opt = os.environ.get(key, "") or ""
                us = int(max(0, int(self.read_timeout_ms)) * 1000) if self.read_timeout_ms > 0 else 0
                parts = [p for p in opt.split("|") if p.strip()]
                kv = []
                for p in parts:
                    if ";" in p:
                        k, v = p.split(";", 1)
                        kv.append((k.strip(), v.strip()))
                    else:
                        kv.append((p.strip(), ""))
                def upsert(k: str, v: str) -> None:
                    for i, (kk, vv) in enumerate(kv):
                        if kk == k:
                            kv[i] = (kk, str(v))
                            return
                    kv.append((k, str(v)))
                if us > 0:
                    upsert("stimeout", us)
                    upsert("rw_timeout", us)
                new_opt = "|".join([f"{k};{v}" if v != "" else k for k, v in kv])
                os.environ[key] = new_opt
            except Exception:
                pass
        src_use = src
        if isinstance(src, str) and src.lower().startswith("rtsp"):
            sep = "&" if "?" in src_use else "?"
            src_use = f"{src_use}{sep}rtsp_transport={self.rtsp_transport}"
        self.src_use = src_use
        self.q: queue.Queue = queue.Queue(maxsize=self.queue_size)
        self.stop_flag = threading.Event()
        self.dropped = 0
        self.read_dropped = 0
        self._cap_lock = threading.Lock()
        self.cap: Optional[cv2.VideoCapture] = None
        self._connected = False
        self._eof = False
        self._source_fps = 0.0
        self._last_frame_mono = time.monotonic()
        self._next_reconnect_mono = 0.0
        self._reconnect_failures = 0
        self._last_log_mono = 0.0
        self._open_capture(initial=True)
        self.thread: Optional[threading.Thread] = None
        if not self.is_file_source:
            self.thread = threading.Thread(target=self._loop, daemon=True)
            self.thread.start()

    def _log(self, msg: str) -> None:
        if not msg:
            return
        if self.reconnect_log_interval <= 0:
            print(msg)
            return
        now = time.monotonic()
        if (now - float(self._last_log_mono)) >= float(self.reconnect_log_interval):
            self._last_log_mono = now
            print(msg)

    def _drop_oldest(self) -> None:
        try:
            _ = self.q.get_nowait()
            self.dropped += 1
        except queue.Empty:
            return

    def _drain_queue(self) -> None:
        try:
            while True:
                _ = self.q.get_nowait()
        except queue.Empty:
            return

    def _create_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture()
        try:
            if self.open_timeout_ms > 0 and hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, float(self.open_timeout_ms))
        except Exception:
            pass
        try:
            if self.read_timeout_ms > 0 and hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, float(self.read_timeout_ms))
        except Exception:
            pass
        return cap

    def _open_once(self, backend: Optional[int]) -> Optional[cv2.VideoCapture]:
        cap = self._create_capture()
        opened = False
        try:
            if backend is None:
                opened = bool(cap.open(self.src_use))
            else:
                opened = bool(cap.open(self.src_use, backend))
        except Exception:
            opened = False
        if not opened:
            try:
                cap.release()
            except Exception:
                pass
            return None
        return cap

    def _open_capture(self, initial: bool = False) -> bool:
        if self.use_opencv:
            if self.is_file_source:
                backends = [None]
                if hasattr(cv2, "CAP_FFMPEG"):
                    backends.append(cv2.CAP_FFMPEG)
            else:
                backends = [cv2.CAP_FFMPEG] if hasattr(cv2, "CAP_FFMPEG") else []
                backends.append(None)
        else:
            backends = [None]
        new_cap: Optional[cv2.VideoCapture] = None
        for backend in backends:
            new_cap = self._open_once(backend)
            if new_cap is not None:
                break
        if new_cap is None:
            if initial:
                self._log(f"[SRC] cannot open source initially: {self.src}")
            return False
        try:
            new_cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self.queue_size))
        except Exception:
            pass
        old_cap: Optional[cv2.VideoCapture] = None
        with self._cap_lock:
            old_cap = self.cap
            self.cap = new_cap
            self._connected = True
            self._eof = False
        try:
            if old_cap is not None:
                old_cap.release()
        except Exception:
            pass
        if self.is_file_source:
            try:
                fps = float(new_cap.get(cv2.CAP_PROP_FPS) or 0.0)
                self._source_fps = fps if math.isfinite(fps) and fps > 0.0 else 0.0
            except Exception:
                self._source_fps = 0.0
        self._last_frame_mono = time.monotonic()
        self._drain_queue()
        return True

    def _backoff_delay(self, failures: int) -> float:
        n = max(0, int(failures))
        base = float(self.reconnect_base_delay)
        cap = float(self.reconnect_max_delay)
        try:
            exp = min(max(0, n - 1), 10)
            delay = base * (2.0 ** exp)
        except Exception:
            delay = base
        delay = float(min(cap, max(base, delay)))
        if self.reconnect_jitter > 0:
            j = (random.random() * 2.0 - 1.0) * float(self.reconnect_jitter) * delay
            delay = max(0.0, delay + j)
        return float(delay)

    def _maybe_reconnect(self, reason: str) -> None:
        if self.is_file_source:
            return
        now = time.monotonic()
        if now < float(self._next_reconnect_mono):
            return
        with self._cap_lock:
            cap = self.cap
            self.cap = None
            self._connected = False
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass
        ok = self._open_capture(initial=False)
        if ok:
            self._reconnect_failures = 0
            self._next_reconnect_mono = 0.0
            self._log(f"[SRC] reconnected: {self.src} (reason={reason})")
        else:
            self._reconnect_failures += 1
            delay = self._backoff_delay(self._reconnect_failures)
            self._next_reconnect_mono = time.monotonic() + delay
            self._log(f"[SRC] reconnect failed: {self.src} (reason={reason}, failures={self._reconnect_failures}, next_retry={delay:.2f}s)")

    def _loop(self):
        while not self.stop_flag.is_set():
            if not self._connected or self.cap is None:
                self._maybe_reconnect(reason="disconnected")
                self.stop_flag.wait(0.02)
                continue
            with self._cap_lock:
                cap = self.cap
            ok = False
            frame = None
            try:
                ok, frame = cap.read() if cap is not None else (False, None)
            except Exception:
                ok, frame = False, None
            now_mono = time.monotonic()
            if ok and frame is not None and getattr(frame, "size", 0) != 0:
                self._last_frame_mono = now_mono
                item = (frame, time.time())
                try:
                    self.q.put_nowait(item)
                except queue.Full:
                    self._drop_oldest()
                    try:
                        self.q.put_nowait(item)
                    except queue.Full:
                        self._drop_oldest()
                continue
            if (now_mono - float(self._last_frame_mono)) >= float(self.freeze_seconds):
                self._maybe_reconnect(reason=f"freeze>{self.freeze_seconds:.2f}s")
            else:
                self.stop_flag.wait(0.002)

    def _read_file_frame(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self._eof:
            return False, None, 0.0
        with self._cap_lock:
            cap = self.cap
        if cap is None:
            self._eof = True
            self._connected = False
            return False, None, 0.0
        ok = False
        frame = None
        try:
            ok, frame = cap.read()
        except Exception:
            ok, frame = False, None
        if ok and frame is not None and getattr(frame, "size", 0) != 0:
            self._last_frame_mono = time.monotonic()
            return True, frame, time.time()
        with self._cap_lock:
            old_cap = self.cap
            self.cap = None
            self._connected = False
            self._eof = True
        try:
            if old_cap is not None:
                old_cap.release()
        except Exception:
            pass
        self._log(f"[SRC] file completed: {self.src}")
        return False, None, 0.0

    def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
        if self.is_file_source:
            return self._read_file_frame()
        try:
            frame, ts = self.q.get(timeout=0.1)
            return True, frame, float(ts)
        except queue.Empty:
            return False, None, 0.0

    def qsize(self) -> int:
        if self.is_file_source:
            return 0
        try:
            return int(self.q.qsize())
        except Exception:
            return 0

    def is_opened(self) -> bool:
        with self._cap_lock:
            return bool(self._connected) and (self.cap is not None)

    def is_finished(self) -> bool:
        return bool(self.is_file_source and self._eof)

    def release(self):
        self.stop_flag.set()
        try:
            if self.thread is not None:
                self.thread.join(timeout=2.0)
        except Exception:
            pass
        with self._cap_lock:
            cap = self.cap
            self.cap = None
            self._connected = False
            self._eof = True if self.is_file_source else self._eof
        try:
            if cap is not None:
                cap.release()
        except Exception:
            pass


class RenderedFrame:
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

    def set(self, frame: np.ndarray, meta: Optional[Dict[str, Any]] = None):
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
            return self._frm, self._ts, dict(self._meta)

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
        if frame is None or ts <= 0:
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
            if frame_ref is None or ts_ref <= 0:
                return None, 0.0
            q = int(max(30, min(95, int(jpeg_quality))))
            ok, enc = cv2.imencode(".jpg", frame_ref, [int(cv2.IMWRITE_JPEG_QUALITY), q])
            if not ok:
                return None, ts_ref
            jpg = enc.tobytes()
            with self._lock:
                if float(self._ts) == ts_ref:
                    self._jpeg = jpg
                    self._jpeg_ts = ts_ref
            return jpg, ts_ref


@dataclass
class _ActiveSession:
    start_ts: float
    last_seen_ts: float
    conf_max: float = 0.0
    conf_sum: float = 0.0
    conf_n: int = 0


@dataclass
class _TrackletRecord:
    tracklet_id: int
    stream_id: int
    camera_db_id: int
    track_id: int
    member_id: int
    name: str
    start_ts: float
    end_ts: float
    frame_count: int
    face_sim_max: float
    face_sim_sum: float
    face_sim_n: int
    start_bbox: Tuple[int, int, int, int]
    end_bbox: Tuple[int, int, int, int]





def _clone_tracklet_record(rec: _TrackletRecord) -> _TrackletRecord:
    return _TrackletRecord(
        tracklet_id=int(rec.tracklet_id),
        stream_id=int(rec.stream_id),
        camera_db_id=int(rec.camera_db_id),
        track_id=int(rec.track_id),
        member_id=int(rec.member_id),
        name=str(rec.name or ""),
        start_ts=float(rec.start_ts),
        end_ts=float(rec.end_ts),
        frame_count=int(rec.frame_count),
        face_sim_max=float(rec.face_sim_max),
        face_sim_sum=float(rec.face_sim_sum),
        face_sim_n=int(rec.face_sim_n),
        start_bbox=tuple(int(v) for v in rec.start_bbox),
        end_bbox=tuple(int(v) for v in rec.end_bbox),
    )


@dataclass
class _NormalizedDataTask:
    action: str
    record: _TrackletRecord


class NormalizedDataDBWriter:
    """
    Writes only known-person tracklet timings into normalized_data.

    Mapping used for the provided schema:
    - member_id: known member id only
    - guest_temp_id: NULL
    - camera_id: DB camera id aligned with --camera-ids
    - movement_type: 1 when the tracklet row is opened, updated to 2 when exit_ts is written
    - entry_ts / exit_ts: tracklet start / end timestamps, same timing basis as the CSV report
    - average_match_value: average face similarity as an integer percent (0..100), or NULL if unavailable
    """

    def __init__(self, db_url: str, max_queue: int = 4096, warn_interval_s: float = 5.0):
        self.db_url = str(db_url)
        self._stop = threading.Event()
        self._q: queue.Queue = queue.Queue(maxsize=max(64, int(max_queue)))
        self._active_row_id_by_tracklet: Dict[int, int] = {}
        self._lock = threading.Lock()
        self._warn_interval_s = float(max(0.0, float(warn_interval_s or 0.0)))
        self._last_warn_mono = 0.0

        Base = declarative_base()

        class NormalizedDataRow(Base):
            __tablename__ = "normalized_data"
            id = Column(BigInteger, primary_key=True)
            member_id = Column(Integer, nullable=True)
            guest_temp_id = Column(String(64), nullable=True)
            camera_id = Column(Integer, nullable=False)
            movement_type = Column(Integer, nullable=False)
            entry_ts = Column(DateTime(timezone=True), nullable=False)
            exit_ts = Column(DateTime(timezone=True), nullable=True)
            average_match_value = Column(Integer, nullable=True)

        self.NormalizedDataRow = NormalizedDataRow
        self.engine = create_engine(self.db_url, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine, expire_on_commit=False)
        self._thr = threading.Thread(target=self._loop, daemon=True)
        self._thr.start()

    def _warn(self, msg: str) -> None:
        if not msg:
            return
        if self._warn_interval_s <= 0:
            print(msg)
            return
        now = time.monotonic()
        if (now - float(self._last_warn_mono)) >= float(self._warn_interval_s):
            self._last_warn_mono = now
            print(msg)

    @staticmethod
    def _ts_to_db_dt(ts: Any) -> Optional[datetime]:
        try:
            ts_f = float(ts)
            if (not math.isfinite(ts_f)) or ts_f <= 0.0:
                return None
            # Keep the same wall-clock basis as the CSV formatting while storing a tz-aware value.
            return datetime.fromtimestamp(ts_f, tz=timezone.utc).astimezone()
        except Exception:
            return None

    @staticmethod
    def _avg_match_value(rec: _TrackletRecord) -> Optional[int]:
        try:
            n = int(rec.face_sim_n)
            if n <= 0:
                return None
            avg = float(rec.face_sim_sum) / float(n)
            if not math.isfinite(avg):
                return None
            return int(max(0, min(100, round(avg * 100.0))))
        except Exception:
            return None

    @staticmethod
    def _should_open(rec: _TrackletRecord) -> bool:
        try:
            return int(rec.member_id) > 0 and bool(str(rec.name or "").strip()) and int(rec.camera_db_id) > 0
        except Exception:
            return False

    def tracklet_started(self, rec: _TrackletRecord) -> None:
        if rec is None or (not self._should_open(rec)):
            return
        task = _NormalizedDataTask(action="open", record=_clone_tracklet_record(rec))
        try:
            self._q.put_nowait(task)
        except queue.Full:
            self._handle_task(task)

    def tracklet_closed(self, rec: _TrackletRecord) -> None:
        if rec is None:
            return
        task = _NormalizedDataTask(action="close", record=_clone_tracklet_record(rec))
        try:
            self._q.put_nowait(task)
        except queue.Full:
            self._handle_task(task)

    def _loop(self) -> None:
        while (not self._stop.is_set()) or (not self._q.empty()):
            try:
                task: _NormalizedDataTask = self._q.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._handle_task(task)
            except Exception as e:
                self._warn(f"[NORMALIZED] task failed: {e}")
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass

    def _handle_task(self, task: _NormalizedDataTask) -> None:
        if task is None or task.record is None:
            return
        action = str(task.action or "").strip().lower()
        if action == "open":
            self._handle_open(task.record)
            return
        if action == "close":
            self._handle_close(task.record)
            return

    def _handle_open(self, rec: _TrackletRecord) -> None:
        if not self._should_open(rec):
            return
        row_id = 0
        entry_dt = self._ts_to_db_dt(rec.start_ts) or datetime.now().astimezone()
        avg_match_value = self._avg_match_value(rec)
        try:
            with self.Session() as session:
                row = self.NormalizedDataRow(
                    member_id=int(rec.member_id),
                    guest_temp_id=None,
                    camera_id=int(rec.camera_db_id),
                    movement_type=1,
                    entry_ts=entry_dt,
                    exit_ts=None,
                    average_match_value=avg_match_value,
                )
                session.add(row)
                session.flush()
                row_id = int(getattr(row, "id", 0) or 0)
                session.commit()
        except Exception as e:
            self._warn(
                f"[NORMALIZED] insert failed member_id={int(rec.member_id)} camera_id={int(rec.camera_db_id)} "
                f"tracklet_id={int(rec.tracklet_id)}: {e}"
            )
            return
        if row_id > 0:
            with self._lock:
                self._active_row_id_by_tracklet[int(rec.tracklet_id)] = int(row_id)

    def _handle_close(self, rec: _TrackletRecord) -> None:
        with self._lock:
            row_id = int(self._active_row_id_by_tracklet.pop(int(rec.tracklet_id), 0) or 0)
        if row_id <= 0:
            return
        end_dt = self._ts_to_db_dt(rec.end_ts) or self._ts_to_db_dt(rec.start_ts) or datetime.now().astimezone()
        avg_match_value = self._avg_match_value(rec)
        try:
            with self.Session() as session:
                row = session.get(self.NormalizedDataRow, row_id)
                if row is None:
                    return
                row.member_id = int(rec.member_id) if int(rec.member_id) > 0 else row.member_id
                if int(getattr(row, "camera_id", 0) or 0) <= 0 and int(rec.camera_db_id) > 0:
                    row.camera_id = int(rec.camera_db_id)
                entry_dt = getattr(row, "entry_ts", None)
                if entry_dt is not None and end_dt < entry_dt:
                    end_dt = entry_dt
                row.exit_ts = end_dt
                row.movement_type = 2
                if avg_match_value is not None:
                    row.average_match_value = avg_match_value
                session.commit()
        except Exception as e:
            self._warn(
                f"[NORMALIZED] update failed row_id={row_id} member_id={int(rec.member_id)} "
                f"camera_id={int(rec.camera_db_id)} tracklet_id={int(rec.tracklet_id)}: {e}"
            )

    def close(self) -> None:
        self._stop.set()
        try:
            self._thr.join(timeout=5.0)
        except Exception:
            pass
        while True:
            try:
                task: _NormalizedDataTask = self._q.get_nowait()
            except queue.Empty:
                break
            try:
                self._handle_task(task)
            except Exception as e:
                self._warn(f"[NORMALIZED] drain failed: {e}")
            finally:
                try:
                    self._q.task_done()
                except Exception:
                    pass

class TrackletSummaryReport:
    report_level = "tracklet"
    is_segmented_report = False

    def __init__(self, num_cams: int, gap_seconds: float = 2.0, time_format: str = "%H:%M:%S",
                 include_unknown: bool = False, normalized_writer: Optional[NormalizedDataDBWriter] = None):
        self.num_cams = int(max(1, num_cams))
        self.gap_seconds = float(max(0.0, gap_seconds))
        self.time_format = str(time_format or "%H:%M:%S")
        self.include_unknown = bool(include_unknown)
        self._normalized_writer = normalized_writer
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._disabled = False
        self._next_tracklet_id = 1
        self._active: Dict[Tuple[int, int], _TrackletRecord] = {}
        self._logs: List[_TrackletRecord] = []

    def stop(self) -> None:
        with self._lock:
            self._disabled = True

    @staticmethod
    def _fallback_path(path: str, suffix: str) -> str:
        try:
            p = Path(path)
            return str(p.with_name(p.stem + suffix + p.suffix))
        except Exception:
            return str(path) + suffix

    @staticmethod
    def _bbox_to_str(bbox: Tuple[int, int, int, int]) -> str:
        try:
            x1, y1, x2, y2 = map(int, bbox)
            return f"{x1},{y1},{x2},{y2}"
        except Exception:
            return ""

    def _fmt_time(self, ts: float) -> str:
        try:
            return datetime.fromtimestamp(float(ts)).strftime(self.time_format)
        except Exception:
            return ""

    @staticmethod
    def _safe_float(x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
            return v if math.isfinite(v) else float(default)
        except Exception:
            return float(default)

    @staticmethod
    def _safe_int(x: Any, default: int = -1) -> int:
        try:
            return int(x)
        except Exception:
            return int(default)

    def _normalize_event(self, ev: Any) -> Optional[Tuple[int, Tuple[int, int, int, int], str, float, int, bool]]:
        tid = -1
        bbox = None
        name = ""
        face_sim = 0.0
        member_id = -1
        is_known = False
        if isinstance(ev, dict):
            tid = self._safe_int(ev.get("tid", -1), -1)
            bbox0 = ev.get("bbox", None)
            if isinstance(bbox0, (list, tuple)) and len(bbox0) >= 4:
                try:
                    bbox = tuple(int(v) for v in bbox0[:4])
                except Exception:
                    bbox = None
            else:
                try:
                    bbox = (
                        self._safe_int(ev.get("x1", 0), 0),
                        self._safe_int(ev.get("y1", 0), 0),
                        self._safe_int(ev.get("x2", 0), 0),
                        self._safe_int(ev.get("y2", 0), 0),
                    )
                except Exception:
                    bbox = None
            name = str(ev.get("name", "") or "").strip()
            face_sim = self._safe_float(ev.get("face_sim", ev.get("sim", 0.0)), 0.0)
            member_id = self._safe_int(ev.get("member_id", -1), -1)
            is_known = bool(ev.get("is_known", bool(name)))
        elif isinstance(ev, (list, tuple)) and len(ev) >= 8:
            tid = self._safe_int(ev[0], -1)
            try:
                bbox = (int(ev[1]), int(ev[2]), int(ev[3]), int(ev[4]))
            except Exception:
                bbox = None
            name = str(ev[5] or "").strip()
            face_sim = self._safe_float(ev[6], 0.0)
            member_id = self._safe_int(ev[7], -1)
            is_known = bool(ev[8]) if len(ev) > 8 else bool(name)
        if tid < 0 or bbox is None:
            return None
        try:
            x1, y1, x2, y2 = map(int, bbox)
        except Exception:
            return None
        if x2 <= x1 or y2 <= y1:
            return None
        if not is_known and name:
            is_known = True
        if not name and (not self.include_unknown):
            return None
        if not name:
            member_id = -1
        return int(tid), (x1, y1, x2, y2), str(name), float(face_sim), int(member_id), bool(is_known)

    def _make_record(self, *, tracklet_id: int, stream_id: int, camera_db_id: int, track_id: int, member_id: int,
                     name: str, start_ts: float, end_ts: float, frame_count: int, face_sim_max: float,
                     face_sim_sum: float, face_sim_n: int, start_bbox: Tuple[int, int, int, int],
                     end_bbox: Tuple[int, int, int, int]) -> _TrackletRecord:
        return _TrackletRecord(
            tracklet_id=int(tracklet_id),
            stream_id=int(stream_id),
            camera_db_id=int(camera_db_id),
            track_id=int(track_id),
            member_id=int(member_id),
            name=str(name or ""),
            start_ts=float(start_ts),
            end_ts=float(end_ts),
            frame_count=int(max(1, frame_count)),
            face_sim_max=float(max(0.0, face_sim_max)),
            face_sim_sum=float(max(0.0, face_sim_sum)),
            face_sim_n=int(max(0, face_sim_n)),
            start_bbox=tuple(int(v) for v in start_bbox),
            end_bbox=tuple(int(v) for v in end_bbox),
        )

    def _start_tracklet_locked(self, stream_id: int, camera_db_id: int, track_id: int, name: str, member_id: int,
                               ts: float, bbox: Tuple[int, int, int, int], face_sim: float) -> _TrackletRecord:
        sim = float(max(0.0, face_sim))
        rec = self._make_record(
            tracklet_id=int(self._next_tracklet_id),
            stream_id=int(stream_id),
            camera_db_id=int(camera_db_id),
            track_id=int(track_id),
            member_id=int(member_id if name else -1),
            name=str(name or ""),
            start_ts=float(ts),
            end_ts=float(ts),
            frame_count=1,
            face_sim_max=float(sim),
            face_sim_sum=float(sim),
            face_sim_n=1 if sim > 0.0 else 0,
            start_bbox=bbox,
            end_bbox=bbox,
        )
        self._next_tracklet_id += 1
        self._active[(int(stream_id), int(track_id))] = rec
        if self._normalized_writer is not None:
            try:
                self._normalized_writer.tracklet_started(rec)
            except Exception as e:
                print(f"[NORMALIZED] start enqueue failed: {e}")
        return rec

    def _extend_tracklet_locked(self, rec: _TrackletRecord, ts: float, bbox: Tuple[int, int, int, int],
                                face_sim: float, camera_db_id: int) -> None:
        rec.end_ts = float(max(float(rec.end_ts), float(ts)))
        rec.frame_count = int(max(1, int(rec.frame_count) + 1))
        rec.end_bbox = tuple(int(v) for v in bbox)
        if int(rec.camera_db_id) <= 0 and int(camera_db_id) > 0:
            rec.camera_db_id = int(camera_db_id)
        sim = float(max(0.0, face_sim))
        if sim > 0.0:
            rec.face_sim_max = max(float(rec.face_sim_max), sim)
            rec.face_sim_sum = float(rec.face_sim_sum) + sim
            rec.face_sim_n = int(rec.face_sim_n) + 1

    def _finalize_tracklet_locked(self, key: Tuple[int, int]) -> None:
        rec = self._active.pop((int(key[0]), int(key[1])), None)
        if rec is None:
            return
        if float(rec.end_ts) < float(rec.start_ts):
            rec.end_ts = float(rec.start_ts)
        rec_copy = self._make_record(
            tracklet_id=int(rec.tracklet_id),
            stream_id=int(rec.stream_id),
            camera_db_id=int(rec.camera_db_id),
            track_id=int(rec.track_id),
            member_id=int(rec.member_id),
            name=str(rec.name),
            start_ts=float(rec.start_ts),
            end_ts=float(rec.end_ts),
            frame_count=int(rec.frame_count),
            face_sim_max=float(rec.face_sim_max),
            face_sim_sum=float(rec.face_sim_sum),
            face_sim_n=int(rec.face_sim_n),
            start_bbox=tuple(int(v) for v in rec.start_bbox),
            end_bbox=tuple(int(v) for v in rec.end_bbox),
        )
        self._logs.append(rec_copy)
        if self._normalized_writer is not None:
            try:
                self._normalized_writer.tracklet_closed(rec_copy)
            except Exception as e:
                print(f"[NORMALIZED] close enqueue failed: {e}")

    def _close_expired_locked(self, now_ts: float) -> None:
        if self.gap_seconds <= 0:
            return
        to_close: List[Tuple[int, int]] = []
        for key, rec in list(self._active.items()):
            if (float(now_ts) - float(rec.end_ts)) > float(self.gap_seconds):
                to_close.append((int(key[0]), int(key[1])))
        for key in to_close:
            self._finalize_tracklet_locked(key)

    def close_all(self) -> None:
        with self._lock:
            for key in list(self._active.keys()):
                self._finalize_tracklet_locked((int(key[0]), int(key[1])))

    def update(self, cam_id: int, present_names: List[str], ts: float, name_to_conf: Optional[Dict[str, float]] = None,
               frame_events: Optional[List[Any]] = None, camera_db_id: Optional[int] = None) -> None:
        if ts <= 0:
            ts = time.time()
        stream_id = int(cam_id)
        cam_db_id = int(camera_db_id) if camera_db_id is not None else -1
        with self._lock:
            if self._disabled:
                return
            self._close_expired_locked(now_ts=float(ts))
            for raw_ev in frame_events or []:
                ev = self._normalize_event(raw_ev)
                if ev is None:
                    continue
                tid, bbox, name, face_sim, member_id, _is_known = ev
                key = (int(stream_id), int(tid))
                current = self._active.get(key)
                name_norm = str(name or "")
                member_norm = int(member_id if name_norm else -1)
                if current is None:
                    self._start_tracklet_locked(
                        stream_id=int(stream_id), camera_db_id=int(cam_db_id), track_id=int(tid),
                        name=name_norm, member_id=int(member_norm), ts=float(ts), bbox=bbox, face_sim=float(face_sim),
                    )
                    continue
                same_identity = (str(current.name or "") == name_norm) and (int(current.member_id) == int(member_norm))
                if not same_identity:
                    self._finalize_tracklet_locked(key)
                    self._start_tracklet_locked(
                        stream_id=int(stream_id), camera_db_id=int(cam_db_id), track_id=int(tid),
                        name=name_norm, member_id=int(member_norm), ts=float(ts), bbox=bbox, face_sim=float(face_sim),
                    )
                    continue
                self._extend_tracklet_locked(current, ts=float(ts), bbox=bbox, face_sim=float(face_sim), camera_db_id=int(cam_db_id))

    def _snapshot_for_export(self, include_active: bool) -> List[_TrackletRecord]:
        with self._lock:
            rows = list(self._logs)
            active_rows = list(self._active.values()) if include_active else []
        for rec in active_rows:
            rows.append(self._make_record(
                tracklet_id=int(rec.tracklet_id),
                stream_id=int(rec.stream_id),
                camera_db_id=int(rec.camera_db_id),
                track_id=int(rec.track_id),
                member_id=int(rec.member_id),
                name=str(rec.name),
                start_ts=float(rec.start_ts),
                end_ts=float(rec.end_ts),
                frame_count=int(rec.frame_count),
                face_sim_max=float(rec.face_sim_max),
                face_sim_sum=float(rec.face_sim_sum),
                face_sim_n=int(rec.face_sim_n),
                start_bbox=tuple(int(v) for v in rec.start_bbox),
                end_bbox=tuple(int(v) for v in rec.end_bbox),
            ))
        rows.sort(key=lambda r: (float(r.start_ts), int(r.stream_id), int(r.tracklet_id)))
        return rows

    def _write_snapshot_csv(self, path: str, rows: List[_TrackletRecord]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp_path = path + ".tmp"
        fallback_path = self._fallback_path(path, "_live")
        header = [
            "tracklet_id", "stream_id", "camera_db_id", "track_id", "member_id", "name",
            "start_time", "end_time", "duration_seconds", "frames",
            "face_sim_max", "face_sim_avg", "start_bbox", "end_bbox",
        ]
        def _write_csv(pth: str) -> None:
            with open(pth, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                for rec in rows:
                    duration = max(0.0, float(rec.end_ts) - float(rec.start_ts))
                    avg_sim = (float(rec.face_sim_sum) / float(rec.face_sim_n)) if int(rec.face_sim_n) > 0 else 0.0
                    nm = str(rec.name or "").strip() or "Unknown"
                    w.writerow([
                        int(rec.tracklet_id),
                        int(rec.stream_id),
                        int(rec.camera_db_id) if int(rec.camera_db_id) > 0 else "",
                        int(rec.track_id),
                        int(rec.member_id) if int(rec.member_id) > 0 else "",
                        nm,
                        self._fmt_time(rec.start_ts),
                        self._fmt_time(rec.end_ts),
                        f"{float(duration):.3f}",
                        int(rec.frame_count),
                        f"{float(rec.face_sim_max):.4f}",
                        f"{float(avg_sim):.4f}",
                        self._bbox_to_str(rec.start_bbox),
                        self._bbox_to_str(rec.end_bbox),
                    ])
        with self._write_lock:
            try:
                _write_csv(tmp_path)
            except Exception as e:
                print("[CSV] write tmp failed:", e)
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                return
            try:
                os.replace(tmp_path, path)
                return
            except Exception as e:
                print("[CSV] replace failed (file may be locked). Writing fallback snapshot:", e)
                try:
                    os.replace(tmp_path, fallback_path)
                    return
                except Exception:
                    try:
                        _write_csv(fallback_path)
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
                        return
                    except Exception as e2:
                        print("[CSV] fallback write failed:", e2)
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass

    def write_csv_live(self, path: str) -> None:
        rows = self._snapshot_for_export(include_active=True)
        self._write_snapshot_csv(path, rows)

    def write_csv(self, path: str) -> None:
        self.close_all()
        rows = self._snapshot_for_export(include_active=False)
        self._write_snapshot_csv(path, rows)


class SummaryReport:
    report_level = "frame"
    is_segmented_report = False
    def __init__(self, num_cams: int, gap_seconds: float = 2.0, time_format: str = "%H:%M:%S"):
        self.num_cams = int(max(1, num_cams))
        self.gap_seconds = float(max(0.0, gap_seconds))
        self.time_format = str(time_format or "%H:%M:%S")
        self._lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._disabled = False
        self._first_seen: Dict[str, float] = {}
        self._active: Dict[Tuple[str, int], _ActiveSession] = {}
        self._logs: Dict[str, Dict[int, List[Tuple[float, float, float]]]] = defaultdict(lambda: defaultdict(list))
        self._total_seconds: Dict[str, float] = defaultdict(float)

    def stop(self) -> None:
        with self._lock:
            self._disabled = True

    def update(self, cam_id: int, present_names: List[str], ts: float, name_to_conf: Optional[Dict[str, float]] = None,
               frame_events: Optional[List[Any]] = None, camera_db_id: Optional[int] = None) -> None:
        if ts <= 0:
            ts = time.time()
        cam_id = int(cam_id)
        with self._lock:
            if self._disabled:
                return
            self._close_expired_locked(now_ts=float(ts))
            for nm in present_names or []:
                name = str(nm).strip()
                if not name:
                    continue
                conf = 0.0
                if isinstance(name_to_conf, dict):
                    try:
                        conf = float(name_to_conf.get(name, 0.0) or 0.0)
                    except Exception:
                        conf = 0.0
                if name not in self._first_seen:
                    self._first_seen[name] = float(ts)
                key = (name, cam_id)
                sess = self._active.get(key)
                if sess is None:
                    self._active[key] = _ActiveSession(float(ts), float(ts), float(conf), float(conf), 1 if conf > 0 else 0)
                else:
                    sess.last_seen_ts = float(ts)
                    if conf > 0:
                        sess.conf_max = max(float(sess.conf_max), float(conf))
                        sess.conf_sum += float(conf)
                        sess.conf_n += 1

    def _close_expired_locked(self, now_ts: float) -> None:
        if self.gap_seconds <= 0:
            return
        to_close: List[Tuple[str, int, _ActiveSession]] = []
        for (name, cam_id), sess in list(self._active.items()):
            if (float(now_ts) - float(sess.last_seen_ts)) > self.gap_seconds:
                to_close.append((name, cam_id, sess))
        for name, cam_id, sess in to_close:
            self._close_session_locked(name, cam_id, sess)

    def _close_session_locked(self, name: str, cam_id: int, sess: _ActiveSession) -> None:
        st = float(sess.start_ts)
        en = float(sess.last_seen_ts)
        if en < st:
            en = st
        conf_max = float(sess.conf_max or 0.0)
        self._logs[name][int(cam_id)].append((st, en, conf_max))
        self._total_seconds[name] += max(0.0, (en - st))
        self._active.pop((name, int(cam_id)), None)

    def close_all(self) -> None:
        with self._lock:
            for (name, cam_id), sess in list(self._active.items()):
                self._close_session_locked(name, cam_id, sess)

    def _fmt_time(self, ts: float) -> str:
        try:
            return datetime.fromtimestamp(float(ts)).strftime(self.time_format)
        except Exception:
            return ""

    def _fmt_total(self, total_seconds: float) -> str:
        try:
            sec_f = float(total_seconds)
        except Exception:
            sec_f = 0.0
        sec_i = int(round(max(0.0, sec_f)))
        if sec_i == 0 and sec_f > 0.0:
            sec_i = 1
        minutes = int(sec_i // 60)
        seconds = int(sec_i % 60)
        m_word = "minute" if minutes == 1 else "minutes"
        s_word = "second" if seconds == 1 else "seconds"
        return f"{minutes} {m_word} {seconds} {s_word}"

    def _snapshot_for_export(self, include_active: bool) -> Tuple[List[str], Dict[str, Dict[int, List[Tuple[float, float, float]]]], Dict[str, float]]:
        with self._lock:
            items = list(self._first_seen.items())
            logs = {k: {ck: list(vv) for ck, vv in cv.items()} for k, cv in self._logs.items()}
            totals = dict(self._total_seconds)
            active_items = list(self._active.items()) if include_active else []
        items.sort(key=lambda kv: kv[1])
        names_order = [nm for nm, _ in items]
        if include_active and active_items:
            for (name, cam_id), sess in active_items:
                st = float(sess.start_ts)
                en = float(sess.last_seen_ts)
                if en < st:
                    en = st
                conf_max = float(sess.conf_max or 0.0)
                logs.setdefault(name, {}).setdefault(int(cam_id), []).append((st, en, conf_max))
                totals[name] = float(totals.get(name, 0.0)) + max(0.0, (en - st))
                if name not in names_order:
                    names_order.append(name)
        return names_order, logs, totals

    @staticmethod
    def _fallback_path(path: str, suffix: str) -> str:
        try:
            p = Path(path)
            return str(p.with_name(p.stem + suffix + p.suffix))
        except Exception:
            return str(path) + suffix

    def _write_snapshot_csv(self, path: str, names_order: List[str], logs: Dict[str, Dict[int, List[Tuple[float, float, float]]]], totals: Dict[str, float]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        header = ["Member"] + [f"c{i+1}" for i in range(self.num_cams)] + ["Total time"]
        tmp_path = path + ".tmp"
        fallback_path = self._fallback_path(path, "_live")
        def _write_csv(pth: str) -> None:
            with open(pth, "w", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow(header)
                for name in names_order:
                    row = [name]
                    cam_map = logs.get(name, {})
                    for cam_id in range(self.num_cams):
                        entries = cam_map.get(cam_id, [])
                        lines = []
                        for idx, (st, en, conf) in enumerate(entries, start=1):
                            lines.append(f"L{idx} - {self._fmt_time(st)} to {self._fmt_time(en)} | conf {float(conf):.2f}")
                        row.append("\n".join(lines))
                    row.append(self._fmt_total(totals.get(name, 0.0)))
                    w.writerow(row)
        with self._write_lock:
            try:
                _write_csv(tmp_path)
            except Exception as e:
                print("[CSV] write tmp failed:", e)
                try:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                except Exception:
                    pass
                return
            try:
                os.replace(tmp_path, path)
                return
            except Exception as e:
                print("[CSV] replace failed (file may be locked). Writing fallback snapshot:", e)
                try:
                    os.replace(tmp_path, fallback_path)
                    return
                except Exception:
                    try:
                        _write_csv(fallback_path)
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass
                        return
                    except Exception as e2:
                        print("[CSV] fallback write failed:", e2)
                        try:
                            if os.path.exists(tmp_path):
                                os.remove(tmp_path)
                        except Exception:
                            pass

    def write_csv_live(self, path: str) -> None:
        names_order, logs, totals = self._snapshot_for_export(include_active=True)
        self._write_snapshot_csv(path, names_order, logs, totals)

    def write_csv(self, path: str) -> None:
        self.close_all()
        names_order, logs, totals = self._snapshot_for_export(include_active=False)
        self._write_snapshot_csv(path, names_order, logs, totals)


def _is_segmented_report(report: Any) -> bool:
    return bool(getattr(report, "is_segmented_report", False))


def live_csv_writer_loop(report, path: str, interval_s: float, stop_evt: threading.Event) -> None:
    base = float(interval_s)
    if base <= 0:
        return
    backoff = base
    while not stop_evt.is_set():
        try:
            if report is not None:
                if _is_segmented_report(report):
                    report.write_csv_live("")
                else:
                    report.write_csv_live(path)
            backoff = base
        except Exception as e:
            print("[WARN] Live CSV write failed:", e)
            backoff = min(60.0, max(base, backoff * 2.0))
        stop_evt.wait(backoff)


class SegmentedSummaryReport:
    report_level = "frame"
    is_segmented_report = True
    def __init__(self, num_cams: int, gap_seconds: float, time_format: str, segment_seconds: float,
                 segments_dir: str, segment_prefix: str = "summary", write_if_any_detection: bool = True,
                 log_prefix: str = "[CSV-SEG]"):
        self.num_cams = int(max(1, num_cams))
        self.gap_seconds = float(max(0.0, gap_seconds))
        self.time_format = str(time_format or "%H:%M:%S")
        self.segment_seconds = float(max(0.0, segment_seconds or 0.0))
        self.segments_dir = str(segments_dir or "csv_output")
        self.segment_prefix = str(segment_prefix or "summary")
        self.write_if_any_detection = bool(write_if_any_detection)
        self.log_prefix = str(log_prefix or "[CSV-SEG]")
        os.makedirs(self.segments_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._disabled = False
        self._seg_index = 0
        self._seg_start_mono = time.monotonic()
        self._seg_start_wall = time.time()
        self._seg_had_detection = False
        self._rep = SummaryReport(num_cams=self.num_cams, gap_seconds=self.gap_seconds, time_format=self.time_format)

    def _make_segment_path(self, seg_start_wall: float, seg_index: int) -> str:
        try:
            ts_tag = datetime.fromtimestamp(float(seg_start_wall)).strftime("%Y%m%d_%H%M%S")
        except Exception:
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.segments_dir, f"{self.segment_prefix}_{ts_tag}_p{int(seg_index):04d}.csv")

    def _finalize_current_locked(self) -> None:
        if self._rep is None:
            return
        if self.write_if_any_detection and (not bool(self._seg_had_detection)):
            return
        out_path = self._make_segment_path(self._seg_start_wall, self._seg_index)
        try:
            self._rep.write_csv(out_path)
            print(f"{self.log_prefix} segment written: {out_path}")
        except Exception as e:
            print(f"{self.log_prefix} segment write failed: {e}")

    def _rotate_if_needed_locked(self, now_mono: float) -> None:
        if self.segment_seconds <= 0:
            return
        if (float(now_mono) - float(self._seg_start_mono)) < float(self.segment_seconds):
            return
        self._finalize_current_locked()
        self._seg_index += 1
        self._seg_start_mono = float(now_mono)
        self._seg_start_wall = time.time()
        self._seg_had_detection = False
        self._rep = SummaryReport(num_cams=self.num_cams, gap_seconds=self.gap_seconds, time_format=self.time_format)

    def stop(self) -> None:
        with self._lock:
            self._disabled = True
            try:
                self._finalize_current_locked()
            except Exception:
                pass
            try:
                if self._rep is not None:
                    self._rep.stop()
            except Exception:
                pass

    def update(self, cam_id: int, present_names: List[str], ts: float, name_to_conf: Optional[Dict[str, float]] = None,
               frame_events: Optional[List[Any]] = None, camera_db_id: Optional[int] = None) -> None:
        with self._lock:
            if self._disabled:
                return
            self._rotate_if_needed_locked(time.monotonic())
            if present_names:
                self._seg_had_detection = True
            if self._rep is not None:
                self._rep.update(cam_id=cam_id, present_names=present_names, ts=ts, name_to_conf=name_to_conf, frame_events=frame_events, camera_db_id=camera_db_id)

    def write_csv_live(self, path: str) -> None:
        with self._lock:
            if self._disabled:
                return
            self._rotate_if_needed_locked(time.monotonic())
            if self.write_if_any_detection and (not bool(self._seg_had_detection)):
                return
            out_path = str(path or "").strip()
            if not out_path:
                out_path = self._make_segment_path(self._seg_start_wall, self._seg_index)
            if self._rep is not None:
                self._rep.write_csv_live(out_path)

    def write_csv(self, path: str) -> None:
        with self._lock:
            if self._disabled:
                return
            try:
                if self._rep is not None:
                    self._rep.write_csv(path)
            except Exception:
                pass
            try:
                self._finalize_current_locked()
            except Exception:
                pass


class SegmentedTrackletSummaryReport:
    report_level = "tracklet"
    is_segmented_report = True

    @staticmethod
    def _has_relevant_event(frame_events: Optional[List[Any]], include_unknown: bool) -> bool:
        if not frame_events:
            return False
        if include_unknown:
            return len(frame_events) > 0
        for ev in frame_events:
            try:
                if isinstance(ev, dict):
                    nm = str(ev.get("name", "") or "").strip()
                    is_known = bool(ev.get("is_known", bool(nm)))
                elif isinstance(ev, (list, tuple)) and len(ev) >= 8:
                    nm = str(ev[5] or "").strip()
                    is_known = bool(ev[8]) if len(ev) > 8 else bool(nm)
                else:
                    continue
            except Exception:
                continue
            if is_known or nm:
                return True
        return False

    def __init__(self, num_cams: int, gap_seconds: float, time_format: str, segment_seconds: float,
                 segments_dir: str, segment_prefix: str = "summary", write_if_any_detection: bool = True,
                 include_unknown: bool = False, log_prefix: str = "[CSV-SEG]",
                 normalized_writer: Optional[NormalizedDataDBWriter] = None):
        self.num_cams = int(max(1, num_cams))
        self.gap_seconds = float(max(0.0, gap_seconds))
        self.time_format = str(time_format or "%H:%M:%S")
        self.segment_seconds = float(max(0.0, segment_seconds or 0.0))
        self.segments_dir = str(segments_dir or "csv_output")
        self.segment_prefix = str(segment_prefix or "summary")
        self.write_if_any_detection = bool(write_if_any_detection)
        self.include_unknown = bool(include_unknown)
        self.log_prefix = str(log_prefix or "[CSV-SEG]")
        self._normalized_writer = normalized_writer
        os.makedirs(self.segments_dir, exist_ok=True)
        self._lock = threading.Lock()
        self._disabled = False
        self._seg_index = 0
        self._seg_start_mono = time.monotonic()
        self._seg_start_wall = time.time()
        self._seg_had_detection = False
        self._rep = TrackletSummaryReport(
            num_cams=self.num_cams, gap_seconds=self.gap_seconds, time_format=self.time_format,
            include_unknown=self.include_unknown, normalized_writer=self._normalized_writer,
        )

    def _make_segment_path(self, seg_start_wall: float, seg_index: int) -> str:
        try:
            ts_tag = datetime.fromtimestamp(float(seg_start_wall)).strftime("%Y%m%d_%H%M%S")
        except Exception:
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(self.segments_dir, f"{self.segment_prefix}_{ts_tag}_p{int(seg_index):04d}.csv")

    def _finalize_current_locked(self) -> None:
        if self._rep is None:
            return
        if self.write_if_any_detection and (not bool(self._seg_had_detection)):
            return
        out_path = self._make_segment_path(self._seg_start_wall, self._seg_index)
        try:
            self._rep.write_csv(out_path)
            print(f"{self.log_prefix} segment written: {out_path}")
        except Exception as e:
            print(f"{self.log_prefix} segment write failed: {e}")

    def _rotate_if_needed_locked(self, now_mono: float) -> None:
        if self.segment_seconds <= 0:
            return
        if (float(now_mono) - float(self._seg_start_mono)) < float(self.segment_seconds):
            return
        self._finalize_current_locked()
        self._seg_index += 1
        self._seg_start_mono = float(now_mono)
        self._seg_start_wall = time.time()
        self._seg_had_detection = False
        self._rep = TrackletSummaryReport(
            num_cams=self.num_cams, gap_seconds=self.gap_seconds, time_format=self.time_format,
            include_unknown=self.include_unknown, normalized_writer=self._normalized_writer,
        )

    def stop(self) -> None:
        with self._lock:
            self._disabled = True
            try:
                self._finalize_current_locked()
            except Exception:
                pass
            try:
                if self._rep is not None:
                    self._rep.stop()
            except Exception:
                pass

    def update(self, cam_id: int, present_names: List[str], ts: float, name_to_conf: Optional[Dict[str, float]] = None,
               frame_events: Optional[List[Any]] = None, camera_db_id: Optional[int] = None) -> None:
        with self._lock:
            if self._disabled:
                return
            self._rotate_if_needed_locked(time.monotonic())
            if present_names or self._has_relevant_event(frame_events, include_unknown=bool(self.include_unknown)):
                self._seg_had_detection = True
            if self._rep is not None:
                self._rep.update(
                    cam_id=cam_id, present_names=present_names, ts=ts, name_to_conf=name_to_conf,
                    frame_events=frame_events, camera_db_id=camera_db_id,
                )

    def write_csv_live(self, path: str) -> None:
        with self._lock:
            if self._disabled:
                return
            self._rotate_if_needed_locked(time.monotonic())
            if self.write_if_any_detection and (not bool(self._seg_had_detection)):
                return
            out_path = str(path or "").strip()
            if not out_path:
                out_path = self._make_segment_path(self._seg_start_wall, self._seg_index)
            if self._rep is not None:
                self._rep.write_csv_live(out_path)

    def write_csv(self, path: str) -> None:
        with self._lock:
            if self._disabled:
                return
            try:
                if self._rep is not None:
                    self._rep.write_csv(path)
            except Exception:
                pass
            try:
                self._finalize_current_locked()
            except Exception:
                pass


def _create_report(args: argparse.Namespace, num_cams: int, seg_s: float, seg_dir: str, seg_prefix: str,
                   normalized_writer: Optional[NormalizedDataDBWriter] = None):
    report_level = str(getattr(args, "report_level", "tracklet") or "tracklet").strip().lower()
    gap_seconds = float(getattr(args, "report_gap_seconds", 2.0) or 2.0)
    time_format = str(getattr(args, "report_time_format", "%H:%M:%S") or "%H:%M:%S")
    write_if_any_detection = (not bool(getattr(args, "csv_segment_write_empty", False)))
    include_unknown = bool(getattr(args, "tracklet_include_unknown", False))
    if seg_s > 0:
        if report_level == "tracklet":
            return SegmentedTrackletSummaryReport(
                num_cams=num_cams, gap_seconds=gap_seconds, time_format=time_format, segment_seconds=float(seg_s),
                segments_dir=seg_dir, segment_prefix=str(seg_prefix), write_if_any_detection=write_if_any_detection,
                include_unknown=include_unknown, normalized_writer=normalized_writer,
            )
        return SegmentedSummaryReport(
            num_cams=num_cams, gap_seconds=gap_seconds, time_format=time_format, segment_seconds=float(seg_s),
            segments_dir=seg_dir, segment_prefix=str(seg_prefix), write_if_any_detection=write_if_any_detection,
        )
    if report_level == "tracklet":
        return TrackletSummaryReport(
            num_cams=num_cams, gap_seconds=gap_seconds, time_format=time_format, include_unknown=include_unknown,
            normalized_writer=normalized_writer,
        )
    return SummaryReport(num_cams=num_cams, gap_seconds=gap_seconds, time_format=time_format)


def _report_label(report: Any) -> str:
    return "tracklet" if str(getattr(report, "report_level", "frame")) == "tracklet" else "summary"


def _update_report_from_meta(report, args: argparse.Namespace, sid: int, camera_db_id: int,
                             meta: Dict[str, Any], ts_use: float) -> None:
    if report is None:
        return
    events = meta.get("events", []) or []
    if bool(getattr(args, "report_use_drawn_only", True)):
        conf_map: Dict[str, float] = {}
        for ev in events:
            try:
                nm = str(ev[5] or "")
                sim = float(ev[6] or 0.0)
            except Exception:
                continue
            if not nm:
                continue
            prev = float(conf_map.get(nm, 0.0))
            if sim > prev:
                conf_map[nm] = sim
        names = sorted(conf_map.keys())
    else:
        names = meta.get("present_names", []) or []
        conf_map = meta.get("present_conf", {}) or {}
        if not isinstance(conf_map, dict):
            conf_map = {}
    report.update(
        cam_id=int(sid),
        present_names=list(names),
        ts=float(ts_use),
        name_to_conf=dict(conf_map),
        frame_events=list(events),
        camera_db_id=int(camera_db_id),
    )


def _create_normalized_data_writer(args: argparse.Namespace) -> Optional[NormalizedDataDBWriter]:
    if not bool(getattr(args, "write_normalized_data", True)):
        return None
    db_url = str(getattr(args, "db_url", "") or "").strip()
    if not db_url:
        return None
    try:
        writer = NormalizedDataDBWriter(db_url=db_url)
        print("[INIT] normalized_data writer: ON (known members only; guest rows ignored)")
        return writer
    except Exception as e:
        print("[WARN] normalized_data writer init failed:", e)
        return None


def _create_tracking_reports(
    args: argparse.Namespace,
    num_cams: int,
    normalized_writer: Optional[NormalizedDataDBWriter] = None,
) -> Tuple[Any, Any, Optional[threading.Event], Optional[threading.Thread]]:
    report = None
    normalized_report = None
    csv_stop_evt: Optional[threading.Event] = None
    csv_thread: Optional[threading.Thread] = None
    writer_attached_to_csv_report = False

    if bool(getattr(args, "save_csv", False)):
        out_dir = "csv_output"
        os.makedirs(out_dir, exist_ok=True)
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_arg = str(getattr(args, "csv", "") or "").strip()
        if (not csv_arg) or (csv_arg == "detections_summary.csv"):
            args.csv = os.path.join(out_dir, f"output_csv_new{ts_tag}.csv")
        else:
            args.csv = csv_arg
            try:
                os.makedirs(os.path.dirname(args.csv) or ".", exist_ok=True)
            except Exception:
                pass
        seg_s = float(getattr(args, "csv_segment_seconds", 0.0) or 0.0)
        if seg_s <= 0:
            seg_s = float(getattr(args, "video_segment_seconds", 0.0) or 0.0)
        seg_dir = str(getattr(args, "csv_segments_dir", "") or "").strip()
        if not seg_dir:
            try:
                seg_dir = os.path.dirname(str(args.csv)) or out_dir
            except Exception:
                seg_dir = out_dir
        os.makedirs(seg_dir, exist_ok=True)
        seg_prefix = str(getattr(args, "csv_segment_prefix", "summary") or "summary").strip() or "summary"
        if seg_prefix == "summary":
            try:
                seg_prefix = Path(str(args.csv)).stem or seg_prefix
            except Exception:
                pass
        report = _create_report(
            args=args,
            num_cams=int(num_cams),
            seg_s=float(seg_s),
            seg_dir=str(seg_dir),
            seg_prefix=str(seg_prefix),
            normalized_writer=normalized_writer if str(getattr(args, "report_level", "tracklet") or "tracklet").strip().lower() == "tracklet" else None,
        )
        writer_attached_to_csv_report = bool(
            normalized_writer is not None and str(getattr(report, "report_level", "frame")) == "tracklet"
        )
        if seg_s > 0:
            print(
                f"[CSV] Segmented {_report_label(report)} CSVs: every {seg_s:.0f}s into folder: {seg_dir} "
                f"(prefix={seg_prefix}, skip_empty={not bool(getattr(args,'csv_segment_write_empty', False))})"
            )
        else:
            print(f"[CSV] Live {_report_label(report)} CSV will be written to: {args.csv}")
        interval = float(getattr(args, "csv_live_interval", 1.0) or 0.0)
        if interval > 0:
            csv_stop_evt = threading.Event()
            csv_thread = threading.Thread(
                target=live_csv_writer_loop,
                args=(report, args.csv, interval, csv_stop_evt),
                daemon=True,
            )
            csv_thread.start()

    if (normalized_writer is not None) and (not writer_attached_to_csv_report):
        normalized_report = TrackletSummaryReport(
            num_cams=int(num_cams),
            gap_seconds=float(getattr(args, "report_gap_seconds", 2.0) or 2.0),
            time_format=str(getattr(args, "report_time_format", "%H:%M:%S") or "%H:%M:%S"),
            include_unknown=False,
            normalized_writer=normalized_writer,
        )
    return report, normalized_report, csv_stop_evt, csv_thread


def _finalize_tracking_reports(report, normalized_report, args: argparse.Namespace) -> None:
    if report is not None:
        try:
            report.stop()
        except Exception:
            pass
        try:
            if bool(getattr(args, "save_csv", False)):
                if _is_segmented_report(report):
                    try:
                        seg_dir = str(getattr(report, "segments_dir", "") or "")
                        print(f"[CSV] Segmented {_report_label(report)} CSV mode: per-segment CSVs are in: {seg_dir}")
                    except Exception:
                        print(f"[CSV] Segmented {_report_label(report)} CSV mode: per-segment CSVs were finalized.")
                else:
                    report.write_csv(args.csv)
                    print(f"[CSV] Final {_report_label(report)} CSV written to {args.csv}")
        except Exception as e:
            print("[CSV] Final write failed:", e)
    if normalized_report is not None:
        try:
            normalized_report.stop()
        except Exception:
            pass
        try:
            normalized_report.close_all()
        except Exception as e:
            print("[WARN] normalized_data report finalization failed:", e)


@dataclass
class DrawItem:
    tid: int
    bbox: Tuple[int, int, int, int]
    name: str
    member_id: int
    face_sim: float
    stable_score: float
    det_conf: Optional[float]
    face_hit: bool
    low_face: bool = False
    low_face_sim: float = 0.0


def _box_area_xyxy(b: Tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = b
    return int(max(0, x2 - x1) * max(0, y2 - y1))


def _priority_tuple(it: DrawItem) -> tuple:
    return (
        1 if it.face_hit else 0,
        float(it.face_sim),
        float(it.stable_score),
        float(it.det_conf or 0.0),
        float(_box_area_xyxy(it.bbox)),
    )


def deduplicate_draw_items(items: List[DrawItem], iou_thresh: float) -> List[DrawItem]:
    if not items:
        return []
    if float(iou_thresh) <= 0.0:
        return list(items)
    known = [it for it in items if it.name]
    unknown = [it for it in items if not it.name]
    if not known:
        return unknown
    groups: Dict[str, List[DrawItem]] = defaultdict(list)
    for it in known:
        groups[it.name].append(it)
    kept: List[DrawItem] = []
    for _name, group in groups.items():
        group_sorted = sorted(group, key=_priority_tuple, reverse=True)
        selected: List[DrawItem] = []
        for it in group_sorted:
            ok = True
            for s in selected:
                if iou_xyxy(it.bbox, s.bbox) >= float(iou_thresh):
                    ok = False
                    break
            if ok:
                selected.append(it)
        kept.extend(selected)
    kept.extend(unknown)
    return kept


def block_duplicate_names(items: List[DrawItem]) -> Tuple[List[DrawItem], set[int]]:
    if not items:
        return [], set()
    groups: Dict[str, List[DrawItem]] = defaultdict(list)
    winner_tid_by_name: Dict[str, int] = {}
    for it in items:
        if not it.name:
            continue
        groups[str(it.name)].append(it)
    for name, group in groups.items():
        best = max(group, key=_priority_tuple)
        winner_tid_by_name[str(name)] = int(best.tid)
    out: List[DrawItem] = []
    demoted_tids: set[int] = set()
    for it in items:
        if not it.name:
            out.append(it)
            continue
        if int(winner_tid_by_name.get(str(it.name), int(it.tid))) == int(it.tid):
            out.append(it)
            continue
        demoted_tids.add(int(it.tid))
        out.append(DrawItem(
            tid=int(it.tid),
            bbox=it.bbox,
            name="",
            member_id=-1,
            face_sim=float(it.face_sim),
            stable_score=float(it.stable_score),
            det_conf=it.det_conf,
            face_hit=bool(it.face_hit),
            low_face=bool(it.low_face),
            low_face_sim=float(it.low_face_sim),
        ))
    return out, demoted_tids


class GlobalNameOwner:
    def __init__(self, hold_seconds: float = 0.5, switch_margin: float = 0.02):
        self.hold_seconds = float(max(0.0, hold_seconds))
        self.switch_margin = float(max(0.0, switch_margin))
        self._lock = threading.Lock()
        self._state: Dict[str, Dict[str, Any]] = {}

    def _cleanup(self, now: float) -> None:
        if self.hold_seconds <= 0:
            return
        dead = []
        for name, st in self._state.items():
            ts = float(st.get("ts", 0.0))
            if (now - ts) > self.hold_seconds:
                dead.append(name)
        for name in dead:
            self._state.pop(name, None)

    def allow(self, name: str, sid: int, score: float) -> bool:
        if not name:
            return False
        now = time.time()
        with self._lock:
            self._cleanup(now)
            st = self._state.get(name)
            if st is None:
                self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
                return True
            owner_sid = int(st.get("sid", -1))
            owner_score = float(st.get("score", 0.0))
            owner_ts = float(st.get("ts", 0.0))
            if owner_sid == int(sid):
                st["score"] = max(owner_score, float(score))
                st["ts"] = now
                return True
            if self.hold_seconds > 0 and (now - owner_ts) > self.hold_seconds:
                self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
                return True
            if float(score) > (owner_score + self.switch_margin):
                self._state[name] = {"sid": int(sid), "score": float(score), "ts": now}
                return True
            return False


def parse_args(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(
        "YOLO -> tracker with member_embeddings DB gallery + rolling embedding updates.",
        conflict_handler="resolve",
    )
    ap.add_argument("--src", nargs="+", required=True, help="Video sources (RTSP/RTMP/HTTP/file).")
    ap.add_argument("--camera-ids", nargs="+", type=int, default=[], help="DB camera_ids aligned with --src order. If omitted, defaults to 1..N.")

    ap.add_argument("--use-db", action="store_true", help="Enable DB gallery.")
    ap.add_argument("--db-url", default="", help="SQLAlchemy DB URL (postgresql://...).")
    ap.add_argument("--db-refresh-seconds", type=float, default=30.0, help="Reload DB gallery every N seconds (0=off).")
    ap.add_argument("--db-max-bank", type=int, default=0, help="Max embeddings per entry to load from *_embeddings_raw (0=all).")
    ap.add_argument("--db-include-inactive", action="store_true", help="Include inactive members.")

    ap.add_argument("--update-db-embeddings", action="store_true", help="Update member_embeddings face banks when face match is strong.")
    ap.add_argument("--update-face-sim-thresh", type=float, default=0.75, help="Minimum face similarity to sample embeddings for DB update.")
    ap.add_argument("--embeddings-slot-mb", type=float, default=0.5, help="Slot size in MB. Total bank = 2 * slot_mb.")
    ap.add_argument("--embeddings-reset-if-gap-days", type=int, default=2, help="If last update is >= this many days ago, clear both slots.")
    ap.add_argument("--embeddings-min-sample-seconds", type=float, default=0.5, help="Min seconds between sampling embeddings per (member,camera).")
    ap.add_argument("--embeddings-flush-seconds", type=float, default=10.0, help="Flush buffered embeddings to DB at least every N seconds.")
    ap.add_argument("--embeddings-log-csv", default="", help="Append-only CSV log for embedding updates (auto if empty).")
    ap.add_argument("--embeddings-samples-log-csv", default="", help="CSV log for each accepted sample (auto if empty).")
    ap.add_argument("--no-update-face-bank", action="store_true", help="Do not update face_embeddings_raw/face_embedding.")
    ap.add_argument("--no-update-body-bank", action="store_true", help="No-op in this face-only build; body bank updates are disabled.")
    ap.add_argument("--embed-extract-face-sim-thresh", type=float, default=0.75, help="Only extract/enqueue embeddings when face_sim >= this.")
    ap.add_argument("--embed-min-face-det-score", type=float, default=0.50, help="Only consider faces when InsightFace det_score >= this.")

    ap.add_argument("--yolo-weights", default="yolov8n.pt")
    ap.add_argument("--yolo-imgsz", type=int, default=1280, help="YOLO inference size (0=ultralytics default)")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--iou", type=float, default=0.20)
    ap.add_argument("--half", action="store_true", help="Enable FP16 where supported")

    ap.add_argument("--cudnn-benchmark", action="store_true", help="Enable cuDNN benchmark")
    ap.add_argument("--reader", choices=["adaptive"], default="adaptive")
    ap.add_argument("--queue-size", type=int, default=128, help="Queue size for ordered readers")
    ap.add_argument("--rtsp-transport", choices=["tcp", "udp"], default="tcp")

    ap.add_argument("--stream-freeze-seconds", type=float, default=2.0)
    ap.add_argument("--stream-open-timeout-ms", type=int, default=1000)
    ap.add_argument("--stream-read-timeout-ms", type=int, default=1000)
    ap.add_argument("--stream-reconnect-base-seconds", type=float, default=0.5)
    ap.add_argument("--stream-reconnect-max-seconds", type=float, default=3.0)
    ap.add_argument("--stream-reconnect-jitter", type=float, default=0.10)
    ap.add_argument("--stream-reconnect-log-interval", type=float, default=2.0)
    ap.add_argument("--resize", type=int, nargs=2, default=[0, 0], help="Force resize W H after reading (0 0 = keep)")
    ap.add_argument("--max-queue-age-ms", type=int, default=1000, help="Drop frames older than this (0=off).")
    ap.add_argument("--max-drain-per-cycle", type=int, default=32, help="Max stale frames to drop per processing cycle.")

    g = ap.add_mutually_exclusive_group()
    g.add_argument("--no-deepsort", action="store_true", help="Disable DeepSORT (fallback to IoU tracker)")
    g.add_argument("--use-deepsort", action="store_true", help="Force DeepSORT")
    g.add_argument("--use-strongsort", action="store_true", help="Force StrongSORT (BoxMOT)")
    g.add_argument("--use-bytetrack", action="store_true", help="Force ByteTrack (BoxMOT)")

    ap.add_argument("--strongsort-reid-weights", default=r"osnet_x1_0_msmt17.pt", help="ReID weights path for StrongSORT (BoxMOT).")
    ap.add_argument("--bytetrack-min-conf", type=float, default=0.10, help="ByteTrack: discard detections below this conf.")
    ap.add_argument("--bytetrack-track-thresh", type=float, default=0.45, help="ByteTrack: high-confidence threshold for first association.")
    ap.add_argument("--bytetrack-match-thresh", type=float, default=0.80, help="ByteTrack: matching threshold.")
    ap.add_argument("--bytetrack-track-buffer", type=int, default=25, help="ByteTrack: track buffer.")
    ap.add_argument("--bytetrack-frame-rate", type=int, default=30, help="ByteTrack: video FPS used internally.")

    ap.add_argument("--reid-model", default="osnet_x1_0")
    ap.add_argument("--reid-weights", default="", help="Optional TorchReID weights path")
    ap.add_argument("--reid-batch-size", type=int, default=16, help="TorchReID batch size")

    ap.add_argument("--max-age", type=int, default=60)
    ap.add_argument("--n-init", type=int, default=3)
    ap.add_argument("--nn-budget", type=int, default=200)
    ap.add_argument("--tracker-max-cosine", type=float, default=0.4)
    ap.add_argument("--tracker-nms-overlap", type=float, default=1.0)
    ap.add_argument("--max-iou-distance", type=float, default=0.30, help="Tracker association IoU threshold.")

    ap.add_argument("--gallery-thresh", type=float, default=0.99)
    ap.add_argument("--gallery-gap", type=float, default=0.22)
    ap.add_argument("--reid-topk", type=int, default=3)
    ap.add_argument("--min-box-wh", type=int, default=40)

    ap.add_argument("--use-face", action="store_true")
    ap.add_argument("--face-model", default="buffalo_l")
    ap.add_argument("--face-det-size", type=int, nargs=2, default=[1280, 1280])
    ap.add_argument("--face-thresh", type=float, default=0.50, help="Minimum face similarity")
    ap.add_argument("--face-gap", type=float, default=0.05, help="Top1-top2 gap required")
    ap.add_argument("--face-every-n", type=int, default=1, help="Run face detector every N frames")
    ap.add_argument("--face-hold-frames", type=int, default=30, help="How long to keep 'face visible' after last linked face match")
    ap.add_argument("--face-provider", choices=["auto", "cuda", "cpu"], default="auto")
    ap.add_argument("--ort-log", action="store_true")
    ap.add_argument("--face-iou-link", type=float, default=0.35, help="Face->person link threshold")
    ap.add_argument("--face-link-mode", choices=["ioa", "iou"], default="ioa")
    ap.add_argument(
        "--face-center-in-person",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require face center inside person box (default: true).",
    )
    ap.add_argument("--min-face-px", type=int, default=24, help="Min face bbox width/height (px) to consider")
    ap.add_argument("--min-face-area-ratio", type=float, default=0.006, help="Min face_area/person_area to accept face link (0=off)")
    ap.add_argument("--face-center-y-max-ratio", type=float, default=0.70, help="Face center must be in top portion of person box (0..1)")
    ap.add_argument("--face-strong-thresh", type=float, default=0.50, help="If face_sim >= this, ignore body conflicts")
    ap.add_argument("--face-override-thresh", type=float, default=0.0, help="Immediate wrong-ID correction threshold. 0=auto based on face thresholds.")
    ap.add_argument("--face-override-gap", type=float, default=0.0, help="Minimum top1-top2 face gap for immediate wrong-ID correction. 0=auto.")
    ap.add_argument("--face-override-det-score", type=float, default=0.0, help="Minimum face detector score for immediate wrong-ID correction. 0=auto.")
    ap.add_argument("--face-confirm-hits", type=int, default=2, help="Face hits required to confirm a new track label when the face is not very strong.")
    ap.add_argument("--face-switch-confirm-hits", type=int, default=3, help="Face hits required before switching an already-assigned track to a different name, unless clear-face override fires.")
    ap.add_argument("--camera-name-hold-frames", type=int, default=0, help="How long a name stays owned by its current track in the same camera. 0=auto.")
    ap.add_argument("--camera-name-switch-margin", type=float, default=0.05, help="New owner must beat the old owner by at least this face score margin.")
    ap.add_argument("--camera-name-switch-hits", type=int, default=2, help="Consecutive approved face hits needed before a different track can steal the same name.")

    ap.add_argument("--name-decay", type=float, default=0.95)
    ap.add_argument("--name-min-score", type=float, default=0.60)
    ap.add_argument("--name-margin", type=float, default=0.10)
    ap.add_argument("--name-ttl", type=int, default=120)
    ap.add_argument("--name-face-weight", type=float, default=1.2)
    ap.add_argument("--name-body-weight", type=float, default=0.5)
    ap.add_argument("--identity-lock-frames", type=int, default=150, help="Lock a confirmed face identity for N frames to prevent swaps.")
    ap.add_argument("--identity-lock-face-thresh", type=float, default=0.50, help="Start/restart identity lock when approved face similarity >= this value.")
    ap.add_argument("--name-continuity-iou", type=float, default=0.05, help="Min IoU to keep reusing a cached name without a new face.")
    ap.add_argument("--name-continuity-center-shift", type=float, default=0.60, help="Max normalized center shift to keep reusing a cached name without a new face.")
    ap.add_argument("--name-break-clear-frames", type=int, default=3, help="Consecutive continuity-break frames before clearing cached identity on a tracker.")

    ap.add_argument("--draw-only-matched", action="store_true", help="Draw tracks only when matched with a detection this frame")
    ap.add_argument("--min-det-conf", type=float, default=0.45, help="Hide boxes below this detection confidence when drawing")
    ap.add_argument("--iou-max-miss", type=int, default=30, help="Max missed frames for IOU fallback before dropping the track")

    ap.add_argument("--allow-duplicate-names", action="store_true", help="Allow the same name multiple times in same camera frame")
    ap.add_argument("--dedup-iou", type=float, default=0.0, help="Duplicate-name suppression IoU threshold")
    ap.add_argument("--no-global-unique-names", action="store_true", help="Disable cross-camera name ownership gate")
    ap.add_argument("--global-hold-seconds", type=float, default=0.5)
    ap.add_argument("--global-switch-margin", type=float, default=0.02)
    ap.add_argument("--show-global-id", action="store_true", help="Append DB member_id to labels")

    ap.add_argument(
    "--hide-unknown",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Hide unknown persons (default: False). Use --hide-unknown to enable.",
)
    ap.add_argument("--no-persist-names", action="store_true", help="Do not persist recognized name when face is not visible.")
    ap.add_argument("--no-show-track-id", action="store_true", help="Do not append tracker ID (T#) in the label.")
    ap.add_argument(
        "--report-use-drawn-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If True (default), summary CSV is driven only by boxes actually drawn.",
    )
    ap.add_argument(
        "--report-level",
        choices=["frame", "tracklet"],
        default="tracklet",
        help="CSV/report granularity. 'tracklet' writes one row per contiguous tracker segment; 'frame' keeps the legacy member-session summary.",
    )
    ap.add_argument("--tracklet-include-unknown", action="store_true", help="Include unknown tracklets in tracklet CSV output.")
    ap.add_argument(
        "--write-normalized-data",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write known tracklet timings into normalized_data (default: true). Use --no-write-normalized-data to disable.",
    )

    ap.add_argument("--show", action="store_true")
    ap.add_argument("--save-csv", action="store_true", help="Save CSV report (tracklet-level by default).")
    ap.add_argument("--csv", default="detections_summary.csv")
    ap.add_argument("--report-gap-seconds", type=float, default=2.0)
    ap.add_argument("--report-time-format", default="%H:%M:%S")
    ap.add_argument("--csv-live-interval", type=float, default=1.0)
    ap.add_argument("--csv-segment-seconds", type=float, default=0.0, help="Write one CSV per segment (seconds). 0=off. If 0, reuses --video-segment-seconds if set.")
    ap.add_argument("--csv-segments-dir", default="", help="Folder for per-segment CSV files.")
    ap.add_argument("--csv-segment-prefix", default="summary", help="Filename prefix for per-segment CSVs.")
    ap.add_argument("--csv-segment-write-empty", action="store_true", help="Write segment CSV even if no detections occurred in that segment.")
    ap.add_argument("--overlay-fps", action="store_true", help="Draw FPS/lag/queue stats on each stream.")

    ap.add_argument("--grid-rows", type=int, default=0)
    ap.add_argument("--grid-cols", type=int, default=0)
    ap.add_argument("--grid-mode", choices=["cover", "contain"], default="cover")
    ap.add_argument("--fullscreen", action="store_true")

    ap.add_argument("--no-save-video", action="store_true")
    ap.add_argument("--video-dir", default="saved_videos")
    ap.add_argument("--video-prefix", default="saved_video")
    ap.add_argument("--video-fps", type=float, default=20.0)
    ap.add_argument("--video-fourcc", default="mp4v")
    ap.add_argument("--video-ext", default=".mp4")
    ap.add_argument("--video-save-height", type=int, default=480)
    ap.add_argument("--video-segment-seconds", type=float, default=3600.0)

    return ap.parse_args(argv)


def process_one_frame(
    frame_idx: int,
    frame_bgr: np.ndarray,
    sid: int,
    camera_db_id: int,
    yolo,
    args,
    deep_tracker,
    iou_tracker: IOUTracker,
    people: list[PersonEntry],
    reid_extractor,
    face_app,
    face_gallery: FaceGallery,
    name_to_member_id: dict[str, int],
    identity_state: dict,
    device_is_cuda: bool,
    global_owner: Optional[GlobalNameOwner] = None,
    ts_cap: float = 0.0,
    embed_updater: Optional[EmbeddingDBUpdater] = None,
    camera_name_owner: Optional[CameraNameOwner] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rw, rh = int(args.resize[0]), int(args.resize[1])
    if rw > 0 and rh > 0:
        frame_bgr = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
    H, W = frame_bgr.shape[:2]
    hide_unknown = bool(getattr(args, "hide_unknown", False))

    tlwh_conf: list[list[float]] = []
    if yolo is not None:
        try:
            res = _yolo_forward_safe(yolo, frame_bgr, args)
            boxes = res[0].boxes if (res and len(res)) else None
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
                cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
                keep = cls == 0
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
                    tlwh_conf.append([x1f, y1f, ww, hh, float(c)])
        except Exception as e:
            print(f"[SRC {sid}] YOLO error:", e)

    dets_np = np.asarray(tlwh_conf, dtype=np.float32)
    if dets_np.ndim != 2:
        dets_np = dets_np.reshape((0, 5)).astype(np.float32)
    dets_dsrt = [([float(x), float(y), float(w), float(h)], float(cf), 0) for x, y, w, h, cf in tlwh_conf]

    out_tracks: List[Any] = []
    if deep_tracker is not None:
        if hasattr(deep_tracker, 'update_tracks'):
            try:
                out_tracks = deep_tracker.update_tracks(dets_dsrt, frame=frame_bgr)
            except Exception as e:
                print(f"[SRC {sid}] DeepSORT update_tracks error:", e)
                out_tracks = iou_tracker.update(dets_np)
        elif hasattr(deep_tracker, 'update'):
            try:
                if len(tlwh_conf) > 0:
                    dets_boxmot = np.asarray([[x, y, x + w, y + h, cf, 0] for x, y, w, h, cf in tlwh_conf], dtype=np.float32)
                else:
                    dets_boxmot = np.zeros((0, 6), dtype=np.float32)
                res_mot = deep_tracker.update(dets_boxmot, frame_bgr)
                out_tracks = boxmot_results_to_tracks(res_mot)
            except Exception as e:
                print(f"[SRC {sid}] BoxMOT tracker update error:", e)
                out_tracks = iou_tracker.update(dets_np)
        else:
            out_tracks = iou_tracker.update(dets_np)
    else:
        out_tracks = iou_tracker.update(dets_np)

    recognized_faces: List[Dict[str, Any]] = []
    low_faces: List[Dict[str, Any]] = []
    do_face = face_app is not None and face_gallery is not None and (not face_gallery.is_empty()) and (frame_idx % max(1, int(args.face_every_n)) == 0)
    if do_face:
        try:
            det_min = float(getattr(args, "embed_min_face_det_score", 0.75))
            with _face_lock:
                faces = face_app.get(np.ascontiguousarray(frame_bgr))
            for f in safe_iter_faces(faces):
                bbox = getattr(f, "bbox", None)
                if bbox is None:
                    continue
                b = np.asarray(bbox).reshape(-1)
                if b.size < 4:
                    continue
                fx1, fy1, fx2, fy2 = map(float, b[:4])
                fw = max(0.0, fx2 - fx1)
                fh = max(0.0, fy2 - fy1)
                if fw < float(args.min_face_px) or fh < float(args.min_face_px):
                    continue
                det_score = float(extract_face_det_score(f))
                if det_score < det_min:
                    continue
                emb = extract_face_embedding(f)
                if emb is None:
                    continue
                emb = l2_normalize(np.asarray(emb, dtype=np.float32))
                best_mid, flabel, fsim, fsecond = best_face_top2(emb, face_gallery)
                if flabel is None or best_mid is None:
                    continue
                gap = float(fsim - fsecond)
                if (fsim >= float(args.face_thresh)) and (gap >= float(args.face_gap)):
                    recognized_faces.append({
                        "bbox": (fx1, fy1, fx2, fy2),
                        "label": str(flabel),
                        "member_id": int(best_mid),
                        "sim": float(fsim),
                        "second": float(fsecond),
                        "gap": float(gap),
                        "det_score": float(det_score),
                        "emb": emb.astype(np.float32),
                    })
                else:
                    if float(fsim) < float(args.face_thresh):
                        low_faces.append({
                            "bbox": (fx1, fy1, fx2, fy2),
                            "sim": float(fsim),
                            "det_score": float(det_score),
                        })
        except Exception as e:
            print(f"[SRC {sid}] FaceAnalysis error:", e)

    out = frame_bgr.copy()
    tracks_info: List[Dict[str, Any]] = []

    for tid in list(identity_state.keys()):
        st = identity_state.get(tid, {})
        if isinstance(st, dict) and "last_seen_frame" not in st:
            st["last_seen_frame"] = -1

    raw_tracks: List[Dict[str, Any]] = []
    for t in out_tracks:
        time_since_update = getattr(t, "time_since_update", 0)
        had_match_this_frame = (time_since_update == 0) or (getattr(t, "last_detection", None) is not None)
        if args.draw_only_matched and not had_match_this_frame:
            continue
        try:
            if hasattr(t, "is_confirmed") and callable(getattr(t, "is_confirmed")) and (not t.is_confirmed()):
                continue
            if hasattr(t, "to_tlbr"):
                ltrb = t.to_tlbr()
            elif hasattr(t, "to_ltrb"):
                ltrb = t.to_ltrb()
            else:
                ltrb = t.to_tlbr()
            x1, y1, x2, y2 = map(int, ltrb)
            x1 = int(max(0, min(W - 1, x1)))
            y1 = int(max(0, min(H - 1, y1)))
            x2 = int(max(0, min(W, x2)))
            y2 = int(max(0, min(H, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            tid = int(getattr(t, "track_id", getattr(t, "track_id_", -1)))
            if tid < 0:
                continue
        except Exception:
            continue
        det_conf = None
        try:
            if hasattr(t, "det_conf") and t.det_conf is not None:
                det_conf = float(t.det_conf)
            elif hasattr(t, "last_detection") and t.last_detection is not None:
                ld = t.last_detection
                if isinstance(ld, (list, tuple)) and len(ld) >= 2:
                    det_conf = float(ld[1])
                elif isinstance(ld, dict):
                    det_conf = float(ld.get("confidence", ld.get("det_conf", 0.0)))
        except Exception:
            det_conf = None
        if args.min_det_conf > 0 and det_conf is not None and det_conf < args.min_det_conf:
            if args.draw_only_matched:
                continue
        raw_tracks.append({"tid": tid, "bbox": (x1, y1, x2, y2), "det_conf": det_conf})

    face_for_tid: Dict[int, Dict[str, Any]] = assign_faces_to_tracks_one_to_one(recognized_faces=recognized_faces, raw_tracks=raw_tracks, args=args)
    low_face_for_tid: Dict[int, Dict[str, Any]] = {}
    if low_faces and raw_tracks:
        raw_unassigned = [tr for tr in raw_tracks if int(tr.get("tid", -1)) not in face_for_tid]
        if raw_unassigned:
            low_face_for_tid = assign_faces_to_tracks_one_to_one(recognized_faces=low_faces, raw_tracks=raw_unassigned, args=args)

    for tr in raw_tracks:
        tid = int(tr["tid"])
        x1, y1, x2, y2 = tr["bbox"]
        det_conf = tr.get("det_conf", None)
        face_label, face_sim, face_gap = "", 0.0, 0.0
        face_det_score = 1.0
        face_member_id = -1
        face_emb = None
        face_hit = False
        low_face_hit = False
        low_face_sim = 0.0
        low_face_det_score = 1.0
        fm = face_for_tid.get(tid)
        if fm is not None:
            face_hit = True
            face_label = str(fm.get("label", ""))
            face_sim = float(fm.get("sim", 0.0))
            face_gap = float(fm.get("gap", 0.0))
            face_member_id = int(fm.get("member_id", -1))
            face_emb = fm.get("emb", None)
            try:
                face_det_score = float(fm.get("det_score", 1.0))
            except Exception:
                face_det_score = 1.0
        lfm = low_face_for_tid.get(tid) if isinstance(low_face_for_tid, dict) else None
        if lfm is not None:
            low_face_hit = True
            try:
                low_face_sim = float(lfm.get("sim", 0.0))
            except Exception:
                low_face_sim = 0.0
            try:
                low_face_det_score = float(lfm.get("det_score", 1.0))
            except Exception:
                low_face_det_score = 1.0

        entry = identity_state.setdefault(tid, make_identity_entry())
        entry["last_seen_frame"] = int(frame_idx)

        approved_face_label = ""
        approved_face_sim = 0.0
        approved_face_member_id = -1
        approved_face_emb = None
        approved_face_det_score = float(face_det_score)
        approved_face_gap = float(face_gap)
        if face_hit and face_label:
            cand_label, cand_mid, cand_sim, cand_ok, force_owner_switch = approve_face_candidate_for_track(
                entry, face_label, int(face_member_id), float(face_sim), float(face_gap), float(face_det_score), args
            )
            if cand_ok and cand_label:
                owner_ok = True
                if camera_name_owner is not None:
                    owner_ok = camera_name_owner.allow(
                        str(cand_label), tid=int(tid), score=float(cand_sim), frame_idx=int(frame_idx), force=bool(force_owner_switch)
                    )
                if owner_ok:
                    approved_face_label = str(cand_label)
                    approved_face_sim = float(cand_sim)
                    approved_face_member_id = int(cand_mid)
                    approved_face_emb = face_emb
                else:
                    face_hit = False
                    face_label = ""
                    face_sim = 0.0
                    face_gap = 0.0
                    face_member_id = -1
                    face_emb = None
        face_hit = bool(approved_face_label)
        face_label = str(approved_face_label)
        face_sim = float(approved_face_sim)
        face_gap = float(approved_face_gap if face_hit else 0.0)
        face_member_id = int(approved_face_member_id if face_hit else -1)
        face_emb = approved_face_emb if face_hit else None
        face_det_score = float(approved_face_det_score if face_hit else 1.0)
        entry["face_vis_ttl"] = max(0, int(entry.get("face_vis_ttl", 0)) - 1)
        if face_hit and face_label:
            entry["face_vis_ttl"] = max(1, int(args.face_hold_frames))
            entry["last_face_label"] = face_label
            entry["last_face_sim"] = float(face_sim)
        tracks_info.append({
            "tid": tid,
            "bbox": (x1, y1, x2, y2),
            "det_conf": det_conf,
            "face_hit": face_hit,
            "face_label": face_label,
            "face_sim": face_sim,
            "face_gap": face_gap,
            "face_det_score": float(face_det_score),
            "face_member_id": int(face_member_id),
            "face_emb": face_emb,
            "face_vis_ttl": int(entry.get("face_vis_ttl", 0)),
            "last_face_sim": float(entry.get("last_face_sim", 0.0)),
            "low_face_hit": bool(low_face_hit),
            "low_face_sim": float(low_face_sim),
            "low_face_det_score": float(low_face_det_score),
        })

    face_winners: Dict[str, Tuple[int, float, int]] = {}
    for r in tracks_info:
        if r["face_hit"] and r["face_label"]:
            lab = str(r["face_label"])
            sim = float(r["face_sim"])
            tid = int(r["tid"])
            mid = int(r.get("face_member_id", -1))
            prev = face_winners.get(lab)
            if prev is None or sim > prev[1]:
                face_winners[lab] = (tid, sim, mid)
    for lab, (winner_tid, sim, mid) in face_winners.items():
        w_ent = identity_state.get(winner_tid)
        if isinstance(w_ent, dict):
            w_ent["assigned_name"] = str(lab)
            if mid > 0:
                w_ent["assigned_member_id"] = int(mid)
            w_ent["assigned_score"] = max(float(w_ent.get("assigned_score", 0.0)), float(sim))
            w_ent["confirmed_face_label"] = str(lab)
            if mid > 0:
                w_ent["confirmed_face_member_id"] = int(mid)
            w_ent["confirmed_face_sim"] = max(float(w_ent.get("confirmed_face_sim", 0.0)), float(sim))

    body_emb_by_tid: Dict[int, np.ndarray] = {}

    if embed_updater is not None and face_winners:
        ts_use = float(ts_cap) if float(ts_cap or 0.0) > 0 else time.time()
        sim_thresh = float(getattr(args, "update_face_sim_thresh", 0.75))
        det_thresh = float(getattr(args, "embed_min_face_det_score", 0.75))
        for lab, (winner_tid, sim, member_id) in face_winners.items():
            sim_f = float(sim or 0.0)
            if sim_f < sim_thresh:
                continue
            if int(member_id) <= 0:
                continue
            fm = face_for_tid.get(int(winner_tid), None)
            if not isinstance(fm, dict):
                continue
            try:
                det_score = float(fm.get("det_score", 1.0))
            except Exception:
                det_score = 1.0
            if det_score < det_thresh:
                continue
            try:
                if hasattr(embed_updater, "can_accept") and (not embed_updater.can_accept(int(member_id), int(camera_db_id))):
                    continue
            except Exception:
                pass
            face_emb = fm.get("emb", None)
            body_emb = None
            embed_updater.enqueue(EmbeddingSample(
                member_id=int(member_id), name=str(lab), camera_id=int(camera_db_id), track_id=int(winner_tid),
                ts=float(ts_use), face_sim=float(sim_f), face_det_score=float(det_score), body_emb=body_emb, face_emb=face_emb,
            ))

    draw_candidates: List[DrawItem] = []
    for r in tracks_info:
        tid = int(r["tid"])
        x1, y1, x2, y2 = r["bbox"]
        face_hit = bool(r["face_hit"])
        face_label = str(r["face_label"])
        face_sim = float(r["face_sim"])
        last_face_sim = float(r.get("last_face_sim", 0.0))
        face_mid = int(r.get("face_member_id", -1))
        body_label = ""
        body_sim = 0.0
        stable_name, stable_score, entry = update_track_identity(
            identity_state,
            tid,
            face_label=face_label if face_hit else "",
            face_sim=float(face_sim if face_hit else 0.0),
            body_label=body_label,
            body_sim=body_sim,
            decay=args.name_decay,
            min_score=args.name_min_score,
            margin=args.name_margin,
            ttl_reset=args.name_ttl,
            w_face=args.name_face_weight,
            w_body=args.name_body_weight,
            lock_frames=int(getattr(args, "identity_lock_frames", 30)),
            lock_face_thresh=float(getattr(args, "identity_lock_face_thresh", 0.50)),
        )
        if stable_name:
            entry["assigned_name"] = str(stable_name)
            if face_hit and face_mid > 0:
                entry["assigned_member_id"] = int(face_mid)
                if bool(entry.get("locked", False)):
                    entry["lock_label"] = str(stable_name)
            else:
                entry["assigned_member_id"] = int(name_to_member_id.get(str(stable_name), entry.get("assigned_member_id", -1)))
            entry["assigned_score"] = float(stable_score)

        curr_bbox = (int(x1), int(y1), int(x2), int(y2))
        same_person = same_person_continuation(
            entry.get("last_good_bbox", None),
            curr_bbox,
            min_iou=float(getattr(args, "name_continuity_iou", 0.05)),
            max_center_shift=float(getattr(args, "name_continuity_center_shift", 0.60)),
        )
        if face_hit and face_label:
            same_person = True
            entry["last_good_bbox"] = curr_bbox
            entry["continuity_break_frames"] = 0
        elif same_person:
            entry["continuity_break_frames"] = 0
        else:
            entry["continuity_break_frames"] = int(entry.get("continuity_break_frames", 0)) + 1

        persist_names = not bool(getattr(args, "no_persist_names", False))
        track_has_approved_face = bool(entry.get("has_approved_face", False)) and bool(entry.get("confirmed_face_label", ""))
        if persist_names:
            keep_cached_name = track_has_approved_face and same_person and (
                bool(entry.get("assigned_name", "")) or bool(stable_name) or int(entry.get("lock_ttl", 0)) > 0 or
                int(entry.get("ttl", 0)) > 0 or int(entry.get("face_vis_ttl", 0)) > 0
            )
            final_name = str(entry.get("assigned_name", "") or stable_name or "") if keep_cached_name else ""
        else:
            final_name = str(stable_name or "") if track_has_approved_face else ""
            if (not face_hit) and (not same_person):
                final_name = ""
        if final_name and str(entry.get("confirmed_face_label", "") or "") and str(final_name) != str(entry.get("confirmed_face_label", "")):
            final_name = ""
            demote_identity_entry(entry, clear_face=True)
        final_mid = int(entry.get("assigned_member_id", -1)) if final_name else -1
        if (not face_hit) and (not same_person):
            final_name = ""
            final_mid = -1
            if int(entry.get("continuity_break_frames", 0)) >= max(1, int(getattr(args, "name_break_clear_frames", 3))):
                entry["last"] = ""
                entry["ttl"] = 0
                entry["assigned_name"] = ""
                entry["assigned_member_id"] = -1
                entry["assigned_score"] = 0.0
                entry["confirmed_face_label"] = ""
                entry["confirmed_face_member_id"] = -1
                entry["confirmed_face_sim"] = 0.0
                entry["has_approved_face"] = False
                entry["pending_face_label"] = ""
                entry["pending_face_member_id"] = -1
                entry["pending_face_count"] = 0
                entry["pending_face_best_sim"] = 0.0
                entry["locked"] = False
                entry["lock_ttl"] = 0
                entry["lock_label"] = ""
                entry["last_good_bbox"] = None
        if final_name and camera_name_owner is not None:
            owner_tid = camera_name_owner.owner_tid(str(final_name), frame_idx=int(frame_idx))
            if owner_tid is not None and int(owner_tid) != int(tid):
                final_name = ""
                final_mid = -1
                demote_identity_entry(entry, clear_face=True)
        if final_name:
            entry["last_good_bbox"] = curr_bbox

        low_face_hit = bool(r.get("low_face_hit", False))
        low_face_sim = float(r.get("low_face_sim", 0.0) or 0.0)
        if hide_unknown and not final_name:
            continue
        disp_face_sim = float(face_sim) if face_hit else float(entry.get("last_face_sim", last_face_sim))
        draw_candidates.append(DrawItem(
            tid=tid,
            bbox=curr_bbox,
            name=str(final_name),
            member_id=int(final_mid),
            face_sim=float(disp_face_sim),
            stable_score=float(stable_score),
            det_conf=r.get("det_conf", None),
            face_hit=bool(face_hit),
            low_face=bool(low_face_hit),
            low_face_sim=float(low_face_sim),
        ))

    if not bool(getattr(args, "allow_duplicate_names", False)):
        draw_final, demoted_tids = block_duplicate_names(draw_candidates)
        for demoted_tid in list(demoted_tids):
            demote_identity_entry(identity_state.get(int(demoted_tid)), clear_face=True)
    else:
        draw_final = draw_candidates
    if bool(getattr(args, "global_unique_names", False)) and global_owner is not None:
        gated: List[DrawItem] = []
        for it in draw_final:
            x1, y1, x2, y2 = it.bbox
            is_known = bool(it.name)
            if hide_unknown and (not is_known):
                continue
            score = float(it.face_sim)
            if global_owner.allow(it.name, sid=int(sid), score=score):
                gated.append(it)
                continue
            demote_identity_entry(identity_state.get(int(it.tid)), clear_face=True)
            gated.append(DrawItem(
                tid=int(it.tid),
                bbox=it.bbox,
                name="",
                member_id=-1,
                face_sim=float(it.face_sim),
                stable_score=float(it.stable_score),
                det_conf=it.det_conf,
                face_hit=bool(it.face_hit),
                low_face=bool(it.low_face),
                low_face_sim=float(it.low_face_sim),
            ))
        draw_final = gated

    present_conf: Dict[str, float] = {}
    for it in draw_final:
        if it.name:
            try:
                present_conf[it.name] = max(float(present_conf.get(it.name, 0.0)), float(it.face_sim or 0.0))
            except Exception:
                present_conf[it.name] = float(present_conf.get(it.name, 0.0))
    present_names = sorted(present_conf.keys())

    shown = 0
    events: List[Tuple[int, int, int, int, int, str, float, int]] = []
    for it in draw_final:
        x1, y1, x2, y2 = it.bbox

        is_known = bool(it.name)

        # 🔴 FIXED LOGIC (STRICT)
        if bool(args.hide_unknown) and not is_known:
            continue

        if is_known:
            color = (0, 255, 0)
            if it.face_hit and float(it.face_sim) < 0.50:
                color = (0, 0, 255)
            label_txt = f"{it.name}"
        else:
            if bool(it.low_face) and float(it.low_face_sim) > 0:
                color = (0, 0, 255)
            else:
                color = (0, 255, 255)
            label_txt = "Unknown"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        cv2.putText(out, label_txt, (x1, max(0, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)

        shown += 1

        event_sim = float(it.face_sim if is_known else it.low_face_sim)
        events.append((int(it.tid), x1, y1, x2, y2,
                    str(it.name), float(event_sim),
                    int(it.member_id), int(1 if is_known else 0)))

    cleanup_frames = max(30, int(getattr(args, "max_age", 15)) + int(getattr(args, "iou_max_miss", 5)) + 10)
    for tid in list(identity_state.keys()):
        st = identity_state.get(tid, {})
        lf = int(st.get("last_seen_frame", -1)) if isinstance(st, dict) else -1
        if lf >= 0 and (int(frame_idx) - lf) > cleanup_frames:
            identity_state.pop(tid, None)

    meta = {
        "tracks": int(len(tracks_info)),
        "shown": int(shown),
        "faces_recognized": int(len(recognized_faces)),
        "do_face": bool(do_face),
        "events": events,
        "present_names": present_names,
        "present_conf": present_conf,
    }
    return out, meta



def processor_thread(
    sid: int,
    camera_db_id: int,
    vs,
    render_store: RenderedFrame,
    yolo,
    args,
    deep_tracker,
    iou_tracker: IOUTracker,
    gallery_mgr: GalleryManager,
    reid_extractor,
    face_app,
    global_owner: Optional[GlobalNameOwner],
    report,
    normalized_report,
    embed_updater: Optional[EmbeddingDBUpdater],
    debug: bool = False,
    save_writer: Optional['SegmentedVideoWriter'] = None,
):
    frame_idx = 0
    identity_state: dict[int, dict] = {}
    camera_name_owner = CameraNameOwner(args)
    last_t = time.time()
    fps_ema = 0.0
    alpha = 0.10
    device_is_cuda = torch.cuda.is_available() and ("cuda" in str(args.device).lower())
    while True:
        ok, frame, ts_cap = vs.read()
        if not ok or frame is None:
            if bool(getattr(vs, "is_file_source", False)) and hasattr(vs, "is_finished") and vs.is_finished():
                break
            time.sleep(0.005)
            continue
        if int(args.max_queue_age_ms) > 0 and (not bool(getattr(vs, "is_file_source", False))):
            now = time.time()
            age_ms = (now - float(ts_cap)) * 1000.0
            dropped_here = 0
            while age_ms > float(args.max_queue_age_ms) and dropped_here < int(args.max_drain_per_cycle):
                try:
                    vs.read_dropped = int(getattr(vs, "read_dropped", 0)) + 1
                except Exception:
                    pass
                ok2, frame2, ts2 = vs.read()
                if not ok2 or frame2 is None:
                    break
                frame, ts_cap = frame2, ts2
                age_ms = (time.time() - float(ts_cap)) * 1000.0
                dropped_here += 1
        try:
            gallery_mgr.maybe_reload(args)
            people_by_cam, face_gallery, name_to_mid = gallery_mgr.snapshot()
            people = people_by_cam.get(int(camera_db_id), [])
            out, meta = process_one_frame(
                frame_idx, frame, sid, int(camera_db_id), yolo, args, deep_tracker, iou_tracker,
                people, reid_extractor, face_app, face_gallery, name_to_mid, identity_state,
                device_is_cuda=device_is_cuda, global_owner=global_owner, ts_cap=float(ts_cap),
                embed_updater=embed_updater, camera_name_owner=camera_name_owner,
            )
            ts_use = float(ts_cap) if ts_cap else time.time()
            if report is not None:
                _update_report_from_meta(report, args=args, sid=int(sid), camera_db_id=int(camera_db_id), meta=meta, ts_use=ts_use)
            if (normalized_report is not None) and (normalized_report is not report):
                _update_report_from_meta(normalized_report, args=args, sid=int(sid), camera_db_id=int(camera_db_id), meta=meta, ts_use=ts_use)
            now = time.time()
            dt = max(1e-6, now - last_t)
            inst_fps = 1.0 / dt
            fps_ema = (1 - alpha) * fps_ema + alpha * inst_fps
            last_t = now
            if args.overlay_fps:
                qsz = int(vs.qsize()) if hasattr(vs, "qsize") else 0
                dropped_cap = int(getattr(vs, "dropped", 0))
                dropped_read = int(getattr(vs, "read_dropped", 0))
                lag_ms = (now - float(ts_cap)) * 1000.0 if ts_cap else 0.0
                lines = [
                    f"SRC {sid} (cam_id={camera_db_id}) | FPS {fps_ema:.1f} | lag {lag_ms:.0f}ms",
                    f"q {qsz} | drop(cap) {dropped_cap} | drop(stale) {dropped_read}",
                    f"tracks {meta.get('tracks', 0)} | shown {meta.get('shown', 0)} | faces {meta.get('faces_recognized', 0)}",
                ]
                y = 22
                for ln in lines:
                    cv2.putText(out, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y += 22
            frame_meta = dict(meta or {})
            frame_meta.update({"fps": float(fps_ema), "sid": int(sid), "camera_db_id": int(camera_db_id), "frame_idx": int(frame_idx)})
            render_store.set(out, meta=frame_meta)
            if save_writer is not None:
                try:
                    save_writer.write(out)
                except Exception:
                    pass
            frame_idx += 1
        except Exception as e:
            if debug:
                print(f"[PROC {sid}] error:", e)
            time.sleep(0.001)


def _norm_ext(ext: str) -> str:
    ext = str(ext or "").strip()
    if not ext:
        return ".mp4"
    if not ext.startswith("."):
        ext = "." + ext
    return ext


def _open_writer_with_fallback(path: str, fps: float, size_wh: Tuple[int, int], preferred_fourcc: str) -> Optional[cv2.VideoWriter]:
    w, h = int(size_wh[0]), int(size_wh[1])
    fps = float(max(1.0, fps))
    codecs = []
    p = str(preferred_fourcc or "mp4v")[:4]
    codecs.append(p)
    for c in ["mp4v", "avc1", "XVID", "MJPG"]:
        if c not in codecs:
            codecs.append(c)
    for c in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*c)
            vw = cv2.VideoWriter(path, fourcc, fps, (w, h))
            if vw is not None and vw.isOpened():
                print(f"[VIDEO] Writer opened: codec={c} fps={fps} size={w}x{h} path={path}")
                return vw
            try:
                if vw is not None:
                    vw.release()
            except Exception:
                pass
        except Exception:
            pass
    print(f"[WARN] Could not open VideoWriter at: {path}")
    return None


class SegmentedVideoWriter:
    def __init__(self, out_dir: str, basename: str = "annotated", fps: float = 20.0,
                 fourcc: str = "mp4v", ext: str = ".mp4", segment_seconds: int = 600,
                 save_height: int = 480, realtime_pacing: bool = True):
        self.out_dir = out_dir
        self.basename = basename
        self.fps = float(max(1.0, float(fps)))
        self.fourcc = str(fourcc)
        self.ext = str(ext) if str(ext).startswith(".") else "." + str(ext)
        self.segment_seconds = int(segment_seconds)
        self.save_height = int(max(0, int(save_height or 0)))
        self.realtime_pacing = bool(realtime_pacing)
        os.makedirs(self.out_dir, exist_ok=True)
        self.run_dir = os.path.join(self.out_dir, "run_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)
        self._vw: Optional[cv2.VideoWriter] = None
        self._size_wh: Optional[Tuple[int, int]] = None
        self._seg_start_mono = 0.0
        self._seg_index = 0
        self._current_path: Optional[str] = None
        self.segments_started = 0
        self.segments_closed = 0
        self._segment_frames_written = 0
        self._period_s: float = 1.0 / float(max(1.0, self.fps))
        self._next_frame_mono: float = 0.0
        self._max_catchup_frames: int = int(max(1, round(self.fps * 2.0)))

    @staticmethod
    def _make_even(x: int) -> int:
        xi = int(x)
        return xi if (xi % 2) == 0 else max(2, xi - 1)

    def _prepare_frame_for_save(self, frame_bgr: np.ndarray) -> np.ndarray:
        if frame_bgr is None:
            return frame_bgr
        if self.save_height <= 0:
            return frame_bgr
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return frame_bgr
        if h <= self.save_height:
            return frame_bgr
        new_h = self._make_even(self.save_height)
        new_w = int(round(w * (new_h / float(h))))
        new_w = self._make_even(max(2, new_w))
        try:
            return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        except Exception:
            return frame_bgr

    def _close_current(self):
        if self._vw is not None:
            try:
                self._vw.release()
            except Exception:
                pass
            self._vw = None
            self.segments_closed += 1
        self._current_path = None
        self._size_wh = None
        self._seg_start_mono = 0.0
        self._segment_frames_written = 0

    def _try_open_path(self, path: str, frame_wh: Tuple[int, int]) -> Optional[cv2.VideoWriter]:
        return _open_writer_with_fallback(path, fps=self.fps, size_wh=frame_wh, preferred_fourcc=self.fourcc)

    def _open_new(self, frame_wh: Tuple[int, int], start_mono: Optional[float] = None):
        self._close_current()
        self._size_wh = frame_wh
        self._seg_start_mono = float(start_mono) if start_mono is not None else time.monotonic()
        ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.run_dir, f"{self.basename}_{ts_tag}_p{self._seg_index:04d}{self.ext}")
        vw = self._try_open_path(path, frame_wh=frame_wh)
        if vw is None:
            try:
                p = Path(path)
                avi_path = str(p.with_suffix(".avi"))
            except Exception:
                avi_path = path + ".avi"
            vw = self._try_open_path(avi_path, frame_wh=frame_wh)
            if vw is not None:
                path = avi_path
                print(f"[VIDEO] Falling back to AVI container: {path}")
        if vw is None:
            return
        self._vw = vw
        self._current_path = path
        self._seg_index += 1
        self.segments_started += 1

    def write(self, frame_bgr: np.ndarray):
        if frame_bgr is None:
            return
        frame_bgr = self._prepare_frame_for_save(frame_bgr)
        h, w = frame_bgr.shape[:2]
        if h <= 0 or w <= 0:
            return
        frame_wh = (int(w), int(h))
        if not self.realtime_pacing:
            if self._vw is None or self._size_wh != frame_wh:
                self._open_new(frame_wh)
                if self._vw is None:
                    return
            if self.segment_seconds > 0:
                seg_elapsed_s = float(self._segment_frames_written) / float(max(1.0, self.fps))
                if seg_elapsed_s >= float(self.segment_seconds):
                    self._open_new(frame_wh)
                    if self._vw is None:
                        return
            try:
                self._vw.write(frame_bgr)
            except Exception:
                self._close_current()
                return
            self._segment_frames_written += 1
            return
        now_mono = time.monotonic()
        if self._next_frame_mono <= 0.0:
            self._next_frame_mono = float(now_mono)
        if self._vw is None or self._size_wh != frame_wh:
            self._open_new(frame_wh, start_mono=self._next_frame_mono)
            if self._vw is None:
                return
        wrote = 0
        while (now_mono >= self._next_frame_mono) and (wrote < self._max_catchup_frames):
            if self.segment_seconds > 0 and (self._next_frame_mono - float(self._seg_start_mono)) >= float(self.segment_seconds):
                self._open_new(frame_wh, start_mono=self._next_frame_mono)
                if self._vw is None:
                    return
            try:
                self._vw.write(frame_bgr)
            except Exception:
                self._close_current()
                return
            self._segment_frames_written += 1
            self._next_frame_mono += float(self._period_s)
            wrote += 1
        if wrote >= self._max_catchup_frames and now_mono >= self._next_frame_mono:
            self._next_frame_mono = float(now_mono) + float(self._period_s)

    def close(self):
        self._close_current()


def main():
    args = parse_args()
    if not args.use_db:
        raise SystemExit("This script is DB-first. Use: --use-db --db-url ...")
    if not args.db_url:
        raise SystemExit("--use-db requires --db-url")
    if not getattr(args, "camera_ids", None):
        args.camera_ids = []
    if len(args.camera_ids) == 0:
        args.camera_ids = list(range(1, len(args.src) + 1))
    if len(args.camera_ids) != len(args.src):
        raise SystemExit("--camera-ids must have the same length as --src")
    args.save_video = not bool(getattr(args, "no_save_video", False))
    args.global_unique_names = (len(args.src) > 1) and (not bool(getattr(args, "no_global_unique_names", False)))
    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True
    try:
        torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
    except Exception:
        pass
    gpu = torch.cuda.is_available() and ("cuda" in str(args.device).lower())
    if gpu:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    if args.half and not gpu:
        print("[WARN] --half requested but CUDA not available; disabling FP16.")
        args.half = False
    print(f"[INIT] device={args.device} cuda_available={torch.cuda.is_available()} half={args.half}")

    gallery_mgr = GalleryManager(args)
    global_owner: Optional[GlobalNameOwner] = None
    if bool(getattr(args, "global_unique_names", False)):
        global_owner = GlobalNameOwner(
            hold_seconds=float(getattr(args, "global_hold_seconds", 0.5)),
            switch_margin=float(getattr(args, "global_switch_margin", 0.02)),
        )
        print(f"[INIT] Global unique names: ON (hold={global_owner.hold_seconds}s, margin={global_owner.switch_margin})")
    else:
        print("[INIT] Global unique names: OFF")

    yolo = None
    if YOLO is not None:
        try:
            weights = args.yolo_weights
            if not Path(weights).exists():
                print(f"[INIT] {weights} not found, falling back to yolov8n.pt")
                weights = "yolov8n.pt"
            yolo = YOLO(weights)
            if gpu:
                try:
                    yolo.to(args.device)
                except Exception as e:
                    print("[WARN] YOLO .to(device) failed:", e)
            print("[INIT] YOLO ready")
        except Exception as e:
            print("[ERROR] YOLO load failed:", e)
            yolo = None
    else:
        print("[WARN] ultralytics not installed; detection disabled")

    reid_extractor = None
    print("[INIT] Face-only identity mode: external body ReID matcher disabled.")

    face_app = init_face_engine(
        args.use_face,
        args.device,
        args.face_model,
        int(args.face_det_size[0]),
        int(args.face_det_size[1]),
        face_provider=getattr(args, "face_provider", "auto"),
        ort_log=getattr(args, "ort_log", False),
    )

    tracker_backend = 'iou'
    if bool(getattr(args, 'use_strongsort', False)):
        tracker_backend = 'strongsort'
    elif bool(getattr(args, 'use_bytetrack', False)):
        tracker_backend = 'bytetrack'
    elif bool(getattr(args, 'use_deepsort', False)):
        tracker_backend = 'deepsort'
    elif bool(getattr(args, 'no_deepsort', False)):
        tracker_backend = 'iou'
    else:
        if DeepSort is not None:
            tracker_backend = 'deepsort'
        elif BoxByteTrack is not None:
            tracker_backend = 'bytetrack'
        else:
            tracker_backend = 'iou'

    strongsort_weights: Optional[Path] = None
    if tracker_backend == 'deepsort' and DeepSort is None:
        print('[WARN] DeepSORT selected but deep-sort-realtime not installed. Falling back to IoU tracker.')
        tracker_backend = 'iou'
    if tracker_backend == 'bytetrack' and BoxByteTrack is None:
        print('[WARN] ByteTrack selected but boxmot not installed. Falling back to IoU tracker.')
        tracker_backend = 'iou'
    if tracker_backend == 'strongsort':
        if BoxStrongSort is None:
            print('[WARN] StrongSORT selected but boxmot not installed. Falling back to IoU tracker.')
            tracker_backend = 'iou'
        else:
            strongsort_weights = resolve_strongsort_reid_weights(args)
            if strongsort_weights is None:
                print('[WARN] StrongSORT selected but no ReID weights found. Falling back to IoU tracker.')
                tracker_backend = 'iou'
    print(f'[INIT] Tracker backend: {tracker_backend}')
    if tracker_backend == 'strongsort' and strongsort_weights is not None:
        print(f'[INIT] StrongSORT ReID weights: {strongsort_weights}')

    normalized_data_writer = _create_normalized_data_writer(args)
    report, normalized_report, csv_stop_evt, csv_thread = _create_tracking_reports(
        args=args,
        num_cams=len(args.src),
        normalized_writer=normalized_data_writer,
    )

    embed_updater: Optional[EmbeddingDBUpdater] = None
    if bool(getattr(args, "update_db_embeddings", False)):
        if not bool(getattr(args, "use_face", False)):
            print("[WARN] --update-db-embeddings requires --use-face. Disabling embedding updates.")
        else:
            os.makedirs("csv_output", exist_ok=True)
            ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_path = str(getattr(args, "embeddings_log_csv", "") or "").strip()
            if not log_path:
                log_path = os.path.join("csv_output", f"embeddings_updates_{ts_tag}.csv")
            samples_log_path = str(getattr(args, "embeddings_samples_log_csv", "") or "").strip()
            if not samples_log_path:
                samples_log_path = os.path.join("csv_output", f"embeddings_samples_{ts_tag}.csv")
            embed_updater = EmbeddingDBUpdater(
                db_url=args.db_url,
                slot_mb=float(getattr(args, "embeddings_slot_mb", 0.5)),
                flush_seconds=float(getattr(args, "embeddings_flush_seconds", 10.0)),
                min_sample_seconds=float(getattr(args, "embeddings_min_sample_seconds", 0.5)),
                min_face_sim=float(getattr(args, "update_face_sim_thresh", 0.75)),
                min_face_det_score=float(getattr(args, "embed_min_face_det_score", 0.75)),
                log_csv_path=log_path,
                update_body=False,
                update_face=not bool(getattr(args, "no_update_face_bank", False)),
                reset_if_gap_days_ge=int(getattr(args, "embeddings_reset_if_gap_days", 2)),
                reset_on_start=False,
                samples_log_csv_path=samples_log_path,
            )
            print(f"[INIT] DB embedding updater: ON (slot_mb={float(getattr(args,'embeddings_slot_mb',0.5)):.3f} => {2*float(getattr(args,'embeddings_slot_mb',0.5)):.3f}MB total) (updates_log={log_path}, samples_log={samples_log_path})")

    streams = []
    for i, raw_src in enumerate(args.src):
        src = raw_src.strip() if isinstance(raw_src, str) else raw_src
        camera_db_id = int(args.camera_ids[i])
        vs = AdaptiveQueueStream(
            src, queue_size=args.queue_size, rtsp_transport=args.rtsp_transport, use_opencv=True,
            freeze_seconds=float(args.stream_freeze_seconds), open_timeout_ms=int(args.stream_open_timeout_ms),
            read_timeout_ms=int(args.stream_read_timeout_ms), reconnect_base_delay=float(args.stream_reconnect_base_seconds),
            reconnect_max_delay=float(args.stream_reconnect_max_seconds), reconnect_jitter=float(args.stream_reconnect_jitter),
            reconnect_log_interval=float(args.stream_reconnect_log_interval),
        )
        deep_tracker = None
        if tracker_backend == 'deepsort':
            try:
                deep_tracker = DeepSort(
                    max_age=int(args.max_age), n_init=int(args.n_init), nn_budget=int(args.nn_budget),
                    max_cosine_distance=float(args.tracker_max_cosine), nms_max_overlap=float(args.tracker_nms_overlap),
                    embedder="torchreid", embedder_gpu=gpu, half=(gpu and args.half), bgr=True,
                )
            except Exception as e:
                print(f"[WARN] DeepSORT init failed for SRC {i}, fallback to IoU tracker:", e)
                deep_tracker = None
        elif tracker_backend == 'bytetrack':
            try:
                deep_tracker = BoxByteTrack(
                    det_thresh=float(args.conf), max_age=int(args.max_age), max_obs=max(50, int(args.max_age) + 5),
                    min_hits=int(args.n_init), iou_threshold=float(getattr(args, 'max_iou_distance', 0.30)),
                    min_conf=float(getattr(args, 'bytetrack_min_conf', 0.10)), track_thresh=float(getattr(args, 'bytetrack_track_thresh', 0.45)),
                    match_thresh=float(getattr(args, 'bytetrack_match_thresh', 0.80)), track_buffer=int(getattr(args, 'bytetrack_track_buffer', 25)),
                    frame_rate=int(getattr(args, 'bytetrack_frame_rate', 30)),
                )
            except Exception as e:
                print(f"[WARN] ByteTrack init failed for SRC {i}, fallback to IoU tracker:", e)
                deep_tracker = None
        elif tracker_backend == 'strongsort':
            try:
                dev = torch.device(args.device) if gpu else torch.device('cpu')
                deep_tracker = BoxStrongSort(
                    reid_weights=strongsort_weights, device=dev, half=(gpu and bool(args.half)),
                    det_thresh=float(args.conf), max_age=int(args.max_age), max_obs=max(50, int(args.max_age) + 5),
                    min_hits=int(args.n_init), iou_threshold=float(getattr(args, 'max_iou_distance', 0.30)),
                    min_conf=float(args.conf), max_cos_dist=float(args.tracker_max_cosine), n_init=int(args.n_init), nn_budget=int(args.nn_budget),
                )
            except Exception as e:
                print(f"[WARN] StrongSORT init failed for SRC {i}, fallback to IoU tracker:", e)
                deep_tracker = None
        iou_tracker = IOUTracker(max_miss=max(1, int(args.iou_max_miss)), iou_thresh=float(getattr(args, 'max_iou_distance', 0.30)))
        streams.append({"sid": i, "camera_db_id": camera_db_id, "src": raw_src, "vs": vs, "deep": deep_tracker, "iou": iou_tracker})

    print(f"[INIT] sources requested: {len(args.src)}")
    for s in streams:
        print(f"[SRC {s['sid']}] cam_id={s['camera_db_id']} open={s['vs'].is_opened()} :: {s['src']}")
    if not any(s["vs"].is_opened() for s in streams):
        print("[ERROR] No sources opened. Check your --src URLs/paths and codecs.")
        for s in streams:
            try:
                s["vs"].release()
            except Exception:
                pass
        try:
            if csv_stop_evt is not None:
                csv_stop_evt.set()
            if csv_thread is not None:
                csv_thread.join(timeout=2.0)
        except Exception:
            pass
        _finalize_tracking_reports(report=report, normalized_report=normalized_report, args=args)
        if normalized_data_writer is not None:
            try:
                normalized_data_writer.close()
            except Exception:
                pass
        if embed_updater is not None:
            try:
                embed_updater.close()
            except Exception:
                pass
        return

    render_map: dict[int, RenderedFrame] = {s["sid"]: RenderedFrame() for s in streams}
    last_good: dict[int, np.ndarray] = {}
    worker_threads: List[threading.Thread] = []
    print("[Main] Running. Press 'q' to quit (when --show).")

    win_name = f"YOLO + {tracker_backend.upper()} + member_embeddings + anti-swap"
    screen_w, screen_h = (0, 0)
    if args.show or args.save_video:
        screen_w, screen_h = _get_screen_resolution(default=(1920, 1080))
    if args.show:
        cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
        try:
            cv2.resizeWindow(win_name, int(screen_w), int(screen_h))
            cv2.moveWindow(win_name, 0, 0)
        except Exception:
            pass
        if bool(getattr(args, "fullscreen", False)):
            try:
                cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            except Exception:
                pass

    seg_writer: Optional[SegmentedVideoWriter] = None
    direct_save_writers: Dict[int, SegmentedVideoWriter] = {}
    direct_single_file_writer = bool(args.save_video and len(streams) == 1 and bool(getattr(streams[0]["vs"], "is_file_source", False)))
    if args.save_video:
        out_dir = str(getattr(args, "video_dir", "saved_videos") or "saved_videos")
        os.makedirs(out_dir, exist_ok=True)
        if direct_single_file_writer:
            only_vs = streams[0]["vs"]
            src_fps = float(getattr(only_vs, "_source_fps", 0.0) or 0.0)
            save_fps = float(src_fps if src_fps > 0.0 else (getattr(args, "video_fps", 20.0) or 20.0))
            seg_writer = SegmentedVideoWriter(
                out_dir=out_dir, basename=str(getattr(args, "video_prefix", "saved_video") or "saved_video"),
                ext=_norm_ext(getattr(args, "video_ext", ".mp4")), fps=save_fps, fourcc=str(getattr(args, "video_fourcc", "mp4v") or "mp4v"),
                segment_seconds=int(float(getattr(args, "video_segment_seconds", 3600.0) or 0.0)), save_height=int(getattr(args, "video_save_height", 480) or 0),
                realtime_pacing=False,
            )
            direct_save_writers[int(streams[0]["sid"])] = seg_writer
            print(f"[INIT] Saving recorded single-video output directly from processing thread: {seg_writer.run_dir}")
            print(f"[INIT] Recorded source FPS for writer: {save_fps:.3f}")
        else:
            seg_writer = SegmentedVideoWriter(
                out_dir=out_dir, basename=str(getattr(args, "video_prefix", "saved_video") or "saved_video"),
                ext=_norm_ext(getattr(args, "video_ext", ".mp4")), fps=float(getattr(args, "video_fps", 20.0) or 20.0),
                fourcc=str(getattr(args, "video_fourcc", "mp4v") or "mp4v"), segment_seconds=int(float(getattr(args, "video_segment_seconds", 3600.0) or 0.0)),
                save_height=int(getattr(args, "video_save_height", 480) or 0), realtime_pacing=True,
            )
            print(f"[INIT] Saving annotated videos to folder: {seg_writer.run_dir}")
            if seg_writer.segment_seconds > 0:
                print(f"[INIT] Video segmentation: ON ({seg_writer.segment_seconds:.0f}s per file)")
            else:
                print("[INIT] Video segmentation: OFF (single file)")

    for s in streams:
        sid = int(s["sid"])
        save_writer = direct_save_writers.get(sid)
        t = threading.Thread(
            target=processor_thread,
            args=(sid, int(s["camera_db_id"]), s["vs"], render_map[sid], yolo, args, s["deep"], s["iou"], gallery_mgr,
                  reid_extractor, face_app, global_owner, report, normalized_report, embed_updater, False),
            kwargs={"save_writer": save_writer},
            daemon=True,
        )
        t.start()
        worker_threads.append(t)

    disp_last = time.time()
    disp_fps_ema = 0.0
    disp_alpha = 0.10
    try:
        while True:
            display_frames: List[np.ndarray] = []
            any_open = False
            for s in streams:
                sid = int(s["sid"])
                vs = s["vs"]
                if not vs.is_opened():
                    display_frames.append(last_good.get(sid, np.zeros((720, 1280, 3), dtype=np.uint8)))
                    continue
                any_open = True
                frm, _ts, _meta = render_map[sid].get()
                if frm is not None:
                    last_good[sid] = frm
                    display_frames.append(frm)
                else:
                    display_frames.append(last_good.get(sid, np.zeros((720, 1280, 3), dtype=np.uint8)))
            if display_frames and (args.show or (args.save_video and (not direct_single_file_writer))):
                if screen_w <= 0 or screen_h <= 0:
                    screen_w, screen_h = _get_screen_resolution(default=(1920, 1080))
                vis = make_grid_view(display_frames, screen_w=int(screen_w), screen_h=int(screen_h),
                                     mode=str(getattr(args, "grid_mode", "cover")),
                                     grid_rows=int(getattr(args, "grid_rows", 0) or 0),
                                     grid_cols=int(getattr(args, "grid_cols", 0) or 0))
                now = time.time()
                dt = max(1e-6, now - disp_last)
                disp_fps = 1.0 / dt
                disp_fps_ema = (1 - disp_alpha) * disp_fps_ema + disp_alpha * disp_fps
                disp_last = now
                cv2.putText(vis, f"DISPLAY FPS {disp_fps_ema:.1f} | cams {len(display_frames)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                if args.save_video and seg_writer is not None and (not direct_single_file_writer):
                    seg_writer.write(vis)
                if args.show:
                    cv2.imshow(win_name, vis)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            if not any_open:
                break
            time.sleep(0.001)
    finally:
        for s in streams:
            try:
                s["vs"].release()
            except Exception:
                pass
        for t in worker_threads:
            try:
                t.join(timeout=5.0)
            except Exception:
                pass
        if seg_writer is not None:
            try:
                seg_writer.close()
            except Exception:
                pass
            print(f"[DONE] Saved videos folder: {seg_writer.run_dir} (segments_closed={seg_writer.segments_closed})")
        if args.show:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
        try:
            if csv_stop_evt is not None:
                csv_stop_evt.set()
            if csv_thread is not None:
                csv_thread.join(timeout=2.0)
        except Exception:
            pass
        _finalize_tracking_reports(report=report, normalized_report=normalized_report, args=args)
        if normalized_data_writer is not None:
            try:
                normalized_data_writer.close()
                print("[DONE] normalized_data writer closed.")
            except Exception as e:
                print("[WARN] normalized_data writer close failed:", e)
        if embed_updater is not None:
            try:
                embed_updater.close()
                print("[DONE] Embedding updater closed (final flush done).")
            except Exception as e:
                print("[WARN] Embedding updater close failed:", e)
        print("Done.")



def parse_pipeline_args(pipeline_args: str | None) -> argparse.Namespace:
    s = str(pipeline_args or "").strip()
    argv = shlex.split(s) if s else []
    return parse_args(argv)


def processor_thread_with_stop(
    sid: int,
    camera_db_id: int,
    vs,
    render_store: RenderedFrame,
    yolo,
    args,
    deep_tracker,
    iou_tracker: IOUTracker,
    gallery_mgr: GalleryManager,
    reid_extractor,
    face_app,
    global_owner: Optional[GlobalNameOwner],
    report,
    normalized_report,
    embed_updater: Optional[EmbeddingDBUpdater],
    stop_evt: threading.Event,
    debug: bool = False,
    save_writer: Optional[SegmentedVideoWriter] = None,
):
    frame_idx = 0
    identity_state: dict[int, dict] = {}
    camera_name_owner = CameraNameOwner(args)
    last_t = time.time()
    fps_ema = 0.0
    alpha = 0.10
    device_is_cuda = torch.cuda.is_available() and ("cuda" in str(args.device).lower())
    while not stop_evt.is_set():
        ok, frame, ts_cap = vs.read()
        if not ok or frame is None:
            if bool(getattr(vs, "is_file_source", False)) and hasattr(vs, "is_finished") and vs.is_finished():
                break
            stop_evt.wait(0.005)
            continue
        if int(args.max_queue_age_ms) > 0 and (not bool(getattr(vs, "is_file_source", False))):
            now = time.time()
            age_ms = (now - float(ts_cap)) * 1000.0
            dropped_here = 0
            while age_ms > float(args.max_queue_age_ms) and dropped_here < int(args.max_drain_per_cycle):
                try:
                    vs.read_dropped = int(getattr(vs, "read_dropped", 0)) + 1
                except Exception:
                    pass
                ok2, frame2, ts2 = vs.read()
                if not ok2 or frame2 is None:
                    break
                frame, ts_cap = frame2, ts2
                age_ms = (time.time() - float(ts_cap)) * 1000.0
                dropped_here += 1
        try:
            gallery_mgr.maybe_reload(args)
            people_by_cam, face_gallery, name_to_mid = gallery_mgr.snapshot()
            people = people_by_cam.get(int(camera_db_id), [])
            out, meta = process_one_frame(
                frame_idx, frame, sid, int(camera_db_id), yolo, args, deep_tracker, iou_tracker,
                people, reid_extractor, face_app, face_gallery, name_to_mid, identity_state,
                device_is_cuda=device_is_cuda, global_owner=global_owner, ts_cap=float(ts_cap),
                embed_updater=embed_updater, camera_name_owner=camera_name_owner,
            )
            ts_use = float(ts_cap) if ts_cap else time.time()
            if report is not None:
                _update_report_from_meta(report, args=args, sid=int(sid), camera_db_id=int(camera_db_id), meta=meta, ts_use=ts_use)
            if (normalized_report is not None) and (normalized_report is not report):
                _update_report_from_meta(normalized_report, args=args, sid=int(sid), camera_db_id=int(camera_db_id), meta=meta, ts_use=ts_use)
            now = time.time()
            dt = max(1e-6, now - last_t)
            inst_fps = 1.0 / dt
            fps_ema = (1 - alpha) * fps_ema + alpha * inst_fps
            last_t = now
            if args.overlay_fps:
                qsz = int(vs.qsize()) if hasattr(vs, "qsize") else 0
                dropped_cap = int(getattr(vs, "dropped", 0))
                dropped_read = int(getattr(vs, "read_dropped", 0))
                lag_ms = (now - float(ts_cap)) * 1000.0 if ts_cap else 0.0
                lines = [
                    f"SRC {sid} (cam_id={camera_db_id}) | FPS {fps_ema:.1f} | lag {lag_ms:.0f}ms",
                    f"q {qsz} | drop(cap) {dropped_cap} | drop(stale) {dropped_read}",
                    f"tracks {meta.get('tracks', 0)} | shown {meta.get('shown', 0)} | faces {meta.get('faces_recognized', 0)}",
                ]
                y = 22
                for ln in lines:
                    cv2.putText(out, ln, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y += 22
            frame_meta = dict(meta or {})
            frame_meta.update({"fps": float(fps_ema), "sid": int(sid), "camera_db_id": int(camera_db_id), "frame_idx": int(frame_idx)})
            render_store.set(out, meta=frame_meta)
            if save_writer is not None:
                try:
                    save_writer.write(out)
                except Exception:
                    pass
            frame_idx += 1
        except Exception as e:
            if debug:
                print(f"[PROC {sid}] error:", e)
            stop_evt.wait(0.001)


def _service_recorded_sources_monitor_loop(stop_evt: threading.Event, streams: List[Dict[str, Any]]) -> None:
    if not streams:
        stop_evt.set()
        return
    while not stop_evt.is_set():
        all_finished = True
        for s in streams:
            vs = s.get("vs")
            if vs is None:
                continue
            if not bool(getattr(vs, "is_file_source", False)):
                all_finished = False
                break
            if hasattr(vs, "is_finished") and (not bool(vs.is_finished())):
                all_finished = False
                break
        if all_finished:
            print("[SERVICE] All recorded video sources completed. Stopping runner.")
            stop_evt.set()
            return
        stop_evt.wait(0.2)


def _service_video_writer_loop(stop_evt: threading.Event, streams: List[Dict[str, Any]], buffers_by_cam: Dict[int, RenderedFrame],
                               args: argparse.Namespace, seg_writer: SegmentedVideoWriter) -> None:
    screen_w, screen_h = _get_screen_resolution(default=(1920, 1080))
    last_good: Dict[int, np.ndarray] = {}
    target_fps = float(max(1.0, float(getattr(args, "video_fps", 20.0) or 20.0)))
    sleep_s = 1.0 / target_fps
    cam_order = [int(s.get("camera_db_id", -1)) for s in streams]
    last_t = time.time()
    fps_ema = 0.0
    alpha = 0.10
    while not stop_evt.is_set():
        frames: List[np.ndarray] = []
        any_frame = False
        for cam_id in cam_order:
            buf = buffers_by_cam.get(int(cam_id))
            frm = None
            if buf is not None:
                frm, _ts, _meta = buf.get()
            if frm is not None:
                any_frame = True
                last_good[int(cam_id)] = frm
                frames.append(frm)
            else:
                frames.append(last_good.get(int(cam_id), np.zeros((720, 1280, 3), dtype=np.uint8)))
        if any_frame and frames:
            vis = make_grid_view(
                frames, screen_w=int(screen_w), screen_h=int(screen_h),
                mode=str(getattr(args, "grid_mode", "cover")),
                grid_rows=int(getattr(args, "grid_rows", 0) or 0),
                grid_cols=int(getattr(args, "grid_cols", 0) or 0),
            )
            now = time.time()
            dt = max(1e-6, now - last_t)
            disp_fps = 1.0 / dt
            fps_ema = (1 - alpha) * fps_ema + alpha * disp_fps
            last_t = now
            cv2.putText(vis, f"DISPLAY FPS {fps_ema:.1f} | cams {len(frames)}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            seg_writer.write(vis)
        stop_evt.wait(sleep_s)


class TrackingRunner:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self._stop_evt = threading.Event()
        self._threads: List[threading.Thread] = []
        self._streams: List[Dict[str, Any]] = []
        self._render_by_cam: Dict[int, RenderedFrame] = {}
        self._gallery_mgr: Optional[GalleryManager] = None
        self._global_owner: Optional[GlobalNameOwner] = None
        self._report = None
        self._normalized_report = None
        self._csv_stop_evt: Optional[threading.Event] = None
        self._csv_thread: Optional[threading.Thread] = None
        self._normalized_data_writer: Optional[NormalizedDataDBWriter] = None
        self._embed_updater: Optional[EmbeddingDBUpdater] = None
        self._yolo = None
        self._reid_extractor = None
        self._face_app = None
        self._seg_writer: Optional[SegmentedVideoWriter] = None
        self._video_thread: Optional[threading.Thread] = None
        self._recorded_sources_monitor_thread: Optional[threading.Thread] = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        args = self.args
        if not bool(getattr(args, "use_db", False)):
            raise RuntimeError("TrackingRunner requires --use-db and --db-url.")
        if not str(getattr(args, "db_url", "") or "").strip():
            raise RuntimeError("TrackingRunner requires --db-url.")
        if not getattr(args, "camera_ids", None):
            args.camera_ids = []
        if len(args.camera_ids) == 0:
            args.camera_ids = list(range(1, len(args.src) + 1))
        if len(args.camera_ids) != len(args.src):
            raise RuntimeError("--camera-ids must have the same length as --src")
        args.save_video = not bool(getattr(args, "no_save_video", False))
        args.global_unique_names = (len(args.src) > 1) and (not bool(getattr(args, "no_global_unique_names", False)))
        if getattr(args, "cudnn_benchmark", False):
            torch.backends.cudnn.benchmark = True
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
        except Exception:
            pass
        gpu = torch.cuda.is_available() and ("cuda" in str(args.device).lower())
        if gpu:
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
            except Exception:
                pass
            try:
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        if bool(getattr(args, "half", False)) and (not gpu):
            args.half = False

        self._gallery_mgr = GalleryManager(args)
        self._global_owner = None
        if bool(getattr(args, "global_unique_names", False)):
            self._global_owner = GlobalNameOwner(
                hold_seconds=float(getattr(args, "global_hold_seconds", 0.5)),
                switch_margin=float(getattr(args, "global_switch_margin", 0.02)),
            )
        yolo = None
        if YOLO is not None:
            try:
                weights = args.yolo_weights
                if weights and (not Path(weights).exists()):
                    weights = "yolov8n.pt"
                yolo = YOLO(weights)
                if gpu:
                    try:
                        yolo.to(args.device)
                    except Exception:
                        pass
            except Exception:
                yolo = None
        self._yolo = yolo
        self._reid_extractor = None
        self._face_app = init_face_engine(
            bool(getattr(args, "use_face", False)), args.device, args.face_model,
            int(args.face_det_size[0]), int(args.face_det_size[1]),
            face_provider=getattr(args, "face_provider", "auto"), ort_log=getattr(args, "ort_log", False),
        )
        self._normalized_data_writer = _create_normalized_data_writer(args)
        self._report, self._normalized_report, self._csv_stop_evt, self._csv_thread = _create_tracking_reports(
            args=args,
            num_cams=len(args.src),
            normalized_writer=self._normalized_data_writer,
        )
        self._embed_updater = None
        if bool(getattr(args, "update_db_embeddings", False)):
            if bool(getattr(args, "use_face", False)):
                os.makedirs("csv_output", exist_ok=True)
                ts_tag = datetime.now().strftime("%Y%m%d_%H%M%S")
                log_path = str(getattr(args, "embeddings_log_csv", "") or "").strip() or os.path.join("csv_output", f"embeddings_updates_{ts_tag}.csv")
                samples_log_path = str(getattr(args, "embeddings_samples_log_csv", "") or "").strip() or os.path.join("csv_output", f"embeddings_samples_{ts_tag}.csv")
                self._embed_updater = EmbeddingDBUpdater(
                    db_url=args.db_url,
                    slot_mb=float(getattr(args, "embeddings_slot_mb", 0.5)),
                    flush_seconds=float(getattr(args, "embeddings_flush_seconds", 10.0)),
                    min_sample_seconds=float(getattr(args, "embeddings_min_sample_seconds", 0.5)),
                    min_face_sim=float(getattr(args, "update_face_sim_thresh", 0.75)),
                    min_face_det_score=float(getattr(args, "embed_min_face_det_score", 0.75)),
                    log_csv_path=log_path,
                    update_body=False,
                    update_face=not bool(getattr(args, "no_update_face_bank", False)),
                    reset_if_gap_days_ge=int(getattr(args, "embeddings_reset_if_gap_days", 2)),
                    reset_on_start=False,
                    samples_log_csv_path=samples_log_path,
                )
        tracker_backend = 'iou'
        if bool(getattr(args, 'use_strongsort', False)):
            tracker_backend = 'strongsort'
        elif bool(getattr(args, 'use_bytetrack', False)):
            tracker_backend = 'bytetrack'
        elif bool(getattr(args, 'use_deepsort', False)):
            tracker_backend = 'deepsort'
        elif bool(getattr(args, 'no_deepsort', False)):
            tracker_backend = 'iou'
        else:
            if DeepSort is not None:
                tracker_backend = 'deepsort'
            elif BoxByteTrack is not None:
                tracker_backend = 'bytetrack'
            else:
                tracker_backend = 'iou'
        strongsort_weights: Optional[Path] = None
        if tracker_backend == 'deepsort' and DeepSort is None:
            tracker_backend = 'iou'
        if tracker_backend == 'bytetrack' and BoxByteTrack is None:
            tracker_backend = 'iou'
        if tracker_backend == 'strongsort':
            if BoxStrongSort is None:
                tracker_backend = 'iou'
            else:
                strongsort_weights = resolve_strongsort_reid_weights(args)
                if strongsort_weights is None:
                    tracker_backend = 'iou'
        print(f'[INIT] (service) Tracker backend: {tracker_backend}')
        if tracker_backend == 'strongsort' and strongsort_weights is not None:
            print(f'[INIT] (service) StrongSORT ReID weights: {strongsort_weights}')

        self._streams = []
        self._render_by_cam = {}
        for sid, raw_src in enumerate(args.src):
            src = raw_src.strip() if isinstance(raw_src, str) else raw_src
            camera_db_id = int(args.camera_ids[sid])
            vs = AdaptiveQueueStream(
                src, queue_size=args.queue_size, rtsp_transport=args.rtsp_transport, use_opencv=True,
                freeze_seconds=float(args.stream_freeze_seconds), open_timeout_ms=int(args.stream_open_timeout_ms),
                read_timeout_ms=int(args.stream_read_timeout_ms), reconnect_base_delay=float(args.stream_reconnect_base_seconds),
                reconnect_max_delay=float(args.stream_reconnect_max_seconds), reconnect_jitter=float(args.stream_reconnect_jitter),
                reconnect_log_interval=float(args.stream_reconnect_log_interval),
            )
            deep_tracker = None
            if tracker_backend == 'deepsort':
                try:
                    deep_tracker = DeepSort(
                        max_age=int(args.max_age), n_init=int(args.n_init), nn_budget=int(args.nn_budget),
                        max_cosine_distance=float(args.tracker_max_cosine), nms_max_overlap=float(args.tracker_nms_overlap),
                        embedder="torchreid", embedder_gpu=gpu, half=(gpu and bool(args.half)), bgr=True,
                    )
                except Exception:
                    deep_tracker = None
            elif tracker_backend == 'bytetrack':
                try:
                    deep_tracker = BoxByteTrack(
                        det_thresh=float(args.conf), max_age=int(args.max_age), max_obs=max(50, int(args.max_age) + 5),
                        min_hits=int(args.n_init), iou_threshold=float(getattr(args, 'max_iou_distance', 0.30)),
                        min_conf=float(getattr(args, 'bytetrack_min_conf', 0.10)), track_thresh=float(getattr(args, 'bytetrack_track_thresh', 0.45)),
                        match_thresh=float(getattr(args, 'bytetrack_match_thresh', 0.80)), track_buffer=int(getattr(args, 'bytetrack_track_buffer', 25)),
                        frame_rate=int(getattr(args, 'bytetrack_frame_rate', 30)),
                    )
                except Exception:
                    deep_tracker = None
            elif tracker_backend == 'strongsort':
                try:
                    dev = torch.device(args.device) if gpu else torch.device('cpu')
                    deep_tracker = BoxStrongSort(
                        reid_weights=strongsort_weights, device=dev, half=(gpu and bool(args.half)),
                        det_thresh=float(args.conf), max_age=int(args.max_age), max_obs=max(50, int(args.max_age) + 5),
                        min_hits=int(args.n_init), iou_threshold=float(getattr(args, 'max_iou_distance', 0.30)),
                        min_conf=float(args.conf), max_cos_dist=float(args.tracker_max_cosine), n_init=int(args.n_init), nn_budget=int(args.nn_budget),
                    )
                except Exception:
                    deep_tracker = None
            iou_tracker = IOUTracker(max_miss=max(1, int(args.iou_max_miss)), iou_thresh=float(getattr(args, 'max_iou_distance', 0.30)))
            buf = RenderedFrame()
            self._render_by_cam[int(camera_db_id)] = buf
            self._streams.append({"sid": sid, "camera_db_id": camera_db_id, "src": raw_src, "vs": vs, "deep": deep_tracker, "iou": iou_tracker, "buf": buf})
        opened_now = any(s["vs"].is_opened() for s in self._streams)
        all_file_sources = bool(self._streams) and all(bool(getattr(s["vs"], "is_file_source", False)) for s in self._streams)
        if (not opened_now) and all_file_sources:
            for s in self._streams:
                try:
                    s["vs"].release()
                except Exception:
                    pass
            try:
                if self._csv_stop_evt is not None:
                    self._csv_stop_evt.set()
                if self._csv_thread is not None:
                    self._csv_thread.join(timeout=2.0)
            except Exception:
                pass
            _finalize_tracking_reports(self._report, self._normalized_report, self.args)
            if self._normalized_data_writer is not None:
                try:
                    self._normalized_data_writer.close()
                except Exception:
                    pass
            if self._embed_updater is not None:
                try:
                    self._embed_updater.close()
                except Exception:
                    pass
            raise RuntimeError("No sources opened. Check --src URLs and codecs.")
        if not opened_now:
            print("[INIT] No sources opened yet. Keeping the runner alive and waiting for RTSP playback to become available...")
        for s in self._streams:
            t = threading.Thread(
                target=processor_thread_with_stop,
                args=(int(s["sid"]), int(s["camera_db_id"]), s["vs"], s["buf"], self._yolo, args, s["deep"], s["iou"],
                      self._gallery_mgr, self._reid_extractor, self._face_app, self._global_owner, self._report,
                      self._normalized_report, self._embed_updater, self._stop_evt, False),
                daemon=True,
            )
            t.start()
            self._threads.append(t)
        if self._streams and all(bool(getattr(s.get("vs"), "is_file_source", False)) for s in self._streams):
            self._recorded_sources_monitor_thread = threading.Thread(
                target=_service_recorded_sources_monitor_loop, args=(self._stop_evt, list(self._streams)), daemon=True
            )
            self._recorded_sources_monitor_thread.start()
        if bool(getattr(args, "save_video", True)):
            out_dir = str(getattr(args, "video_dir", "saved_videos") or "saved_videos")
            os.makedirs(out_dir, exist_ok=True)
            self._seg_writer = SegmentedVideoWriter(
                out_dir=out_dir, basename=str(getattr(args, "video_prefix", "saved_video") or "saved_video"),
                ext=_norm_ext(getattr(args, "video_ext", ".mp4")), fps=float(getattr(args, "video_fps", 20.0) or 20.0),
                fourcc=str(getattr(args, "video_fourcc", "mp4v") or "mp4v"),
                segment_seconds=int(float(getattr(args, "video_segment_seconds", 3600.0) or 0.0)),
                save_height=int(getattr(args, "video_save_height", 480) or 0),
            )
            print(f"[INIT] (service) Saving annotated GRID videos to folder: {self._seg_writer.run_dir}")
            if self._seg_writer.segment_seconds > 0:
                print(f"[INIT] (service) Video segmentation: ON ({self._seg_writer.segment_seconds:.0f}s per file)")
            else:
                print("[INIT] (service) Video segmentation: OFF (single file)")
            self._video_thread = threading.Thread(
                target=_service_video_writer_loop,
                args=(self._stop_evt, list(self._streams), dict(self._render_by_cam), args, self._seg_writer),
                daemon=True,
            )
            self._video_thread.start()
        self._started = True

    def stop(self) -> None:
        if not self._started:
            return
        self._stop_evt.set()
        for s in self._streams:
            try:
                s["vs"].release()
            except Exception:
                pass
        try:
            if self._csv_stop_evt is not None:
                self._csv_stop_evt.set()
            if self._csv_thread is not None:
                self._csv_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self._video_thread is not None:
                self._video_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self._recorded_sources_monitor_thread is not None:
                self._recorded_sources_monitor_thread.join(timeout=2.0)
        except Exception:
            pass
        try:
            if self._seg_writer is not None:
                self._seg_writer.close()
        except Exception:
            pass
        _finalize_tracking_reports(self._report, self._normalized_report, self.args)
        if self._normalized_data_writer is not None:
            try:
                self._normalized_data_writer.close()
            except Exception:
                pass
        if self._embed_updater is not None:
            try:
                self._embed_updater.close()
            except Exception:
                pass
        for t in self._threads:
            try:
                t.join(timeout=0.2)
            except Exception:
                pass
        self._started = False

    def get_camera_buffer(self, cam_id: int) -> Optional[RenderedFrame]:
        return self._render_by_cam.get(int(cam_id))

    def list_db_cameras(self, active_only: bool = True) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for s in self._streams:
            cam_id = int(s.get("camera_db_id", -1))
            out.append({
                "id": cam_id,
                "camera_id": cam_id,
                "src": str(s.get("src", "")),
                "running": bool(getattr(s.get("vs", None), "is_opened", lambda: False)()),
            })
        out.sort(key=lambda x: int(x.get("id", 0)))
        return out

    def status(self) -> Dict[str, Any]:
        cams = sorted(list(self._render_by_cam.keys()))
        return {
            "running": bool(self._started),
            "camera_ids": cams,
            "num_cameras": len(cams),
            "save_csv": bool(getattr(self.args, "save_csv", False)),
            "csv_path": str(getattr(self.args, "csv", "") or ""),
            "write_normalized_data": bool(self._normalized_data_writer is not None),
            "update_db_embeddings": bool(getattr(self.args, "update_db_embeddings", False)),
            "save_video": bool(getattr(self.args, "save_video", True)),
            "video_dir": str(getattr(self._seg_writer, "run_dir", "") or ""),
        }

    def write_report_snapshot(self, path: str | None = None) -> str:
        if self._report is None:
            return ""
        out_path = str(path or getattr(self.args, "csv", "") or "").strip()
        if not out_path:
            out_path = "detections_summary_snapshot.csv"
        try:
            self._report.write_csv_live(out_path)
        except Exception:
            pass
        return out_path


# === Added in stable recovery build: known-person raw-ID reattach + tracker-id UI ===

@dataclass
class _DrawItemStable:
    tid: int
    raw_tid: int
    bbox: Tuple[int, int, int, int]
    name: str
    member_id: int
    face_sim: float
    stable_score: float
    det_conf: Optional[float]
    face_hit: bool
    low_face: bool = False
    low_face_sim: float = 0.0


def _priority_tuple_stable(it: _DrawItemStable) -> tuple:
    return (
        1 if it.face_hit else 0,
        float(it.face_sim),
        float(it.stable_score),
        float(it.det_conf or 0.0),
        float(_box_area_xyxy(it.bbox)),
    )


def _block_duplicate_names_stable(items: List[_DrawItemStable]) -> Tuple[List[_DrawItemStable], set[int]]:
    if not items:
        return [], set()
    groups: Dict[str, List[_DrawItemStable]] = defaultdict(list)
    winner_tid_by_name: Dict[str, int] = {}
    for it in items:
        if not it.name:
            continue
        groups[str(it.name)].append(it)
    for name, group in groups.items():
        best = max(group, key=_priority_tuple_stable)
        winner_tid_by_name[str(name)] = int(best.tid)
    out: List[_DrawItemStable] = []
    demoted_tids: set[int] = set()
    for it in items:
        if not it.name:
            out.append(it)
            continue
        if int(winner_tid_by_name.get(str(it.name), int(it.tid))) == int(it.tid):
            out.append(it)
            continue
        demoted_tids.add(int(it.tid))
        out.append(_DrawItemStable(
            tid=int(it.tid),
            raw_tid=int(it.raw_tid),
            bbox=it.bbox,
            name="",
            member_id=-1,
            face_sim=float(it.face_sim),
            stable_score=float(it.stable_score),
            det_conf=it.det_conf,
            face_hit=bool(it.face_hit),
            low_face=bool(it.low_face),
            low_face_sim=float(it.low_face_sim),
        ))
    return out, demoted_tids


@dataclass
class _KnownTrackSnapshot:
    logical_tid: int
    raw_tid: int
    bbox: Tuple[int, int, int, int]
    name: str
    member_id: int
    has_approved_face: bool


@dataclass
class _LostKnownTrack:
    logical_tid: int
    raw_tid: int
    bbox: Tuple[int, int, int, int]
    center: Tuple[float, float]
    name: str
    member_id: int
    lost_frame: int


class KnownTrackReattachManager:
    def __init__(self, args, sid: int, camera_db_id: int):
        self.enabled = bool(getattr(args, "known_id_reattach", True))
        self.hold_frames = int(max(1, int(getattr(args, "known_id_reattach_frames", 45) or 45)))
        self.center_px = float(max(1.0, float(getattr(args, "known_id_reattach_center_px", 50.0) or 50.0)))
        self.size_ratio_min = float(max(0.10, min(1.0, float(getattr(args, "known_id_reattach_size_ratio", 0.60) or 0.60))))
        self.ambiguity_margin_px = float(max(0.0, float(getattr(args, "known_id_reattach_ambiguity_px", 12.0) or 12.0)))
        self.sid = int(sid)
        self.camera_db_id = int(camera_db_id)
        self._active_by_raw: Dict[int, _KnownTrackSnapshot] = {}
        self._lost: Dict[int, _LostKnownTrack] = {}

    @staticmethod
    def _bbox_area(bbox: Tuple[int, int, int, int]) -> float:
        return float(max(0, int(bbox[2]) - int(bbox[0])) * max(0, int(bbox[3]) - int(bbox[1])))

    @staticmethod
    def _center(bbox: Tuple[int, int, int, int]) -> Tuple[float, float]:
        return _box_center_xyxy(bbox)

    def _is_named_snapshot(self, snap: Optional[_KnownTrackSnapshot]) -> bool:
        if snap is None:
            return False
        return bool(str(snap.name or "").strip()) and bool(snap.has_approved_face) and int(snap.member_id) > 0

    def _cleanup(self, frame_idx: int) -> None:
        dead = []
        for logical_tid, lost in self._lost.items():
            if (int(frame_idx) - int(lost.lost_frame)) > int(self.hold_frames):
                dead.append(int(logical_tid))
        for logical_tid in dead:
            self._lost.pop(int(logical_tid), None)

    def begin_frame(self, raw_tracks: List[Dict[str, Any]], frame_idx: int) -> Dict[int, int]:
        if (not self.enabled) or (not raw_tracks):
            return {int(tr.get("tid", -1)): int(tr.get("tid", -1)) for tr in raw_tracks if int(tr.get("tid", -1)) >= 0}
        frame_idx = int(frame_idx)
        self._cleanup(frame_idx)
        curr_bbox_by_raw: Dict[int, Tuple[int, int, int, int]] = {}
        for tr in raw_tracks:
            try:
                raw_tid = int(tr.get("tid", -1))
                bbox = tuple(int(v) for v in tr.get("bbox", (0, 0, 0, 0))[:4])
            except Exception:
                continue
            if raw_tid < 0:
                continue
            curr_bbox_by_raw[raw_tid] = bbox
        current_raws = set(curr_bbox_by_raw.keys())
        raw_to_logical: Dict[int, int] = {}

        # keep current raw->logical mapping when the raw tracker id itself continues
        for raw_tid, snap in list(self._active_by_raw.items()):
            if int(raw_tid) in current_raws:
                raw_to_logical[int(raw_tid)] = int(snap.logical_tid)
                self._lost.pop(int(snap.logical_tid), None)
                continue
            if self._is_named_snapshot(snap):
                self._lost[int(snap.logical_tid)] = _LostKnownTrack(
                    logical_tid=int(snap.logical_tid),
                    raw_tid=int(snap.raw_tid),
                    bbox=tuple(int(v) for v in snap.bbox),
                    center=self._center(snap.bbox),
                    name=str(snap.name),
                    member_id=int(snap.member_id),
                    lost_frame=int(frame_idx - 1),
                )
        self._cleanup(frame_idx)

        unmatched_raws = [int(tid) for tid in curr_bbox_by_raw.keys() if int(tid) not in raw_to_logical]
        if not unmatched_raws or (not self._lost):
            return raw_to_logical

        candidates: List[Tuple[float, int, int, float, float, float]] = []
        by_raw: Dict[int, List[Tuple[float, int, int, float, float, float]]] = defaultdict(list)
        by_logical: Dict[int, List[Tuple[float, int, int, float, float, float]]] = defaultdict(list)
        for raw_tid in unmatched_raws:
            bbox = curr_bbox_by_raw.get(int(raw_tid))
            if bbox is None:
                continue
            cx, cy = self._center(bbox)
            area = max(1.0, self._bbox_area(bbox))
            for logical_tid, lost in list(self._lost.items()):
                dx = abs(float(cx) - float(lost.center[0]))
                dy = abs(float(cy) - float(lost.center[1]))
                if dx > self.center_px or dy > self.center_px:
                    continue
                lost_area = max(1.0, self._bbox_area(lost.bbox))
                ratio = float(min(area, lost_area) / max(area, lost_area))
                if ratio < self.size_ratio_min:
                    continue
                score = float(dx + dy - (10.0 * ratio))
                item = (score, int(raw_tid), int(logical_tid), float(dx), float(dy), float(ratio))
                candidates.append(item)
                by_raw[int(raw_tid)].append(item)
                by_logical[int(logical_tid)].append(item)
        if not candidates:
            return raw_to_logical

        ambiguous_raws: set[int] = set()
        for raw_tid, vals in by_raw.items():
            vals = sorted(vals, key=lambda x: x[0])
            if len(vals) >= 2 and abs(float(vals[1][0]) - float(vals[0][0])) <= self.ambiguity_margin_px:
                ambiguous_raws.add(int(raw_tid))
        ambiguous_logs: set[int] = set()
        for logical_tid, vals in by_logical.items():
            vals = sorted(vals, key=lambda x: x[0])
            if len(vals) >= 2 and abs(float(vals[1][0]) - float(vals[0][0])) <= self.ambiguity_margin_px:
                ambiguous_logs.add(int(logical_tid))

        used_raws = set(raw_to_logical.keys())
        used_logs = set(int(v) for v in raw_to_logical.values())
        for score, raw_tid, logical_tid, dx, dy, ratio in sorted(candidates, key=lambda x: x[0]):
            if int(raw_tid) in used_raws or int(logical_tid) in used_logs:
                continue
            if int(raw_tid) in ambiguous_raws or int(logical_tid) in ambiguous_logs:
                continue
            lost = self._lost.pop(int(logical_tid), None)
            if lost is None:
                continue
            raw_to_logical[int(raw_tid)] = int(logical_tid)
            used_raws.add(int(raw_tid))
            used_logs.add(int(logical_tid))
            try:
                print(
                    f"[ID-REATTACH][src={self.sid} cam={self.camera_db_id}] "
                    f"{str(lost.name)}: raw T{int(lost.raw_tid)} -> T{int(raw_tid)} "
                    f"(dx={float(dx):.1f}, dy={float(dy):.1f}, size={float(ratio):.2f})"
                )
            except Exception:
                pass
        return raw_to_logical

    def end_frame(self, track_snapshots: List[Dict[str, Any]], frame_idx: int) -> None:
        frame_idx = int(frame_idx)
        new_active: Dict[int, _KnownTrackSnapshot] = {}
        for item in track_snapshots or []:
            try:
                raw_tid = int(item.get("raw_tid", -1))
                logical_tid = int(item.get("logical_tid", raw_tid))
                bbox = tuple(int(v) for v in item.get("bbox", (0, 0, 0, 0))[:4])
                name = str(item.get("name", "") or "")
                member_id = int(item.get("member_id", -1))
                has_approved_face = bool(item.get("has_approved_face", False))
            except Exception:
                continue
            if raw_tid < 0:
                continue
            new_active[raw_tid] = _KnownTrackSnapshot(
                logical_tid=int(logical_tid),
                raw_tid=int(raw_tid),
                bbox=tuple(int(v) for v in bbox),
                name=str(name),
                member_id=int(member_id),
                has_approved_face=bool(has_approved_face),
            )
            self._lost.pop(int(logical_tid), None)
        self._active_by_raw = new_active
        self._cleanup(frame_idx)


_known_reattach_mgr_store: Dict[Tuple[int, int, int], KnownTrackReattachManager] = {}


def _get_known_reattach_mgr(args, sid: int, camera_db_id: int, identity_state: dict) -> KnownTrackReattachManager:
    key = (int(sid), int(camera_db_id), int(id(identity_state)))
    mgr = _known_reattach_mgr_store.get(key)
    if mgr is None:
        mgr = KnownTrackReattachManager(args=args, sid=int(sid), camera_db_id=int(camera_db_id))
        _known_reattach_mgr_store[key] = mgr
    return mgr


def process_one_frame(
    frame_idx: int,
    frame_bgr: np.ndarray,
    sid: int,
    camera_db_id: int,
    yolo,
    args,
    deep_tracker,
    iou_tracker: IOUTracker,
    people: list[PersonEntry],
    reid_extractor,
    face_app,
    face_gallery: FaceGallery,
    name_to_member_id: dict[str, int],
    identity_state: dict,
    device_is_cuda: bool,
    global_owner: Optional[GlobalNameOwner] = None,
    ts_cap: float = 0.0,
    embed_updater: Optional[EmbeddingDBUpdater] = None,
    camera_name_owner: Optional[CameraNameOwner] = None,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    rw, rh = int(args.resize[0]), int(args.resize[1])
    if rw > 0 and rh > 0:
        frame_bgr = cv2.resize(frame_bgr, (rw, rh), interpolation=cv2.INTER_LINEAR)
    H, W = frame_bgr.shape[:2]

    tlwh_conf: list[list[float]] = []
    if yolo is not None:
        try:
            res = _yolo_forward_safe(yolo, frame_bgr, args)
            boxes = res[0].boxes if (res and len(res)) else None
            if boxes is not None:
                xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
                cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
                keep = cls == 0
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
                    tlwh_conf.append([x1f, y1f, ww, hh, float(c)])
        except Exception as e:
            print(f"[SRC {sid}] YOLO error:", e)

    dets_np = np.asarray(tlwh_conf, dtype=np.float32)
    if dets_np.ndim != 2:
        dets_np = dets_np.reshape((0, 5)).astype(np.float32)
    dets_dsrt = [([float(x), float(y), float(w), float(h)], float(cf), 0) for x, y, w, h, cf in tlwh_conf]

    out_tracks: List[Any] = []
    if deep_tracker is not None:
        if hasattr(deep_tracker, 'update_tracks'):
            try:
                out_tracks = deep_tracker.update_tracks(dets_dsrt, frame=frame_bgr)
            except Exception as e:
                print(f"[SRC {sid}] DeepSORT update_tracks error:", e)
                out_tracks = iou_tracker.update(dets_np)
        elif hasattr(deep_tracker, 'update'):
            try:
                if len(tlwh_conf) > 0:
                    dets_boxmot = np.asarray([[x, y, x + w, y + h, cf, 0] for x, y, w, h, cf in tlwh_conf], dtype=np.float32)
                else:
                    dets_boxmot = np.zeros((0, 6), dtype=np.float32)
                res_mot = deep_tracker.update(dets_boxmot, frame_bgr)
                out_tracks = boxmot_results_to_tracks(res_mot)
            except Exception as e:
                print(f"[SRC {sid}] BoxMOT tracker update error:", e)
                out_tracks = iou_tracker.update(dets_np)
        else:
            out_tracks = iou_tracker.update(dets_np)
    else:
        out_tracks = iou_tracker.update(dets_np)

    recognized_faces: List[Dict[str, Any]] = []
    low_faces: List[Dict[str, Any]] = []
    do_face = face_app is not None and face_gallery is not None and (not face_gallery.is_empty()) and (frame_idx % max(1, int(args.face_every_n)) == 0)
    if do_face:
        try:
            det_min = float(getattr(args, "embed_min_face_det_score", 0.75))
            with _face_lock:
                faces = face_app.get(np.ascontiguousarray(frame_bgr))
            for f in safe_iter_faces(faces):
                bbox = getattr(f, "bbox", None)
                if bbox is None:
                    continue
                b = np.asarray(bbox).reshape(-1)
                if b.size < 4:
                    continue
                fx1, fy1, fx2, fy2 = map(float, b[:4])
                fw = max(0.0, fx2 - fx1)
                fh = max(0.0, fy2 - fy1)
                if fw < float(args.min_face_px) or fh < float(args.min_face_px):
                    continue
                det_score = float(extract_face_det_score(f))
                if det_score < det_min:
                    continue
                emb = extract_face_embedding(f)
                if emb is None:
                    continue
                emb = l2_normalize(np.asarray(emb, dtype=np.float32))
                best_mid, flabel, fsim, fsecond = best_face_top2(emb, face_gallery)
                if flabel is None or best_mid is None:
                    continue
                gap = float(fsim - fsecond)
                if (fsim >= float(args.face_thresh)) and (gap >= float(args.face_gap)):
                    recognized_faces.append({
                        "bbox": (fx1, fy1, fx2, fy2),
                        "label": str(flabel),
                        "member_id": int(best_mid),
                        "sim": float(fsim),
                        "second": float(fsecond),
                        "gap": float(gap),
                        "det_score": float(det_score),
                        "emb": emb.astype(np.float32),
                    })
                else:
                    if float(fsim) < float(args.face_thresh):
                        low_faces.append({
                            "bbox": (fx1, fy1, fx2, fy2),
                            "sim": float(fsim),
                            "det_score": float(det_score),
                        })
        except Exception as e:
            print(f"[SRC {sid}] FaceAnalysis error:", e)

    out = frame_bgr.copy()
    tracks_info: List[Dict[str, Any]] = []

    for tid in list(identity_state.keys()):
        if not isinstance(tid, int):
            continue
        st = identity_state.get(tid, {})
        if isinstance(st, dict) and "last_seen_frame" not in st:
            st["last_seen_frame"] = -1

    raw_tracks: List[Dict[str, Any]] = []
    for t in out_tracks:
        time_since_update = getattr(t, "time_since_update", 0)
        had_match_this_frame = (time_since_update == 0) or (getattr(t, "last_detection", None) is not None)
        if args.draw_only_matched and not had_match_this_frame:
            continue
        try:
            if hasattr(t, "is_confirmed") and callable(getattr(t, "is_confirmed")) and (not t.is_confirmed()):
                continue
            if hasattr(t, "to_tlbr"):
                ltrb = t.to_tlbr()
            elif hasattr(t, "to_ltrb"):
                ltrb = t.to_ltrb()
            else:
                ltrb = t.to_tlbr()
            x1, y1, x2, y2 = map(int, ltrb)
            x1 = int(max(0, min(W - 1, x1)))
            y1 = int(max(0, min(H - 1, y1)))
            x2 = int(max(0, min(W, x2)))
            y2 = int(max(0, min(H, y2)))
            if x2 <= x1 or y2 <= y1:
                continue
            tid = int(getattr(t, "track_id", getattr(t, "track_id_", -1)))
            if tid < 0:
                continue
        except Exception:
            continue
        det_conf = None
        try:
            if hasattr(t, "det_conf") and t.det_conf is not None:
                det_conf = float(t.det_conf)
            elif hasattr(t, "last_detection") and t.last_detection is not None:
                ld = t.last_detection
                if isinstance(ld, (list, tuple)) and len(ld) >= 2:
                    det_conf = float(ld[1])
                elif isinstance(ld, dict):
                    det_conf = float(ld.get("confidence", ld.get("det_conf", 0.0)))
        except Exception:
            det_conf = None
        if args.min_det_conf > 0 and det_conf is not None and det_conf < args.min_det_conf:
            if args.draw_only_matched:
                continue
        raw_tracks.append({"tid": tid, "bbox": (x1, y1, x2, y2), "det_conf": det_conf})

    known_reattach_mgr = _get_known_reattach_mgr(args=args, sid=int(sid), camera_db_id=int(camera_db_id), identity_state=identity_state)
    raw_to_logical: Dict[int, int] = known_reattach_mgr.begin_frame(raw_tracks=raw_tracks, frame_idx=int(frame_idx)) if known_reattach_mgr is not None else {int(tr.get("tid", -1)): int(tr.get("tid", -1)) for tr in raw_tracks}

    face_for_tid: Dict[int, Dict[str, Any]] = assign_faces_to_tracks_one_to_one(recognized_faces=recognized_faces, raw_tracks=raw_tracks, args=args)
    low_face_for_tid: Dict[int, Dict[str, Any]] = {}
    if low_faces and raw_tracks:
        raw_unassigned = [tr for tr in raw_tracks if int(tr.get("tid", -1)) not in face_for_tid]
        if raw_unassigned:
            low_face_for_tid = assign_faces_to_tracks_one_to_one(recognized_faces=low_faces, raw_tracks=raw_unassigned, args=args)

    for tr in raw_tracks:
        raw_tid = int(tr["tid"])
        logical_tid = int(raw_to_logical.get(raw_tid, raw_tid))
        x1, y1, x2, y2 = tr["bbox"]
        det_conf = tr.get("det_conf", None)
        face_label, face_sim, face_gap = "", 0.0, 0.0
        face_det_score = 1.0
        face_member_id = -1
        face_emb = None
        face_hit = False
        low_face_hit = False
        low_face_sim = 0.0
        low_face_det_score = 1.0
        fm = face_for_tid.get(raw_tid)
        if fm is not None:
            face_hit = True
            face_label = str(fm.get("label", ""))
            face_sim = float(fm.get("sim", 0.0))
            face_gap = float(fm.get("gap", 0.0))
            face_member_id = int(fm.get("member_id", -1))
            face_emb = fm.get("emb", None)
            try:
                face_det_score = float(fm.get("det_score", 1.0))
            except Exception:
                face_det_score = 1.0
        lfm = low_face_for_tid.get(raw_tid) if isinstance(low_face_for_tid, dict) else None
        if lfm is not None:
            low_face_hit = True
            try:
                low_face_sim = float(lfm.get("sim", 0.0))
            except Exception:
                low_face_sim = 0.0
            try:
                low_face_det_score = float(lfm.get("det_score", 1.0))
            except Exception:
                low_face_det_score = 1.0

        entry = identity_state.setdefault(logical_tid, make_identity_entry())
        entry["last_seen_frame"] = int(frame_idx)

        approved_face_label = ""
        approved_face_sim = 0.0
        approved_face_member_id = -1
        approved_face_emb = None
        approved_face_det_score = float(face_det_score)
        approved_face_gap = float(face_gap)
        force_owner_switch = False
        if face_hit and face_label:
            cand_label, cand_mid, cand_sim, cand_ok, force_owner_switch = approve_face_candidate_for_track(
                entry, face_label, int(face_member_id), float(face_sim), float(face_gap), float(face_det_score), args
            )
            if cand_ok and cand_label:
                owner_ok = True
                if camera_name_owner is not None:
                    owner_ok = camera_name_owner.allow(
                        str(cand_label), tid=int(logical_tid), score=float(cand_sim), frame_idx=int(frame_idx), force=bool(force_owner_switch)
                    )
                if owner_ok:
                    approved_face_label = str(cand_label)
                    approved_face_sim = float(cand_sim)
                    approved_face_member_id = int(cand_mid)
                    approved_face_emb = face_emb
                else:
                    face_hit = False
                    face_label = ""
                    face_sim = 0.0
                    face_gap = 0.0
                    face_member_id = -1
                    face_emb = None
        face_hit = bool(approved_face_label)
        face_label = str(approved_face_label)
        face_sim = float(approved_face_sim)
        face_gap = float(approved_face_gap if face_hit else 0.0)
        face_member_id = int(approved_face_member_id if face_hit else -1)
        face_emb = approved_face_emb if face_hit else None
        face_det_score = float(approved_face_det_score if face_hit else 1.0)
        entry["face_vis_ttl"] = max(0, int(entry.get("face_vis_ttl", 0)) - 1)
        if face_hit and face_label:
            entry["face_vis_ttl"] = max(1, int(args.face_hold_frames))
            entry["last_face_label"] = face_label
            entry["last_face_sim"] = float(face_sim)
        tracks_info.append({
            "tid": int(logical_tid),
            "raw_tid": int(raw_tid),
            "bbox": (x1, y1, x2, y2),
            "det_conf": det_conf,
            "face_hit": face_hit,
            "face_label": face_label,
            "face_sim": face_sim,
            "face_gap": face_gap,
            "face_det_score": float(face_det_score),
            "face_member_id": int(face_member_id),
            "face_emb": face_emb,
            "face_vis_ttl": int(entry.get("face_vis_ttl", 0)),
            "last_face_sim": float(entry.get("last_face_sim", 0.0)),
            "low_face_hit": bool(low_face_hit),
            "low_face_sim": float(low_face_sim),
            "low_face_det_score": float(low_face_det_score),
            "was_reattached": bool(int(logical_tid) != int(raw_tid)),
        })

    face_winners: Dict[str, Tuple[int, float, int, int]] = {}
    for r in tracks_info:
        if r["face_hit"] and r["face_label"]:
            lab = str(r["face_label"])
            sim = float(r["face_sim"])
            logical_tid = int(r["tid"])
            raw_tid = int(r.get("raw_tid", logical_tid))
            mid = int(r.get("face_member_id", -1))
            prev = face_winners.get(lab)
            if prev is None or sim > prev[1]:
                face_winners[lab] = (logical_tid, sim, mid, raw_tid)
    for lab, (winner_logical_tid, sim, mid, _winner_raw_tid) in face_winners.items():
        w_ent = identity_state.get(int(winner_logical_tid))
        if isinstance(w_ent, dict):
            w_ent["assigned_name"] = str(lab)
            if mid > 0:
                w_ent["assigned_member_id"] = int(mid)
            w_ent["assigned_score"] = max(float(w_ent.get("assigned_score", 0.0)), float(sim))
            w_ent["confirmed_face_label"] = str(lab)
            if mid > 0:
                w_ent["confirmed_face_member_id"] = int(mid)
            w_ent["confirmed_face_sim"] = max(float(w_ent.get("confirmed_face_sim", 0.0)), float(sim))
            w_ent["has_approved_face"] = True

    if embed_updater is not None and face_winners:
        ts_use = float(ts_cap) if float(ts_cap or 0.0) > 0 else time.time()
        sim_thresh = float(getattr(args, "update_face_sim_thresh", 0.75))
        det_thresh = float(getattr(args, "embed_min_face_det_score", 0.75))
        for lab, (winner_logical_tid, sim, member_id, winner_raw_tid) in face_winners.items():
            sim_f = float(sim or 0.0)
            if sim_f < sim_thresh:
                continue
            if int(member_id) <= 0:
                continue
            fm = face_for_tid.get(int(winner_raw_tid), None)
            if not isinstance(fm, dict):
                continue
            try:
                det_score = float(fm.get("det_score", 1.0))
            except Exception:
                det_score = 1.0
            if det_score < det_thresh:
                continue
            try:
                if hasattr(embed_updater, "can_accept") and (not embed_updater.can_accept(int(member_id), int(camera_db_id))):
                    continue
            except Exception:
                pass
            face_emb = fm.get("emb", None)
            body_emb = None
            embed_updater.enqueue(EmbeddingSample(
                member_id=int(member_id), name=str(lab), camera_id=int(camera_db_id), track_id=int(winner_logical_tid),
                ts=float(ts_use), face_sim=float(sim_f), face_det_score=float(det_score), body_emb=body_emb, face_emb=face_emb,
            ))

    draw_candidates: List[_DrawItemStable] = []
    for r in tracks_info:
        logical_tid = int(r["tid"])
        raw_tid = int(r.get("raw_tid", logical_tid))
        x1, y1, x2, y2 = r["bbox"]
        face_hit = bool(r["face_hit"])
        face_label = str(r["face_label"])
        face_sim = float(r["face_sim"])
        last_face_sim = float(r.get("last_face_sim", 0.0))
        face_mid = int(r.get("face_member_id", -1))
        body_label = ""
        body_sim = 0.0
        stable_name, stable_score, entry = update_track_identity(
            identity_state,
            logical_tid,
            face_label=face_label if face_hit else "",
            face_sim=float(face_sim if face_hit else 0.0),
            body_label=body_label,
            body_sim=body_sim,
            decay=args.name_decay,
            min_score=args.name_min_score,
            margin=args.name_margin,
            ttl_reset=args.name_ttl,
            w_face=args.name_face_weight,
            w_body=args.name_body_weight,
            lock_frames=int(getattr(args, "identity_lock_frames", 30)),
            lock_face_thresh=float(getattr(args, "identity_lock_face_thresh", 0.50)),
        )
        if stable_name:
            entry["assigned_name"] = str(stable_name)
            if face_hit and face_mid > 0:
                entry["assigned_member_id"] = int(face_mid)
                if bool(entry.get("locked", False)):
                    entry["lock_label"] = str(stable_name)
            else:
                entry["assigned_member_id"] = int(name_to_member_id.get(str(stable_name), entry.get("assigned_member_id", -1)))
            entry["assigned_score"] = float(stable_score)

        curr_bbox = (int(x1), int(y1), int(x2), int(y2))
        same_person = same_person_continuation(
            entry.get("last_good_bbox", None),
            curr_bbox,
            min_iou=float(getattr(args, "name_continuity_iou", 0.05)),
            max_center_shift=float(getattr(args, "name_continuity_center_shift", 0.60)),
        )
        was_reattached = bool(r.get("was_reattached", False))
        if was_reattached and bool(entry.get("has_approved_face", False)):
            same_person = True
            entry["last_good_bbox"] = curr_bbox
            entry["continuity_break_frames"] = 0
        elif face_hit and face_label:
            same_person = True
            entry["last_good_bbox"] = curr_bbox
            entry["continuity_break_frames"] = 0
        elif same_person:
            entry["continuity_break_frames"] = 0
        else:
            entry["continuity_break_frames"] = int(entry.get("continuity_break_frames", 0)) + 1

        persist_names = not bool(getattr(args, "no_persist_names", False))
        track_has_approved_face = bool(entry.get("has_approved_face", False)) and bool(entry.get("confirmed_face_label", ""))
        if persist_names:
            keep_cached_name = track_has_approved_face and same_person and (
                bool(entry.get("assigned_name", "")) or bool(stable_name) or int(entry.get("lock_ttl", 0)) > 0 or
                int(entry.get("ttl", 0)) > 0 or int(entry.get("face_vis_ttl", 0)) > 0
            )
            final_name = str(entry.get("assigned_name", "") or stable_name or "") if keep_cached_name else ""
        else:
            final_name = str(stable_name or "") if track_has_approved_face else ""
            if (not face_hit) and (not same_person):
                final_name = ""
        if final_name and str(entry.get("confirmed_face_label", "") or "") and str(final_name) != str(entry.get("confirmed_face_label", "")):
            final_name = ""
            demote_identity_entry(entry, clear_face=True)
        final_mid = int(entry.get("assigned_member_id", -1)) if final_name else -1
        if (not face_hit) and (not same_person):
            final_name = ""
            final_mid = -1
            if int(entry.get("continuity_break_frames", 0)) >= max(1, int(getattr(args, "name_break_clear_frames", 3))):
                entry["last"] = ""
                entry["ttl"] = 0
                entry["assigned_name"] = ""
                entry["assigned_member_id"] = -1
                entry["assigned_score"] = 0.0
                entry["confirmed_face_label"] = ""
                entry["confirmed_face_member_id"] = -1
                entry["confirmed_face_sim"] = 0.0
                entry["has_approved_face"] = False
                entry["pending_face_label"] = ""
                entry["pending_face_member_id"] = -1
                entry["pending_face_count"] = 0
                entry["pending_face_best_sim"] = 0.0
                entry["locked"] = False
                entry["lock_ttl"] = 0
                entry["lock_label"] = ""
                entry["last_good_bbox"] = None
        if final_name and camera_name_owner is not None:
            owner_tid = camera_name_owner.owner_tid(str(final_name), frame_idx=int(frame_idx))
            if owner_tid is not None and int(owner_tid) != int(logical_tid):
                final_name = ""
                final_mid = -1
                demote_identity_entry(entry, clear_face=True)
        if final_name:
            entry["last_good_bbox"] = curr_bbox

        low_face_hit = bool(r.get("low_face_hit", False))
        low_face_sim = float(r.get("low_face_sim", 0.0) or 0.0)
        if bool(getattr(args, "hide_unknown", False)) and not final_name:
            if not low_face_hit:
                continue
        disp_face_sim = float(face_sim) if face_hit else float(entry.get("last_face_sim", last_face_sim))
        draw_candidates.append(_DrawItemStable(
            tid=int(logical_tid),
            raw_tid=int(raw_tid),
            bbox=curr_bbox,
            name=str(final_name),
            member_id=int(final_mid),
            face_sim=float(disp_face_sim),
            stable_score=float(stable_score),
            det_conf=r.get("det_conf", None),
            face_hit=bool(face_hit),
            low_face=bool(low_face_hit),
            low_face_sim=float(low_face_sim),
        ))

    if not bool(getattr(args, "allow_duplicate_names", False)):
        draw_final, demoted_tids = _block_duplicate_names_stable(draw_candidates)
        for demoted_tid in list(demoted_tids):
            demote_identity_entry(identity_state.get(int(demoted_tid)), clear_face=True)
    else:
        draw_final = draw_candidates
    if bool(getattr(args, "global_unique_names", False)) and global_owner is not None:
        gated: List[_DrawItemStable] = []
        for it in draw_final:
            if not it.name:
                gated.append(it)
                continue
            score = float(it.face_sim)
            if global_owner.allow(it.name, sid=int(sid), score=score):
                gated.append(it)
                continue
            demote_identity_entry(identity_state.get(int(it.tid)), clear_face=True)
            gated.append(_DrawItemStable(
                tid=int(it.tid),
                raw_tid=int(it.raw_tid),
                bbox=it.bbox,
                name="",
                member_id=-1,
                face_sim=float(it.face_sim),
                stable_score=float(it.stable_score),
                det_conf=it.det_conf,
                face_hit=bool(it.face_hit),
                low_face=bool(it.low_face),
                low_face_sim=float(it.low_face_sim),
            ))
        draw_final = gated

    present_conf: Dict[str, float] = {}
    for it in draw_final:
        if it.name:
            try:
                present_conf[it.name] = max(float(present_conf.get(it.name, 0.0)), float(it.face_sim or 0.0))
            except Exception:
                present_conf[it.name] = float(present_conf.get(it.name, 0.0))
    present_names = sorted(present_conf.keys())

    shown = 0
    events: List[Tuple[int, int, int, int, int, str, float, int]] = []
    for it in draw_final:
        x1, y1, x2, y2 = it.bbox

        # 🔥 Better known check
        is_known = bool(it.name) and int(it.member_id) > 0

        # 🔴 FIXED (no leaks)
        if bool(args.hide_unknown) and not is_known:
            continue

        show_tid = int(it.tid)
        show_raw_tid = int(it.raw_tid if int(it.raw_tid) > 0 else it.tid)

        if is_known:
            color = (0, 255, 0)
            label_txt = (
                f"{it.name} (T{show_tid})"
                if show_raw_tid == show_tid
                else f"{it.name} (T{show_tid}->{show_raw_tid})"
            )
        else:
            color = (0, 255, 255)
            label_txt = f"Unknown (T{show_raw_tid})"

        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)

        cv2.putText(
            out,
            label_txt,
            (x1, max(0, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            color,
            4
        )

        shown += 1

        event_sim = float(it.face_sim if is_known else it.low_face_sim)

        events.append((
            int(it.tid), x1, y1, x2, y2,
            str(it.name), float(event_sim),
            int(it.member_id),
            int(1 if is_known else 0)
        ))
    if known_reattach_mgr is not None:
        snaps: List[Dict[str, Any]] = []
        for it in draw_final:
            entry = identity_state.get(int(it.tid), {})
            has_approved_face = bool(entry.get("has_approved_face", False)) and bool(entry.get("confirmed_face_label", ""))
            snaps.append({
                "raw_tid": int(it.raw_tid if int(it.raw_tid) > 0 else it.tid),
                "logical_tid": int(it.tid),
                "bbox": tuple(int(v) for v in it.bbox),
                "name": str(it.name or ""),
                "member_id": int(it.member_id if int(it.member_id) > 0 else entry.get("assigned_member_id", -1)),
                "has_approved_face": bool(has_approved_face),
            })
        known_reattach_mgr.end_frame(track_snapshots=snaps, frame_idx=int(frame_idx))

    cleanup_frames = max(30, int(getattr(args, "max_age", 15)) + int(getattr(args, "iou_max_miss", 5)) + 10)
    for tid in list(identity_state.keys()):
        if not isinstance(tid, int):
            continue
        st = identity_state.get(tid, {})
        lf = int(st.get("last_seen_frame", -1)) if isinstance(st, dict) else -1
        if lf >= 0 and (int(frame_idx) - lf) > cleanup_frames:
            identity_state.pop(tid, None)

    meta = {
        "tracks": int(len(tracks_info)),
        "shown": int(shown),
        "faces_recognized": int(len(recognized_faces)),
        "do_face": bool(do_face),
        "events": events,
        "present_names": present_names,
        "present_conf": present_conf,
    }
    return out, meta


if __name__ == "__main__":
    try:
        print("[BOOT] tracking_face_only_anti_swap_fixed_20260320.py starting...")
        main()
    except SystemExit:
        raise
    except Exception:
        import traceback
        print("[FATAL] Unhandled exception:")
        traceback.print_exc()
        sys.exit(1)
