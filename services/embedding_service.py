# # app/services/embedding_service.py
# from __future__ import annotations

# import csv
# import sys
# import os 
# from dotenv import load_dotenv
# from pathlib import Path
# import time
# import threading
# from pathlib import Path
# from datetime import datetime, timezone
# from typing import Optional, List, Dict, Tuple, Callable, Any
# from queue import Queue, Empty
# import logging
# import shutil
# import random
# import io
# import gzip
# import zlib
# import re
# import inspect
# from contextlib import contextmanager
# from typing import Generator

# import cv2
# import numpy as np
# import torch

# from app.core.constants import (
#     RTSP_STREAMS,
#     YOLO_WEIGHTS,
#     DEVICE,
#     CONF_THRES,
#     IOU_THRES,
#     EMB_CSV,
#     CROPS_ROOT,
# )
# from app.db.session import get_db, engine  # engine kept for compatibility
# from app.db.models import Member, MemberEmbedding

# try:
#     from app.db.models import Camera
# except Exception:
#     Camera = None  # type: ignore


# BASE_DIR = Path(__file__).resolve().parents[2]
# load_dotenv(BASE_DIR / ".env") 

# # ===================== DB Session Wrapper =====================

# @contextmanager
# def db_session():
#     """
#     Works with both styles:
#       - get_db() returns Session (old style)
#       - get_db() yields Session (FastAPI dependency style -> generator)
#     """
#     db_obj: Any = get_db()

#     # FastAPI dependency style: generator that yields a Session
#     if inspect.isgenerator(db_obj):
#         gen: Generator = db_obj  # type: ignore
#         try:
#             db = next(gen)  # actual Session
#         except StopIteration:
#             raise RuntimeError("get_db() did not yield a DB session")
#         try:
#             yield db
#         finally:
#             try:
#                 gen.close()
#             except Exception:
#                 pass
#         return

#     # Old style: get_db() returned an actual Session
#     db = db_obj
#     try:
#         yield db
#     finally:
#         try:
#             db.close()
#         except Exception:
#             pass


# # ===================== Runtime perf knobs =====================

# try:
#     torch.backends.cudnn.benchmark = True
# except Exception:
#     pass
# try:
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")
# except Exception:
#     pass
# try:
#     cv2.setNumThreads(1)
# except Exception:
#     pass


# # ===================== Optional libs =====================

# try:
#     from ultralytics import YOLO
# except Exception:
#     YOLO = None

# try:
#     from torchreid.utils import FeatureExtractor as TorchreidExtractor
# except Exception as e:
#     print("-----------------TorchReID import error:", e)
#     TorchreidExtractor = None

# try:
#     from insightface.app import FaceAnalysis
#     INSIGHT_OK = True
# except Exception:
#     FaceAnalysis = None
#     INSIGHT_OK = False

# try:
#     import onnxruntime as ort
# except Exception:
#     ort = None


# # ===================== Config =====================

# EXPECTED_EMBED_DIM = 512

# RAW_STORE_LIMIT = int(os.getenv("RAW_STORE_LIMIT", "256"))

# USE_FACE = True
# FACE_MODEL = "buffalo_l"
# FACE_DET_SIZE = (1280, 1280)
# FACE_PROVIDER = "auto"
# FACE_MIN_SCORE = 0.5
# FACE_MIN_SIZE = 32
# FACE_IOU_LINK = 0.05
# FACE_OVER_FACE_LINK = 0.60
# FACE_EVERY = int(os.getenv("FACE_EVERY", "2"))

# YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))

# CAP_QUEUE_MAX = int(os.getenv("CAP_QUEUE_MAX", "2"))
# IO_QUEUE_MAX = int(os.getenv("IO_QUEUE_MAX", "1024"))

# VIEWER_SLEEP_MS = int(os.getenv("VIEWER_SLEEP_MS", "5"))

# REID_MODELS = ["osnet_x1_0", "osnet_x0_25"]
# BODY_TTA_FLIP = True
# FACE_TTA_FLIP = True

# BACK_BODY_GALLERY_NAME = "gallery_body"
# BACK_HEAD_CUT_RATIO = float(os.getenv("BACK_HEAD_CUT_RATIO", "0.0"))
# BACK_SHAPE_SEED = int(os.getenv("BACK_SHAPE_SEED", "1337"))

# BACK_REID_WEIGHT = float(os.getenv("BACK_REID_WEIGHT", "0.65"))
# BACK_SHAPE_WEIGHT = float(os.getenv("BACK_SHAPE_WEIGHT", "0.35"))

# BACK_HOG_WIN = (64, 128)
# BACK_HOG_BLOCK = (16, 16)
# BACK_HOG_STRIDE = (8, 8)
# BACK_HOG_CELL = (8, 8)
# BACK_HOG_BINS = 9

# SAVE_MIN_INTERVAL_MS = int(os.getenv("SAVE_MIN_INTERVAL_MS", "200"))
# SAVE_MAX_DETS_PER_FRAME = int(os.getenv("SAVE_MAX_DETS_PER_FRAME", "3"))
# JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

# YOLO_REQ_QUEUE_MAX = int(os.getenv("YOLO_REQ_QUEUE_MAX", "64"))
# YOLO_BATCH_MAX = int(os.getenv("YOLO_BATCH_MAX", "8"))
# YOLO_BATCH_WAIT_MS = int(os.getenv("YOLO_BATCH_WAIT_MS", "10"))
# YOLO_RESP_TIMEOUT_S = float(os.getenv("YOLO_RESP_TIMEOUT_S", "0.7"))

# MAX_BODY_IMAGES = int(os.getenv("MAX_BODY_IMAGES", "600"))
# MAX_FACE_IMAGES = int(os.getenv("MAX_FACE_IMAGES", "600"))
# MAX_BACK_BODY_IMAGES = int(os.getenv("MAX_BACK_BODY_IMAGES", "600"))
# GALLERY_SAMPLE_SEED = int(os.getenv("GALLERY_SAMPLE_SEED", "1337"))

# # Debug aids (default off)
# DEBUG_SAVE_NO_DETS_EVERY_SEC = float(os.getenv("DEBUG_SAVE_NO_DETS_EVERY_SEC", "0"))  # 0 disables
# DEBUG_LOG_NO_DET_EVERY = int(os.getenv("DEBUG_LOG_NO_DET_EVERY", "150"))

# # ===================== DB Camera -> RTSP (CRITICAL FIX) =====================

# USE_DB_CAMERA_CONFIG = os.getenv("USE_DB_CAMERA_CONFIG", "1").lower() in ("1", "true", "yes", "y", "on")
# ONLY_ACTIVE_CAMERAS = os.getenv("ONLY_ACTIVE_CAMERAS", "1").lower() in ("1", "true", "yes", "y", "on")

# # Recommended:
# #   RTSP_URL_TEMPLATE="rtsp://user:pass@{ip}:554/Streaming/Channels/101"
# RTSP_URL_TEMPLATE = os.getenv("RTSP_URL_TEMPLATE", "").strip()

# # Optional non-template builder:
# RTSP_USERNAME = os.getenv("RTSP_USERNAME", os.getenv("RTSP_USER", "")).strip()
# RTSP_PASSWORD = os.getenv("RTSP_PASSWORD", os.getenv("RTSP_PASS", "")).strip()
# RTSP_PORT = os.getenv("RTSP_PORT", "554").strip()
# RTSP_PATH = os.getenv("RTSP_PATH", "").strip()
# RTSP_SCHEME = os.getenv("RTSP_SCHEME", "rtsp").strip() or "rtsp"
# RTSP_STREAM = os.getenv("RTSP_STREAM", "").strip() or ""
# RTSP_CHANNEL = os.getenv("RTSP_CHANNEL", "1").strip()   # camera channel
# RTSP_SUBTYPE = os.getenv("RTSP_SUBTYPE", "0").strip() 

# # ===================== Logging =====================

# logger = logging.getLogger("app.embedding_service")
# if not logger.handlers:
#     ch = logging.StreamHandler()
#     ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
#     logger.addHandler(ch)
# logger.setLevel(logging.INFO)


# # ===================== Globals =====================

# yolo_model: YOLO | None = None
# reid_extractor: TorchreidExtractor | None = None
# reid_extractors: List[TorchreidExtractor] = []
# face_app: FaceAnalysis | None = None

# reid_lock = threading.Lock()
# csv_lock = threading.Lock()
# frames_lock = threading.Lock()

# latest_frames: Dict[int, np.ndarray] = {}

# capture_threads: Dict[int, threading.Thread] = {}
# extract_threads: Dict[int, threading.Thread] = {}
# cap_queues: Dict[int, Queue] = {}

# io_thread: Optional[threading.Thread] = None
# io_queue: Queue | None = None

# yolo_thread: Optional[threading.Thread] = None
# yolo_req_queue: Queue | None = None

# viewer_thread: Optional[threading.Thread] = None
# stop_event = threading.Event()
# is_running = False

# current_member_id: Optional[int] = None
# current_member_name: Optional[str] = None
# current_camera_ids: List[int] = []
# current_camera_streams: Dict[int, str] = {}

# progress_lock = threading.Lock()
# progress_state: Dict[Tuple[int, int], Dict[str, object]] = {}

# _back_hog = None
# _back_proj = None
# _back_proj_in_dim = None

# OVERALL_CAMERA_ID = -1


# # ===================== Cropping =====================

# def crop_with_padding(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
#     H, W = frame.shape[:2]
#     x1, y1, x2, y2 = bbox

#     x1 = max(0, min(W - 1, int(x1)))
#     y1 = max(0, min(H - 1, int(y1)))
#     x2 = max(0, min(W, int(x2)))
#     y2 = max(0, min(H, int(y2)))

#     if x2 <= x1 or y2 <= y1:
#         return np.zeros((0, 0, 3), dtype=np.uint8)

#     return frame[y1:y2, x1:x2].copy()


# def crop_back_body_no_face(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
#     x1, y1, x2, y2 = bbox
#     h = max(0, int(y2) - int(y1))
#     if h <= 0:
#         return np.zeros((0, 0, 3), dtype=np.uint8)
#     y1b = int(y1 + h * float(max(0.0, min(0.45, BACK_HEAD_CUT_RATIO))))
#     if y1b >= y2 - 2:
#         return np.zeros((0, 0, 3), dtype=np.uint8)
#     return crop_with_padding(frame, (x1, y1b, x2, y2))


# # ===================== IO Worker =====================

# def start_workers_if_needed() -> None:
#     global io_thread, io_queue, yolo_thread, yolo_req_queue

#     if io_queue is None:
#         io_queue = Queue(maxsize=IO_QUEUE_MAX)

#     if io_thread is None or not io_thread.is_alive():
#         io_thread = threading.Thread(target=io_worker, name="io_worker", daemon=True)
#         io_thread.start()

#     if yolo_req_queue is None:
#         yolo_req_queue = Queue(maxsize=YOLO_REQ_QUEUE_MAX)

#     if yolo_thread is None or not yolo_thread.is_alive():
#         if yolo_model is not None:
#             yolo_thread = threading.Thread(target=yolo_worker, name="yolo_worker", daemon=True)
#             yolo_thread.start()


# def io_worker() -> None:
#     ensure_csv_header()
#     csv_path = Path(EMB_CSV)
#     _ensure_dir(csv_path.parent if csv_path.parent.as_posix() not in (".", "") else Path("."))

#     f = None
#     writer = None
#     flush_every = 128
#     pending = 0
#     last_flush = time.time()

#     try:
#         f = open(csv_path, "a", newline="", encoding="utf-8")
#         writer = csv.writer(f)

#         while True:
#             if stop_event.is_set() and (io_queue is None or io_queue.empty()):
#                 break

#             try:
#                 job = io_queue.get(timeout=0.1)  # type: ignore
#             except Empty:
#                 if pending and (time.time() - last_flush) > 1.0 and f is not None:
#                     try:
#                         f.flush()
#                     except Exception:
#                         pass
#                     pending = 0
#                     last_flush = time.time()
#                 continue
#             except Exception:
#                 break

#             try:
#                 crop_path: Path = job["crop_path"]
#                 _ensure_dir(crop_path.parent)

#                 img = job["image"]
#                 ok = False
#                 if str(crop_path).lower().endswith((".jpg", ".jpeg")):
#                     ok = bool(cv2.imwrite(str(crop_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]))
#                 else:
#                     ok = bool(cv2.imwrite(str(crop_path), img))

#                 if not ok:
#                     logger.error("cv2.imwrite failed: %s", crop_path)

#                 if writer is not None:
#                     writer.writerow(job["csv_row"])
#                     pending += 1

#                 if f is not None and (pending >= flush_every or (time.time() - last_flush) > 1.0):
#                     try:
#                         f.flush()
#                     except Exception:
#                         pass
#                     pending = 0
#                     last_flush = time.time()
#             except Exception:
#                 logger.exception("IO worker error")
#             finally:
#                 try:
#                     io_queue.task_done()  # type: ignore
#                 except Exception:
#                     pass
#     finally:
#         try:
#             if f is not None:
#                 f.flush()
#                 f.close()
#         except Exception:
#             pass


# # ===================== YOLO Batch Worker =====================

# def yolo_worker() -> None:
#     if yolo_model is None:
#         return

#     gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
#     device = DEVICE if gpu else "cpu"

#     while True:
#         if stop_event.is_set() and (yolo_req_queue is None or yolo_req_queue.empty()):
#             break
#         try:
#             req0 = yolo_req_queue.get(timeout=0.1)  # type: ignore
#         except Empty:
#             continue
#         except Exception:
#             break

#         batch = [req0]
#         t0 = time.time()
#         while len(batch) < max(1, int(YOLO_BATCH_MAX)) and ((time.time() - t0) * 1000.0) < float(YOLO_BATCH_WAIT_MS):
#             try:
#                 batch.append(yolo_req_queue.get_nowait())  # type: ignore
#             except Empty:
#                 break
#             except Exception:
#                 break

#         frames = [b["frame"] for b in batch]
#         try:
#             with torch.inference_mode():
#                 results = yolo_model.predict(
#                     frames,
#                     conf=float(CONF_THRES),
#                     iou=float(IOU_THRES),
#                     imgsz=int(YOLO_IMGSZ),
#                     verbose=False,
#                     device=device,
#                 )
#         except Exception:
#             logger.exception("YOLO batch failed")
#             results = [None for _ in frames]

#         for req, res in zip(batch, results):
#             H, W = int(req["H"]), int(req["W"])
#             outs: List[tuple[float, float, float, float, float]] = []

#             try:
#                 boxes = getattr(res, "boxes", None) if res is not None else None
#                 if boxes is not None:
#                     xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
#                     confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
#                     cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
#                     keep = (cls == 0)
#                     xyxy, confs = xyxy[keep], confs[keep]

#                     for (x1, y1, x2, y2), c in zip(xyxy, confs):
#                         x1f = float(max(0, min(W - 1, x1)))
#                         y1f = float(max(0, min(H - 1, y1)))
#                         x2f = float(max(0, min(W - 1, x2)))
#                         y2f = float(max(0, min(H - 1, y2)))
#                         if (x2f - x1f) < 4 or (y2f - y1f) < 4:
#                             continue
#                         outs.append((x1f, y1f, x2f, y2f, float(c)))
#             except Exception:
#                 outs = []

#             try:
#                 respq: Queue = req["resp"]
#                 respq.put_nowait(outs)
#             except Exception:
#                 pass

#             try:
#                 yolo_req_queue.task_done()  # type: ignore
#             except Exception:
#                 pass


# def yolo_infer(frame: np.ndarray, H: int, W: int, timeout_s: float = YOLO_RESP_TIMEOUT_S) -> List[tuple[float, float, float, float, float]]:
#     if yolo_model is None or yolo_req_queue is None:
#         return []
#     resp = Queue(maxsize=1)
#     try:
#         yolo_req_queue.put_nowait({"frame": frame, "H": H, "W": W, "resp": resp})
#     except Exception:
#         return []
#     try:
#         return resp.get(timeout=float(timeout_s))
#     except Exception:
#         return []


# # ===================== Progress helpers =====================

# def _pkey(member_id: int, camera_id: int) -> Tuple[int, int]:
#     return (int(member_id), int(camera_id))


# def _progress_reset(member_id: int, member_name: str, camera_id: int) -> None:
#     with progress_lock:
#         progress_state[_pkey(member_id, camera_id)] = {
#             "member_id": int(member_id),
#             "member_name": member_name,
#             "camera_id": int(camera_id),
#             "stage": "idle",
#             "total_body": 0,
#             "total_face": 0,
#             "total_back_body": 0,
#             "done_body": 0,
#             "done_face": 0,
#             "done_back_body": 0,
#             "work_total": 0,
#             "work_done": 0,
#             "percent": 0,
#             "message": "waiting",
#         }


# def _progress_set(member_id: int, camera_id: int, **kv) -> None:
#     with progress_lock:
#         st = progress_state.get(_pkey(member_id, camera_id))
#         if not st:
#             st = {}
#             progress_state[_pkey(member_id, camera_id)] = st
#         st.update(kv)

#         tb = int(st.get("total_body", 0))
#         tf = int(st.get("total_face", 0))
#         tbb = int(st.get("total_back_body", 0))

#         db_ = int(st.get("done_body", 0))
#         df_ = int(st.get("done_face", 0))
#         dbb = int(st.get("done_back_body", 0))

#         db_ = max(0, min(db_, tb))
#         df_ = max(0, min(df_, tf))
#         dbb = max(0, min(dbb, tbb))

#         st["done_body"], st["done_face"], st["done_back_body"] = db_, df_, dbb

#         work_total = int(st.get("work_total", 0))
#         work_done = int(st.get("work_done", 0))
#         work_done = max(0, min(work_done, max(0, work_total)))
#         st["work_done"] = work_done

#         stage = str(st.get("stage", "")).lower()

#         if stage.startswith("embedding_") and work_total > 0:
#             pct = int(round((work_done / max(1, work_total)) * 100))
#         elif stage == "Body embeddings" and tb > 0:
#             pct = int(round((db_ / max(1, tb)) * 100))
#         elif stage == "Face embeddings" and tf > 0:
#             pct = int(round((df_ / max(1, tf)) * 100))
#         elif stage in ("Back body embeddings", "embedding_back") and tbb > 0:
#             pct = int(round((dbb / max(1, tbb)) * 100))
#         else:
#             tot = max(1, tb + tf + tbb)
#             pct = int(round(((db_ + df_ + dbb) / tot) * 100))

#         st["percent"] = max(0, min(100, pct))


# def _default_progress(member_id: int, camera_id: int) -> Dict[str, object]:
#     return {
#         "member_id": int(member_id),
#         "camera_id": int(camera_id),
#         "stage": "unknown",
#         "percent": 0,
#         "message": "no job",
#         "total_body": 0,
#         "total_face": 0,
#         "total_back_body": 0,
#         "done_body": 0,
#         "done_face": 0,
#         "done_back_body": 0,
#         "work_total": 0,
#         "work_done": 0,
#     }


# def get_progress_for_member(member_id: int, camera_id: Optional[int] = None) -> Dict[str, object]:
#     mid = int(member_id)
#     with progress_lock:
#         if camera_id is not None:
#             st = progress_state.get(_pkey(mid, int(camera_id)))
#             return dict(st) if st else _default_progress(mid, int(camera_id))

#         st_overall = progress_state.get(_pkey(mid, OVERALL_CAMERA_ID))
#         if st_overall:
#             return dict(st_overall)

#         cams = {cid: dict(st) for (m, cid), st in progress_state.items() if m == mid and cid != OVERALL_CAMERA_ID}
#         if len(cams) == 1:
#             return next(iter(cams.values()))
#         if cams:
#             return {"member_id": mid, "stage": "multi", "cameras": cams}
#         return _default_progress(mid, OVERALL_CAMERA_ID)


# # ===================== Utils =====================

# def l2_normalize(v: np.ndarray) -> np.ndarray:
#     v = np.asarray(v, dtype=np.float32).reshape(-1)
#     n = float(np.linalg.norm(v))
#     if n == 0.0 or not np.isfinite(n):
#         return v
#     return v / n


# def _as_512f(vec: np.ndarray | list | None) -> Optional[np.ndarray]:
#     if vec is None:
#         return None
#     try:
#         a = np.asarray(vec, dtype=np.float32).reshape(-1)
#         if a.size != EXPECTED_EMBED_DIM:
#             return None
#         if not np.isfinite(a).all():
#             return None
#         return l2_normalize(a)
#     except Exception:
#         return None


# def _pack_raw_bank(vectors: Optional[List[np.ndarray]]) -> Optional[bytes]:
#     if not vectors:
#         return None
#     cleaned: List[np.ndarray] = []
#     for v in vectors[: max(1, int(RAW_STORE_LIMIT))]:
#         vv = _as_512f(v)
#         if vv is not None:
#             cleaned.append(vv.astype(np.float32))
#     if not cleaned:
#         return None
#     arr = np.stack(cleaned, axis=0).astype(np.float32)
#     buf = io.BytesIO()
#     np.save(buf, arr, allow_pickle=False)
#     return gzip.compress(buf.getvalue(), compresslevel=6)


# def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
#     return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# def _safe_slug(s: str, max_len: int = 64) -> str:
#     s = (s or "").strip()
#     s = re.sub(r"[^\w\-\.]+", "_", s, flags=re.UNICODE)
#     s = re.sub(r"_+", "_", s).strip("_")
#     if not s:
#         s = "member"
#     return s[:max_len]


# # ===================== Back-body shape embedding =====================

# def _init_back_shape_embedder() -> None:
#     global _back_hog
#     if _back_hog is None:
#         try:
#             _back_hog = cv2.HOGDescriptor(
#                 _winSize=BACK_HOG_WIN,
#                 _blockSize=BACK_HOG_BLOCK,
#                 _blockStride=BACK_HOG_STRIDE,
#                 _cellSize=BACK_HOG_CELL,
#                 _nbins=BACK_HOG_BINS,
#             )
#             logger.info("Back-body HOG ready (dim=%s)", int(_back_hog.getDescriptorSize()))
#         except Exception:
#             logger.exception("Failed to init HOG descriptor for back-body embeddings")
#             _back_hog = None


# def _get_back_proj(in_dim: int) -> Optional[np.ndarray]:
#     global _back_proj, _back_proj_in_dim
#     if in_dim <= 0:
#         return None
#     if _back_proj is None or _back_proj_in_dim != int(in_dim):
#         rng = np.random.RandomState(int(BACK_SHAPE_SEED))
#         mat = rng.normal(
#             loc=0.0,
#             scale=float(1.0 / max(1.0, np.sqrt(in_dim))),
#             size=(EXPECTED_EMBED_DIM, in_dim),
#         ).astype(np.float32)
#         _back_proj = mat
#         _back_proj_in_dim = int(in_dim)
#         logger.info("Back-body projection matrix built (seed=%s, in_dim=%s)", BACK_SHAPE_SEED, in_dim)
#     return _back_proj


# def _back_shape_embed(img_bgr: np.ndarray) -> Optional[np.ndarray]:
#     if img_bgr is None or img_bgr.size == 0:
#         return None
#     _init_back_shape_embedder()
#     if _back_hog is None:
#         return None
#     try:
#         g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#         g = cv2.resize(g, BACK_HOG_WIN, interpolation=cv2.INTER_AREA)
#         hog = _back_hog.compute(g)
#         if hog is None:
#             return None
#         hog = np.asarray(hog, dtype=np.float32).reshape(-1)
#         if hog.size <= 0 or not np.isfinite(hog).all():
#             return None
#         hn = float(np.linalg.norm(hog))
#         if hn > 0:
#             hog = hog / hn
#         proj = _get_back_proj(int(hog.size))
#         if proj is None:
#             return None
#         emb = proj @ hog
#         return _as_512f(emb)
#     except Exception:
#         return None


# # ===================== Geometry helpers =====================

# def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
#     inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
#     inter = iw * ih
#     a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
#     b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
#     denom = a_area + b_area - inter
#     return inter / denom if denom > 0 else 0.0


# def inter_over_face(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
#     inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
#     inter = iw * ih
#     face_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
#     return inter / face_area if face_area > 0 else 0.0


# def face_center_in(a: Tuple[int, int, int, int], b: Tuple[float, float, float, float]) -> bool:
#     fx1, fy1, fx2, fy2 = a
#     px1, py1, px2, py2 = b
#     cx = (fx1 + fx2) * 0.5
#     cy = (fy1 + fy2) * 0.5
#     return (px1 <= cx <= px2) and (py1 <= cy <= py2)


# def safe_iter_faces(obj: Any) -> List[Any]:
#     if obj is None:
#         return []
#     try:
#         return list(obj)
#     except TypeError:
#         return [obj]


# def extract_face_embedding(face: Any):
#     emb = getattr(face, "normed_embedding", None)
#     if emb is None:
#         emb = getattr(face, "embedding", None)
#     return emb


# def _cuda_ep_loadable() -> bool:
#     if ort is None:
#         return False
#     try:
#         if sys.platform.startswith("mac"):
#             return False
#         from pathlib import Path as _P
#         import ctypes as _ct
#         capi_dir = _P(ort.__file__).parent / "capi"
#         name = "onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so"
#         lib_path = capi_dir / name
#         if not lib_path.exists():
#             return False
#         _ct.CDLL(str(lib_path))
#         return True
#     except Exception:
#         return False


# # ===================== Models init =====================

# def init_face_engine(use_face: bool, device: str, face_model: str, det_w: int, det_h: int, face_provider: str):
#     if not use_face:
#         return None
#     if not INSIGHT_OK:
#         logger.warning("insightface not installed; face recognition disabled.")
#         return None
#     try:
#         is_cuda = ("cuda" in device.lower()) and torch.cuda.is_available()
#         cuda_ok = _cuda_ep_loadable()

#         providers = ["CPUExecutionProvider"]
#         if face_provider == "cuda":
#             if cuda_ok:
#                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#         elif face_provider == "auto":
#             if is_cuda and cuda_ok:
#                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

#         app = FaceAnalysis(name=face_model, providers=providers)
#         ctx_id = 0 if providers[0].startswith("CUDA") else -1
#         try:
#             app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
#         except TypeError:
#             app.prepare(ctx_id=ctx_id)

#         logger.info("InsightFace ready (model=%s, providers=%s).", face_model, providers)

#         try:
#             dummy = np.zeros((max(8, det_h), max(8, det_w), 3), dtype=np.uint8)
#             app.get(dummy)
#         except Exception:
#             pass
#         return app
#     except Exception:
#         logger.exception("InsightFace init failed")
#         return None


# def init_models() -> None:
#     global yolo_model, reid_extractor, reid_extractors, face_app

#     gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
#     logger.info("init_models device=%s gpu_available=%s", DEVICE, torch.cuda.is_available())

#     # YOLO
#     if YOLO is not None:
#         try:
#             weights = YOLO_WEIGHTS
#             if not Path(weights).exists():
#                 logger.warning("%s not found, falling back to yolov8n.pt", weights)
#             yolo = YOLO(weights if Path(YOLO_WEIGHTS).exists() else "yolov8n.pt")
#             if gpu:
#                 try:
#                     yolo.to(DEVICE)
#                 except Exception:
#                     pass
#             try:
#                 yolo.fuse()
#             except Exception:
#                 pass
#             try:
#                 with torch.inference_mode():
#                     _dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
#                     yolo.predict(
#                         _dummy,
#                         device=DEVICE if gpu else "cpu",
#                         conf=0.25,
#                         iou=0.5,
#                         imgsz=YOLO_IMGSZ,
#                         verbose=False,
#                     )
#             except Exception:
#                 pass
#             yolo_model = yolo
#             logger.info("YOLO ready (gpu=%s)", gpu)
#         except Exception:
#             logger.exception("YOLO load failed; detection disabled.")
#             yolo_model = None
#     else:
#         logger.warning("ultralytics not installed; detection disabled.")
#         yolo_model = None

#     # TorchReID ensemble
#     reid_extractors.clear()
#     if TorchreidExtractor is not None:
#         try:
#             dev = DEVICE if gpu else "cpu"
#             for m in REID_MODELS:
#                 try:
#                     ext = TorchreidExtractor(model_name=m, device=dev)
#                     try:
#                         dummy = np.zeros((256, 128, 3), dtype=np.uint8)
#                         _ = ext([dummy])
#                     except Exception:
#                         pass
#                     reid_extractors.append(ext)
#                     logger.info("TorchReID ready: %s (%s)", m, dev)
#                 except Exception:
#                     logger.exception("TorchReID init failed for %s", m)
#             reid_extractor = reid_extractors[0] if reid_extractors else None
#         except Exception:
#             logger.exception("TorchReID init failed; body embeddings disabled.")
#             reid_extractors.clear()
#             reid_extractor = None
#     else:
#         logger.warning("torchreid not installed; body embeddings disabled.")
#         reid_extractors.clear()
#         reid_extractor = None

#     # InsightFace
#     face_app = init_face_engine(USE_FACE, DEVICE, FACE_MODEL, FACE_DET_SIZE[0], FACE_DET_SIZE[1], FACE_PROVIDER)

#     # Back-body init
#     _init_back_shape_embedder()

#     # Workers
#     start_workers_if_needed()


# # ===================== CSV + gallery helpers =====================

# def _ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)


# def ensure_csv_header() -> None:
#     csv_path = Path(EMB_CSV)
#     _ensure_dir(csv_path.parent if csv_path.parent.as_posix() not in (".", "") else Path("."))

#     expected_header = [
#         "member_id", "member_name", "ts", "camera_id", "frame_idx", "det_idx",
#         "x1", "y1", "x2", "y2", "conf_or_score",
#         "body_embedding", "face_embedding", "crop_path", "kind",
#     ]

#     def _write_new():
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             w = csv.writer(f)
#             w.writerow(expected_header)

#     if not csv_path.exists():
#         with csv_lock:
#             if not csv_path.exists():
#                 _write_new()
#                 logger.info("Created CSV header: %s", EMB_CSV)
#         return

#     try:
#         with open(csv_path, "r", newline="", encoding="utf-8") as f:
#             r = csv.reader(f)
#             hdr = next(r, None)
#     except Exception:
#         hdr = None

#     if hdr == expected_header:
#         return

#     with csv_lock:
#         try:
#             with open(csv_path, "r", newline="", encoding="utf-8") as f:
#                 r = csv.reader(f)
#                 hdr2 = next(r, None)
#         except Exception:
#             hdr2 = None
#         if hdr2 == expected_header:
#             return

#         bak = csv_path.with_suffix(csv_path.suffix + f".bak_{int(time.time())}")
#         try:
#             shutil.copy2(csv_path, bak)
#             logger.warning("CSV header mismatch. Backed up old CSV to: %s", bak)
#         except Exception:
#             logger.warning("CSV header mismatch, but backup failed. Rewriting header anyway.")
#         _write_new()
#         logger.info("Rewrote CSV header: %s", EMB_CSV)


# def _member_folder(member_id: int, member_name: Optional[str] = None) -> str:
#     mid = int(member_id)
#     if member_name:
#         return f"{mid}_{_safe_slug(str(member_name))}"
#     return str(mid)


# def _gallery_body_dir(member_id: int, camera_id: int, member_name: Optional[str] = None) -> Path:
#     d = Path(CROPS_ROOT) / "gallery" / _member_folder(member_id, member_name) / f"cam_{int(camera_id)}"
#     _ensure_dir(d)
#     return d


# def _gallery_face_dir(member_id: int, camera_id: int, member_name: Optional[str] = None) -> Path:
#     d = Path(CROPS_ROOT) / "gallery_face" / _member_folder(member_id, member_name) / f"cam_{int(camera_id)}"
#     _ensure_dir(d)
#     return d


# def _gallery_back_body_dir(member_id: int, camera_id: int, member_name: Optional[str] = None) -> Path:
#     d = Path(CROPS_ROOT) / BACK_BODY_GALLERY_NAME / _member_folder(member_id, member_name) / f"cam_{int(camera_id)}"
#     _ensure_dir(d)
#     return d


# def _epoch_ms() -> int:
#     return int(time.time() * 1000) * 1000 + random.randint(0, 999)


# # ===================== Face helpers (capture-time) =====================

# def detect_faces_raw(frame: np.ndarray):
#     outs: List[Dict] = []
#     if face_app is None:
#         return outs
#     try:
#         faces = face_app.get(np.ascontiguousarray(frame))
#         for f in safe_iter_faces(faces):
#             bbox = getattr(f, "bbox", None)
#             if bbox is None:
#                 continue
#             b = np.asarray(bbox).reshape(-1)
#             if b.size < 4:
#                 continue
#             x1, y1, x2, y2 = map(float, b[:4])
#             score = float(getattr(f, "det_score", 0.0))
#             outs.append({"bbox": (x1, y1, x2, y2), "score": score})
#     except Exception:
#         logger.exception("FaceAnalysis error")
#     return outs


# def link_face_to_person(face_bbox: Tuple[float, float, float, float], person_bbox: Tuple[int, int, int, int]) -> bool:
#     if face_center_in(tuple(map(int, face_bbox)), person_bbox):
#         return True
#     if iou_xyxy(person_bbox, face_bbox) >= FACE_IOU_LINK:
#         return True
#     if inter_over_face(person_bbox, face_bbox) >= FACE_OVER_FACE_LINK:
#         return True
#     return False


# def _face_big_enough(face_bbox: Tuple[float, float, float, float]) -> bool:
#     x1, y1, x2, y2 = face_bbox
#     return min(float(x2 - x1), float(y2 - y1)) >= float(FACE_MIN_SIZE)


# # ===================== RTSP helpers =====================

# def _configure_rtsp_capture(cap: cv2.VideoCapture) -> None:
#     try:
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     except Exception:
#         pass


# def _read_latest_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
#     grabbed = cap.grab()
#     if not grabbed:
#         ok, fr = cap.read()
#         return ok, (fr if ok else None)
#     for _ in range(3):
#         if not cap.grab():
#             break
#     ok, frame = cap.retrieve()
#     return ok, (frame if ok else None)


# # ===================== Stream / Camera mapping =====================

# def _normalize_camera_ids(camera_ids: Any) -> Optional[List[int]]:
#     if camera_ids is None:
#         return None
#     if isinstance(camera_ids, int):
#         return [int(camera_ids)]
#     if isinstance(camera_ids, str):
#         parts = [p.strip() for p in camera_ids.split(",") if p.strip()]
#         out: List[int] = []
#         for p in parts:
#             try:
#                 out.append(int(p))
#             except Exception:
#                 continue
#         return out if out else None
#     if isinstance(camera_ids, (list, tuple, set)):
#         out: List[int] = []
#         for x in camera_ids:
#             try:
#                 out.append(int(x))
#             except Exception:
#                 continue
#         return out if out else None
#     return None


# def _camera_streams_map(streams: Any) -> Dict[int, str]:
#     out: Dict[int, str] = {}

#     if isinstance(streams, str):
#         s = streams.strip()
#         return {0: s} if s else {}

#     if isinstance(streams, dict):
#         for k, v in streams.items():
#             try:
#                 cid = int(k)
#             except Exception:
#                 continue
#             url = ""
#             if isinstance(v, str):
#                 url = v
#             elif isinstance(v, dict):
#                 url = str(v.get("url") or v.get("rtsp_url") or v.get("rtsp") or v.get("stream") or "")
#             elif isinstance(v, (list, tuple)) and len(v) >= 1:
#                 url = str(v[0])
#             if url:
#                 out[cid] = url
#         return out

#     if isinstance(streams, (list, tuple)):
#         for idx, item in enumerate(streams):
#             if isinstance(item, str):
#                 out[int(idx)] = item
#             elif isinstance(item, dict):
#                 cid = int(item.get("id", idx))
#                 url = str(item.get("url") or item.get("rtsp_url") or item.get("rtsp") or item.get("stream") or "")
#                 if url:
#                     out[cid] = url
#             elif isinstance(item, (list, tuple)):
#                 if len(item) >= 2:
#                     try:
#                         cid = int(item[0])
#                     except Exception:
#                         cid = int(idx)
#                     url = str(item[1])
#                     if url:
#                         out[cid] = url
#                 elif len(item) == 1:
#                     out[int(idx)] = str(item[0])
#     return out


# def _get_member_display_name(member: Any, fallback: str = "unknown") -> str:
#     for attr in ("first_name", "name", "full_name", "username"):
#         v = getattr(member, attr, None)
#         if v:
#             return str(v)
#     return fallback


# def _build_rtsp_url(ip: str, camera_id: int, camera_name: str = "") -> str:
#     ip = (ip or "").strip()
#     if not ip:
#         return ""

#     # If already full RTSP URL, return directly
#     if ip.lower().startswith(("rtsp://", "rtsps://")):
#         return ip

#     # Build Dahua RTSP format
#     auth = RTSP_USERNAME
#     if RTSP_PASSWORD:
#         auth += f":{RTSP_PASSWORD}"
#     auth += "@"

#     return (
#         f"rtsp://{auth}{ip}:{RTSP_PORT}"
#         f"/cam/realmonitor?channel={RTSP_CHANNEL}&subtype={RTSP_SUBTYPE}"
#     )


# def _db_get_camera_ids(
#     camera_ids: Optional[List[int]] = None,
#     only_active: Optional[bool] = None
# ) -> List[int]:
#     if Camera is None:
#         return []

#     try:
#         with db_session() as db:
#             q = db.query(Camera.id)

#             if camera_ids:
#                 q = q.filter(Camera.id.in_(map(int, camera_ids)))

#             if only_active is True:
#                 q = q.filter(Camera.is_active.is_(True))

#             rows = q.all()

#         return sorted(r[0] for r in rows)

#     except Exception as e:
#         logger.exception(f"Failed to load camera ids from DB: {e}")
#         return []


# def _camera_streams_from_db(camera_ids: Optional[List[int]] = None) -> Dict[int, str]:
#     if Camera is None:
#         return {}

#     out: Dict[int, str] = {}
#     try:
#         with db_session() as db:
#             q = db.query(Camera)
#             if camera_ids:
#                 q = q.filter(Camera.id.in_([int(x) for x in camera_ids]))
#             if ONLY_ACTIVE_CAMERAS:
#                 q = q.filter(Camera.is_active.is_(True))
#             cams = q.all()
#     except Exception:
#         logger.exception("Failed to query cameras from DB")
#         return out

#     for cam in cams:
#         try:
#             cid = int(getattr(cam, "id"))
#         except Exception:
#             continue
#         url = _build_rtsp_url(str(getattr(cam, "ip_address", "") or ""), cid, str(getattr(cam, "name", "") or "")) 
#         if url:
#             out[cid] = url

#     return out


# def _get_streams_map(camera_ids: Optional[Any] = None) -> Dict[int, str]:
#     ids = _normalize_camera_ids(camera_ids)

#     if USE_DB_CAMERA_CONFIG:
#         m = _camera_streams_from_db(ids)
#         if m:
#             return m
#         if ids is not None:
#             # user requested IDs but DB could not resolve them
#             return {}

#     return _camera_streams_map(RTSP_STREAMS)


# def _validate_camera_ids_exist(camera_ids: List[int]) -> Tuple[List[int], List[int]]:
#     camera_ids = [int(x) for x in camera_ids if str(x).isdigit()]
#     if not camera_ids:
#         return [], []
#     if Camera is None:
#         return camera_ids, []
#     existing = set(_db_get_camera_ids(camera_ids=camera_ids, only_active=None))
#     valid = [cid for cid in camera_ids if cid in existing]
#     missing = [cid for cid in camera_ids if cid not in existing]
#     return valid, missing


# # ===================== Capture & processing loops =====================

# def capture_thread_fn(camera_id: int, rtsp_url: str):
#     q = cap_queues[camera_id]

#     # Prefer FFMPEG, but fall back if needed
#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#     if not cap.isOpened():
#         cap = cv2.VideoCapture(rtsp_url)

#     if not cap.isOpened():
#         logger.error("[cap cam=%d] cannot open RTSP: %s", camera_id, rtsp_url)
#         return

#     _configure_rtsp_capture(cap)
#     logger.info("[cap cam=%d] started on %s", camera_id, rtsp_url)

#     try:
#         while not stop_event.is_set():
#             ok, frame = _read_latest_frame(cap)
#             if not ok or frame is None:
#                 time.sleep(0.01)
#                 continue

#             while not q.empty():
#                 try:
#                     q.get_nowait()
#                 except Exception:
#                     break
#             try:
#                 q.put_nowait(frame)
#             except Exception:
#                 pass
#     finally:
#         cap.release()
#         logger.info("[cap cam=%d] stopped", camera_id)


# def embedding_loop_for_cam(camera_id: int, rtsp_url: str):
#     global current_member_id, current_member_name

#     ensure_csv_header()
#     start_workers_if_needed()

#     member_id = int(current_member_id or 0)
#     member_name = str(current_member_name or "unknown")

#     body_dir = _gallery_body_dir(member_id, camera_id, member_name)
#     face_dir = _gallery_face_dir(member_id, camera_id, member_name)
#     back_dir = _gallery_back_body_dir(member_id, camera_id, member_name)

#     logger.info("[Loop cam=%d] dirs: body=%s face=%s back=%s", camera_id, body_dir, face_dir, back_dir)

#     frame_idx = 0
#     last_save_ms = 0
#     last_debug_save = 0.0
#     no_det_counter = 0

#     if yolo_model is None:
#         logger.warning("[Loop cam=%d] YOLO model is None -> no detections, no crops", camera_id)

#     logger.info("[Loop cam=%d] started on %s", camera_id, rtsp_url)
#     q = cap_queues[camera_id]

#     try:
#         while not stop_event.is_set():
#             try:
#                 frame = q.get(timeout=0.2)
#             except Empty:
#                 continue
#             except Exception:
#                 break
#             if frame is None:
#                 continue

#             frame_idx += 1
#             H, W = frame.shape[:2]

#             now_ms = int(time.time() * 1000)
#             allow_save = (now_ms - last_save_ms) >= int(SAVE_MIN_INTERVAL_MS)
#             if allow_save:
#                 last_save_ms = now_ms

#             # submit YOLO request
#             respq = Queue(maxsize=1)
#             yolo_ok = (yolo_model is not None and yolo_req_queue is not None)
#             if yolo_ok:
#                 try:
#                     yolo_req_queue.put_nowait({"frame": frame, "H": H, "W": W, "resp": respq})
#                 except Exception:
#                     yolo_ok = False

#             faces_checked = bool(USE_FACE and face_app is not None and (frame_idx % max(1, int(FACE_EVERY)) == 0))
#             faces = detect_faces_raw(frame) if faces_checked else []

#             detections: List[tuple[float, float, float, float, float]] = []
#             if yolo_ok:
#                 try:
#                     detections = respq.get(timeout=float(YOLO_RESP_TIMEOUT_S))
#                 except Exception:
#                     detections = []
#             else:
#                 detections = yolo_infer(frame, H, W, timeout_s=float(YOLO_RESP_TIMEOUT_S))

#             det_boxes_i: List[tuple[int, int, int, int, float]] = []
#             for (x1, y1, x2, y2, conf) in detections:
#                 x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
#                 x1i, y1i = max(0, x1i), max(0, y1i)
#                 x2i, y2i = min(W, x2i), min(H, y2i)
#                 if x2i <= x1i or y2i <= y1i:
#                     continue
#                 det_boxes_i.append((x1i, y1i, x2i, y2i, conf))

#             if not det_boxes_i:
#                 no_det_counter += 1
#                 if DEBUG_LOG_NO_DET_EVERY > 0 and (no_det_counter % DEBUG_LOG_NO_DET_EVERY) == 0:
#                     logger.info("[Loop cam=%d] no detections for %d frames", camera_id, no_det_counter)

#                 # Optional debug: save a raw frame periodically if no detections
#                 if DEBUG_SAVE_NO_DETS_EVERY_SEC > 0:
#                     now = time.time()
#                     if (now - last_debug_save) >= float(DEBUG_SAVE_NO_DETS_EVERY_SEC):
#                         last_debug_save = now
#                         dbg_dir = Path(CROPS_ROOT) / "debug_frames" / _member_folder(member_id, member_name) / f"cam_{camera_id}"
#                         _ensure_dir(dbg_dir)
#                         dbg_path = dbg_dir / f"frame_{int(now)}.jpg"
#                         try:
#                             cv2.imwrite(str(dbg_path), frame)
#                             logger.info("[Loop cam=%d] DEBUG saved frame: %s", camera_id, dbg_path)
#                         except Exception:
#                             pass

#             # Save crops (rate-limited)
#             if allow_save and det_boxes_i:
#                 ts = time.time()
#                 for det_idx, (x1i, y1i, x2i, y2i, conf) in enumerate(det_boxes_i[: max(1, int(SAVE_MAX_DETS_PER_FRAME))]):

#                     # (1) regular body gallery
#                     crop_bgr = crop_with_padding(frame, (x1i, y1i, x2i, y2i))
#                     if crop_bgr.size > 0:
#                         fname = f"person_{_epoch_ms()}.jpg"
#                         path = body_dir / fname
#                         row = [
#                             member_id, member_name, ts,
#                             int(camera_id), frame_idx, det_idx,
#                             x1i, y1i, x2i, y2i,
#                             conf,
#                             "", "", str(path), "body",
#                         ]
#                         try:
#                             if io_queue is not None:
#                                 io_queue.put_nowait({"crop_path": path, "image": crop_bgr, "csv_row": row})
#                         except Exception:
#                             pass

#                     # (3) back-body / no-face gallery
#                     if faces_checked:
#                         has_linked_face = False
#                         if faces:
#                             t_xyxy = (x1i, y1i, x2i, y2i)
#                             for fm in faces:
#                                 fb = fm.get("bbox")
#                                 if not fb:
#                                     continue
#                                 score = float(fm.get("score", 0.0))
#                                 if score < float(FACE_MIN_SCORE):
#                                     continue
#                                 if not _face_big_enough(tuple(map(float, fb))):
#                                     continue
#                                 if link_face_to_person(tuple(map(float, fb)), t_xyxy):
#                                     has_linked_face = True
#                                     break

#                         if not has_linked_face:
#                             back_crop = crop_back_body_no_face(frame, (x1i, y1i, x2i, y2i))
#                             if back_crop.size > 0:
#                                 fname = f"back_{_epoch_ms()}.jpg"
#                                 path = back_dir / fname
#                                 row = [
#                                     member_id, member_name, ts,
#                                     int(camera_id), frame_idx, det_idx,
#                                     x1i, y1i, x2i, y2i,
#                                     conf,
#                                     "", "", str(path), "back_body",
#                                 ]
#                                 try:
#                                     if io_queue is not None:
#                                         io_queue.put_nowait({"crop_path": path, "image": back_crop, "csv_row": row})
#                                 except Exception:
#                                     pass

#                 # (2) gallery_face
#                 if faces and faces_checked:
#                     for det_idx, (x1i, y1i, x2i, y2i, _conf) in enumerate(det_boxes_i[: max(1, int(SAVE_MAX_DETS_PER_FRAME))]):
#                         t_xyxy = (x1i, y1i, x2i, y2i)
#                         best_score = -1.0
#                         best_box = None
#                         for fm in faces:
#                             fb = fm.get("bbox")
#                             if not fb:
#                                 continue
#                             if link_face_to_person(tuple(map(float, fb)), t_xyxy):
#                                 s = float(fm.get("score", 0.0))
#                                 if s > best_score:
#                                     best_score = s
#                                     best_box = fb

#                         if best_box is None:
#                             continue

#                         fx1, fy1, fx2, fy2 = map(int, best_box)
#                         if best_score < FACE_MIN_SCORE:
#                             continue
#                         if min(fx2 - fx1, fy2 - fy1) < FACE_MIN_SIZE:
#                             continue

#                         body_crop = crop_with_padding(frame, (x1i, y1i, x2i, y2i))
#                         if body_crop.size <= 0:
#                             continue

#                         fname = f"person_{_epoch_ms()}.jpg"
#                         path = face_dir / fname
#                         row = [
#                             member_id, member_name, ts,
#                             int(camera_id), frame_idx, det_idx,
#                             x1i, y1i, x2i, y2i,
#                             best_score,
#                             "", "", str(path), "face",
#                         ]
#                         try:
#                             if io_queue is not None:
#                                 io_queue.put_nowait({"crop_path": path, "image": body_crop, "csv_row": row})
#                         except Exception:
#                             pass

#             with frames_lock:
#                 latest_frames[camera_id] = frame

#     finally:
#         logger.info("[Loop cam=%d] stopped", camera_id)


# # ===================== Extract from folders helpers =====================

# def _list_images(root: Path) -> List[Path]:
#     if not root.exists():
#         return []
#     pats = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
#     files: List[Path] = []
#     for p in pats:
#         files.extend(root.glob(p))
#     return sorted(files)


# def _stable_seed(*parts: str) -> int:
#     s = "|".join(parts).encode("utf-8", errors="ignore")
#     return int(zlib.crc32(s) & 0xFFFFFFFF)


# def _sample_paths(paths: List[Path], max_n: int, seed: int) -> List[Path]:
#     if max_n <= 0 or len(paths) <= max_n:
#         return paths
#     rng = random.Random(int(seed))
#     idxs = sorted(rng.sample(range(len(paths)), int(max_n)))
#     return [paths[i] for i in idxs]


# def _list_images_sampled(root: Path, max_n: int, member_id: int, camera_id: int, kind: str) -> List[Path]:
#     paths = _list_images(root)
#     seed = _stable_seed(str(member_id), str(camera_id), kind, str(GALLERY_SAMPLE_SEED))
#     return _sample_paths(paths, int(max_n), seed)


# def _load_bgr(path: Path) -> Optional[np.ndarray]:
#     try:
#         img = cv2.imread(str(path), cv2.IMREAD_COLOR)
#         return img if img is not None else None
#     except Exception:
#         return None


# def _mean_embed(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
#     if not vectors:
#         return None
#     arr = np.stack(vectors, axis=0).astype(np.float32)
#     m = np.mean(arr, axis=0)
#     return l2_normalize(m)


# def _count_images(member_id: int, camera_id: int, member_name: str) -> Tuple[int, int, int]:
#     body = len(_list_images_sampled(_gallery_body_dir(member_id, camera_id, member_name), MAX_BODY_IMAGES, member_id, camera_id, "body"))
#     face = len(_list_images_sampled(_gallery_face_dir(member_id, camera_id, member_name), MAX_FACE_IMAGES, member_id, camera_id, "face"))
#     back_body = len(_list_images_sampled(_gallery_back_body_dir(member_id, camera_id, member_name), MAX_BACK_BODY_IMAGES, member_id, camera_id, "back_body"))
#     return body, face, back_body


# def _ahash8(img_bgr: np.ndarray) -> int:
#     try:
#         g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#         g = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
#         avg = float(g.mean())
#         bits = (g > avg).astype(np.uint8)
#         out = 0
#         for i, b in enumerate(bits.flatten()):
#             out |= (int(b) & 1) << i
#         return out
#     except Exception:
#         return random.getrandbits(64)


# def _is_blurry(img_bgr: np.ndarray, var_thr: float = 30.0) -> bool:
#     try:
#         g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#         return float(cv2.Laplacian(g, cv2.CV_64F).var()) < var_thr
#     except Exception:
#         return False


# def _prep_reid(img_bgr: np.ndarray) -> np.ndarray:
#     h, w = img_bgr.shape[:2]
#     if h < 32 or w < 16:
#         raise ValueError("Crop too small for ReID")
#     img = cv2.resize(img_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
#     return _to_rgb(img)


# def _compute_body_bank_and_centroid_from_gallery(
#     member_id: int,
#     camera_id: int,
#     member_name: str,
#     on_file_done: Optional[Callable[[int], None]] = None,
#     on_work_done: Optional[Callable[[int], None]] = None,
# ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
#     if not reid_extractors:
#         logger.warning("TorchReID not available; skipping body embedding.")
#         return None, None

#     root = _gallery_body_dir(member_id, camera_id, member_name)
#     imgs = _list_images_sampled(root, MAX_BODY_IMAGES, member_id, camera_id, "body")
#     if not imgs:
#         logger.warning("No BODY crops found for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
#     B = 64 if gpu else 32

#     prepared: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
#     seen_hashes: set[int] = set()
#     per_image_units = len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)

#     for p in imgs:
#         img = _load_bgr(p)
#         if img is None or img.size == 0:
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue
#         h, w = img.shape[:2]
#         if min(h, w) < 32 or _is_blurry(img):
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue
#         hval = _ahash8(img)
#         if hval in seen_hashes:
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue
#         seen_hashes.add(hval)

#         try:
#             rgb = _prep_reid(img)
#         except Exception:
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue

#         rgb_flip = cv2.flip(rgb, 1) if BODY_TTA_FLIP else None
#         prepared.append((rgb, rgb_flip))

#     if not prepared:
#         logger.warning("No valid BODY crops for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     per_model_embs: List[List[np.ndarray]] = [[] for _ in reid_extractors]

#     for midx, ext in enumerate(reid_extractors):
#         batch_imgs: List[np.ndarray] = []
#         idx_map: List[int] = []
#         for i, (orig, flip) in enumerate(prepared):
#             batch_imgs.append(orig)
#             idx_map.append(i)
#             if flip is not None:
#                 batch_imgs.append(flip)
#                 idx_map.append(i)

#         feats_norm: List[np.ndarray] = []
#         for start in range(0, len(batch_imgs), B):
#             chunk = batch_imgs[start:start + B]
#             try:
#                 with reid_lock, torch.inference_mode():
#                     feats = ext(chunk)
#                 for f in feats:
#                     f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
#                     f512 = _as_512f(f)
#                     feats_norm.append(f512 if f512 is not None else np.zeros((EXPECTED_EMBED_DIM,), np.float32))
#             except Exception:
#                 logger.exception("TorchReID batch failed (model=%s)", REID_MODELS[midx])
#                 feats_norm.extend([np.zeros((EXPECTED_EMBED_DIM,), np.float32) for _ in chunk])

#             if on_work_done:
#                 on_work_done(len(chunk))

#         img_accum = [[] for _ in prepared]
#         for img_i, feat in zip(idx_map, feats_norm):
#             img_accum[img_i].append(feat)

#         fused = [
#             l2_normalize(np.mean(np.stack(v, axis=0), axis=0)) if v else np.zeros((EXPECTED_EMBED_DIM,), np.float32)
#             for v in img_accum
#         ]
#         per_model_embs[midx] = fused

#     vectors_per_image: List[np.ndarray] = []
#     for i in range(len(prepared)):
#         parts = [per_model_embs[m][i] for m in range(len(reid_extractors))]
#         fused_img = l2_normalize(np.mean(np.stack(parts, axis=0), axis=0))
#         vectors_per_image.append(fused_img)
#         if on_file_done:
#             on_file_done(1)

#     centroid = _mean_embed(vectors_per_image)
#     return (vectors_per_image if vectors_per_image else None), centroid


# def _compute_face_bank_and_centroid_from_gallery(
#     member_id: int,
#     camera_id: int,
#     member_name: str,
#     on_file_done: Optional[Callable[[int], None]] = None,
#     on_work_done: Optional[Callable[[int], None]] = None,
# ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
#     if face_app is None:
#         logger.warning("InsightFace not available; skipping face embedding.")
#         return None, None

#     root = _gallery_face_dir(member_id, camera_id, member_name)
#     imgs = _list_images_sampled(root, MAX_FACE_IMAGES, member_id, camera_id, "face")
#     if not imgs:
#         logger.warning("No FACE crops found for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     vectors: List[np.ndarray] = []
#     seen_hashes: set[int] = set()
#     MAX_SIDE = 640
#     MIN_BODY_SIDE = 40
#     per_image_units = (2 if FACE_TTA_FLIP else 1)

#     for p in imgs:
#         img = _load_bgr(p)
#         if img is None or img.size == 0:
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue

#         h, w = img.shape[:2]
#         if min(h, w) < MIN_BODY_SIDE or _is_blurry(img):
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue

#         hval = _ahash8(img)
#         if hval in seen_hashes:
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue
#         seen_hashes.add(hval)

#         if max(h, w) > MAX_SIDE:
#             scale = MAX_SIDE / float(max(h, w))
#             img_proc = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
#         else:
#             img_proc = img

#         def _best_face(bgr_img) -> Optional[np.ndarray]:
#             try:
#                 faces = face_app.get(np.ascontiguousarray(bgr_img))
#             except Exception:
#                 faces = []
#             best, best_score = None, -1.0
#             for f in safe_iter_faces(faces):
#                 bbox = getattr(f, "bbox", None)
#                 if bbox is None:
#                     continue
#                 b = np.asarray(bbox).reshape(-1)
#                 if b.size < 4:
#                     continue
#                 x1, y1, x2, y2 = b[:4].astype(float)
#                 area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
#                 score = float(getattr(f, "det_score", 0.0))
#                 ranking = score * (area ** 0.5)
#                 if ranking > best_score:
#                     best, best_score = f, ranking
#             if best is None:
#                 return None
#             det_score = float(getattr(best, "det_score", 0.0))
#             if det_score < FACE_MIN_SCORE:
#                 return None
#             emb = extract_face_embedding(best)
#             return _as_512f(emb)

#         e0 = _best_face(img_proc)
#         if on_work_done:
#             on_work_done(1)

#         e1 = None
#         if FACE_TTA_FLIP:
#             try:
#                 e1 = _best_face(cv2.flip(img_proc, 1))
#             except Exception:
#                 e1 = None
#             if on_work_done:
#                 on_work_done(1)

#         feats = [e for e in (e0, e1) if e is not None]
#         if feats:
#             fused = l2_normalize(np.mean(np.stack(feats, axis=0), axis=0))
#             vectors.append(fused)

#         if on_file_done:
#             on_file_done(1)

#     if not vectors:
#         logger.warning("No valid FACE embeddings for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     centroid = _mean_embed(vectors)
#     return (vectors if vectors else None), centroid


# def _compute_back_body_bank_and_centroid_from_gallery(
#     member_id: int,
#     camera_id: int,
#     member_name: str,
#     on_file_done: Optional[Callable[[int], None]] = None,
#     on_work_done: Optional[Callable[[int], None]] = None,
# ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
#     root = _gallery_back_body_dir(member_id, camera_id, member_name)
#     imgs = _list_images_sampled(root, MAX_BACK_BODY_IMAGES, member_id, camera_id, "back_body")
#     if not imgs:
#         logger.warning("No BACK-BODY crops found for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
#     B = 64 if gpu else 32

#     seen_hashes: set[int] = set()

#     prepared_reid: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []
#     shape_vecs: List[Optional[np.ndarray]] = []
#     keep_mask: List[bool] = []
#     reid_ready: List[bool] = []

#     reid_units = (len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)) if reid_extractors else 0
#     per_image_units = 1 + int(reid_units)

#     for p in imgs:
#         img = _load_bgr(p)
#         if img is None or img.size == 0:
#             shape_vecs.append(None)
#             prepared_reid.append((None, None))
#             keep_mask.append(False)
#             reid_ready.append(False)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             if on_file_done:
#                 on_file_done(1)
#             continue

#         h, w = img.shape[:2]
#         if min(h, w) < 32 or _is_blurry(img):
#             shape_vecs.append(None)
#             prepared_reid.append((None, None))
#             keep_mask.append(False)
#             reid_ready.append(False)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             if on_file_done:
#                 on_file_done(1)
#             continue

#         hval = _ahash8(img)
#         if hval in seen_hashes:
#             shape_vecs.append(None)
#             prepared_reid.append((None, None))
#             keep_mask.append(False)
#             reid_ready.append(False)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             if on_file_done:
#                 on_file_done(1)
#             continue
#         seen_hashes.add(hval)

#         shp = _back_shape_embed(img)
#         shape_vecs.append(shp)
#         if on_work_done:
#             on_work_done(1)

#         rgb = None
#         rgb_flip = None
#         can_reid = False
#         if reid_extractors:
#             try:
#                 rgb = _prep_reid(img)
#                 rgb_flip = cv2.flip(rgb, 1) if BODY_TTA_FLIP else None
#                 can_reid = True
#             except Exception:
#                 can_reid = False

#         prepared_reid.append((rgb, rgb_flip))
#         reid_ready.append(bool(can_reid))

#         keep = (shp is not None) or bool(can_reid)
#         keep_mask.append(bool(keep))

#         if not keep:
#             if on_work_done and reid_units > 0:
#                 on_work_done(int(reid_units))
#             if on_file_done:
#                 on_file_done(1)
#             continue

#         if not can_reid and reid_units > 0:
#             if on_work_done:
#                 on_work_done(int(reid_units))

#     reid_vectors_per_image: List[Optional[np.ndarray]] = [None for _ in prepared_reid]

#     if reid_extractors:
#         per_model_embs: List[List[np.ndarray]] = [[] for _ in reid_extractors]

#         for midx, ext in enumerate(reid_extractors):
#             batch_imgs: List[np.ndarray] = []
#             idx_map: List[int] = []
#             for i, (orig, flip) in enumerate(prepared_reid):
#                 if not reid_ready[i] or orig is None:
#                     continue
#                 batch_imgs.append(orig)
#                 idx_map.append(i)
#                 if flip is not None:
#                     batch_imgs.append(flip)
#                     idx_map.append(i)

#             feats_norm: List[np.ndarray] = []
#             for start in range(0, len(batch_imgs), B):
#                 chunk = batch_imgs[start:start + B]
#                 try:
#                     with reid_lock, torch.inference_mode():
#                         feats = ext(chunk)
#                     for f in feats:
#                         f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
#                         f512 = _as_512f(f)
#                         feats_norm.append(f512 if f512 is not None else np.zeros((EXPECTED_EMBED_DIM,), np.float32))
#                 except Exception:
#                     logger.exception("TorchReID batch failed (model=%s) [back_body]", REID_MODELS[midx])
#                     feats_norm.extend([np.zeros((EXPECTED_EMBED_DIM,), np.float32) for _ in chunk])

#                 if on_work_done:
#                     on_work_done(len(chunk))

#             img_accum: Dict[int, List[np.ndarray]] = {}
#             for img_i, feat in zip(idx_map, feats_norm):
#                 img_accum.setdefault(img_i, []).append(feat)

#             fused_list: List[np.ndarray] = []
#             for i in range(len(prepared_reid)):
#                 vv = img_accum.get(i, [])
#                 fused_list.append(
#                     l2_normalize(np.mean(np.stack(vv, axis=0), axis=0)) if vv else np.zeros((EXPECTED_EMBED_DIM,), np.float32)
#                 )
#             per_model_embs[midx] = fused_list

#         for i in range(len(prepared_reid)):
#             if not reid_ready[i]:
#                 continue
#             parts = [per_model_embs[m][i] for m in range(len(reid_extractors))]
#             if parts:
#                 reid_vectors_per_image[i] = l2_normalize(np.mean(np.stack(parts, axis=0), axis=0))

#     fused_vectors: List[np.ndarray] = []
#     for i in range(len(prepared_reid)):
#         if not keep_mask[i]:
#             continue

#         v_reid = reid_vectors_per_image[i]
#         v_shape = shape_vecs[i] if i < len(shape_vecs) else None

#         parts2: List[np.ndarray] = []
#         if v_reid is not None:
#             parts2.append(v_reid * float(BACK_REID_WEIGHT))
#         if v_shape is not None:
#             parts2.append(v_shape * float(BACK_SHAPE_WEIGHT))

#         if parts2:
#             fused = l2_normalize(np.sum(np.stack(parts2, axis=0), axis=0))
#             fused_vectors.append(fused)

#         if on_file_done:
#             on_file_done(1)

#     if not fused_vectors:
#         logger.warning("No valid BACK-BODY embeddings for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     centroid = _mean_embed(fused_vectors)
#     return (fused_vectors if fused_vectors else None), centroid


# # ===================== DB update (FIX FK camera_id) =====================

# def _update_db_embeddings(
#     member_id: int,
#     camera_id: int,
#     body_emb: Optional[np.ndarray],
#     face_emb: Optional[np.ndarray],
#     back_body_emb: Optional[np.ndarray],
#     body_bank: Optional[List[np.ndarray]] = None,
#     face_bank: Optional[List[np.ndarray]] = None,
#     back_body_bank: Optional[List[np.ndarray]] = None,
# ) -> Dict[str, object]:
#     with db_session() as db:
#         try:
#             m = db.query(Member).filter(Member.id == int(member_id)).first()
#             if not m:
#                 logger.warning("Member ID %s not found. Skipping DB update.", member_id)
#                 return {"status": "error", "message": "member not found"}

#             # IMPORTANT: camera FK validation
#             if Camera is not None:
#                 cam = db.query(Camera).filter(Camera.id == int(camera_id)).first()
#                 if not cam:
#                     logger.error("Camera ID %s not found in cameras table. FK would fail.", camera_id)
#                     db.rollback()
#                     return {"status": "error", "message": "camera not found", "member_id": int(member_id), "camera_id": int(camera_id)}

#             rec = (
#                 db.query(MemberEmbedding)
#                 .filter(MemberEmbedding.member_id == int(member_id), MemberEmbedding.camera_id == int(camera_id))
#                 .first()
#             )
#             if rec is None:
#                 rec = MemberEmbedding(member_id=int(member_id), camera_id=int(camera_id))
#                 db.add(rec)

#             changed = False

#             if body_emb is not None:
#                 rec.body_embedding = body_emb.tolist()
#                 changed = True
#             if face_emb is not None:
#                 rec.face_embedding = face_emb.tolist()
#                 changed = True
#             if back_body_emb is not None:
#                 rec.back_body_embedding = back_body_emb.tolist()
#                 changed = True

#             if body_bank is not None:
#                 rec.body_embeddings_raw = _pack_raw_bank(body_bank)
#                 changed = True
#             if face_bank is not None:
#                 rec.face_embeddings_raw = _pack_raw_bank(face_bank)
#                 changed = True
#             if back_body_bank is not None:
#                 rec.back_body_embeddings_raw = _pack_raw_bank(back_body_bank)
#                 changed = True

#             if changed:
#                 rec.last_embedding_update_ts = datetime.now(timezone.utc)
#                 db.commit()
#                 logger.info(
#                     "DB updated: member=%s cam=%s body=%s face=%s back=%s body_raw=%s face_raw=%s back_raw=%s",
#                     member_id,
#                     camera_id,
#                     body_emb is not None,
#                     face_emb is not None,
#                     back_body_emb is not None,
#                     body_bank is not None,
#                     face_bank is not None,
#                     back_body_bank is not None,
#                 )
#                 return {
#                     "status": "ok",
#                     "member_id": int(member_id),
#                     "camera_id": int(camera_id),
#                     "body": body_emb is not None,
#                     "face": face_emb is not None,
#                     "back_body": back_body_emb is not None,
#                     "body_raw": body_bank is not None,
#                     "face_raw": face_bank is not None,
#                     "back_body_raw": back_body_bank is not None,
#                 }

#             db.rollback()
#             logger.info("No embeddings to update for member=%s cam=%s", member_id, camera_id)
#             return {"status": "no_embeddings", "member_id": int(member_id), "camera_id": int(camera_id)}

#         except Exception:
#             db.rollback()
#             logger.exception("DB error on update")
#             return {"status": "error", "message": "db error", "member_id": int(member_id), "camera_id": int(camera_id)}


# # ===================== Background extraction workers =====================

# def _extract_worker(member_id: int, member_name: str, camera_id: int):
#     try:
#         _progress_reset(member_id, member_name, camera_id)
#         _progress_set(member_id, camera_id, stage="Scanning", message="Counting images...")
#         total_body, total_face, total_back = _count_images(member_id, camera_id, member_name)

#         if total_body + total_face + total_back == 0:
#             _progress_set(
#                 member_id,
#                 camera_id,
#                 total_body=0,
#                 total_face=0,
#                 total_back_body=0,
#                 stage="error",
#                 message="No images to process",
#             )
#             return

#         _progress_set(
#             member_id,
#             camera_id,
#             total_body=total_body,
#             total_face=total_face,
#             total_back_body=total_back,
#             done_body=0,
#             done_face=0,
#             done_back_body=0,
#             percent=0,
#         )

#         body_bank = None
#         body_cent = None
#         if total_body > 0 and reid_extractors:
#             units_per = len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)
#             _progress_set(
#                 member_id,
#                 camera_id,
#                 stage="Body embeddings",
#                 message=f"Processing body ({total_body})",
#                 work_total=int(total_body * units_per),
#                 work_done=0,
#             )

#             def on_body_files(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, done_body=int(st.get("done_body", 0)) + int(n))

#             def on_body_work(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, work_done=int(st.get("work_done", 0)) + int(n))

#             body_bank, body_cent = _compute_body_bank_and_centroid_from_gallery(
#                 member_id, camera_id, member_name, on_file_done=on_body_files, on_work_done=on_body_work
#             )
#         else:
#             _progress_set(member_id, camera_id, message="Skipping body (no images or model unavailable)")

#         face_bank = None
#         face_cent = None
#         if total_face > 0 and face_app is not None:
#             units_per = (2 if FACE_TTA_FLIP else 1)
#             _progress_set(
#                 member_id,
#                 camera_id,
#                 stage="Face embeddings",
#                 message=f"Processing face ({total_face})",
#                 work_total=int(total_face * units_per),
#                 work_done=0,
#             )

#             def on_face_files(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, done_face=int(st.get("done_face", 0)) + int(n))

#             def on_face_work(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, work_done=int(st.get("work_done", 0)) + int(n))

#             face_bank, face_cent = _compute_face_bank_and_centroid_from_gallery(
#                 member_id, camera_id, member_name, on_file_done=on_face_files, on_work_done=on_face_work
#             )
#         else:
#             _progress_set(member_id, camera_id, message="Skipping face (no images or model unavailable)")

#         back_bank = None
#         back_cent = None
#         if total_back > 0:
#             reid_units = (len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)) if reid_extractors else 0
#             units_per = 1 + reid_units
#             _progress_set(
#                 member_id,
#                 camera_id,
#                 stage="Back body embeddings",
#                 message=f"Processing back-body ({total_back})",
#                 work_total=int(total_back * units_per),
#                 work_done=0,
#             )

#             def on_back_files(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, done_back_body=int(st.get("done_back_body", 0)) + int(n))

#             def on_back_work(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, work_done=int(st.get("work_done", 0)) + int(n))

#             back_bank, back_cent = _compute_back_body_bank_and_centroid_from_gallery(
#                 member_id, camera_id, member_name, on_file_done=on_back_files, on_work_done=on_back_work
#             )
#         else:
#             _progress_set(member_id, camera_id, message="Skipping back-body (no images)")

#         # ---- Validation: ensure all six embedding inputs are present before DB update ----
#         required_pairs = [
#             ("Body embeddings", body_cent),
#             ("Face embeddings", face_cent),
#             ("Back body embeddings", back_cent),
#             ("Body bank", body_bank),
#             ("Face bank", face_bank),
#             ("Back body bank", back_bank),
#         ]
#         missing = [name for name, val in required_pairs if val is None]
#         if missing:
#             missing_list = ", ".join(missing)
#             friendly = (
#                 f"Missing required embedding data: {missing_list}. "
#                 "Please provide the missing embedding to proceed."
#             )
#             logger.error("Extract worker validation failed: %s (member=%s cam=%s)", missing_list, member_id, camera_id)
#             _progress_set(
#                 member_id,
#                 camera_id,
#                 stage="error",
#                 message=friendly,
#                 percent=100,
#             )
#             return

#         # ---- call the DB update as before ----
#         res = _update_db_embeddings(
#             member_id,
#             camera_id,
#             body_cent,
#             face_cent,
#             back_cent,
#             body_bank=body_bank,
#             face_bank=face_bank,
#             back_body_bank=back_bank,
#         )
#         status = str(res.get("status", "unknown"))
#         _progress_set(member_id, camera_id, stage="Done", message=("Embeddings saved" if status == "ok" else status), percent=100)

#     except Exception as e:
#         logger.exception("extract worker failed")
#         _progress_set(member_id, camera_id, stage="error", message=f"error: {e}")

# def _extract_worker_multi(member_id: int, member_name: str, camera_ids: List[int]):
#     camera_ids = [int(c) for c in camera_ids if isinstance(c, int) or str(c).isdigit()]
#     if not camera_ids:
#         return

#     _progress_reset(member_id, member_name, OVERALL_CAMERA_ID)
#     _progress_set(member_id, OVERALL_CAMERA_ID, stage="Starting", message=f"Processing {len(camera_ids)} camera(s)", percent=0)

#     done = 0
#     total = max(1, len(camera_ids))

#     for cid in camera_ids:
#         _progress_set(member_id, OVERALL_CAMERA_ID, stage="running", message=f"Extracting camera {cid} ({done+1}/{total})")
#         _extract_worker(member_id, member_name, cid)
#         done += 1
#         _progress_set(member_id, OVERALL_CAMERA_ID, percent=int(round((done / total) * 100)))

#     _progress_set(member_id, OVERALL_CAMERA_ID, stage="Done", message="All cameras processed", percent=100)


# def extract_embeddings_async(member_id: int, camera_ids: Optional[Any] = None) -> Dict[str, object]:
#     if not reid_extractors or (USE_FACE and face_app is None):
#         init_models()

#     cam_ids = _normalize_camera_ids(camera_ids)
#     if cam_ids is None:
#         cam_ids = _db_get_camera_ids(camera_ids=None, only_active=True)
#         if not cam_ids:
#             cam_ids = sorted(_camera_streams_map(RTSP_STREAMS).keys())
#     else:
#         cam_ids = [int(x) for x in cam_ids]

#     valid_ids, missing = _validate_camera_ids_exist(cam_ids)
#     if missing:
#         return {
#             "status": "error",
#             "message": "Some camera_ids do not exist in cameras tablesssssssss",
#             "missing_camera_ids": missing,
#             "valid_camera_ids": valid_ids,
#         }
#     if not valid_ids:
#         return {"status": "error", "message": "No valid camera_ids to extract", "requested_camera_ids": cam_ids}

#     with db_session() as db:
#         try:
#             member = db.query(Member).filter(Member.id == int(member_id)).first()
#             if not member:
#                 return {"status": "error", "message": f"member id {member_id} not found"}
#             member_name = _get_member_display_name(member, fallback=f"member_{member_id}")
#         except Exception:
#             db.rollback()
#             logger.exception("DB error during member lookup.")
#             return {"status": "error", "message": "DB error"}

#     t = threading.Thread(
#         target=_extract_worker_multi,
#         args=(int(member_id), member_name, valid_ids),
#         daemon=True,
#         name=f"extract-{member_id}",
#     )
#     t.start()
#     return {"status": "started", "member_id": int(member_id), "member_name": member_name, "camera_ids": valid_ids}


# def extract_embeddings_sync(member_id: int, camera_ids: Optional[Any] = None) -> Dict[str, object]:
#     if not reid_extractors or (USE_FACE and face_app is None):
#         init_models()

#     cam_ids = _normalize_camera_ids(camera_ids)
#     if cam_ids is None:
#         cam_ids = _db_get_camera_ids(camera_ids=None, only_active=True)
#         if not cam_ids:
#             cam_ids = sorted(_camera_streams_map(RTSP_STREAMS).keys())
#     else:
#         cam_ids = [int(x) for x in cam_ids]

#     valid_ids, missing = _validate_camera_ids_exist(cam_ids)
#     if missing:
#         return {
#             "status": "error",
#             "message": "Some camera_ids do not exist in cameras table",
#             "missing_camera_ids": missing,
#             "valid_camera_ids": valid_ids,
#         }
#     if not valid_ids:
#         return {"status": "error", "message": "No valid camera_ids to extract", "requested_camera_ids": cam_ids}

#     with db_session() as db:
#         try:
#             member = db.query(Member).filter(Member.id == int(member_id)).first()
#             if not member:
#                 return {"status": "error", "message": f"member id {member_id} not found"}
#             member_name = _get_member_display_name(member, fallback=f"member_{member_id}")
#         except Exception:
#             db.rollback()
#             logger.exception("DB error during member lookup.")
#             return {"status": "error", "message": "DB error"}

#     results: Dict[int, Dict[str, object]] = {}
#     for cid in valid_ids:
#         _extract_worker(int(member_id), member_name, int(cid))
#         results[int(cid)] = get_progress_for_member(int(member_id), int(cid))

#     return {"status": "Done", "member_id": int(member_id), "member_name": member_name, "cameras": results}


# # ===================== Control API =====================

# def start_extraction(member_id: int, camera_ids: Optional[Any] = None, show_viewer: bool = True) -> dict:
#     """
#     FIX:
#       - camera_ids refer to cameras.id (DB FK)
#       - RTSP URL built from Camera.ip_address using RTSP_URL_TEMPLATE
#     """
#     global is_running, current_member_id, current_member_name, current_camera_ids, current_camera_streams, viewer_thread

#     if is_running:
#         return {
#             "status": "ok",
#             "message": "Already running",
#             "num_cams": len(current_camera_ids),
#             "member_id": current_member_id,
#             "member_name": current_member_name,
#             "camera_ids": list(current_camera_ids),
#         }

#     selected = _normalize_camera_ids(camera_ids)
#     streams_map = _get_streams_map(selected)

#     if not streams_map:
#         avail = sorted(_get_streams_map(None).keys())
#         return {
#             "status": "error",
#             "message": "No valid cameras/RTSP streams configured",
#             "requested_camera_ids": selected,
#             "available_camera_ids": avail,
#             "hint": "Populate cameras table and set RTSP_URL_TEMPLATE like your working RTSP URL (replace IP with {ip}).",
#         }

#     if selected is None:
#         selected_ids = sorted(streams_map.keys())
#     else:
#         missing = [int(cid) for cid in selected if int(cid) not in streams_map]
#         if missing:
#             avail = sorted(_get_streams_map(None).keys())
#             return {
#                 "status": "error",
#                 "message": "Some requested camera_ids were not found (or are inactive).",
#                 "missing_camera_ids": missing,
#                 "available_camera_ids": avail,
#                 "hint": "If you need inactive cameras too, set ONLY_ACTIVE_CAMERAS=0",
#             }
#         selected_ids = [int(cid) for cid in selected]

#     with db_session() as db:
#         try:
#             member = db.query(Member).filter(Member.id == int(member_id)).first()
#             if not member:
#                 return {"status": "error", "message": f"member id {member_id} not found"}
#             member_name = _get_member_display_name(member, fallback=f"member_{member_id}")
#         except Exception:
#             db.rollback()
#             logger.exception("DB error during member lookup (start_extraction).")
#             return {"status": "error", "message": "DB error"}

#     if yolo_model is None or (USE_FACE and face_app is None) or (not reid_extractors):
#         init_models()

#     stop_event.clear()
#     with frames_lock:
#         latest_frames.clear()
#     for d in (capture_threads, extract_threads, cap_queues):
#         d.clear()

#     current_member_id = int(member_id)
#     current_member_name = member_name
#     current_camera_ids = list(selected_ids)
#     current_camera_streams = {cid: streams_map[cid] for cid in selected_ids}

#     _progress_reset(int(member_id), member_name, OVERALL_CAMERA_ID)
#     _progress_set(int(member_id), OVERALL_CAMERA_ID, stage="Starting", message="initializing", percent=0)

#     start_workers_if_needed()

#     for cid in selected_ids:
#         rtsp_url = streams_map[cid]
#         cap_queues[cid] = Queue(maxsize=CAP_QUEUE_MAX)

#         t_cap = threading.Thread(target=capture_thread_fn, name=f"cap_cam_{cid}", args=(cid, rtsp_url), daemon=True)
#         t_ext = threading.Thread(target=embedding_loop_for_cam, name=f"ext_cam_{cid}", args=(cid, rtsp_url), daemon=True)

#         capture_threads[cid] = t_cap
#         extract_threads[cid] = t_ext

#         t_cap.start()
#         t_ext.start()

#     if show_viewer and (viewer_thread is None or not viewer_thread.is_alive()):
#         viewer_thread = threading.Thread(target=viewer_loop, name="viewer", daemon=True)
#         viewer_thread.start()

#     is_running = True
#     _progress_set(int(member_id), OVERALL_CAMERA_ID, stage="running", message="processing", percent=0)
#     logger.info("Extraction started: member_id=%s name=%s camera_ids=%s", member_id, member_name, selected_ids)

#     return {
#         "status": "ok",
#         "message": "started",
#         "num_cams": len(selected_ids),
#         "member_id": int(member_id),
#         "member_name": member_name,
#         "camera_ids": selected_ids,
#         "rtsp_streams": {cid: streams_map[cid] for cid in selected_ids},
#     }


# def viewer_loop():
#     logger.info("[Viewer] started")
#     window_name = "Multi-RTSP Viewer"
#     window_created = False
#     try:
#         while not stop_event.is_set():
#             with frames_lock:
#                 frames = [latest_frames[idx] for idx in sorted(latest_frames.keys()) if latest_frames.get(idx) is not None]

#             if not frames:
#                 time.sleep(0.01)
#                 continue

#             if not window_created:
#                 try:
#                     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#                     window_created = True
#                 except Exception:
#                     pass

#             min_h = min(f.shape[0] for f in frames)
#             resized = []
#             for f in frames:
#                 h, w = f.shape[:2]
#                 if h != min_h:
#                     new_w = int(w * (min_h / h))
#                     f = cv2.resize(f, (new_w, min_h))
#                 resized.append(f)

#             try:
#                 combined = np.concatenate(resized, axis=1)
#             except Exception:
#                 time.sleep(0.01)
#                 continue

#             try: 
#                 cv2.imshow(window_name, combined)
#             except Exception:
#                 window_created = False
#                 time.sleep(0.01)
#                 continue

#             key = cv2.waitKey(1) & 0xFF
#             if key in (ord("q"), 27):
#                 logger.info("[Viewer] key pressed, stopping...")
#                 stop_event.set()
#                 break

#             time.sleep(VIEWER_SLEEP_MS / 1000.0)
#     finally:
#         try:
#             cv2.destroyAllWindows()
#         except Exception:
#             pass
#         logger.info("[Viewer] stopped")


# def stop_extraction(reason: str = "user") -> dict:
#     global is_running, current_member_id, current_member_name, current_camera_ids, current_camera_streams
#     try:
#         mid_snapshot = current_member_id
#         mname_snapshot = current_member_name
#         cams_snapshot = list(current_camera_ids)

#         stop_event.set()

#         for d in (capture_threads, extract_threads):
#             for _cid, th in list(d.items()):
#                 try:
#                     th.join(timeout=1.5)
#                 except Exception:
#                     pass

#         try:
#             if io_queue is not None:
#                 io_queue.join()
#         except Exception:
#             pass

#         with frames_lock:
#             latest_frames.clear()
#         capture_threads.clear()
#         extract_threads.clear()
#         cap_queues.clear()

#         is_running = False
#         logger.info("Extraction stopped (%s).", reason)

#         if mid_snapshot is not None and cams_snapshot:
#             if not reid_extractors or (USE_FACE and face_app is None):
#                 init_models()
#             threading.Thread(
#                 target=_extract_worker_multi,
#                 args=(int(mid_snapshot), str(mname_snapshot or "unknown"), cams_snapshot),
#                 name=f"auto-extract-{mid_snapshot}",
#                 daemon=True,
#             ).start()
#             logger.info("Auto-extract triggered for member_id=%s cameras=%s after stop.", mid_snapshot, cams_snapshot)

#         return {"status": "ok", "message": f"Stopped capturing ", "member_id": mid_snapshot, "camera_ids": cams_snapshot}
#     finally:
#         current_member_id = None
#         current_member_name = None
#         current_camera_ids = []
#         current_camera_streams = {}


# def remove_embeddings(member_id: int, camera_id: Optional[int] = None) -> Dict[str, object]:
#     with db_session() as db:
#         try:
#             q = db.query(MemberEmbedding).filter(MemberEmbedding.member_id == int(member_id))
#             if camera_id is not None:
#                 q = q.filter(MemberEmbedding.camera_id == int(camera_id))

#             rows = q.all()
#             if not rows:
#                 return {"status": "error", "message": "no embeddings rows found", "member_id": int(member_id), "camera_id": camera_id}

#             for rec in rows:
#                 rec.body_embedding = None
#                 rec.face_embedding = None
#                 rec.back_body_embedding = None
#                 rec.body_embeddings_raw = None
#                 rec.face_embeddings_raw = None
#                 rec.back_body_embeddings_raw = None
#                 rec.last_embedding_update_ts = datetime.now(timezone.utc)

#             db.commit()
#             logger.warning("Embeddings removed for member_id=%s camera_id=%s (rows=%d)", member_id, camera_id, len(rows))
#             return {"status": "ok", "member_id": int(member_id), "camera_id": camera_id, "rows": len(rows)}
#         except Exception:
#             db.rollback()
#             logger.exception("remove embeddings: DB error")
#             return {"status": "error", "message": "DB error", "member_id": int(member_id), "camera_id": camera_id}


# def get_status() -> Dict[str, object]:
#     streams_map = _get_streams_map(None)
#     return {
#         "running": bool(is_running),
#         "configured_num_cams": len(streams_map),
#         "configured_camera_ids": sorted(streams_map.keys()),
#         "running_camera_ids": list(current_camera_ids),
#         "rtsp_streams": streams_map,
#         "member_id": current_member_id,
#         "member_name": current_member_name,
#     }

# ----------------------------------------------------------------------------------------------------------------------------------
# # app/services/embedding_service.py
# from __future__ import annotations

# import csv
# import sys
# import os 
# from dotenv import load_dotenv
# from pathlib import Path
# import time
# import threading
# from pathlib import Path
# from datetime import datetime, timezone
# from typing import Optional, List, Dict, Tuple, Callable, Any
# from queue import Queue, Empty
# import logging
# import shutil
# import random
# import io
# import gzip
# import zlib
# import re
# import inspect
# from contextlib import contextmanager
# from typing import Generator

# import cv2
# import numpy as np
# import torch

# from app.core.constants import (
#     RTSP_STREAMS,
#     YOLO_WEIGHTS,
#     DEVICE,
#     CONF_THRES,
#     IOU_THRES,
#     EMB_CSV,
#     CROPS_ROOT,
# )
# from app.db.session import get_db, engine  # engine kept for compatibility
# from app.db.models import Member, MemberEmbedding

# try:
#     from app.db.models import Camera
# except Exception:
#     Camera = None  # type: ignore


# BASE_DIR = Path(__file__).resolve().parents[2]
# load_dotenv(BASE_DIR / ".env") 

# # ===================== DB Session Wrapper =====================

# @contextmanager
# def db_session():
#     """
#     Works with both styles:
#       - get_db() returns Session (old style)
#       - get_db() yields Session (FastAPI dependency style -> generator)
#     """
#     db_obj: Any = get_db()

#     # FastAPI dependency style: generator that yields a Session
#     if inspect.isgenerator(db_obj):
#         gen: Generator = db_obj  # type: ignore
#         try:
#             db = next(gen)  # actual Session
#         except StopIteration:
#             raise RuntimeError("get_db() did not yield a DB session")
#         try:
#             yield db
#         finally:
#             try:
#                 gen.close()
#             except Exception:
#                 pass
#         return

#     # Old style: get_db() returned an actual Session
#     db = db_obj
#     try:
#         yield db
#     finally:
#         try:
#             db.close()
#         except Exception:
#             pass


# # ===================== Runtime perf knobs =====================

# try:
#     torch.backends.cudnn.benchmark = True
# except Exception:
#     pass
# try:
#     if hasattr(torch, "set_float32_matmul_precision"):
#         torch.set_float32_matmul_precision("high")
# except Exception:
#     pass
# try:
#     cv2.setNumThreads(1)
# except Exception:
#     pass


# # ===================== Optional libs =====================

# try:
#     from ultralytics import YOLO
# except Exception:
#     YOLO = None

# try:
#     from torchreid.utils import FeatureExtractor as TorchreidExtractor
# except Exception as e:
#     print("-----------------TorchReID import error:", e)
#     TorchreidExtractor = None

# try:
#     from insightface.app import FaceAnalysis
#     INSIGHT_OK = True
# except Exception:
#     FaceAnalysis = None
#     INSIGHT_OK = False

# try:
#     import onnxruntime as ort
# except Exception:
#     ort = None


# # ===================== Config =====================

# EXPECTED_EMBED_DIM = 512

# RAW_STORE_LIMIT = int(os.getenv("RAW_STORE_LIMIT", "256"))

# USE_FACE = True
# FACE_MODEL = "buffalo_l"
# FACE_DET_SIZE = (1280, 1280)
# FACE_PROVIDER = "auto"
# FACE_MIN_SCORE = 0.5
# FACE_MIN_SIZE = 32
# FACE_IOU_LINK = 0.05
# FACE_OVER_FACE_LINK = 0.60
# FACE_EVERY = int(os.getenv("FACE_EVERY", "1"))

# YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))

# CAP_QUEUE_MAX = int(os.getenv("CAP_QUEUE_MAX", "2"))
# IO_QUEUE_MAX = int(os.getenv("IO_QUEUE_MAX", "1024"))

# VIEWER_SLEEP_MS = int(os.getenv("VIEWER_SLEEP_MS", "5"))

# REID_MODELS = ["osnet_x1_0", "osnet_x0_25"]
# BODY_TTA_FLIP = True
# FACE_TTA_FLIP = True

# BACK_BODY_GALLERY_NAME = "gallery_body"
# BACK_HEAD_CUT_RATIO = float(os.getenv("BACK_HEAD_CUT_RATIO", "0.0"))
# BACK_SHAPE_SEED = int(os.getenv("BACK_SHAPE_SEED", "1337"))

# BACK_REID_WEIGHT = float(os.getenv("BACK_REID_WEIGHT", "0.65"))
# BACK_SHAPE_WEIGHT = float(os.getenv("BACK_SHAPE_WEIGHT", "0.35"))

# BACK_HOG_WIN = (64, 128)
# BACK_HOG_BLOCK = (16, 16)
# BACK_HOG_STRIDE = (8, 8)
# BACK_HOG_CELL = (8, 8)
# BACK_HOG_BINS = 9

# SAVE_MIN_INTERVAL_MS = int(os.getenv("SAVE_MIN_INTERVAL_MS", "200"))
# SAVE_MAX_DETS_PER_FRAME = int(os.getenv("SAVE_MAX_DETS_PER_FRAME", "3"))
# JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

# YOLO_REQ_QUEUE_MAX = int(os.getenv("YOLO_REQ_QUEUE_MAX", "64"))
# YOLO_BATCH_MAX = int(os.getenv("YOLO_BATCH_MAX", "8"))
# YOLO_BATCH_WAIT_MS = int(os.getenv("YOLO_BATCH_WAIT_MS", "10"))
# YOLO_RESP_TIMEOUT_S = float(os.getenv("YOLO_RESP_TIMEOUT_S", "0.7"))

# MAX_BODY_IMAGES = int(os.getenv("MAX_BODY_IMAGES", "600"))
# MAX_FACE_IMAGES = int(os.getenv("MAX_FACE_IMAGES", "600"))
# MAX_BACK_BODY_IMAGES = int(os.getenv("MAX_BACK_BODY_IMAGES", "600"))
# GALLERY_SAMPLE_SEED = int(os.getenv("GALLERY_SAMPLE_SEED", "1337"))

# # Debug aids (default off)
# DEBUG_SAVE_NO_DETS_EVERY_SEC = float(os.getenv("DEBUG_SAVE_NO_DETS_EVERY_SEC", "0"))  # 0 disables
# DEBUG_LOG_NO_DET_EVERY = int(os.getenv("DEBUG_LOG_NO_DET_EVERY", "150"))

# # ===================== DB Camera -> RTSP (CRITICAL FIX) =====================

# USE_DB_CAMERA_CONFIG = os.getenv("USE_DB_CAMERA_CONFIG", "1").lower() in ("1", "true", "yes", "y", "on")
# ONLY_ACTIVE_CAMERAS = os.getenv("ONLY_ACTIVE_CAMERAS", "1").lower() in ("1", "true", "yes", "y", "on")

# # Recommended:
# #   RTSP_URL_TEMPLATE="rtsp://user:pass@{ip}:554/Streaming/Channels/101"
# RTSP_URL_TEMPLATE = os.getenv("RTSP_URL_TEMPLATE", "").strip()

# # Optional non-template builder:
# RTSP_USERNAME = os.getenv("RTSP_USERNAME", os.getenv("RTSP_USER", "")).strip()
# RTSP_PASSWORD = os.getenv("RTSP_PASSWORD", os.getenv("RTSP_PASS", "")).strip()
# RTSP_PORT = os.getenv("RTSP_PORT", "554").strip()
# RTSP_PATH = os.getenv("RTSP_PATH", "").strip()
# RTSP_SCHEME = os.getenv("RTSP_SCHEME", "rtsp").strip() or "rtsp"
# RTSP_STREAM = os.getenv("RTSP_STREAM", "").strip() or ""


# # ===================== Logging =====================

# logger = logging.getLogger("app.embedding_service")
# if not logger.handlers:
#     ch = logging.StreamHandler()
#     ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
#     logger.addHandler(ch)
# logger.setLevel(logging.INFO)


# # ===================== Globals =====================

# yolo_model: YOLO | None = None
# reid_extractor: TorchreidExtractor | None = None
# reid_extractors: List[TorchreidExtractor] = []
# face_app: FaceAnalysis | None = None

# reid_lock = threading.Lock()
# csv_lock = threading.Lock()
# frames_lock = threading.Lock()

# latest_frames: Dict[int, np.ndarray] = {}

# capture_threads: Dict[int, threading.Thread] = {}
# extract_threads: Dict[int, threading.Thread] = {}
# cap_queues: Dict[int, Queue] = {}

# io_thread: Optional[threading.Thread] = None
# io_queue: Queue | None = None

# yolo_thread: Optional[threading.Thread] = None
# yolo_req_queue: Queue | None = None

# viewer_thread: Optional[threading.Thread] = None
# stop_event = threading.Event()
# is_running = False

# current_member_id: Optional[int] = None
# current_member_name: Optional[str] = None
# current_camera_ids: List[int] = []
# current_camera_streams: Dict[int, str] = {}

# progress_lock = threading.Lock()
# progress_state: Dict[Tuple[int, int], Dict[str, object]] = {}

# _back_hog = None
# _back_proj = None
# _back_proj_in_dim = None

# OVERALL_CAMERA_ID = -1


# # ===================== Cropping =====================

# def crop_with_padding(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
#     H, W = frame.shape[:2]
#     x1, y1, x2, y2 = bbox

#     x1 = max(0, min(W - 1, int(x1)))
#     y1 = max(0, min(H - 1, int(y1)))
#     x2 = max(0, min(W, int(x2)))
#     y2 = max(0, min(H, int(y2)))

#     if x2 <= x1 or y2 <= y1:
#         return np.zeros((0, 0, 3), dtype=np.uint8)

#     return frame[y1:y2, x1:x2].copy()


# def crop_back_body_no_face(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
#     x1, y1, x2, y2 = bbox
#     h = max(0, int(y2) - int(y1))
#     if h <= 0:
#         return np.zeros((0, 0, 3), dtype=np.uint8)
#     y1b = int(y1 + h * float(max(0.0, min(0.45, BACK_HEAD_CUT_RATIO))))
#     if y1b >= y2 - 2:
#         return np.zeros((0, 0, 3), dtype=np.uint8)
#     return crop_with_padding(frame, (x1, y1b, x2, y2))


# # ===================== IO Worker =====================

# def start_workers_if_needed() -> None:
#     global io_thread, io_queue, yolo_thread, yolo_req_queue

#     if io_queue is None:
#         io_queue = Queue(maxsize=IO_QUEUE_MAX)

#     if io_thread is None or not io_thread.is_alive():
#         io_thread = threading.Thread(target=io_worker, name="io_worker", daemon=True)
#         io_thread.start()

#     if yolo_req_queue is None:
#         yolo_req_queue = Queue(maxsize=YOLO_REQ_QUEUE_MAX)

#     if yolo_thread is None or not yolo_thread.is_alive():
#         if yolo_model is not None:
#             yolo_thread = threading.Thread(target=yolo_worker, name="yolo_worker", daemon=True)
#             yolo_thread.start()


# def io_worker() -> None:
#     ensure_csv_header()
#     csv_path = Path(EMB_CSV)
#     _ensure_dir(csv_path.parent if csv_path.parent.as_posix() not in (".", "") else Path("."))

#     f = None
#     writer = None
#     flush_every = 128
#     pending = 0
#     last_flush = time.time()

#     try:
#         f = open(csv_path, "a", newline="", encoding="utf-8")
#         writer = csv.writer(f)

#         while True:
#             if stop_event.is_set() and (io_queue is None or io_queue.empty()):
#                 break

#             try:
#                 job = io_queue.get(timeout=0.1)  # type: ignore
#             except Empty:
#                 if pending and (time.time() - last_flush) > 1.0 and f is not None:
#                     try:
#                         f.flush()
#                     except Exception:
#                         pass
#                     pending = 0
#                     last_flush = time.time()
#                 continue
#             except Exception:
#                 break

#             try:
#                 crop_path: Path = job["crop_path"]
#                 _ensure_dir(crop_path.parent)

#                 img = job["image"]
#                 ok = False
#                 if str(crop_path).lower().endswith((".jpg", ".jpeg")):
#                     ok = bool(cv2.imwrite(str(crop_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]))
#                 else:
#                     ok = bool(cv2.imwrite(str(crop_path), img))

#                 if not ok:
#                     logger.error("cv2.imwrite failed: %s", crop_path)

#                 if writer is not None:
#                     writer.writerow(job["csv_row"])
#                     pending += 1

#                 if f is not None and (pending >= flush_every or (time.time() - last_flush) > 1.0):
#                     try:
#                         f.flush()
#                     except Exception:
#                         pass
#                     pending = 0
#                     last_flush = time.time()
#             except Exception:
#                 logger.exception("IO worker error")
#             finally:
#                 try:
#                     io_queue.task_done()  # type: ignore
#                 except Exception:
#                     pass
#     finally:
#         try:
#             if f is not None:
#                 f.flush()
#                 f.close()
#         except Exception:
#             pass


# # ===================== YOLO Batch Worker =====================

# def yolo_worker() -> None:
#     if yolo_model is None:
#         return

#     gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
#     device = DEVICE if gpu else "cpu"

#     while True:
#         if stop_event.is_set() and (yolo_req_queue is None or yolo_req_queue.empty()):
#             break
#         try:
#             req0 = yolo_req_queue.get(timeout=0.1)  # type: ignore
#         except Empty:
#             continue
#         except Exception:
#             break

#         batch = [req0]
#         t0 = time.time()
#         while len(batch) < max(1, int(YOLO_BATCH_MAX)) and ((time.time() - t0) * 1000.0) < float(YOLO_BATCH_WAIT_MS):
#             try:
#                 batch.append(yolo_req_queue.get_nowait())  # type: ignore
#             except Empty:
#                 break
#             except Exception:
#                 break

#         frames = [b["frame"] for b in batch]
#         try:
#             with torch.inference_mode():
#                 results = yolo_model.predict(
#                     frames,
#                     conf=float(CONF_THRES),
#                     iou=float(IOU_THRES),
#                     imgsz=int(YOLO_IMGSZ),
#                     verbose=False,
#                     device=device,
#                 )
#         except Exception:
#             logger.exception("YOLO batch failed")
#             results = [None for _ in frames]

#         for req, res in zip(batch, results):
#             H, W = int(req["H"]), int(req["W"])
#             outs: List[tuple[float, float, float, float, float]] = []

#             try:
#                 boxes = getattr(res, "boxes", None) if res is not None else None
#                 if boxes is not None:
#                     xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
#                     confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
#                     cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
#                     keep = (cls == 0)
#                     xyxy, confs = xyxy[keep], confs[keep]

#                     for (x1, y1, x2, y2), c in zip(xyxy, confs):
#                         x1f = float(max(0, min(W - 1, x1)))
#                         y1f = float(max(0, min(H - 1, y1)))
#                         x2f = float(max(0, min(W - 1, x2)))
#                         y2f = float(max(0, min(H - 1, y2)))
#                         if (x2f - x1f) < 4 or (y2f - y1f) < 4:
#                             continue
#                         outs.append((x1f, y1f, x2f, y2f, float(c)))
#             except Exception:
#                 outs = []

#             try:
#                 respq: Queue = req["resp"]
#                 respq.put_nowait(outs)
#             except Exception:
#                 pass

#             try:
#                 yolo_req_queue.task_done()  # type: ignore
#             except Exception:
#                 pass


# def yolo_infer(frame: np.ndarray, H: int, W: int, timeout_s: float = YOLO_RESP_TIMEOUT_S) -> List[tuple[float, float, float, float, float]]:
#     if yolo_model is None or yolo_req_queue is None:
#         return []
#     resp = Queue(maxsize=1)
#     try:
#         yolo_req_queue.put_nowait({"frame": frame, "H": H, "W": W, "resp": resp})
#     except Exception:
#         return []
#     try:
#         return resp.get(timeout=float(timeout_s))
#     except Exception:
#         return []


# # ===================== Progress helpers =====================

# def _pkey(member_id: int, camera_id: int) -> Tuple[int, int]:
#     return (int(member_id), int(camera_id))


# def _progress_reset(member_id: int, member_name: str, camera_id: int) -> None:
#     with progress_lock:
#         progress_state[_pkey(member_id, camera_id)] = {
#             "member_id": int(member_id),
#             "member_name": member_name,
#             "camera_id": int(camera_id),
#             "stage": "idle",
#             "total_body": 0,
#             "total_face": 0,
#             "total_back_body": 0,
#             "done_body": 0,
#             "done_face": 0,
#             "done_back_body": 0,
#             "work_total": 0,
#             "work_done": 0,
#             "percent": 0,
#             "message": "waiting",
#         }


# def _progress_set(member_id: int, camera_id: int, **kv) -> None:
#     with progress_lock:
#         st = progress_state.get(_pkey(member_id, camera_id))
#         if not st:
#             st = {}
#             progress_state[_pkey(member_id, camera_id)] = st
#         st.update(kv)

#         tb = int(st.get("total_body", 0))
#         tf = int(st.get("total_face", 0))
#         tbb = int(st.get("total_back_body", 0))

#         db_ = int(st.get("done_body", 0))
#         df_ = int(st.get("done_face", 0))
#         dbb = int(st.get("done_back_body", 0))

#         db_ = max(0, min(db_, tb))
#         df_ = max(0, min(df_, tf))
#         dbb = max(0, min(dbb, tbb))

#         st["done_body"], st["done_face"], st["done_back_body"] = db_, df_, dbb

#         work_total = int(st.get("work_total", 0))
#         work_done = int(st.get("work_done", 0))
#         work_done = max(0, min(work_done, max(0, work_total)))
#         st["work_done"] = work_done

#         stage = str(st.get("stage", "")).lower()

#         if stage.startswith("embedding_") and work_total > 0:
#             pct = int(round((work_done / max(1, work_total)) * 100))
#         elif stage == "Body embeddings" and tb > 0:
#             pct = int(round((db_ / max(1, tb)) * 100))
#         elif stage == "Face embeddings" and tf > 0:
#             pct = int(round((df_ / max(1, tf)) * 100))
#         elif stage in ("Back body embeddings", "embedding_back") and tbb > 0:
#             pct = int(round((dbb / max(1, tbb)) * 100))
#         else:
#             tot = max(1, tb + tf + tbb)
#             pct = int(round(((db_ + df_ + dbb) / tot) * 100))

#         st["percent"] = max(0, min(100, pct))


# def _default_progress(member_id: int, camera_id: int) -> Dict[str, object]:
#     return {
#         "member_id": int(member_id),
#         "camera_id": int(camera_id),
#         "stage": "unknown",
#         "percent": 0,
#         "message": "no job",
#         "total_body": 0,
#         "total_face": 0,
#         "total_back_body": 0,
#         "done_body": 0,
#         "done_face": 0,
#         "done_back_body": 0,
#         "work_total": 0,
#         "work_done": 0,
#     }


# def get_progress_for_member(member_id: int, camera_id: Optional[int] = None) -> Dict[str, object]:
#     mid = int(member_id)
#     with progress_lock:
#         if camera_id is not None:
#             st = progress_state.get(_pkey(mid, int(camera_id)))
#             return dict(st) if st else _default_progress(mid, int(camera_id))

#         st_overall = progress_state.get(_pkey(mid, OVERALL_CAMERA_ID))
#         if st_overall:
#             return dict(st_overall)

#         cams = {cid: dict(st) for (m, cid), st in progress_state.items() if m == mid and cid != OVERALL_CAMERA_ID}
#         if len(cams) == 1:
#             return next(iter(cams.values()))
#         if cams:
#             return {"member_id": mid, "stage": "multi", "cameras": cams}
#         return _default_progress(mid, OVERALL_CAMERA_ID)


# # ===================== Utils =====================

# def l2_normalize(v: np.ndarray) -> np.ndarray:
#     v = np.asarray(v, dtype=np.float32).reshape(-1)
#     n = float(np.linalg.norm(v))
#     if n == 0.0 or not np.isfinite(n):
#         return v
#     return v / n


# def _as_512f(vec: np.ndarray | list | None) -> Optional[np.ndarray]:
#     if vec is None:
#         return None
#     try:
#         a = np.asarray(vec, dtype=np.float32).reshape(-1)
#         if a.size != EXPECTED_EMBED_DIM:
#             return None
#         if not np.isfinite(a).all():
#             return None
#         return l2_normalize(a)
#     except Exception:
#         return None


# def _pack_raw_bank(vectors: Optional[List[np.ndarray]]) -> Optional[bytes]:
#     if not vectors:
#         return None
#     cleaned: List[np.ndarray] = []
#     for v in vectors[: max(1, int(RAW_STORE_LIMIT))]:
#         vv = _as_512f(v)
#         if vv is not None:
#             cleaned.append(vv.astype(np.float32))
#     if not cleaned:
#         return None
#     arr = np.stack(cleaned, axis=0).astype(np.float32)
#     buf = io.BytesIO()
#     np.save(buf, arr, allow_pickle=False)
#     return gzip.compress(buf.getvalue(), compresslevel=6)


# def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
#     return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


# def _safe_slug(s: str, max_len: int = 64) -> str:
#     s = (s or "").strip()
#     s = re.sub(r"[^\w\-\.]+", "_", s, flags=re.UNICODE)
#     s = re.sub(r"_+", "_", s).strip("_")
#     if not s:
#         s = "member"
#     return s[:max_len]


# # ===================== Back-body shape embedding =====================

# def _init_back_shape_embedder() -> None:
#     global _back_hog
#     if _back_hog is None:
#         try:
#             _back_hog = cv2.HOGDescriptor(
#                 _winSize=BACK_HOG_WIN,
#                 _blockSize=BACK_HOG_BLOCK,
#                 _blockStride=BACK_HOG_STRIDE,
#                 _cellSize=BACK_HOG_CELL,
#                 _nbins=BACK_HOG_BINS,
#             )
#             logger.info("Back-body HOG ready (dim=%s)", int(_back_hog.getDescriptorSize()))
#         except Exception:
#             logger.exception("Failed to init HOG descriptor for back-body embeddings")
#             _back_hog = None


# def _get_back_proj(in_dim: int) -> Optional[np.ndarray]:
#     global _back_proj, _back_proj_in_dim
#     if in_dim <= 0:
#         return None
#     if _back_proj is None or _back_proj_in_dim != int(in_dim):
#         rng = np.random.RandomState(int(BACK_SHAPE_SEED))
#         mat = rng.normal(
#             loc=0.0,
#             scale=float(1.0 / max(1.0, np.sqrt(in_dim))),
#             size=(EXPECTED_EMBED_DIM, in_dim),
#         ).astype(np.float32)
#         _back_proj = mat
#         _back_proj_in_dim = int(in_dim)
#         logger.info("Back-body projection matrix built (seed=%s, in_dim=%s)", BACK_SHAPE_SEED, in_dim)
#     return _back_proj


# def _back_shape_embed(img_bgr: np.ndarray) -> Optional[np.ndarray]:
#     if img_bgr is None or img_bgr.size == 0:
#         return None
#     _init_back_shape_embedder()
#     if _back_hog is None:
#         return None
#     try:
#         g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#         g = cv2.resize(g, BACK_HOG_WIN, interpolation=cv2.INTER_AREA)
#         hog = _back_hog.compute(g)
#         if hog is None:
#             return None
#         hog = np.asarray(hog, dtype=np.float32).reshape(-1)
#         if hog.size <= 0 or not np.isfinite(hog).all():
#             return None
#         hn = float(np.linalg.norm(hog))
#         if hn > 0:
#             hog = hog / hn
#         proj = _get_back_proj(int(hog.size))
#         if proj is None:
#             return None
#         emb = proj @ hog
#         return _as_512f(emb)
#     except Exception:
#         return None


# # ===================== Geometry helpers =====================

# def iou_xyxy(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
#     inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
#     inter = iw * ih
#     a_area = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
#     b_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
#     denom = a_area + b_area - inter
#     return inter / denom if denom > 0 else 0.0


# def inter_over_face(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
#     ax1, ay1, ax2, ay2 = a
#     bx1, by1, bx2, by2 = b
#     inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
#     inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
#     iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
#     inter = iw * ih
#     face_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
#     return inter / face_area if face_area > 0 else 0.0


# def face_center_in(a: Tuple[int, int, int, int], b: Tuple[float, float, float, float]) -> bool:
#     fx1, fy1, fx2, fy2 = a
#     px1, py1, px2, py2 = b
#     cx = (fx1 + fx2) * 0.5
#     cy = (fy1 + fy2) * 0.5
#     return (px1 <= cx <= px2) and (py1 <= cy <= py2)


# def safe_iter_faces(obj: Any) -> List[Any]:
#     if obj is None:
#         return []
#     try:
#         return list(obj)
#     except TypeError:
#         return [obj]


# def extract_face_embedding(face: Any):
#     emb = getattr(face, "normed_embedding", None)
#     if emb is None:
#         emb = getattr(face, "embedding", None)
#     return emb


# def _cuda_ep_loadable() -> bool:
#     if ort is None:
#         return False
#     try:
#         if sys.platform.startswith("mac"):
#             return False
#         from pathlib import Path as _P
#         import ctypes as _ct
#         capi_dir = _P(ort.__file__).parent / "capi"
#         name = "onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so"
#         lib_path = capi_dir / name
#         if not lib_path.exists():
#             return False
#         _ct.CDLL(str(lib_path))
#         return True
#     except Exception:
#         return False


# # ===================== Models init =====================

# def init_face_engine(use_face: bool, device: str, face_model: str, det_w: int, det_h: int, face_provider: str):
#     if not use_face:
#         return None
#     if not INSIGHT_OK:
#         logger.warning("insightface not installed; face recognition disabled.")
#         return None
#     try:
#         is_cuda = ("cuda" in device.lower()) and torch.cuda.is_available()
#         cuda_ok = _cuda_ep_loadable()

#         providers = ["CPUExecutionProvider"]
#         if face_provider == "cuda":
#             if cuda_ok:
#                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
#         elif face_provider == "auto":
#             if is_cuda and cuda_ok:
#                 providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

#         app = FaceAnalysis(name=face_model, providers=providers)
#         ctx_id = 0 if providers[0].startswith("CUDA") else -1
#         try:
#             app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
#         except TypeError:
#             app.prepare(ctx_id=ctx_id)

#         logger.info("InsightFace ready (model=%s, providers=%s).", face_model, providers)

#         try:
#             dummy = np.zeros((max(8, det_h), max(8, det_w), 3), dtype=np.uint8)
#             app.get(dummy)
#         except Exception:
#             pass
#         return app
#     except Exception:
#         logger.exception("InsightFace init failed")
#         return None


# def init_models() -> None:
#     global yolo_model, reid_extractor, reid_extractors, face_app

#     gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
#     logger.info("init_models device=%s gpu_available=%s", DEVICE, torch.cuda.is_available())

#     # YOLO
#     if YOLO is not None:
#         try:
#             weights = YOLO_WEIGHTS
#             if not Path(weights).exists():
#                 logger.warning("%s not found, falling back to yolov8n.pt", weights)
#             yolo = YOLO(weights if Path(YOLO_WEIGHTS).exists() else "yolov8n.pt")
#             if gpu:
#                 try:
#                     yolo.to(DEVICE)
#                 except Exception:
#                     pass
#             try:
#                 yolo.fuse()
#             except Exception:
#                 pass
#             try:
#                 with torch.inference_mode():
#                     _dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
#                     yolo.predict(
#                         _dummy,
#                         device=DEVICE if gpu else "cpu",
#                         conf=0.25,
#                         iou=0.5,
#                         imgsz=YOLO_IMGSZ,
#                         verbose=False,
#                     )
#             except Exception:
#                 pass
#             yolo_model = yolo
#             logger.info("YOLO ready (gpu=%s)", gpu)
#         except Exception:
#             logger.exception("YOLO load failed; detection disabled.")
#             yolo_model = None
#     else:
#         logger.warning("ultralytics not installed; detection disabled.")
#         yolo_model = None

#     # TorchReID ensemble
#     reid_extractors.clear()
#     if TorchreidExtractor is not None:
#         try:
#             dev = DEVICE if gpu else "cpu"
#             for m in REID_MODELS:
#                 try:
#                     ext = TorchreidExtractor(model_name=m, device=dev)
#                     try:
#                         dummy = np.zeros((256, 128, 3), dtype=np.uint8)
#                         _ = ext([dummy])
#                     except Exception:
#                         pass
#                     reid_extractors.append(ext)
#                     logger.info("TorchReID ready: %s (%s)", m, dev)
#                 except Exception:
#                     logger.exception("TorchReID init failed for %s", m)
#             reid_extractor = reid_extractors[0] if reid_extractors else None
#         except Exception:
#             logger.exception("TorchReID init failed; body embeddings disabled.")
#             reid_extractors.clear()
#             reid_extractor = None
#     else:
#         logger.warning("torchreid not installed; body embeddings disabled.")
#         reid_extractors.clear()
#         reid_extractor = None

#     # InsightFace
#     face_app = init_face_engine(USE_FACE, DEVICE, FACE_MODEL, FACE_DET_SIZE[0], FACE_DET_SIZE[1], FACE_PROVIDER)

#     # Back-body init
#     _init_back_shape_embedder()

#     # Workers
#     start_workers_if_needed()


# # ===================== CSV + gallery helpers =====================

# def _ensure_dir(p: Path) -> None:
#     p.mkdir(parents=True, exist_ok=True)


# def ensure_csv_header() -> None:
#     csv_path = Path(EMB_CSV)
#     _ensure_dir(csv_path.parent if csv_path.parent.as_posix() not in (".", "") else Path("."))

#     expected_header = [
#         "member_id", "member_name", "ts", "camera_id", "frame_idx", "det_idx",
#         "x1", "y1", "x2", "y2", "conf_or_score",
#         "body_embedding", "face_embedding", "crop_path", "kind",
#     ]

#     def _write_new():
#         with open(csv_path, "w", newline="", encoding="utf-8") as f:
#             w = csv.writer(f)
#             w.writerow(expected_header)

#     if not csv_path.exists():
#         with csv_lock:
#             if not csv_path.exists():
#                 _write_new()
#                 logger.info("Created CSV header: %s", EMB_CSV)
#         return

#     try:
#         with open(csv_path, "r", newline="", encoding="utf-8") as f:
#             r = csv.reader(f)
#             hdr = next(r, None)
#     except Exception:
#         hdr = None

#     if hdr == expected_header:
#         return

#     with csv_lock:
#         try:
#             with open(csv_path, "r", newline="", encoding="utf-8") as f:
#                 r = csv.reader(f)
#                 hdr2 = next(r, None)
#         except Exception:
#             hdr2 = None
#         if hdr2 == expected_header:
#             return

#         bak = csv_path.with_suffix(csv_path.suffix + f".bak_{int(time.time())}")
#         try:
#             shutil.copy2(csv_path, bak)
#             logger.warning("CSV header mismatch. Backed up old CSV to: %s", bak)
#         except Exception:
#             logger.warning("CSV header mismatch, but backup failed. Rewriting header anyway.")
#         _write_new()
#         logger.info("Rewrote CSV header: %s", EMB_CSV)


# def _member_folder(member_id: int, member_name: Optional[str] = None) -> str:
#     mid = int(member_id)
#     if member_name:
#         return f"{mid}_{_safe_slug(str(member_name))}"
#     return str(mid)


# def _gallery_body_dir(member_id: int, camera_id: int, member_name: Optional[str] = None) -> Path:
#     d = Path(CROPS_ROOT) / "gallery" / _member_folder(member_id, member_name) / f"cam_{int(camera_id)}"
#     _ensure_dir(d)
#     return d


# def _gallery_face_dir(member_id: int, camera_id: int, member_name: Optional[str] = None) -> Path:
#     d = Path(CROPS_ROOT) / "gallery_face" / _member_folder(member_id, member_name) / f"cam_{int(camera_id)}"
#     _ensure_dir(d)
#     return d


# def _gallery_back_body_dir(member_id: int, camera_id: int, member_name: Optional[str] = None) -> Path:
#     d = Path(CROPS_ROOT) / BACK_BODY_GALLERY_NAME / _member_folder(member_id, member_name) / f"cam_{int(camera_id)}"
#     _ensure_dir(d)
#     return d


# def _epoch_ms() -> int:
#     return int(time.time() * 1000) * 1000 + random.randint(0, 999)


# # ===================== Face helpers (capture-time) =====================

# def detect_faces_raw(frame: np.ndarray):
#     outs: List[Dict] = []
#     if face_app is None:
#         return outs
#     try:
#         faces = face_app.get(np.ascontiguousarray(frame))
#         for f in safe_iter_faces(faces):
#             bbox = getattr(f, "bbox", None)
#             if bbox is None:
#                 continue
#             b = np.asarray(bbox).reshape(-1)
#             if b.size < 4:
#                 continue
#             x1, y1, x2, y2 = map(float, b[:4])
#             score = float(getattr(f, "det_score", 0.0))
#             outs.append({"bbox": (x1, y1, x2, y2), "score": score})
#     except Exception:
#         logger.exception("FaceAnalysis error")
#     return outs


# def link_face_to_person(face_bbox: Tuple[float, float, float, float], person_bbox: Tuple[int, int, int, int]) -> bool:
#     if face_center_in(tuple(map(int, face_bbox)), person_bbox):
#         return True
#     if iou_xyxy(person_bbox, face_bbox) >= FACE_IOU_LINK:
#         return True
#     if inter_over_face(person_bbox, face_bbox) >= FACE_OVER_FACE_LINK:
#         return True
#     return False


# def _face_big_enough(face_bbox: Tuple[float, float, float, float]) -> bool:
#     x1, y1, x2, y2 = face_bbox
#     return min(float(x2 - x1), float(y2 - y1)) >= float(FACE_MIN_SIZE)


# # ===================== RTSP helpers =====================

# def _configure_rtsp_capture(cap: cv2.VideoCapture) -> None:
#     try:
#         cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
#     except Exception:
#         pass


# def _read_latest_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
#     grabbed = cap.grab()
#     if not grabbed:
#         ok, fr = cap.read()
#         return ok, (fr if ok else None)
#     for _ in range(3):
#         if not cap.grab():
#             break
#     ok, frame = cap.retrieve()
#     return ok, (frame if ok else None)


# # ===================== Stream / Camera mapping =====================

# def _normalize_camera_ids(camera_ids: Any) -> Optional[List[int]]:
#     if camera_ids is None:
#         return None
#     if isinstance(camera_ids, int):
#         return [int(camera_ids)]
#     if isinstance(camera_ids, str):
#         parts = [p.strip() for p in camera_ids.split(",") if p.strip()]
#         out: List[int] = []
#         for p in parts:
#             try:
#                 out.append(int(p))
#             except Exception:
#                 continue
#         return out if out else None
#     if isinstance(camera_ids, (list, tuple, set)):
#         out: List[int] = []
#         for x in camera_ids:
#             try:
#                 out.append(int(x))
#             except Exception:
#                 continue
#         return out if out else None
#     return None


# def _camera_streams_map(streams: Any) -> Dict[int, str]:
#     out: Dict[int, str] = {}

#     if isinstance(streams, str):
#         s = streams.strip()
#         return {0: s} if s else {}

#     if isinstance(streams, dict):
#         for k, v in streams.items():
#             try:
#                 cid = int(k)
#             except Exception:
#                 continue
#             url = ""
#             if isinstance(v, str):
#                 url = v
#             elif isinstance(v, dict):
#                 url = str(v.get("url") or v.get("rtsp_url") or v.get("rtsp") or v.get("stream") or "")
#             elif isinstance(v, (list, tuple)) and len(v) >= 1:
#                 url = str(v[0])
#             if url:
#                 out[cid] = url
#         return out

#     if isinstance(streams, (list, tuple)):
#         for idx, item in enumerate(streams):
#             if isinstance(item, str):
#                 out[int(idx)] = item
#             elif isinstance(item, dict):
#                 cid = int(item.get("id", idx))
#                 url = str(item.get("url") or item.get("rtsp_url") or item.get("rtsp") or item.get("stream") or "")
#                 if url:
#                     out[cid] = url
#             elif isinstance(item, (list, tuple)):
#                 if len(item) >= 2:
#                     try:
#                         cid = int(item[0])
#                     except Exception:
#                         cid = int(idx)
#                     url = str(item[1])
#                     if url:
#                         out[cid] = url
#                 elif len(item) == 1:
#                     out[int(idx)] = str(item[0])
#     return out


# def _get_member_display_name(member: Any, fallback: str = "unknown") -> str:
#     for attr in ("first_name", "name", "full_name", "username"):
#         v = getattr(member, attr, None)
#         if v:
#             return str(v)
#     return fallback


# def _build_rtsp_url(ip: str, camera_id: int, camera_name: str = "") -> str:
#     ip = (ip or "").strip()
#     if not ip:
#         return ""

#     # Allow storing full RTSP URL in ip_address field
#     if ip.lower().startswith(("rtsp://", "rtsps://")):
#         return ip 
#     if RTSP_URL_TEMPLATE:
#         try:
#             return RTSP_URL_TEMPLATE.format(
#                 ip=ip,
#                 camera_id=int(camera_id),
#                 id=int(camera_id),
#                 name=str(camera_name or ""),
#                 user=RTSP_USERNAME,
#                 username=RTSP_USERNAME,
#                 password=RTSP_PASSWORD,
#                 port=RTSP_PORT,
#                 stream=RTSP_STREAM,
#             )
#         except Exception:
#             logger.exception("RTSP_URL_TEMPLATE formatting failed; falling back to basic builder")

#     auth = ""
#     if RTSP_USERNAME:
#         auth = RTSP_USERNAME
#         if RTSP_PASSWORD:
#             auth += f":{RTSP_PASSWORD}"
#         auth += "@"

#     port = f":{RTSP_PORT}" if RTSP_PORT else ""
#     path = RTSP_PATH or ""
#     if path and not path.startswith("/"):
#         path = "/" + path

#     return f"{RTSP_SCHEME}://{auth}{ip}{port}{path}"


# def _db_get_camera_ids(
#     camera_ids: Optional[List[int]] = None,
#     only_active: Optional[bool] = None
# ) -> List[int]:
#     if Camera is None:
#         return []

#     try:
#         with db_session() as db:
#             q = db.query(Camera.id)

#             if camera_ids:
#                 q = q.filter(Camera.id.in_(map(int, camera_ids)))

#             if only_active is True:
#                 q = q.filter(Camera.is_active.is_(True))

#             rows = q.all()

#         return sorted(r[0] for r in rows)

#     except Exception as e:
#         logger.exception(f"Failed to load camera ids from DB: {e}")
#         return []


# def _camera_streams_from_db(camera_ids: Optional[List[int]] = None) -> Dict[int, str]:
#     if Camera is None:
#         return {}

#     out: Dict[int, str] = {}
#     try:
#         with db_session() as db:
#             q = db.query(Camera)
#             if camera_ids:
#                 q = q.filter(Camera.id.in_([int(x) for x in camera_ids]))
#             if ONLY_ACTIVE_CAMERAS:
#                 q = q.filter(Camera.is_active.is_(True))
#             cams = q.all()
#     except Exception:
#         logger.exception("Failed to query cameras from DB")
#         return out

#     for cam in cams:
#         try:
#             cid = int(getattr(cam, "id"))
#         except Exception:
#             continue
#         url = _build_rtsp_url(str(getattr(cam, "ip_address", "") or ""), cid, str(getattr(cam, "name", "") or "")) 
#         if url:
#             out[cid] = url

#     return out


# def _get_streams_map(camera_ids: Optional[Any] = None) -> Dict[int, str]:
#     ids = _normalize_camera_ids(camera_ids)

#     if USE_DB_CAMERA_CONFIG:
#         m = _camera_streams_from_db(ids)
#         if m:
#             return m
#         if ids is not None:
#             # user requested IDs but DB could not resolve them
#             return {}

#     return _camera_streams_map(RTSP_STREAMS)


# def _validate_camera_ids_exist(camera_ids: List[int]) -> Tuple[List[int], List[int]]:
#     camera_ids = [int(x) for x in camera_ids if str(x).isdigit()]
#     if not camera_ids:
#         return [], []
#     if Camera is None:
#         return camera_ids, []
#     existing = set(_db_get_camera_ids(camera_ids=camera_ids, only_active=None))
#     valid = [cid for cid in camera_ids if cid in existing]
#     missing = [cid for cid in camera_ids if cid not in existing]
#     return valid, missing


# # ===================== Capture & processing loops =====================

# def capture_thread_fn(camera_id: int, rtsp_url: str):
#     q = cap_queues[camera_id]

#     # Prefer FFMPEG, but fall back if needed
#     cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
#     if not cap.isOpened():
#         cap = cv2.VideoCapture(rtsp_url)

#     if not cap.isOpened():
#         logger.error("[cap cam=%d] cannot open RTSP: %s", camera_id, rtsp_url)
#         return

#     _configure_rtsp_capture(cap)
#     logger.info("[cap cam=%d] started on %s", camera_id, rtsp_url)

#     try:
#         while not stop_event.is_set():
#             ok, frame = _read_latest_frame(cap)
#             if not ok or frame is None:
#                 time.sleep(0.01)
#                 continue

#             while not q.empty():
#                 try:
#                     q.get_nowait()
#                 except Exception:
#                     break
#             try:
#                 q.put_nowait(frame)
#             except Exception:
#                 pass
#     finally:
#         cap.release()
#         logger.info("[cap cam=%d] stopped", camera_id)


# def embedding_loop_for_cam(camera_id: int, rtsp_url: str):
#     global current_member_id, current_member_name

#     ensure_csv_header()
#     start_workers_if_needed()

#     member_id = int(current_member_id or 0)
#     member_name = str(current_member_name or "unknown")

#     body_dir = _gallery_body_dir(member_id, camera_id, member_name)
#     face_dir = _gallery_face_dir(member_id, camera_id, member_name)
#     back_dir = _gallery_back_body_dir(member_id, camera_id, member_name)

#     logger.info("[Loop cam=%d] dirs: body=%s face=%s back=%s", camera_id, body_dir, face_dir, back_dir)

#     frame_idx = 0
#     last_save_ms = 0
#     last_debug_save = 0.0
#     no_det_counter = 0

#     if yolo_model is None:
#         logger.warning("[Loop cam=%d] YOLO model is None -> no detections, no crops", camera_id)

#     logger.info("[Loop cam=%d] started on %s", camera_id, rtsp_url)
#     q = cap_queues[camera_id]

#     try:
#         while not stop_event.is_set():
#             try:
#                 frame = q.get(timeout=0.2)
#             except Empty:
#                 continue
#             except Exception:
#                 break
#             if frame is None:
#                 continue

#             frame_idx += 1
#             H, W = frame.shape[:2]

#             now_ms = int(time.time() * 1000)
#             allow_save = (now_ms - last_save_ms) >= int(SAVE_MIN_INTERVAL_MS)
#             if allow_save:
#                 last_save_ms = now_ms

#             # submit YOLO request
#             respq = Queue(maxsize=1)
#             yolo_ok = (yolo_model is not None and yolo_req_queue is not None)
#             if yolo_ok:
#                 try:
#                     yolo_req_queue.put_nowait({"frame": frame, "H": H, "W": W, "resp": respq})
#                 except Exception:
#                     yolo_ok = False

#             faces_checked = bool(USE_FACE and face_app is not None and (frame_idx % max(1, int(FACE_EVERY)) == 0))
#             faces = detect_faces_raw(frame) if faces_checked else []

#             detections: List[tuple[float, float, float, float, float]] = []
#             if yolo_ok:
#                 try:
#                     detections = respq.get(timeout=float(YOLO_RESP_TIMEOUT_S))
#                 except Exception:
#                     detections = []
#             else:
#                 detections = yolo_infer(frame, H, W, timeout_s=float(YOLO_RESP_TIMEOUT_S))

#             det_boxes_i: List[tuple[int, int, int, int, float]] = []
#             for (x1, y1, x2, y2, conf) in detections:
#                 x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
#                 x1i, y1i = max(0, x1i), max(0, y1i)
#                 x2i, y2i = min(W, x2i), min(H, y2i)
#                 if x2i <= x1i or y2i <= y1i:
#                     continue
#                 det_boxes_i.append((x1i, y1i, x2i, y2i, conf))

#             if not det_boxes_i:
#                 no_det_counter += 1
#                 if DEBUG_LOG_NO_DET_EVERY > 0 and (no_det_counter % DEBUG_LOG_NO_DET_EVERY) == 0:
#                     logger.info("[Loop cam=%d] no detections for %d frames", camera_id, no_det_counter)

#                 # Optional debug: save a raw frame periodically if no detections
#                 if DEBUG_SAVE_NO_DETS_EVERY_SEC > 0:
#                     now = time.time()
#                     if (now - last_debug_save) >= float(DEBUG_SAVE_NO_DETS_EVERY_SEC):
#                         last_debug_save = now
#                         dbg_dir = Path(CROPS_ROOT) / "debug_frames" / _member_folder(member_id, member_name) / f"cam_{camera_id}"
#                         _ensure_dir(dbg_dir)
#                         dbg_path = dbg_dir / f"frame_{int(now)}.jpg"
#                         try:
#                             cv2.imwrite(str(dbg_path), frame)
#                             logger.info("[Loop cam=%d] DEBUG saved frame: %s", camera_id, dbg_path)
#                         except Exception:
#                             pass

#             # Save crops (rate-limited)
#             if allow_save and det_boxes_i:
#                 ts = time.time()
#                 for det_idx, (x1i, y1i, x2i, y2i, conf) in enumerate(det_boxes_i[: max(1, int(SAVE_MAX_DETS_PER_FRAME))]):

#                     # (1) regular body gallery
#                     crop_bgr = crop_with_padding(frame, (x1i, y1i, x2i, y2i))
#                     if crop_bgr.size > 0:
#                         fname = f"person_{_epoch_ms()}.jpg"
#                         path = body_dir / fname
#                         row = [
#                             member_id, member_name, ts,
#                             int(camera_id), frame_idx, det_idx,
#                             x1i, y1i, x2i, y2i,
#                             conf,
#                             "", "", str(path), "body",
#                         ]
#                         try:
#                             if io_queue is not None:
#                                 io_queue.put_nowait({"crop_path": path, "image": crop_bgr, "csv_row": row})
#                         except Exception:
#                             pass

#                     # (3) back-body / no-face gallery
#                     if faces_checked:
#                         has_linked_face = False
#                         if faces:
#                             t_xyxy = (x1i, y1i, x2i, y2i)
#                             for fm in faces:
#                                 fb = fm.get("bbox")
#                                 if not fb:
#                                     continue
#                                 score = float(fm.get("score", 0.0))
#                                 if score < float(FACE_MIN_SCORE):
#                                     continue
#                                 if not _face_big_enough(tuple(map(float, fb))):
#                                     continue
#                                 if link_face_to_person(tuple(map(float, fb)), t_xyxy):
#                                     has_linked_face = True
#                                     break

#                         if not has_linked_face:
#                             back_crop = crop_back_body_no_face(frame, (x1i, y1i, x2i, y2i))
#                             if back_crop.size > 0:
#                                 fname = f"back_{_epoch_ms()}.jpg"
#                                 path = back_dir / fname
#                                 row = [
#                                     member_id, member_name, ts,
#                                     int(camera_id), frame_idx, det_idx,
#                                     x1i, y1i, x2i, y2i,
#                                     conf,
#                                     "", "", str(path), "back_body",
#                                 ]
#                                 try:
#                                     if io_queue is not None:
#                                         io_queue.put_nowait({"crop_path": path, "image": back_crop, "csv_row": row})
#                                 except Exception:
#                                     pass

#                 # (2) gallery_face
#                 if faces and faces_checked:
#                     for det_idx, (x1i, y1i, x2i, y2i, _conf) in enumerate(det_boxes_i[: max(1, int(SAVE_MAX_DETS_PER_FRAME))]):
#                         t_xyxy = (x1i, y1i, x2i, y2i)
#                         best_score = -1.0
#                         best_box = None
#                         for fm in faces:
#                             fb = fm.get("bbox")
#                             if not fb:
#                                 continue
#                             if link_face_to_person(tuple(map(float, fb)), t_xyxy):
#                                 s = float(fm.get("score", 0.0))
#                                 if s > best_score:
#                                     best_score = s
#                                     best_box = fb

#                         if best_box is None:
#                             continue

#                         fx1, fy1, fx2, fy2 = map(int, best_box)
#                         if best_score < FACE_MIN_SCORE:
#                             continue
#                         if min(fx2 - fx1, fy2 - fy1) < FACE_MIN_SIZE:
#                             continue

#                         body_crop = crop_with_padding(frame, (x1i, y1i, x2i, y2i))
#                         if body_crop.size <= 0:
#                             continue

#                         fname = f"person_{_epoch_ms()}.jpg"
#                         path = face_dir / fname
#                         row = [
#                             member_id, member_name, ts,
#                             int(camera_id), frame_idx, det_idx,
#                             x1i, y1i, x2i, y2i,
#                             best_score,
#                             "", "", str(path), "face",
#                         ]
#                         try:
#                             if io_queue is not None:
#                                 io_queue.put_nowait({"crop_path": path, "image": body_crop, "csv_row": row})
#                         except Exception:
#                             pass

#             with frames_lock:
#                 latest_frames[camera_id] = frame

#     finally:
#         logger.info("[Loop cam=%d] stopped", camera_id)


# # ===================== Extract from folders helpers =====================

# def _list_images(root: Path) -> List[Path]:
#     if not root.exists():
#         return []
#     pats = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
#     files: List[Path] = []
#     for p in pats:
#         files.extend(root.glob(p))
#     return sorted(files)


# def _stable_seed(*parts: str) -> int:
#     s = "|".join(parts).encode("utf-8", errors="ignore")
#     return int(zlib.crc32(s) & 0xFFFFFFFF)


# def _sample_paths(paths: List[Path], max_n: int, seed: int) -> List[Path]:
#     if max_n <= 0 or len(paths) <= max_n:
#         return paths
#     rng = random.Random(int(seed))
#     idxs = sorted(rng.sample(range(len(paths)), int(max_n)))
#     return [paths[i] for i in idxs]


# def _list_images_sampled(root: Path, max_n: int, member_id: int, camera_id: int, kind: str) -> List[Path]:
#     paths = _list_images(root)
#     seed = _stable_seed(str(member_id), str(camera_id), kind, str(GALLERY_SAMPLE_SEED))
#     return _sample_paths(paths, int(max_n), seed)


# def _load_bgr(path: Path) -> Optional[np.ndarray]:
#     try:
#         img = cv2.imread(str(path), cv2.IMREAD_COLOR)
#         return img if img is not None else None
#     except Exception:
#         return None


# def _mean_embed(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
#     if not vectors:
#         return None
#     arr = np.stack(vectors, axis=0).astype(np.float32)
#     m = np.mean(arr, axis=0)
#     return l2_normalize(m)


# def _count_images(member_id: int, camera_id: int, member_name: str) -> Tuple[int, int, int]:
#     body = len(_list_images_sampled(_gallery_body_dir(member_id, camera_id, member_name), MAX_BODY_IMAGES, member_id, camera_id, "body"))
#     face = len(_list_images_sampled(_gallery_face_dir(member_id, camera_id, member_name), MAX_FACE_IMAGES, member_id, camera_id, "face"))
#     back_body = len(_list_images_sampled(_gallery_back_body_dir(member_id, camera_id, member_name), MAX_BACK_BODY_IMAGES, member_id, camera_id, "back_body"))
#     return body, face, back_body


# def _ahash8(img_bgr: np.ndarray) -> int:
#     try:
#         g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#         g = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
#         avg = float(g.mean())
#         bits = (g > avg).astype(np.uint8)
#         out = 0
#         for i, b in enumerate(bits.flatten()):
#             out |= (int(b) & 1) << i
#         return out
#     except Exception:
#         return random.getrandbits(64)


# def _is_blurry(img_bgr: np.ndarray, var_thr: float = 30.0) -> bool:
#     try:
#         g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
#         return float(cv2.Laplacian(g, cv2.CV_64F).var()) < var_thr
#     except Exception:
#         return False


# def _prep_reid(img_bgr: np.ndarray) -> np.ndarray:
#     h, w = img_bgr.shape[:2]
#     if h < 32 or w < 16:
#         raise ValueError("Crop too small for ReID")
#     img = cv2.resize(img_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
#     return _to_rgb(img)


# def _compute_body_bank_and_centroid_from_gallery(
#     member_id: int,
#     camera_id: int,
#     member_name: str,
#     on_file_done: Optional[Callable[[int], None]] = None,
#     on_work_done: Optional[Callable[[int], None]] = None,
# ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
#     if not reid_extractors:
#         logger.warning("TorchReID not available; skipping body embedding.")
#         return None, None

#     root = _gallery_body_dir(member_id, camera_id, member_name)
#     imgs = _list_images_sampled(root, MAX_BODY_IMAGES, member_id, camera_id, "body")
#     if not imgs:
#         logger.warning("No BODY crops found for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
#     B = 64 if gpu else 32

#     prepared: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
#     seen_hashes: set[int] = set()
#     per_image_units = len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)

#     for p in imgs:
#         img = _load_bgr(p)
#         if img is None or img.size == 0:
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue
#         h, w = img.shape[:2]
#         if min(h, w) < 32 or _is_blurry(img):
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue
#         hval = _ahash8(img)
#         if hval in seen_hashes:
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue
#         seen_hashes.add(hval)

#         try:
#             rgb = _prep_reid(img)
#         except Exception:
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue

#         rgb_flip = cv2.flip(rgb, 1) if BODY_TTA_FLIP else None
#         prepared.append((rgb, rgb_flip))

#     if not prepared:
#         logger.warning("No valid BODY crops for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     per_model_embs: List[List[np.ndarray]] = [[] for _ in reid_extractors]

#     for midx, ext in enumerate(reid_extractors):
#         batch_imgs: List[np.ndarray] = []
#         idx_map: List[int] = []
#         for i, (orig, flip) in enumerate(prepared):
#             batch_imgs.append(orig)
#             idx_map.append(i)
#             if flip is not None:
#                 batch_imgs.append(flip)
#                 idx_map.append(i)

#         feats_norm: List[np.ndarray] = []
#         for start in range(0, len(batch_imgs), B):
#             chunk = batch_imgs[start:start + B]
#             try:
#                 with reid_lock, torch.inference_mode():
#                     feats = ext(chunk)
#                 for f in feats:
#                     f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
#                     f512 = _as_512f(f)
#                     feats_norm.append(f512 if f512 is not None else np.zeros((EXPECTED_EMBED_DIM,), np.float32))
#             except Exception:
#                 logger.exception("TorchReID batch failed (model=%s)", REID_MODELS[midx])
#                 feats_norm.extend([np.zeros((EXPECTED_EMBED_DIM,), np.float32) for _ in chunk])

#             if on_work_done:
#                 on_work_done(len(chunk))

#         img_accum = [[] for _ in prepared]
#         for img_i, feat in zip(idx_map, feats_norm):
#             img_accum[img_i].append(feat)

#         fused = [
#             l2_normalize(np.mean(np.stack(v, axis=0), axis=0)) if v else np.zeros((EXPECTED_EMBED_DIM,), np.float32)
#             for v in img_accum
#         ]
#         per_model_embs[midx] = fused

#     vectors_per_image: List[np.ndarray] = []
#     for i in range(len(prepared)):
#         parts = [per_model_embs[m][i] for m in range(len(reid_extractors))]
#         fused_img = l2_normalize(np.mean(np.stack(parts, axis=0), axis=0))
#         vectors_per_image.append(fused_img)
#         if on_file_done:
#             on_file_done(1)

#     centroid = _mean_embed(vectors_per_image)
#     return (vectors_per_image if vectors_per_image else None), centroid


# def _compute_face_bank_and_centroid_from_gallery(
#     member_id: int,
#     camera_id: int,
#     member_name: str,
#     on_file_done: Optional[Callable[[int], None]] = None,
#     on_work_done: Optional[Callable[[int], None]] = None,
# ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
#     if face_app is None:
#         logger.warning("InsightFace not available; skipping face embedding.")
#         return None, None

#     root = _gallery_face_dir(member_id, camera_id, member_name)
#     imgs = _list_images_sampled(root, MAX_FACE_IMAGES, member_id, camera_id, "face")
#     if not imgs:
#         logger.warning("No FACE crops found for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     vectors: List[np.ndarray] = []
#     seen_hashes: set[int] = set()
#     MAX_SIDE = 640
#     MIN_BODY_SIDE = 40
#     per_image_units = (2 if FACE_TTA_FLIP else 1)

#     for p in imgs:
#         img = _load_bgr(p)
#         if img is None or img.size == 0:
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue

#         h, w = img.shape[:2]
#         if min(h, w) < MIN_BODY_SIDE or _is_blurry(img):
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue

#         hval = _ahash8(img)
#         if hval in seen_hashes:
#             if on_file_done:
#                 on_file_done(1)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             continue
#         seen_hashes.add(hval)

#         if max(h, w) > MAX_SIDE:
#             scale = MAX_SIDE / float(max(h, w))
#             img_proc = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
#         else:
#             img_proc = img

#         def _best_face(bgr_img) -> Optional[np.ndarray]:
#             try:
#                 faces = face_app.get(np.ascontiguousarray(bgr_img))
#             except Exception:
#                 faces = []
#             best, best_score = None, -1.0
#             for f in safe_iter_faces(faces):
#                 bbox = getattr(f, "bbox", None)
#                 if bbox is None:
#                     continue
#                 b = np.asarray(bbox).reshape(-1)
#                 if b.size < 4:
#                     continue
#                 x1, y1, x2, y2 = b[:4].astype(float)
#                 area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
#                 score = float(getattr(f, "det_score", 0.0))
#                 ranking = score * (area ** 0.5)
#                 if ranking > best_score:
#                     best, best_score = f, ranking
#             if best is None:
#                 return None
#             det_score = float(getattr(best, "det_score", 0.0))
#             if det_score < FACE_MIN_SCORE:
#                 return None
#             emb = extract_face_embedding(best)
#             return _as_512f(emb)

#         e0 = _best_face(img_proc)
#         if on_work_done:
#             on_work_done(1)

#         e1 = None
#         if FACE_TTA_FLIP:
#             try:
#                 e1 = _best_face(cv2.flip(img_proc, 1))
#             except Exception:
#                 e1 = None
#             if on_work_done:
#                 on_work_done(1)

#         feats = [e for e in (e0, e1) if e is not None]
#         if feats:
#             fused = l2_normalize(np.mean(np.stack(feats, axis=0), axis=0))
#             vectors.append(fused)

#         if on_file_done:
#             on_file_done(1)

#     if not vectors:
#         logger.warning("No valid FACE embeddings for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     centroid = _mean_embed(vectors)
#     return (vectors if vectors else None), centroid


# def _compute_back_body_bank_and_centroid_from_gallery(
#     member_id: int,
#     camera_id: int,
#     member_name: str,
#     on_file_done: Optional[Callable[[int], None]] = None,
#     on_work_done: Optional[Callable[[int], None]] = None,
# ) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
#     root = _gallery_back_body_dir(member_id, camera_id, member_name)
#     imgs = _list_images_sampled(root, MAX_BACK_BODY_IMAGES, member_id, camera_id, "back_body")
#     if not imgs:
#         logger.warning("No BACK-BODY crops found for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
#     B = 64 if gpu else 32

#     seen_hashes: set[int] = set()

#     prepared_reid: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []
#     shape_vecs: List[Optional[np.ndarray]] = []
#     keep_mask: List[bool] = []
#     reid_ready: List[bool] = []

#     reid_units = (len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)) if reid_extractors else 0
#     per_image_units = 1 + int(reid_units)

#     for p in imgs:
#         img = _load_bgr(p)
#         if img is None or img.size == 0:
#             shape_vecs.append(None)
#             prepared_reid.append((None, None))
#             keep_mask.append(False)
#             reid_ready.append(False)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             if on_file_done:
#                 on_file_done(1)
#             continue

#         h, w = img.shape[:2]
#         if min(h, w) < 32 or _is_blurry(img):
#             shape_vecs.append(None)
#             prepared_reid.append((None, None))
#             keep_mask.append(False)
#             reid_ready.append(False)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             if on_file_done:
#                 on_file_done(1)
#             continue

#         hval = _ahash8(img)
#         if hval in seen_hashes:
#             shape_vecs.append(None)
#             prepared_reid.append((None, None))
#             keep_mask.append(False)
#             reid_ready.append(False)
#             if on_work_done:
#                 on_work_done(per_image_units)
#             if on_file_done:
#                 on_file_done(1)
#             continue
#         seen_hashes.add(hval)

#         shp = _back_shape_embed(img)
#         shape_vecs.append(shp)
#         if on_work_done:
#             on_work_done(1)

#         rgb = None
#         rgb_flip = None
#         can_reid = False
#         if reid_extractors:
#             try:
#                 rgb = _prep_reid(img)
#                 rgb_flip = cv2.flip(rgb, 1) if BODY_TTA_FLIP else None
#                 can_reid = True
#             except Exception:
#                 can_reid = False

#         prepared_reid.append((rgb, rgb_flip))
#         reid_ready.append(bool(can_reid))

#         keep = (shp is not None) or bool(can_reid)
#         keep_mask.append(bool(keep))

#         if not keep:
#             if on_work_done and reid_units > 0:
#                 on_work_done(int(reid_units))
#             if on_file_done:
#                 on_file_done(1)
#             continue

#         if not can_reid and reid_units > 0:
#             if on_work_done:
#                 on_work_done(int(reid_units))

#     reid_vectors_per_image: List[Optional[np.ndarray]] = [None for _ in prepared_reid]

#     if reid_extractors:
#         per_model_embs: List[List[np.ndarray]] = [[] for _ in reid_extractors]

#         for midx, ext in enumerate(reid_extractors):
#             batch_imgs: List[np.ndarray] = []
#             idx_map: List[int] = []
#             for i, (orig, flip) in enumerate(prepared_reid):
#                 if not reid_ready[i] or orig is None:
#                     continue
#                 batch_imgs.append(orig)
#                 idx_map.append(i)
#                 if flip is not None:
#                     batch_imgs.append(flip)
#                     idx_map.append(i)

#             feats_norm: List[np.ndarray] = []
#             for start in range(0, len(batch_imgs), B):
#                 chunk = batch_imgs[start:start + B]
#                 try:
#                     with reid_lock, torch.inference_mode():
#                         feats = ext(chunk)
#                     for f in feats:
#                         f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
#                         f512 = _as_512f(f)
#                         feats_norm.append(f512 if f512 is not None else np.zeros((EXPECTED_EMBED_DIM,), np.float32))
#                 except Exception:
#                     logger.exception("TorchReID batch failed (model=%s) [back_body]", REID_MODELS[midx])
#                     feats_norm.extend([np.zeros((EXPECTED_EMBED_DIM,), np.float32) for _ in chunk])

#                 if on_work_done:
#                     on_work_done(len(chunk))

#             img_accum: Dict[int, List[np.ndarray]] = {}
#             for img_i, feat in zip(idx_map, feats_norm):
#                 img_accum.setdefault(img_i, []).append(feat)

#             fused_list: List[np.ndarray] = []
#             for i in range(len(prepared_reid)):
#                 vv = img_accum.get(i, [])
#                 fused_list.append(
#                     l2_normalize(np.mean(np.stack(vv, axis=0), axis=0)) if vv else np.zeros((EXPECTED_EMBED_DIM,), np.float32)
#                 )
#             per_model_embs[midx] = fused_list

#         for i in range(len(prepared_reid)):
#             if not reid_ready[i]:
#                 continue
#             parts = [per_model_embs[m][i] for m in range(len(reid_extractors))]
#             if parts:
#                 reid_vectors_per_image[i] = l2_normalize(np.mean(np.stack(parts, axis=0), axis=0))

#     fused_vectors: List[np.ndarray] = []
#     for i in range(len(prepared_reid)):
#         if not keep_mask[i]:
#             continue

#         v_reid = reid_vectors_per_image[i]
#         v_shape = shape_vecs[i] if i < len(shape_vecs) else None

#         parts2: List[np.ndarray] = []
#         if v_reid is not None:
#             parts2.append(v_reid * float(BACK_REID_WEIGHT))
#         if v_shape is not None:
#             parts2.append(v_shape * float(BACK_SHAPE_WEIGHT))

#         if parts2:
#             fused = l2_normalize(np.sum(np.stack(parts2, axis=0), axis=0))
#             fused_vectors.append(fused)

#         if on_file_done:
#             on_file_done(1)

#     if not fused_vectors:
#         logger.warning("No valid BACK-BODY embeddings for member=%s cam=%s", member_id, camera_id)
#         return None, None

#     centroid = _mean_embed(fused_vectors)
#     return (fused_vectors if fused_vectors else None), centroid


# # ===================== DB update (FIX FK camera_id) =====================

# def _update_db_embeddings(
#     member_id: int,
#     camera_id: int,
#     body_emb: Optional[np.ndarray],
#     face_emb: Optional[np.ndarray],
#     back_body_emb: Optional[np.ndarray],
#     body_bank: Optional[List[np.ndarray]] = None,
#     face_bank: Optional[List[np.ndarray]] = None,
#     back_body_bank: Optional[List[np.ndarray]] = None,
# ) -> Dict[str, object]:
#     with db_session() as db:
#         try:
#             m = db.query(Member).filter(Member.id == int(member_id)).first()
#             if not m:
#                 logger.warning("Member ID %s not found. Skipping DB update.", member_id)
#                 return {"status": "error", "message": "member not found"}

#             # IMPORTANT: camera FK validation
#             if Camera is not None:
#                 cam = db.query(Camera).filter(Camera.id == int(camera_id)).first()
#                 if not cam:
#                     logger.error("Camera ID %s not found in cameras table. FK would fail.", camera_id)
#                     db.rollback()
#                     return {"status": "error", "message": "camera not found", "member_id": int(member_id), "camera_id": int(camera_id)}

#             rec = (
#                 db.query(MemberEmbedding)
#                 .filter(MemberEmbedding.member_id == int(member_id), MemberEmbedding.camera_id == int(camera_id))
#                 .first()
#             )
#             if rec is None:
#                 rec = MemberEmbedding(member_id=int(member_id), camera_id=int(camera_id))
#                 db.add(rec)

#             changed = False

#             if body_emb is not None:
#                 rec.body_embedding = body_emb.tolist()
#                 changed = True
#             if face_emb is not None:
#                 rec.face_embedding = face_emb.tolist()
#                 changed = True
#             if back_body_emb is not None:
#                 rec.back_body_embedding = back_body_emb.tolist()
#                 changed = True

#             if body_bank is not None:
#                 rec.body_embeddings_raw = _pack_raw_bank(body_bank)
#                 changed = True
#             if face_bank is not None:
#                 rec.face_embeddings_raw = _pack_raw_bank(face_bank)
#                 changed = True
#             if back_body_bank is not None:
#                 rec.back_body_embeddings_raw = _pack_raw_bank(back_body_bank)
#                 changed = True

#             if changed:
#                 rec.last_embedding_update_ts = datetime.now(timezone.utc)
#                 db.commit()
#                 logger.info(
#                     "DB updated: member=%s cam=%s body=%s face=%s back=%s body_raw=%s face_raw=%s back_raw=%s",
#                     member_id,
#                     camera_id,
#                     body_emb is not None,
#                     face_emb is not None,
#                     back_body_emb is not None,
#                     body_bank is not None,
#                     face_bank is not None,
#                     back_body_bank is not None,
#                 )
#                 return {
#                     "status": "ok",
#                     "member_id": int(member_id),
#                     "camera_id": int(camera_id),
#                     "body": body_emb is not None,
#                     "face": face_emb is not None,
#                     "back_body": back_body_emb is not None,
#                     "body_raw": body_bank is not None,
#                     "face_raw": face_bank is not None,
#                     "back_body_raw": back_body_bank is not None,
#                 }

#             db.rollback()
#             logger.info("No embeddings to update for member=%s cam=%s", member_id, camera_id)
#             return {"status": "no_embeddings", "member_id": int(member_id), "camera_id": int(camera_id)}

#         except Exception:
#             db.rollback()
#             logger.exception("DB error on update")
#             return {"status": "error", "message": "db error", "member_id": int(member_id), "camera_id": int(camera_id)}


# # ===================== Background extraction workers =====================

# def _extract_worker(member_id: int, member_name: str, camera_id: int):
#     try:
#         _progress_reset(member_id, member_name, camera_id)
#         _progress_set(member_id, camera_id, stage="Scanning", message="Counting images...")
#         total_body, total_face, total_back = _count_images(member_id, camera_id, member_name)

#         if total_body + total_face + total_back == 0:
#             _progress_set(
#                 member_id,
#                 camera_id,
#                 total_body=0,
#                 total_face=0,
#                 total_back_body=0,
#                 stage="error",
#                 message="No images to process",
#             )
#             return

#         _progress_set(
#             member_id,
#             camera_id,
#             total_body=total_body,
#             total_face=total_face,
#             total_back_body=total_back,
#             done_body=0,
#             done_face=0,
#             done_back_body=0,
#             percent=0,
#         )

#         body_bank = None
#         body_cent = None
#         if total_body > 0 and reid_extractors:
#             units_per = len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)
#             _progress_set(
#                 member_id,
#                 camera_id,
#                 stage="Body embeddings",
#                 message=f"Processing body ({total_body})",
#                 work_total=int(total_body * units_per),
#                 work_done=0,
#             )

#             def on_body_files(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, done_body=int(st.get("done_body", 0)) + int(n))

#             def on_body_work(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, work_done=int(st.get("work_done", 0)) + int(n))

#             body_bank, body_cent = _compute_body_bank_and_centroid_from_gallery(
#                 member_id, camera_id, member_name, on_file_done=on_body_files, on_work_done=on_body_work
#             )
#         else:
#             _progress_set(member_id, camera_id, message="Skipping body (no images or model unavailable)")

#         face_bank = None
#         face_cent = None
#         if total_face > 0 and face_app is not None:
#             units_per = (2 if FACE_TTA_FLIP else 1)
#             _progress_set(
#                 member_id,
#                 camera_id,
#                 stage="Face embeddings",
#                 message=f"Processing face ({total_face})",
#                 work_total=int(total_face * units_per),
#                 work_done=0,
#             )

#             def on_face_files(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, done_face=int(st.get("done_face", 0)) + int(n))

#             def on_face_work(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, work_done=int(st.get("work_done", 0)) + int(n))

#             face_bank, face_cent = _compute_face_bank_and_centroid_from_gallery(
#                 member_id, camera_id, member_name, on_file_done=on_face_files, on_work_done=on_face_work
#             )
#         else:
#             _progress_set(member_id, camera_id, message="Skipping face (no images or model unavailable)")

#         back_bank = None
#         back_cent = None
#         if total_back > 0:
#             reid_units = (len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)) if reid_extractors else 0
#             units_per = 1 + reid_units
#             _progress_set(
#                 member_id,
#                 camera_id,
#                 stage="Back body embeddings",
#                 message=f"Processing back-body ({total_back})",
#                 work_total=int(total_back * units_per),
#                 work_done=0,
#             )

#             def on_back_files(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, done_back_body=int(st.get("done_back_body", 0)) + int(n))

#             def on_back_work(n: int):
#                 st = get_progress_for_member(member_id, camera_id)
#                 _progress_set(member_id, camera_id, work_done=int(st.get("work_done", 0)) + int(n))

#             back_bank, back_cent = _compute_back_body_bank_and_centroid_from_gallery(
#                 member_id, camera_id, member_name, on_file_done=on_back_files, on_work_done=on_back_work
#             )
#         else:
#             _progress_set(member_id, camera_id, message="Skipping back-body (no images)")

#         # ---- Validation: ensure all six embedding inputs are present before DB update ----
#         required_pairs = [
#             ("Body embeddings", body_cent),
#             ("Face embeddings", face_cent),
#             ("Back body embeddings", back_cent),
#             ("Body bank", body_bank),
#             ("Face bank", face_bank),
#             ("Back body bank", back_bank),
#         ]
#         missing = [name for name, val in required_pairs if val is None]
#         if missing:
#             missing_list = ", ".join(missing)
#             friendly = (
#                 f"Missing required embedding data: {missing_list}. "
#                 "Please provide the missing embedding to proceed."
#             )
#             logger.error("Extract worker validation failed: %s (member=%s cam=%s)", missing_list, member_id, camera_id)
#             _progress_set(
#                 member_id,
#                 camera_id,
#                 stage="error",
#                 message=friendly,
#                 percent=100,
#             )
#             return

#         # ---- call the DB update as before ----
#         res = _update_db_embeddings(
#             member_id,
#             camera_id,
#             body_cent,
#             face_cent,
#             back_cent,
#             body_bank=body_bank,
#             face_bank=face_bank,
#             back_body_bank=back_bank,
#         )
#         status = str(res.get("status", "unknown"))
#         _progress_set(member_id, camera_id, stage="Done", message=("Embeddings saved" if status == "ok" else status), percent=100)

#     except Exception as e:
#         logger.exception("extract worker failed")
#         _progress_set(member_id, camera_id, stage="error", message=f"error: {e}")

# def _extract_worker_multi(member_id: int, member_name: str, camera_ids: List[int]):
#     camera_ids = [int(c) for c in camera_ids if isinstance(c, int) or str(c).isdigit()]
#     if not camera_ids:
#         return

#     _progress_reset(member_id, member_name, OVERALL_CAMERA_ID)
#     _progress_set(member_id, OVERALL_CAMERA_ID, stage="Starting", message=f"Processing {len(camera_ids)} camera(s)", percent=0)

#     done = 0
#     total = max(1, len(camera_ids))

#     for cid in camera_ids:
#         _progress_set(member_id, OVERALL_CAMERA_ID, stage="running", message=f"Extracting camera {cid} ({done+1}/{total})")
#         _extract_worker(member_id, member_name, cid)
#         done += 1
#         _progress_set(member_id, OVERALL_CAMERA_ID, percent=int(round((done / total) * 100)))

#     _progress_set(member_id, OVERALL_CAMERA_ID, stage="Done", message="All cameras processed", percent=100)


# def extract_embeddings_async(member_id: int, camera_ids: Optional[Any] = None) -> Dict[str, object]:
#     if not reid_extractors or (USE_FACE and face_app is None):
#         init_models()

#     cam_ids = _normalize_camera_ids(camera_ids)
#     if cam_ids is None:
#         cam_ids = _db_get_camera_ids(camera_ids=None, only_active=True)
#         if not cam_ids:
#             cam_ids = sorted(_camera_streams_map(RTSP_STREAMS).keys())
#     else:
#         cam_ids = [int(x) for x in cam_ids]

#     valid_ids, missing = _validate_camera_ids_exist(cam_ids)
#     if missing:
#         return {
#             "status": "error",
#             "message": "Some camera_ids do not exist in cameras tablesssssssss",
#             "missing_camera_ids": missing,
#             "valid_camera_ids": valid_ids,
#         }
#     if not valid_ids:
#         return {"status": "error", "message": "No valid camera_ids to extract", "requested_camera_ids": cam_ids}

#     with db_session() as db:
#         try:
#             member = db.query(Member).filter(Member.id == int(member_id)).first()
#             if not member:
#                 return {"status": "error", "message": f"member id {member_id} not found"}
#             member_name = _get_member_display_name(member, fallback=f"member_{member_id}")
#         except Exception:
#             db.rollback()
#             logger.exception("DB error during member lookup.")
#             return {"status": "error", "message": "DB error"}

#     t = threading.Thread(
#         target=_extract_worker_multi,
#         args=(int(member_id), member_name, valid_ids),
#         daemon=True,
#         name=f"extract-{member_id}",
#     )
#     t.start()
#     return {"status": "started", "member_id": int(member_id), "member_name": member_name, "camera_ids": valid_ids}


# def extract_embeddings_sync(member_id: int, camera_ids: Optional[Any] = None) -> Dict[str, object]:
#     if not reid_extractors or (USE_FACE and face_app is None):
#         init_models()

#     cam_ids = _normalize_camera_ids(camera_ids)
#     if cam_ids is None:
#         cam_ids = _db_get_camera_ids(camera_ids=None, only_active=True)
#         if not cam_ids:
#             cam_ids = sorted(_camera_streams_map(RTSP_STREAMS).keys())
#     else:
#         cam_ids = [int(x) for x in cam_ids]

#     valid_ids, missing = _validate_camera_ids_exist(cam_ids)
#     if missing:
#         return {
#             "status": "error",
#             "message": "Some camera_ids do not exist in cameras table",
#             "missing_camera_ids": missing,
#             "valid_camera_ids": valid_ids,
#         }
#     if not valid_ids:
#         return {"status": "error", "message": "No valid camera_ids to extract", "requested_camera_ids": cam_ids}

#     with db_session() as db:
#         try:
#             member = db.query(Member).filter(Member.id == int(member_id)).first()
#             if not member:
#                 return {"status": "error", "message": f"member id {member_id} not found"}
#             member_name = _get_member_display_name(member, fallback=f"member_{member_id}")
#         except Exception:
#             db.rollback()
#             logger.exception("DB error during member lookup.")
#             return {"status": "error", "message": "DB error"}

#     results: Dict[int, Dict[str, object]] = {}
#     for cid in valid_ids:
#         _extract_worker(int(member_id), member_name, int(cid))
#         results[int(cid)] = get_progress_for_member(int(member_id), int(cid))

#     return {"status": "Done", "member_id": int(member_id), "member_name": member_name, "cameras": results}


# # ===================== Control API =====================

# def start_extraction(member_id: int, camera_ids: Optional[Any] = None, show_viewer: bool = True) -> dict:
#     """
#     FIX:
#       - camera_ids refer to cameras.id (DB FK)
#       - RTSP URL built from Camera.ip_address using RTSP_URL_TEMPLATE
#     """
#     global is_running, current_member_id, current_member_name, current_camera_ids, current_camera_streams, viewer_thread

#     if is_running:
#         return {
#             "status": "ok",
#             "message": "Already running",
#             "num_cams": len(current_camera_ids),
#             "member_id": current_member_id,
#             "member_name": current_member_name,
#             "camera_ids": list(current_camera_ids),
#         }

#     selected = _normalize_camera_ids(camera_ids)
#     streams_map = _get_streams_map(selected)

#     if not streams_map:
#         avail = sorted(_get_streams_map(None).keys())
#         return {
#             "status": "error",
#             "message": "No valid cameras/RTSP streams configured",
#             "requested_camera_ids": selected,
#             "available_camera_ids": avail,
#             "hint": "Populate cameras table and set RTSP_URL_TEMPLATE like your working RTSP URL (replace IP with {ip}).",
#         }

#     if selected is None:
#         selected_ids = sorted(streams_map.keys())
#     else:
#         missing = [int(cid) for cid in selected if int(cid) not in streams_map]
#         if missing:
#             avail = sorted(_get_streams_map(None).keys())
#             return {
#                 "status": "error",
#                 "message": "Some requested camera_ids were not found (or are inactive).",
#                 "missing_camera_ids": missing,
#                 "available_camera_ids": avail,
#                 "hint": "If you need inactive cameras too, set ONLY_ACTIVE_CAMERAS=0",
#             }
#         selected_ids = [int(cid) for cid in selected]

#     with db_session() as db:
#         try:
#             member = db.query(Member).filter(Member.id == int(member_id)).first()
#             if not member:
#                 return {"status": "error", "message": f"member id {member_id} not found"}
#             member_name = _get_member_display_name(member, fallback=f"member_{member_id}")
#         except Exception:
#             db.rollback()
#             logger.exception("DB error during member lookup (start_extraction).")
#             return {"status": "error", "message": "DB error"}

#     if yolo_model is None or (USE_FACE and face_app is None) or (not reid_extractors):
#         init_models()

#     stop_event.clear()
#     with frames_lock:
#         latest_frames.clear()
#     for d in (capture_threads, extract_threads, cap_queues):
#         d.clear()

#     current_member_id = int(member_id)
#     current_member_name = member_name
#     current_camera_ids = list(selected_ids)
#     current_camera_streams = {cid: streams_map[cid] for cid in selected_ids}

#     _progress_reset(int(member_id), member_name, OVERALL_CAMERA_ID)
#     _progress_set(int(member_id), OVERALL_CAMERA_ID, stage="Starting", message="initializing", percent=0)

#     start_workers_if_needed()

#     for cid in selected_ids:
#         rtsp_url = streams_map[cid]
#         cap_queues[cid] = Queue(maxsize=CAP_QUEUE_MAX)

#         t_cap = threading.Thread(target=capture_thread_fn, name=f"cap_cam_{cid}", args=(cid, rtsp_url), daemon=True)
#         t_ext = threading.Thread(target=embedding_loop_for_cam, name=f"ext_cam_{cid}", args=(cid, rtsp_url), daemon=True)

#         capture_threads[cid] = t_cap
#         extract_threads[cid] = t_ext

#         t_cap.start()
#         t_ext.start()

#     if show_viewer and (viewer_thread is None or not viewer_thread.is_alive()):
#         viewer_thread = threading.Thread(target=viewer_loop, name="viewer", daemon=True)
#         viewer_thread.start()

#     is_running = True
#     _progress_set(int(member_id), OVERALL_CAMERA_ID, stage="running", message="processing", percent=0)
#     logger.info("Extraction started: member_id=%s name=%s camera_ids=%s", member_id, member_name, selected_ids)

#     return {
#         "status": "ok",
#         "message": "started",
#         "num_cams": len(selected_ids),
#         "member_id": int(member_id),
#         "member_name": member_name,
#         "camera_ids": selected_ids,
#         "rtsp_streams": {cid: streams_map[cid] for cid in selected_ids},
#     }


# def viewer_loop():
#     logger.info("[Viewer] started")
#     window_name = "Multi-RTSP Viewer"
#     window_created = False
#     try:
#         while not stop_event.is_set():
#             with frames_lock:
#                 frames = [latest_frames[idx] for idx in sorted(latest_frames.keys()) if latest_frames.get(idx) is not None]

#             if not frames:
#                 time.sleep(0.01)
#                 continue

#             if not window_created:
#                 try:
#                     cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
#                     window_created = True
#                 except Exception:
#                     pass

#             min_h = min(f.shape[0] for f in frames)
#             resized = []
#             for f in frames:
#                 h, w = f.shape[:2]
#                 if h != min_h:
#                     new_w = int(w * (min_h / h))
#                     f = cv2.resize(f, (new_w, min_h))
#                 resized.append(f)

#             try:
#                 combined = np.concatenate(resized, axis=1)
#             except Exception:
#                 time.sleep(0.01)
#                 continue

#             try: 
#                 cv2.imshow(window_name, combined)
#             except Exception:
#                 window_created = False
#                 time.sleep(0.01)
#                 continue

#             key = cv2.waitKey(1) & 0xFF
#             if key in (ord("q"), 27):
#                 logger.info("[Viewer] key pressed, stopping...")
#                 stop_event.set()
#                 break

#             time.sleep(VIEWER_SLEEP_MS / 1000.0)
#     finally:
#         try:
#             cv2.destroyAllWindows()
#         except Exception:
#             pass
#         logger.info("[Viewer] stopped")


# def stop_extraction(reason: str = "user") -> dict:
#     global is_running, current_member_id, current_member_name, current_camera_ids, current_camera_streams
#     try:
#         mid_snapshot = current_member_id
#         mname_snapshot = current_member_name
#         cams_snapshot = list(current_camera_ids)

#         stop_event.set()

#         for d in (capture_threads, extract_threads):
#             for _cid, th in list(d.items()):
#                 try:
#                     th.join(timeout=1.5)
#                 except Exception:
#                     pass

#         try:
#             if io_queue is not None:
#                 io_queue.join()
#         except Exception:
#             pass

#         with frames_lock:
#             latest_frames.clear()
#         capture_threads.clear()
#         extract_threads.clear()
#         cap_queues.clear()

#         is_running = False
#         logger.info("Extraction stopped (%s).", reason)

#         if mid_snapshot is not None and cams_snapshot:
#             if not reid_extractors or (USE_FACE and face_app is None):
#                 init_models()
#             threading.Thread(
#                 target=_extract_worker_multi,
#                 args=(int(mid_snapshot), str(mname_snapshot or "unknown"), cams_snapshot),
#                 name=f"auto-extract-{mid_snapshot}",
#                 daemon=True,
#             ).start()
#             logger.info("Auto-extract triggered for member_id=%s cameras=%s after stop.", mid_snapshot, cams_snapshot)

#         return {"status": "ok", "message": f"Stopped capturing ", "member_id": mid_snapshot, "camera_ids": cams_snapshot}
#     finally:
#         current_member_id = None
#         current_member_name = None
#         current_camera_ids = []
#         current_camera_streams = {}


# def remove_embeddings(member_id: int, camera_id: Optional[int] = None) -> Dict[str, object]:
#     with db_session() as db:
#         try:
#             q = db.query(MemberEmbedding).filter(MemberEmbedding.member_id == int(member_id))
#             if camera_id is not None:
#                 q = q.filter(MemberEmbedding.camera_id == int(camera_id))

#             rows = q.all()
#             if not rows:
#                 return {"status": "error", "message": "no embeddings rows found", "member_id": int(member_id), "camera_id": camera_id}

#             for rec in rows:
#                 rec.body_embedding = None
#                 rec.face_embedding = None
#                 rec.back_body_embedding = None
#                 rec.body_embeddings_raw = None
#                 rec.face_embeddings_raw = None
#                 rec.back_body_embeddings_raw = None
#                 rec.last_embedding_update_ts = datetime.now(timezone.utc)

#             db.commit()
#             logger.warning("Embeddings removed for member_id=%s camera_id=%s (rows=%d)", member_id, camera_id, len(rows))
#             return {"status": "ok", "member_id": int(member_id), "camera_id": camera_id, "rows": len(rows)}
#         except Exception:
#             db.rollback()
#             logger.exception("remove embeddings: DB error")
#             return {"status": "error", "message": "DB error", "member_id": int(member_id), "camera_id": camera_id}


# def get_status() -> Dict[str, object]:
#     streams_map = _get_streams_map(None)
#     return {
#         "running": bool(is_running),
#         "configured_num_cams": len(streams_map),
#         "configured_camera_ids": sorted(streams_map.keys()),
#         "running_camera_ids": list(current_camera_ids),
#         "rtsp_streams": streams_map,
#         "member_id": current_member_id,
#         "member_name": current_member_name,
#     }


#-----------------------------------------------------------------------------------------------------------------------------------

from __future__ import annotations

import csv
import sys
import os
from dotenv import load_dotenv
from pathlib import Path
import time
import threading
from datetime import datetime, timezone
from typing import Optional, List, Dict, Tuple, Callable, Any
from queue import Queue, Empty
import logging
import shutil
import random
import io
import gzip
import zlib
import re
import inspect
from contextlib import contextmanager
from typing import Generator

import cv2
import numpy as np
import torch

from app.core.constants import (
    RTSP_STREAMS,
    YOLO_WEIGHTS,
    DEVICE,
    CONF_THRES,
    IOU_THRES,
    EMB_CSV,
    CROPS_ROOT,
)
from app.db.session import get_db, engine  # engine kept for compatibility
from app.db.models import Member, MemberEmbedding

try:
    from app.db.models import Camera
except Exception:
    Camera = None  # type: ignore


BASE_DIR = Path(__file__).resolve().parents[2]
load_dotenv(BASE_DIR / ".env")

# ===================== DB Session Wrapper =====================

@contextmanager
def db_session():
    """
    Works with both styles:
      - get_db() returns Session (old style)
      - get_db() yields Session (FastAPI dependency style -> generator)
    """
    db_obj: Any = get_db()

    # FastAPI dependency style: generator that yields a Session
    if inspect.isgenerator(db_obj):
        gen: Generator = db_obj  # type: ignore
        try:
            db = next(gen)  # actual Session
        except StopIteration:
            raise RuntimeError("get_db() did not yield a DB session")
        try:
            yield db
        finally:
            try:
                gen.close()
            except Exception:
                pass
        return

    # Old style: get_db() returned an actual Session
    db = db_obj
    try:
        yield db
    finally:
        try:
            db.close()
        except Exception:
            pass


# ===================== Runtime perf knobs =====================

try:
    torch.backends.cudnn.benchmark = True
except Exception:
    pass
try:
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
except Exception:
    pass
try:
    cv2.setNumThreads(1)
except Exception:
    pass


# ===================== Optional libs =====================

try:
    from ultralytics import YOLO
except Exception:
    YOLO = None

try:
    from torchreid.utils import FeatureExtractor as TorchreidExtractor
except Exception as e:
    print("-----------------TorchReID import error:", e)
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


# ===================== Config =====================

EXPECTED_EMBED_DIM = 512

RAW_STORE_LIMIT = int(os.getenv("RAW_STORE_LIMIT", "256"))

USE_FACE = True
FACE_MODEL = "buffalo_l"
FACE_DET_SIZE = (1280, 1280)
FACE_PROVIDER = "auto"
FACE_MIN_SCORE = 0.5
FACE_MIN_SIZE = 32
FACE_IOU_LINK = 0.05
FACE_OVER_FACE_LINK = 0.60
FACE_EVERY = int(os.getenv("FACE_EVERY", "1"))

YOLO_IMGSZ = int(os.getenv("YOLO_IMGSZ", "1280"))

CAP_QUEUE_MAX = int(os.getenv("CAP_QUEUE_MAX", "2"))
IO_QUEUE_MAX = int(os.getenv("IO_QUEUE_MAX", "1024"))

VIEWER_SLEEP_MS = int(os.getenv("VIEWER_SLEEP_MS", "5"))

REID_MODELS = ["osnet_x1_0", "osnet_x0_25"]
BODY_TTA_FLIP = True
FACE_TTA_FLIP = True

BACK_BODY_GALLERY_NAME = "gallery_body"
BACK_HEAD_CUT_RATIO = float(os.getenv("BACK_HEAD_CUT_RATIO", "0.0"))
BACK_SHAPE_SEED = int(os.getenv("BACK_SHAPE_SEED", "1337"))

BACK_REID_WEIGHT = float(os.getenv("BACK_REID_WEIGHT", "0.65"))
BACK_SHAPE_WEIGHT = float(os.getenv("BACK_SHAPE_WEIGHT", "0.35"))

BACK_HOG_WIN = (64, 128)
BACK_HOG_BLOCK = (16, 16)
BACK_HOG_STRIDE = (8, 8)
BACK_HOG_CELL = (8, 8)
BACK_HOG_BINS = 9

SAVE_MIN_INTERVAL_MS = int(os.getenv("SAVE_MIN_INTERVAL_MS", "200"))
SAVE_MAX_DETS_PER_FRAME = int(os.getenv("SAVE_MAX_DETS_PER_FRAME", "3"))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", "85"))

YOLO_REQ_QUEUE_MAX = int(os.getenv("YOLO_REQ_QUEUE_MAX", "64"))
YOLO_BATCH_MAX = int(os.getenv("YOLO_BATCH_MAX", "8"))
YOLO_BATCH_WAIT_MS = int(os.getenv("YOLO_BATCH_WAIT_MS", "10"))
YOLO_RESP_TIMEOUT_S = float(os.getenv("YOLO_RESP_TIMEOUT_S", "0.7"))

MAX_BODY_IMAGES = int(os.getenv("MAX_BODY_IMAGES", "600"))
MAX_FACE_IMAGES = int(os.getenv("MAX_FACE_IMAGES", "600"))
MAX_BACK_BODY_IMAGES = int(os.getenv("MAX_BACK_BODY_IMAGES", "600"))
GALLERY_SAMPLE_SEED = int(os.getenv("GALLERY_SAMPLE_SEED", "1337"))

# Debug aids (default off)
DEBUG_SAVE_NO_DETS_EVERY_SEC = float(os.getenv("DEBUG_SAVE_NO_DETS_EVERY_SEC", "0"))  # 0 disables
DEBUG_LOG_NO_DET_EVERY = int(os.getenv("DEBUG_LOG_NO_DET_EVERY", "150"))

# ===================== DB Camera -> RTSP =====================

USE_DB_CAMERA_CONFIG = os.getenv("USE_DB_CAMERA_CONFIG", "1").lower() in ("1", "true", "yes", "y", "on")
ONLY_ACTIVE_CAMERAS = os.getenv("ONLY_ACTIVE_CAMERAS", "1").lower() in ("1", "true", "yes", "y", "on")

# Desired format:
# rtsp://admin:Admin%40123@192.168.1.161:554/video/live?channel=1&subtype=0
#
# In the cameras table, Camera.ip_address should contain only the IP, for example:
# 192.168.1.161
#
# You can still store a full rtsp:// URL in Camera.ip_address; if you do, it is returned unchanged.
RTSP_URL_TEMPLATE = os.getenv(
    "RTSP_URL_TEMPLATE",
    "rtsp://{username}:{password}@{ip}:{port}/video/live?channel={channel}&subtype={subtype}",
).strip()

RTSP_USERNAME = os.getenv("RTSP_USERNAME", os.getenv("RTSP_USER", "admin")).strip()
RTSP_PASSWORD = os.getenv("RTSP_PASSWORD", os.getenv("RTSP_PASS", "Admin%40123")).strip()
RTSP_PORT = os.getenv("RTSP_PORT", "554").strip()
RTSP_PATH = os.getenv("RTSP_PATH", "/video/live").strip() or "/video/live"
RTSP_SCHEME = os.getenv("RTSP_SCHEME", "rtsp").strip() or "rtsp"
RTSP_CHANNEL = os.getenv("RTSP_CHANNEL", "1").strip() or "1"
RTSP_SUBTYPE = os.getenv("RTSP_SUBTYPE", "0").strip() or "0"

# Kept for compatibility with your older Streaming/channels template.
RTSP_STREAM = os.getenv("RTSP_STREAM", "").strip() or ""


# ===================== Logging =====================

logger = logging.getLogger("app.embedding_service")
if not logger.handlers:
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("[%(levelname)s] %(message)s"))
    logger.addHandler(ch)
logger.setLevel(logging.INFO)


# ===================== Globals =====================

yolo_model: YOLO | None = None
reid_extractor: TorchreidExtractor | None = None
reid_extractors: List[TorchreidExtractor] = []
face_app: FaceAnalysis | None = None

reid_lock = threading.Lock()
csv_lock = threading.Lock()
frames_lock = threading.Lock()

latest_frames: Dict[int, np.ndarray] = {}

capture_threads: Dict[int, threading.Thread] = {}
extract_threads: Dict[int, threading.Thread] = {}
cap_queues: Dict[int, Queue] = {}

io_thread: Optional[threading.Thread] = None
io_queue: Queue | None = None

yolo_thread: Optional[threading.Thread] = None
yolo_req_queue: Queue | None = None

viewer_thread: Optional[threading.Thread] = None
stop_event = threading.Event()
is_running = False

current_member_id: Optional[int] = None
current_member_name: Optional[str] = None
current_camera_ids: List[int] = []
current_camera_streams: Dict[int, str] = {}

progress_lock = threading.Lock()
progress_state: Dict[Tuple[int, int], Dict[str, object]] = {}

_back_hog = None
_back_proj = None
_back_proj_in_dim = None

OVERALL_CAMERA_ID = -1


# ===================== Cropping =====================

def crop_with_padding(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = bbox

    x1 = max(0, min(W - 1, int(x1)))
    y1 = max(0, min(H - 1, int(y1)))
    x2 = max(0, min(W, int(x2)))
    y2 = max(0, min(H, int(y2)))

    if x2 <= x1 or y2 <= y1:
        return np.zeros((0, 0, 3), dtype=np.uint8)

    return frame[y1:y2, x1:x2].copy()


def crop_back_body_no_face(frame: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    h = max(0, int(y2) - int(y1))
    if h <= 0:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    y1b = int(y1 + h * float(max(0.0, min(0.45, BACK_HEAD_CUT_RATIO))))
    if y1b >= y2 - 2:
        return np.zeros((0, 0, 3), dtype=np.uint8)
    return crop_with_padding(frame, (x1, y1b, x2, y2))


# ===================== IO Worker =====================

def start_workers_if_needed() -> None:
    global io_thread, io_queue, yolo_thread, yolo_req_queue

    if io_queue is None:
        io_queue = Queue(maxsize=IO_QUEUE_MAX)

    if io_thread is None or not io_thread.is_alive():
        io_thread = threading.Thread(target=io_worker, name="io_worker", daemon=True)
        io_thread.start()

    if yolo_req_queue is None:
        yolo_req_queue = Queue(maxsize=YOLO_REQ_QUEUE_MAX)

    if yolo_thread is None or not yolo_thread.is_alive():
        if yolo_model is not None:
            yolo_thread = threading.Thread(target=yolo_worker, name="yolo_worker", daemon=True)
            yolo_thread.start()


def io_worker() -> None:
    ensure_csv_header()
    csv_path = Path(EMB_CSV)
    _ensure_dir(csv_path.parent if csv_path.parent.as_posix() not in (".", "") else Path("."))

    f = None
    writer = None
    flush_every = 128
    pending = 0
    last_flush = time.time()

    try:
        f = open(csv_path, "a", newline="", encoding="utf-8")
        writer = csv.writer(f)

        while True:
            if stop_event.is_set() and (io_queue is None or io_queue.empty()):
                break

            try:
                job = io_queue.get(timeout=0.1)  # type: ignore
            except Empty:
                if pending and (time.time() - last_flush) > 1.0 and f is not None:
                    try:
                        f.flush()
                    except Exception:
                        pass
                    pending = 0
                    last_flush = time.time()
                continue
            except Exception:
                break

            try:
                crop_path: Path = job["crop_path"]
                _ensure_dir(crop_path.parent)

                img = job["image"]
                ok = False
                if str(crop_path).lower().endswith((".jpg", ".jpeg")):
                    ok = bool(cv2.imwrite(str(crop_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), int(JPEG_QUALITY)]))
                else:
                    ok = bool(cv2.imwrite(str(crop_path), img))

                if not ok:
                    logger.error("cv2.imwrite failed: %s", crop_path)

                if writer is not None:
                    writer.writerow(job["csv_row"])
                    pending += 1

                if f is not None and (pending >= flush_every or (time.time() - last_flush) > 1.0):
                    try:
                        f.flush()
                    except Exception:
                        pass
                    pending = 0
                    last_flush = time.time()
            except Exception:
                logger.exception("IO worker error")
            finally:
                try:
                    io_queue.task_done()  # type: ignore
                except Exception:
                    pass
    finally:
        try:
            if f is not None:
                f.flush()
                f.close()
        except Exception:
            pass


# ===================== YOLO Batch Worker =====================

def yolo_worker() -> None:
    if yolo_model is None:
        return

    gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
    device = DEVICE if gpu else "cpu"

    while True:
        if stop_event.is_set() and (yolo_req_queue is None or yolo_req_queue.empty()):
            break
        try:
            req0 = yolo_req_queue.get(timeout=0.1)  # type: ignore
        except Empty:
            continue
        except Exception:
            break

        batch = [req0]
        t0 = time.time()
        while len(batch) < max(1, int(YOLO_BATCH_MAX)) and ((time.time() - t0) * 1000.0) < float(YOLO_BATCH_WAIT_MS):
            try:
                batch.append(yolo_req_queue.get_nowait())  # type: ignore
            except Empty:
                break
            except Exception:
                break

        frames = [b["frame"] for b in batch]
        try:
            with torch.inference_mode():
                results = yolo_model.predict(
                    frames,
                    conf=float(CONF_THRES),
                    iou=float(IOU_THRES),
                    imgsz=int(YOLO_IMGSZ),
                    verbose=False,
                    device=device,
                )
        except Exception:
            logger.exception("YOLO batch failed")
            results = [None for _ in frames]

        for req, res in zip(batch, results):
            H, W = int(req["H"]), int(req["W"])
            outs: List[tuple[float, float, float, float, float]] = []

            try:
                boxes = getattr(res, "boxes", None) if res is not None else None
                if boxes is not None:
                    xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
                    confs = boxes.conf.detach().cpu().numpy().astype(np.float32)
                    cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
                    keep = (cls == 0)
                    xyxy, confs = xyxy[keep], confs[keep]

                    for (x1, y1, x2, y2), c in zip(xyxy, confs):
                        x1f = float(max(0, min(W - 1, x1)))
                        y1f = float(max(0, min(H - 1, y1)))
                        x2f = float(max(0, min(W - 1, x2)))
                        y2f = float(max(0, min(H - 1, y2)))
                        if (x2f - x1f) < 4 or (y2f - y1f) < 4:
                            continue
                        outs.append((x1f, y1f, x2f, y2f, float(c)))
            except Exception:
                outs = []

            try:
                respq: Queue = req["resp"]
                respq.put_nowait(outs)
            except Exception:
                pass

            try:
                yolo_req_queue.task_done()  # type: ignore
            except Exception:
                pass


def yolo_infer(frame: np.ndarray, H: int, W: int, timeout_s: float = YOLO_RESP_TIMEOUT_S) -> List[tuple[float, float, float, float, float]]:
    if yolo_model is None or yolo_req_queue is None:
        return []
    resp = Queue(maxsize=1)
    try:
        yolo_req_queue.put_nowait({"frame": frame, "H": H, "W": W, "resp": resp})
    except Exception:
        return []
    try:
        return resp.get(timeout=float(timeout_s))
    except Exception:
        return []


# ===================== Progress helpers =====================

def _pkey(member_id: int, camera_id: int) -> Tuple[int, int]:
    return (int(member_id), int(camera_id))


def _progress_reset(member_id: int, member_name: str, camera_id: int) -> None:
    with progress_lock:
        progress_state[_pkey(member_id, camera_id)] = {
            "member_id": int(member_id),
            "member_name": member_name,
            "camera_id": int(camera_id),
            "stage": "idle",
            "total_body": 0,
            "total_face": 0,
            "total_back_body": 0,
            "done_body": 0,
            "done_face": 0,
            "done_back_body": 0,
            "work_total": 0,
            "work_done": 0,
            "percent": 0,
            "message": "waiting",
        }


def _progress_set(member_id: int, camera_id: int, **kv) -> None:
    with progress_lock:
        st = progress_state.get(_pkey(member_id, camera_id))
        if not st:
            st = {}
            progress_state[_pkey(member_id, camera_id)] = st
        st.update(kv)

        tb = int(st.get("total_body", 0))
        tf = int(st.get("total_face", 0))
        tbb = int(st.get("total_back_body", 0))

        db_ = int(st.get("done_body", 0))
        df_ = int(st.get("done_face", 0))
        dbb = int(st.get("done_back_body", 0))

        db_ = max(0, min(db_, tb))
        df_ = max(0, min(df_, tf))
        dbb = max(0, min(dbb, tbb))

        st["done_body"], st["done_face"], st["done_back_body"] = db_, df_, dbb

        work_total = int(st.get("work_total", 0))
        work_done = int(st.get("work_done", 0))
        work_done = max(0, min(work_done, max(0, work_total)))
        st["work_done"] = work_done

        stage = str(st.get("stage", "")).lower()

        if stage.startswith("embedding_") and work_total > 0:
            pct = int(round((work_done / max(1, work_total)) * 100))
        elif stage == "body embeddings" and tb > 0:
            pct = int(round((db_ / max(1, tb)) * 100))
        elif stage == "face embeddings" and tf > 0:
            pct = int(round((df_ / max(1, tf)) * 100))
        elif stage in ("back body embeddings", "embedding_back") and tbb > 0:
            pct = int(round((dbb / max(1, tbb)) * 100))
        else:
            tot = max(1, tb + tf + tbb)
            pct = int(round(((db_ + df_ + dbb) / tot) * 100))

        st["percent"] = max(0, min(100, pct))


def _default_progress(member_id: int, camera_id: int) -> Dict[str, object]:
    return {
        "member_id": int(member_id),
        "camera_id": int(camera_id),
        "stage": "unknown",
        "percent": 0,
        "message": "no job",
        "total_body": 0,
        "total_face": 0,
        "total_back_body": 0,
        "done_body": 0,
        "done_face": 0,
        "done_back_body": 0,
        "work_total": 0,
        "work_done": 0,
    }


def get_progress_for_member(member_id: int, camera_id: Optional[int] = None) -> Dict[str, object]:
    mid = int(member_id)
    with progress_lock:
        if camera_id is not None:
            st = progress_state.get(_pkey(mid, int(camera_id)))
            return dict(st) if st else _default_progress(mid, int(camera_id))

        st_overall = progress_state.get(_pkey(mid, OVERALL_CAMERA_ID))
        if st_overall:
            return dict(st_overall)

        cams = {cid: dict(st) for (m, cid), st in progress_state.items() if m == mid and cid != OVERALL_CAMERA_ID}
        if len(cams) == 1:
            return next(iter(cams.values()))
        if cams:
            return {"member_id": mid, "stage": "multi", "cameras": cams}
        return _default_progress(mid, OVERALL_CAMERA_ID)


# ===================== Utils =====================

def l2_normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float32).reshape(-1)
    n = float(np.linalg.norm(v))
    if n == 0.0 or not np.isfinite(n):
        return v
    return v / n


def _as_512f(vec: np.ndarray | list | None) -> Optional[np.ndarray]:
    if vec is None:
        return None
    try:
        a = np.asarray(vec, dtype=np.float32).reshape(-1)
        if a.size != EXPECTED_EMBED_DIM:
            return None
        if not np.isfinite(a).all():
            return None
        return l2_normalize(a)
    except Exception:
        return None


def _pack_raw_bank(vectors: Optional[List[np.ndarray]]) -> Optional[bytes]:
    if not vectors:
        return None
    cleaned: List[np.ndarray] = []
    for v in vectors[: max(1, int(RAW_STORE_LIMIT))]:
        vv = _as_512f(v)
        if vv is not None:
            cleaned.append(vv.astype(np.float32))
    if not cleaned:
        return None
    arr = np.stack(cleaned, axis=0).astype(np.float32)
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return gzip.compress(buf.getvalue(), compresslevel=6)


def _to_rgb(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)


def _safe_slug(s: str, max_len: int = 64) -> str:
    s = (s or "").strip()
    s = re.sub(r"[^\w\-\.]+", "_", s, flags=re.UNICODE)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        s = "member"
    return s[:max_len]


# ===================== Back-body shape embedding =====================

def _init_back_shape_embedder() -> None:
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
            logger.info("Back-body HOG ready (dim=%s)", int(_back_hog.getDescriptorSize()))
        except Exception:
            logger.exception("Failed to init HOG descriptor for back-body embeddings")
            _back_hog = None


def _get_back_proj(in_dim: int) -> Optional[np.ndarray]:
    global _back_proj, _back_proj_in_dim
    if in_dim <= 0:
        return None
    if _back_proj is None or _back_proj_in_dim != int(in_dim):
        rng = np.random.RandomState(int(BACK_SHAPE_SEED))
        mat = rng.normal(
            loc=0.0,
            scale=float(1.0 / max(1.0, np.sqrt(in_dim))),
            size=(EXPECTED_EMBED_DIM, in_dim),
        ).astype(np.float32)
        _back_proj = mat
        _back_proj_in_dim = int(in_dim)
        logger.info("Back-body projection matrix built (seed=%s, in_dim=%s)", BACK_SHAPE_SEED, in_dim)
    return _back_proj


def _back_shape_embed(img_bgr: np.ndarray) -> Optional[np.ndarray]:
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
        proj = _get_back_proj(int(hog.size))
        if proj is None:
            return None
        emb = proj @ hog
        return _as_512f(emb)
    except Exception:
        return None


# ===================== Geometry helpers =====================

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
    return inter / denom if denom > 0 else 0.0


def inter_over_face(a: Tuple[float, float, float, float], b: Tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1, inter_y1 = max(ax1, bx1), max(ay1, by1)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, inter_x2 - inter_x1), max(0.0, inter_y2 - inter_y1)
    inter = iw * ih
    face_area = max(0.0, (bx2 - bx1)) * max(0.0, (by2 - by1))
    return inter / face_area if face_area > 0 else 0.0


def face_center_in(a: Tuple[int, int, int, int], b: Tuple[float, float, float, float]) -> bool:
    fx1, fy1, fx2, fy2 = a
    px1, py1, px2, py2 = b
    cx = (fx1 + fx2) * 0.5
    cy = (fy1 + fy2) * 0.5
    return (px1 <= cx <= px2) and (py1 <= cy <= py2)


def safe_iter_faces(obj: Any) -> List[Any]:
    if obj is None:
        return []
    try:
        return list(obj)
    except TypeError:
        return [obj]


def extract_face_embedding(face: Any):
    emb = getattr(face, "normed_embedding", None)
    if emb is None:
        emb = getattr(face, "embedding", None)
    return emb


def _cuda_ep_loadable() -> bool:
    if ort is None:
        return False
    try:
        if sys.platform.startswith("mac"):
            return False
        from pathlib import Path as _P
        import ctypes as _ct
        capi_dir = _P(ort.__file__).parent / "capi"
        name = "onnxruntime_providers_cuda.dll" if os.name == "nt" else "libonnxruntime_providers_cuda.so"
        lib_path = capi_dir / name
        if not lib_path.exists():
            return False
        _ct.CDLL(str(lib_path))
        return True
    except Exception:
        return False


# ===================== Models init =====================

def init_face_engine(use_face: bool, device: str, face_model: str, det_w: int, det_h: int, face_provider: str):
    if not use_face:
        return None
    if not INSIGHT_OK:
        logger.warning("insightface not installed; face recognition disabled.")
        return None
    try:
        is_cuda = ("cuda" in device.lower()) and torch.cuda.is_available()
        cuda_ok = _cuda_ep_loadable()

        providers = ["CPUExecutionProvider"]
        if face_provider == "cuda":
            if cuda_ok:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif face_provider == "auto":
            if is_cuda and cuda_ok:
                providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

        app = FaceAnalysis(name=face_model, providers=providers)
        ctx_id = 0 if providers[0].startswith("CUDA") else -1
        try:
            app.prepare(ctx_id=ctx_id, det_size=(det_w, det_h))
        except TypeError:
            app.prepare(ctx_id=ctx_id)

        logger.info("InsightFace ready (model=%s, providers=%s).", face_model, providers)

        try:
            dummy = np.zeros((max(8, det_h), max(8, det_w), 3), dtype=np.uint8)
            app.get(dummy)
        except Exception:
            pass
        return app
    except Exception:
        logger.exception("InsightFace init failed")
        return None


def init_models() -> None:
    global yolo_model, reid_extractor, reid_extractors, face_app

    gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
    logger.info("init_models device=%s gpu_available=%s", DEVICE, torch.cuda.is_available())

    # YOLO
    if YOLO is not None:
        try:
            weights = YOLO_WEIGHTS
            if not Path(weights).exists():
                logger.warning("%s not found, falling back to yolov8n.pt", weights)
            yolo = YOLO(weights if Path(YOLO_WEIGHTS).exists() else "yolov8n.pt")
            if gpu:
                try:
                    yolo.to(DEVICE)
                except Exception:
                    pass
            try:
                yolo.fuse()
            except Exception:
                pass
            try:
                with torch.inference_mode():
                    _dummy = np.zeros((YOLO_IMGSZ, YOLO_IMGSZ, 3), dtype=np.uint8)
                    yolo.predict(
                        _dummy,
                        device=DEVICE if gpu else "cpu",
                        conf=0.25,
                        iou=0.5,
                        imgsz=YOLO_IMGSZ,
                        verbose=False,
                    )
            except Exception:
                pass
            yolo_model = yolo
            logger.info("YOLO ready (gpu=%s)", gpu)
        except Exception:
            logger.exception("YOLO load failed; detection disabled.")
            yolo_model = None
    else:
        logger.warning("ultralytics not installed; detection disabled.")
        yolo_model = None

    # TorchReID ensemble
    reid_extractors.clear()
    if TorchreidExtractor is not None:
        try:
            dev = DEVICE if gpu else "cpu"
            for m in REID_MODELS:
                try:
                    ext = TorchreidExtractor(model_name=m, device=dev)
                    try:
                        dummy = np.zeros((256, 128, 3), dtype=np.uint8)
                        _ = ext([dummy])
                    except Exception:
                        pass
                    reid_extractors.append(ext)
                    logger.info("TorchReID ready: %s (%s)", m, dev)
                except Exception:
                    logger.exception("TorchReID init failed for %s", m)
            reid_extractor = reid_extractors[0] if reid_extractors else None
        except Exception:
            logger.exception("TorchReID init failed; body embeddings disabled.")
            reid_extractors.clear()
            reid_extractor = None
    else:
        logger.warning("torchreid not installed; body embeddings disabled.")
        reid_extractors.clear()
        reid_extractor = None

    # InsightFace
    face_app = init_face_engine(USE_FACE, DEVICE, FACE_MODEL, FACE_DET_SIZE[0], FACE_DET_SIZE[1], FACE_PROVIDER)

    # Back-body init
    _init_back_shape_embedder()

    # Workers
    start_workers_if_needed()


# ===================== CSV + gallery helpers =====================

def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def ensure_csv_header() -> None:
    csv_path = Path(EMB_CSV)
    _ensure_dir(csv_path.parent if csv_path.parent.as_posix() not in (".", "") else Path("."))

    expected_header = [
        "member_id", "member_name", "ts", "camera_id", "frame_idx", "det_idx",
        "x1", "y1", "x2", "y2", "conf_or_score",
        "body_embedding", "face_embedding", "crop_path", "kind",
    ]

    def _write_new():
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(expected_header)

    if not csv_path.exists():
        with csv_lock:
            if not csv_path.exists():
                _write_new()
                logger.info("Created CSV header: %s", EMB_CSV)
        return

    try:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            hdr = next(r, None)
    except Exception:
        hdr = None

    if hdr == expected_header:
        return

    with csv_lock:
        try:
            with open(csv_path, "r", newline="", encoding="utf-8") as f:
                r = csv.reader(f)
                hdr2 = next(r, None)
        except Exception:
            hdr2 = None
        if hdr2 == expected_header:
            return

        bak = csv_path.with_suffix(csv_path.suffix + f".bak_{int(time.time())}")
        try:
            shutil.copy2(csv_path, bak)
            logger.warning("CSV header mismatch. Backed up old CSV to: %s", bak)
        except Exception:
            logger.warning("CSV header mismatch, but backup failed. Rewriting header anyway.")
        _write_new()
        logger.info("Rewrote CSV header: %s", EMB_CSV)


def _member_folder(member_id: int, member_name: Optional[str] = None) -> str:
    mid = int(member_id)
    if member_name:
        return f"{mid}_{_safe_slug(str(member_name))}"
    return str(mid)


def _gallery_body_dir(member_id: int, camera_id: int, member_name: Optional[str] = None) -> Path:
    d = Path(CROPS_ROOT) / "gallery" / _member_folder(member_id, member_name) / f"cam_{int(camera_id)}"
    _ensure_dir(d)
    return d


def _gallery_face_dir(member_id: int, camera_id: int, member_name: Optional[str] = None) -> Path:
    d = Path(CROPS_ROOT) / "gallery_face" / _member_folder(member_id, member_name) / f"cam_{int(camera_id)}"
    _ensure_dir(d)
    return d


def _gallery_back_body_dir(member_id: int, camera_id: int, member_name: Optional[str] = None) -> Path:
    d = Path(CROPS_ROOT) / BACK_BODY_GALLERY_NAME / _member_folder(member_id, member_name) / f"cam_{int(camera_id)}"
    _ensure_dir(d)
    return d


def _epoch_ms() -> int:
    return int(time.time() * 1000) * 1000 + random.randint(0, 999)


# ===================== Face helpers (capture-time) =====================

def detect_faces_raw(frame: np.ndarray):
    outs: List[Dict] = []
    if face_app is None:
        return outs
    try:
        faces = face_app.get(np.ascontiguousarray(frame))
        for f in safe_iter_faces(faces):
            bbox = getattr(f, "bbox", None)
            if bbox is None:
                continue
            b = np.asarray(bbox).reshape(-1)
            if b.size < 4:
                continue
            x1, y1, x2, y2 = map(float, b[:4])
            score = float(getattr(f, "det_score", 0.0))
            outs.append({"bbox": (x1, y1, x2, y2), "score": score})
    except Exception:
        logger.exception("FaceAnalysis error")
    return outs


def link_face_to_person(face_bbox: Tuple[float, float, float, float], person_bbox: Tuple[int, int, int, int]) -> bool:
    if face_center_in(tuple(map(int, face_bbox)), person_bbox):
        return True
    if iou_xyxy(person_bbox, face_bbox) >= FACE_IOU_LINK:
        return True
    if inter_over_face(person_bbox, face_bbox) >= FACE_OVER_FACE_LINK:
        return True
    return False


def _face_big_enough(face_bbox: Tuple[float, float, float, float]) -> bool:
    x1, y1, x2, y2 = face_bbox
    return min(float(x2 - x1), float(y2 - y1)) >= float(FACE_MIN_SIZE)


# ===================== RTSP helpers =====================

def _configure_rtsp_capture(cap: cv2.VideoCapture) -> None:
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass


def _read_latest_frame(cap: cv2.VideoCapture) -> Tuple[bool, Optional[np.ndarray]]:
    grabbed = cap.grab()
    if not grabbed:
        ok, fr = cap.read()
        return ok, (fr if ok else None)
    for _ in range(3):
        if not cap.grab():
            break
    ok, frame = cap.retrieve()
    return ok, (frame if ok else None)


# ===================== Stream / Camera mapping =====================

def _normalize_camera_ids(camera_ids: Any) -> Optional[List[int]]:
    if camera_ids is None:
        return None
    if isinstance(camera_ids, int):
        return [int(camera_ids)]
    if isinstance(camera_ids, str):
        parts = [p.strip() for p in camera_ids.split(",") if p.strip()]
        out: List[int] = []
        for p in parts:
            try:
                out.append(int(p))
            except Exception:
                continue
        return out if out else None
    if isinstance(camera_ids, (list, tuple, set)):
        out: List[int] = []
        for x in camera_ids:
            try:
                out.append(int(x))
            except Exception:
                continue
        return out if out else None
    return None


def _camera_streams_map(streams: Any) -> Dict[int, str]:
    out: Dict[int, str] = {}

    if isinstance(streams, str):
        s = streams.strip()
        return {0: s} if s else {}

    if isinstance(streams, dict):
        for k, v in streams.items():
            try:
                cid = int(k)
            except Exception:
                continue
            url = ""
            if isinstance(v, str):
                url = v
            elif isinstance(v, dict):
                url = str(v.get("url") or v.get("rtsp_url") or v.get("rtsp") or v.get("stream") or "")
            elif isinstance(v, (list, tuple)) and len(v) >= 1:
                url = str(v[0])
            if url:
                out[cid] = url
        return out

    if isinstance(streams, (list, tuple)):
        for idx, item in enumerate(streams):
            if isinstance(item, str):
                out[int(idx)] = item
            elif isinstance(item, dict):
                cid = int(item.get("id", idx))
                url = str(item.get("url") or item.get("rtsp_url") or item.get("rtsp") or item.get("stream") or "")
                if url:
                    out[cid] = url
            elif isinstance(item, (list, tuple)):
                if len(item) >= 2:
                    try:
                        cid = int(item[0])
                    except Exception:
                        cid = int(idx)
                    url = str(item[1])
                    if url:
                        out[cid] = url
                elif len(item) == 1:
                    out[int(idx)] = str(item[0])
    return out


def _get_member_display_name(member: Any, fallback: str = "unknown") -> str:
    for attr in ("first_name", "name", "full_name", "username"):
        v = getattr(member, attr, None)
        if v:
            return str(v)
    return fallback


def _build_rtsp_url(ip: str, camera_id: int, camera_name: str = "") -> str:
    """
    Build an RTSP URL from Camera.ip_address.

    Expected DB value:
        Camera.ip_address = "192.168.1.161"

    Output with the default RTSP_URL_TEMPLATE:
        rtsp://admin:Admin%40123@192.168.1.161:554/video/live?channel=1&subtype=0

    If Camera.ip_address already contains a full rtsp:// URL, it is returned unchanged.
    """
    ip = (ip or "").strip()
    if not ip:
        return ""

    # Allow storing full RTSP URL in ip_address field.
    if ip.lower().startswith(("rtsp://", "rtsps://")):
        return ip

    if RTSP_URL_TEMPLATE:
        try:
            return RTSP_URL_TEMPLATE.format(
                ip=ip,
                camera_id=int(camera_id),
                id=int(camera_id),
                name=str(camera_name or ""),
                user=RTSP_USERNAME,
                username=RTSP_USERNAME,
                password=RTSP_PASSWORD,
                port=RTSP_PORT,
                stream=RTSP_STREAM,
                channel=RTSP_CHANNEL,
                subtype=RTSP_SUBTYPE,
            )
        except Exception:
            logger.exception("RTSP_URL_TEMPLATE formatting failed; falling back to basic builder")

    auth = ""
    if RTSP_USERNAME:
        auth = RTSP_USERNAME
        if RTSP_PASSWORD:
            auth += f":{RTSP_PASSWORD}"
        auth += "@"

    port = f":{RTSP_PORT}" if RTSP_PORT else ""
    path = RTSP_PATH or "/video/live"
    if path and not path.startswith("/"):
        path = "/" + path

    if "?" in path:
        return f"{RTSP_SCHEME}://{auth}{ip}{port}{path}"

    return f"{RTSP_SCHEME}://{auth}{ip}{port}{path}?channel={RTSP_CHANNEL}&subtype={RTSP_SUBTYPE}"


def _db_get_camera_ids(
    camera_ids: Optional[List[int]] = None,
    only_active: Optional[bool] = None
) -> List[int]:
    if Camera is None:
        return []

    try:
        with db_session() as db:
            q = db.query(Camera.id)

            if camera_ids:
                q = q.filter(Camera.id.in_(map(int, camera_ids)))

            if only_active is True:
                q = q.filter(Camera.is_active.is_(True))

            rows = q.all()

        return sorted(r[0] for r in rows)

    except Exception as e:
        logger.exception("Failed to load camera ids from DB: %s", e)
        return []


def _camera_streams_from_db(camera_ids: Optional[List[int]] = None) -> Dict[int, str]:
    if Camera is None:
        return {}

    out: Dict[int, str] = {}
    try:
        with db_session() as db:
            q = db.query(Camera)
            if camera_ids:
                q = q.filter(Camera.id.in_([int(x) for x in camera_ids]))
            if ONLY_ACTIVE_CAMERAS:
                q = q.filter(Camera.is_active.is_(True))
            cams = q.all()
    except Exception:
        logger.exception("Failed to query cameras from DB")
        return out

    for cam in cams:
        try:
            cid = int(getattr(cam, "id"))
        except Exception:
            continue
        url = _build_rtsp_url(str(getattr(cam, "ip_address", "") or ""), cid, str(getattr(cam, "name", "") or ""))
        if url:
            out[cid] = url

    return out


def _get_streams_map(camera_ids: Optional[Any] = None) -> Dict[int, str]:
    ids = _normalize_camera_ids(camera_ids)

    if USE_DB_CAMERA_CONFIG:
        m = _camera_streams_from_db(ids)
        if m:
            return m
        if ids is not None:
            # user requested IDs but DB could not resolve them
            return {}

    return _camera_streams_map(RTSP_STREAMS)


def _validate_camera_ids_exist(camera_ids: List[int]) -> Tuple[List[int], List[int]]:
    camera_ids = [int(x) for x in camera_ids if str(x).isdigit()]
    if not camera_ids:
        return [], []
    if Camera is None:
        return camera_ids, []
    existing = set(_db_get_camera_ids(camera_ids=camera_ids, only_active=None))
    valid = [cid for cid in camera_ids if cid in existing]
    missing = [cid for cid in camera_ids if cid not in existing]
    return valid, missing


# ===================== Capture & processing loops =====================

def capture_thread_fn(camera_id: int, rtsp_url: str):
    q = cap_queues[camera_id]

    # Prefer FFMPEG, but fall back if needed
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap = cv2.VideoCapture(rtsp_url)

    if not cap.isOpened():
        logger.error("[cap cam=%d] cannot open RTSP: %s", camera_id, rtsp_url)
        return

    _configure_rtsp_capture(cap)
    logger.info("[cap cam=%d] started on %s", camera_id, rtsp_url)

    try:
        while not stop_event.is_set():
            ok, frame = _read_latest_frame(cap)
            if not ok or frame is None:
                time.sleep(0.01)
                continue

            while not q.empty():
                try:
                    q.get_nowait()
                except Exception:
                    break
            try:
                q.put_nowait(frame)
            except Exception:
                pass
    finally:
        cap.release()
        logger.info("[cap cam=%d] stopped", camera_id)


def embedding_loop_for_cam(camera_id: int, rtsp_url: str):
    global current_member_id, current_member_name

    ensure_csv_header()
    start_workers_if_needed()

    member_id = int(current_member_id or 0)
    member_name = str(current_member_name or "unknown")

    body_dir = _gallery_body_dir(member_id, camera_id, member_name)
    face_dir = _gallery_face_dir(member_id, camera_id, member_name)
    back_dir = _gallery_back_body_dir(member_id, camera_id, member_name)

    logger.info("[Loop cam=%d] dirs: body=%s face=%s back=%s", camera_id, body_dir, face_dir, back_dir)

    frame_idx = 0
    last_save_ms = 0
    last_debug_save = 0.0
    no_det_counter = 0

    if yolo_model is None:
        logger.warning("[Loop cam=%d] YOLO model is None -> no detections, no crops", camera_id)

    logger.info("[Loop cam=%d] started on %s", camera_id, rtsp_url)
    q = cap_queues[camera_id]

    try:
        while not stop_event.is_set():
            try:
                frame = q.get(timeout=0.2)
            except Empty:
                continue
            except Exception:
                break
            if frame is None:
                continue

            frame_idx += 1
            H, W = frame.shape[:2]

            now_ms = int(time.time() * 1000)
            allow_save = (now_ms - last_save_ms) >= int(SAVE_MIN_INTERVAL_MS)
            if allow_save:
                last_save_ms = now_ms

            # submit YOLO request
            respq = Queue(maxsize=1)
            yolo_ok = (yolo_model is not None and yolo_req_queue is not None)
            if yolo_ok:
                try:
                    yolo_req_queue.put_nowait({"frame": frame, "H": H, "W": W, "resp": respq})
                except Exception:
                    yolo_ok = False

            faces_checked = bool(USE_FACE and face_app is not None and (frame_idx % max(1, int(FACE_EVERY)) == 0))
            faces = detect_faces_raw(frame) if faces_checked else []

            detections: List[tuple[float, float, float, float, float]] = []
            if yolo_ok:
                try:
                    detections = respq.get(timeout=float(YOLO_RESP_TIMEOUT_S))
                except Exception:
                    detections = []
            else:
                detections = yolo_infer(frame, H, W, timeout_s=float(YOLO_RESP_TIMEOUT_S))

            det_boxes_i: List[tuple[int, int, int, int, float]] = []
            for (x1, y1, x2, y2, conf) in detections:
                x1i, y1i, x2i, y2i = map(int, [x1, y1, x2, y2])
                x1i, y1i = max(0, x1i), max(0, y1i)
                x2i, y2i = min(W, x2i), min(H, y2i)
                if x2i <= x1i or y2i <= y1i:
                    continue
                det_boxes_i.append((x1i, y1i, x2i, y2i, conf))

            if not det_boxes_i:
                no_det_counter += 1
                if DEBUG_LOG_NO_DET_EVERY > 0 and (no_det_counter % DEBUG_LOG_NO_DET_EVERY) == 0:
                    logger.info("[Loop cam=%d] no detections for %d frames", camera_id, no_det_counter)

                # Optional debug: save a raw frame periodically if no detections
                if DEBUG_SAVE_NO_DETS_EVERY_SEC > 0:
                    now = time.time()
                    if (now - last_debug_save) >= float(DEBUG_SAVE_NO_DETS_EVERY_SEC):
                        last_debug_save = now
                        dbg_dir = Path(CROPS_ROOT) / "debug_frames" / _member_folder(member_id, member_name) / f"cam_{camera_id}"
                        _ensure_dir(dbg_dir)
                        dbg_path = dbg_dir / f"frame_{int(now)}.jpg"
                        try:
                            cv2.imwrite(str(dbg_path), frame)
                            logger.info("[Loop cam=%d] DEBUG saved frame: %s", camera_id, dbg_path)
                        except Exception:
                            pass

            # Save crops (rate-limited)
            if allow_save and det_boxes_i:
                ts = time.time()
                for det_idx, (x1i, y1i, x2i, y2i, conf) in enumerate(det_boxes_i[: max(1, int(SAVE_MAX_DETS_PER_FRAME))]):

                    # (1) regular body gallery
                    crop_bgr = crop_with_padding(frame, (x1i, y1i, x2i, y2i))
                    if crop_bgr.size > 0:
                        fname = f"person_{_epoch_ms()}.jpg"
                        path = body_dir / fname
                        row = [
                            member_id, member_name, ts,
                            int(camera_id), frame_idx, det_idx,
                            x1i, y1i, x2i, y2i,
                            conf,
                            "", "", str(path), "body",
                        ]
                        try:
                            if io_queue is not None:
                                io_queue.put_nowait({"crop_path": path, "image": crop_bgr, "csv_row": row})
                        except Exception:
                            pass

                    # (3) back-body / no-face gallery
                    if faces_checked:
                        has_linked_face = False
                        if faces:
                            t_xyxy = (x1i, y1i, x2i, y2i)
                            for fm in faces:
                                fb = fm.get("bbox")
                                if not fb:
                                    continue
                                score = float(fm.get("score", 0.0))
                                if score < float(FACE_MIN_SCORE):
                                    continue
                                if not _face_big_enough(tuple(map(float, fb))):
                                    continue
                                if link_face_to_person(tuple(map(float, fb)), t_xyxy):
                                    has_linked_face = True
                                    break

                        if not has_linked_face:
                            back_crop = crop_back_body_no_face(frame, (x1i, y1i, x2i, y2i))
                            if back_crop.size > 0:
                                fname = f"back_{_epoch_ms()}.jpg"
                                path = back_dir / fname
                                row = [
                                    member_id, member_name, ts,
                                    int(camera_id), frame_idx, det_idx,
                                    x1i, y1i, x2i, y2i,
                                    conf,
                                    "", "", str(path), "back_body",
                                ]
                                try:
                                    if io_queue is not None:
                                        io_queue.put_nowait({"crop_path": path, "image": back_crop, "csv_row": row})
                                except Exception:
                                    pass

                # (2) gallery_face
                if faces and faces_checked:
                    for det_idx, (x1i, y1i, x2i, y2i, _conf) in enumerate(det_boxes_i[: max(1, int(SAVE_MAX_DETS_PER_FRAME))]):
                        t_xyxy = (x1i, y1i, x2i, y2i)
                        best_score = -1.0
                        best_box = None
                        for fm in faces:
                            fb = fm.get("bbox")
                            if not fb:
                                continue
                            if link_face_to_person(tuple(map(float, fb)), t_xyxy):
                                s = float(fm.get("score", 0.0))
                                if s > best_score:
                                    best_score = s
                                    best_box = fb

                        if best_box is None:
                            continue

                        fx1, fy1, fx2, fy2 = map(int, best_box)
                        if best_score < FACE_MIN_SCORE:
                            continue
                        if min(fx2 - fx1, fy2 - fy1) < FACE_MIN_SIZE:
                            continue

                        body_crop = crop_with_padding(frame, (x1i, y1i, x2i, y2i))
                        if body_crop.size <= 0:
                            continue

                        fname = f"person_{_epoch_ms()}.jpg"
                        path = face_dir / fname
                        row = [
                            member_id, member_name, ts,
                            int(camera_id), frame_idx, det_idx,
                            x1i, y1i, x2i, y2i,
                            best_score,
                            "", "", str(path), "face",
                        ]
                        try:
                            if io_queue is not None:
                                io_queue.put_nowait({"crop_path": path, "image": body_crop, "csv_row": row})
                        except Exception:
                            pass

            with frames_lock:
                latest_frames[camera_id] = frame

    finally:
        logger.info("[Loop cam=%d] stopped", camera_id)


# ===================== Extract from folders helpers =====================

def _list_images(root: Path) -> List[Path]:
    if not root.exists():
        return []
    pats = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    files: List[Path] = []
    for p in pats:
        files.extend(root.glob(p))
    return sorted(files)


def _stable_seed(*parts: str) -> int:
    s = "|".join(parts).encode("utf-8", errors="ignore")
    return int(zlib.crc32(s) & 0xFFFFFFFF)


def _sample_paths(paths: List[Path], max_n: int, seed: int) -> List[Path]:
    if max_n <= 0 or len(paths) <= max_n:
        return paths
    rng = random.Random(int(seed))
    idxs = sorted(rng.sample(range(len(paths)), int(max_n)))
    return [paths[i] for i in idxs]


def _list_images_sampled(root: Path, max_n: int, member_id: int, camera_id: int, kind: str) -> List[Path]:
    paths = _list_images(root)
    seed = _stable_seed(str(member_id), str(camera_id), kind, str(GALLERY_SAMPLE_SEED))
    return _sample_paths(paths, int(max_n), seed)


def _load_bgr(path: Path) -> Optional[np.ndarray]:
    try:
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        return img if img is not None else None
    except Exception:
        return None


def _mean_embed(vectors: List[np.ndarray]) -> Optional[np.ndarray]:
    if not vectors:
        return None
    arr = np.stack(vectors, axis=0).astype(np.float32)
    m = np.mean(arr, axis=0)
    return l2_normalize(m)


def _count_images(member_id: int, camera_id: int, member_name: str) -> Tuple[int, int, int]:
    body = len(_list_images_sampled(_gallery_body_dir(member_id, camera_id, member_name), MAX_BODY_IMAGES, member_id, camera_id, "body"))
    face = len(_list_images_sampled(_gallery_face_dir(member_id, camera_id, member_name), MAX_FACE_IMAGES, member_id, camera_id, "face"))
    back_body = len(_list_images_sampled(_gallery_back_body_dir(member_id, camera_id, member_name), MAX_BACK_BODY_IMAGES, member_id, camera_id, "back_body"))
    return body, face, back_body


def _ahash8(img_bgr: np.ndarray) -> int:
    try:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (8, 8), interpolation=cv2.INTER_AREA)
        avg = float(g.mean())
        bits = (g > avg).astype(np.uint8)
        out = 0
        for i, b in enumerate(bits.flatten()):
            out |= (int(b) & 1) << i
        return out
    except Exception:
        return random.getrandbits(64)


def _is_blurry(img_bgr: np.ndarray, var_thr: float = 30.0) -> bool:
    try:
        g = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(g, cv2.CV_64F).var()) < var_thr
    except Exception:
        return False


def _prep_reid(img_bgr: np.ndarray) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    if h < 32 or w < 16:
        raise ValueError("Crop too small for ReID")
    img = cv2.resize(img_bgr, (128, 256), interpolation=cv2.INTER_LINEAR)
    return _to_rgb(img)


def _compute_body_bank_and_centroid_from_gallery(
    member_id: int,
    camera_id: int,
    member_name: str,
    on_file_done: Optional[Callable[[int], None]] = None,
    on_work_done: Optional[Callable[[int], None]] = None,
) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
    if not reid_extractors:
        logger.warning("TorchReID not available; skipping body embedding.")
        return None, None

    root = _gallery_body_dir(member_id, camera_id, member_name)
    imgs = _list_images_sampled(root, MAX_BODY_IMAGES, member_id, camera_id, "body")
    if not imgs:
        logger.warning("No BODY crops found for member=%s cam=%s", member_id, camera_id)
        return None, None

    gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
    B = 64 if gpu else 32

    prepared: List[Tuple[np.ndarray, Optional[np.ndarray]]] = []
    seen_hashes: set[int] = set()
    per_image_units = len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)

    for p in imgs:
        img = _load_bgr(p)
        if img is None or img.size == 0:
            if on_file_done:
                on_file_done(1)
            if on_work_done:
                on_work_done(per_image_units)
            continue
        h, w = img.shape[:2]
        if min(h, w) < 32 or _is_blurry(img):
            if on_file_done:
                on_file_done(1)
            if on_work_done:
                on_work_done(per_image_units)
            continue
        hval = _ahash8(img)
        if hval in seen_hashes:
            if on_file_done:
                on_file_done(1)
            if on_work_done:
                on_work_done(per_image_units)
            continue
        seen_hashes.add(hval)

        try:
            rgb = _prep_reid(img)
        except Exception:
            if on_file_done:
                on_file_done(1)
            if on_work_done:
                on_work_done(per_image_units)
            continue

        rgb_flip = cv2.flip(rgb, 1) if BODY_TTA_FLIP else None
        prepared.append((rgb, rgb_flip))

    if not prepared:
        logger.warning("No valid BODY crops for member=%s cam=%s", member_id, camera_id)
        return None, None

    per_model_embs: List[List[np.ndarray]] = [[] for _ in reid_extractors]

    for midx, ext in enumerate(reid_extractors):
        batch_imgs: List[np.ndarray] = []
        idx_map: List[int] = []
        for i, (orig, flip) in enumerate(prepared):
            batch_imgs.append(orig)
            idx_map.append(i)
            if flip is not None:
                batch_imgs.append(flip)
                idx_map.append(i)

        feats_norm: List[np.ndarray] = []
        for start in range(0, len(batch_imgs), B):
            chunk = batch_imgs[start:start + B]
            try:
                with reid_lock, torch.inference_mode():
                    feats = ext(chunk)
                for f in feats:
                    f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
                    f512 = _as_512f(f)
                    feats_norm.append(f512 if f512 is not None else np.zeros((EXPECTED_EMBED_DIM,), np.float32))
            except Exception:
                logger.exception("TorchReID batch failed (model=%s)", REID_MODELS[midx])
                feats_norm.extend([np.zeros((EXPECTED_EMBED_DIM,), np.float32) for _ in chunk])

            if on_work_done:
                on_work_done(len(chunk))

        img_accum = [[] for _ in prepared]
        for img_i, feat in zip(idx_map, feats_norm):
            img_accum[img_i].append(feat)

        fused = [
            l2_normalize(np.mean(np.stack(v, axis=0), axis=0)) if v else np.zeros((EXPECTED_EMBED_DIM,), np.float32)
            for v in img_accum
        ]
        per_model_embs[midx] = fused

    vectors_per_image: List[np.ndarray] = []
    for i in range(len(prepared)):
        parts = [per_model_embs[m][i] for m in range(len(reid_extractors))]
        fused_img = l2_normalize(np.mean(np.stack(parts, axis=0), axis=0))
        vectors_per_image.append(fused_img)
        if on_file_done:
            on_file_done(1)

    centroid = _mean_embed(vectors_per_image)
    return (vectors_per_image if vectors_per_image else None), centroid


def _compute_face_bank_and_centroid_from_gallery(
    member_id: int,
    camera_id: int,
    member_name: str,
    on_file_done: Optional[Callable[[int], None]] = None,
    on_work_done: Optional[Callable[[int], None]] = None,
) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
    if face_app is None:
        logger.warning("InsightFace not available; skipping face embedding.")
        return None, None

    root = _gallery_face_dir(member_id, camera_id, member_name)
    imgs = _list_images_sampled(root, MAX_FACE_IMAGES, member_id, camera_id, "face")
    if not imgs:
        logger.warning("No FACE crops found for member=%s cam=%s", member_id, camera_id)
        return None, None

    vectors: List[np.ndarray] = []
    seen_hashes: set[int] = set()
    MAX_SIDE = 640
    MIN_BODY_SIDE = 40
    per_image_units = (2 if FACE_TTA_FLIP else 1)

    for p in imgs:
        img = _load_bgr(p)
        if img is None or img.size == 0:
            if on_file_done:
                on_file_done(1)
            if on_work_done:
                on_work_done(per_image_units)
            continue

        h, w = img.shape[:2]
        if min(h, w) < MIN_BODY_SIDE or _is_blurry(img):
            if on_file_done:
                on_file_done(1)
            if on_work_done:
                on_work_done(per_image_units)
            continue

        hval = _ahash8(img)
        if hval in seen_hashes:
            if on_file_done:
                on_file_done(1)
            if on_work_done:
                on_work_done(per_image_units)
            continue
        seen_hashes.add(hval)

        if max(h, w) > MAX_SIDE:
            scale = MAX_SIDE / float(max(h, w))
            img_proc = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_LINEAR)
        else:
            img_proc = img

        def _best_face(bgr_img) -> Optional[np.ndarray]:
            try:
                faces = face_app.get(np.ascontiguousarray(bgr_img))
            except Exception:
                faces = []
            best, best_score = None, -1.0
            for f in safe_iter_faces(faces):
                bbox = getattr(f, "bbox", None)
                if bbox is None:
                    continue
                b = np.asarray(bbox).reshape(-1)
                if b.size < 4:
                    continue
                x1, y1, x2, y2 = b[:4].astype(float)
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                score = float(getattr(f, "det_score", 0.0))
                ranking = score * (area ** 0.5)
                if ranking > best_score:
                    best, best_score = f, ranking
            if best is None:
                return None
            det_score = float(getattr(best, "det_score", 0.0))
            if det_score < FACE_MIN_SCORE:
                return None
            emb = extract_face_embedding(best)
            return _as_512f(emb)

        e0 = _best_face(img_proc)
        if on_work_done:
            on_work_done(1)

        e1 = None
        if FACE_TTA_FLIP:
            try:
                e1 = _best_face(cv2.flip(img_proc, 1))
            except Exception:
                e1 = None
            if on_work_done:
                on_work_done(1)

        feats = [e for e in (e0, e1) if e is not None]
        if feats:
            fused = l2_normalize(np.mean(np.stack(feats, axis=0), axis=0))
            vectors.append(fused)

        if on_file_done:
            on_file_done(1)

    if not vectors:
        logger.warning("No valid FACE embeddings for member=%s cam=%s", member_id, camera_id)
        return None, None

    centroid = _mean_embed(vectors)
    return (vectors if vectors else None), centroid


def _compute_back_body_bank_and_centroid_from_gallery(
    member_id: int,
    camera_id: int,
    member_name: str,
    on_file_done: Optional[Callable[[int], None]] = None,
    on_work_done: Optional[Callable[[int], None]] = None,
) -> Tuple[Optional[List[np.ndarray]], Optional[np.ndarray]]:
    root = _gallery_back_body_dir(member_id, camera_id, member_name)
    imgs = _list_images_sampled(root, MAX_BACK_BODY_IMAGES, member_id, camera_id, "back_body")
    if not imgs:
        logger.warning("No BACK-BODY crops found for member=%s cam=%s", member_id, camera_id)
        return None, None

    gpu = torch.cuda.is_available() and ("cuda" in str(DEVICE).lower())
    B = 64 if gpu else 32

    seen_hashes: set[int] = set()

    prepared_reid: List[Tuple[Optional[np.ndarray], Optional[np.ndarray]]] = []
    shape_vecs: List[Optional[np.ndarray]] = []
    keep_mask: List[bool] = []
    reid_ready: List[bool] = []

    reid_units = (len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)) if reid_extractors else 0
    per_image_units = 1 + int(reid_units)

    for p in imgs:
        img = _load_bgr(p)
        if img is None or img.size == 0:
            shape_vecs.append(None)
            prepared_reid.append((None, None))
            keep_mask.append(False)
            reid_ready.append(False)
            if on_work_done:
                on_work_done(per_image_units)
            if on_file_done:
                on_file_done(1)
            continue

        h, w = img.shape[:2]
        if min(h, w) < 32 or _is_blurry(img):
            shape_vecs.append(None)
            prepared_reid.append((None, None))
            keep_mask.append(False)
            reid_ready.append(False)
            if on_work_done:
                on_work_done(per_image_units)
            if on_file_done:
                on_file_done(1)
            continue

        hval = _ahash8(img)
        if hval in seen_hashes:
            shape_vecs.append(None)
            prepared_reid.append((None, None))
            keep_mask.append(False)
            reid_ready.append(False)
            if on_work_done:
                on_work_done(per_image_units)
            if on_file_done:
                on_file_done(1)
            continue
        seen_hashes.add(hval)

        shp = _back_shape_embed(img)
        shape_vecs.append(shp)
        if on_work_done:
            on_work_done(1)

        rgb = None
        rgb_flip = None
        can_reid = False
        if reid_extractors:
            try:
                rgb = _prep_reid(img)
                rgb_flip = cv2.flip(rgb, 1) if BODY_TTA_FLIP else None
                can_reid = True
            except Exception:
                can_reid = False

        prepared_reid.append((rgb, rgb_flip))
        reid_ready.append(bool(can_reid))

        keep = (shp is not None) or bool(can_reid)
        keep_mask.append(bool(keep))

        if not keep:
            if on_work_done and reid_units > 0:
                on_work_done(int(reid_units))
            if on_file_done:
                on_file_done(1)
            continue

        if not can_reid and reid_units > 0:
            if on_work_done:
                on_work_done(int(reid_units))

    reid_vectors_per_image: List[Optional[np.ndarray]] = [None for _ in prepared_reid]

    if reid_extractors:
        per_model_embs: List[List[np.ndarray]] = [[] for _ in reid_extractors]

        for midx, ext in enumerate(reid_extractors):
            batch_imgs: List[np.ndarray] = []
            idx_map: List[int] = []
            for i, (orig, flip) in enumerate(prepared_reid):
                if not reid_ready[i] or orig is None:
                    continue
                batch_imgs.append(orig)
                idx_map.append(i)
                if flip is not None:
                    batch_imgs.append(flip)
                    idx_map.append(i)

            feats_norm: List[np.ndarray] = []
            for start in range(0, len(batch_imgs), B):
                chunk = batch_imgs[start:start + B]
                try:
                    with reid_lock, torch.inference_mode():
                        feats = ext(chunk)
                    for f in feats:
                        f = f.detach().cpu().numpy() if hasattr(f, "detach") else np.asarray(f)
                        f512 = _as_512f(f)
                        feats_norm.append(f512 if f512 is not None else np.zeros((EXPECTED_EMBED_DIM,), np.float32))
                except Exception:
                    logger.exception("TorchReID batch failed (model=%s) [back_body]", REID_MODELS[midx])
                    feats_norm.extend([np.zeros((EXPECTED_EMBED_DIM,), np.float32) for _ in chunk])

                if on_work_done:
                    on_work_done(len(chunk))

            img_accum: Dict[int, List[np.ndarray]] = {}
            for img_i, feat in zip(idx_map, feats_norm):
                img_accum.setdefault(img_i, []).append(feat)

            fused_list: List[np.ndarray] = []
            for i in range(len(prepared_reid)):
                vv = img_accum.get(i, [])
                fused_list.append(
                    l2_normalize(np.mean(np.stack(vv, axis=0), axis=0)) if vv else np.zeros((EXPECTED_EMBED_DIM,), np.float32)
                )
            per_model_embs[midx] = fused_list

        for i in range(len(prepared_reid)):
            if not reid_ready[i]:
                continue
            parts = [per_model_embs[m][i] for m in range(len(reid_extractors))]
            if parts:
                reid_vectors_per_image[i] = l2_normalize(np.mean(np.stack(parts, axis=0), axis=0))

    fused_vectors: List[np.ndarray] = []
    for i in range(len(prepared_reid)):
        if not keep_mask[i]:
            continue

        v_reid = reid_vectors_per_image[i]
        v_shape = shape_vecs[i] if i < len(shape_vecs) else None

        parts2: List[np.ndarray] = []
        if v_reid is not None:
            parts2.append(v_reid * float(BACK_REID_WEIGHT))
        if v_shape is not None:
            parts2.append(v_shape * float(BACK_SHAPE_WEIGHT))

        if parts2:
            fused = l2_normalize(np.sum(np.stack(parts2, axis=0), axis=0))
            fused_vectors.append(fused)

        if on_file_done:
            on_file_done(1)

    if not fused_vectors:
        logger.warning("No valid BACK-BODY embeddings for member=%s cam=%s", member_id, camera_id)
        return None, None

    centroid = _mean_embed(fused_vectors)
    return (fused_vectors if fused_vectors else None), centroid


# ===================== DB update (FIX FK camera_id) =====================

def _update_db_embeddings(
    member_id: int,
    camera_id: int,
    body_emb: Optional[np.ndarray],
    face_emb: Optional[np.ndarray],
    back_body_emb: Optional[np.ndarray],
    body_bank: Optional[List[np.ndarray]] = None,
    face_bank: Optional[List[np.ndarray]] = None,
    back_body_bank: Optional[List[np.ndarray]] = None,
) -> Dict[str, object]:
    with db_session() as db:
        try:
            m = db.query(Member).filter(Member.id == int(member_id)).first()
            if not m:
                logger.warning("Member ID %s not found. Skipping DB update.", member_id)
                return {"status": "error", "message": "member not found"}

            # IMPORTANT: camera FK validation
            if Camera is not None:
                cam = db.query(Camera).filter(Camera.id == int(camera_id)).first()
                if not cam:
                    logger.error("Camera ID %s not found in cameras table. FK would fail.", camera_id)
                    db.rollback()
                    return {"status": "error", "message": "camera not found", "member_id": int(member_id), "camera_id": int(camera_id)}

            rec = (
                db.query(MemberEmbedding)
                .filter(MemberEmbedding.member_id == int(member_id), MemberEmbedding.camera_id == int(camera_id))
                .first()
            )
            if rec is None:
                rec = MemberEmbedding(member_id=int(member_id), camera_id=int(camera_id))
                db.add(rec)

            changed = False

            if body_emb is not None:
                rec.body_embedding = body_emb.tolist()
                changed = True
            if face_emb is not None:
                rec.face_embedding = face_emb.tolist()
                changed = True
            if back_body_emb is not None:
                rec.back_body_embedding = back_body_emb.tolist()
                changed = True

            if body_bank is not None:
                rec.body_embeddings_raw = _pack_raw_bank(body_bank)
                changed = True
            if face_bank is not None:
                rec.face_embeddings_raw = _pack_raw_bank(face_bank)
                changed = True
            if back_body_bank is not None:
                rec.back_body_embeddings_raw = _pack_raw_bank(back_body_bank)
                changed = True

            if changed:
                rec.last_embedding_update_ts = datetime.now(timezone.utc)
                db.commit()
                logger.info(
                    "DB updated: member=%s cam=%s body=%s face=%s back=%s body_raw=%s face_raw=%s back_raw=%s",
                    member_id,
                    camera_id,
                    body_emb is not None,
                    face_emb is not None,
                    back_body_emb is not None,
                    body_bank is not None,
                    face_bank is not None,
                    back_body_bank is not None,
                )
                return {
                    "status": "ok",
                    "member_id": int(member_id),
                    "camera_id": int(camera_id),
                    "body": body_emb is not None,
                    "face": face_emb is not None,
                    "back_body": back_body_emb is not None,
                    "body_raw": body_bank is not None,
                    "face_raw": face_bank is not None,
                    "back_body_raw": back_body_bank is not None,
                }

            db.rollback()
            logger.info("No embeddings to update for member=%s cam=%s", member_id, camera_id)
            return {"status": "no_embeddings", "member_id": int(member_id), "camera_id": int(camera_id)}

        except Exception:
            db.rollback()
            logger.exception("DB error on update")
            return {"status": "error", "message": "db error", "member_id": int(member_id), "camera_id": int(camera_id)}


# ===================== Background extraction workers =====================

def _extract_worker(member_id: int, member_name: str, camera_id: int):
    try:
        _progress_reset(member_id, member_name, camera_id)
        _progress_set(member_id, camera_id, stage="Scanning", message="Counting images...")
        total_body, total_face, total_back = _count_images(member_id, camera_id, member_name)

        if total_body + total_face + total_back == 0:
            _progress_set(
                member_id,
                camera_id,
                total_body=0,
                total_face=0,
                total_back_body=0,
                stage="error",
                message="No images to process",
            )
            return

        _progress_set(
            member_id,
            camera_id,
            total_body=total_body,
            total_face=total_face,
            total_back_body=total_back,
            done_body=0,
            done_face=0,
            done_back_body=0,
            percent=0,
        )

        body_bank = None
        body_cent = None
        if total_body > 0 and reid_extractors:
            units_per = len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)
            _progress_set(
                member_id,
                camera_id,
                stage="Body embeddings",
                message=f"Processing body ({total_body})",
                work_total=int(total_body * units_per),
                work_done=0,
            )

            def on_body_files(n: int):
                st = get_progress_for_member(member_id, camera_id)
                _progress_set(member_id, camera_id, done_body=int(st.get("done_body", 0)) + int(n))

            def on_body_work(n: int):
                st = get_progress_for_member(member_id, camera_id)
                _progress_set(member_id, camera_id, work_done=int(st.get("work_done", 0)) + int(n))

            body_bank, body_cent = _compute_body_bank_and_centroid_from_gallery(
                member_id, camera_id, member_name, on_file_done=on_body_files, on_work_done=on_body_work
            )
        else:
            _progress_set(member_id, camera_id, message="Skipping body (no images or model unavailable)")

        face_bank = None
        face_cent = None
        if total_face > 0 and face_app is not None:
            units_per = (2 if FACE_TTA_FLIP else 1)
            _progress_set(
                member_id,
                camera_id,
                stage="Face embeddings",
                message=f"Processing face ({total_face})",
                work_total=int(total_face * units_per),
                work_done=0,
            )

            def on_face_files(n: int):
                st = get_progress_for_member(member_id, camera_id)
                _progress_set(member_id, camera_id, done_face=int(st.get("done_face", 0)) + int(n))

            def on_face_work(n: int):
                st = get_progress_for_member(member_id, camera_id)
                _progress_set(member_id, camera_id, work_done=int(st.get("work_done", 0)) + int(n))

            face_bank, face_cent = _compute_face_bank_and_centroid_from_gallery(
                member_id, camera_id, member_name, on_file_done=on_face_files, on_work_done=on_face_work
            )
        else:
            _progress_set(member_id, camera_id, message="Skipping face (no images or model unavailable)")

        back_bank = None
        back_cent = None
        if total_back > 0:
            reid_units = (len(reid_extractors) * (2 if BODY_TTA_FLIP else 1)) if reid_extractors else 0
            units_per = 1 + reid_units
            _progress_set(
                member_id,
                camera_id,
                stage="Back body embeddings",
                message=f"Processing back-body ({total_back})",
                work_total=int(total_back * units_per),
                work_done=0,
            )

            def on_back_files(n: int):
                st = get_progress_for_member(member_id, camera_id)
                _progress_set(member_id, camera_id, done_back_body=int(st.get("done_back_body", 0)) + int(n))

            def on_back_work(n: int):
                st = get_progress_for_member(member_id, camera_id)
                _progress_set(member_id, camera_id, work_done=int(st.get("work_done", 0)) + int(n))

            back_bank, back_cent = _compute_back_body_bank_and_centroid_from_gallery(
                member_id, camera_id, member_name, on_file_done=on_back_files, on_work_done=on_back_work
            )
        else:
            _progress_set(member_id, camera_id, message="Skipping back-body (no images)")

        # Validation: ensure all six embedding inputs are present before DB update.
        required_pairs = [
            ("Body embeddings", body_cent),
            ("Face embeddings", face_cent),
            ("Back body embeddings", back_cent),
            ("Body bank", body_bank),
            ("Face bank", face_bank),
            ("Back body bank", back_bank),
        ]
        missing = [name for name, val in required_pairs if val is None]
        if missing:
            missing_list = ", ".join(missing)
            friendly = (
                f"Missing required embedding data: {missing_list}. "
                "Please provide the missing embedding to proceed."
            )
            logger.error("Extract worker validation failed: %s (member=%s cam=%s)", missing_list, member_id, camera_id)
            _progress_set(
                member_id,
                camera_id,
                stage="error",
                message=friendly,
                percent=100,
            )
            return

        res = _update_db_embeddings(
            member_id,
            camera_id,
            body_cent,
            face_cent,
            back_cent,
            body_bank=body_bank,
            face_bank=face_bank,
            back_body_bank=back_bank,
        )
        status = str(res.get("status", "unknown"))
        _progress_set(member_id, camera_id, stage="Done", message=("Embeddings saved" if status == "ok" else status), percent=100)

    except Exception as e:
        logger.exception("extract worker failed")
        _progress_set(member_id, camera_id, stage="error", message=f"error: {e}")


def _extract_worker_multi(member_id: int, member_name: str, camera_ids: List[int]):
    camera_ids = [int(c) for c in camera_ids if isinstance(c, int) or str(c).isdigit()]
    if not camera_ids:
        return

    _progress_reset(member_id, member_name, OVERALL_CAMERA_ID)
    _progress_set(member_id, OVERALL_CAMERA_ID, stage="Starting", message=f"Processing {len(camera_ids)} camera(s)", percent=0)

    done = 0
    total = max(1, len(camera_ids))

    for cid in camera_ids:
        _progress_set(member_id, OVERALL_CAMERA_ID, stage="running", message=f"Extracting camera {cid} ({done + 1}/{total})")
        _extract_worker(member_id, member_name, cid)
        done += 1
        _progress_set(member_id, OVERALL_CAMERA_ID, percent=int(round((done / total) * 100)))

    _progress_set(member_id, OVERALL_CAMERA_ID, stage="Done", message="All cameras processed", percent=100)


def extract_embeddings_async(member_id: int, camera_ids: Optional[Any] = None) -> Dict[str, object]:
    if not reid_extractors or (USE_FACE and face_app is None):
        init_models()

    cam_ids = _normalize_camera_ids(camera_ids)
    if cam_ids is None:
        cam_ids = _db_get_camera_ids(camera_ids=None, only_active=True)
        if not cam_ids:
            cam_ids = sorted(_camera_streams_map(RTSP_STREAMS).keys())
    else:
        cam_ids = [int(x) for x in cam_ids]

    valid_ids, missing = _validate_camera_ids_exist(cam_ids)
    if missing:
        return {
            "status": "error",
            "message": "Some camera_ids do not exist in cameras table",
            "missing_camera_ids": missing,
            "valid_camera_ids": valid_ids,
        }
    if not valid_ids:
        return {"status": "error", "message": "No valid camera_ids to extract", "requested_camera_ids": cam_ids}

    with db_session() as db:
        try:
            member = db.query(Member).filter(Member.id == int(member_id)).first()
            if not member:
                return {"status": "error", "message": f"member id {member_id} not found"}
            member_name = _get_member_display_name(member, fallback=f"member_{member_id}")
        except Exception:
            db.rollback()
            logger.exception("DB error during member lookup.")
            return {"status": "error", "message": "DB error"}

    t = threading.Thread(
        target=_extract_worker_multi,
        args=(int(member_id), member_name, valid_ids),
        daemon=True,
        name=f"extract-{member_id}",
    )
    t.start()
    return {"status": "started", "member_id": int(member_id), "member_name": member_name, "camera_ids": valid_ids}


def extract_embeddings_sync(member_id: int, camera_ids: Optional[Any] = None) -> Dict[str, object]:
    if not reid_extractors or (USE_FACE and face_app is None):
        init_models()

    cam_ids = _normalize_camera_ids(camera_ids)
    if cam_ids is None:
        cam_ids = _db_get_camera_ids(camera_ids=None, only_active=True)
        if not cam_ids:
            cam_ids = sorted(_camera_streams_map(RTSP_STREAMS).keys())
    else:
        cam_ids = [int(x) for x in cam_ids]

    valid_ids, missing = _validate_camera_ids_exist(cam_ids)
    if missing:
        return {
            "status": "error",
            "message": "Some camera_ids do not exist in cameras table",
            "missing_camera_ids": missing,
            "valid_camera_ids": valid_ids,
        }
    if not valid_ids:
        return {"status": "error", "message": "No valid camera_ids to extract", "requested_camera_ids": cam_ids}

    with db_session() as db:
        try:
            member = db.query(Member).filter(Member.id == int(member_id)).first()
            if not member:
                return {"status": "error", "message": f"member id {member_id} not found"}
            member_name = _get_member_display_name(member, fallback=f"member_{member_id}")
        except Exception:
            db.rollback()
            logger.exception("DB error during member lookup.")
            return {"status": "error", "message": "DB error"}

    results: Dict[int, Dict[str, object]] = {}
    for cid in valid_ids:
        _extract_worker(int(member_id), member_name, int(cid))
        results[int(cid)] = get_progress_for_member(int(member_id), int(cid))

    return {"status": "Done", "member_id": int(member_id), "member_name": member_name, "cameras": results}


# ===================== Control API =====================

def start_extraction(member_id: int, camera_ids: Optional[Any] = None, show_viewer: bool = True) -> dict:
    """
    camera_ids refer to cameras.id (DB FK).
    RTSP URL is built from Camera.ip_address using RTSP_URL_TEMPLATE.
    """
    global is_running, current_member_id, current_member_name, current_camera_ids, current_camera_streams, viewer_thread

    if is_running:
        return {
            "status": "ok",
            "message": "Already running",
            "num_cams": len(current_camera_ids),
            "member_id": current_member_id,
            "member_name": current_member_name,
            "camera_ids": list(current_camera_ids),
        }

    selected = _normalize_camera_ids(camera_ids)
    streams_map = _get_streams_map(selected)

    if not streams_map:
        avail = sorted(_get_streams_map(None).keys())
        return {
            "status": "error",
            "message": "No valid cameras/RTSP streams configured",
            "requested_camera_ids": selected,
            "available_camera_ids": avail,
            "hint": "Populate cameras table. Store only the camera IP in cameras.ip_address and set RTSP_URL_TEMPLATE.",
        }

    if selected is None:
        selected_ids = sorted(streams_map.keys())
    else:
        missing = [int(cid) for cid in selected if int(cid) not in streams_map]
        if missing:
            avail = sorted(_get_streams_map(None).keys())
            return {
                "status": "error",
                "message": "Some requested camera_ids were not found (or are inactive).",
                "missing_camera_ids": missing,
                "available_camera_ids": avail,
                "hint": "If you need inactive cameras too, set ONLY_ACTIVE_CAMERAS=0",
            }
        selected_ids = [int(cid) for cid in selected]

    with db_session() as db:
        try:
            member = db.query(Member).filter(Member.id == int(member_id)).first()
            if not member:
                return {"status": "error", "message": f"member id {member_id} not found"}
            member_name = _get_member_display_name(member, fallback=f"member_{member_id}")
        except Exception:
            db.rollback()
            logger.exception("DB error during member lookup (start_extraction).")
            return {"status": "error", "message": "DB error"}

    if yolo_model is None or (USE_FACE and face_app is None) or (not reid_extractors):
        init_models()

    stop_event.clear()
    with frames_lock:
        latest_frames.clear()
    for d in (capture_threads, extract_threads, cap_queues):
        d.clear()

    current_member_id = int(member_id)
    current_member_name = member_name
    current_camera_ids = list(selected_ids)
    current_camera_streams = {cid: streams_map[cid] for cid in selected_ids}

    _progress_reset(int(member_id), member_name, OVERALL_CAMERA_ID)
    _progress_set(int(member_id), OVERALL_CAMERA_ID, stage="Starting", message="initializing", percent=0)

    start_workers_if_needed()

    for cid in selected_ids:
        rtsp_url = streams_map[cid]
        cap_queues[cid] = Queue(maxsize=CAP_QUEUE_MAX)

        t_cap = threading.Thread(target=capture_thread_fn, name=f"cap_cam_{cid}", args=(cid, rtsp_url), daemon=True)
        t_ext = threading.Thread(target=embedding_loop_for_cam, name=f"ext_cam_{cid}", args=(cid, rtsp_url), daemon=True)

        capture_threads[cid] = t_cap
        extract_threads[cid] = t_ext

        t_cap.start()
        t_ext.start()

    if show_viewer and (viewer_thread is None or not viewer_thread.is_alive()):
        viewer_thread = threading.Thread(target=viewer_loop, name="viewer", daemon=True)
        viewer_thread.start()

    is_running = True
    _progress_set(int(member_id), OVERALL_CAMERA_ID, stage="running", message="processing", percent=0)
    logger.info("Extraction started: member_id=%s name=%s camera_ids=%s", member_id, member_name, selected_ids)

    return {
        "status": "ok",
        "message": "started",
        "num_cams": len(selected_ids),
        "member_id": int(member_id),
        "member_name": member_name,
        "camera_ids": selected_ids,
        "rtsp_streams": {cid: streams_map[cid] for cid in selected_ids},
    }


def viewer_loop():
    logger.info("[Viewer] started")
    window_name = "Multi-RTSP Viewer"
    window_created = False
    try:
        while not stop_event.is_set():
            with frames_lock:
                frames = [latest_frames[idx] for idx in sorted(latest_frames.keys()) if latest_frames.get(idx) is not None]

            if not frames:
                time.sleep(0.01)
                continue

            if not window_created:
                try:
                    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                    window_created = True
                except Exception:
                    pass

            min_h = min(f.shape[0] for f in frames)
            resized = []
            for f in frames:
                h, w = f.shape[:2]
                if h != min_h:
                    new_w = int(w * (min_h / h))
                    f = cv2.resize(f, (new_w, min_h))
                resized.append(f)

            try:
                combined = np.concatenate(resized, axis=1)
            except Exception:
                time.sleep(0.01)
                continue

            try:
                cv2.imshow(window_name, combined)
            except Exception:
                window_created = False
                time.sleep(0.01)
                continue

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                logger.info("[Viewer] key pressed, stopping...")
                stop_event.set()
                break

            time.sleep(VIEWER_SLEEP_MS / 1000.0)
    finally:
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass
        logger.info("[Viewer] stopped")


def stop_extraction(reason: str = "user") -> dict:
    global is_running, current_member_id, current_member_name, current_camera_ids, current_camera_streams
    try:
        mid_snapshot = current_member_id
        mname_snapshot = current_member_name
        cams_snapshot = list(current_camera_ids)

        stop_event.set()

        for d in (capture_threads, extract_threads):
            for _cid, th in list(d.items()):
                try:
                    th.join(timeout=1.5)
                except Exception:
                    pass

        try:
            if io_queue is not None:
                io_queue.join()
        except Exception:
            pass

        with frames_lock:
            latest_frames.clear()
        capture_threads.clear()
        extract_threads.clear()
        cap_queues.clear()

        is_running = False
        logger.info("Extraction stopped (%s).", reason)

        if mid_snapshot is not None and cams_snapshot:
            if not reid_extractors or (USE_FACE and face_app is None):
                init_models()
            threading.Thread(
                target=_extract_worker_multi,
                args=(int(mid_snapshot), str(mname_snapshot or "unknown"), cams_snapshot),
                name=f"auto-extract-{mid_snapshot}",
                daemon=True,
            ).start()
            logger.info("Auto-extract triggered for member_id=%s cameras=%s after stop.", mid_snapshot, cams_snapshot)

        return {"status": "ok", "message": "Stopped capturing", "member_id": mid_snapshot, "camera_ids": cams_snapshot}
    finally:
        current_member_id = None
        current_member_name = None
        current_camera_ids = []
        current_camera_streams = {}


def remove_embeddings(member_id: int, camera_id: Optional[int] = None) -> Dict[str, object]:
    with db_session() as db:
        try:
            q = db.query(MemberEmbedding).filter(MemberEmbedding.member_id == int(member_id))
            if camera_id is not None:
                q = q.filter(MemberEmbedding.camera_id == int(camera_id))

            rows = q.all()
            if not rows:
                return {"status": "error", "message": "no embeddings rows found", "member_id": int(member_id), "camera_id": camera_id}

            for rec in rows:
                rec.body_embedding = None
                rec.face_embedding = None
                rec.back_body_embedding = None
                rec.body_embeddings_raw = None
                rec.face_embeddings_raw = None
                rec.back_body_embeddings_raw = None
                rec.last_embedding_update_ts = datetime.now(timezone.utc)

            db.commit()
            logger.warning("Embeddings removed for member_id=%s camera_id=%s (rows=%d)", member_id, camera_id, len(rows))
            return {"status": "ok", "member_id": int(member_id), "camera_id": camera_id, "rows": len(rows)}
        except Exception:
            db.rollback()
            logger.exception("remove embeddings: DB error")
            return {"status": "error", "message": "DB error", "member_id": int(member_id), "camera_id": camera_id}


def get_status() -> Dict[str, object]:
    streams_map = _get_streams_map(None)
    return {
        "running": bool(is_running),
        "configured_num_cams": len(streams_map),
        "configured_camera_ids": sorted(streams_map.keys()),
        "running_camera_ids": list(current_camera_ids),
        "rtsp_streams": streams_map,
        "member_id": current_member_id,
        "member_name": current_member_name,
    }
