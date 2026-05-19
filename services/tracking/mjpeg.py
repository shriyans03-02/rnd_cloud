
from __future__ import annotations

import time
from typing import List, Optional, Tuple

import cv2
import numpy as np

from app.services.tracking.pipeline import RenderedFrame


_BOUNDARY = b"--frame\r\n"
_HEADER = b"Content-Type: image/jpeg\r\n\r\n"
_TAIL = b"\r\n"


def _ensure_frame(x) -> Optional[np.ndarray]:
    """Defensive: accept only a real numpy image."""
    if x is None:
        return None
    if isinstance(x, np.ndarray):
        return x
    if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], np.ndarray):
        return x[0]
    return None


def _encode_jpeg(frame_bgr: np.ndarray, quality: int) -> Optional[bytes]:
    q = int(max(30, min(95, int(quality))))
    try:
        ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
        if not ok:
            return None
        return enc.tobytes()
    except Exception:
        return None


def _placeholder(w: int = 1280, h: int = 720, text: str = "Waiting for frames...") -> np.ndarray:
    img = np.zeros((h, w, 3), dtype=np.uint8)
    cv2.putText(img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return img


def _sleep_until(deadline: float) -> None:
    """Sleeps until monotonic deadline (best-effort), avoids negative sleep."""
    now = time.monotonic()
    dt = deadline - now
    if dt > 0:
        time.sleep(dt)


def _snapshot_from_get_result(ret) -> Tuple[Optional[np.ndarray], float, Optional[int]]:
    """Normalize `buf.get()` return values across versions.

    Supported get() shapes seen across v1/v2 buffers:
      - frame
      - (frame, ts)
      - (frame, ts, meta)
      - (frame, ts, meta, seq)

    Returns: (frame, ts, seq)
    """
    frame: Optional[np.ndarray] = None
    ts: float = 0.0
    seq: Optional[int] = None

    if isinstance(ret, (list, tuple)):
        if len(ret) >= 1:
            frame = _ensure_frame(ret[0])
        if len(ret) >= 2 and ret[1] is not None:
            try:
                ts = float(ret[1])
            except Exception:
                ts = 0.0
        if len(ret) >= 4 and ret[3] is not None:
            try:
                seq = int(ret[3])
            except Exception:
                seq = None
    else:
        frame = _ensure_frame(ret)

    return frame, float(ts), seq




def _frame_order_key(ts: float, seq: Optional[int]) -> Tuple[int, float]:
    """Return a comparable freshness key for raw/processed frame snapshots.

    Prefer sequence numbers when available; otherwise use wall-clock timestamp.
    The first tuple item makes seq-based keys always comparable ahead of
    timestamp-only keys without treating None as newer.
    """
    if seq is not None:
        try:
            return (1, float(int(seq)))
        except Exception:
            pass
    try:
        return (0, float(ts or 0.0))
    except Exception:
        return (0, 0.0)


def _is_newer_frame(ts: float, seq: Optional[int], last_seq: Optional[int], last_ts: float) -> bool:
    """True only when candidate is newer than the last frame yielded."""
    if seq is not None and last_seq is not None:
        return int(seq) > int(last_seq)
    if seq is not None and last_seq is None:
        # If previous frames only had timestamps, allow seq frames only when we
        # do not have a meaningful timestamp comparison available.
        return last_ts <= 0 or float(ts or 0.0) > float(last_ts)
    if seq is None and last_seq is not None:
        # Avoid switching back to timestamp-only older frames after seq frames.
        return float(ts or 0.0) > float(last_ts)
    return float(ts or 0.0) > float(last_ts)

def mjpeg_generator(
    buf: RenderedFrame,
    max_fps: int = 15,
    jpeg_quality: int = 80,
    raw_fallback: Optional[RenderedFrame] = None,
    fallback_after_ms: int = 250,
):
    """Processed-only low-latency MJPEG generator.

    IMPORTANT: this generator intentionally ignores ``raw_fallback``.

    The UI must show only frames that have completed the AI pipeline
    (YOLO/tracker/face/identity/drawing).  Mixing raw capture frames with
    processed frames creates visible forward/backward motion because the raw
    frame is newer than the last processed frame.  Therefore this generator:

      - reads only ``buf`` (the processed RenderedFrame),
      - never reads raw capture frames,
      - repeats the last processed JPEG while the AI worker is busy,
      - sends a placeholder only until the first processed frame exists,
      - never queues encoded JPEG frames.

    ``raw_fallback`` and ``fallback_after_ms`` remain in the signature only so
    older route/service code can still call this function safely.
    """
    max_fps = int(max(1, min(30, max_fps)))
    min_dt = 1.0 / float(max_fps)
    jpeg_quality = int(max(30, min(95, int(jpeg_quality))))

    # Change tracking. Prefer RenderedFrame.wait_for_seq() because get() in
    # some app versions returns only (frame, ts, meta) and does not expose seq.
    last_seq = -1
    last_ts = 0.0
    last_key: Optional[Tuple[float, Optional[int], int]] = None
    last_jpg: Optional[bytes] = None
    placeholder_jpg: Optional[bytes] = None

    next_send = time.monotonic()

    if buf is not None and hasattr(buf, "add_client"):
        try:
            buf.add_client()
        except Exception:
            pass

    try:
        while True:
            _sleep_until(next_send)

            frame: Optional[np.ndarray] = None
            ts = 0.0
            seq: Optional[int] = None

            # Preferred path: non-blocking processed-frame snapshot with seq.
            if hasattr(buf, "wait_for_seq"):
                try:
                    ret = buf.wait_for_seq(last_seq, timeout=0.0)  # type: ignore[attr-defined]
                    frame, ts, seq = _snapshot_from_get_result(ret)
                except Exception:
                    frame, ts, seq = None, 0.0, None

            # Fallback for older buffers.
            if frame is None and hasattr(buf, "get"):
                try:
                    frame, ts, seq = _snapshot_from_get_result(buf.get())  # type: ignore[misc]
                except Exception:
                    frame, ts, seq = None, 0.0, None

            if frame is None:
                if placeholder_jpg is None:
                    placeholder_jpg = _encode_jpeg(
                        _placeholder(text="Waiting for first processed AI frame..."),
                        jpeg_quality,
                    )
                if placeholder_jpg is not None:
                    yield _BOUNDARY + _HEADER + placeholder_jpg + _TAIL
            else:
                # Build a stable key. If seq is available, it is the source of truth.
                # Otherwise use ts. If neither exists, fall back to object id.
                key = (
                    float(ts or 0.0),
                    int(seq) if seq is not None else None,
                    int(id(frame)) if (not ts and seq is None) else 0,
                )

                changed = last_jpg is None or key != last_key
                if changed:
                    jpg: Optional[bytes] = None
                    if hasattr(buf, "wait_jpeg") and ts > 0:
                        try:
                            jpg, jt = buf.wait_jpeg(last_ts, timeout=0.0, jpeg_quality=jpeg_quality)  # type: ignore[attr-defined]
                            if jpg is not None:
                                last_ts = float(jt or ts)
                        except Exception:
                            jpg = None
                    if jpg is None:
                        jpg = _encode_jpeg(frame, jpeg_quality)
                        if jpg is not None and ts > 0:
                            last_ts = float(ts)
                    if jpg is not None:
                        last_jpg = jpg
                        last_key = key
                        if seq is not None:
                            last_seq = int(seq)
                # If no new processed frame arrived, repeat last processed JPEG.
                if last_jpg is not None:
                    yield _BOUNDARY + _HEADER + last_jpg + _TAIL

            now_mono = time.monotonic()
            next_send += min_dt
            if next_send < (now_mono - 0.25):
                next_send = now_mono + min_dt

    finally:
        if buf is not None and hasattr(buf, "remove_client"):
            try:
                buf.remove_client()
            except Exception:
                pass

def _grid_size(n: int, rows: int, cols: int) -> Tuple[int, int]:
    if rows > 0 and cols > 0:
        return rows, cols
    if rows > 0:
        cols = int(np.ceil(n / rows))
        return rows, cols
    if cols > 0:
        rows = int(np.ceil(n / cols))
        return rows, cols
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _resize_cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    scale = max(w / float(iw), h / float(ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    x1 = max(0, (nw - w) // 2)
    y1 = max(0, (nh - h) // 2)
    out = r[y1 : y1 + h, x1 : x1 + w]
    if out.shape[0] != h or out.shape[1] != w:
        out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
    return out


def _resize_contain(img: np.ndarray, w: int, h: int) -> np.ndarray:
    ih, iw = img.shape[:2]
    if iw <= 0 or ih <= 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    scale = min(w / float(iw), h / float(ih))
    nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
    r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    out = np.zeros((h, w, 3), dtype=np.uint8)
    x1 = (w - nw) // 2
    y1 = (h - nh) // 2
    out[y1 : y1 + nh, x1 : x1 + nw] = r
    return out


def _make_grid(frames: List[np.ndarray], out_w: int, out_h: int, mode: str, rows: int, cols: int) -> np.ndarray:
    n = len(frames)
    rows, cols = _grid_size(n, rows, cols)
    cell_w = max(1, out_w // cols)
    cell_h = max(1, out_h // rows)

    resize_fn = _resize_cover if str(mode).lower() == "cover" else _resize_contain

    tiles: List[np.ndarray] = []
    for i in range(rows * cols):
        if i < n:
            tiles.append(resize_fn(frames[i], cell_w, cell_h))
        else:
            tiles.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

    row_imgs = []
    idx = 0
    for _r in range(rows):
        row_imgs.append(np.concatenate(tiles[idx : idx + cols], axis=1))
        idx += cols

    grid = np.concatenate(row_imgs, axis=0)
    if grid.shape[1] != out_w or grid.shape[0] != out_h:
        grid = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return grid


def mjpeg_generator_multi(
    bufs: List[RenderedFrame],
    max_fps: int = 12,
    jpeg_quality: int = 80,
    out_w: int = 1280,
    out_h: int = 720,
    grid_mode: str = "cover",
    grid_rows: int = 0,
    grid_cols: int = 0,
):
    """Multi-cam MJPEG grid stream.

    Also defensive about `buf.get()` return shapes (1/2/3/4 values).

    It will rebuild the grid only when any source changes (ts/seq/id), and
    otherwise reuses the last encoded grid to reduce CPU.
    """
    max_fps = int(max(1, min(30, max_fps)))
    min_dt = 1.0 / float(max_fps)

    # Per-buffer change detectors
    last_ts: List[float] = [0.0 for _ in bufs]
    last_seq: List[Optional[int]] = [None for _ in bufs]
    last_id: List[Optional[int]] = [None for _ in bufs]

    last_jpg: Optional[bytes] = None

    next_send = time.monotonic()

    # Optional: mark active clients
    for b in bufs:
        if hasattr(b, "add_client"):
            try:
                b.add_client()
            except Exception:
                pass

    try:
        while True:
            _sleep_until(next_send)

            frames: List[np.ndarray] = []
            any_changed = False

            for i, b in enumerate(bufs):
                ret = None
                if hasattr(b, "get"):
                    try:
                        ret = b.get()  # type: ignore[misc]
                    except Exception:
                        ret = None

                frame, ts, seq = _snapshot_from_get_result(ret)
                if frame is None:
                    frame = _placeholder(text=f"Waiting cam {i} ...")

                # Determine if this source changed since last tick
                if seq is not None:
                    if last_seq[i] is None or int(seq) != int(last_seq[i]):
                        any_changed = True
                        last_seq[i] = int(seq)
                elif ts > 0:
                    if float(ts) > float(last_ts[i]):
                        any_changed = True
                        last_ts[i] = float(ts)
                else:
                    cur_id = id(frame)
                    if last_id[i] is None or int(cur_id) != int(last_id[i]):
                        any_changed = True
                        last_id[i] = int(cur_id)

                frames.append(frame)

            if (not any_changed) and (last_jpg is not None):
                yield _BOUNDARY + _HEADER + last_jpg + _TAIL
            else:
                grid = _make_grid(
                    frames,
                    out_w=int(out_w),
                    out_h=int(out_h),
                    mode=str(grid_mode),
                    rows=int(grid_rows),
                    cols=int(grid_cols),
                )
                jpg = _encode_jpeg(grid, jpeg_quality)
                if jpg is None:
                    jpg = _encode_jpeg(_placeholder(text="JPEG encode error"), jpeg_quality)
                    if jpg is None:
                        time.sleep(0.02)
                        continue
                last_jpg = jpg
                yield _BOUNDARY + _HEADER + last_jpg + _TAIL

            now = time.monotonic()
            next_send += min_dt
            if next_send < (now - 0.25):
                next_send = now + min_dt

    finally:
        for b in bufs:
            if hasattr(b, "remove_client"):
                try:
                    b.remove_client()
                except Exception:
                    pass


# from __future__ import annotations

# import time
# from typing import List, Optional, Tuple

# import cv2
# import numpy as np

# from app.services.tracking.pipeline import RenderedFrame


# _BOUNDARY = b"--frame\r\n"
# _HEADER = b"Content-Type: image/jpeg\r\n\r\n"
# _TAIL = b"\r\n"


# def _ensure_frame(x) -> Optional[np.ndarray]:
#     """Defensive: accept only a real numpy image."""
#     if x is None:
#         return None
#     if isinstance(x, np.ndarray):
#         return x
#     if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], np.ndarray):
#         return x[0]
#     return None


# def _encode_jpeg(frame_bgr: np.ndarray, quality: int) -> Optional[bytes]:
#     q = int(max(30, min(95, int(quality))))
#     try:
#         ok, enc = cv2.imencode(".jpg", frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), q])
#         if not ok:
#             return None
#         return enc.tobytes()
#     except Exception:
#         return None


# def _placeholder(w: int = 1280, h: int = 720, text: str = "Waiting for frames...") -> np.ndarray:
#     img = np.zeros((h, w, 3), dtype=np.uint8)
#     cv2.putText(img, text, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
#     return img


# def _sleep_until(deadline: float) -> None:
#     """Sleeps until monotonic deadline (best-effort), avoids negative sleep."""
#     now = time.monotonic()
#     dt = deadline - now
#     if dt > 0:
#         time.sleep(dt)


# def _snapshot_from_get_result(ret) -> Tuple[Optional[np.ndarray], float, Optional[int]]:
#     """Normalize `buf.get()` return values across versions.

#     Supported get() shapes seen across v1/v2 buffers:
#       - frame
#       - (frame, ts)
#       - (frame, ts, meta)
#       - (frame, ts, meta, seq)

#     Returns: (frame, ts, seq)
#     """
#     frame: Optional[np.ndarray] = None
#     ts: float = 0.0
#     seq: Optional[int] = None

#     if isinstance(ret, (list, tuple)):
#         if len(ret) >= 1:
#             frame = _ensure_frame(ret[0])
#         if len(ret) >= 2 and ret[1] is not None:
#             try:
#                 ts = float(ret[1])
#             except Exception:
#                 ts = 0.0
#         if len(ret) >= 4 and ret[3] is not None:
#             try:
#                 seq = int(ret[3])
#             except Exception:
#                 seq = None
#     else:
#         frame = _ensure_frame(ret)

#     return frame, float(ts), seq


# def mjpeg_generator(buf: RenderedFrame, max_fps: int = 15, jpeg_quality: int = 80):
#     """Streaming MJPEG generator.

#     Fixes the crash you hit:
#       ValueError: too many values to unpack (expected 3)

#     That happens because different RenderedFrame buffers return different tuple
#     sizes from `get()`. This implementation is defensive and supports:
#       - wait_jpeg() buffers (preferred, cached JPEG)
#       - get() buffers returning 1/2/3/4 values

#     It also:
#       - runs on a monotonic FPS scheduler (less jitter)
#       - reuses last JPEG if no new frame is available
#     """
#     max_fps = int(max(1, min(30, max_fps)))
#     min_dt = 1.0 / float(max_fps)

#     has_wait_jpeg = hasattr(buf, "wait_jpeg")
#     has_get = hasattr(buf, "get")

#     # Change detectors
#     last_ts = 0.0
#     last_seq: Optional[int] = None
#     last_frame_id: Optional[int] = None

#     last_jpg: Optional[bytes] = None

#     next_send = time.monotonic()

#     # Optional: some buffers enable jpeg caching only when a client is attached
#     if hasattr(buf, "add_client"):
#         try:
#             buf.add_client()
#         except Exception:
#             pass

#     try:
#         while True:
#             _sleep_until(next_send)

#             updated = False

#             # Preferred path: ask buffer for cached JPEG (fast & low CPU)
#             if has_wait_jpeg:
#                 try:
#                     # Try a couple of non-blocking pulls to land on the newest ts
#                     for _ in range(3):
#                         jpg, ts = buf.wait_jpeg(last_ts, timeout=0.0, jpeg_quality=jpeg_quality)  # type: ignore[attr-defined]
#                         if jpg is not None and float(ts) > float(last_ts):
#                             last_jpg = jpg
#                             last_ts = float(ts)
#                             updated = True
#                         else:
#                             break
#                 except Exception:
#                     # ignore and fall back to get()
#                     pass

#             # Fallback: pull raw frame via get() and encode here
#             if (not updated) and has_get:
#                 try:
#                     ret = buf.get()  # type: ignore[misc]
#                 except Exception:
#                     ret = None

#                 frame, ts, seq = _snapshot_from_get_result(ret)
#                 if frame is not None:
#                     if seq is not None:
#                         changed = (last_seq is None) or (int(seq) != int(last_seq))
#                     elif ts > 0:
#                         changed = float(ts) > float(last_ts)
#                     else:
#                         cur_id = id(frame)
#                         changed = (last_frame_id is None) or (int(cur_id) != int(last_frame_id))
#                         last_frame_id = int(cur_id)

#                     if last_jpg is None or changed:
#                         jpg = _encode_jpeg(frame, jpeg_quality)
#                         if jpg is not None:
#                             last_jpg = jpg
#                             if ts > 0:
#                                 last_ts = float(ts)
#                             if seq is not None:
#                                 last_seq = int(seq)

#             if last_jpg is None:
#                 ph = _placeholder(text="Starting camera / waiting for first frame...")
#                 last_jpg = _encode_jpeg(ph, jpeg_quality)
#                 if last_jpg is None:
#                     time.sleep(0.02)
#                     continue

#             yield _BOUNDARY + _HEADER + last_jpg + _TAIL

#             now = time.monotonic()
#             next_send += min_dt
#             if next_send < (now - 0.25):
#                 next_send = now + min_dt

#     finally:
#         if hasattr(buf, "remove_client"):
#             try:
#                 buf.remove_client()
#             except Exception:
#                 pass


# def _grid_size(n: int, rows: int, cols: int) -> Tuple[int, int]:
#     if rows > 0 and cols > 0:
#         return rows, cols
#     if rows > 0:
#         cols = int(np.ceil(n / rows))
#         return rows, cols
#     if cols > 0:
#         rows = int(np.ceil(n / cols))
#         return rows, cols
#     cols = int(np.ceil(np.sqrt(n)))
#     rows = int(np.ceil(n / cols))
#     return rows, cols


# def _resize_cover(img: np.ndarray, w: int, h: int) -> np.ndarray:
#     ih, iw = img.shape[:2]
#     if iw <= 0 or ih <= 0:
#         return np.zeros((h, w, 3), dtype=np.uint8)
#     scale = max(w / float(iw), h / float(ih))
#     nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
#     r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
#     x1 = max(0, (nw - w) // 2)
#     y1 = max(0, (nh - h) // 2)
#     out = r[y1 : y1 + h, x1 : x1 + w]
#     if out.shape[0] != h or out.shape[1] != w:
#         out = cv2.resize(out, (w, h), interpolation=cv2.INTER_LINEAR)
#     return out


# def _resize_contain(img: np.ndarray, w: int, h: int) -> np.ndarray:
#     ih, iw = img.shape[:2]
#     if iw <= 0 or ih <= 0:
#         return np.zeros((h, w, 3), dtype=np.uint8)
#     scale = min(w / float(iw), h / float(ih))
#     nw, nh = max(1, int(iw * scale)), max(1, int(ih * scale))
#     r = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
#     out = np.zeros((h, w, 3), dtype=np.uint8)
#     x1 = (w - nw) // 2
#     y1 = (h - nh) // 2
#     out[y1 : y1 + nh, x1 : x1 + nw] = r
#     return out


# def _make_grid(frames: List[np.ndarray], out_w: int, out_h: int, mode: str, rows: int, cols: int) -> np.ndarray:
#     n = len(frames)
#     rows, cols = _grid_size(n, rows, cols)
#     cell_w = max(1, out_w // cols)
#     cell_h = max(1, out_h // rows)

#     resize_fn = _resize_cover if str(mode).lower() == "cover" else _resize_contain

#     tiles: List[np.ndarray] = []
#     for i in range(rows * cols):
#         if i < n:
#             tiles.append(resize_fn(frames[i], cell_w, cell_h))
#         else:
#             tiles.append(np.zeros((cell_h, cell_w, 3), dtype=np.uint8))

#     row_imgs = []
#     idx = 0
#     for _r in range(rows):
#         row_imgs.append(np.concatenate(tiles[idx : idx + cols], axis=1))
#         idx += cols

#     grid = np.concatenate(row_imgs, axis=0)
#     if grid.shape[1] != out_w or grid.shape[0] != out_h:
#         grid = cv2.resize(grid, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
#     return grid


# def mjpeg_generator_multi(
#     bufs: List[RenderedFrame],
#     max_fps: int = 12,
#     jpeg_quality: int = 80,
#     out_w: int = 1280,
#     out_h: int = 720,
#     grid_mode: str = "cover",
#     grid_rows: int = 0,
#     grid_cols: int = 0,
# ):
#     """Multi-cam MJPEG grid stream.

#     Also defensive about `buf.get()` return shapes (1/2/3/4 values).

#     It will rebuild the grid only when any source changes (ts/seq/id), and
#     otherwise reuses the last encoded grid to reduce CPU.
#     """
#     max_fps = int(max(1, min(30, max_fps)))
#     min_dt = 1.0 / float(max_fps)

#     # Per-buffer change detectors
#     last_ts: List[float] = [0.0 for _ in bufs]
#     last_seq: List[Optional[int]] = [None for _ in bufs]
#     last_id: List[Optional[int]] = [None for _ in bufs]

#     last_jpg: Optional[bytes] = None

#     next_send = time.monotonic()

#     # Optional: mark active clients
#     for b in bufs:
#         if hasattr(b, "add_client"):
#             try:
#                 b.add_client()
#             except Exception:
#                 pass

#     try:
#         while True:
#             _sleep_until(next_send)

#             frames: List[np.ndarray] = []
#             any_changed = False

#             for i, b in enumerate(bufs):
#                 ret = None
#                 if hasattr(b, "get"):
#                     try:
#                         ret = b.get()  # type: ignore[misc]
#                     except Exception:
#                         ret = None

#                 frame, ts, seq = _snapshot_from_get_result(ret)
#                 if frame is None:
#                     frame = _placeholder(text=f"Waiting cam {i} ...")

#                 # Determine if this source changed since last tick
#                 if seq is not None:
#                     if last_seq[i] is None or int(seq) != int(last_seq[i]):
#                         any_changed = True
#                         last_seq[i] = int(seq)
#                 elif ts > 0:
#                     if float(ts) > float(last_ts[i]):
#                         any_changed = True
#                         last_ts[i] = float(ts)
#                 else:
#                     cur_id = id(frame)
#                     if last_id[i] is None or int(cur_id) != int(last_id[i]):
#                         any_changed = True
#                         last_id[i] = int(cur_id)

#                 frames.append(frame)

#             if (not any_changed) and (last_jpg is not None):
#                 yield _BOUNDARY + _HEADER + last_jpg + _TAIL
#             else:
#                 grid = _make_grid(
#                     frames,
#                     out_w=int(out_w),
#                     out_h=int(out_h),
#                     mode=str(grid_mode),
#                     rows=int(grid_rows),
#                     cols=int(grid_cols),
#                 )
#                 jpg = _encode_jpeg(grid, jpeg_quality)
#                 if jpg is None:
#                     jpg = _encode_jpeg(_placeholder(text="JPEG encode error"), jpeg_quality)
#                     if jpg is None:
#                         time.sleep(0.02)
#                         continue
#                 last_jpg = jpg
#                 yield _BOUNDARY + _HEADER + last_jpg + _TAIL

#             now = time.monotonic()
#             next_send += min_dt
#             if next_send < (now - 0.25):
#                 next_send = now + min_dt

#     finally:
#         for b in bufs:
#             if hasattr(b, "remove_client"):
#                 try:
#                     b.remove_client()
#                 except Exception:
#                     pass