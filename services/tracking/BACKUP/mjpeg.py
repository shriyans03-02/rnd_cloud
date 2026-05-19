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


def mjpeg_generator(buf: RenderedFrame, max_fps: int = 15, jpeg_quality: int = 80):
    """
    Smoother MJPEG:
    - Stable monotonic scheduler (less jitter)
    - Prefers buf.wait_jpeg() (cached JPEG, far less CPU)
    - Drops old frames (sends latest)
    - Reuses last_jpg if no new data (smooth continuous stream)
    """
    max_fps = int(max(1, min(30, max_fps)))
    min_dt = 1.0 / float(max_fps)

    has_wait_jpeg = hasattr(buf, "wait_jpeg")

    last_seq = -1
    last_ts = 0.0
    last_jpg: Optional[bytes] = None

    next_send = time.monotonic()

    # Tell buffer we have an active MJPEG client (helps caching policy)
    if hasattr(buf, "add_client"):
        try:
            buf.add_client()
        except Exception:
            pass

    try:
        while True:
            _sleep_until(next_send)

            if has_wait_jpeg:
                # try a few non-blocking pulls to land on newest ts
                for _ in range(3):
                    jpg, ts = buf.wait_jpeg(last_ts, timeout=0.0, jpeg_quality=jpeg_quality)  # type: ignore[attr-defined]
                    if ts > last_ts and jpg is not None:
                        last_ts = float(ts)
                        last_jpg = jpg
                    else:
                        break
            else:
                for _ in range(3):
                    frm, _ts, _meta, seq = buf.wait_for_seq(last_seq, timeout=0.0)
                    if int(seq) != int(last_seq):
                        last_seq = int(seq)
                    else:
                        break

            if last_jpg is None:
                frm, _ts, _meta, seq = buf.wait_for_seq(last_seq, timeout=0.0)
                last_seq = int(seq)
                frame = _ensure_frame(frm)
                if frame is None:
                    frame = _placeholder(text="Starting camera / waiting for first frame...")

                last_jpg = _encode_jpeg(frame, jpeg_quality)
                if last_jpg is None:
                    last_jpg = _encode_jpeg(_placeholder(text="JPEG encode error"), jpeg_quality)
                    if last_jpg is None:
                        time.sleep(0.02)
                        continue

            yield _BOUNDARY + _HEADER + last_jpg + _TAIL

            now = time.monotonic()
            next_send += min_dt
            if next_send < (now - 0.25):
                next_send = now + min_dt

    finally:
        if hasattr(buf, "remove_client"):
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
    out = r[y1:y1 + h, x1:x1 + w]
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
    out[y1:y1 + nh, x1:x1 + nw] = r
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
        row_imgs.append(np.concatenate(tiles[idx:idx + cols], axis=1))
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
    """
    Smoother multi-cam MJPEG:
    - Stable monotonic FPS scheduler
    - Always uses latest frames (drops old)
    - Reuses last encoded grid if nothing changes (reduces CPU → smoother)
    """
    max_fps = int(max(1, min(30, max_fps)))
    min_dt = 1.0 / float(max_fps)

    last_seqs = [-1 for _ in bufs]
    last_jpg: Optional[bytes] = None

    next_send = time.monotonic()

    # mark clients (optional)
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

            # Pull latest from each buffer without blocking
            for i, b in enumerate(bufs):
                frm, _ts, _meta, seq = b.wait_for_seq(last_seqs[i], timeout=0.0)
                s = int(seq)
                if s != int(last_seqs[i]):
                    any_changed = True
                    last_seqs[i] = s

                frame = _ensure_frame(frm)
                if frame is None:
                    frame = _placeholder(text=f"Waiting cam {i} ...")
                frames.append(frame)

            # If nothing changed and we have a cached grid jpg, reuse it
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
