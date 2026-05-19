from __future__ import annotations

import json
import queue
import threading
import time
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Deque, Dict, Iterable, List, Optional

import redis as sync_redis
from sqlalchemy import insert

from app.core.config import settings
from app.db.base import Base
from app.db.models.detection import Detection
from app.db.session import SessionLocal, engine
from app.services.ws_service import METADATA_CHANNEL

LATEST_META_KEY = "detection:latest:{camera_id}"
META_TTL_SECONDS = 10
SYNC_HISTORY_SECONDS = 30.0
SYNC_HISTORY_FRAMES = 1200
KNOWN_ONLY = True


def _as_int(value: Any, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except Exception:
        return default


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except Exception:
        return default


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return bool(value)
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "y", "known"}:
        return True
    if text in {"0", "false", "no", "n", "unknown", ""}:
        return False
    return default


def _clean_name(*values: Any) -> str:
    for value in values:
        text = str(value or "").strip()
        if not text:
            continue
        if text.lower() in {"unknown", "none", "null", "-"}:
            continue
        return text
    return ""


def _bbox_list(value: Any) -> List[int]:
    try:
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except Exception:
                value = value.strip().strip("[]").split(",")
        vals = list(value or [])[:4]
        if len(vals) != 4:
            return [0, 0, 0, 0]
        return [int(round(float(str(v).strip()))) for v in vals]
    except Exception:
        return [0, 0, 0, 0]


def ensure_detection_table() -> None:
    """Create the detection table when migrations are not being used."""
    import app.db.models  # noqa: F401 - registers referenced tables

    Base.metadata.create_all(bind=engine, tables=[Detection.__table__])


def _is_known_track(track: Dict[str, Any]) -> bool:
    """Only recognized members are eligible for UI overlay and DB storage."""
    if not isinstance(track, dict):
        return False
    name = _clean_name(track.get("name"), track.get("person_name"), track.get("member_name"), track.get("label"))
    member_id = _as_int(track.get("member_id"), -1)
    return bool(member_id > 0 and name and name.lower() != "unknown")


def _track_from_tuple(ev: Any) -> Optional[Dict[str, Any]]:
    try:
        # Legacy tuple shape:
        # (track_id, x1, y1, x2, y2, name, score, member_id, is_known)
        track_id = _as_int(ev[0], -1)
        bbox = [_as_int(ev[1]), _as_int(ev[2]), _as_int(ev[3]), _as_int(ev[4])]
        name = _clean_name(ev[5] if len(ev) > 5 else "")
        score = _as_float(ev[6] if len(ev) > 6 else 0.0, 0.0)
        member_id = _as_int(ev[7] if len(ev) > 7 else -1, -1)
        tuple_known = _as_bool(ev[8] if len(ev) > 8 else True, True)
        is_known = bool(tuple_known and member_id > 0 and name)
    except Exception:
        return None
    if track_id < 0:
        return None
    return {
        "track_id": int(track_id),
        "raw_track_id": int(track_id),
        "bbox": bbox,
        "name": name if is_known else None,
        "label": name if is_known else "",
        "member_id": int(member_id) if is_known else None,
        "is_known": bool(is_known),
        "face_conf": float(score),
        "det_conf": None,
    }


def _normalize_track(track: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(track, dict):
        return _track_from_tuple(track)

    track_id = _as_int(
        track.get("track_id", track.get("tid", track.get("logical_tid", track.get("id", -1)))),
        -1,
    )
    if track_id < 0:
        return None

    raw_track_id = _as_int(track.get("raw_track_id", track.get("raw_tid", track_id)), track_id)
    bbox = _bbox_list(track.get("bbox"))
    name = _clean_name(track.get("name"), track.get("person_name"), track.get("member_name"), track.get("label"))
    member_id = _as_int(track.get("member_id"), -1)

    # Do not trust `is_known` alone. Some pipeline branches have written
    # is_known=false even after the identity is resolved. The real rule for UI
    # and audit storage is: recognized name + valid member_id.
    is_known = bool(member_id > 0 and name)

    face_conf = _as_float(
        track.get("face_conf", track.get("face_sim", track.get("sim", track.get("score", 0.0)))),
        0.0,
    )

    return {
        "track_id": int(track_id),
        "raw_track_id": int(raw_track_id),
        "bbox": bbox,
        "name": name if is_known else None,
        "label": name if is_known else "",
        "member_id": int(member_id) if is_known else None,
        "is_known": bool(is_known),
        "face_conf": float(face_conf),
        "det_conf": None if track.get("det_conf") is None else _as_float(track.get("det_conf"), 0.0),
    }


def _normalize_tracks(frame_meta: Dict[str, Any], *, known_only: bool = True) -> List[Dict[str, Any]]:
    tracks_raw = frame_meta.get("visible_tracks")
    if tracks_raw is None:
        tracks_raw = frame_meta.get("security_events")
    if tracks_raw is None:
        tracks_raw = frame_meta.get("events", [])

    out: List[Dict[str, Any]] = []
    for item in tracks_raw or []:
        tr = _normalize_track(item)
        if tr is None:
            continue
        if known_only and not _is_known_track(tr):
            continue
        out.append(tr)
    return out


def empty_metadata_payload(camera_id: int) -> Dict[str, Any]:
    now = time.time()
    return {
        "event": "detection_metadata",
        "camera_id": int(camera_id),
        "timestamp": float(now),
        "capture_ts": float(now),
        "frame_seq": None,
        "processing_ts": float(now),
        "processing_latency_ms": 0.0,
        "server_now": float(now),
        "recommended_video_delay_ms": 900,
        "frame_width": None,
        "frame_height": None,
        "fps": None,
        "tracks": [],
        "shown": 0,
        "known_only": True,
        "all_track_count": 0,
        "known_track_count": 0,
        "present_names": [],
        "present_conf": {},
        "source": "empty",
    }


def build_metadata_payload(
    *,
    camera_id: int,
    timestamp: float,
    frame_seq: Optional[int],
    frame_meta: Dict[str, Any],
    fps: Optional[float] = None,
) -> Dict[str, Any]:
    processing_ts = time.time()
    ts = float(timestamp or processing_ts)

    all_tracks = _normalize_tracks(frame_meta or {}, known_only=False)
    known_tracks = [tr for tr in all_tracks if _is_known_track(tr)]
    tracks = known_tracks if KNOWN_ONLY else all_tracks

    frame_width = _as_int((frame_meta or {}).get("frame_width"), 0)
    frame_height = _as_int((frame_meta or {}).get("frame_height"), 0)

    present_conf: Dict[str, float] = {}
    for tr in tracks:
        name = _clean_name(tr.get("name"), tr.get("label"))
        if not name:
            continue
        present_conf[name] = max(float(present_conf.get(name, 0.0)), _as_float(tr.get("face_conf"), 0.0))

    latency_ms = max(0.0, (processing_ts - ts) * 1000.0)
    recommended_delay_ms = int(max(500, min(1500, latency_ms + 180.0)))

    return {
        "event": "detection_metadata",
        "camera_id": int(camera_id),
        "timestamp": float(ts),
        "capture_ts": float(ts),
        "frame_seq": int(frame_seq) if frame_seq is not None else None,
        "processing_ts": float(processing_ts),
        "processing_latency_ms": float(latency_ms),
        "server_now": float(processing_ts),
        "recommended_video_delay_ms": int(recommended_delay_ms),
        "frame_width": int(frame_width) if frame_width > 0 else None,
        "frame_height": int(frame_height) if frame_height > 0 else None,
        "fps": None if fps is None else float(fps),
        "tracks": tracks,
        "shown": int(len(tracks)),
        "known_only": bool(KNOWN_ONLY),
        "all_track_count": int(len(all_tracks)),
        "known_track_count": int(len(known_tracks)),
        "present_names": sorted(present_conf.keys()),
        "present_conf": present_conf,
        "source": "live",
    }


def detection_rows_from_payload(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    camera_id = _as_int(payload.get("camera_id"), -1)
    if camera_id <= 0:
        return rows

    timestamp = _as_float(payload.get("timestamp", payload.get("capture_ts")), time.time())
    frame_seq = payload.get("frame_seq")
    frame_width = payload.get("frame_width")
    frame_height = payload.get("frame_height")
    processing_latency_ms = payload.get("processing_latency_ms")

    for track in payload.get("tracks", []) or []:
        tr = _normalize_track(track)
        if tr is None or not _is_known_track(tr):
            continue
        rows.append({
            "camera_id": int(camera_id),
            "member_id": int(tr["member_id"]),
            "timestamp": float(timestamp),
            "frame_seq": int(frame_seq) if frame_seq is not None else None,
            "track_id": int(tr["track_id"]),
            "raw_track_id": int(tr.get("raw_track_id") or tr["track_id"]),
            "bbox": list(tr.get("bbox") or [0, 0, 0, 0]),
            "person_name": _clean_name(tr.get("name"), tr.get("label")),
            "is_known": True,
            "face_conf": tr.get("face_conf"),
            "det_conf": tr.get("det_conf"),
            "frame_width": int(frame_width) if frame_width else None,
            "frame_height": int(frame_height) if frame_height else None,
            "processing_latency_ms": float(processing_latency_ms) if processing_latency_ms is not None else None,
        })
    return rows


def latest_payload_from_db(camera_id: int, max_age_ms: int = 2500) -> Optional[Dict[str, Any]]:
    """Return recent known-person metadata from PostgreSQL when Redis/pubsub missed it."""
    camera_id = _as_int(camera_id, -1)
    if camera_id <= 0:
        return None

    max_age_ms = int(max(250, min(30000, int(max_age_ms or 2500))))
    cutoff = datetime.now(timezone.utc) - timedelta(milliseconds=max_age_ms)

    db = SessionLocal()
    try:
        latest = (
            db.query(Detection)
            .filter(Detection.camera_id == camera_id)
            .filter(Detection.is_known.is_(True))
            .filter(Detection.member_id.isnot(None))
            .filter(Detection.member_id > 0)
            .filter(Detection.person_name.isnot(None))
            .filter(Detection.person_name != "")
            .filter(Detection.created_ts >= cutoff)
            .order_by(Detection.created_ts.desc(), Detection.id.desc())
            .first()
        )
        if latest is None:
            return None

        q = (
            db.query(Detection)
            .filter(Detection.camera_id == camera_id)
            .filter(Detection.is_known.is_(True))
            .filter(Detection.member_id.isnot(None))
            .filter(Detection.member_id > 0)
            .filter(Detection.person_name.isnot(None))
            .filter(Detection.person_name != "")
        )
        if latest.frame_seq is not None:
            q = q.filter(Detection.frame_seq == latest.frame_seq)
        else:
            q = q.filter(Detection.timestamp == latest.timestamp)

        rows = q.order_by(Detection.id.asc()).all()
        tracks: List[Dict[str, Any]] = []
        present_conf: Dict[str, float] = {}
        for row in rows:
            name = _clean_name(row.person_name)
            member_id = _as_int(row.member_id, -1)
            if member_id <= 0 or not name:
                continue
            bbox = _bbox_list(row.bbox)
            track = {
                "track_id": int(row.track_id),
                "raw_track_id": int(row.raw_track_id or row.track_id),
                "bbox": bbox,
                "name": name,
                "label": name,
                "member_id": int(member_id),
                "is_known": True,
                "face_conf": None if row.face_conf is None else float(row.face_conf),
                "det_conf": None if row.det_conf is None else float(row.det_conf),
            }
            tracks.append(track)
            present_conf[name] = max(float(present_conf.get(name, 0.0)), float(row.face_conf or 0.0))

        now = time.time()
        return {
            "event": "detection_metadata",
            "camera_id": int(camera_id),
            "timestamp": float(latest.timestamp),
            "capture_ts": float(latest.timestamp),
            "frame_seq": int(latest.frame_seq) if latest.frame_seq is not None else None,
            "processing_ts": float(now),
            "processing_latency_ms": None,
            "server_now": float(now),
            "recommended_video_delay_ms": 900,
            "frame_width": int(latest.frame_width) if latest.frame_width else None,
            "frame_height": int(latest.frame_height) if latest.frame_height else None,
            "fps": None,
            "tracks": tracks,
            "shown": int(len(tracks)),
            "known_only": True,
            "all_track_count": int(len(tracks)),
            "known_track_count": int(len(tracks)),
            "present_names": sorted(present_conf.keys()),
            "present_conf": present_conf,
            "source": "postgres_fallback",
        }
    finally:
        db.close()


class DetectionMetadataPublisher:
    """
    Publishes live overlay metadata independently from PostgreSQL writes.

    Live path: latest metadata per camera goes to Redis/WebSocket immediately.
    DB path: known-person rows are inserted on a separate thread. Slow DB writes
    cannot delay boxes in the UI.
    """

    def __init__(self, max_db_queue: int = 2000):
        self._db_queue: "queue.Queue[Dict[str, Any]]" = queue.Queue(maxsize=max(50, int(max_db_queue)))
        self._stop = threading.Event()
        self._live_event = threading.Event()
        self._live_lock = threading.Lock()
        self._live_latest: Dict[int, Dict[str, Any]] = {}
        self._published_latest: Dict[int, Dict[str, Any]] = {}
        self._history_by_camera: Dict[int, Deque[Dict[str, Any]]] = {}
        self._live_thread: Optional[threading.Thread] = None
        self._db_thread: Optional[threading.Thread] = None
        self._redis = None
        self._last_error_ts = 0.0
        self._started = False
        self._start_lock = threading.Lock()

    def start(self) -> None:
        with self._start_lock:
            if self._started:
                return
            try:
                ensure_detection_table()
            except Exception as exc:
                self._log_error(f"ensure table failed: {exc}")

            self._stop.clear()
            self._live_thread = threading.Thread(target=self._run_live, name="detection-metadata-live", daemon=True)
            self._db_thread = threading.Thread(target=self._run_db, name="detection-metadata-db", daemon=True)
            self._live_thread.start()
            self._db_thread.start()
            self._started = True

    def stop(self) -> None:
        self._stop.set()
        self._live_event.set()
        for t in (self._live_thread, self._db_thread):
            if t is not None:
                try:
                    t.join(timeout=3.0)
                except Exception:
                    pass
        self._live_thread = None
        self._db_thread = None
        self._started = False
        if self._redis is not None:
            try:
                self._redis.close()
            except Exception:
                pass
        self._redis = None

    def submit_nowait(self, payload: Dict[str, Any]) -> None:
        if not self._started:
            try:
                self.start()
            except Exception as exc:
                self._log_error(f"start failed: {exc}")
                return

        camera_id = _as_int((payload or {}).get("camera_id"), -1)
        if camera_id <= 0:
            return

        payload = dict(payload or {})
        payload["sync_ready"] = True

        # Live overlay path: keep only the newest payload per camera, but keep
        # a short in-memory history by frame_seq for synced MJPEG release.
        with self._live_lock:
            self._remember_payload_locked(int(camera_id), payload)
            self._live_latest[int(camera_id)] = payload
        self._live_event.set()

        # DB audit path: only known-person payloads need insertion.
        if not payload.get("tracks"):
            return
        try:
            self._db_queue.put_nowait(payload)
        except queue.Full:
            # DB fell behind. Drop old audit frames, never block inference/UI.
            dropped = 0
            while dropped < 25:
                try:
                    self._db_queue.get_nowait()
                    self._db_queue.task_done()
                    dropped += 1
                except Exception:
                    break
            try:
                self._db_queue.put_nowait(payload)
            except Exception:
                self._log_error("db queue full; dropped metadata frame")
        except Exception as exc:
            self._log_error(f"submit failed: {exc}")

    def _remember_payload_locked(self, camera_id: int, payload: Dict[str, Any]) -> None:
        hist = self._history_by_camera.get(int(camera_id))
        if hist is None:
            hist = deque(maxlen=SYNC_HISTORY_FRAMES)
            self._history_by_camera[int(camera_id)] = hist

        p = dict(payload or {})
        hist.append(p)

        now = time.time()
        cutoff = now - float(SYNC_HISTORY_SECONDS)
        while len(hist) > 1:
            try:
                ts = _as_float(hist[0].get("capture_ts", hist[0].get("timestamp")), now)
            except Exception:
                ts = now
            if ts < cutoff:
                hist.popleft()
                continue
            break

    def get_latest_payload(self, camera_id: int) -> Optional[Dict[str, Any]]:
        with self._live_lock:
            p = self._published_latest.get(int(camera_id))
            if p is None:
                hist = self._history_by_camera.get(int(camera_id))
                if hist:
                    p = hist[-1]
            return dict(p) if isinstance(p, dict) else None

    def get_payload_for_frame(
        self,
        camera_id: int,
        *,
        frame_seq: Optional[int] = None,
        capture_ts: Optional[float] = None,
        tolerance_ms: int = 80,
    ) -> Optional[Dict[str, Any]]:
        cid = int(camera_id)
        tol_s = max(0.0, float(tolerance_ms or 0) / 1000.0)
        with self._live_lock:
            hist = list(self._history_by_camera.get(cid) or [])
        if not hist:
            return None

        if frame_seq is not None:
            try:
                target_seq = int(frame_seq)
            except Exception:
                target_seq = None
            if target_seq is not None:
                for p in reversed(hist):
                    try:
                        pseq = int(p.get("frame_seq")) if p.get("frame_seq") is not None else None
                    except Exception:
                        pseq = None
                    if pseq == target_seq:
                        return dict(p)

        if capture_ts is not None and float(capture_ts or 0.0) > 0.0:
            target_ts = float(capture_ts)
            best = None
            best_delta = None
            for p in reversed(hist):
                pts = _as_float(p.get("capture_ts", p.get("timestamp")), 0.0)
                if pts <= 0:
                    continue
                delta = abs(float(pts) - target_ts)
                if best_delta is None or delta < best_delta:
                    best = p
                    best_delta = delta
            if best is not None and best_delta is not None and best_delta <= tol_s:
                return dict(best)

        return None

    def publish_display_payload(self, payload: Dict[str, Any]) -> None:
        """Publish metadata at the moment synced MJPEG releases its matching raw frame.

        This does not write to PostgreSQL and does not add a duplicate audit row.
        It only updates Redis/WebSocket so the canvas receives the exact frame's
        metadata when that frame is sent to the browser.
        """
        if not self._started:
            try:
                self.start()
            except Exception as exc:
                self._log_error(f"start failed: {exc}")
                return
        p = dict(payload or {})
        camera_id = _as_int(p.get("camera_id"), -1)
        if camera_id <= 0:
            return
        p["source"] = "sync_display"
        p["sync_display"] = True
        p["server_now"] = float(time.time())
        self._publish_payload(p)

    def _get_redis(self):
        if self._redis is not None:
            return self._redis
        self._redis = sync_redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_keepalive=True,
            health_check_interval=30,
        )
        return self._redis

    def _run_live(self) -> None:
        while not self._stop.is_set():
            self._live_event.wait(timeout=0.25)
            self._live_event.clear()

            with self._live_lock:
                batch = list(self._live_latest.values())
                self._live_latest.clear()

            if not batch:
                continue

            for payload in batch:
                self._publish_payload(payload)

        # Flush any last clear/detection message on shutdown.
        with self._live_lock:
            batch = list(self._live_latest.values())
            self._live_latest.clear()
        for payload in batch:
            self._publish_payload(payload)

    def _publish_payload(self, payload: Dict[str, Any]) -> None:
        try:
            redis = self._get_redis()
            camera_id = _as_int(payload.get("camera_id"), -1)
            if camera_id <= 0:
                return
            data = json.dumps(payload, separators=(",", ":"))
            with self._live_lock:
                self._published_latest[int(camera_id)] = dict(payload)
            redis.setex(LATEST_META_KEY.format(camera_id=camera_id), META_TTL_SECONDS, data)
            redis.publish(METADATA_CHANNEL, data)
        except Exception as exc:
            self._log_error(f"redis publish failed: {exc}")
            try:
                if self._redis is not None:
                    self._redis.close()
            except Exception:
                pass
            self._redis = None

    def _run_db(self) -> None:
        while (not self._stop.is_set()) or (not self._db_queue.empty()):
            batch: List[Dict[str, Any]] = []
            try:
                item = self._db_queue.get(timeout=0.2)
                batch.append(item)
            except queue.Empty:
                continue
            except Exception:
                continue

            while len(batch) < 200:
                try:
                    batch.append(self._db_queue.get_nowait())
                except queue.Empty:
                    break
                except Exception:
                    break

            try:
                self._insert_batch(batch)
            except Exception as exc:
                self._log_error(f"db batch failed: {exc}")
            finally:
                for _ in batch:
                    try:
                        self._db_queue.task_done()
                    except Exception:
                        pass

    def _insert_batch(self, batch: Iterable[Dict[str, Any]]) -> None:
        rows: List[Dict[str, Any]] = []
        for payload in batch:
            rows.extend(detection_rows_from_payload(payload))
        if not rows:
            return
        db = SessionLocal()
        try:
            db.execute(insert(Detection), rows)
            db.commit()
        except Exception as exc:
            db.rollback()
            self._log_error(f"postgres insert failed: {exc}")
        finally:
            db.close()

    def _log_error(self, msg: str) -> None:
        now = time.monotonic()
        if (now - self._last_error_ts) >= 2.0:
            self._last_error_ts = now
            print(f"[DetectionMetadata] {msg}")


def publish_display_payload(payload: Dict[str, Any]) -> None:
    _publisher.publish_display_payload(dict(payload or {}))


def get_latest_live_payload(camera_id: int) -> Optional[Dict[str, Any]]:
    return _publisher.get_latest_payload(int(camera_id))


def get_live_payload_for_frame(
    camera_id: int,
    *,
    frame_seq: Optional[int] = None,
    capture_ts: Optional[float] = None,
    tolerance_ms: int = 80,
) -> Optional[Dict[str, Any]]:
    return _publisher.get_payload_for_frame(
        int(camera_id),
        frame_seq=frame_seq,
        capture_ts=capture_ts,
        tolerance_ms=int(tolerance_ms),
    )


_publisher = DetectionMetadataPublisher()


def start_detection_metadata_publisher() -> None:
    _publisher.start()


def stop_detection_metadata_publisher() -> None:
    _publisher.stop()


def submit_detection_metadata(
    *,
    camera_id: int,
    timestamp: float,
    frame_seq: Optional[int],
    frame_meta: Dict[str, Any],
    fps: Optional[float] = None,
) -> None:
    payload = build_metadata_payload(
        camera_id=int(camera_id),
        timestamp=float(timestamp or time.time()),
        frame_seq=frame_seq,
        frame_meta=dict(frame_meta or {}),
        fps=fps,
    )
    _publisher.submit_nowait(payload)
