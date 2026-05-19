from __future__ import annotations

"""
FastAPI-ready person tracking runner.

Key rules implemented:
- Cameras are taken from DB (cameras.ip_address) when --use-db is enabled.
- In DB mode, if --src has no numeric camera IDs, ALL active cameras are started at startup.
- Unknown people are NEVER drawn.
- Only known / recognized people are drawn with GREEN boxes.

Ingestion:
- Every tracked person (known or unknown) is logged to rotating raw tables.
- guest_data_vector stores UNKNOWN embeddings only (face or body) — NO bbox coordinates.
- match_value stores similarity (0..100):
	* known: face similarity to gallery
	* unknown: best similarity to gallery (face-best if face seen else body-best)
"""

import argparse
import calendar
import csv
import gzip
import json
import os
import queue
import shlex
import subprocess
import sys
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import quote, unquote, urlsplit, urlunsplit

import cv2
import numpy as np
import torch

# Optional deps
try:
	from ultralytics import YOLO
except Exception:
	YOLO = None

try:
	from deep_sort_realtime.deepsort_tracker import DeepSort
except Exception:
	DeepSort = None

try:
	from torchreid.utils import FeatureExtractor as TorchreidExtractor
except Exception:
	TorchreidExtractor = None

try:
	from insightface.app import FaceAnalysis
	_INSIGHT_OK = True
except Exception:
	FaceAnalysis = None
	_INSIGHT_OK = False

try:
	import onnxruntime as ort  # noqa: F401
except Exception:
	ort = None

# SQLAlchemy / pgvector
from sqlalchemy import (
	ARRAY,
	BigInteger,
	Boolean,
	Column,
	DateTime,
	Float,
	ForeignKey,
	Integer,
	LargeBinary,
	String,
	create_engine,
	func,
	select,
	text,
)
from sqlalchemy.orm import declarative_base, sessionmaker

try:
	from pgvector.sqlalchemy import Vector
except Exception:
	Vector = None


EXPECTED_DIM = 512

_yolo_lock = threading.Lock()
_reid_lock = threading.Lock()
_face_lock = threading.Lock()


# ---------------------------
# Helpers
# ---------------------------
def l2_normalize(v: np.ndarray) -> np.ndarray:
	v = np.asarray(v, dtype=np.float32).reshape(-1)
	n = float(np.linalg.norm(v))
	if n <= 0 or not np.isfinite(n):
		return v
	return v / n


def l2_normalize_rows(m: np.ndarray) -> np.ndarray:
	m = np.asarray(m, dtype=np.float32)
	if m.ndim != 2:
		return m
	n = np.linalg.norm(m, axis=1, keepdims=True)
	n = np.where((n <= 0) | (~np.isfinite(n)), 1.0, n)
	return m / n


def safe_iter(x):
	if x is None:
		return []
	try:
		return list(x)
	except TypeError:
		return [x]


def extract_face_embedding(face):
	emb = getattr(face, "normed_embedding", None)
	if emb is None:
		emb = getattr(face, "embedding", None)
	return emb


def _to_rgb(bgr: np.ndarray) -> np.ndarray:
	return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _looks_like_url(s: str) -> bool:
	s = str(s or "").strip().lower()
	return "://" in s or s.startswith("rtsp:") or s.startswith("http:") or s.startswith("https:")


def _is_int_str(s: str) -> bool:
	s = str(s or "").strip()
	if not s:
		return False
	if s.startswith("-"):
		return s[1:].isdigit()
	return s.isdigit()


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


def iou_xyxy(a, b) -> float:
	ax1, ay1, ax2, ay2 = a
	bx1, by1, bx2, by2 = b
	ix1, iy1 = max(ax1, bx1), max(ay1, by1)
	ix2, iy2 = min(ax2, bx2), min(ay2, by2)
	iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
	inter = iw * ih
	aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
	ba = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
	denom = aa + ba - inter
	return float(inter / denom) if denom > 0 else 0.0


def ioa_xyxy(inner, outer) -> float:
	ix1, iy1, ix2, iy2 = inner
	ox1, oy1, ox2, oy2 = outer
	x1, y1 = max(ix1, ox1), max(iy1, oy1)
	x2, y2 = min(ix2, ox2), min(iy2, oy2)
	iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
	inter = iw * ih
	ia = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
	return float(inter / ia) if ia > 0 else 0.0


def _point_in_xyxy(px: float, py: float, box) -> bool:
	x1, y1, x2, y2 = box
	return (px >= x1) and (px <= x2) and (py >= y1) and (py <= y2)


# ---------------------------
# DB gallery
# ---------------------------
def decode_bank_blob(raw: bytes | None) -> np.ndarray | None:
	if not raw:
		return None
	for mode in ("gzip", "plain"):
		try:
			data = gzip.decompress(raw) if mode == "gzip" else raw
			arr = np.load(BytesIO(data), allow_pickle=False)
			arr = np.asarray(arr, dtype=np.float32)
			if arr.ndim == 2 and arr.shape[1] == EXPECTED_DIM and np.isfinite(arr).all():
				return l2_normalize_rows(arr)
		except Exception:
			pass
	return None


def _as_vec512(x) -> np.ndarray | None:
	if x is None:
		return None
	try:
		a = np.asarray(x, dtype=np.float32).reshape(-1)
		if a.size != EXPECTED_DIM or not np.isfinite(a).all():
			return None
		return l2_normalize(a)
	except Exception:
		return None


@dataclass
class PersonEntry:
	member_id: int
	name: str
	body_bank: Optional[np.ndarray]
	body_centroid: Optional[np.ndarray]
	face_centroid: Optional[np.ndarray]


@dataclass
class FaceGallery:
	names: List[str]
	mat: np.ndarray  # [N,512] L2-normalized

	def is_empty(self) -> bool:
		return (not self.names) or (self.mat is None) or (self.mat.size == 0)


@dataclass
class _Agg:
	body_vecs: List[np.ndarray]
	face_vecs: List[np.ndarray]
	body_banks: List[np.ndarray]
	face_banks: List[np.ndarray]


class GalleryData:
	def __init__(self):
		self.people_global: List[PersonEntry] = []
		self.face_global: FaceGallery = FaceGallery([], np.zeros((0, EXPECTED_DIM), np.float32))
		self.people_by_cam: Dict[int, List[PersonEntry]] = {}
		self.face_by_cam: Dict[int, FaceGallery] = {}


def _member_label(member_number: str | None, first_name: str | None, last_name: str | None, member_id: int) -> str:
	fn = (first_name or "").strip()
	ln = (last_name or "").strip()
	mn = (member_number or "").strip()
	if fn and ln:
		return f"{fn} {ln}".strip()
	if fn:
		return fn
	if mn:
		return mn
	return f"member_{int(member_id)}"


def build_galleries_from_db(db_url: str, active_only: bool = True, max_bank_per_member: int = 0) -> GalleryData:
	Base = declarative_base()

	class MemberRow(Base):
		__tablename__ = "members"
		id = Column(Integer, primary_key=True)
		member_number = Column(String(16))
		first_name = Column(String(64))
		last_name = Column(String(64))
		is_active = Column(Boolean)

	class MemberEmbeddingRow(Base):
		__tablename__ = "member_embeddings"
		id = Column(BigInteger, primary_key=True)

		member_id = Column(Integer, ForeignKey("members.id"))
		camera_id = Column(Integer, ForeignKey("cameras.id"))

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

		last_embedding_update_ts = Column(DateTime(timezone=True), nullable=True, server_default=func.now())

	engine = create_engine(db_url, pool_pre_ping=True)
	Session = sessionmaker(bind=engine)

	member_meta: Dict[int, Dict[str, Any]] = {}
	agg_global: Dict[int, _Agg] = defaultdict(lambda: _Agg([], [], [], []))
	agg_by_cam: Dict[int, Dict[int, _Agg]] = defaultdict(lambda: defaultdict(lambda: _Agg([], [], [], [])))

	with Session() as session:
		stmt = (
			select(
				MemberRow.id,
				MemberRow.member_number,
				MemberRow.first_name,
				MemberRow.last_name,
				MemberRow.is_active,
				MemberEmbeddingRow.camera_id,
				MemberEmbeddingRow.body_embedding,
				MemberEmbeddingRow.face_embedding,
				MemberEmbeddingRow.back_body_embedding,
				MemberEmbeddingRow.body_embeddings_raw,
				MemberEmbeddingRow.face_embeddings_raw,
				MemberEmbeddingRow.back_body_embeddings_raw,
			)
			.select_from(MemberRow)
			.join(MemberEmbeddingRow, MemberEmbeddingRow.member_id == MemberRow.id, isouter=False)
		)

		rows = session.execute(stmt).all()

		raw_loaded_body: Dict[int, int] = defaultdict(int)
		raw_loaded_face: Dict[int, int] = defaultdict(int)
		raw_loaded_body_cam: Dict[Tuple[int, int], int] = defaultdict(int)
		raw_loaded_face_cam: Dict[Tuple[int, int], int] = defaultdict(int)

		for r in rows:
			mid = int(r.id)
			is_active_m = bool(r.is_active) if r.is_active is not None else True
			if active_only and (not is_active_m):
				continue

			cam_id = int(r.camera_id)

			if mid not in member_meta:
				member_meta[mid] = {
					"member_number": r.member_number,
					"first_name": r.first_name,
					"last_name": r.last_name,
					"is_active": is_active_m,
				}

			b = _as_vec512(r.body_embedding)
			f = _as_vec512(r.face_embedding)
			bb = _as_vec512(r.back_body_embedding)

			if b is not None:
				agg_global[mid].body_vecs.append(b)
				agg_by_cam[cam_id][mid].body_vecs.append(b)
			if bb is not None:
				agg_global[mid].body_vecs.append(bb)
				agg_by_cam[cam_id][mid].body_vecs.append(bb)
			if f is not None:
				agg_global[mid].face_vecs.append(f)
				agg_by_cam[cam_id][mid].face_vecs.append(f)

			max_bank = int(max_bank_per_member or 0)

			# body bank
			if r.body_embeddings_raw and (max_bank <= 0 or raw_loaded_body[mid] < max_bank):
				bank = decode_bank_blob(r.body_embeddings_raw)
				if bank is not None and bank.size > 0:
					remain = max_bank - raw_loaded_body[mid] if max_bank > 0 else bank.shape[0]
					take = bank[: max(0, remain)] if max_bank > 0 else bank
					if take.size > 0:
						agg_global[mid].body_banks.append(take)
						raw_loaded_body[mid] += int(take.shape[0])

					key = (mid, cam_id)
					if max_bank <= 0 or raw_loaded_body_cam[key] < max_bank:
						remain2 = max_bank - raw_loaded_body_cam[key] if max_bank > 0 else bank.shape[0]
						take2 = bank[: max(0, remain2)] if max_bank > 0 else bank
						if take2.size > 0:
							agg_by_cam[cam_id][mid].body_banks.append(take2)
							raw_loaded_body_cam[key] += int(take2.shape[0])

			# face bank
			if r.face_embeddings_raw and (max_bank <= 0 or raw_loaded_face[mid] < max_bank):
				bank = decode_bank_blob(r.face_embeddings_raw)
				if bank is not None and bank.size > 0:
					remain = max_bank - raw_loaded_face[mid] if max_bank > 0 else bank.shape[0]
					take = bank[: max(0, remain)] if max_bank > 0 else bank
					if take.size > 0:
						agg_global[mid].face_banks.append(take)
						raw_loaded_face[mid] += int(take.shape[0])

					key = (mid, cam_id)
					if max_bank <= 0 or raw_loaded_face_cam[key] < max_bank:
						remain2 = max_bank - raw_loaded_face_cam[key] if max_bank > 0 else bank.shape[0]
						take2 = bank[: max(0, remain2)] if max_bank > 0 else bank
						if take2.size > 0:
							agg_by_cam[cam_id][mid].face_banks.append(take2)
							raw_loaded_face_cam[key] += int(take2.shape[0])

			# back-body bank
			if r.back_body_embeddings_raw and (max_bank <= 0 or raw_loaded_body[mid] < max_bank):
				bank = decode_bank_blob(r.back_body_embeddings_raw)
				if bank is not None and bank.size > 0:
					remain = max_bank - raw_loaded_body[mid] if max_bank > 0 else bank.shape[0]
					take = bank[: max(0, remain)] if max_bank > 0 else bank
					if take.size > 0:
						agg_global[mid].body_banks.append(take)
						raw_loaded_body[mid] += int(take.shape[0])

					key = (mid, cam_id)
					if max_bank <= 0 or raw_loaded_body_cam[key] < max_bank:
						remain2 = max_bank - raw_loaded_body_cam[key] if max_bank > 0 else bank.shape[0]
						take2 = bank[: max(0, remain2)] if max_bank > 0 else bank
						if take2.size > 0:
							agg_by_cam[cam_id][mid].body_banks.append(take2)
							raw_loaded_body_cam[key] += int(take2.shape[0])

	def _build(agg: Dict[int, _Agg]) -> Tuple[List[PersonEntry], FaceGallery]:
		ppl: List[PersonEntry] = []
		fnames: List[str] = []
		fvecs: List[np.ndarray] = []

		for mid, a in agg.items():
			meta = member_meta.get(mid, {})
			name = _member_label(meta.get("member_number"), meta.get("first_name"), meta.get("last_name"), mid)

			body_bank = None
			if a.body_banks:
				body_bank = l2_normalize_rows(np.concatenate([np.asarray(x, np.float32) for x in a.body_banks if x is not None], axis=0))
			body_cent = None
			if a.body_vecs:
				body_cent = l2_normalize(np.mean(np.stack(a.body_vecs, axis=0), axis=0))
			elif body_bank is not None and body_bank.size > 0:
				body_cent = l2_normalize(np.mean(body_bank, axis=0))
			if body_bank is None and body_cent is not None:
				body_bank = body_cent.reshape(1, -1).astype(np.float32)

			face_cent = None
			if a.face_vecs:
				face_cent = l2_normalize(np.mean(np.stack(a.face_vecs, axis=0), axis=0))
			elif a.face_banks:
				fb = l2_normalize_rows(np.concatenate([np.asarray(x, np.float32) for x in a.face_banks if x is not None], axis=0))
				if fb.size > 0:
					face_cent = l2_normalize(np.mean(fb, axis=0))

			ppl.append(PersonEntry(member_id=int(mid), name=str(name), body_bank=body_bank, body_centroid=body_cent, face_centroid=face_cent))
			if face_cent is not None:
				fnames.append(str(name))
				fvecs.append(face_cent.astype(np.float32))

		fmat = l2_normalize_rows(np.stack(fvecs, axis=0)) if fvecs else np.zeros((0, EXPECTED_DIM), np.float32)
		return ppl, FaceGallery(fnames, fmat)

	gd = GalleryData()
	gd.people_global, gd.face_global = _build(agg_global)
	for cam_id, mm in agg_by_cam.items():
		ppl, fg = _build(mm)
		gd.people_by_cam[int(cam_id)] = ppl
		gd.face_by_cam[int(cam_id)] = fg

	return gd


class GalleryManager:
	def __init__(self, args: argparse.Namespace):
		self._lock = threading.Lock()
		self._reload_lock = threading.Lock()
		self.data = GalleryData()
		self.last_load_ts = 0.0
		self.load(args)

	def load(self, args: argparse.Namespace) -> None:
		active_only = not bool(getattr(args, "db_include_inactive", False))
		max_bank = int(getattr(args, "db_max_bank", 0) or 0)
		gd = build_galleries_from_db(args.db_url, active_only=active_only, max_bank_per_member=max_bank)
		with self._lock:
			self.data = gd
			self.last_load_ts = time.time()

	def maybe_reload(self, args: argparse.Namespace) -> None:
		period = float(getattr(args, "db_refresh_seconds", 0.0) or 0.0)
		if period <= 0:
			return
		if (time.time() - self.last_load_ts) < period:
			return
		if not self._reload_lock.acquire(blocking=False):
			return
		try:
			if (time.time() - self.last_load_ts) < period:
				return
			self.load(args)
			print("[DB] Gallery reloaded")
		except Exception as e:
			print("[DB] Gallery reload failed:", e)
		finally:
			self._reload_lock.release()

	def snapshot(self, camera_id: Optional[int]) -> Tuple[List[PersonEntry], FaceGallery]:
		with self._lock:
			if camera_id is None:
				return self.data.people_global, self.data.face_global
			cid = int(camera_id)
			ppl = self.data.people_by_cam.get(cid)
			fg = self.data.face_by_cam.get(cid)
			if ppl is not None and fg is not None and (not fg.is_empty()):
				return ppl, fg
			return self.data.people_global, self.data.face_global


# ---------------------------
# Stream readers
# ---------------------------
class AdaptiveQueueStream:
	def __init__(self, src: str, queue_size: int, rtsp_transport: str):
		self.src = src
		src_use = src
		if isinstance(src, str) and src.lower().startswith("rtsp"):
			sep = "&" if "?" in src_use else "?"
			src_use = f"{src_use}{sep}rtsp_transport={rtsp_transport}"
		self.cap = cv2.VideoCapture(src_use, cv2.CAP_FFMPEG)
		self.ok = self.cap.isOpened()
		try:
			self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(queue_size)))
		except Exception:
			pass
		self.q: queue.Queue = queue.Queue(maxsize=max(1, int(queue_size)))
		self.stop_flag = threading.Event()
		self.read_dropped = 0
		self.thread = threading.Thread(target=self._loop, daemon=True)
		self.thread.start()

	def _loop(self):
		while not self.stop_flag.is_set() and self.ok:
			ok, frame = self.cap.read()
			if not ok or frame is None:
				time.sleep(0.01)
				continue
			item = (frame, time.time())
			try:
				self.q.put_nowait(item)
			except queue.Full:
				try:
					_ = self.q.get_nowait()
				except Exception:
					pass
				try:
					self.q.put_nowait(item)
				except Exception:
					pass

	def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
		try:
			frame, ts = self.q.get(timeout=0.1)
			return True, frame, float(ts)
		except queue.Empty:
			return False, None, 0.0

	def is_opened(self) -> bool:
		return bool(self.ok)

	def release(self):
		self.stop_flag.set()
		try:
			self.thread.join(timeout=1.0)
		except Exception:
			pass
		try:
			self.cap.release()
		except Exception:
			pass


class LatestStream:
	def __init__(self, src: str, rtsp_buffer: int, decode_skip: int, rtsp_transport: str):
		self.src = src
		src_use = src
		if isinstance(src, str) and src.lower().startswith("rtsp"):
			sep = "&" if "?" in src_use else "?"
			src_use = f"{src_use}{sep}rtsp_transport={rtsp_transport}"
		self.cap = cv2.VideoCapture(src_use, cv2.CAP_FFMPEG)
		self.ok = self.cap.isOpened()
		try:
			self.cap.set(cv2.CAP_PROP_BUFFERSIZE, max(1, int(rtsp_buffer)))
		except Exception:
			pass
		self.decode_skip = max(0, int(decode_skip))
		self.q = queue.Queue(maxsize=1)
		self.stop_flag = threading.Event()
		self.thread = threading.Thread(target=self._loop, daemon=True)
		self.thread.start()

	def _loop(self):
		idx = 0
		while not self.stop_flag.is_set() and self.ok:
			ok, frame = self.cap.read()
			if not ok or frame is None:
				time.sleep(0.01)
				continue
			if self.decode_skip > 0 and (idx % (self.decode_skip + 1)) != 0:
				idx += 1
				continue
			idx += 1
			item = (frame, time.time())
			if not self.q.empty():
				try:
					_ = self.q.get_nowait()
				except Exception:
					pass
			try:
				self.q.put_nowait(item)
			except Exception:
				pass

	def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
		try:
			frame, ts = self.q.get(timeout=0.1)
			return True, frame, float(ts)
		except queue.Empty:
			return False, None, 0.0

	def is_opened(self) -> bool:
		return bool(self.ok)

	def release(self):
		self.stop_flag.set()
		try:
			self.thread.join(timeout=1.0)
		except Exception:
			pass
		try:
			self.cap.release()
		except Exception:
			pass


class FFmpegStream:
	def __init__(self, src: str, queue_size: int, rtsp_transport: str, use_cuda: bool, force_w: int = 0, force_h: int = 0):
		self.src = _sanitize_rtsp_url(src) if isinstance(src, str) and src.lower().startswith("rtsp") else src
		self.q = queue.Queue(maxsize=max(1, int(queue_size)))
		self.stop_flag = threading.Event()
		self.ok = False

		# probe size
		probed_w = probed_h = 0
		try:
			cap = cv2.VideoCapture(self.src, cv2.CAP_FFMPEG)
			if cap.isOpened():
				probed_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 0
				probed_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 0
			cap.release()
		except Exception:
			pass

		out_w = int(force_w) if force_w and force_w > 0 else int(probed_w or 1280)
		out_h = int(force_h) if force_h and force_h > 0 else int(probed_h or 720)
		self.width, self.height = out_w, out_h
		self.frame_bytes = self.width * self.height * 3

		scheme = ""
		try:
			scheme = urlsplit(self.src).scheme.lower()
		except Exception:
			pass

		cmd = ["ffmpeg", "-hide_banner", "-loglevel", "warning"]
		if scheme == "rtsp":
			cmd += ["-rtsp_transport", rtsp_transport]
		cmd += ["-i", self.src, "-an", "-dn", "-sn", "-vsync", "0", "-fflags", "nobuffer", "-flags", "low_delay"]

		vf = f"scale={out_w}:{out_h}"
		if use_cuda:
			vf = f"hwdownload,format=bgr24,scale={out_w}:{out_h}"
			cmd = cmd[:]
			cmd += ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]

		cmd += ["-vf", vf, "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1"]

		self.proc = None
		try:
			self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=10**8)
			self.ok = True
		except Exception as e:
			print("[FFMPEG] start failed:", e)
			self.ok = False

		self.thread = threading.Thread(target=self._loop, daemon=True)
		self.thread.start()

	def _loop(self):
		if not self.proc or not self.proc.stdout or not self.ok:
			return
		fb = int(self.frame_bytes)
		w, h = int(self.width), int(self.height)
		while not self.stop_flag.is_set():
			buf = self.proc.stdout.read(fb)
			if not buf or len(buf) < fb:
				time.sleep(0.005)
				continue
			frame = np.frombuffer(buf, dtype=np.uint8).reshape((h, w, 3))
			item = (frame, time.time())
			try:
				self.q.put_nowait(item)
			except queue.Full:
				try:
					_ = self.q.get_nowait()
				except Exception:
					pass
				try:
					self.q.put_nowait(item)
				except Exception:
					pass

	def read(self) -> Tuple[bool, Optional[np.ndarray], float]:
		try:
			frame, ts = self.q.get(timeout=0.1)
			return True, frame, float(ts)
		except queue.Empty:
			return False, None, 0.0

	def is_opened(self) -> bool:
		return bool(self.ok)

	def release(self):
		self.stop_flag.set()
		try:
			self.thread.join(timeout=1.0)
		except Exception:
			pass
		try:
			if self.proc:
				self.proc.terminate()
				self.proc.kill()
		except Exception:
			pass


# ---------------------------
# MJPEG frame buffer
# ---------------------------
class FrameBuffer:
	def __init__(self):
		self._lock = threading.Lock()
		self._cond = threading.Condition(self._lock)
		self._frame_bgr: Optional[np.ndarray] = None
		self._ts: float = 0.0
		self._meta: Dict[str, Any] = {}
		self._seq = 0

		self._jpeg: Optional[bytes] = None
		self._jpeg_ts: float = 0.0
		self._encode_lock = threading.Lock()
		self._clients = 0

	def add_client(self):
		with self._lock:
			self._clients += 1

	def remove_client(self):
		with self._lock:
			self._clients = max(0, self._clients - 1)

	def set(self, frame_bgr: np.ndarray, meta: Optional[Dict[str, Any]] = None):
		with self._cond:
			self._frame_bgr = frame_bgr
			self._ts = time.time()
			self._meta = dict(meta or {})
			self._seq += 1
			self._jpeg = None
			self._jpeg_ts = 0.0
			self._cond.notify_all()

	def get(self) -> Optional[np.ndarray]:
		with self._lock:
			return self._frame_bgr

	def wait_jpeg(self, last_ts: float, timeout: float, jpeg_quality: int = 80) -> Tuple[Optional[bytes], float]:
		with self._cond:
			if self._ts <= float(last_ts):
				self._cond.wait(timeout=float(timeout))
			ts = float(self._ts)
			frame = self._frame_bgr
			clients = int(self._clients)

		if frame is None or ts <= 0:
			return None, 0.0
		if clients <= 0:
			return None, ts

		with self._lock:
			if self._jpeg is not None and self._jpeg_ts == ts:
				return self._jpeg, ts

		with self._encode_lock:
			with self._lock:
				if self._jpeg is not None and self._jpeg_ts == ts:
					return self._jpeg, ts
				frame_ref = self._frame_bgr
				ts_copy = float(self._ts)

			if frame_ref is None or ts_copy <= 0:
				return None, 0.0

			q = int(max(30, min(95, int(jpeg_quality))))
			ok, enc = cv2.imencode(".jpg", frame_ref, [int(cv2.IMWRITE_JPEG_QUALITY), q])
			if not ok:
				return None, ts_copy

			jpg = enc.tobytes()

			with self._lock:
				if float(self._ts) == ts_copy:
					self._jpeg = jpg
					self._jpeg_ts = ts_copy

			return jpg, ts_copy


RenderedFrame = FrameBuffer  # external import alias


# ---------------------------
# Summary report (CSV)
# ---------------------------
@dataclass
class _ActiveSession:
	start_ts: float
	last_seen_ts: float


class SummaryReport:
	def __init__(self, num_cams: int, gap_seconds: float = 2.0, time_format: str = "%H:%M:%S"):
		self.num_cams = max(1, int(num_cams))
		self.gap_seconds = float(max(0.0, gap_seconds))
		self.time_format = str(time_format or "%H:%M:%S")
		self._lock = threading.Lock()
		self._disabled = False

		self._first_seen: Dict[str, float] = {}
		self._active: Dict[Tuple[str, int], _ActiveSession] = {}
		self._logs: Dict[str, Dict[int, List[Tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
		self._total_seconds: Dict[str, float] = defaultdict(float)

	def stop(self):
		with self._lock:
			self._disabled = True

	def _close_expired_locked(self, now_ts: float):
		if self.gap_seconds <= 0:
			return
		to_close = []
		for (name, cam_id), sess in list(self._active.items()):
			if (now_ts - sess.last_seen_ts) > self.gap_seconds:
				to_close.append((name, cam_id, sess))
		for name, cam_id, sess in to_close:
			self._close_session_locked(name, cam_id, sess)

	def _close_session_locked(self, name: str, cam_id: int, sess: _ActiveSession):
		st = float(sess.start_ts)
		en = float(sess.last_seen_ts)
		if en < st:
			en = st
		self._logs[name][int(cam_id)].append((st, en))
		self._total_seconds[name] += max(0.0, (en - st))
		self._active.pop((name, int(cam_id)), None)

	def close_all(self):
		with self._lock:
			for (name, cam_id), sess in list(self._active.items()):
				self._close_session_locked(name, cam_id, sess)

	def update(self, cam_id: int, present_names: List[str], ts: float):
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
				if name not in self._first_seen:
					self._first_seen[name] = float(ts)

				key = (name, cam_id)
				sess = self._active.get(key)
				if sess is None:
					self._active[key] = _ActiveSession(start_ts=float(ts), last_seen_ts=float(ts))
				else:
					sess.last_seen_ts = float(ts)

	def _fmt_time(self, ts: float) -> str:
		try:
			return datetime.fromtimestamp(float(ts)).strftime(self.time_format)
		except Exception:
			return ""

	def _fmt_total(self, total_seconds: float) -> str:
		sec_i = int(round(max(0.0, float(total_seconds))))
		if sec_i == 0 and total_seconds > 0.0:
			sec_i = 1
		minutes = sec_i // 60
		seconds = sec_i % 60
		return f"{minutes} {'minute' if minutes == 1 else 'minutes'} {seconds} {'second' if seconds == 1 else 'seconds'}"

	def write_csv(self, path: str):
		self.close_all()
		with self._lock:
			items = list(self._first_seen.items())
			logs = {k: {ck: list(vv) for ck, vv in cv.items()} for k, cv in self._logs.items()}
			totals = dict(self._total_seconds)

		items.sort(key=lambda kv: kv[1])
		names_order = [nm for nm, _ in items]
		os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

		header = ["Member"] + [f"c{i+1}" for i in range(self.num_cams)] + ["Total time"]
		with open(path, "w", newline="", encoding="utf-8") as f:
			w = csv.writer(f)
			w.writerow(header)
			for name in names_order:
				row = [name]
				cam_map = logs.get(name, {})
				for cam_id in range(self.num_cams):
					entries = cam_map.get(cam_id, [])
					lines = [f"L{idx} - {self._fmt_time(st)} to {self._fmt_time(en)}" for idx, (st, en) in enumerate(entries, start=1)]
					row.append("\n".join(lines))
				row.append(self._fmt_total(totals.get(name, 0.0)))
				w.writerow(row)

	def write_csv_snapshot(self, path: str):
		now_ts = time.time()
		with self._lock:
			if self._disabled:
				return
			self._close_expired_locked(now_ts=now_ts)
			items = list(self._first_seen.items())
			logs = {k: {ck: list(vv) for ck, vv in cv.items()} for k, cv in self._logs.items()}
			totals = dict(self._total_seconds)

			for (name, cam_id), sess in self._active.items():
				st = float(sess.start_ts)
				en = float(sess.last_seen_ts)
				logs.setdefault(name, {}).setdefault(int(cam_id), []).append((st, en))
				totals[name] = totals.get(name, 0.0) + max(0.0, (en - st))

		items.sort(key=lambda kv: kv[1])
		names_order = [nm for nm, _ in items]
		os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

		header = ["Member"] + [f"c{i+1}" for i in range(self.num_cams)] + ["Total time"]
		with open(path, "w", newline="", encoding="utf-8") as f:
			w = csv.writer(f)
			w.writerow(header)
			for name in names_order:
				row = [name]
				cam_map = logs.get(name, {})
				for cam_id in range(self.num_cams):
					entries = cam_map.get(cam_id, [])
					lines = [f"L{idx} - {self._fmt_time(st)} to {self._fmt_time(en)}" for idx, (st, en) in enumerate(entries, start=1)]
					row.append("\n".join(lines))
				row.append(self._fmt_total(totals.get(name, 0.0)))
				w.writerow(row)


# ---------------------------
# Raw ingestion manager
# ---------------------------
@dataclass
class DetectionRow:
	data_ts: datetime
	camera_id: int
	member_id: Optional[int]
	guest_temp_id: Optional[str]
	guest_data_vector: Optional[dict]
	match_value: int


def _align_next_rotation(now_utc: datetime, rotate_seconds: int) -> datetime:
	"""Return the next rotation boundary in UTC as a **naive UTC datetime**.

	IMPORTANT:
	- datetime.utcnow() returns a naive datetime representing UTC.
	- Using .timestamp() on a naive datetime interprets it as *local time* on many systems,
	  which can make the computed rotation time be in the past -> rotate repeatedly.
	- We must compute epoch seconds treating the input as UTC, regardless of local timezone.
	"""
	rotate_seconds = int(max(1, rotate_seconds))

	# Treat naive datetimes as UTC deterministically (no local-time assumptions)
	if getattr(now_utc, "tzinfo", None) is None:
		sec = int(calendar.timegm(now_utc.timetuple()))
	else:
		# If aware, convert to UTC then take epoch
		sec = int(now_utc.timestamp())

	nxt = ((sec // rotate_seconds) + 1) * rotate_seconds
	return datetime.utcfromtimestamp(nxt)


class RawIngestionManager:
	def __init__(
		self,
		db_url: str,
		# dynamic table creation
		rotate_seconds: int = 300,
		flush_interval: float = 1.0,
		batch_size: int = 500,
		retention_minutes: int = 0,
		enable_cleanup: bool = False,
	):
	
		self.engine = create_engine(db_url, pool_pre_ping=True)

		self.rotate_seconds = int(max(60, rotate_seconds))
		self.flush_interval = float(max(0.1, flush_interval))
		self.batch_size = int(max(1, batch_size))

		self.retention_minutes = int(max(0, retention_minutes))
		self.enable_cleanup = bool(enable_cleanup and self.retention_minutes > 0)

		self._q: "queue.Queue[DetectionRow]" = queue.Queue(maxsize=200000)
		self._stop = threading.Event()
		self._thread = threading.Thread(target=self._loop, daemon=True)

		self._table_lock = threading.Lock()
		self._cur_table: Optional[str] = None
		self._next_rotate_at: datetime = _align_next_rotation(datetime.utcnow(), self.rotate_seconds)
		self._last_cleanup_at: float = 0.0

		with self.engine.begin() as conn:
			conn.execute(text("SELECT 1"))

		self._rotate_table(force=True)
		self._thread.start()

	def stop(self):
		self._stop.set()
		try:
			self._thread.join(timeout=3.0)
		except Exception:
			pass
		try:
			self._flush(max_rows=1_000_000)
		except Exception as e:
			print("[INGEST] final flush failed:", e)

	def push_many(self, rows: List[DetectionRow]):
		for r in rows or []:
			try:
				self._q.put_nowait(r)
			except queue.Full:
				try:
					_ = self._q.get_nowait()
				except Exception:
					pass
				try:
					self._q.put_nowait(r)
				except Exception:
					pass

	def _make_table_name(self) -> str:
		return f"raw_data_{uuid.uuid4().hex}"

	def _rotate_table(self, force: bool = False):
		now_utc = datetime.utcnow()
		if (not force) and (now_utc < self._next_rotate_at):
			return

		new_table = self._make_table_name()
		ddl = f"""
		CREATE TABLE IF NOT EXISTS {new_table} (
		  data_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
		  camera_id INTEGER NOT NULL CHECK (camera_id >= 0),
		  member_id INTEGER CHECK (member_id >= 0),
		  guest_temp_id VARCHAR(64),
		  guest_data_vector JSONB,
		  match_value INTEGER CHECK (match_value >= 0)
		);
		"""
		try:
			with self.engine.begin() as conn:
				conn.execute(text(ddl)) 
				conn.execute(
					text(
						"""
						CREATE TABLE IF NOT EXISTS raw_data_registry (
							table_name TEXT PRIMARY KEY,
							created_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
						);
						"""
					)
				)
				conn.execute(
					text(
						"INSERT INTO raw_data_registry (table_name) VALUES (:tname) "
						"ON CONFLICT (table_name) DO NOTHING"
					),
					{"tname": new_table},
				)
		except Exception as e:
			print("[INGEST] table create failed:", e)
			return

		with self._table_lock:
			self._cur_table = new_table
			self._next_rotate_at = _align_next_rotation(now_utc, self.rotate_seconds)

		print(f"[INGEST] Rotated raw table -> {new_table} (next at {self._next_rotate_at.isoformat()}Z)")

	def _flush(self, max_rows: int) -> int:
		rows: List[DetectionRow] = []
		for _ in range(max_rows):
			try:
				rows.append(self._q.get_nowait())
			except queue.Empty:
				break
		if not rows:
			return 0

		self._rotate_table(force=False)
		with self._table_lock:
			table = self._cur_table
		if not table:
			return 0

		ins = text(
			f"""
			INSERT INTO {table} (data_ts, camera_id, member_id, guest_temp_id, guest_data_vector, match_value)
			VALUES (:data_ts, :camera_id, :member_id, :guest_temp_id, CAST(:guest_data_vector AS JSONB), :match_value)
			"""
		)

		def _default(o):
			if isinstance(o, (np.floating,)):
				return float(o)
			if isinstance(o, (np.integer,)):
				return int(o)
			if isinstance(o, (np.ndarray,)):
				return o.astype(float).tolist()
			return str(o)

		payload = []
		for r in rows:
			gv = None
			if r.guest_data_vector is not None:
				raw = r.guest_data_vector
				if isinstance(raw, dict):
					# safety: do not store bbox or confidences
					drop = {"bbox", "det_conf", "conf", "confidence", "face_ran", "last_face_sim"}
					raw = {k: v for k, v in raw.items() if k not in drop}
				if isinstance(raw, str):
					try:
						json.loads(raw)
						gv = raw
					except Exception:
						gv = json.dumps(raw, default=_default)
				else:
					gv = json.dumps(raw, default=_default)

			payload.append(
				{
					"data_ts": r.data_ts,
					"camera_id": int(r.camera_id),
					"member_id": int(r.member_id) if r.member_id is not None else None,
					"guest_temp_id": str(r.guest_temp_id) if r.guest_temp_id else None,
					"guest_data_vector": gv,
					"match_value": int(max(0, r.match_value)),
				}
			)

		try:
			with self.engine.begin() as conn:
				conn.execute(ins, payload)
			return len(rows)
		except Exception as e:
			print("[INGEST] insert failed:", e)
			return 0

	def _cleanup_old_tables(self):
		if not self.enable_cleanup:
			return

		now = time.time()
		if (now - self._last_cleanup_at) < 60.0:
			return
		self._last_cleanup_at = now

		cutoff = datetime.utcnow() - timedelta(minutes=self.retention_minutes)

		try:
			with self.engine.begin() as conn:
				# Always ensure registry exists
				conn.execute(text("""
					CREATE TABLE IF NOT EXISTS raw_data_registry (
					table_name TEXT PRIMARY KEY,
					created_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
					normalized BOOLEAN DEFAULT FALSE
					);
				"""))

				rows = conn.execute(
					text("""
						SELECT table_name
						FROM raw_data_registry
						WHERE created_ts < :cutoff
						ORDER BY created_ts ASC
					"""),
					{"cutoff": cutoff},
				).fetchall()

				with self._table_lock:
					cur = self._cur_table

				for (tname,) in rows:
					# Never drop the active table
					if cur and tname == cur:
						continue

					# Safety: allow only expected table pattern
					if not tname.startswith("raw_data_"):
						print(f"[INGEST] skipped suspicious table name: {tname}")
						continue

					try:
						conn.execute(text(f'DROP TABLE IF EXISTS "{tname}"'))
						conn.execute(
							text("DELETE FROM raw_data_registry WHERE table_name=:t"),
							{"t": tname},
						)
						print(f"[INGEST] Dropped old raw table: {tname}")
					except Exception as e:
						print(f"[INGEST] failed dropping {tname}: {e}")

		except Exception as e:
			print(f"[INGEST] cleanup failed: {e}")


	def _loop(self):
		last_flush = 0.0
		while not self._stop.is_set():
			self._rotate_table(force=False)
			try:
				qsz = int(self._q.qsize())
			except Exception:
				qsz = 0

			now = time.time()
			if qsz >= self.batch_size or (now - last_flush) >= self.flush_interval:
				n = self._flush(max_rows=max(self.batch_size, 1))
				if n > 0:
					last_flush = now

			self._cleanup_old_tables()
			time.sleep(0.02)


# ---------------------------
# Embedding / matching
# ---------------------------
def best_face_label_top2(emb: np.ndarray | None, face_gallery: FaceGallery) -> Tuple[Optional[str], float, float]:
	if emb is None or face_gallery is None or face_gallery.is_empty():
		return None, 0.0, 0.0
	q = l2_normalize(np.asarray(emb, np.float32).reshape(-1))
	if q.size != EXPECTED_DIM or not np.isfinite(q).all():
		return None, 0.0, 0.0
	sims = face_gallery.mat @ q
	if sims.size == 0:
		return None, 0.0, 0.0
	if sims.size == 1:
		return face_gallery.names[0], float(sims[0]), 0.0
	idxs = np.argpartition(sims, -2)[-2:]
	i1, i2 = int(idxs[0]), int(idxs[1])
	if sims[i2] > sims[i1]:
		i1, i2 = i2, i1
	return face_gallery.names[i1], float(sims[i1]), float(sims[i2])


def best_face_sim_to_gallery(emb: np.ndarray | None, face_gallery: FaceGallery) -> float:
	if emb is None or face_gallery is None or face_gallery.is_empty():
		return 0.0
	q = l2_normalize(np.asarray(emb, np.float32).reshape(-1))
	if q.size != EXPECTED_DIM or not np.isfinite(q).all():
		return 0.0
	sims = face_gallery.mat @ q
	return float(np.max(sims)) if sims.size else 0.0


def extract_body_embeddings_batch(extractor, crops_bgr: List[np.ndarray], device_is_cuda: bool, use_half: bool) -> Optional[np.ndarray]:
	if extractor is None or not crops_bgr:
		return None
	crops_rgb = []
	for c in crops_bgr:
		if c is None or c.size == 0:
			crops_rgb.append(np.zeros((1, 1, 3), np.uint8))
		else:
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

	if hasattr(feats, "detach"):
		feats = feats.detach().cpu().numpy()
	else:
		feats = np.asarray(feats)

	feats = np.asarray(feats, dtype=np.float32)
	if feats.ndim == 1:
		feats = feats.reshape(1, -1)
	if feats.ndim != 2 or feats.shape[1] != EXPECTED_DIM or not np.isfinite(feats).all():
		return None
	return l2_normalize_rows(feats)


def best_body_sim_to_gallery(emb: np.ndarray | None, people: List[PersonEntry], topk: int = 3) -> float:
	if emb is None or not people:
		return 0.0
	q = l2_normalize(np.asarray(emb, np.float32).reshape(-1))
	if q.size != EXPECTED_DIM or not np.isfinite(q).all():
		return 0.0
	best = 0.0
	k_req = max(1, int(topk))
	for p in people:
		bank = p.body_bank
		if bank is None or bank.size == 0:
			continue
		sims = bank @ q
		if sims.ndim != 1 or sims.size == 0:
			continue
		k = min(k_req, sims.size)
		s = float(np.max(sims)) if k <= 1 else float(np.mean(np.partition(sims, -k)[-k:]))
		if s > best:
			best = s
	return float(best)


# ---------------------------
# Tracker (IoU fallback)
# ---------------------------
class IOUTrack:
	def __init__(self, tlwh, tid):
		self.tlwh = np.array(tlwh, dtype=np.float32)
		self.tid = int(tid)
		self.miss = 0


class IOUTracker:
	def __init__(self, max_miss: int = 5, iou_thresh: float = 0.3):
		self.tracks: List[IOUTrack] = []
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
			t_xyxy = np.array([t_x, t_y, t_x + t_w, t_y + t_h], np.float32)
			best_j, best_iou = -1, 0.0
			for j, d in enumerate(dets):
				if j in assigned:
					continue
				x, y, w, h = d[:4]
				d_xyxy = np.array([x, y, x + w, y + h], np.float32)
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

		outs = []
		for t in self.tracks:
			x, y, w, h = t.tlwh

			class Dummy:
				pass

			o = Dummy()
			o.track_id = t.tid
			o.is_confirmed = lambda: True
			o.to_tlbr = lambda: (x, y, x + w, y + h)
			o.time_since_update = 0
			o.det_conf = None
			outs.append(o)

		return outs


# ---------------------------
# Face engine init
# ---------------------------
def init_face_engine(args: argparse.Namespace):
	if not bool(getattr(args, "use_face", False)):
		return None
	if not _INSIGHT_OK:
		print("[WARN] insightface not installed; face disabled.")
		return None
	providers = ["CPUExecutionProvider"]
	device = str(getattr(args, "device", "cpu")).lower()
	if "cuda" in device and torch.cuda.is_available():
		# attempt CUDA EP if available
		providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]

	try:
		app = FaceAnalysis(name=str(getattr(args, "face_model", "buffalo_l")), providers=providers)
		det_w, det_h = getattr(args, "face_det_size", [1280, 1280])
		ctx_id = 0 if providers[0].startswith("CUDA") else -1
		try:
			app.prepare(ctx_id=ctx_id, det_size=(int(det_w), int(det_h)))
		except TypeError:
			app.prepare(ctx_id=ctx_id)
		print(f"[INIT] InsightFace ready providers={providers}")
		return app
	except Exception as e:
		print("[WARN] InsightFace init failed:", e)
		return None


# ---------------------------
# RTSP from DB camera IP
# ---------------------------
def _rtsp_url_from_db_camera_ip(ip: str) -> str:
	ip = str(ip or "").strip()
	username = os.environ.get("RTSP_USER", "admin")
	password = os.environ.get("RTSP_PASS", "admin")
	port = os.environ.get("RTSP_PORT", "554")
	stream = os.environ.get("RTSP_STREAM", "101")
	scheme = os.environ.get("RTSP_SCHEME", "rtsp")
	path = os.environ.get("RTSP_PATH", "Streaming/channels/")

	user_enc = quote(unquote(username or ""), safe="")
	pass_enc = quote(unquote(password or ""), safe="")

	tmpl = os.environ.get("RTSP_URL_TEMPLATE", "").strip()
	if tmpl:
		try:
			return tmpl.format(username=user_enc, password=pass_enc, ip=ip, port=port, stream=stream, scheme=scheme, path=path)
		except Exception:
			pass

	path = str(path or "").lstrip("/")
	if not path.endswith("/"):
		path += "/"
	return f"{scheme}://{user_enc}:{pass_enc}@{ip}:{port}/{path}{stream}"


# ---------------------------
# Core frame processing
# ---------------------------
def _yolo_forward(yolo, frame_bgr: np.ndarray, args: argparse.Namespace):
	if yolo is None:
		return None
	with _yolo_lock, torch.inference_mode():
		try:
			return yolo(
				frame_bgr,
				conf=float(getattr(args, "conf", 0.3)),
				iou=float(getattr(args, "iou", 0.4)),
				verbose=False,
				device=str(getattr(args, "device", "cpu")),
				half=bool(getattr(args, "half", False)),
				imgsz=int(getattr(args, "yolo_imgsz", 0)) or None,
			)
		except TypeError:
			return yolo(
				frame_bgr,
				conf=float(getattr(args, "conf", 0.3)),
				iou=float(getattr(args, "iou", 0.4)),
				verbose=False,
				device=str(getattr(args, "device", "cpu")),
				half=bool(getattr(args, "half", False)),
			)


def process_one_frame(
	frame_idx: int,
	frame_bgr: np.ndarray,
	sid: int,
	camera_id_for_logging: int,
	yolo,
	args: argparse.Namespace,
	deep_tracker,
	iou_tracker: IOUTracker,
	people: List[PersonEntry],
	reid_extractor,
	face_app,
	face_gallery: FaceGallery,
	identity_state: Dict[int, Dict[str, Any]],
	device_is_cuda: bool,
) -> Tuple[np.ndarray, Dict[str, Any], List[DetectionRow]]:
	"""
	Returns (annotated_frame, meta, detection_rows)

	Drawing rule: ONLY known.
	"""
	H, W = frame_bgr.shape[:2]

	# 1) YOLO persons
	tlwh_conf: List[List[float]] = []
	res = _yolo_forward(yolo, frame_bgr, args) if yolo is not None else None
	if res is not None and len(res):
		boxes = res[0].boxes
		if boxes is not None:
			xyxy = boxes.xyxy.detach().cpu().numpy().astype(np.float32)
			conf = boxes.conf.detach().cpu().numpy().astype(np.float32)
			cls = boxes.cls.detach().cpu().numpy().astype(np.int32)
			keep = cls == 0
			xyxy, conf = xyxy[keep], conf[keep]
			min_wh = int(getattr(args, "min_box_wh", 40))
			for (x1, y1, x2, y2), c in zip(xyxy, conf):
				x1f = float(max(0, min(W - 1, x1)))
				y1f = float(max(0, min(H - 1, y1)))
				x2f = float(max(0, min(W - 1, x2)))
				y2f = float(max(0, min(H - 1, y2)))
				ww = float(max(1.0, x2f - x1f))
				hh = float(max(1.0, y2f - y1f))
				if ww < min_wh or hh < min_wh:
					continue
				tlwh_conf.append([x1f, y1f, ww, hh, float(c)])

	dets_np = np.asarray(tlwh_conf, np.float32)
	if dets_np.ndim != 2:
		dets_np = dets_np.reshape((0, 5)).astype(np.float32)

	dets_dsrt = [([float(x), float(y), float(w), float(h)], float(cf), 0) for x, y, w, h, cf in tlwh_conf]

	# 2) Tracker
	if deep_tracker is not None:
		try:
			out_tracks = deep_tracker.update_tracks(dets_dsrt, frame=frame_bgr)
		except Exception as e:
			print(f"[SRC {sid}] DeepSORT error:", e)
			out_tracks = []
	else:
		out_tracks = iou_tracker.update(dets_np)

	# 3) Faces (every n)
	faces_all: List[Dict[str, Any]] = []
	do_face = bool(face_app is not None) and (frame_idx % max(1, int(getattr(args, "face_every_n", 1))) == 0)
	store_guest_emb = bool(getattr(args, "store_guest_embeddings", True))

	if do_face:
		try:
			with _face_lock:
				faces = face_app.get(np.ascontiguousarray(frame_bgr))
			for f in safe_iter(faces):
				bbox = getattr(f, "bbox", None)
				if bbox is None:
					continue
				b = np.asarray(bbox).reshape(-1)
				if b.size < 4:
					continue
				fx1, fy1, fx2, fy2 = map(float, b[:4])
				fw, fh = max(0.0, fx2 - fx1), max(0.0, fy2 - fy1)
				if fw < float(getattr(args, "min_face_px", 24)) or fh < float(getattr(args, "min_face_px", 24)):
					continue

				emb = extract_face_embedding(f)
				if emb is None:
					continue
				emb = l2_normalize(np.asarray(emb, np.float32))

				flabel, fsim, f2 = best_face_label_top2(emb, face_gallery)
				best_sim = best_face_sim_to_gallery(emb, face_gallery)

				recognized = False
				gap = float(fsim - f2)
				if flabel is not None and (fsim >= float(getattr(args, "face_thresh", 0.4))) and (gap >= float(getattr(args, "face_gap", 0.05))):
					recognized = True

				faces_all.append(
					{
						"bbox": (fx1, fy1, fx2, fy2),
						"recognized": recognized,
						"label": str(flabel) if recognized else "",
						"sim": float(fsim) if recognized else 0.0,
						"best_sim": float(best_sim),
						"emb": emb if store_guest_emb else None,
					}
				)
		except Exception as e:
			print(f"[SRC {sid}] Face error:", e)

	# 4) Build per-track info + link faces
	tracks_info: List[Dict[str, Any]] = []
	for t in out_tracks:
		try:
			if hasattr(t, "is_confirmed") and callable(getattr(t, "is_confirmed")) and (not t.is_confirmed()):
				continue
			ltrb = t.to_tlbr() if hasattr(t, "to_tlbr") else t.to_ltrb()
			x1, y1, x2, y2 = map(int, ltrb)
			x1 = int(max(0, min(W - 1, x1)))
			y1 = int(max(0, min(H - 1, y1)))
			x2 = int(max(0, min(W, x2)))
			y2 = int(max(0, min(H, y2)))
			if x2 <= x1 or y2 <= y1:
				continue
			tid = int(getattr(t, "track_id", -1))
		except Exception:
			continue

		face_seen = False
		face_recognized = False
		face_label = ""
		face_sim = 0.0
		face_best_sim = 0.0
		face_emb = None

		if faces_all:
			t_xyxy = (float(x1), float(y1), float(x2), float(y2))
			best_idx = -1
			best_score = -1e9
			link_mode = str(getattr(args, "face_link_mode", "ioa"))
			link_thresh = float(getattr(args, "face_iou_link", 0.35))
			center_in = bool(getattr(args, "face_center_in_person", False))
			max_y_ratio = float(getattr(args, "face_center_y_max_ratio", 0.70))

			for idx, fm in enumerate(faces_all):
				fbox = fm["bbox"]
				cx = 0.5 * (float(fbox[0]) + float(fbox[2]))
				cy = 0.5 * (float(fbox[1]) + float(fbox[3]))
				if center_in and (not _point_in_xyxy(cx, cy, t_xyxy)):
					continue
				top_limit = float(y1) + max_y_ratio * float(max(1, (y2 - y1)))
				if cy > top_limit:
					continue

				link = ioa_xyxy(fbox, t_xyxy) if link_mode == "ioa" else iou_xyxy(t_xyxy, fbox)
				if link < link_thresh:
					continue

				score = float(link) + 0.10 * float(fm.get("best_sim", 0.0))
				if score > best_score:
					best_score = score
					best_idx = idx

			if best_idx >= 0:
				fm = faces_all[best_idx]
				face_seen = True
				face_recognized = bool(fm.get("recognized", False))
				face_label = str(fm.get("label", "")) if face_recognized else ""
				face_sim = float(fm.get("sim", 0.0)) if face_recognized else 0.0
				face_best_sim = float(fm.get("best_sim", 0.0))
				face_emb = fm.get("emb", None)

		# state
		st = identity_state.setdefault(tid, {"name": "", "face_vis_ttl": 0, "last_face_sim": 0.0, "guest_temp_id": f"guest_{camera_id_for_logging}_{tid}"})
		if do_face:
			st["face_vis_ttl"] = max(0, int(st.get("face_vis_ttl", 0)) - 1)

		if face_recognized and face_label:
			st["name"] = face_label
			st["last_face_sim"] = float(face_sim)
			st["face_vis_ttl"] = max(1, int(getattr(args, "face_hold_frames", 2)))

		tracks_info.append(
			{
				"tid": tid,
				"bbox": (x1, y1, x2, y2),
				"face_seen": face_seen,
				"face_recognized": face_recognized,
				"face_label": face_label,
				"face_sim": face_sim,
				"face_best_sim": face_best_sim,
				"face_emb": face_emb,
				"show_label": bool(st.get("name")) and int(st.get("face_vis_ttl", 0)) > 0,
				"stable_name": str(st.get("name", "")),
				"stable_face_sim": float(st.get("last_face_sim", 0.0)),
				"guest_temp_id": str(st.get("guest_temp_id", "")),
			}
		)

	# 5) Body embeddings for unknown without face (for guest_data_vector)
	body_emb_by_tid: Dict[int, np.ndarray] = {}
	body_best_sim_by_tid: Dict[int, float] = {}
	if store_guest_emb and reid_extractor is not None:
		need = [(r["tid"], r["bbox"]) for r in tracks_info if (not r["show_label"]) and (not r["face_seen"])]
		if need:
			tids = [t for t, _ in need]
			boxes = [b for _, b in need]
			crops = [frame_bgr[y1:y2, x1:x2] for (x1, y1, x2, y2) in boxes]
			feats = extract_body_embeddings_batch(reid_extractor, crops, device_is_cuda=device_is_cuda, use_half=bool(getattr(args, "half", False)))
			if feats is not None:
				for tid, emb in zip(tids, feats):
					body_emb_by_tid[int(tid)] = emb
					body_best_sim_by_tid[int(tid)] = best_body_sim_to_gallery(emb, people, topk=int(getattr(args, "reid_topk", 3)))

	# 6) Build detections + draw only known
	out = frame_bgr.copy()
	det_rows: List[DetectionRow] = []
	present_names: List[str] = []

	name_to_member_id = {p.name: int(p.member_id) for p in (people or [])}

	now_dt = datetime.utcnow()
	shown = 0
	for r in tracks_info:
		tid = int(r["tid"])
		x1, y1, x2, y2 = r["bbox"]

		if r["show_label"]:
			name = str(r["stable_name"])
			member_id = name_to_member_id.get(name)
			mv = float(r.get("stable_face_sim", 0.0))
			match_value = int(max(0, min(100, round(mv * 100.0))))
			det_rows.append(DetectionRow(data_ts=now_dt, camera_id=int(camera_id_for_logging), member_id=member_id, guest_temp_id=None, guest_data_vector=None, match_value=match_value))
			present_names.append(name)

			# draw GREEN only for known
			cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
			cv2.putText(out, f"{name} | SIM {mv:.2f}", (x1, max(0, y1 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
			shown += 1
		else:
			# unknown: do not draw
			if r["face_seen"]:
				mv = float(r.get("face_best_sim", 0.0))
				guest_vec = {"tid": tid, "face_embedding": np.asarray(r["face_emb"], np.float32).reshape(-1).astype(float).tolist()} if (store_guest_emb and r.get("face_emb") is not None) else {"tid": tid}
			else:
				mv = float(body_best_sim_by_tid.get(tid, 0.0))
				emb = body_emb_by_tid.get(tid)
				guest_vec = {"tid": tid, "body_embedding": np.asarray(emb, np.float32).reshape(-1).astype(float).tolist()} if (store_guest_emb and emb is not None) else {"tid": tid}

			match_value = int(max(0, min(100, round(mv * 100.0))))
			det_rows.append(DetectionRow(data_ts=now_dt, camera_id=int(camera_id_for_logging), member_id=None, guest_temp_id=str(r["guest_temp_id"]), guest_data_vector=guest_vec, match_value=match_value))

	meta = {"tracks": len(tracks_info), "shown": shown, "faces_total": len(faces_all), "do_face": do_face, "present_names": sorted(set(present_names)), "camera_id": int(camera_id_for_logging)}
	return out, meta, det_rows


# ---------------------------
# Processor thread
# ---------------------------
def processor_thread(
	sid: int,
	camera_id_for_gallery: Optional[int],
	camera_id_for_logging: int,
	vs,
	out_buf: FrameBuffer,
	yolo,
	args: argparse.Namespace,
	deep_tracker,
	iou_tracker: IOUTracker,
	gallery_mgr: GalleryManager,
	reid_extractor,
	face_app,
	report: Optional[SummaryReport],
	report_cam_index: Optional[int],
	ingest: Optional[RawIngestionManager],
	stop_event: threading.Event,
):
	frame_idx = 0
	identity_state: Dict[int, Dict[str, Any]] = {}

	device_is_cuda = torch.cuda.is_available() and ("cuda" in str(getattr(args, "device", "cpu")).lower())

	last_t = time.time()
	fps_ema = 0.0
	alpha = 0.10

	while not stop_event.is_set():
		ok, frame, ts_cap = vs.read()
		if not ok or frame is None:
			time.sleep(0.005)
			continue

		try:
			gallery_mgr.maybe_reload(args)
			people, face_gallery = gallery_mgr.snapshot(camera_id=camera_id_for_gallery)

			out, meta, det_rows = process_one_frame(
				frame_idx=frame_idx,
				frame_bgr=frame,
				sid=sid,
				camera_id_for_logging=int(camera_id_for_logging),
				yolo=yolo,
				args=args,
				deep_tracker=deep_tracker,
				iou_tracker=iou_tracker,
				people=people,
				reid_extractor=reid_extractor,
				face_app=face_app,
				face_gallery=face_gallery,
				identity_state=identity_state,
				device_is_cuda=device_is_cuda,
			)

			if ingest is not None and det_rows:
				ingest.push_many(det_rows)

			if bool(getattr(args, "overlay_fps", False)):
				now = time.time()
				dt = max(1e-6, now - last_t)
				inst_fps = 1.0 / dt
				fps_ema = (1 - alpha) * fps_ema + alpha * inst_fps
				last_t = now
				cv2.putText(out, f"FPS: {fps_ema:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

			if report is not None and report_cam_index is not None:
				names = meta.get("present_names", []) or []
				ts_use = float(ts_cap) if ts_cap else time.time()
				report.update(cam_id=int(report_cam_index), present_names=list(names), ts=ts_use)

			out_buf.set(out, meta)
			frame_idx += 1

		except Exception as e:
			print(f"[PROC {sid}] error:", e)
			time.sleep(0.01)


# ---------------------------
# Args parsing (kept compatible with your PIPELINE_ARGS)
# ---------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
	"""
	Full argument parser (kept intentionally permissive for compatibility with existing PIPELINE_ARGS).
	Many arguments are accepted even if this runner doesn't use every single one.
	"""
	ap = argparse.ArgumentParser(conflict_handler="resolve")

	ap.add_argument("--src", nargs="*", default=[], help="RTSP URLs OR camera IDs. If empty -> DB mode start ALL.")

	# DB
	ap.add_argument("--use-db", action="store_true")
	ap.add_argument("--db-url", default="")
	ap.add_argument("--db-refresh-seconds", type=float, default=60.0)
	ap.add_argument("--db-max-bank", type=int, default=256)
	ap.add_argument("--db-include-inactive", action="store_true")

	# YOLO
	ap.add_argument("--yolo-weights", default="yolov8n.pt")
	ap.add_argument("--yolo-imgsz", type=int, default=1280)
	ap.add_argument("--device", default="cuda:0")
	ap.add_argument("--conf", type=float, default=0.30)
	ap.add_argument("--iou", type=float, default=0.40)
	ap.add_argument("--half", action="store_true")
	ap.add_argument("--cudnn-benchmark", action="store_true")

	# Readers / stream
	ap.add_argument("--reader", choices=["latest", "adaptive", "ffmpeg"], default="adaptive")
	ap.add_argument("--queue-size", type=int, default=128)
	ap.add_argument("--rtsp-buffer", type=int, default=2)
	ap.add_argument("--decode-skip", type=int, default=0)
	ap.add_argument("--rtsp-transport", choices=["tcp", "udp"], default="tcp")
	ap.add_argument("--ffmpeg-cuda", action="store_true")
	ap.add_argument("--ffmpeg-width", type=int, default=0)
	ap.add_argument("--ffmpeg-height", type=int, default=0)
	ap.add_argument("--resize", type=int, nargs=2, default=[0, 0])

	# Adaptive reader extras (accepted for compatibility)
	ap.add_argument("--max-queue-age-ms", type=int, default=1000)
	ap.add_argument("--max-drain-per-cycle", type=int, default=32)

	# Tracking
	ap.add_argument("--use-deepsort", action="store_true")
	ap.add_argument("--no-deepsort", action="store_true")
	ap.add_argument("--max-age", type=int, default=15)
	ap.add_argument("--n-init", type=int, default=3)
	ap.add_argument("--nn-budget", type=int, default=200)
	ap.add_argument("--tracker-max-cosine", type=float, default=0.4)
	ap.add_argument("--tracker-nms-overlap", type=float, default=1.0)
	ap.add_argument("--iou-max-miss", type=int, default=5)

	# TorchReID / body embeddings
	ap.add_argument("--reid-model", default="osnet_x0_25")
	ap.add_argument("--reid-weights", default="")
	ap.add_argument("--reid-batch-size", type=int, default=16)
	ap.add_argument("--reid-topk", type=int, default=3)
	ap.add_argument("--gallery-thresh", type=float, default=0.70)
	ap.add_argument("--gallery-gap", type=float, default=0.08)
	ap.add_argument("--min-box-wh", type=int, default=40)

	# Face
	ap.add_argument("--use-face", action="store_true")
	ap.add_argument("--face-model", default="buffalo_l")
	ap.add_argument("--face-det-size", type=int, nargs=2, default=[1280, 1280])
	ap.add_argument("--face-thresh", type=float, default=0.40)
	ap.add_argument("--face-gap", type=float, default=0.05)
	ap.add_argument("--face-every-n", type=int, default=1)
	ap.add_argument("--face-hold-frames", type=int, default=2)
	ap.add_argument("--face-provider", choices=["auto", "cuda", "cpu"], default="auto")
	ap.add_argument("--ort-log", action="store_true")
	ap.add_argument("--face-iou-link", type=float, default=0.35)
	ap.add_argument("--face-link-mode", choices=["ioa", "iou"], default="ioa")
	ap.add_argument("--face-center-in-person", action="store_true")
	ap.add_argument("--min-face-px", type=int, default=24)
	ap.add_argument("--min-face-area-ratio", type=float, default=0.006)
	ap.add_argument("--face-center-y-max-ratio", type=float, default=0.70)
	ap.add_argument("--face-strong-thresh", type=float, default=0.50)

	# Identity smoothing (accepted, only partially used by this runner)
	ap.add_argument("--name-decay", type=float, default=0.85)
	ap.add_argument("--name-min-score", type=float, default=0.60)
	ap.add_argument("--name-margin", type=float, default=0.30)
	ap.add_argument("--name-ttl", type=int, default=20)
	ap.add_argument("--name-face-weight", type=float, default=1.2)
	ap.add_argument("--name-body-weight", type=float, default=0.5)

	# Drawing / filtering (accepted, but the runner always hides unknown boxes by design)
	ap.add_argument("--draw-only-matched", action="store_true")
	ap.add_argument("--min-det-conf", type=float, default=0.45)

	# Dup suppression / global naming
	ap.add_argument("--allow-duplicate-names", action="store_true")
	ap.add_argument("--dedup-iou", type=float, default=0.0)
	if hasattr(argparse, "BooleanOptionalAction"):
		ap.add_argument("--global-unique-names", action=argparse.BooleanOptionalAction, default=True)
	else:
		ap.add_argument("--global-unique-names", action="store_true", default=True)
	ap.add_argument("--global-hold-seconds", type=float, default=0.5)
	ap.add_argument("--global-switch-margin", type=float, default=0.02)
	ap.add_argument("--show-global-id", action="store_true")

	# UI / debug
	ap.add_argument("--show", action="store_true")
	ap.add_argument("--overlay-fps", action="store_true")

	# CSV report
	ap.add_argument("--save-csv", action="store_true")
	ap.add_argument("--csv", default="detections_summary.csv")
	ap.add_argument("--report-gap-seconds", type=float, default=2.0)
	ap.add_argument("--report-time-format", default="%H:%M:%S")

	# Ingestion
	ap.add_argument("--enable-ingest", action="store_true")
	ap.add_argument("--raw-rotate-seconds", type=int, default=300)
	ap.add_argument("--ingest-flush-interval", type=float, default=1.0)
	ap.add_argument("--ingest-batch-size", type=int, default=500)
	ap.add_argument("--raw-retention-minutes", type=int, default=0)
	if hasattr(argparse, "BooleanOptionalAction"):
		ap.add_argument("--raw-cleanup", action=argparse.BooleanOptionalAction, default=False)
	else:
		ap.add_argument("--raw-cleanup", action="store_true", default=False)

	# default ON (compatible with --store-guest-embeddings / --no-store-guest-embeddings)
	if hasattr(argparse, "BooleanOptionalAction"):
		ap.add_argument("--store-guest-embeddings", action=argparse.BooleanOptionalAction, default=True)
	else:
		ap.add_argument("--store-guest-embeddings", action="store_true", default=True)

	return ap.parse_args(argv)


def parse_pipeline_args(pipeline_args: str | None) -> argparse.Namespace:
	s = str(pipeline_args or "").strip()
	argv = shlex.split(s) if s else []
	return parse_args(argv)


# ---------------------------
# Global name gate (optional) - kept for status compatibility
# ---------------------------
class GlobalNameOwner:
	def __init__(self, hold_seconds: float = 0.5, switch_margin: float = 0.02):
		self.hold_seconds = float(max(0.0, hold_seconds))
		self.switch_margin = float(max(0.0, switch_margin))
		self._lock = threading.Lock()
		self._state: Dict[str, Dict[str, Any]] = {}

	def _cleanup(self, now: float):
		if self.hold_seconds <= 0:
			return
		dead = []
		for name, st in self._state.items():
			if (now - float(st.get("ts", 0.0))) > self.hold_seconds:
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


# ---------------------------
# Runner
# ---------------------------
class TrackingRunner:
	def __init__(self, args: argparse.Namespace):
		self.args = args
		self._lock = threading.Lock()
		self._started = False

		self.gallery_mgr: Optional[GalleryManager] = None
		self.yolo = None
		self.reid_extractor = None
		self.face_app = None

		self.report: Optional[SummaryReport] = None
		self.ingest: Optional[RawIngestionManager] = None

		self.stop_event = threading.Event()

		self._mode: str = "db"
		self._manual_states: Dict[int, Dict[str, Any]] = {}
		self._db_states: Dict[int, Dict[str, Any]] = {}
		self._camid_to_report_index: Dict[int, int] = {}

		self._db_engine = None
		self._Session = None

	# --- DB cameras ---
	def _fetch_db_cameras(self, active_only: bool = True) -> List[Dict[str, Any]]:
		if self._Session is None:
			return []
		Base = declarative_base()

		class CameraRow(Base):
			__tablename__ = "cameras"
			id = Column(Integer, primary_key=True)
			name = Column(String(64))
			ip_address = Column(String(64))
			is_active = Column(Boolean)

		cams = []
		with self._Session() as session:
			rows = session.execute(
				select(CameraRow.id, CameraRow.name, CameraRow.ip_address, CameraRow.is_active).order_by(CameraRow.id.asc())
			).all()
			for r in rows:
				is_active_c = bool(r.is_active) if r.is_active is not None else True
				if active_only and (not is_active_c):
					continue
				cams.append({"id": int(r.id), "name": str(r.name or ""), "ip_address": str(r.ip_address or ""), "is_active": is_active_c})
		return cams

	def _build_report_camera_index_map(self):
		self._camid_to_report_index = {}
		cams = self._fetch_db_cameras(active_only=True)
		for idx, cam in enumerate(cams):
			self._camid_to_report_index[int(cam["id"])] = int(idx)

	# --- init ---
	def start(self):
		with self._lock:
			if self._started:
				return
			self._started = True

		args = self.args

		if not bool(getattr(args, "use_db", False)):
			raise RuntimeError("DB-first runner: set --use-db in PIPELINE_ARGS")
		if not str(getattr(args, "db_url", "")).strip():
			raise RuntimeError("--db-url is required")

		src_list = list(getattr(args, "src", []) or [])
		self._mode = "manual" if any(_looks_like_url(s) for s in src_list) else "db"
		print(f"[INIT] mode={self._mode}")

		# gallery
		self.gallery_mgr = GalleryManager(args)

		# db session for cameras
		if self._mode == "db":
			self._db_engine = create_engine(args.db_url, pool_pre_ping=True)
			self._Session = sessionmaker(bind=self._db_engine)
			self._build_report_camera_index_map()

		# yolo
		if YOLO is not None:
			weights = str(getattr(args, "yolo_weights", "yolov8n.pt"))
			if not Path(weights).exists():
				print(f"[INIT] {weights} not found, using yolov8n.pt")
				weights = "yolov8n.pt"
			try:
				self.yolo = YOLO(weights)
			except Exception as e:
				print("[WARN] YOLO init failed:", e)
				self.yolo = None
		else:
			print("[WARN] ultralytics not installed; YOLO disabled")
			self.yolo = None

		# TorchReID (optional)
		gpu = torch.cuda.is_available() and ("cuda" in str(getattr(args, "device", "cpu")).lower())
		if TorchreidExtractor is not None:
			try:
				dev = str(getattr(args, "device", "cpu")) if gpu else "cpu"
				weights = str(getattr(args, "reid_weights", ""))
				if weights and Path(weights).exists():
					self.reid_extractor = TorchreidExtractor(model_name=str(getattr(args, "reid_model", "osnet_x0_25")), model_path=weights, device=dev)
				else:
					self.reid_extractor = TorchreidExtractor(model_name=str(getattr(args, "reid_model", "osnet_x0_25")), device=dev)
			except Exception as e:
				print("[WARN] TorchReID init failed:", e)
				self.reid_extractor = None

		# face
		self.face_app = init_face_engine(args)

		# report
		if bool(getattr(args, "save_csv", False)):
			num_cams = max(1, len(self._camid_to_report_index) if self._mode == "db" else len(src_list))
			self.report = SummaryReport(num_cams=num_cams, gap_seconds=float(getattr(args, "report_gap_seconds", 2.0)), time_format=str(getattr(args, "report_time_format", "%H:%M:%S")))
		else:
			self.report = None

		# ingest
		if bool(getattr(args, "enable_ingest", False)):
			self.ingest = RawIngestionManager(
				db_url=args.db_url,
				rotate_seconds=int(getattr(args, "raw_rotate_seconds", 300)),
				flush_interval=float(getattr(args, "ingest_flush_interval", 1.0)),
				batch_size=int(getattr(args, "ingest_batch_size", 500)),
				retention_minutes=int(getattr(args, "raw_retention_minutes", 0)),
				enable_cleanup=bool(getattr(args, "raw_cleanup", False)),
			)

		# start sources
		if self._mode == "manual":
			self._start_manual_sources(src_list)
		else:
			cam_ids = [int(s) for s in src_list if _is_int_str(s)]
			if cam_ids:
				for cid in cam_ids:
					self.ensure_camera_started(cid)
				print(f"[Runner] Started {len(cam_ids)} camera(s) from --src IDs.")
			else:
				n = self.start_all_db_cameras(active_only=True)
				print(f"[Runner] Auto-started {n} camera(s) from DB at startup.")

	# --- open stream ---
	def _open_stream(self, src: str):
		args = self.args
		if str(getattr(args, "reader", "adaptive")) == "latest":
			return LatestStream(src, rtsp_buffer=int(getattr(args, "rtsp_buffer", 2)), decode_skip=int(getattr(args, "decode_skip", 0)), rtsp_transport=str(getattr(args, "rtsp_transport", "tcp")))
		if str(getattr(args, "reader", "adaptive")) == "ffmpeg":
			return FFmpegStream(
				src,
				queue_size=int(getattr(args, "queue_size", 128)),
				rtsp_transport=str(getattr(args, "rtsp_transport", "tcp")),
				use_cuda=bool(getattr(args, "ffmpeg_cuda", False)),
				force_w=int(getattr(args, "ffmpeg_width", 0)),
				force_h=int(getattr(args, "ffmpeg_height", 0)),
			)
		return AdaptiveQueueStream(src, queue_size=int(getattr(args, "queue_size", 128)), rtsp_transport=str(getattr(args, "rtsp_transport", "tcp")))

	def _make_deepsort(self, gpu: bool):
		args = self.args
		if DeepSort is None:
			return None
		try:
			return DeepSort(
				max_age=int(getattr(args, "max_age", 15)),
				n_init=int(getattr(args, "n_init", 3)),
				nn_budget=int(getattr(args, "nn_budget", 200)),
				max_cosine_distance=float(getattr(args, "tracker_max_cosine", 0.4)),
				nms_max_overlap=float(getattr(args, "tracker_nms_overlap", 1.0)),
				embedder="torchreid",
				embedder_gpu=bool(gpu),
				half=(bool(gpu) and bool(getattr(args, "half", False))),
				bgr=True,
			)
		except Exception as e:
			print("[WARN] DeepSort init failed:", e)
			return None

	def _should_use_deepsort(self) -> bool:
		if bool(getattr(self.args, "no_deepsort", False)):
			return False
		if bool(getattr(self.args, "use_deepsort", False)):
			return DeepSort is not None
		return DeepSort is not None

	# --- manual mode ---
	def _start_manual_sources(self, src_list: List[str]):
		gpu = torch.cuda.is_available() and ("cuda" in str(getattr(self.args, "device", "cpu")).lower())
		for i, raw_src in enumerate(src_list):
			cam_id = int(i)
			src = str(raw_src).strip()
			vs = self._open_stream(src)
			deep = self._make_deepsort(gpu) if self._should_use_deepsort() else None
			iou = IOUTracker(max_miss=int(getattr(self.args, "iou_max_miss", 5)), iou_thresh=0.3)
			buf = FrameBuffer()
			st = {"cam_id": cam_id, "sid": cam_id, "src": src, "vs": vs, "deep": deep, "iou": iou, "buf": buf, "thread": None}
			self._manual_states[cam_id] = st

			if not vs.is_opened():
				print(f"[SRC {cam_id}] open=False :: {src}")
				continue

			t = threading.Thread(
				target=processor_thread,
				args=(cam_id, None, cam_id, vs, buf, self.yolo, self.args, deep, iou, self.gallery_mgr, self.reid_extractor, self.face_app, self.report, cam_id, self.ingest, self.stop_event),
				daemon=True,
			)
			t.start()
			st["thread"] = t
			print(f"[SRC {cam_id}] open=True :: {src}")

	# --- DB mode ---
	def _start_db_camera_from_row(self, cam: Dict[str, Any]) -> bool:
		cam_id = int(cam.get("id", -1))
		if cam_id < 0:
			return False

		with self._lock:
			if cam_id in self._db_states:
				vs0 = self._db_states[cam_id].get("vs")
				if vs0 is not None and vs0.is_opened():
					return True
				# restart if not opened
				try:
					if vs0 is not None:
						vs0.release()
				except Exception:
					pass
				self._db_states.pop(cam_id, None)

		ip = str(cam.get("ip_address", "")).strip()
		if not ip:
			print(f"[SRC {cam_id}] skipped: empty ip_address")
			return False

		src = _rtsp_url_from_db_camera_ip(ip)
		vs = self._open_stream(src)
		opened = vs.is_opened()
		gpu = torch.cuda.is_available() and ("cuda" in str(getattr(self.args, "device", "cpu")).lower())
		deep = self._make_deepsort(gpu) if self._should_use_deepsort() else None
		iou = IOUTracker(max_miss=int(getattr(self.args, "iou_max_miss", 5)), iou_thresh=0.3)
		buf = FrameBuffer()

		sid = self._camid_to_report_index.get(cam_id, cam_id)
		report_idx = self._camid_to_report_index.get(cam_id)

		st = {"cam_id": cam_id, "sid": sid, "camera": dict(cam), "src": src, "vs": vs, "deep": deep, "iou": iou, "buf": buf, "thread": None, "started_at": time.time()}
		with self._lock:
			self._db_states[cam_id] = st

		print(f"[SRC {cam_id}] open={opened} :: {src}")
		if not opened:
			return False

		t = threading.Thread(
			target=processor_thread,
			args=(sid, cam_id, cam_id, vs, buf, self.yolo, self.args, deep, iou, self.gallery_mgr, self.reid_extractor, self.face_app, self.report, report_idx, self.ingest, self.stop_event),
			daemon=True,
		)
		t.start()
		with self._lock:
			if cam_id in self._db_states:
				self._db_states[cam_id]["thread"] = t
		return True

	def start_all_db_cameras(self, active_only: bool = True) -> int:
		cams = self._fetch_db_cameras(active_only=active_only)
		n = 0
		for cam in cams:
			if self._start_db_camera_from_row(cam):
				n += 1
		return n

	def _get_db_camera(self, camera_id: int) -> Optional[Dict[str, Any]]:
		if self._Session is None:
			return None
		Base = declarative_base()

		class CameraRow(Base):
			__tablename__ = "cameras"
			id = Column(Integer, primary_key=True)
			name = Column(String(64))
			ip_address = Column(String(64))
			is_active = Column(Boolean)

		with self._Session() as session:
			row = session.execute(select(CameraRow.id, CameraRow.name, CameraRow.ip_address, CameraRow.is_active).where(CameraRow.id == int(camera_id))).first()
			if not row:
				return None
			return {"id": int(row.id), "name": str(row.name or ""), "ip_address": str(row.ip_address or ""), "is_active": bool(row.is_active) if row.is_active is not None else True}

	def ensure_camera_started(self, camera_id: int) -> bool:
		cam_id = int(camera_id)
		with self._lock:
			st = self._db_states.get(cam_id)
			if st:
				vs = st.get("vs")
				if vs is not None and vs.is_opened():
					return True
				# restart
				try:
					if vs is not None:
						vs.release()
				except Exception:
					pass
				self._db_states.pop(cam_id, None)

		cam = self._get_db_camera(cam_id)
		if cam is None:
			return False
		return self._start_db_camera_from_row(cam)

	# --- API helpers ---
	def get_camera_buffer(self, cam_id: int) -> Optional[FrameBuffer]:
		if self._mode == "manual":
			st = self._manual_states.get(int(cam_id))
			return st.get("buf") if st else None

		ok = self.ensure_camera_started(int(cam_id))
		if not ok:
			return None
		with self._lock:
			st = self._db_states.get(int(cam_id))
		return st.get("buf") if st else None

	def list_db_cameras(self, active_only: bool = True) -> List[Dict[str, Any]]:
		if self._mode == "manual":
			out = []
			for cid, st in sorted(self._manual_states.items()):
				vs = st.get("vs")
				out.append({"id": int(cid), "mode": "manual", "opened": bool(vs.is_opened()) if vs else False, "src": str(st.get("src", ""))})
			return out

		cams = self._fetch_db_cameras(active_only=active_only)
		with self._lock:
			states = dict(self._db_states)

		out = []
		for cam in cams:
			cid = int(cam["id"])
			st = states.get(cid)
			vs = st.get("vs") if st else None
			out.append(
				{
					"id": cid,
					"name": str(cam.get("name", "")),
					"ip_address": str(cam.get("ip_address", "")),
					"is_active": bool(cam.get("is_active", True)),
					"running": bool(st is not None),
					"opened": bool(vs.is_opened()) if vs else False,
				}
			)
		return out

	def status(self) -> Dict[str, Any]:
		out = {"started": bool(self._started), "mode": self._mode, "ingest_enabled": bool(self.ingest is not None), "running": []}
		if self._mode == "manual":
			for cid, st in sorted(self._manual_states.items()):
				vs = st.get("vs")
				out["running"].append({"id": int(cid), "opened": bool(vs.is_opened()) if vs else False, "src": str(st.get("src", ""))})
		else:
			with self._lock:
				items = list(self._db_states.items())
			for cid, st in sorted(items):
				vs = st.get("vs")
				out["running"].append({"id": int(cid), "opened": bool(vs.is_opened()) if vs else False, "src": str(st.get("src", ""))})
		return out

	def write_report_snapshot(self, path: Optional[str] = None) -> str:
		if self.report is None:
			raise RuntimeError("CSV reporting not enabled. Add --save-csv")
		out_path = str(path or getattr(self.args, "csv", "detections_summary.csv"))
		self.report.write_csv_snapshot(out_path)
		return out_path

	# --- stop ---
	def stop(self):
		with self._lock:
			if not self._started:
				return
			self._started = False

		self.stop_event.set()

		if self._mode == "manual":
			states = list(self._manual_states.values())
		else:
			with self._lock:
				states = list(self._db_states.values())

		for st in states:
			try:
				st["vs"].release()
			except Exception:
				pass

		for st in states:
			t = st.get("thread")
			if t is not None:
				try:
					t.join(timeout=2.0)
				except Exception:
					pass

		if self.ingest is not None:
			try:
				self.ingest.stop()
			except Exception as e:
				print("[WARN] ingest stop failed:", e)

		if self.report is not None:
			try:
				self.report.stop()
				self.report.write_csv(str(getattr(self.args, "csv", "detections_summary.csv")))
			except Exception as e:
				print("[WARN] report write failed:", e)


def main():
	args = parse_args()
	runner = TrackingRunner(args)
	runner.start()
	try:
		while True:
			time.sleep(1.0)
	except KeyboardInterrupt:
		pass
	finally:
		runner.stop()


if __name__ == "__main__":
	main()
