from __future__ import annotations

import gzip
from dataclasses import dataclass
from io import BytesIO
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

from app.utils.embeddings import l2_normalize, l2_normalize_rows

EXPECTED_DIM = 512

# --- SQLAlchemy / pgvector ---
try:
    from sqlalchemy import (
        Column,
        Integer,
        String,
        LargeBinary,
        Boolean,
        BigInteger,
        DateTime,
        ForeignKey,
        create_engine,
        select,
        Float,
    )
    from sqlalchemy.orm import declarative_base, sessionmaker
    from sqlalchemy.dialects.postgresql import ARRAY
except Exception as e:
    raise RuntimeError("SQLAlchemy is required for DB gallery mode") from e

try:
    from pgvector.sqlalchemy import Vector
except Exception:
    Vector = None


def decode_bank_gzip_npy(raw: bytes | None) -> Optional[np.ndarray]:
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


def _display_name(first: str, last: Optional[str], member_number: Optional[str]) -> str:
    first = (first or "").strip()
    last = (last or "").strip()
    member_number = (member_number or "").strip()

    full = (first + " " + last).strip() if last else first
    if member_number:
        if full:
            return f"{full} ({member_number})"
        return member_number
    return full or "UNKNOWN"


@dataclass
class PersonEntry:
    user_id: int               # member_id
    name: str                  # display name
    body_bank: Optional[np.ndarray]
    body_centroid: Optional[np.ndarray]
    face_centroid: Optional[np.ndarray]


@dataclass
class FaceGallery:
    names: List[str]
    mat: np.ndarray

    def is_empty(self) -> bool:
        return (not self.names) or (self.mat is None) or (self.mat.size == 0)


class GalleryRepository:
    """
    Loads:
      - members (id, first_name, last_name, member_number, is_active)
      - member_embeddings (many rows per member, possibly per camera)
    Aggregates embeddings PER MEMBER.
    """

    def __init__(self, db_url: str):
        self.db_url = db_url

        Base = declarative_base()

        class MemberRow(Base):
            __tablename__ = "members"
            id = Column(Integer, primary_key=True)
            member_number = Column(String(16), nullable=False)
            first_name = Column(String(64), nullable=False)
            last_name = Column(String(64), nullable=True)
            is_active = Column(Boolean, nullable=False)

        class MemberEmbeddingRow(Base):
            __tablename__ = "member_embeddings"
            id = Column(BigInteger, primary_key=True)
            member_id = Column(Integer, ForeignKey("members.id", ondelete="CASCADE"), nullable=False)
            camera_id = Column(Integer, ForeignKey("cameras.id", ondelete="CASCADE"), nullable=False)

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

        self.MemberRow = MemberRow
        self.MemberEmbeddingRow = MemberEmbeddingRow

        self.engine = create_engine(self.db_url, pool_pre_ping=True)
        self.Session = sessionmaker(bind=self.engine)

    def load(
        self,
        active_only: bool = True,
        max_bank_per_member: int = 0,
    ) -> Tuple[List[PersonEntry], FaceGallery]:
        people: List[PersonEntry] = []
        face_names: List[str] = []
        face_vecs: List[np.ndarray] = []

        # Per member accumulators
        acc: Dict[int, Dict[str, Any]] = {}

        with self.Session() as session:
            M = self.MemberRow
            E = self.MemberEmbeddingRow

            stmt = (
                select(
                    M.id.label("mid"),
                    M.member_number,
                    M.first_name,
                    M.last_name,
                    M.is_active,
                    E.camera_id,
                    E.body_embedding,
                    E.face_embedding,
                    E.body_embeddings_raw,
                    E.face_embeddings_raw,
                    E.last_embedding_update_ts,
                )
                .select_from(M)
                .join(E, E.member_id == M.id, isouter=True)
            )

            if active_only:
                stmt = stmt.where(M.is_active.is_(True))

            rows = session.execute(stmt).all()

            for r in rows:
                mid = int(r.mid)
                if mid not in acc:
                    name = _display_name(r.first_name, r.last_name, r.member_number)
                    acc[mid] = {
                        "user_id": mid,
                        "name": name,
                        "body_chunks": [],        # list[(ts_float, np.ndarray Nx512)]
                        "body_centroid": None,
                        "body_centroid_ts": -1.0,
                        "face_centroid": None,
                        "face_centroid_ts": -1.0,
                        "face_raw_candidates": [],  # list[(ts_float, bytes)]
                    }

                st = acc[mid]
                ts = r.last_embedding_update_ts
                ts_f = float(ts.timestamp()) if ts is not None else 0.0

                # Body centroid candidate
                b_cent = _as_vec512(r.body_embedding)
                if b_cent is not None and ts_f >= float(st["body_centroid_ts"]):
                    st["body_centroid"] = b_cent
                    st["body_centroid_ts"] = ts_f

                # Face centroid candidate
                f_cent = _as_vec512(r.face_embedding)
                if f_cent is not None and ts_f >= float(st["face_centroid_ts"]):
                    st["face_centroid"] = f_cent
                    st["face_centroid_ts"] = ts_f

                # Body bank chunks (prefer raw; otherwise use centroid as 1-row bank)
                if r.body_embeddings_raw:
                    bank = decode_bank_gzip_npy(r.body_embeddings_raw)
                    if bank is not None and bank.ndim == 2 and bank.shape[1] == EXPECTED_DIM:
                        st["body_chunks"].append((ts_f, bank))
                elif b_cent is not None:
                    st["body_chunks"].append((ts_f, b_cent.reshape(1, -1).astype(np.float32)))

                # Store face raw only for fallback (if no face_embedding at all)
                if r.face_embeddings_raw:
                    st["face_raw_candidates"].append((ts_f, r.face_embeddings_raw))

        # Build final people + face gallery
        for mid, st in acc.items():
            # Build body bank
            body_bank = None
            chunks = list(st["body_chunks"])
            chunks.sort(key=lambda x: x[0], reverse=True)

            if chunks:
                mats = [c for _, c in chunks if c is not None and c.size > 0]
                if mats:
                    body_bank = np.concatenate(mats, axis=0).astype(np.float32)
                    body_bank = l2_normalize_rows(body_bank)

            m = int(max_bank_per_member or 0)
            if m > 0 and body_bank is not None and len(body_bank) > m:
                body_bank = body_bank[:m]

            body_centroid = st["body_centroid"]
            if body_centroid is None and body_bank is not None and len(body_bank) > 0:
                body_centroid = l2_normalize(np.mean(body_bank, axis=0))

            # Face centroid fallback: if no face_embedding, compute from raw (newest-first)
            face_centroid = st["face_centroid"]
            if face_centroid is None:
                raws = list(st["face_raw_candidates"])
                raws.sort(key=lambda x: x[0], reverse=True)
                # Take first few newest raw banks until we can build a centroid
                face_vecs_tmp = []
                for _, raw in raws[:4]:
                    bank = decode_bank_gzip_npy(raw)
                    if bank is None or bank.size == 0:
                        continue
                    face_vecs_tmp.append(np.mean(bank, axis=0))
                if face_vecs_tmp:
                    face_centroid = l2_normalize(np.mean(np.stack(face_vecs_tmp, axis=0), axis=0))

            # Ensure body_bank exists if centroid exists
            if body_bank is None and body_centroid is not None:
                body_bank = body_centroid.reshape(1, -1).astype(np.float32)

            people.append(
                PersonEntry(
                    user_id=int(st["user_id"]),
                    name=str(st["name"]),
                    body_bank=body_bank,
                    body_centroid=body_centroid,
                    face_centroid=face_centroid,
                )
            )

            if face_centroid is not None:
                face_names.append(str(st["name"]))
                face_vecs.append(face_centroid.astype(np.float32))

        face_mat = (
            l2_normalize_rows(np.stack(face_vecs, axis=0))
            if face_vecs
            else np.zeros((0, EXPECTED_DIM), dtype=np.float32)
        )

        fg = FaceGallery(face_names, face_mat)
        return people, fg
