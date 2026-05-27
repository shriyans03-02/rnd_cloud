from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Iterable, List, Optional, Sequence

from sqlalchemy import Boolean, Column, Integer, String, create_engine, select
from sqlalchemy.orm import declarative_base, sessionmaker


FALSE_VALUES = {"0", "false", "no", "off", "n"}
TRUE_VALUES = {"1", "true", "yes", "on", "y"}


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(str(name))
    if raw is None or str(raw).strip() == "":
        return bool(default)
    return str(raw).strip().lower() in TRUE_VALUES


def env_int(name: str, default: int) -> int:
    raw = os.environ.get(str(name))
    try:
        if raw is None or str(raw).strip() == "":
            return int(default)
        return int(str(raw).strip())
    except Exception:
        return int(default)


@dataclass(frozen=True)
class CameraSpec:
    id: int
    name: str = ""
    ip_address: str = ""
    is_active: bool = True


def parse_camera_expr(expr: str | None) -> List[int]:
    """Parse '1-8,11,15-18' into sorted unique camera IDs."""
    text = str(expr or "").strip()
    if not text:
        return []
    out: set[int] = set()
    for part in text.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            try:
                start, end = int(a.strip()), int(b.strip())
            except Exception:
                continue
            lo, hi = min(start, end), max(start, end)
            for x in range(lo, hi + 1):
                if x > 0:
                    out.add(int(x))
        else:
            try:
                x = int(part)
            except Exception:
                continue
            if x > 0:
                out.add(int(x))
    return sorted(out)


def _coerce_selected_ids(selected_ids: Optional[Iterable[int]]) -> set[int]:
    out: set[int] = set()
    for raw in selected_ids or []:
        try:
            v = int(raw)
        except Exception:
            continue
        if v > 0:
            out.add(v)
    return out


def list_active_db_cameras(db_url: str, selected_ids: Optional[Iterable[int]] = None) -> List[CameraSpec]:
    """Return active cameras from DB.  On errors, return [] instead of breaking API startup."""
    db_url = str(db_url or "").strip()
    if not db_url:
        return []
    selected = _coerce_selected_ids(selected_ids)
    BaseLocal = declarative_base()

    class CameraRow(BaseLocal):
        __tablename__ = "cameras"
        id = Column(Integer, primary_key=True)
        name = Column(String)
        ip_address = Column(String)
        is_active = Column(Boolean)

    engine = None
    try:
        engine = create_engine(db_url, pool_pre_ping=True)
        Session = sessionmaker(bind=engine)
        with Session() as session:
            rows = session.execute(
                select(CameraRow.id, CameraRow.name, CameraRow.ip_address, CameraRow.is_active)
            ).all()
        cams: list[CameraSpec] = []
        for r in rows:
            try:
                cam_id = int(r[0])
            except Exception:
                continue
            if selected and cam_id not in selected:
                continue
            active = bool(r[3]) if r[3] is not None else True
            if not active:
                continue
            cams.append(CameraSpec(
                id=cam_id,
                name=str(r[1] or f"Camera {cam_id}"),
                ip_address=str(r[2] or ""),
                is_active=True,
            ))
        cams.sort(key=lambda x: int(x.id))
        return cams
    except Exception as exc:
        print(f"[AI-WORKER] DB camera list failed: {type(exc).__name__}: {exc}")
        return []
    finally:
        try:
            if engine is not None:
                engine.dispose()
        except Exception:
            pass


def split_camera_ids(camera_ids: Sequence[int], *, max_per_worker: int = 8, workers: int = 0) -> List[List[int]]:
    ids = [int(x) for x in camera_ids if int(x) > 0]
    if not ids:
        return []
    if workers and int(workers) > 0:
        n_workers = max(1, int(workers))
    else:
        max_per = max(1, int(max_per_worker or 8))
        n_workers = int(math.ceil(len(ids) / float(max_per)))
    n_workers = max(1, min(n_workers, len(ids)))
    chunks: list[list[int]] = [[] for _ in range(n_workers)]
    # Contiguous split keeps related nearby camera IDs together and makes logs easier.
    chunk_size = int(math.ceil(len(ids) / float(n_workers)))
    chunks = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
    return [c for c in chunks if c]


def camera_ids_from_args(args: Any) -> List[int]:
    ids = []
    for raw in getattr(args, "camera_ids", []) or []:
        try:
            v = int(raw)
        except Exception:
            continue
        if v > 0:
            ids.append(v)
    return sorted(set(ids))
