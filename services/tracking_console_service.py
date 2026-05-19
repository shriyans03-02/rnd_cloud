from typing import Dict, List, Optional

from app.db.session import SessionLocal
from app.repositories.tracking_console_repo import TrackingConsoleRepository


class TrackingConsoleService:

    @staticmethod
    def list_members(
        search: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict]:
        """
        Returns active members that have complete embeddings.
        Delegates to the same repo method used by /tracing/active/names
        so the filter condition is always in sync.
        """
        db = SessionLocal()
        try:
            members = TrackingConsoleRepository.list_all_active(
                db, search=search, limit=limit
            )
            return [
                {
                    "id": m.id,
                    "name": f"{m.first_name} {m.last_name or ''}".strip(),
                }
                for m in members
            ]
        finally:
            db.close()

    @staticmethod
    def list_site_locations(search: Optional[str] = None) -> List[Dict]:
        """
        Returns active site locations that belong to a fully-active hierarchy
        and have at least one active camera assigned.
        Delegates to the same repo method used by /tracing/active/site_locations
        so the filter condition is always in sync.
        """
        db = SessionLocal()
        try:
            return TrackingConsoleRepository.list_site_locations(db, search=search)
        finally:
            db.close()

    @staticmethod
    def resolve_cameras_for_members(member_ids: List[int]) -> List[Dict]:
        """
        Returns the deduplicated list of cameras on which the given members
        are currently live (movement_type=1, exit_ts IS NULL).

        Before querying NormalizedData, validates member_ids against the same
        active+embedding filter used by list_members / /tracing/active/names.
        """
        db = SessionLocal()
        try:
            # ── Validate: keep only members that pass the active+embedding check ──
            valid_members = TrackingConsoleRepository.list_all_active(db, limit=10_000)
            valid_id_set = {m.id for m in valid_members}
            filtered_ids = [mid for mid in member_ids if mid in valid_id_set]

            if not filtered_ids:
                return []

            return TrackingConsoleRepository.get_live_cameras_for_members(
                db, filtered_ids
            )
        finally:
            db.close()

    @staticmethod
    def resolve_cameras_for_locations(site_location_ids: List[int]) -> List[Dict]:
        """
        Returns the deduplicated list of active cameras assigned to the given
        site locations.

        Before querying Camera, validates site_location_ids against the same
        fully-active-hierarchy + active-camera filter used by list_site_locations
        / /tracing/active/site_locations.
        """
        db = SessionLocal()
        try:
            # ── Validate: keep only locations that pass the hierarchy+camera check ──
            all_valid = TrackingConsoleRepository.list_site_locations(db)
            valid_id_set = {loc["id"] for loc in all_valid}
            filtered_ids = [lid for lid in site_location_ids if lid in valid_id_set]

            if not filtered_ids:
                return []

            return TrackingConsoleRepository.get_cameras_for_locations(
                db, filtered_ids
            )
        finally:
            db.close()