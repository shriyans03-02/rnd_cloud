from typing import Optional, Dict, Any, List
from datetime import datetime

from app.db.session import SessionLocal
from app.repositories.member_tracing_repo import MemberTracingRepository
from app.core.constants import MOVEMENT_TYPES


class MemberTracingService:

    @staticmethod
    def get_member_trace(
        member_id: int,
        date: datetime,
        entry_ts: Optional[datetime],
        exit_ts: Optional[datetime],
    ) -> Dict[str, Any]:
        db = SessionLocal()
        try:
            # ── 1. Member details (with access groups) ───────────────────────
            member = MemberTracingRepository.get_member_details(db, member_id)
            if not member:
                return {"error": "Member not found"}

            member_info = {
                "id": member.id,
                "member_number": member.member_number,
                "first_name": member.first_name,
                "last_name": member.last_name or "",
                "full_name": f"{member.first_name} {member.last_name or ''}".strip(),
                "department": member.department.name if member.department else None,
            }

            # ── 2. Build allowed site_location_id set ────────────────────────
            allowed_site_location_ids = MemberTracingRepository.get_allowed_site_location_ids(member)

            # ── 3. Raw events ────────────────────────────────────────────────
            events = MemberTracingRepository.get_member_events(
                db, member_id, date, entry_ts, exit_ts
            )

            if not events:
                return {
                    "member": member_info,
                    "locations": [],
                    "timeline": [],
                    "gaps": [],
                    "summary": {
                        "total_locations_visited": 0,
                        "total_visits": 0,
                        "total_duration_seconds": 0,
                        "first_seen": None,
                        "last_seen": None,
                        "is_live": False,
                        "undetected_seconds": 0,
                        "time_outside_zones_seconds": 0,
                    },
                }

            # ── 4. Build flat chronological timeline ─────────────────────────
            timeline: List[Dict[str, Any]] = []
            for r in events:
                cam_name = r.camera.name if r.camera else None
                # ✅ NEW: capture camera_id for playback
                cam_id = r.camera_id if r.camera else None
                loc_name = None
                site_location_id = None

                if r.camera and r.camera.site_location_rel:
                    loc_name = r.camera.site_location_rel.name
                    site_location_id = r.camera.site_location_id

                duration_sec = None
                if r.entry_ts and r.exit_ts:
                    duration_sec = int((r.exit_ts - r.entry_ts).total_seconds())

                # ── Authorization check per visit ────────────────────────────
                if site_location_id is None:
                    is_authorized = True
                elif member.access_groups:
                    is_authorized = site_location_id in allowed_site_location_ids
                else:
                    is_authorized = True

                timeline.append({
                    "event_id":            r.id,
                    "camera":              cam_name,
                    "camera_id":           cam_id,          # ✅ NEW
                    "location":            loc_name or cam_name or "Unknown",
                    "site_location_id":    site_location_id,
                    "movement_type":       r.movement_type,
                    "movement_label":      MOVEMENT_TYPES.get(r.movement_type, "Unknown"),
                    "entry_ts":            r.entry_ts.isoformat() if r.entry_ts else None,
                    "exit_ts":             r.exit_ts.isoformat()  if r.exit_ts  else None,
                    "duration_seconds":    duration_sec,
                    "average_match_value": r.average_match_value,
                    "is_authorized":       is_authorized,
                })

            # ── 5. Build locations list ───────────────────────────────────────
            location_map: Dict[str, Dict] = {}

            for t in timeline:
                lk = t["location"]
                if lk not in location_map:
                    location_map[lk] = {
                        "location": lk,
                        "camera":   t["camera"],
                        "camera_id": t["camera_id"],        # ✅ NEW
                        "visits":   [],
                        "total_duration_seconds": 0,
                    }
                visit_num = len(location_map[lk]["visits"]) + 1
                location_map[lk]["visits"].append({
                    "visit_number":        visit_num,
                    "entry_ts":            t["entry_ts"],
                    "exit_ts":             t["exit_ts"],
                    "duration_seconds":    t["duration_seconds"],
                    "movement_label":      t["movement_label"],
                    "average_match_value": t["average_match_value"],
                    "is_authorized":       t["is_authorized"],
                    "camera_id":           t["camera_id"],  # ✅ NEW — per-visit camera_id
                })
                if t["duration_seconds"] is not None:
                    location_map[lk]["total_duration_seconds"] += t["duration_seconds"]

            locations_list = list(location_map.values())

            # ── 6. Live detection ─────────────────────────────────────────────
            last_event = events[-1]
            is_live = last_event.exit_ts is None

            # ── 7. Summary ────────────────────────────────────────────────────
            all_entry_times = [t["entry_ts"] for t in timeline if t["entry_ts"]]
            all_exit_times  = [t["exit_ts"]  for t in timeline if t["exit_ts"]]
            total_duration  = sum(
                t["duration_seconds"] for t in timeline if t["duration_seconds"] is not None
            )

            first_seen = min(all_entry_times) if all_entry_times else None
            last_seen  = max(all_exit_times)  if all_exit_times  else None

            # ── 8. Undetected time ────────────────────────────────────────────
            undetected_seconds = 0
            if first_seen:
                first_dt = datetime.fromisoformat(first_seen)
                all_ts = [t for t in all_entry_times + all_exit_times if t]
                if all_ts:
                    last_any_ts = max(all_ts)
                    end_dt = datetime.fromisoformat(last_any_ts)
                    wall_clock_seconds = int((end_dt - first_dt).total_seconds())
                    undetected_seconds = max(0, wall_clock_seconds - total_duration)

            # ── 9. Time outside designated zones ─────────────────────────────
            time_outside_zones_seconds = sum(
                t["duration_seconds"]
                for t in timeline
                if not t["is_authorized"] and t["duration_seconds"] is not None
            )

            # ── 10. Inter-location gaps ───────────────────────────────────────
            gaps: List[Dict[str, Any]] = []
            for i in range(len(timeline) - 1):
                a = timeline[i]
                b = timeline[i + 1]

                if a["location"] == b["location"]:
                    continue

                exit_ts_a  = a["exit_ts"]
                entry_ts_b = b["entry_ts"]

                if not exit_ts_a or not entry_ts_b:
                    continue

                exit_dt  = datetime.fromisoformat(exit_ts_a)
                entry_dt = datetime.fromisoformat(entry_ts_b)
                gap_sec  = int((entry_dt - exit_dt).total_seconds())

                if gap_sec <= 0:
                    continue

                gaps.append({
                    "from_location": a["location"],
                    "to_location":   b["location"],
                    "exit_ts":       exit_ts_a,
                    "entry_ts":      entry_ts_b,
                    "gap_seconds":   gap_sec,
                })

            summary = {
                "total_locations_visited":    len(location_map),
                "total_visits":               len(timeline),
                "total_duration_seconds":     total_duration,
                "first_seen":                 first_seen,
                "last_seen":                  last_seen,
                "is_live":                    is_live,
                "undetected_seconds":         undetected_seconds,
                "time_outside_zones_seconds": time_outside_zones_seconds,
            }

            return {
                "member":    member_info,
                "locations": locations_list,
                "timeline":  timeline,
                "gaps":      gaps,
                "summary":   summary,
            }

        finally:
            db.close()


    @staticmethod
    def list_active_member_names(
        search: Optional[str] = None, limit: int = 50
    ) -> List[Dict]:
        db = SessionLocal()
        try:
            members = MemberTracingRepository.list_all_active(db, search=search, limit=limit)
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
        db = SessionLocal()
        try:
            return MemberTracingRepository.list_site_locations(db, search=search)
        finally:
            db.close()

    @staticmethod
    def get_cameras_by_location(site_location_id: int) -> List[Dict]:
        db = SessionLocal()
        try:
            return MemberTracingRepository.get_cameras_by_site_location(db, site_location_id)
        finally:
            db.close()