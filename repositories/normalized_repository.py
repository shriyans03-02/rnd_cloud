from sqlalchemy import text
import traceback

class NormalizedRepository:
    def __init__(self, engine):
        self.engine = engine

    def bulk_insert(self, rows):
        print("==== Executing Bulk Insert ====", flush=True)

        if rows is None:
            print("ROWS is None", flush=True)
            return

        rows = list(rows)

        if len(rows) == 0:
            print("ROWS is empty", flush=True)
            return

        print("ROWS COUNT:", len(rows), flush=True)

        stmt = text("""
            INSERT INTO normalized_data (
                member_id,
                guest_temp_id,
                average_guest_data_vector,
                camera_id,
                movement_type,
                movement_ts,
                average_match_value
            )
            VALUES (
                :member_id,
                :guest_temp_id,
                :average_guest_data_vector,
                :camera_id,
                :movement_type,
                :movement_ts,
                :average_match_value
            )
        """)

        print("QUERY READY", flush=True)

        try:
            with self.engine.begin() as conn:
                conn.execute(stmt, rows)
                return True    
            print("INSERT SUCCESS", flush=True)
        except Exception as e:
            print("❌ INSERT ERROR:", e, flush=True)
            print("========================================= Executing Bulk Insert =============================")
            if not rows:
                print("ROWS is None", flush=True)
                return 

            stmt = text("""
                INSERT INTO normalized_data (
                    member_id,
                    guest_temp_id,
                    average_guest_data_vector,
                    camera_id,
                    movement_type,
                    movement_ts,
                    average_match_value
                )
                VALUES (
                    :member_id,
                    :guest_temp_id,
                    :average_guest_data_vector,
                    :camera_id,
                    :movement_type,
                    :movement_ts,
                    :average_match_value
                )
            """)
            print("***************** INSERT QRY **********************",stmt)
            try:
                with self.engine.begin() as conn:
                    conn.execute(stmt, rows)
                print("STEP: Insert executed", flush=True)
                return True
            except Exception as e:
                print("❌ INSERT ERROR:", e, flush=True)
                print(traceback.format_exc())
                return False