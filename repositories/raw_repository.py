from sqlalchemy import  text

class RawRepository:
    def __init__(self, engine):
        self.engine = engine

    def ensure_raw_data_registry(self):
        with self.engine.begin() as conn:
            conn.execute(text("""
                CREATE TABLE IF NOT EXISTS raw_data_registry (
                table_name TEXT PRIMARY KEY,
                created_ts TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
                normalized BOOLEAN DEFAULT FALSE
                );
            """))
            
    def get_next_unprocessed_table(self):
        with self.engine.begin() as conn:
            row = conn.execute(
                text("""
                    SELECT table_name
                    FROM raw_data_registry
                    WHERE normalized = FALSE
                    ORDER BY created_ts
                    LIMIT 1
                """)
            ).first()
            return row[0] if row else None

    def stream_table(self, table_name):
        query = f"SELECT * FROM {table_name} ORDER BY data_ts" 
        with self.engine.connect() as conn:   # ✅ use connect, not begin
            result = conn.execute(text(query))
            rows = result.fetchall()

        print(f"Fetched {len(rows)} rows from {table_name}")
        return rows

    def mark_normalized(self, table_name):
        with self.engine.begin() as conn:
            conn.execute(
                text("UPDATE raw_data_registry SET normalized = TRUE WHERE table_name=:t"),
                {"t": table_name},
            )

    def drop_table(self, table_name):
        with self.engine.begin() as conn:
            conn.execute(text(f"DROP TABLE IF EXISTS {table_name}"))
