from pathlib import Path

import database


def test_database_records_are_searchable(tmp_path):
    db_path = tmp_path / "analytics_test.db"
    original_path = database.DB_PATH

    try:
        database.DB_PATH = db_path
        database.init_db()

        database.log_event("cam_a", "ANPR Match", "Plate ABC123 detected")
        database.log_event("cam_b", "Face Analytics", "Face analytics completed")
        database.upsert_vehicle_record("cam_a", 7, "car", "ABC123")
        database.upsert_plate_read("cam_a", 7, "ABC123", "car", 0.91)
        database.upsert_face_record("cam_a", 11, "alice", "Woman", None, True)

        events = database.get_recent_events(camera_id="cam_a")
        plates = database.get_plate_reads(query="ABC", camera_id="cam_a")
        faces = database.get_face_records(query="alice", camera_id="cam_a")

        assert len(events) == 1
        assert events[0]["camera_id"] == "cam_a"
        assert plates[0]["plate_text"] == "ABC123"
        assert faces[0]["identity"] == "alice"
        assert faces[0]["watchlist_hit"] == 1
    finally:
        database.DB_PATH = original_path


def test_database_upserts_update_existing_records(tmp_path):
    db_path = tmp_path / "analytics_upsert.db"
    original_path = database.DB_PATH

    try:
        database.DB_PATH = db_path
        database.init_db()

        database.upsert_plate_read("cam_a", 1, "XYZ999", "car", 0.55)
        database.upsert_plate_read("cam_a", 9, "XYZ999", "truck", 0.88)

        records = database.get_plate_reads(camera_id="cam_a")
        assert len(records) == 1
        assert records[0]["tracker_id"] == 9
        assert records[0]["vehicle_type"] == "truck"
    finally:
        database.DB_PATH = original_path


def test_legacy_event_schema_is_still_writable(tmp_path):
    db_path = tmp_path / "legacy.db"
    original_path = database.DB_PATH

    try:
        database.DB_PATH = db_path
        conn = database._connect()
        conn.execute(
            """
            CREATE TABLE events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL,
                details TEXT NOT NULL,
                camera_id TEXT NOT NULL DEFAULT 'camera_1'
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                vehicle_count INTEGER,
                people_count INTEGER,
                zone_count INTEGER
            )
            """
        )
        conn.commit()
        conn.close()

        database.init_db()
        database.log_event("cam_legacy", "Face Analytics", "Legacy schema write works")

        events = database.get_recent_events(camera_id="cam_legacy")
        assert len(events) == 1
        assert events[0]["detail"] == "Legacy schema write works"
    finally:
        database.DB_PATH = original_path
