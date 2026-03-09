from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "analytics.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH, timeout=30, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def _get_columns(conn: sqlite3.Connection, table_name: str) -> set[str]:
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def init_db() -> None:
    conn = _connect()
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL DEFAULT 'camera_1',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                event_type TEXT NOT NULL,
                detail TEXT NOT NULL
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL DEFAULT 'camera_1',
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                vehicle_count INTEGER,
                people_count INTEGER,
                zone_count INTEGER
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS vehicle_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                tracker_id INTEGER NOT NULL,
                vehicle_type TEXT NOT NULL,
                plate_text TEXT,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(camera_id, tracker_id)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS plate_reads (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                tracker_id INTEGER,
                plate_text TEXT NOT NULL,
                vehicle_type TEXT NOT NULL,
                confidence REAL,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(camera_id, plate_text)
            )
            """
        )
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS face_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                camera_id TEXT NOT NULL,
                tracker_id INTEGER NOT NULL,
                identity TEXT,
                gender TEXT,
                age INTEGER,
                watchlist_hit INTEGER NOT NULL DEFAULT 0,
                first_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                last_seen DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(camera_id, tracker_id)
            )
            """
        )

        event_columns = _get_columns(conn, "events")
        if "camera_id" not in event_columns:
            conn.execute("ALTER TABLE events ADD COLUMN camera_id TEXT NOT NULL DEFAULT 'camera_1'")
        if "detail" not in event_columns:
            conn.execute("ALTER TABLE events ADD COLUMN detail TEXT")
            if "details" in event_columns:
                conn.execute("UPDATE events SET detail = details WHERE detail IS NULL OR detail = ''")

        metric_columns = _get_columns(conn, "metrics")
        if "camera_id" not in metric_columns:
            conn.execute("ALTER TABLE metrics ADD COLUMN camera_id TEXT NOT NULL DEFAULT 'camera_1'")

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_camera_timestamp ON events(camera_id, timestamp DESC)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_plate_reads_plate ON plate_reads(plate_text)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_face_records_identity ON face_records(identity)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vehicle_records_camera_tracker ON vehicle_records(camera_id, tracker_id)"
        )
        conn.commit()
    finally:
        conn.close()


def log_event(camera_id: str, event_type: str, detail: str) -> None:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect()
        event_columns = _get_columns(conn, "events")
        timestamp = datetime.utcnow().isoformat(timespec="seconds")
        if "details" in event_columns:
            conn.execute(
                """
                INSERT INTO events (camera_id, timestamp, event_type, detail, details)
                VALUES (?, ?, ?, ?, ?)
                """,
                (camera_id, timestamp, event_type, detail, detail),
            )
        else:
            conn.execute(
                "INSERT INTO events (camera_id, timestamp, event_type, detail) VALUES (?, ?, ?, ?)",
                (camera_id, timestamp, event_type, detail),
            )
        conn.commit()
    except Exception as exc:
        print(f"DB Error: {exc}")
    finally:
        if conn is not None:
            conn.close()


def store_metric(camera_id: str, vehicle_count: int, people_count: int, zone_count: int) -> None:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO metrics (camera_id, vehicle_count, people_count, zone_count)
            VALUES (?, ?, ?, ?)
            """,
            (camera_id, vehicle_count, people_count, zone_count),
        )
        conn.commit()
    except Exception as exc:
        print(f"DB Error: {exc}")
    finally:
        if conn is not None:
            conn.close()


def upsert_vehicle_record(
    camera_id: str,
    tracker_id: int,
    vehicle_type: str,
    plate_text: Optional[str] = None,
) -> None:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO vehicle_records (camera_id, tracker_id, vehicle_type, plate_text)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(camera_id, tracker_id) DO UPDATE SET
                vehicle_type = excluded.vehicle_type,
                plate_text = COALESCE(excluded.plate_text, vehicle_records.plate_text),
                last_seen = CURRENT_TIMESTAMP
            """,
            (camera_id, tracker_id, vehicle_type, plate_text),
        )
        conn.commit()
    except Exception as exc:
        print(f"DB Error: {exc}")
    finally:
        if conn is not None:
            conn.close()


def upsert_plate_read(
    camera_id: str,
    tracker_id: Optional[int],
    plate_text: str,
    vehicle_type: str,
    confidence: Optional[float],
) -> None:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO plate_reads (camera_id, tracker_id, plate_text, vehicle_type, confidence)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(camera_id, plate_text) DO UPDATE SET
                tracker_id = COALESCE(excluded.tracker_id, plate_reads.tracker_id),
                vehicle_type = excluded.vehicle_type,
                confidence = COALESCE(excluded.confidence, plate_reads.confidence),
                last_seen = CURRENT_TIMESTAMP
            """,
            (camera_id, tracker_id, plate_text, vehicle_type, confidence),
        )
        conn.commit()
    except Exception as exc:
        print(f"DB Error: {exc}")
    finally:
        if conn is not None:
            conn.close()


def upsert_face_record(
    camera_id: str,
    tracker_id: int,
    identity: Optional[str],
    gender: Optional[str],
    age: Optional[int],
    watchlist_hit: bool,
) -> None:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect()
        conn.execute(
            """
            INSERT INTO face_records (camera_id, tracker_id, identity, gender, age, watchlist_hit)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT(camera_id, tracker_id) DO UPDATE SET
                identity = COALESCE(excluded.identity, face_records.identity),
                gender = COALESCE(excluded.gender, face_records.gender),
                age = COALESCE(excluded.age, face_records.age),
                watchlist_hit = excluded.watchlist_hit,
                last_seen = CURRENT_TIMESTAMP
            """,
            (camera_id, tracker_id, identity, gender, age, int(watchlist_hit)),
        )
        conn.commit()
    except Exception as exc:
        print(f"DB Error: {exc}")
    finally:
        if conn is not None:
            conn.close()


def _build_filter_clause(camera_id: Optional[str], query: Optional[str]) -> tuple[str, List[Any]]:
    clauses: List[str] = []
    params: List[Any] = []

    if camera_id:
        clauses.append("camera_id = ?")
        params.append(camera_id)

    if query:
        clauses.append("(detail LIKE ? OR event_type LIKE ?)")
        like_query = f"%{query}%"
        params.extend([like_query, like_query])

    if not clauses:
        return "", params
    return " WHERE " + " AND ".join(clauses), params


def get_recent_events(
    limit: int = 50,
    query: Optional[str] = None,
    camera_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect()
        where_clause, params = _build_filter_clause(camera_id, query)
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT camera_id, timestamp, event_type AS type, detail
            FROM events
            {where_clause}
            ORDER BY timestamp DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        print(f"DB Error: {exc}")
        return []
    finally:
        if conn is not None:
            conn.close()


def get_plate_reads(
    limit: int = 50,
    query: Optional[str] = None,
    camera_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect()
        clauses: List[str] = []
        params: List[Any] = []

        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)

        if query:
            clauses.append("(plate_text LIKE ? OR vehicle_type LIKE ?)")
            like_query = f"%{query}%"
            params.extend([like_query, like_query])

        where_clause = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT camera_id, tracker_id, plate_text, vehicle_type, confidence, first_seen, last_seen
            FROM plate_reads
            {where_clause}
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        print(f"DB Error: {exc}")
        return []
    finally:
        if conn is not None:
            conn.close()


def get_face_records(
    limit: int = 50,
    query: Optional[str] = None,
    camera_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    conn: Optional[sqlite3.Connection] = None
    try:
        conn = _connect()
        clauses: List[str] = []
        params: List[Any] = []

        if camera_id:
            clauses.append("camera_id = ?")
            params.append(camera_id)

        if query:
            clauses.append("(COALESCE(identity, '') LIKE ? OR COALESCE(gender, '') LIKE ?)")
            like_query = f"%{query}%"
            params.extend([like_query, like_query])

        where_clause = f" WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(limit)
        rows = conn.execute(
            f"""
            SELECT camera_id, tracker_id, identity, gender, age, watchlist_hit, first_seen, last_seen
            FROM face_records
            {where_clause}
            ORDER BY last_seen DESC
            LIMIT ?
            """,
            params,
        ).fetchall()
        return [dict(row) for row in rows]
    except Exception as exc:
        print(f"DB Error: {exc}")
        return []
    finally:
        if conn is not None:
            conn.close()


init_db()
