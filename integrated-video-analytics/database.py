import sqlite3
import time
from typing import List, Dict, Any

DB_PATH = 'analytics.db'

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Events table for generic logs like faces and plates
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            event_type TEXT NOT NULL,
            detail TEXT NOT NULL
        )
    ''')
    
    # Store aggregate counts periodically if needed
    c.execute('''
        CREATE TABLE IF NOT EXISTS metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            vehicle_count INTEGER,
            people_count INTEGER,
            zone_count INTEGER
        )
    ''')
    
    conn.commit()
    conn.close()

def log_event(event_type: str, detail: str):
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO events (event_type, detail) VALUES (?, ?)", (event_type, detail))
        conn.commit()
    except Exception as e:
        print(f"DB Error: {e}")
    finally:
        if conn:
            conn.close()

def get_recent_events(limit: int = 50) -> List[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()
        c.execute("SELECT timestamp, event_type as type, detail FROM events ORDER BY timestamp DESC LIMIT ?", (limit,))
        rows = c.fetchall()
        
        # Convert to dict format expected by the frontend
        return [{"time": row['timestamp'].split()[1], "type": row['type'], "detail": row['detail']} for row in rows]
    except Exception as e:
        print(f"DB Error: {e}")
        return []
    finally:
        if conn:
            conn.close()

# Initialize when imported
init_db()
