# Member 3 Extension Point: SQLite Alert Storage

This directory is reserved for **Member 3** to implement lightweight SQLite database persistence for alert records and human verification history.

---

## 1. Responsibilities for Member 3

1. Create a simple SQLite table to record all generated safety events.
2. Persist event metadata (event type, camera ID, track ID, timestamp, confidence, verification status).
3. Allow updating alert status when the human operator clicks **"Verify"** or **"Dismiss"** in the Streamlit dashboard.
4. Provide query utilities to display recent alert history and audit logs.

---

## 2. Recommended SQLite Schema

```sql
CREATE TABLE IF NOT EXISTS safety_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,         -- e.g. 'possible_fall', 'distress_gesture'
    track_id INTEGER,                 -- Tracking ID of the person (or NULL)
    timestamp REAL NOT NULL,          -- Video timestamp in seconds
    camera_id TEXT NOT NULL,          -- e.g. 'CAM_01'
    confidence REAL NOT NULL,         -- Confidence score (0.0 - 1.0)
    status TEXT NOT NULL DEFAULT 'pending', -- 'pending', 'verified', 'dismissed'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    notes TEXT                        -- Optional human operator comments
);
```

---

## 3. Sample Python SQLite Implementation Pattern

```python
import sqlite3
from typing import List, Dict, Any

DB_PATH = "data/alerts.sqlite"

def init_db(db_path: str = DB_PATH) -> None:
    """Initialize alerts table if it does not exist."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS safety_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type TEXT NOT NULL,
                track_id INTEGER,
                timestamp REAL NOT NULL,
                camera_id TEXT NOT NULL,
                confidence REAL NOT NULL,
                status TEXT NOT NULL DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                notes TEXT
            )
        """)
        conn.commit()

def insert_alert(event: Dict[str, Any], db_path: str = DB_PATH) -> int:
    """Insert a new safety event with pending status."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO safety_alerts (event_type, track_id, timestamp, camera_id, confidence, status)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event["event_type"],
            event.get("track_id"),
            event["timestamp"],
            event.get("camera_id", "CAM_01"),
            event["confidence"],
            event.get("status", "pending")
        ))
        conn.commit()
        return cursor.lastrowid

def update_alert_status(alert_id: int, status: str, notes: str = "", db_path: str = DB_PATH) -> None:
    """Update verification status ('verified' or 'dismissed')."""
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("""
            UPDATE safety_alerts
            SET status = ?, notes = ?
            WHERE id = ?
        """, (status, notes, alert_id))
        conn.commit()
```

> **Note**: Database files (`*.sqlite`, `*.db`, `*.sqlite3`) are ignored in `.gitignore` to avoid checking runtime databases into source control.
